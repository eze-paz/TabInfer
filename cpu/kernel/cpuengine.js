// Pure-CPU Qwen3/Bonsai inference engine (single-thread, correctness-first).
// Heavy ternary matmuls run in cpukern.wasm (relaxed-SIMD gemv_tern); the light
// ops (RMSNorm, per-head QK-norm, RoPE, GQA attention + KV cache, SwiGLU, argmax)
// run in JS over Float32Arrays that view wasm output directly (zero-copy reads).
// Weights: _proj/lm_head are ternary-packed into wasm linear memory; embed stays
// f16 (row-gathered per token); norms stay f32. UNTRACKED spike.
const CPUEngine = (function () {
  const CFG = { L: 28, H: 2048, nH: 16, nKV: 8, hd: 128, I: 6144, theta: 1e6, eps: 1e-6, vocab: 151936, G: 64 };
  let W = null;             // wasm exports
  let heapTop = 0;          // bump pointer in wasm linear memory
  const wt = {};            // ternary weight registry: name -> {oCodes, oScales, N, K}
  let embedF16 = null;      // Uint16Array [vocab*H] (row-gathered per token)
  const nrm = {};           // norm/qk-norm weights: name -> Float32Array
  let sAct = 0, sXsum = 0, sOut = 0; // scratch offsets (int8 act[K], i32 xsum[K/64], f32 out[N])

  // ---- half/bfloat decode ----
  function f16f(h) { const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024); }
  const _bb = new ArrayBuffer(4), _bf = new Float32Array(_bb), _bu = new Uint32Array(_bb);
  function bf16f(h) { _bu[0] = h << 16; return _bf[0]; }

  const align16 = x => (x + 15) & ~15;
  async function initWasm() {
    const buf = await (await fetch('cpukern.wasm')).arrayBuffer();
    const r = await WebAssembly.instantiate(buf, {});
    W = r.instance.exports;
    heapTop = align16(W.heap_base());
  }
  function gr0(bytes) { // bump-allocate `bytes` of zeroed wasm memory, return offset
    const off = heapTop; heapTop = align16(heapTop + bytes);
    const need = Math.ceil(heapTop / 65536) + 1, have = W.memory.buffer.byteLength / 65536;
    if (need > have) W.memory.grow(need - have);
    return off;
  }

  // ---- ternary pack: rows×K bf16 weights -> interleaved 2-bit codes + f32 scales (G=64) ----
  // matches cpukern gemv_tern: per 64-block, code c -> byte (c&15) at bits 2*(c>>4);
  // code = weight_sign+1 (0=-1,1=0,2=+1); scale = per-64 group absmax.
  function packTernary(getBits, N, K) {
    const bpr = K / 4, ng = K / CFG.G;
    const oCodes = gr0(N * bpr), oScales = gr0(N * ng * 4);
    // views taken AFTER both allocs (grow may detach); re-take per row batch is unnecessary since no more grows here
    const codes = new Uint8Array(W.memory.buffer, oCodes, N * bpr);
    const scales = new Float32Array(W.memory.buffer, oScales, N * ng);
    for (let n = 0; n < N; n++) {
      const rowBits = n * K, rC = n * bpr, rS = n * ng;
      for (let g = 0; g < ng; g++) {
        const base = g * CFG.G;
        let s = 0;
        for (let j = 0; j < CFG.G; j++) { const v = Math.abs(bf16f(getBits(rowBits + base + j))); if (v > s) s = v; }
        scales[rS + g] = s;
        const thr = s * 0.5;
        for (let j = 0; j < CFG.G; j++) {
          const w = bf16f(getBits(rowBits + base + j));
          const code = Math.abs(w) < thr ? 1 : (w < 0 ? 0 : 2);
          if (code === 1) continue; // 1 is the zero bit-pattern; already zeroed
          const c = base + j;                       // 0..63 within block? base is within-row; block index:
          const blk = c >> 6, within = c & 63;
          codes[rC + blk * 16 + (within & 15)] |= code << (2 * (within >> 4));
        }
      }
    }
    return { oCodes, oScales, N, K };
  }

  // ---- safetensors shard parse + dispatch ----
  function parseShard(ab, onName) {
    const dv = new DataView(ab);
    const hlen = Number(dv.getBigUint64(0, true));
    const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 8, hlen)));
    const dataStart = 8 + hlen;
    for (const name of Object.keys(hdr)) {
      if (name === '__metadata__') continue;
      const info = hdr[name], [b0, b1] = info.data_offsets;
      onName(name, info.dtype, info.shape, ab, dataStart + b0, b1 - b0);
    }
  }
  const isTern = n => n.endsWith('_proj.weight') || n === 'lm_head.weight';

  async function load(root, onProgress) {
    const idx = await (await fetch(root + 'model.safetensors.index.json')).json();
    const shards = [...new Set(Object.values(idx.weight_map))].sort();
    for (let si = 0; si < shards.length; si++) {
      onProgress && onProgress({ phase: 'download', shard: si + 1, of: shards.length });
      const ab = await (await fetch(root + shards[si])).arrayBuffer();
      parseShard(ab, (name, dtype, shape, buf, off, len) => {
        if (name === 'model.embed_tokens.weight') {
          embedF16 = new Uint16Array(buf.slice(off, off + len)); // F16, kept
        } else if (isTern(name)) {
          const u16 = new Uint16Array(buf, off, len / 2);        // BF16 halves
          const [N, K] = shape;
          wt[name] = packTernary(i => u16[i], N, K);
        } else {                                                  // norms: F16 -> f32
          const u16 = new Uint16Array(buf, off, len / 2);
          const a = new Float32Array(u16.length);
          for (let i = 0; i < u16.length; i++) a[i] = f16f(u16[i]);
          nrm[name] = a;
        }
        onProgress && onProgress({ phase: 'pack', name });
      });
    }
    // scratch: act int8[maxK], xsum i32[maxK/64], out f32[maxN]
    const maxK = CFG.I, maxN = CFG.vocab;
    sAct = gr0(maxK); sXsum = gr0((maxK / CFG.G) * 4); sOut = gr0(maxN * 4);
    onProgress && onProgress({ phase: 'ready' });
  }

  // FAST LOAD: pre-packed binary (pack_bin.py) — memcpy into wasm, no JS packing (~sec vs 70s).
  async function loadBin(url, onProgress) {
    _ternAny = true;
    const ab = await (await fetch(url)).arrayBuffer();
    const dv = new DataView(ab);
    const hlen = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, hlen)));
    const dataBase = 4 + hlen;
    const T = header.tensors;
    for (const name in T) {
      const t = T[name];
      if (t.kind === 'tern') {
        const oCodes = gr0(t.codesLen), oScales = gr0(t.scalesLen);
        new Uint8Array(W.memory.buffer, oCodes, t.codesLen).set(new Uint8Array(ab, dataBase + t.codesOff, t.codesLen));
        new Uint8Array(W.memory.buffer, oScales, t.scalesLen).set(new Uint8Array(ab, dataBase + t.scalesOff, t.scalesLen));
        wt[name] = { oCodes, oScales, N: t.N, K: t.K, tern: true };
      } else if (t.kind === 'embed_f16') {
        embedF16 = new Uint16Array(ab.slice(dataBase + t.off, dataBase + t.off + t.len));
      } else { // f32 norms
        nrm[name] = new Float32Array(ab.slice(dataBase + t.off, dataBase + t.off + t.len));
      }
    }
    const maxK = CFG.I, maxN = CFG.vocab;
    sAct = gr0(maxK); sXsum = gr0((maxK / CFG.G) * 4); sOut = gr0(maxN * 4);
    onProgress && onProgress({ phase: 'ready' });
  }

  // ---- matmul: y[N] = ternary(name) · x[K]  (x: Float32Array) ----
  let _prof = null; // {mm, mmLm, quant, gemv, scale, attn, norm, rope, embed}
  const _now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  // cached wasm-memory views (buffer stable after load — no grows in the forward)
  let _actV = null, _xsumV = null, _rawV = null;
  function ensureViews() {
    if (_actV && _actV.buffer === W.memory.buffer) return;
    _actV = new Int8Array(W.memory.buffer, sAct, CFG.I);
    _xsumV = new Int32Array(W.memory.buffer, sXsum, CFG.I / CFG.G);
    _rawV = new Float32Array(W.memory.buffer, sOut, CFG.vocab);
  }
  // matmul into a caller-provided buffer (no per-call alloc). out must hold >= N.
  function matmul(name, x, out) {
    const rec = wt[name], K = rec.K, N = rec.N, ng = K / CFG.G;
    ensureViews();
    const t0 = _prof ? _now() : 0;
    let amax = 0; for (let k = 0; k < K; k++) { const a = x[k] < 0 ? -x[k] : x[k]; if (a > amax) amax = a; }
    const asc = amax / 127 || 1e-9, inv = 1 / asc;
    const act = _actV, xsum = _xsumV;
    for (let k = 0; k < K; k++) { let q = Math.round(x[k] * inv); q = q < -127 ? -127 : q > 127 ? 127 : q; act[k] = q; }
    for (let g = 0; g < ng; g++) { let t = 0, b = g * CFG.G; for (let j = 0; j < CFG.G; j++) t += act[b + j]; xsum[g] = t; }
    const t1 = _prof ? _now() : 0;
    W.gemv_tern(sOut, rec.oCodes, rec.oScales, sAct, sXsum, N, K);
    const t2 = _prof ? _now() : 0;
    const raw = _rawV;
    for (let n = 0; n < N; n++) out[n] = raw[n] * asc;
    if (_prof) { const e = _now(); _prof.quant += t1 - t0; _prof.gemv += t2 - t1; _prof.scale += e - t2; _prof.mm += e - t0; if (name === 'lm_head.weight') _prof.mmLm += e - t0; }
    return out;
  }

  // ---- light ops (write into provided buffer) ----
  function rmsnorm(x, w, n, out) {
    let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i];
    const inv = 1 / Math.sqrt(s / n + CFG.eps);
    for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    return out;
  }
  function ropeInPlace(v, off, pos) { // rotate_half over hd dims at v[off..off+hd)
    const hd = CFG.hd, half = hd / 2;
    for (let i = 0; i < half; i++) {
      const fr = Math.pow(CFG.theta, -(2 * i) / hd), a = pos * fr, c = Math.cos(a), s = Math.sin(a);
      const x0 = v[off + i], x1 = v[off + i + half];
      v[off + i] = x0 * c - x1 * s;
      v[off + i + half] = x1 * c + x0 * s;
    }
  }

  // KV cache: kv[layer] = { k:[Float32Array(nKV*hd)], v:[...] } appended per position
  let _dbg = null;
  const _rms = (a, n) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * a[i]; return Math.sqrt(s / n); };
  const _dp = (tag, a, n) => { if (_dbg) _dbg.push(tag + '=' + _rms(a, n).toFixed(4)); };
  let _bufs = null;
  function _ensureBufs() {
    if (_bufs) return; const { H, hd, nH, nKV, I, vocab } = CFG;
    _bufs = {
      x: new Float32Array(H), xn: new Float32Array(H), xn2: new Float32Array(H),
      q: new Float32Array(nH * hd), k: new Float32Array(nKV * hd), v: new Float32Array(nKV * hd),
      attn: new Float32Array(nH * hd), gate: new Float32Array(I), up: new Float32Array(I), swi: new Float32Array(I),
      o: new Float32Array(H), xf: new Float32Array(H), logits: new Float32Array(vocab),
      hn: new Float32Array(hd), scores: new Float32Array(16384),
    };
  }
  function forward(tokenId, pos, kv) {
    const { H, hd, nH, nKV, L, I } = CFG; _ensureBufs(); const B = _bufs;
    const x = B.x, te = _prof ? _now() : 0;
    for (let i = 0; i < H; i++) x[i] = f16f(embedF16[tokenId * H + i]);
    if (_prof) _prof.embed += _now() - te;
    _dp('embed', x, H);

    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      let tn = _prof ? _now() : 0;
      rmsnorm(x, nrm[p + 'input_layernorm.weight'], H, B.xn);
      if (_prof) _prof.norm += _now() - tn;
      const q = matmul(p + 'self_attn.q_proj.weight', B.xn, B.q);
      const k = matmul(p + 'self_attn.k_proj.weight', B.xn, B.k);
      const v = matmul(p + 'self_attn.v_proj.weight', B.xn, B.v);
      const qnw = nrm[p + 'self_attn.q_norm.weight'], knw = nrm[p + 'self_attn.k_norm.weight'];
      tn = _prof ? _now() : 0;
      for (let h = 0; h < nH; h++) { const off = h * hd; rmsnorm(q.subarray(off, off + hd), qnw, hd, B.hn); for (let i = 0; i < hd; i++) q[off + i] = B.hn[i]; ropeInPlace(q, off, pos); }
      for (let h = 0; h < nKV; h++) { const off = h * hd; rmsnorm(k.subarray(off, off + hd), knw, hd, B.hn); for (let i = 0; i < hd; i++) k[off + i] = B.hn[i]; ropeInPlace(k, off, pos); }
      if (_prof) _prof.rope += _now() - tn;
      kv[l].k.push(k.slice()); kv[l].v.push(v.slice());   // KV must persist (copy)
      tn = _prof ? _now() : 0;
      const attn = B.attn; attn.fill(0); const scale = 1 / Math.sqrt(hd), T = kv[l].k.length, qpk = nH / nKV;
      const KVk = kv[l].k, KVv = kv[l].v, scores = B.scores;
      for (let h = 0; h < nH; h++) {
        const kvh = (h / qpk) | 0, qo = h * hd, ko = kvh * hd; let mx = -Infinity;
        for (let t = 0; t < T; t++) { const kt = KVk[t]; let d = 0; for (let i = 0; i < hd; i++) d += q[qo + i] * kt[ko + i]; d *= scale; scores[t] = d; if (d > mx) mx = d; }
        let sum = 0; for (let t = 0; t < T; t++) { const e = Math.exp(scores[t] - mx); scores[t] = e; sum += e; }
        const isum = 1 / sum;
        for (let t = 0; t < T; t++) { const w = scores[t] * isum, vt = KVv[t]; for (let i = 0; i < hd; i++) attn[qo + i] += w * vt[ko + i]; }
      }
      if (_prof) _prof.attn += _now() - tn;
      if (l === 0) { _dp('l0.xn', B.xn, H); _dp('l0.q', q, nH * hd); _dp('l0.attn', attn, nH * hd); }
      const o = matmul(p + 'self_attn.o_proj.weight', attn, B.o);
      for (let i = 0; i < H; i++) x[i] += o[i];
      if (l === 0) _dp('l0.afterAttn', x, H);
      tn = _prof ? _now() : 0;
      rmsnorm(x, nrm[p + 'post_attention_layernorm.weight'], H, B.xn2);
      if (_prof) _prof.norm += _now() - tn;
      const gate = matmul(p + 'mlp.gate_proj.weight', B.xn2, B.gate);
      const up = matmul(p + 'mlp.up_proj.weight', B.xn2, B.up);
      const swi = B.swi; for (let i = 0; i < I; i++) { const g = gate[i]; swi[i] = (g / (1 + Math.exp(-g))) * up[i]; }
      const down = matmul(p + 'mlp.down_proj.weight', swi, B.o);
      for (let i = 0; i < H; i++) x[i] += down[i];
      if (l === 0) _dp('l0.afterMLP', x, H);
    }
    _dp('final.x', x, H);
    rmsnorm(x, nrm['model.norm.weight'], H, B.xf);
    return matmul('lm_head.weight', B.xf, B.logits);
  }
  function profReset() { _prof = { mm: 0, mmLm: 0, quant: 0, gemv: 0, scale: 0, attn: 0, norm: 0, rope: 0, embed: 0 }; }
  function profGet() { return _prof; }
  function profOff() { _prof = null; }

  function newKV() { return Array.from({ length: CFG.L }, () => ({ k: [], v: [] })); }
  function argmax(a) { let bi = 0, bv = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; }

  async function generate(promptIds, maxTok, onTok) {
    const kv = newKV(); let pos = 0, last = 0;
    for (let i = 0; i < promptIds.length; i++) { const lg = forward(promptIds[i], pos++, kv); if (i === promptIds.length - 1) last = argmax(lg); }
    const out = [last]; onTok && onTok(last);
    for (let t = 1; t < maxTok; t++) {
      const lg = forward(last, pos++, kv); last = argmax(lg); out.push(last); onTok && onTok(last);
    }
    return out;
  }

  // debug: decode a packed ternary weight row back to f32 (for pack/kernel round-trip checks)
  function decodeRow(name, n) {
    const rec = wt[name], K = rec.K, ng = K / CFG.G, bpr = K / 4;
    const codes = new Uint8Array(W.memory.buffer, rec.oCodes + n * bpr, bpr);
    const scales = new Float32Array(W.memory.buffer, rec.oScales + n * ng * 4, ng);
    const out = new Float32Array(K);
    for (let b = 0; b < K / 64; b++) {
      const s = scales[b];
      for (let pl = 0; pl < 4; pl++) for (let i = 0; i < 16; i++) {
        const code = (codes[b * 16 + i] >> (2 * pl)) & 3;
        out[b * 64 + pl * 16 + i] = (code - 1) * s;
      }
    }
    return out;
  }
  return { initWasm, load, loadBin, forward, generate, newKV, argmax, matmul, CFG, _wt: wt, _nrm: nrm, _mem: () => W.memory, _decodeRow: decodeRow,
    _setDbg: (v) => { _dbg = v; }, profReset, profGet, profOff, };
})();
if (typeof window !== 'undefined') window.CPUEngine = CPUEngine;
