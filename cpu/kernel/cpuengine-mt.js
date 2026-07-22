// Multi-threaded pure-CPU Bonsai forward. Coordinator (this) runs the graph
// (embed/norm/rope/attention/quant) on the main thread and dispatches each
// ternary matmul to a row-partitioned worker pool via a SharedArrayBuffer Atomics
// barrier. Weights are partitioned across workers (no duplication). Reuses
// cpukern.wasm (gemv_tern) + bonsai17.cpu.bin.
const CPUEngineMT = (function () {
  const CFG = { L: 28, H: 2048, nH: 16, nKV: 8, hd: 128, I: 6144, theta: 1e6, eps: 1e-6, vocab: 151936, G: 64 };
  // ctrl slot layout — ONE CACHE LINE (16 i32 = 64B) PER CONTENDED WORD. The v1 layout put
  // GEN + params + all DONE flags in the first 64 bytes: every worker's DONE store invalidated
  // the line every spinner was reading (false sharing) → ~1ms barriers. Now: GEN on line 0,
  // params on line 1 (written before the GEN bump, read once per round), worker w's DONE flag
  // at its own line (32 + 16w).
  // Batched barrier: one dispatch computes up to 3 matmuls sharing ONE input activation
  // (q/k/v share xn; gate/up share xn2). Cuts barriers 197→~113 AND quantizes each shared
  // input once instead of per-matmul. Line 0: GEN. Line 1 (slots 16..31): batch spec.
  const GEN = 0, NMAT = 16, KK = 17, ASC = 18, MATID0 = 19, OUTOFF0 = 22, JOBTYPE = 25, LAYER = 26, TLEN = 27, SPIN = 28, DONEBASE = 32, DONESTRIDE = 16;
  const CTRL_I32 = 32 + 16 * 16; // room for 16 workers
  const MAXCTX = 512; // KV positions held in the shared buffer
  let Wn = 8, workers = [], ctrl, sabAct, sabXsum, sabOut, sabQ, sabAttn, sabKV, _kvStride = 0;
  let embedF16 = null; const nrm = {}, mats = {}; // name -> {mid,N,K}
  // MAIN-THREAD wasm instance (private memory) for the SIMD serial ops (quant + rmsnorm)
  // that otherwise run as scalar JS on the critical path, blocking every worker.
  let MW = null, mheap = 0, qDstOff = 0, qXsumOff = 0; const nrmOff = {}; // norm-weight → wasm offset
  async function initMainWasm() {
    const buf = await (await fetch('cpukern.wasm')).arrayBuffer();
    MW = (await WebAssembly.instantiate(buf, {})).instance.exports;
  }
  const align16 = x => (x + 15) & ~15;
  function mgrowTo(topBytes) { const have = MW.memory.buffer.byteLength; if (topBytes > have) MW.memory.grow(Math.ceil((topBytes - have) / 65536) + 1); }
  function malloc(bytes) { const off = align16(mheap); mheap = off + bytes; mgrowTo(mheap); return off; }
  const f32view = off => new Float32Array(MW.memory.buffer, off, undefined); // caller slices
  const _b = new ArrayBuffer(4), _bf = new Float32Array(_b), _bi = new Int32Array(_b);
  const f2i = (v) => { _bf[0] = v; return _bi[0]; };
  function f16f(h) { const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024); }

  async function load(binUrl, nWorkers) {
    if (workers.length) { try { Atomics.store(ctrl, NMAT, -1); Atomics.add(ctrl, GEN, 1); workers.forEach(w => w.terminate()); } catch (_) {} workers = []; }
    Wn = nWorkers || 8;
    const ab = await (await fetch(binUrl)).arrayBuffer();
    const dv = new DataView(ab), hlen = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, hlen)));
    const dataBase = 4 + hlen, T = header.tensors;
    // coordinator keeps embed + norms; matrix meta (mid via sorted tern names)
    const ternNames = Object.keys(T).filter(n => T[n].kind === 'tern').sort();
    ternNames.forEach((n, mid) => { mats[n] = { mid, N: T[n].N, K: T[n].K }; });
    for (const name in T) {
      const t = T[name];
      if (t.kind === 'embed_f16') embedF16 = new Uint16Array(ab.slice(dataBase + t.off, dataBase + t.off + t.len));
      else if (t.kind === 'f32') nrm[name] = new Float32Array(ab.slice(dataBase + t.off, dataBase + t.off + t.len));
    }
    await initMainWasm();
    buildMainBufs();  // wasm-backed hot buffers + norm weights into main wasm memory
    // SAB: ctrl + act(i8) + xsum(i32) + out(f32) + q(f32 nH*hd) + attn(f32 nH*hd)
    //      + KV(f32, L layers × {K,V} × MAXCTX × nKV*hd) — KV shared so workers can do attention.
    const maxK = CFG.I, maxN = CFG.vocab, H = CFG.H, hd = CFG.hd, nH = CFG.nH, nKV = CFG.nKV, L = CFG.L;
    const kvStride = MAXCTX * nKV * hd;                 // floats per (layer, K or V)
    const CTRL = CTRL_I32 * 4, actOff = CTRL, xsumOff = actOff + maxK, outOff = xsumOff + (maxK / 64) * 4;
    const qOff = outOff + maxN * 4, attnOff = qOff + nH * hd * 4, kvOff = attnOff + nH * hd * 4;
    const sabBytes = kvOff + L * 2 * kvStride * 4;
    const sab = new SharedArrayBuffer(sabBytes);
    ctrl = new Int32Array(sab, 0, CTRL_I32);
    Atomics.store(ctrl, SPIN, (globalThis.__SPIN | 0) || 400);
    sabAct = new Int8Array(sab, actOff, maxK);
    sabXsum = new Int32Array(sab, xsumOff, maxK / 64);
    sabOut = new Float32Array(sab, outOff, maxN);
    sabQ = new Float32Array(sab, qOff, nH * hd);
    sabAttn = new Float32Array(sab, attnOff, nH * hd);
    sabKV = new Float32Array(sab, kvOff, L * 2 * kvStride);
    _kvStride = kvStride;
    // Build each worker's row-partition into its OWN small ArrayBuffer and TRANSFER it
    // (main fetches the 1.16GB binary once; workers never hold the whole thing → no 8×9GB OOM).
    const srcU8 = new Uint8Array(ab);
    const readies = [];
    for (let w = 0; w < Wn; w++) {
      // size this worker's partition
      let bytes = 0; const ents = [];
      for (const name of ternNames) {
        const t = T[name], N = t.N, K = t.K, ng = K / 64, bpr = K / 4;
        const r0 = Math.floor(w * N / Wn), r1 = Math.floor((w + 1) * N / Wn), rows = r1 - r0;
        const cl = rows * bpr, sl = rows * ng * 4;
        ents.push({ mid: mats[name].mid, r0, rows, K, cl, sl, csrc: dataBase + t.codesOff + r0 * bpr, ssrc: dataBase + t.scalesOff + r0 * ng * 4 });
        bytes += cl + sl;
      }
      const part = new Uint8Array(bytes); let o = 0; const layout = [];
      for (const e of ents) {
        const co = o; part.set(srcU8.subarray(e.csrc, e.csrc + e.cl), o); o += e.cl;
        const so = o; part.set(srcU8.subarray(e.ssrc, e.ssrc + e.sl), o); o += e.sl;
        layout.push({ mid: e.mid, r0: e.r0, rows: e.rows, K: e.K, codesOff: co, codesLen: e.cl, scalesOff: so, scalesLen: e.sl });
      }
      const wk = new Worker('cpukern-fwd-worker.js'); workers.push(wk);
      wk.onerror = e => { (globalThis.__wErr = globalThis.__wErr || []).push('onerror w' + wk.__wid + ': ' + (e.message || e)); };
      wk.__wid = w;
      readies.push(new Promise(res => { wk.onmessage = ev => { if (ev.data.workerError) (globalThis.__wErr = globalThis.__wErr || []).push('w' + wk.__wid + ': ' + ev.data.workerError); if (ev.data.ready) res(); }; }));
      wk.postMessage({ cmd: 'init', wid: w, Wn, sab, part: part.buffer, layout, maxK, maxN, actOff, xsumOff, outOff, qOff, attnOff, kvOff, kvStride, nH, nKV, hd }, [part.buffer]);
    }
    await Promise.all(readies);
  }

  // cached scratch views into main-wasm memory (stable — no malloc after buildMainBufs)
  let qDstI8 = null, qXsumI32 = null, qXinF32 = null, qXinOff = 0;
  // Allocate every main-wasm buffer FIRST (malloc only grows + records offsets), THEN create
  // all views against the final buffer — growing detaches earlier views, so views come last.
  function buildMainBufs() {
    const { H, hd, nH, nKV, I, vocab } = CFG, maxK = CFG.I;
    mheap = MW.heap_base();
    const oDst = malloc(maxK), oXsum = malloc((maxK / 64) * 4), oXin = malloc(maxK * 4);
    const oX = malloc(H * 4), oXn = malloc(H * 4), oXn2 = malloc(H * 4), oXf = malloc(H * 4),
      oAttn = malloc(nH * hd * 4), oSwi = malloc(I * 4);
    const normNames = Object.keys(nrm), nO = {};
    for (const nm of normNames) nO[nm] = malloc(nrm[nm].length * 4);
    const buf = MW.memory.buffer;   // final buffer — no more mallocs past here
    qDstI8 = new Int8Array(buf, oDst, maxK); qXsumI32 = new Int32Array(buf, oXsum, maxK / 64);
    qXinF32 = new Float32Array(buf, oXin, maxK); qXinOff = oXin; qDstOff = oDst; qXsumOff = oXsum;
    const wv = (off, n, key) => { const a = new Float32Array(buf, off, n); a.__off = off; return a; };
    B = {
      x: wv(oX, H), xn: wv(oXn, H), xn2: wv(oXn2, H), xf: wv(oXf, H), attn: wv(oAttn, nH * hd), swi: wv(oSwi, I),
      q: new Float32Array(nH * hd), k: new Float32Array(nKV * hd), v: new Float32Array(nKV * hd),
      gate: new Float32Array(I), up: new Float32Array(I), o: new Float32Array(H), logits: new Float32Array(vocab),
      hn: new Float32Array(hd), cosT: new Float32Array(hd / 2), sinT: new Float32Array(hd / 2),
    };
    for (const nm of normNames) { new Float32Array(buf, nO[nm], nrm[nm].length).set(nrm[nm]); nrmOff[nm] = nO[nm]; }
  }

  let P = null; // {quant, barrier, gather, n}
  function profReset() { P = { quant: 0, barrier: 0, gather: 0, n: 0 }; }
  function profGet() { return P; }
  const _now = () => performance.now();
  // Synchronous spin barrier: workers run on their own cores while main polls ARRIVED
  // (~µs vs main-thread waitAsync's ~1.8ms macrotask latency — the whole ballgame at 197/tok).
  function waitBarrier(g) { for (let w = 0; w < Wn; w++) { const s = DONEBASE + w * DONESTRIDE; while (Atomics.load(ctrl, s) !== g) { /* spin */ } } }
  // Batched threaded matmul: `names` (1..3) all consume the SAME input x → quantize x ONCE,
  // dispatch all of them in one barrier, gather each into outs[i]. Workers loop over the batch.
  function mmBatch(names, x, outs) {
    const K = mats[names[0]].K;
    const t0 = P ? _now() : 0;
    let asc;
    if (globalThis.__jsQuant) { // A/B: legacy scalar-JS quant (fused xsum)
      let amax = 0; for (let k = 0; k < K; k++) { const a = x[k] < 0 ? -x[k] : x[k]; if (a > amax) amax = a; }
      asc = amax / 127 || 1e-9; const inv = 1 / asc; const G = CFG.G; let gi = 0, gs = 0;
      for (let k = 0; k < K; k++) { let q = Math.round(x[k] * inv); q = q < -127 ? -127 : q > 127 ? 127 : q; sabAct[k] = q; gs += q; if ((k + 1) % G === 0) { sabXsum[gi++] = gs; gs = 0; } }
    } else {
      // SIMD quant in the main wasm instance (was scalar JS on the critical path). x must live
      // in wasm memory (all matmul inputs are wasm-backed and carry __off); guard-copy otherwise.
      let xo = x.__off; if (xo === undefined) { qXinF32.set(x.subarray(0, K)); xo = qXinOff; }
      asc = MW.quantize(qDstOff, qXsumOff, xo, K);
      sabAct.set(qDstI8.subarray(0, K));            // int8 activation → SAB (small memcpy)
      sabXsum.set(qXsumI32.subarray(0, K / 64));    // per-group sums → SAB
    }
    let off = 0; const offs = [];
    for (let i = 0; i < names.length; i++) { offs.push(off); Atomics.store(ctrl, MATID0 + i, mats[names[i]].mid); Atomics.store(ctrl, OUTOFF0 + i, off); off += mats[names[i]].N; }
    Atomics.store(ctrl, ASC, f2i(asc)); Atomics.store(ctrl, KK, K); Atomics.store(ctrl, NMAT, names.length);
    Atomics.store(ctrl, JOBTYPE, 0);
    const t1 = P ? _now() : 0;
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    waitBarrier(g);
    const t2 = P ? _now() : 0;
    for (let i = 0; i < names.length; i++) outs[i].set(sabOut.subarray(offs[i], offs[i] + mats[names[i]].N));
    if (P) { P.quant += t1 - t0; P.barrier += t2 - t1; P.gather += _now() - t2; P.n++; }
    return outs;
  }
  const mmT = (name, x, out) => mmBatch([name], x, [out])[0];
  // Threaded GQA attention: q (post qk-norm+rope) is in sabQ, K/V history in sabKV[layer];
  // workers split the nH heads, each computes score·softmax·(Σ w·V) → sabAttn. Read into `out`.
  function attnThreaded(l, T, out) {
    const t1 = P ? _now() : 0;
    Atomics.store(ctrl, JOBTYPE, 1); Atomics.store(ctrl, LAYER, l); Atomics.store(ctrl, TLEN, T);
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    waitBarrier(g);
    out.set(sabAttn.subarray(0, CFG.nH * CFG.hd));
    if (P) { P.attn = (P.attn || 0) + (_now() - t1); }
    return out;
  }

  // JS rmsnorm — used only for the small per-head qk-norm (over hd=128, on the non-wasm B.q/B.k).
  // The big H-sized norms go through MW.rmsnorm (SIMD).
  function rmsnorm(x, w, n, out) { let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i]; const inv = 1 / Math.sqrt(s / n + CFG.eps); for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i]; return out; }
  // rope with cos/sin PRECOMPUTED once per token (B.cosT/B.sinT) — they depend only on (pos,i),
  // identical across all 24 q/k heads, so computing them once kills 23/24 of the transcendentals.
  function ropeC(v, off) { const hd = CFG.hd, half = hd / 2, cT = B.cosT, sT = B.sinT; for (let i = 0; i < half; i++) { const c = cT[i], s = sT[i], x0 = v[off + i], x1 = v[off + i + half]; v[off + i] = x0 * c - x1 * s; v[off + i + half] = x1 * c + x0 * s; } }

  let B = null;
  function ensureBufs() { if (!B) throw new Error('CPUEngineMT: call load() before forward()'); }

  function forward(tokenId, pos, kv) {
    const { H, hd, nH, nKV, L, I, eps, theta } = CFG; ensureBufs(); const x = B.x;
    for (let i = 0; i < H; i++) x[i] = f16f(embedF16[tokenId * H + i]);
    // precompute rope cos/sin for this position (shared by all heads)
    { const half = hd / 2; for (let i = 0; i < half; i++) { const a = pos * Math.pow(theta, -(2 * i) / hd); B.cosT[i] = Math.cos(a); B.sinT[i] = Math.sin(a); } }
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      MW.rmsnorm(B.xn.__off, x.__off, nrmOff[p + 'input_layernorm.weight'], H, eps);
      if (globalThis.__noBatch) { mmT(p + 'self_attn.q_proj.weight', B.xn, B.q); mmT(p + 'self_attn.k_proj.weight', B.xn, B.k); mmT(p + 'self_attn.v_proj.weight', B.xn, B.v); }
      else mmBatch([p + 'self_attn.q_proj.weight', p + 'self_attn.k_proj.weight', p + 'self_attn.v_proj.weight'], B.xn, [B.q, B.k, B.v]);
      const qnw = nrm[p + 'self_attn.q_norm.weight'], knw = nrm[p + 'self_attn.k_norm.weight'];
      for (let h = 0; h < nH; h++) { const off = h * hd; rmsnorm(B.q.subarray(off, off + hd), qnw, hd, B.hn); for (let i = 0; i < hd; i++) B.q[off + i] = B.hn[i]; ropeC(B.q, off); }
      for (let h = 0; h < nKV; h++) { const off = h * hd; rmsnorm(B.k.subarray(off, off + hd), knw, hd, B.hn); for (let i = 0; i < hd; i++) B.k[off + i] = B.hn[i]; ropeC(B.k, off); }
      // publish q (post norm+rope) and this token's K,V into the shared buffers, then thread attention
      sabQ.set(B.q);
      const kBase = l * 2 * _kvStride + pos * nKV * hd, vBase = l * 2 * _kvStride + _kvStride + pos * nKV * hd;
      for (let i = 0; i < nKV * hd; i++) { sabKV[kBase + i] = B.k[i]; sabKV[vBase + i] = B.v[i]; }
      attnThreaded(l, pos + 1, B.attn);
      mmT(p + 'self_attn.o_proj.weight', B.attn, B.o);
      for (let i = 0; i < H; i++) x[i] += B.o[i];
      MW.rmsnorm(B.xn2.__off, x.__off, nrmOff[p + 'post_attention_layernorm.weight'], H, eps);
      if (globalThis.__noBatch) { mmT(p + 'mlp.gate_proj.weight', B.xn2, B.gate); mmT(p + 'mlp.up_proj.weight', B.xn2, B.up); }
      else mmBatch([p + 'mlp.gate_proj.weight', p + 'mlp.up_proj.weight'], B.xn2, [B.gate, B.up]);
      for (let i = 0; i < I; i++) { const g = B.gate[i]; B.swi[i] = (g / (1 + Math.exp(-g))) * B.up[i]; }
      mmT(p + 'mlp.down_proj.weight', B.swi, B.o);
      for (let i = 0; i < H; i++) x[i] += B.o[i];
    }
    MW.rmsnorm(B.xf.__off, x.__off, nrmOff['model.norm.weight'], H, eps);
    mmT('lm_head.weight', B.xf, B.logits);
    return B.logits;
  }
  function newKV() { return Array.from({ length: CFG.L }, () => ({ k: [], v: [] })); }
  function argmax(a) { let bi = 0, bv = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; }
  function stop() { Atomics.store(ctrl, NMAT, -1); Atomics.add(ctrl, GEN, 1); Atomics.notify(ctrl, GEN); setTimeout(() => workers.forEach(w => w.terminate()), 50); }

  const setSpin = (v) => { if (ctrl) Atomics.store(ctrl, SPIN, v | 0); };
  // per-worker gemv busy-µs accumulators (slot DONEBASE+w*STRIDE+1) — load-balance probe
  const busyReset = () => { for (let w = 0; w < Wn; w++) Atomics.store(ctrl, DONEBASE + w * DONESTRIDE + 1, 0); };
  const busyTimes = () => { const a = []; for (let w = 0; w < Wn; w++) a.push(Atomics.load(ctrl, DONEBASE + w * DONESTRIDE + 1)); return a; };
  return { load, forward, newKV, argmax, stop, CFG, _mats: mats, profReset, profGet, setSpin, busyReset, busyTimes };
})();
if (typeof window !== 'undefined') window.CPUEngineMT = CPUEngineMT;
