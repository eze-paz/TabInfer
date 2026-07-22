// Forward-pool worker: holds a contiguous ROW-PARTITION of every ternary matrix
// (rows [r0,r1) of matrix mid). On each barrier release it runs gemv_tern over its
// rows of the requested matrix, reading the shared int8 activation, and writes its
// output rows (×asc) into the shared output buffer. Weights partitioned → no dup.
// ctrl slots: one cache line per contended word (see cpuengine-mt.js) — false sharing
// in the packed v1 layout cost ~1ms/barrier.
const GEN = 0, NMAT = 16, KK = 17, ASC = 18, MATID0 = 19, OUTOFF0 = 22, JOBTYPE = 25, LAYER = 26, TLEN = 27, SPIN = 28, BATCH = 29, BASEPOS = 30, PHASE = 31, DONEBASE = 32, DONESTRIDE = 16, CTRL_I32 = 32 + 16 * 16;
const MAXCTX = 512;   // per-worker int8 KV capacity (must match coordinator's context ceiling)
let W = null, wid = 0, Wn = 1, aParts = 1;   // aParts = phase-barrier participants (Wn + main)
let ctrl, sabAsc, sabAct, sabXsum, sabOut, sabQ, sabAttn, sabKcur, sabVcur, sabCos, sabSin;
let aNH, aNKV, aHD, aScale, aEps, aMaxK, aMaxN, aBMAX;   // attention/batch params
let A = null;                               // int8-attention state (buffers, owned kvh KV stores)
let sAct, sXsum, sOutLocal;                 // wasm scratch offsets
const meta = [];                            // mid -> {oCodes,oScales,r0,rows,K}
const _b = new ArrayBuffer(4), _bf = new Float32Array(_b), _bi = new Int32Array(_b);
const i32f = (v) => { _bi[0] = v; return _bf[0]; };

const align16 = x => (x + 15) & ~15;
// rope in-place on an hd-length wasm f32 view, using column `cb`'s cos/sin (cb=col*hd/2)
function ropeV(v, cb) { const half = aHD >> 1; for (let i = 0; i < half; i++) { const c = sabCos[cb + i], s = sabSin[cb + i], x0 = v[i], x1 = v[i + half]; v[i] = x0 * c - x1 * s; v[i + half] = x1 * c + x0 * s; } }
// int8 GQA attention over B columns. q read from qA[qBase + c*(nH*hd) + h*hd]; k/v from
// {k,v}A[{k,v}Base + c*(nKV*hd) + kvh*hd]. Applies qk-norm+rope, appends k/v to per-(layer,kvh)
// int8 KV, SIMD scores/softmax/weighted-V → sabAttn[c*nH*hd + h*hd]. Shared by JOBTYPE 1 & 2.
function doAttn(qA, qBase, kA, kBase, vA, vBase) {
  const l = Atomics.load(ctrl, LAYER), B = Atomics.load(ctrl, BATCH) || 1, p0 = Atomics.load(ctrl, BASEPOS);
  const hd = aHD, half = hd >> 1, nH = aNH, nKV = aNKV, qpk = A.qpk, scale = aScale, qcs = nH * hd, kcs = nKV * hd;
  const lqn = A.qnOff + l * hd * 4, lkn = A.knOff + l * hd * 4;
  const seen = {};
  for (let h = A.h0; h < A.h1; h++) {
    const kvh = (h / qpk) | 0; if (seen[kvh]) continue; seen[kvh] = 1;
    const st = A.kv[l * nKV + kvh];
    for (let c = 0; c < B; c++) {
      const pos = p0 + c; if (pos >= MAXCTX) break;
      const ko = kBase + c * kcs + kvh * hd;
      A.kfV.set(kA.subarray(ko, ko + hd)); W.rmsnorm(A.kfOff, A.kfOff, lkn, hd, aEps); ropeV(A.kfV, c * half);
      st.ksV[pos] = W.quant_vec(st.ki8 + pos * hd, A.kfOff, hd);
      const vo = vBase + c * kcs + kvh * hd;
      A.vfV.set(vA.subarray(vo, vo + hd)); st.vsV[pos] = W.quant_vec(st.vi8 + pos * hd, A.vfOff, hd);
    }
  }
  for (let h = A.h0; h < A.h1; h++) {
    const kvh = (h / qpk) | 0, st = A.kv[l * nKV + kvh];
    for (let c = 0; c < B; c++) {
      const T = p0 + c + 1, qo = qBase + c * qcs + h * hd;
      A.qfV.set(qA.subarray(qo, qo + hd)); W.rmsnorm(A.qfOff, A.qfOff, lqn, hd, aEps); ropeV(A.qfV, c * half);
      const qs = W.quant_vec(A.qbufOff, A.qfOff, hd);
      W.attn_scores(A.scoreOff, A.qbufOff, qs, st.ki8, st.ks, T, hd, scale);
      const sc = A.scoreV; let mx = -Infinity; for (let t = 0; t < T; t++) { const v = sc[t]; if (v > mx) mx = v; }
      let sum = 0; const wv = A.wV; for (let t = 0; t < T; t++) { const e = Math.exp(sc[t] - mx); wv[t] = e; sum += e; }
      const isum = 1 / sum; for (let t = 0; t < T; t++) wv[t] *= isum;
      W.attn_accv(A.outOff, A.wOff, st.vi8, st.vs, T, hd);
      sabAttn.set(A.outV.subarray(0, hd), c * nH * hd + h * hd);
    }
  }
}
async function initWasm() {
  const buf = await (await fetch('cpukern.wasm')).arrayBuffer();
  W = (await WebAssembly.instantiate(buf, {})).instance.exports;
}
let heapTop = 0;
function gr0(bytes) {
  const off = heapTop; heapTop = align16(heapTop + bytes);
  const need = Math.ceil(heapTop / 65536) + 1, have = W.memory.buffer.byteLength / 65536;
  if (need > have) W.memory.grow(need - have);
  return off;
}

async function setup(msg) {
  ({ wid, Wn } = msg);
  const sab = msg.sab;
  ctrl = new Int32Array(sab, 0, CTRL_I32);
  const B = msg.BMAX, nH = msg.nH, nKV = msg.nKV, hd = msg.hd;
  sabAsc = new Float32Array(sab, msg.ascOff, B);
  sabAct = new Int8Array(sab, msg.actOff, B * msg.maxK);
  sabXsum = new Int32Array(sab, msg.xsumOff, B * (msg.maxK / 64));
  sabOut = new Float32Array(sab, msg.outOff, B * msg.maxN);
  sabQ = new Float32Array(sab, msg.qOff, B * nH * hd);
  sabAttn = new Float32Array(sab, msg.attnOff, B * nH * hd);
  sabKcur = new Float32Array(sab, msg.kcurOff, B * nKV * hd);
  sabVcur = new Float32Array(sab, msg.vcurOff, B * nKV * hd);
  sabCos = new Float32Array(sab, msg.cosOff, B * (hd / 2));
  sabSin = new Float32Array(sab, msg.sinOff, B * (hd / 2));
  aNH = nH; aNKV = nKV; aHD = hd; aScale = 1 / Math.sqrt(hd); aEps = msg.eps; aMaxK = msg.maxK; aMaxN = msg.maxN; aBMAX = B;
  aParts = Wn + (msg.mainOn ? 1 : 0);   // fused-job worker-side barrier participant count
  await initWasm();
  heapTop = align16(W.heap_base());
  // receive my pre-extracted row-partition (transferred ArrayBuffer) + layout
  const part = new Uint8Array(msg.part);
  for (const e of msg.layout) {
    const oCodes = gr0(e.codesLen), oScales = gr0(e.scalesLen);
    if (e.rows > 0) {
      new Uint8Array(W.memory.buffer, oCodes, e.codesLen).set(part.subarray(e.codesOff, e.codesOff + e.codesLen));
      new Uint8Array(W.memory.buffer, oScales, e.scalesLen).set(part.subarray(e.scalesOff, e.scalesOff + e.scalesLen));
    }
    meta[e.mid] = { oCodes, oScales, r0: e.r0, rows: e.rows, K: e.K };
  }
  // gemv scratch
  sAct = gr0(msg.maxK); sXsum = gr0((msg.maxK / 64) * 4);
  sOutLocal = gr0(Math.ceil(msg.maxN / Wn + 64) * 4);
  // ── int8 attention state: scratch + a private int8 KV store per (layer, owned kvh) ──
  // KV is PER-LAYER (each layer has its own K/V history) — key = l*nKV + kvh.
  const L = msg.L, qpk = nH / nKV;
  const h0 = Math.floor(wid * nH / Wn), h1 = Math.floor((wid + 1) * nH / Wn);
  const qbufOff = gr0(hd), qfOff = gr0(hd * 4), kfOff = gr0(hd * 4), vfOff = gr0(hd * 4);
  const scoreOff = gr0(MAXCTX * 4), wOff = gr0(MAXCTX * 4), outOff = gr0(hd * 4);
  const qnOff = gr0(L * hd * 4), knOff = gr0(L * hd * 4);   // qk-norm weights (all layers)
  const ownedKvh = []; { const seen = {}; for (let h = h0; h < h1; h++) { const kvh = (h / qpk) | 0; if (!seen[kvh]) { seen[kvh] = 1; ownedKvh.push(kvh); } } }
  const kv = {};   // (l*nKV+kvh) -> {ki8,ks,vi8,vs} offsets
  for (let l = 0; l < L; l++) for (const kvh of ownedKvh) kv[l * nKV + kvh] = { ki8: gr0(MAXCTX * hd), ks: gr0(MAXCTX * 4), vi8: gr0(MAXCTX * hd), vs: gr0(MAXCTX * 4) };
  const buf = W.memory.buffer;   // memory stable after here (no more gr0)
  new Float32Array(buf, qnOff, L * hd).set(msg.qNormAll);
  new Float32Array(buf, knOff, L * hd).set(msg.kNormAll);
  for (const k in kv) { kv[k].ksV = new Float32Array(buf, kv[k].ks, MAXCTX); kv[k].vsV = new Float32Array(buf, kv[k].vs, MAXCTX); }
  A = { h0, h1, qpk, hd, nKV, qbufOff, qfOff, kfOff, vfOff, scoreOff, wOff, outOff, qnOff, knOff, kv,
    qfV: new Float32Array(buf, qfOff, hd), kfV: new Float32Array(buf, kfOff, hd), vfV: new Float32Array(buf, vfOff, hd),
    scoreV: new Float32Array(buf, scoreOff, MAXCTX), wV: new Float32Array(buf, wOff, MAXCTX), outV: new Float32Array(buf, outOff, hd) };
  postMessage({ ready: true });
  // barrier loop — SPIN briefly (hot path: next matmul arrives within ~100µs), then PARK
  // via Atomics.wait. Pure spin made orphaned pools burn 100% CPU forever (the bug that
  // poisoned a whole benchmarking session); parked waiters cost nothing and the ~µs spin
  // window still catches the hot case without a thread wake.
  let gen = 0;
  for (;;) {
    // Spin LONG enough to stay hot across a whole token's dispatch sequence. A token issues
    // ~5 barriers/layer × 28 = ~140 dispatches; between two consecutive ones main runs only
    // small vector ops (norm/rope/swiglu/quant, ~0.1-0.4ms). Parking after ~200 spins (~µs)
    // meant EVERY dispatch found the worker parked → ~200-400µs wake × 140 = ~50ms/token of
    // pure wake latency (measured: 71ms/token floor at 23% util = workers idle 77%). Attention
    // is now threaded onto the workers themselves, so the old "spinners starve main's serial
    // attention" regression no longer applies — the remaining main serial work is small. So
    // spin ~1-2ms worth of iterations (SPINLIM) before parking; forwards run back-to-back
    // during generation so the pool stays hot the whole time, and an orphaned/between-turns
    // pool still parks after one idle window (SPINLIM iters) instead of burning CPU forever.
    // PARK-FAST wins on this 2P+8E chip: hot-spinning (SPINLIM 250k) pushed util 23→62% but
    // REGRESSED tok/s 71→133ms — all-core activity kills P-core turbo and saturates memory
    // bandwidth, starving the critical-path main-thread serial work. Park quickly; eat the
    // ~200µs wake per dispatch instead. (Measured 2026-07-22.)
    const SPINLIM = Atomics.load(ctrl, SPIN) || 400;
    let spins = 0;
    while (Atomics.load(ctrl, GEN) === gen) {
      if (++spins > SPINLIM) { Atomics.wait(ctrl, GEN, gen); spins = 0; }
    }
    gen = Atomics.load(ctrl, GEN);
    const nmat = Atomics.load(ctrl, NMAT);
    if (nmat < 0) return;
    const jt = Atomics.load(ctrl, JOBTYPE);
    if (jt === 1) {  // standalone attention (used by batched forwardN): q←sabQ, k←sabKcur, v←sabVcur
      doAttn(sabQ, 0, sabKcur, 0, sabVcur, 0);
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    if (jt === 2) {  // FUSED qkv-gemv + attention (single-token). Phase 1 gemv → sabOut; worker-side
      // spin-barrier (no main round-trip); phase 2 attention reads q/k/v directly from sabOut.
      const K = Atomics.load(ctrl, KK), nmat2 = Atomics.load(ctrl, NMAT), asc = sabAsc[0];
      const sActV = new Int8Array(W.memory.buffer, sAct, K), sXsumV = new Int32Array(W.memory.buffer, sXsum, K / 64);
      const loV = new Float32Array(W.memory.buffer, sOutLocal, Math.ceil(aMaxN / Wn) + 64);
      sActV.set(sabAct.subarray(0, K)); sXsumV.set(sabXsum.subarray(0, K / 64));
      for (let b = 0; b < nmat2; b++) {
        const m = meta[Atomics.load(ctrl, MATID0 + b)], outOff = Atomics.load(ctrl, OUTOFF0 + b);
        if (m.rows > 0) { W.gemv_tern(sOutLocal, m.oCodes, m.oScales, sAct, sXsum, m.rows, m.K); for (let i = 0; i < m.rows; i++) sabOut[outOff + m.r0 + i] = loV[i] * asc; }
      }
      Atomics.add(ctrl, PHASE, 1);
      while (Atomics.load(ctrl, PHASE) < aParts) { /* cheap: all workers hot */ }
      const Nq = aNH * aHD, Nk = aNKV * aHD;   // q at sabOut[0..Nq), k at [Nq..), v at [Nq+Nk..)
      doAttn(sabOut, 0, sabOut, Nq, sabOut, Nq + Nk);
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    const K = Atomics.load(ctrl, KK), B = Atomics.load(ctrl, BATCH) || 1, maxN = aMaxN, ng = K / 64;
    const _t0 = performance.now();
    const sActV = new Int8Array(W.memory.buffer, sAct, K), sXsumV = new Int32Array(W.memory.buffer, sXsum, ng);
    const loV = new Float32Array(W.memory.buffer, sOutLocal, Math.ceil(maxN / Wn) + 64);
    // B columns (speculative verify) share ONE barrier; column 0 == single-token path.
    for (let c = 0; c < B; c++) {
      sActV.set(sabAct.subarray(c * K, c * K + K));
      sXsumV.set(sabXsum.subarray(c * ng, c * ng + ng));
      const asc = sabAsc[c], obase = c * maxN;
      for (let b = 0; b < nmat; b++) {
        const m = meta[Atomics.load(ctrl, MATID0 + b)], outOff = Atomics.load(ctrl, OUTOFF0 + b);
        if (m.rows > 0) {
          W.gemv_tern(sOutLocal, m.oCodes, m.oScales, sAct, sXsum, m.rows, m.K);
          for (let i = 0; i < m.rows; i++) sabOut[obase + outOff + m.r0 + i] = loV[i] * asc;
        }
      }
    }
    Atomics.add(ctrl, DONEBASE + wid * DONESTRIDE + 1, ((performance.now() - _t0) * 1000) | 0);
    Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);   // own cache line — no false sharing
  }
}
self.onerror = (m, src, ln, col, err) => { try { postMessage({ workerError: String(err && err.stack || m) + ' @' + ln + ':' + col }); } catch (_) {} };
onmessage = (e) => { if (e.data.cmd === 'init') setup(e.data).catch(err => { try { postMessage({ workerError: 'setup: ' + String(err && err.stack || err) }); } catch (_) {} }); };
