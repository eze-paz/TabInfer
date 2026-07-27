// Forward-pool worker: holds a contiguous ROW-PARTITION of every ternary matrix
// (rows [r0,r1) of matrix mid). On each barrier release it runs gemv_tern over its
// rows of the requested matrix, reading the shared int8 activation, and writes its
// output rows (×asc) into the shared output buffer. Weights partitioned → no dup.
// ctrl slots: one cache line per contended word (see cpuengine-mt.js) — false sharing
// in the packed v1 layout cost ~1ms/barrier.
const GEN = 0, NMAT = 16, KK = 17, ASC = 18, MATID0 = 19, OUTOFF0 = 22, JOBTYPE = 25, LAYER = 26, TLEN = 27, SPIN = 28, BATCH = 29, BASEPOS = 30, PHASE = 31, DONEBASE = 32, DONESTRIDE = 16, PBCNT = 288, PBGEN = 304, CURSOR = 320, RESID = 336, AMAX = 337, CUR_A = 352, CUR_B = 368, CUR_C = 384, CUR_D = 400, CUR_E = 416, GLUEGEN = 432, CTRL_I32 = 32 + 16 * 16 + 160;
const CH = 128; // work-stealing chunk rows (shared-weights mode)
const MAXCTX = 2048;   // per-worker int8 KV capacity (must match coordinator's context ceiling)
let W = null, wid = 0, Wn = 1, aParts = 1;   // aParts = phase-barrier participants (Wn + main)
let ctrl, sabAsc, sabAct, sabXsum, sabOut, sabQ, sabAttn, sabKcur, sabVcur, sabCos, sabSin, sabX, argValV, argIdxV;
let aNH, aNKV, aHD, aScale, aEps, aMaxK, aMaxN, aBMAX;   // attention/batch params
let A = null;                               // int8-attention state (buffers, owned kvh KV stores)
let sAct, sXsum, sOutLocal;                 // wasm scratch offsets
// MEGA-token state: per-layer matrix mids, norm-weight wasm offsets, private glue scratch + views
let MEGA = null, pXOff = 0, pSwiOff = 0, pXv = null, pSwiV = null, sActV = null, sXsumV = null, loV = null;
// CHUNKED shared-weights mode: second wasm instance bound to the SHARED memory holding ALL
// weights; gemv rows are claimed dynamically (CURSOR) so fast cores take more chunks.
let SW = null, swLayout = null, swActOff = 0, swXsumOff = 0, swOutOff = 0, swOutV = null;
// increment-2 shared regions: residual x, shared-KV attention, per-participant scratch
let sxV = null, swAttnV = null, swF32 = null, SH = null, SP = null;
const meta = [];                            // mid -> {oCodes,oScales,r0,rows,K}
const _b = new ArrayBuffer(4), _bf = new Float32Array(_b), _bi = new Int32Array(_b);
const i32f = (v) => { _bi[0] = v; return _bf[0]; };

const align16 = x => (x + 15) & ~15;
// rope in-place on an hd-length wasm f32 view, using column `cb`'s cos/sin (cb=col*hd/2)
function ropeV(v, cb) { const half = aHD >> 1; for (let i = 0; i < half; i++) { const c = sabCos[cb + i], s = sabSin[cb + i], x0 = v[i], x1 = v[i + half]; v[i] = x0 * c - x1 * s; v[i + half] = x1 * c + x0 * s; } }
// int8 GQA attention over B columns. q read from qA[qBase + c*(nH*hd) + h*hd]; k/v from
// {k,v}A[{k,v}Base + c*(nKV*hd) + kvh*hd]. Applies qk-norm+rope, appends k/v to per-(layer,kvh)
// int8 KV, SIMD scores/softmax/weighted-V → sabAttn[c*nH*hd + h*hd]. Shared by JOBTYPE 1 & 2.
function doAttn(qA, qBase, kA, kBase, vA, vBase, l, p0, B) {
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
// Worker-side sense-reversing phase barrier (mega-token). Inter-phase gaps are µs (worker
// imbalance ~1.01, no main in the loop), so spin briefly; park-fallback only for OS noise.
// Last arriver resets the counter THEN bumps the generation (seq-cst order guarantees no
// next-phase arrival can see a stale counter), and notifies any parked stragglers.
function pbar() {
  const g = Atomics.load(ctrl, PBGEN);
  if (Atomics.add(ctrl, PBCNT, 1) + 1 === Wn) {
    Atomics.store(ctrl, PBCNT, 0);
    Atomics.add(ctrl, PBGEN, 1);
    Atomics.notify(ctrl, PBGEN);
  } else {
    const t0 = performance.now();
    let s = 0, parks = 0;
    while (Atomics.load(ctrl, PBGEN) === g) { if (++s > 30000) { parks++; Atomics.wait(ctrl, PBGEN, g); s = 0; } }
    // debug accumulators (own cache line): +3 total wait µs, +4 park-fallback count
    Atomics.add(ctrl, DONEBASE + wid * DONESTRIDE + 3, ((performance.now() - t0) * 1000) | 0);
    if (parks) Atomics.add(ctrl, DONEBASE + wid * DONESTRIDE + 4, parks);
  }
}
// In-job barrier for JOBTYPE=7 mega-layer: Wn+1 participants (workers + main).
function pbarT(target) {
  const g = Atomics.load(ctrl, PBGEN);
  if (Atomics.add(ctrl, PBCNT, 1) + 1 === target) {
    Atomics.store(ctrl, PBCNT, 0);
    Atomics.add(ctrl, PBGEN, 1);
    Atomics.notify(ctrl, PBGEN);
  } else {
    let s = 0;
    while (Atomics.load(ctrl, PBGEN) === g) { if (++s > 30000) { Atomics.wait(ctrl, PBGEN, g); s = 0; } }
  }
}
// wait for main's glue bump past `gTarget` (glue is µs-scale wasm — spin, park fallback)
function glueWaitW(gTarget) {
  let s = 0;
  for (;;) { const v = Atomics.load(ctrl, GLUEGEN); if (v >= gTarget) return; if (++s > 30000) { Atomics.wait(ctrl, GLUEGEN, v); s = 0; } }
}
// work-stolen gemv chunks over SHARED weights from cursor `curSlot`; resid folds ×asc into
// shared x; amax tracks this worker's argmax candidate. asc read from sabAsc[0] at call time.
function claimChunksW(mids, offs4, curSlot, resid, amax) {
  const asc = sabAsc[0];
  const nchs = mids.map(m => Math.ceil(swLayout[m].N / CH));
  let tot = 0; for (const n2 of nchs) tot += n2;
  const _t4 = performance.now();
  let bv = -Infinity, bi = 0;
  for (;;) {
    const c = Atomics.add(ctrl, curSlot, 1);
    if (c >= tot) break;
    let b = 0, rem = c; while (rem >= nchs[b]) { rem -= nchs[b]; b++; }
    const Lw = swLayout[mids[b]], row0 = rem * CH, rows = Math.min(CH, Lw.N - row0), ngw = Lw.K / 64;
    if (SH.lut) SW.gemv_lut_tern_s(swOutOff + (offs4[b] + row0) * 4, Lw.widxOff + rem * Lw.cw, Lw.scalesBOff + rem * Lw.cs, SH.tblOff, rows, Lw.K);
    else SW.gemv_tern(swOutOff + (offs4[b] + row0) * 4, Lw.codesOff + row0 * (Lw.K / 4), Lw.scalesOff + row0 * ngw * 4, swActOff, swXsumOff, rows, Lw.K);
    const ob = offs4[b] + row0;
    if (resid) { for (let i = 0; i < rows; i++) sxV[ob + i] += swOutV[ob + i] * asc; }
    else if (amax) { for (let i = 0; i < rows; i++) { const v = swOutV[ob + i] * asc; swOutV[ob + i] = v; if (v > bv) { bv = v; bi = ob + i; } } }
    else { for (let i = 0; i < rows; i++) swOutV[ob + i] *= asc; }
  }
  if (amax) { argValV[wid] = bv; argIdxV[wid] = bi; }
  Atomics.add(ctrl, DONEBASE + wid * DONESTRIDE + 1, ((performance.now() - _t4) * 1000) | 0);
}
// work-stolen shared-KV attention units (one unit = one kvh: append k/v, then its heads)
function attnUnitsW(l, pos, curSlot) {
  const T = pos + 1, hd = aHD, nKV = aNKV, qpk = aNH / nKV, scale = aScale;
  const Nq = aNH * hd, Nk = nKV * hd;
  for (;;) {
    const kvh = Atomics.add(ctrl, curSlot, 1);
    if (kvh >= nKV) break;
    const kb = SH.kvOff + (l * nKV + kvh) * SH.kvStride;
    const ki8 = kb, ks = kb + SH.kvKS, vi8 = kb + SH.kvVI, vs = kb + SH.kvVS;
    if (pos < MAXCTX) {
      SP.kfV.set(swOutV.subarray(Nq + kvh * hd, Nq + kvh * hd + hd));
      SW.rmsnorm(SP.kfOff, SP.kfOff, SH.qknK + l * hd * 4, hd, aEps); ropeV(SP.kfV, 0);
      swF32[(ks >> 2) + pos] = SW.quant_vec(ki8 + pos * hd, SP.kfOff, hd);
      SP.kfV.set(swOutV.subarray(Nq + Nk + kvh * hd, Nq + Nk + kvh * hd + hd));
      swF32[(vs >> 2) + pos] = SW.quant_vec(vi8 + pos * hd, SP.kfOff, hd);
    }
    for (let h = kvh * qpk; h < (kvh + 1) * qpk; h++) {
      SP.qfV.set(swOutV.subarray(h * hd, h * hd + hd));
      SW.rmsnorm(SP.qfOff, SP.qfOff, SH.qknQ + l * hd * 4, hd, aEps); ropeV(SP.qfV, 0);
      const qs = SW.quant_vec(SP.qi8Off, SP.qfOff, hd);
      SW.attn_scores(SP.scOff, SP.qi8Off, qs, ki8, ks, T, hd, scale);
      const sc = SP.scV; let mx = -Infinity; for (let t = 0; t < T; t++) { const v = sc[t]; if (v > mx) mx = v; }
      let sum = 0; for (let t = 0; t < T; t++) { const e = Math.exp(sc[t] - mx); sc[t] = e; sum += e; }
      const isum = 1 / sum; for (let t = 0; t < T; t++) sc[t] *= isum;
      SW.attn_accv(SH.attnOff2 + h * hd * 4, SP.scOff, vi8, vs, T, hd);
    }
  }
}
// gemv my rows of matrix `mid` from the private quantized act (sAct/sXsum), write ×asc into
// sabOut[outBase + myRows]; resid variant adds into sabX[myRows] instead (o/down projections).
function gemvRows(mid, outBase, asc) {
  const m = meta[mid]; if (m.rows <= 0) return;
  W.gemv_tern(sOutLocal, m.oCodes, m.oScales, sAct, sXsum, m.rows, m.K);
  for (let i = 0; i < m.rows; i++) sabOut[outBase + m.r0 + i] = loV[i] * asc;
}
function gemvRowsResid(mid, asc) {
  const m = meta[mid]; if (m.rows <= 0) return;
  W.gemv_tern(sOutLocal, m.oCodes, m.oScales, sAct, sXsum, m.rows, m.K);
  for (let i = 0; i < m.rows; i++) sabX[m.r0 + i] += loV[i] * asc;
}
async function initWasm() {
  const buf = await (await fetch('cpukern.wasm?v=2')).arrayBuffer();
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
  // ── MEGA-token scratch: layer norm weights (input/post/final) + private glue buffers ──
  const H = msg.mega.H, I = msg.mega.I;
  const nrmBase = gr0((2 * L + 1) * H * 4);
  pXOff = gr0(H * 4); pSwiOff = gr0(I * 4);
  const buf = W.memory.buffer;   // memory stable after here (no more gr0)
  new Float32Array(buf, qnOff, L * hd).set(msg.qNormAll);
  new Float32Array(buf, knOff, L * hd).set(msg.kNormAll);
  new Float32Array(buf, nrmBase, (2 * L + 1) * H).set(msg.normsAll);
  MEGA = { H, I, L, layers: msg.mega.layers, lm: msg.mega.lm, inOff: nrmBase, postOff: nrmBase + L * H * 4, finOff: nrmBase + 2 * L * H * 4 };
  pXv = new Float32Array(buf, pXOff, H); pSwiV = new Float32Array(buf, pSwiOff, I);
  sActV = new Int8Array(buf, sAct, msg.maxK); sXsumV = new Int32Array(buf, sXsum, msg.maxK / 64);
  loV = new Float32Array(buf, sOutLocal, Math.ceil(msg.maxN / Wn) + 64);
  sabX = new Float32Array(sab, msg.xOff, H);
  argValV = new Float32Array(sab, msg.argValOff, 16); argIdxV = new Int32Array(sab, msg.argIdxOff, 16);
  for (const k in kv) { kv[k].ksV = new Float32Array(buf, kv[k].ks, MAXCTX); kv[k].vsV = new Float32Array(buf, kv[k].vs, MAXCTX); }
  A = { h0, h1, qpk, hd, nKV, qbufOff, qfOff, kfOff, vfOff, scoreOff, wOff, outOff, qnOff, knOff, kv,
    qfV: new Float32Array(buf, qfOff, hd), kfV: new Float32Array(buf, kfOff, hd), vfV: new Float32Array(buf, vfOff, hd),
    scoreV: new Float32Array(buf, scoreOff, MAXCTX), wV: new Float32Array(buf, wOff, MAXCTX), outV: new Float32Array(buf, outOff, hd) };
  if (msg.shared) {   // chunked shared-weights mode: bind a second instance to the SHARED memory
    const smod = await WebAssembly.compile(await (await fetch('cpukern-shared.wasm?v=2')).arrayBuffer());
    const inst = await WebAssembly.instantiate(smod, { env: { memory: msg.shared.mem } });
    inst.exports.__stack_pointer.value = msg.shared.stackTop;   // distinct stack per instance
    SW = inst.exports;
    swLayout = msg.shared.layout;
    swActOff = msg.shared.actOff; swXsumOff = msg.shared.xsumOff; swOutOff = msg.shared.outOff;
    const sbuf = msg.shared.mem.buffer;
    swOutV = new Float32Array(sbuf, swOutOff, msg.shared.outLen);
    SH = msg.shared;   // {sxOff, attnOff2, kvOff, kvStride, pscrBase, pscrStride, qknQ, qknK, ...}
    sxV = new Float32Array(sbuf, SH.sxOff, msg.mega.H);
    swAttnV = new Float32Array(sbuf, SH.attnOff2, nH * hd);
    swF32 = new Float32Array(sbuf, 0, sbuf.byteLength >> 2);
    const ps = SH.pscrBase + wid * SH.pscrStride;
    SP = { kfOff: ps, qfOff: ps + 512, qi8Off: ps + 1024, scOff: ps + 4096,
      kfV: new Float32Array(sbuf, ps, hd), qfV: new Float32Array(sbuf, ps + 512, hd), scV: new Float32Array(sbuf, ps + 4096, MAXCTX) };
  }
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
      doAttn(sabQ, 0, sabKcur, 0, sabVcur, 0, Atomics.load(ctrl, LAYER), Atomics.load(ctrl, BASEPOS), Atomics.load(ctrl, BATCH) || 1);
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
      doAttn(sabOut, 0, sabOut, Nq, sabOut, Nq + Nk, Atomics.load(ctrl, LAYER), Atomics.load(ctrl, BASEPOS), Atomics.load(ctrl, BATCH) || 1);
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    if (jt === 3) {
      // MEGA-TOKEN: the ENTIRE forward in ONE dispatch. Workers self-sequence through all
      // phases via µs spin phase-barriers (pbar) — no main round-trips. Serial glue
      // (rmsnorm/quant/swiglu) is computed REDUNDANTLY by every worker (~tens of µs each,
      // SIMD, parallel — wall-cost same as main doing it once, but zero dispatch machinery).
      // Main only writes embed(x)+cos/sin before the GEN bump and reduces argmax after.
      const pos = Atomics.load(ctrl, BASEPOS);
      const { H, I, L, layers, inOff, postOff, finOff } = MEGA;
      const Nq = aNH * aHD, Nk = aNKV * aHD;
      const PROG = DONEBASE + wid * DONESTRIDE + 2;   // debug: my layer*8+phase progress word
      for (let l = 0; l < L; l++) {
        const M = layers[l];
        Atomics.store(ctrl, PROG, l * 8 + 1);
        // (a) xn = rmsnorm(x, input_norm); quant → private act; qkv gemv on my rows
        pXv.set(sabX);
        W.rmsnorm(pXOff, pXOff, inOff + l * H * 4, H, aEps);
        const a1 = W.quantize(sAct, sXsum, pXOff, H);
        gemvRows(M.q, 0, a1); gemvRows(M.k, Nq, a1); gemvRows(M.v, Nq + Nk, a1);
        pbar();                                    // full q/k/v visible
        Atomics.store(ctrl, PROG, l * 8 + 2);
        // (b) attention on my heads (q/k/v straight from sabOut; private per-layer int8 KV)
        doAttn(sabOut, 0, sabOut, Nq, sabOut, Nq + Nk, l, pos, 1);
        pbar();                                    // full attn visible
        Atomics.store(ctrl, PROG, l * 8 + 3);
        // (c) o_proj: quant(attn) → my rows → residual-add into my x rows
        pXv.set(sabAttn.subarray(0, H));           // nH*hd == H
        const a2 = W.quantize(sAct, sXsum, pXOff, H);
        gemvRowsResid(M.o, a2);
        pbar();                                    // full x updated
        Atomics.store(ctrl, PROG, l * 8 + 4);
        // (d) xn2 = rmsnorm(x, post_norm); gate/up gemv on my rows
        pXv.set(sabX);
        W.rmsnorm(pXOff, pXOff, postOff + l * H * 4, H, aEps);
        const a3 = W.quantize(sAct, sXsum, pXOff, H);
        gemvRows(M.gate, 0, a3); gemvRows(M.up, I, a3);
        pbar();                                    // full gate/up visible
        Atomics.store(ctrl, PROG, l * 8 + 5);
        // (e) swiglu (redundant, full I) → quant → down gemv → residual into my x rows
        for (let i = 0; i < I; i++) { const g2 = sabOut[i]; pSwiV[i] = (g2 / (1 + Math.exp(-g2))) * sabOut[I + i]; }
        const a4 = W.quantize(sAct, sXsum, pSwiOff, I);
        gemvRowsResid(M.down, a4);
        pbar();                                    // full x updated — layer done
      }
      Atomics.store(ctrl, PROG, 9999);
      // final norm + my lm_head slice + my argmax candidate
      pXv.set(sabX);
      W.rmsnorm(pXOff, pXOff, finOff, H, aEps);
      const a5 = W.quantize(sAct, sXsum, pXOff, H);
      const lm = meta[MEGA.lm];
      let bv = -Infinity, bi = 0;
      if (lm.rows > 0) {
        W.gemv_tern(sOutLocal, lm.oCodes, lm.oScales, sAct, sXsum, lm.rows, lm.K);
        for (let i = 0; i < lm.rows; i++) { const v = loV[i] * a5; if (v > bv) { bv = v; bi = lm.r0 + i; } }
      }
      argValV[wid] = bv; argIdxV[wid] = bi;
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    if (jt === 4) {
      // CHUNKED gemv over SHARED weights (single-phase dispatch)
      const nmat4 = Atomics.load(ctrl, NMAT);
      const resid = Atomics.load(ctrl, RESID) === 1, amax = Atomics.load(ctrl, AMAX) === 1;
      const mids = [], offs4 = [];
      for (let b = 0; b < nmat4; b++) { mids.push(Atomics.load(ctrl, MATID0 + b)); offs4.push(Atomics.load(ctrl, OUTOFF0 + b)); }
      claimChunksW(mids, offs4, CURSOR, resid, amax);
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    if (jt === 6) {
      // SHARED-KV ATTENTION, work-stolen by kvh unit (single-phase dispatch)
      attnUnitsW(Atomics.load(ctrl, LAYER), Atomics.load(ctrl, BASEPOS), CURSOR);
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    if (jt === 7) {
      // MEGA-LAYER: an ENTIRE layer in ONE dispatch — 5 work-stolen phases (qkv / attention /
      // o+resid / gate,up / down+resid) separated by µs in-job barriers (Wn+1 participants incl
      // main). Main does the µs glue (quant/norm/swiglu on its shared instance) at the gaps and
      // bumps GLUEGEN; each phase's asc is read from sabAsc[0] AFTER the glue wait. Replaces 5
      // dispatch wakes per layer with 1 — the measured 77%-of-token gemv-dispatch overhead.
      const l = Atomics.load(ctrl, LAYER), pos = Atomics.load(ctrl, BASEPOS);
      const M = MEGA.layers[l], I = MEGA.I;
      const Nq = aNH * aHD, Nk = aNKV * aHD;
      const gBase = Atomics.load(ctrl, GLUEGEN);
      claimChunksW([M.q, M.k, M.v], [0, Nq, Nq + Nk], CUR_A, 0, 0);
      pbarT(Wn + 1);
      attnUnitsW(l, pos, CUR_B);
      pbarT(Wn + 1);
      glueWaitW(gBase + 1);                       // main: quant(attn) → asc
      claimChunksW([M.o], [0], CUR_C, 1, 0);
      pbarT(Wn + 1);
      glueWaitW(gBase + 2);                       // main: post-norm + quant → asc
      claimChunksW([M.gate, M.up], [0, I], CUR_D, 0, 0);
      pbarT(Wn + 1);
      glueWaitW(gBase + 3);                       // main: swiglu + quant → asc
      claimChunksW([M.down], [0], CUR_E, 1, 0);
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    const K = Atomics.load(ctrl, KK), B = Atomics.load(ctrl, BATCH) || 1, maxN = aMaxN, ng = K / 64;
    const _t0 = performance.now();
    // module-level preallocated views (sActV/sXsumV/loV) — per-dispatch view alloc removed;
    // NOTE: local `const loV` here previously TDZ-shadowed the module view used by jt===3.
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
