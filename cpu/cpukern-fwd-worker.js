// Forward-pool worker: holds a contiguous ROW-PARTITION of every ternary matrix
// (rows [r0,r1) of matrix mid). On each barrier release it runs gemv_tern over its
// rows of the requested matrix, reading the shared int8 activation, and writes its
// output rows (×asc) into the shared output buffer. Weights partitioned → no dup.
// ctrl slots: one cache line per contended word (see cpuengine-mt.js) — false sharing
// in the packed v1 layout cost ~1ms/barrier.
const GEN = 0, NMAT = 16, KK = 17, ASC = 18, MATID0 = 19, OUTOFF0 = 22, JOBTYPE = 25, LAYER = 26, TLEN = 27, SPIN = 28, DONEBASE = 32, DONESTRIDE = 16, CTRL_I32 = 32 + 16 * 16;
let W = null, wid = 0, Wn = 1;
let ctrl, sabAct, sabXsum, sabOut, sabQ, sabAttn, sabKV;
let aNH, aNKV, aHD, aKVstride, aScale, aScores;   // attention params
let sAct, sXsum, sOutLocal;                 // wasm scratch offsets
const meta = [];                            // mid -> {oCodes,oScales,r0,rows,K}
const _b = new ArrayBuffer(4), _bf = new Float32Array(_b), _bi = new Int32Array(_b);
const i32f = (v) => { _bi[0] = v; return _bf[0]; };

const align16 = x => (x + 15) & ~15;
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
  const ACT_OFF = msg.actOff, XSUM_OFF = msg.xsumOff, OUT_OFF = msg.outOff;
  sabAct = new Int8Array(sab, ACT_OFF, msg.maxK);
  sabXsum = new Int32Array(sab, XSUM_OFF, msg.maxK / 64);
  sabOut = new Float32Array(sab, OUT_OFF, msg.maxN);
  sabQ = new Float32Array(sab, msg.qOff, msg.nH * msg.hd);
  sabAttn = new Float32Array(sab, msg.attnOff, msg.nH * msg.hd);
  sabKV = new Float32Array(sab, msg.kvOff, (msg.kvOff !== undefined) ? (sab.byteLength - msg.kvOff) / 4 : 0);
  aNH = msg.nH; aNKV = msg.nKV; aHD = msg.hd; aKVstride = msg.kvStride; aScale = 1 / Math.sqrt(msg.hd);
  aScores = new Float32Array(4096);
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
  // scratch
  sAct = gr0(msg.maxK); sXsum = gr0((msg.maxK / 64) * 4);
  sOutLocal = gr0(Math.ceil(msg.maxN / Wn + 64) * 4);
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
    if (Atomics.load(ctrl, JOBTYPE) === 1) {
      // THREADED GQA ATTENTION: split heads across workers.
      const l = Atomics.load(ctrl, LAYER), T = Atomics.load(ctrl, TLEN);
      const nH = aNH, hd = aHD, nKV = aNKV, qpk = nH / nKV, scale = aScale;
      const kBase = l * 2 * aKVstride, vBase = l * 2 * aKVstride + aKVstride, row = nKV * hd;
      const h0 = Math.floor(wid * nH / Wn), h1 = Math.floor((wid + 1) * nH / Wn), sc = aScores;
      for (let h = h0; h < h1; h++) {
        const kvh = (h / qpk) | 0, qo = h * hd, ko = kvh * hd; let mx = -Infinity;
        for (let t = 0; t < T; t++) { const kb = kBase + t * row + ko; let d = 0; for (let i = 0; i < hd; i++) d += sabQ[qo + i] * sabKV[kb + i]; d *= scale; sc[t] = d; if (d > mx) mx = d; }
        let sum = 0; for (let t = 0; t < T; t++) { const e = Math.exp(sc[t] - mx); sc[t] = e; sum += e; }
        const isum = 1 / sum;
        for (let i = 0; i < hd; i++) sabAttn[qo + i] = 0;
        for (let t = 0; t < T; t++) { const w = sc[t] * isum, vb = vBase + t * row + ko; for (let i = 0; i < hd; i++) sabAttn[qo + i] += w * sabKV[vb + i]; }
      }
      Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);
      continue;
    }
    const K = Atomics.load(ctrl, KK), asc = i32f(Atomics.load(ctrl, ASC));
    // shared int8 activation → my wasm mem, once for the whole batch
    new Int8Array(W.memory.buffer, sAct, K).set(sabAct.subarray(0, K));
    new Int32Array(W.memory.buffer, sXsum, K / 64).set(sabXsum.subarray(0, K / 64));
    for (let b = 0; b < nmat; b++) {
      const m = meta[Atomics.load(ctrl, MATID0 + b)], outOff = Atomics.load(ctrl, OUTOFF0 + b);
      if (m.rows > 0) {
        W.gemv_tern(sOutLocal, m.oCodes, m.oScales, sAct, sXsum, m.rows, m.K);
        const lo = new Float32Array(W.memory.buffer, sOutLocal, m.rows);
        for (let i = 0; i < m.rows; i++) sabOut[outOff + m.r0 + i] = lo[i] * asc;
      }
    }
    Atomics.store(ctrl, DONEBASE + wid * DONESTRIDE, gen);   // own cache line — no false sharing
  }
}
self.onerror = (m, src, ln, col, err) => { try { postMessage({ workerError: String(err && err.stack || m) + ' @' + ln + ':' + col }); } catch (_) {} };
onmessage = (e) => { if (e.data.cmd === 'init') setup(e.data).catch(err => { try { postMessage({ workerError: 'setup: ' + String(err && err.stack || err) }); } catch (_) {} }); };
