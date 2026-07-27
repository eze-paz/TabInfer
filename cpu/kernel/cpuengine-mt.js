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
  const GEN = 0, NMAT = 16, KK = 17, ASC = 18, MATID0 = 19, OUTOFF0 = 22, JOBTYPE = 25, LAYER = 26, TLEN = 27, SPIN = 28, BATCH = 29, BASEPOS = 30, PHASE = 31, DONEBASE = 32, DONESTRIDE = 16, PBCNT = 288, PBGEN = 304, CURSOR = 320, RESID = 336, AMAX = 337, CUR_A = 352, CUR_B = 368, CUR_C = 384, CUR_D = 400, CUR_E = 416, GLUEGEN = 432;
  const CTRL_I32 = 32 + 16 * 16 + 160; // 16 workers + mega/chunk/mega-layer sync words (own cache lines)
  const CH = 128; // work-stealing chunk rows (shared-weights mode)
  const MAXCTX = 2048; // KV positions held in the shared buffer
  const BMAX = 6;     // max tokens verified per batched (speculative) forward
  let Wn = 8, workers = [], ctrl, sabAsc, sabAct, sabXsum, sabOut, sabQ, sabAttn, sabKcur, sabVcur, sabCos, sabSin, sabX, argValV, argIdxV;
  let embedF16 = null; const nrm = {}, mats = {}; // name -> {mid,N,K}
  // MAIN-THREAD wasm instance (private memory) for the SIMD serial ops (quant + rmsnorm)
  // that otherwise run as scalar JS on the critical path, blocking every worker.
  let MW = null, mheap = 0, qDstOff = 0, qXsumOff = 0; const nrmOff = {}; // norm-weight → wasm offset
  // main-thread gemv PARTICIPATION: main holds its own row-slice of every matrix in MW memory
  // and computes it (via MW.gemv_tern) while the workers compute theirs — using the P-core it
  // otherwise burns spin-waiting in waitBarrier. mainMeta[mid] = {codesOff,scalesOff,r0,rows,K}.
  let mainOn = true, mainOutOff = 0, mainOutV = null; const mainMeta = {};
  // CHUNKED shared-weights mode: ONE WebAssembly.Memory({shared}) holds ALL weights; workers +
  // main claim CH-row chunks dynamically (CURSOR) → per-phase straggler ≈ one chunk. Set by
  // load() when globalThis.__chunkMode is true.
  let chunkOn = false, lutOn = false, SWmem = null, MS = null, swLayout = null;
  const BP = 4;   // max batched-prefill columns (gemm_tern_b4)
  let swActOff = 0, swXsumOff = 0, swOutOff = 0, swActI8 = null, swXsumI32 = null, swOutV = null;
  // increment 1+2 shared regions (glue on the shared instance + shared-KV attention)
  let SG = null;   // {sxOff,sxnOff,sswiOff,nrmIn,nrmPost,nrmFin,qknQ,qknK,kvOff,kvStride,kvKS,kvVI,kvVS,attnOff2,pscrBase,pscrStride}
  let sxV = null, sxnV = null, swAttnV = null, swSwiGV = null, swSwiUV = null, swF32 = null, MSP = null, megaMap = null;
  async function initMainWasm() {
    const buf = await (await fetch((globalThis.__cpukernBase || '') + 'cpukern.wasm?v=2')).arrayBuffer();
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
    chunkOn = !!globalThis.__chunkMode || !!globalThis.__lutMode;
    lutOn = !!globalThis.__lutMode;   // LUT implies chunked; widx layout REPLACES packed codes
    mainOn = !chunkOn && globalThis.__mainCompute !== false;   // chunked: main participates via chunk-claim instead
    const parts = Wn + (mainOn ? 1 : 0);   // main participates as a (Wn+1)-th gemv slice
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
    buildMainBufs(ab, dataBase, ternNames, T, parts);  // wasm buffers + norm weights + main's weight slice
    // ── CHUNKED shared-weights setup: ALL weights once in one shared wasm memory ──
    let sharedInit = null;
    if (chunkOn) {
      const PAGE = 65536, STACK = 1 << 20;
      let top = 1024;   // leave the module's own data/stack-default region alone at 0.. (kernels have no statics of note; start at 1KB, align below)
      const align16b = x => (x + 15) & ~15;
      swLayout = {};
      for (const name of ternNames) {
        const t = T[name], ng2 = t.K / 64;
        if (lutOn) {   // chunk-major LUT layouts (widx = 2× codes bytes — the DRAM-side cost)
          const widxOff = align16b(top); top = widxOff + t.N * (t.K / 2);
          const scalesBOff = align16b(top); top = scalesBOff + t.N * ng2 * 4;
          swLayout[mats[name].mid] = { widxOff, scalesBOff, N: t.N, K: t.K, cw: CH * (t.K / 2), cs: CH * ng2 * 4 };
        } else {
          const codesOff = align16b(top); top = codesOff + t.N * (t.K / 4);
          const scalesOff = align16b(top); top = scalesOff + t.N * ng2 * 4;
          swLayout[mats[name].mid] = { codesOff, scalesOff, N: t.N, K: t.K };
        }
      }
      const tblOff0 = align16b(top); top = tblOff0 + 65536;   // LUT table (K/2×16 B, ≤48KB @K=6144)
      swActOff = align16b(top); top = swActOff + BP * CFG.I;
      swXsumOff = align16b(top); top = swXsumOff + BP * (CFG.I / 64) * 4;
      swOutOff = align16b(top); top = swOutOff + BP * CFG.vocab * 4;
      // glue + shared-KV attention regions (increments 1+2)
      const H2 = CFG.H, L2 = CFG.L, hd2 = CFG.hd, nKV2 = CFG.nKV, nH2 = CFG.nH;
      const sxOff = align16b(top); top = sxOff + BP * H2 * 4;   // residual stream per column
      const sxnOff = align16b(top); top = sxnOff + H2 * 4;
      const sswiOff = align16b(top); top = sswiOff + CFG.I * 4;
      const nrmInOff = align16b(top); top = nrmInOff + L2 * H2 * 4;
      const nrmPostOff = align16b(top); top = nrmPostOff + L2 * H2 * 4;
      const nrmFinOff = align16b(top); top = nrmFinOff + H2 * 4;
      const qknQOff = align16b(top); top = qknQOff + L2 * hd2 * 4;
      const qknKOff = align16b(top); top = qknKOff + L2 * hd2 * 4;
      const kvKS = MAXCTX * hd2, kvVI = kvKS + MAXCTX * 4, kvVS = kvVI + MAXCTX * hd2;
      const kvStride = kvVS + MAXCTX * 4;                  // ki8 | ks | vi8 | vs per (l,kvh)
      const kvOff2 = align16b(top); top = kvOff2 + L2 * nKV2 * kvStride;
      const attnOff2 = align16b(top); top = attnOff2 + nH2 * hd2 * 4;
      const pscrStride = 16384, pscrBase = align16b(top); top = pscrBase + (Wn + 1) * pscrStride;
      SG = { sxOff, sxnOff, sswiOff, nrmIn: nrmInOff, nrmPost: nrmPostOff, nrmFin: nrmFinOff, qknQ: qknQOff, qknK: qknKOff, kvOff: kvOff2, kvStride, kvKS, kvVI, kvVS, attnOff2, pscrBase, pscrStride, tblOff: tblOff0, lut: lutOn };
      const stacksBase = align16b(top);
      top = stacksBase + (Wn + 1) * STACK;                 // one stack region per worker + main
      const pages = Math.ceil(top / PAGE) + 4;
      SWmem = new WebAssembly.Memory({ initial: pages, maximum: 24576, shared: true });
      const swU8 = new Uint8Array(SWmem.buffer), srcAll = new Uint8Array(ab);
      for (const name of ternNames) {                       // copy (or LUT-repack) ALL weights ONCE
        const t = T[name], Lw = swLayout[mats[name].mid], ng2 = t.K / 64;
        if (lutOn) {
          // chunk-major widx: [chunk][localBlock b<8][group g<K/2][16 lanes]; index byte =
          // lo/hi nibble of the packed 2-bit code byte ((c1<<2)|c0 == nibble, by construction).
          const K2 = t.K, bpr = K2 / 4, ng = K2 / 2, sgN = K2 / 64;
          const cBase = dataBase + t.codesOff, sBase = dataBase + t.scalesOff;
          const srcF = new Float32Array(ab.slice(sBase, sBase + t.N * sgN * 4));   // alignment-safe copy
          const dstF = new Float32Array(SWmem.buffer, Lw.scalesBOff, t.N * sgN);
          for (let c0 = 0; c0 < t.N / CH; c0++) {
            const wBase = Lw.widxOff + c0 * Lw.cw, sBBase = c0 * (CH * sgN);
            for (let b2 = 0; b2 < CH / 16; b2++) {
              for (let lane = 0; lane < 16; lane++) {
                const row = c0 * CH + b2 * 16 + lane, rB = cBase + row * bpr, wRow = wBase + (b2 * ng) * 16 + lane;
                // source codes are INTERLEAVED per 64-block (code c → byte c&15, bits 2·(c>>4))
                for (let g = 0; g < ng; g++) {
                  const c0i = 2 * g, w0 = c0i & 63, w1 = (c0i + 1) & 63, bB = rB + (c0i >> 6) * 16;
                  const v0 = (srcAll[bB + (w0 & 15)] >> (2 * (w0 >> 4))) & 3;
                  const v1 = (srcAll[bB + (w1 & 15)] >> (2 * (w1 >> 4))) & 3;
                  swU8[wRow + g * 16] = (v1 << 2) | v0;
                }
              }
              for (let sg = 0; sg < sgN; sg++) for (let lane = 0; lane < 16; lane++)
                dstF[sBBase + (sg * (CH / 16) + b2) * 16 + lane] = srcF[(c0 * CH + b2 * 16 + lane) * sgN + sg];
            }
          }
        } else {
          swU8.set(srcAll.subarray(dataBase + t.codesOff, dataBase + t.codesOff + t.N * (t.K / 4)), Lw.codesOff);
          swU8.set(srcAll.subarray(dataBase + t.scalesOff, dataBase + t.scalesOff + t.N * ng2 * 4), Lw.scalesOff);
        }
      }
      const smod = await WebAssembly.compile(await (await fetch((globalThis.__cpukernBase || '') + 'cpukern-shared.wasm?v=2')).arrayBuffer());
      const mi = await WebAssembly.instantiate(smod, { env: { memory: SWmem } });
      mi.exports.__stack_pointer.value = stacksBase + (Wn + 1) * STACK;   // main takes the top stack
      MS = mi.exports;
      swActI8 = new Int8Array(SWmem.buffer, swActOff, CFG.I);
      swXsumI32 = new Int32Array(SWmem.buffer, swXsumOff, CFG.I / 64);
      swOutV = new Float32Array(SWmem.buffer, swOutOff, BP * CFG.vocab);   // BP columns
      sxV = new Float32Array(SWmem.buffer, SG.sxOff, H2);
      sxnV = new Float32Array(SWmem.buffer, SG.sxnOff, H2);
      swAttnV = new Float32Array(SWmem.buffer, SG.attnOff2, nH2 * hd2);
      swSwiGV = new Float32Array(SWmem.buffer, swOutOff, CFG.I);                 // gate at OUT[0..I)
      swSwiUV = new Float32Array(SWmem.buffer, swOutOff + CFG.I * 4, CFG.I);     // up at OUT[I..2I)
      swF32 = new Float32Array(SWmem.buffer, 0, SWmem.buffer.byteLength >> 2);
      const mps = SG.pscrBase + Wn * SG.pscrStride;   // main = participant index Wn
      MSP = { kfOff: mps, qfOff: mps + 512, qi8Off: mps + 1024, scOff: mps + 4096,
        kfV: new Float32Array(SWmem.buffer, mps, hd2), qfV: new Float32Array(SWmem.buffer, mps + 512, hd2), scV: new Float32Array(SWmem.buffer, mps + 4096, MAXCTX) };
      sharedInit = (w) => ({ mem: SWmem, layout: swLayout, stackTop: stacksBase + (w + 1) * STACK, actOff: swActOff, xsumOff: swXsumOff, outOff: swOutOff, outLen: BP * CFG.vocab, ...SG });
    }
    // SAB: ctrl + act(i8) + xsum(i32) + out(f32) + q(f32 nH*hd) + attn(f32 nH*hd)
    //      + kcur/vcur (f32 nKV*hd) — ONLY the current token's k,v. The KV history now lives
    //      as int8 in each worker's PRIVATE wasm memory (per head it owns), so no giant f32
    //      KV in the SAB and attention dots run SIMD (see cpukern-fwd-worker.js).
    const maxK = CFG.I, maxN = CFG.vocab, H = CFG.H, hd = CFG.hd, nH = CFG.nH, nKV = CFG.nKV, L = CFG.L;
    // batched buffers hold up to BMAX columns (speculative verify); single-token path uses col 0.
    const CTRL = CTRL_I32 * 4, ascOff = CTRL, actOff = ascOff + BMAX * 4;
    const xsumOff = actOff + BMAX * maxK, outOff = xsumOff + BMAX * (maxK / 64) * 4;
    const qOff = outOff + BMAX * maxN * 4, attnOff = qOff + BMAX * nH * hd * 4;
    const kcurOff = attnOff + BMAX * nH * hd * 4, vcurOff = kcurOff + BMAX * nKV * hd * 4;
    const cosOff = vcurOff + BMAX * nKV * hd * 4, sinOff = cosOff + BMAX * (hd / 2) * 4;
    const xOff = sinOff + BMAX * (hd / 2) * 4;                    // residual x (mega-token)
    const argValOff = xOff + H * 4, argIdxOff = argValOff + 16 * 4; // per-worker argmax candidates
    const sabBytes = argIdxOff + 16 * 4;
    const sab = new SharedArrayBuffer(sabBytes);
    ctrl = new Int32Array(sab, 0, CTRL_I32);
    Atomics.store(ctrl, SPIN, (globalThis.__SPIN | 0) || 400);
    Atomics.store(ctrl, BATCH, 1);
    Atomics.store(ctrl, PBCNT, 0); Atomics.store(ctrl, PBGEN, 0);
    sabAsc = new Float32Array(sab, ascOff, BMAX);
    sabAct = new Int8Array(sab, actOff, BMAX * maxK);
    sabXsum = new Int32Array(sab, xsumOff, BMAX * (maxK / 64));
    sabOut = new Float32Array(sab, outOff, BMAX * maxN);
    sabQ = new Float32Array(sab, qOff, BMAX * nH * hd);
    sabAttn = new Float32Array(sab, attnOff, BMAX * nH * hd);
    sabKcur = new Float32Array(sab, kcurOff, BMAX * nKV * hd);
    sabVcur = new Float32Array(sab, vcurOff, BMAX * nKV * hd);
    sabCos = new Float32Array(sab, cosOff, BMAX * (hd / 2));
    sabSin = new Float32Array(sab, sinOff, BMAX * (hd / 2));
    sabX = new Float32Array(sab, xOff, H);
    argValV = new Float32Array(sab, argValOff, 16);
    argIdxV = new Int32Array(sab, argIdxOff, 16);
    // qk-norm weights for ALL layers, concatenated → workers apply qk-norm on their heads
    const qNormAll = new Float32Array(L * hd), kNormAll = new Float32Array(L * hd);
    for (let l = 0; l < L; l++) { qNormAll.set(nrm[`model.layers.${l}.self_attn.q_norm.weight`], l * hd); kNormAll.set(nrm[`model.layers.${l}.self_attn.k_norm.weight`], l * hd); }
    // mega-token: input/post/final norm weights + per-layer matrix mids → workers run the
    // whole forward themselves (redundant glue), one dispatch per token.
    const normsAll = new Float32Array((2 * L + 1) * H);
    for (let l = 0; l < L; l++) {
      normsAll.set(nrm[`model.layers.${l}.input_layernorm.weight`], l * H);
      normsAll.set(nrm[`model.layers.${l}.post_attention_layernorm.weight`], (L + l) * H);
    }
    normsAll.set(nrm['model.norm.weight'], 2 * L * H);
    const megaLayers = [];
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      megaLayers.push({ q: mats[p + 'self_attn.q_proj.weight'].mid, k: mats[p + 'self_attn.k_proj.weight'].mid, v: mats[p + 'self_attn.v_proj.weight'].mid,
        o: mats[p + 'self_attn.o_proj.weight'].mid, gate: mats[p + 'mlp.gate_proj.weight'].mid, up: mats[p + 'mlp.up_proj.weight'].mid, down: mats[p + 'mlp.down_proj.weight'].mid });
    }
    const mega = { H, I: CFG.I, layers: megaLayers, lm: mats['lm_head.weight'].mid };
    megaMap = megaLayers;
    if (chunkOn) {   // norm weights → shared memory (used by MS glue + jt===6 attention on both sides)
      for (let l = 0; l < L; l++) {
        swF32.set(nrm[`model.layers.${l}.input_layernorm.weight`], (SG.nrmIn >> 2) + l * H);
        swF32.set(nrm[`model.layers.${l}.post_attention_layernorm.weight`], (SG.nrmPost >> 2) + l * H);
      }
      swF32.set(nrm['model.norm.weight'], SG.nrmFin >> 2);
      swF32.set(qNormAll, SG.qknQ >> 2);
      swF32.set(kNormAll, SG.qknK >> 2);
    }
    // Build each worker's row-partition into its OWN small ArrayBuffer and TRANSFER it
    // (main fetches the 1.16GB binary once; workers never hold the whole thing → no 8×9GB OOM).
    const srcU8 = new Uint8Array(ab);
    const readies = [];
    for (let w = 0; w < Wn; w++) {
      // size this worker's partition (chunked mode: weights live in the shared memory — no partition)
      let bytes = 0; const ents = [];
      if (!chunkOn) for (const name of ternNames) {
        const t = T[name], N = t.N, K = t.K, ng = K / 64, bpr = K / 4;
        const r0 = Math.floor(w * N / parts), r1 = Math.floor((w + 1) * N / parts), rows = r1 - r0;
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
      const wk = new Worker((globalThis.__cpukernBase || '') + 'cpukern-fwd-worker.js'); workers.push(wk);
      wk.onerror = e => { (globalThis.__wErr = globalThis.__wErr || []).push('onerror w' + wk.__wid + ': ' + (e.message || e)); };
      wk.__wid = w;
      readies.push(new Promise(res => { wk.onmessage = ev => { if (ev.data.workerError) (globalThis.__wErr = globalThis.__wErr || []).push('w' + wk.__wid + ': ' + ev.data.workerError); if (ev.data.ready) res(); }; }));
      wk.postMessage({ cmd: 'init', wid: w, Wn, sab, part: part.buffer, layout, maxK, maxN, BMAX, ascOff, actOff, xsumOff, outOff, qOff, attnOff, kcurOff, vcurOff, cosOff, sinOff, nH, nKV, hd, L: CFG.L, eps: CFG.eps, qNormAll, kNormAll, mainOn, xOff, argValOff, argIdxOff, mega, normsAll, shared: sharedInit ? sharedInit(w) : null }, [part.buffer]);
    }
    await Promise.all(readies);
  }

  // cached scratch views into main-wasm memory (stable — no malloc after buildMainBufs)
  let qDstI8 = null, qXsumI32 = null, qXinF32 = null, qXinOff = 0;
  // Allocate every main-wasm buffer FIRST (malloc only grows + records offsets), THEN create
  // all views against the final buffer — growing detaches earlier views, so views come last.
  function buildMainBufs(ab, dataBase, ternNames, T, parts) {
    const { H, hd, nH, nKV, I, vocab } = CFG, maxK = CFG.I;
    mheap = MW.heap_base();
    // ── PHASE 1: all mallocs (grows memory; record offsets only) ──
    const oDst = malloc(maxK), oXsum = malloc((maxK / 64) * 4), oXin = malloc(maxK * 4);
    const oOut = malloc((Math.ceil(vocab / parts) + 64) * 4);   // main's gemv output slice
    const oX = malloc(H * 4), oXn = malloc(H * 4), oXn2 = malloc(H * 4), oXf = malloc(H * 4),
      oAttn = malloc(nH * hd * 4), oSwi = malloc(I * 4);
    const normNames = Object.keys(nrm), nO = {};
    for (const nm of normNames) nO[nm] = malloc(nrm[nm].length * 4);
    // batched hidden-state buffer offsets (B columns) for forwardN / speculative verify
    const nbo = {}; for (const [k, n] of [['x', H], ['xn', H], ['xn2', H], ['xf', H], ['attn', nH * hd], ['swi', I]]) nbo[k] = { off: malloc(BMAX * n * 4), n };
    // main's weight slice: rows [Wn*N/parts, N) of every matrix (mainOn only; else parts==Wn)
    const srcU8 = new Uint8Array(ab); const mEnt = [];
    if (mainOn) for (const name of ternNames) {
      const t = T[name], N = t.N, K = t.K, ng = K / 64, bpr = K / 4;
      const r0 = Math.floor((parts - 1) * N / parts), rows = N - r0;
      const oc = malloc(rows * bpr), os = malloc(rows * ng * 4);
      mEnt.push({ mid: mats[name].mid, r0, rows, K, oc, os, csrc: dataBase + t.codesOff + r0 * bpr, ssrc: dataBase + t.scalesOff + r0 * ng * 4, cl: rows * bpr, sl: rows * ng * 4 });
    }
    // ── PHASE 2: create all views against the now-final buffer, then write ──
    const buf = MW.memory.buffer;   // final buffer — no more mallocs past here
    qDstI8 = new Int8Array(buf, oDst, maxK); qXsumI32 = new Int32Array(buf, oXsum, maxK / 64);
    qXinF32 = new Float32Array(buf, oXin, maxK); qXinOff = oXin; qDstOff = oDst; qXsumOff = oXsum;
    mainOutOff = oOut; mainOutV = new Float32Array(buf, oOut, Math.ceil(vocab / parts) + 64);
    const wv = (off, n) => { const a = new Float32Array(buf, off, n); a.__off = off; return a; };
    B = {
      x: wv(oX, H), xn: wv(oXn, H), xn2: wv(oXn2, H), xf: wv(oXf, H), attn: wv(oAttn, nH * hd), swi: wv(oSwi, I),
      q: new Float32Array(nH * hd), k: new Float32Array(nKV * hd), v: new Float32Array(nKV * hd),
      gate: new Float32Array(I), up: new Float32Array(I), o: new Float32Array(H), logits: new Float32Array(vocab),
      hn: new Float32Array(hd), cosT: new Float32Array(hd / 2), sinT: new Float32Array(hd / 2),
    };
    for (const nm of normNames) { new Float32Array(buf, nO[nm], nrm[nm].length).set(nrm[nm]); nrmOff[nm] = nO[nm]; }
    NB = { q: new Float32Array(BMAX * nH * hd), k: new Float32Array(BMAX * nKV * hd), v: new Float32Array(BMAX * nKV * hd),
      gate: new Float32Array(BMAX * I), up: new Float32Array(BMAX * I), o: new Float32Array(BMAX * H) };
    for (const k in nbo) { const a = new Float32Array(buf, nbo[k].off, BMAX * nbo[k].n); a.__off = nbo[k].off; a.stride = nbo[k].n; NB[k] = a; }
    for (const e of mEnt) {   // copy main's weight slice into MW memory
      new Uint8Array(buf, e.oc, e.cl).set(srcU8.subarray(e.csrc, e.csrc + e.cl));
      new Uint8Array(buf, e.os, e.sl).set(srcU8.subarray(e.ssrc, e.ssrc + e.sl));
      mainMeta[e.mid] = { codesOff: e.oc, scalesOff: e.os, r0: e.r0, rows: e.rows, K: e.K };
    }
  }

  let P = null; // {quant, barrier, gather, n}
  function profReset() { P = { quant: 0, barrier: 0, gather: 0, n: 0 }; }
  function profGet() { return P; }
  const _now = () => performance.now();
  // Synchronous spin barrier: workers run on their own cores while main polls ARRIVED
  // (~µs vs main-thread waitAsync's ~1.8ms macrotask latency — the whole ballgame at 197/tok).
  function waitBarrier(g) { for (let w = 0; w < Wn; w++) { const s = DONEBASE + w * DONESTRIDE; while (Atomics.load(ctrl, s) !== g) { /* spin */ } } }
  // bounded variant for the chunked paths: a stalled/crashed worker throws diagnostics
  // instead of freezing the page (worker errors can't surface while main spins).
  function waitBarrierTO(g, label) {
    const t0 = performance.now();
    let it = 0;
    for (let w = 0; w < Wn; w++) {
      const s = DONEBASE + w * DONESTRIDE;
      while (Atomics.load(ctrl, s) !== g) {
        if ((++it & 4095) === 0 && performance.now() - t0 > 6000) {
          const done = []; for (let x2 = 0; x2 < Wn; x2++) done.push(Atomics.load(ctrl, DONEBASE + x2 * DONESTRIDE));
          throw new Error('stall@' + label + ' gen=' + g + ' done=' + JSON.stringify(done) + ' cursor=' + Atomics.load(ctrl, CURSOR) + ' jt=' + Atomics.load(ctrl, JOBTYPE));
        }
      }
    }
  }
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
      if (mainOn) { qDstI8.set(sabAct.subarray(0, K)); qXsumI32.set(sabXsum.subarray(0, K / 64)); }  // keep main's copy valid
    } else {
      // SIMD quant in the main wasm instance (was scalar JS on the critical path). x must live
      // in wasm memory (all matmul inputs are wasm-backed and carry __off); guard-copy otherwise.
      let xo = x.__off; if (xo === undefined) { qXinF32.set(x.subarray(0, K)); xo = qXinOff; }
      asc = MW.quantize(qDstOff, qXsumOff, xo, K);
      if (chunkOn) {                                 // chunked: activation lives in SHARED memory
        swActI8.set(qDstI8.subarray(0, K));
        swXsumI32.set(qXsumI32.subarray(0, K / 64));
      } else {
        sabAct.set(qDstI8.subarray(0, K));           // int8 activation → SAB (small memcpy)
        sabXsum.set(qXsumI32.subarray(0, K / 64));   // per-group sums → SAB
      }
    }
    if (globalThis.__act6) {  // TEST: requantize activation to int6 (±31) to gauge LUT-precision quality
      const K64 = K / 64;
      for (let k = 0; k < K; k++) { let q = Math.round(sabAct[k] * 31 / 127); q = q < -31 ? -31 : q > 31 ? 31 : q; sabAct[k] = q; }
      for (let g = 0; g < K64; g++) { let t = 0, b0 = g * 64; for (let j = 0; j < 64; j++) t += sabAct[b0 + j]; sabXsum[g] = t; }
      asc *= 127 / 31;
      if (mainOn) { qDstI8.set(sabAct.subarray(0, K)); qXsumI32.set(sabXsum.subarray(0, K64)); }
    }
    sabAsc[0] = asc;
    const t1 = P ? _now() : 0;
    const offs = dispatchMM(names, 1);
    const t2 = P ? _now() : 0;
    const srcV = chunkOn ? swOutV : sabOut;
    for (let i = 0; i < names.length; i++) outs[i].set(srcV.subarray(offs[i], offs[i] + mats[names[i]].N));
    if (P) { P.quant += t1 - t0; P.barrier += t2 - t1; P.gather += _now() - t2; P.n++; }
    return outs;
  }
  // Dispatch `names` matmuls over B activation columns (in sabAct/sabXsum/sabAsc) in ONE barrier;
  // main computes its own row-slice for every column while workers do theirs. Outputs land in
  // sabOut, column-strided by maxN. Returns per-matrix base offsets.
  function dispatchMM(names, B) {
    const maxN = CFG.vocab, K = mats[names[0]].K, ng = K / 64;
    let off = 0; const offs = [];
    for (let i = 0; i < names.length; i++) { offs.push(off); Atomics.store(ctrl, MATID0 + i, mats[names[i]].mid); Atomics.store(ctrl, OUTOFF0 + i, off); off += mats[names[i]].N; }
    if (chunkOn) {
      if (B !== 1) throw new Error('chunked mode: batched forwardN unsupported');
      dispatchChunk(names, offs, 0, 0);
      return offs;
    }
    Atomics.store(ctrl, KK, K); Atomics.store(ctrl, NMAT, names.length); Atomics.store(ctrl, JOBTYPE, 0); Atomics.store(ctrl, BATCH, B);
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    if (mainOn) {
      for (let c = 0; c < B; c++) {
        if (B > 1) { qDstI8.set(sabAct.subarray(c * K, c * K + K)); qXsumI32.set(sabXsum.subarray(c * ng, c * ng + ng)); }
        const asc = sabAsc[c], obase = c * maxN;
        for (let i = 0; i < names.length; i++) {
          const m = mainMeta[mats[names[i]].mid]; if (!m || m.rows <= 0) continue;
          MW.gemv_tern(mainOutOff, m.codesOff, m.scalesOff, qDstOff, qXsumOff, m.rows, m.K);
          const base = obase + offs[i] + m.r0; for (let j = 0; j < m.rows; j++) sabOut[base + j] = mainOutV[j] * asc;
        }
      }
    }
    waitBarrier(g);
    return offs;
  }
  const mmT = (name, x, out) => mmBatch([name], x, [out])[0];
  // Chunked dispatch core: workers + main claim CH-row chunks; resid folds ×asc into shared x
  // (o/down), amax tracks per-participant argmax candidates (lm_head). Act must already be in
  // the shared act region and sabAsc[0] set. `offs` = per-matrix output bases in OUT.
  function dispatchChunk(names, offs, resid, amax, B) {
    B = B || 1;
    const K = mats[names[0]].K, asc = sabAsc[0];
    const mids = names.map(n => mats[n].mid), nchs = mids.map(m => Math.ceil(swLayout[m].N / CH));
    const tot = nchs.reduce((a, b2) => a + b2, 0);
    for (let i = 0; i < names.length; i++) { Atomics.store(ctrl, MATID0 + i, mids[i]); Atomics.store(ctrl, OUTOFF0 + i, offs[i]); }
    Atomics.store(ctrl, KK, K); Atomics.store(ctrl, NMAT, names.length); Atomics.store(ctrl, JOBTYPE, 4); Atomics.store(ctrl, BATCH, B);
    Atomics.store(ctrl, RESID, resid); Atomics.store(ctrl, AMAX, amax); Atomics.store(ctrl, CURSOR, 0);
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    mainClaim(mids, offs, CURSOR, resid, amax, B);
    waitBarrierTO(g, 'chunk');
  }
  // main's work-stealing chunk loop (mirror of the worker's claimChunksW; participant index Wn)
  function mainClaim(mids, offs, curSlot, resid, amax, B) {
    B = B || 1;
    const asc = sabAsc[0];
    const nchs = mids.map(m => Math.ceil(swLayout[m].N / CH));
    let tot = 0; for (const n2 of nchs) tot += n2;
    let bv = -Infinity, bi = 0;
    for (;;) {
      const c = Atomics.add(ctrl, curSlot, 1);
      if (c >= tot) break;
      let b2 = 0, rem = c; while (rem >= nchs[b2]) { rem -= nchs[b2]; b2++; }
      const Lw = swLayout[mids[b2]], row0 = rem * CH, rows = Math.min(CH, Lw.N - row0), ngw = Lw.K / 64;
      if (B === 4) {   // BATCHED: one unpack feeds 4 columns (out cols strided by vocab)
        MS.gemm_tern_b4(swOutOff + (offs[b2] + row0) * 4, CFG.vocab, Lw.codesOff + row0 * (Lw.K / 4), Lw.scalesOff + row0 * ngw * 4, swActOff, swXsumOff, rows, Lw.K);
        const obb = offs[b2] + row0;
        for (let c = 0; c < 4; c++) { const a2 = sabAsc[c], cb = c * CFG.vocab + obb; for (let i = 0; i < rows; i++) swOutV[cb + i] *= a2; }
        continue;
      }
      if (lutOn) MS.gemv_lut_tern_s(swOutOff + (offs[b2] + row0) * 4, Lw.widxOff + rem * Lw.cw, Lw.scalesBOff + rem * Lw.cs, SG.tblOff, rows, Lw.K);
      else MS.gemv_tern(swOutOff + (offs[b2] + row0) * 4, Lw.codesOff + row0 * (Lw.K / 4), Lw.scalesOff + row0 * ngw * 4, swActOff, swXsumOff, rows, Lw.K);
      const ob = offs[b2] + row0;
      if (resid) { for (let i = 0; i < rows; i++) sxV[ob + i] += swOutV[ob + i] * asc; }
      else if (amax) { for (let i = 0; i < rows; i++) { const v = swOutV[ob + i] * asc; swOutV[ob + i] = v; if (v > bv) { bv = v; bi = ob + i; } } }
      else { for (let i = 0; i < rows; i++) swOutV[ob + i] *= asc; }
    }
    if (amax) { argValV[Wn] = bv; argIdxV[Wn] = bi; }
  }
  // main arrives at the in-job mega-layer barrier (Wn+1 participants); bounded spin.
  function pbarM(label) {
    const g = Atomics.load(ctrl, PBGEN);
    if (Atomics.add(ctrl, PBCNT, 1) + 1 === Wn + 1) { Atomics.store(ctrl, PBCNT, 0); Atomics.add(ctrl, PBGEN, 1); Atomics.notify(ctrl, PBGEN); return; }
    let it = 0;
    while (Atomics.load(ctrl, PBGEN) === g) {
      if (++it > 3e8) throw new Error('pbarM stall@' + label + ' pbcnt=' + Atomics.load(ctrl, PBCNT));
    }
  }
  function glueBump() { Atomics.add(ctrl, GLUEGEN, 1); Atomics.notify(ctrl, GLUEGEN); }
  // Main-side mirror of the jt===6 shared-KV attention unit loop (main = participant Wn).
  function attnChunk(l, pos) {
    Atomics.store(ctrl, JOBTYPE, 6); Atomics.store(ctrl, LAYER, l); Atomics.store(ctrl, BASEPOS, pos);
    Atomics.store(ctrl, NMAT, 1); Atomics.store(ctrl, CURSOR, 0);
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    mainAttnUnits(l, pos, CURSOR);
    waitBarrierTO(g, 'attn6');
  }
  // main's work-stealing attention-unit loop (mirror of the worker's attnUnitsW)
  function mainAttnUnits(l, pos, curSlot) {
    const { hd, nKV, nH } = CFG, qpk = nH / nKV, scale = 1 / Math.sqrt(hd), T = pos + 1;
    const Nq = nH * hd, Nk = nKV * hd, eps = CFG.eps;
    for (;;) {
      const kvh = Atomics.add(ctrl, curSlot, 1);
      if (kvh >= nKV) break;
      const kb = SG.kvOff + (l * nKV + kvh) * SG.kvStride;
      const ki8 = kb, ks = kb + SG.kvKS, vi8 = kb + SG.kvVI, vs = kb + SG.kvVS;
      if (pos < MAXCTX) {
        MSP.kfV.set(swOutV.subarray(Nq + kvh * hd, Nq + kvh * hd + hd));
        MS.rmsnorm(MSP.kfOff, MSP.kfOff, SG.qknK + l * hd * 4, hd, eps); ropeS(MSP.kfV);
        swF32[(ks >> 2) + pos] = MS.quant_vec(ki8 + pos * hd, MSP.kfOff, hd);
        MSP.kfV.set(swOutV.subarray(Nq + Nk + kvh * hd, Nq + Nk + kvh * hd + hd));
        swF32[(vs >> 2) + pos] = MS.quant_vec(vi8 + pos * hd, MSP.kfOff, hd);
      }
      for (let h = kvh * qpk; h < (kvh + 1) * qpk; h++) {
        MSP.qfV.set(swOutV.subarray(h * hd, h * hd + hd));
        MS.rmsnorm(MSP.qfOff, MSP.qfOff, SG.qknQ + l * hd * 4, hd, eps); ropeS(MSP.qfV);
        const qs = MS.quant_vec(MSP.qi8Off, MSP.qfOff, hd);
        MS.attn_scores(MSP.scOff, MSP.qi8Off, qs, ki8, ks, T, hd, scale);
        const sc = MSP.scV; let mx = -Infinity; for (let t = 0; t < T; t++) { const v = sc[t]; if (v > mx) mx = v; }
        let sum = 0; for (let t = 0; t < T; t++) { const e = Math.exp(sc[t] - mx); sc[t] = e; sum += e; }
        const isum = 1 / sum; for (let t = 0; t < T; t++) sc[t] *= isum;
        MS.attn_accv(SG.attnOff2 + h * hd * 4, MSP.scOff, vi8, vs, T, hd);
      }
    }
  }
  // rope on a shared f32 view using this token's cos/sin (col 0)
  function ropeS(v) { const half = CFG.hd >> 1; for (let i = 0; i < half; i++) { const c = sabCos[i], s = sabSin[i], x0 = v[i], x1 = v[i + half]; v[i] = x0 * c - x1 * s; v[i + half] = x1 * c + x0 * s; } }
  // Quantize column c of a batched dispatch. Column strides MUST be K / K/64 —
  // that's what gemm_tern_b4 assumes for act/xsum.
  function quantForGemvCol(srcOff, K2, c) {
    const a = MS.quantize(swActOff + c * K2, swXsumOff + c * (K2 / 64) * 4, srcOff, K2);
    sabAsc[c] = a;
  }
  function quantForGemv(srcOff, K2) {
    let a = MS.quantize(swActOff, swXsumOff, srcOff, K2);
    if (lutOn) { MS.build_lut_tern(SG.tblOff, swActOff, K2); a *= 127 / 63; }  // int7-act units
    sabAsc[0] = a;
  }

  // BATCHED PREFILL (B columns, positions p0..p0+B-1). Only the GEMVs are batched
  // (gemm_tern_b4); attention runs per column through the existing single-position
  // attnChunk — column c's q/k/v are copied into the slot attnChunk already reads,
  // so attention/KV/argmax code is untouched. Returns the argmax of the LAST column
  // (prefill only needs the token after the final prompt token).
  function forwardChunkN(tokenIds, p0) {
    const { H, hd, nH, nKV, L, I, eps, theta, vocab } = CFG, half = hd / 2;
    const B = tokenIds.length;
    if (B !== 4) throw new Error('forwardChunkN: B must be 4');
    const Nq = nH * hd, Nk = nKV * hd, qkv = Nq + 2 * Nk;
    for (let c = 0; c < B; c++) { const tid = tokenIds[c], o = (SG.sxOff >> 2) + c * H;
      for (let i = 0; i < H; i++) swF32[o + i] = f16f(embedF16[tid * H + i]); }
    const cosAll = new Float32Array(B * half), sinAll = new Float32Array(B * half);
    for (let c = 0; c < B; c++) { const pos = p0 + c, cb = c * half;
      for (let i = 0; i < half; i++) { const a = pos * Math.pow(theta, -(2 * i) / hd); cosAll[cb + i] = Math.cos(a); sinAll[cb + i] = Math.sin(a); } }
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      for (let c = 0; c < B; c++) { MS.rmsnorm(SG.sxnOff, SG.sxOff + c * H * 4, SG.nrmIn + l * H * 4, H, eps); quantForGemvCol(SG.sxnOff, H, c); }
      dispatchChunk([p + 'self_attn.q_proj.weight', p + 'self_attn.k_proj.weight', p + 'self_attn.v_proj.weight'], [0, Nq, Nq + Nk], 0, 0, B);
      for (let c = 0; c < B; c++) {
        if (c > 0) swOutV.copyWithin(0, c * vocab, c * vocab + qkv);              // col c's qkv → the slot attnChunk reads
        sabCos.set(cosAll.subarray(c * half, c * half + half));                    // EVERY col (incl. 0): sabCos[0..] is clobbered each pass
        sabSin.set(sinAll.subarray(c * half, c * half + half));
        attnChunk(l, p0 + c);                                                     // writes K/V at p0+c, causal over p0+c+1
        sxnV.set(swAttnV); quantForGemvCol(SG.sxnOff, H, c);
      }
      dispatchChunk([p + 'self_attn.o_proj.weight'], [0], 0, 0, B);
      for (let c = 0; c < B; c++) { const xo = (SG.sxOff >> 2) + c * H, so = c * vocab; for (let i = 0; i < H; i++) swF32[xo + i] += swOutV[so + i]; }
      for (let c = 0; c < B; c++) { MS.rmsnorm(SG.sxnOff, SG.sxOff + c * H * 4, SG.nrmPost + l * H * 4, H, eps); quantForGemvCol(SG.sxnOff, H, c); }
      dispatchChunk([p + 'mlp.gate_proj.weight', p + 'mlp.up_proj.weight'], [0, I], 0, 0, B);
      for (let c = 0; c < B; c++) { MS.swiglu(SG.sswiOff, swOutOff + c * vocab * 4, swOutOff + (c * vocab + I) * 4, I); quantForGemvCol(SG.sswiOff, I, c); }
      dispatchChunk([p + 'mlp.down_proj.weight'], [0], 0, 0, B);
      for (let c = 0; c < B; c++) { const xo = (SG.sxOff >> 2) + c * H, so = c * vocab; for (let i = 0; i < H; i++) swF32[xo + i] += swOutV[so + i]; }
    }
    const cL = B - 1;
    MS.rmsnorm(SG.sxnOff, SG.sxOff + cL * H * 4, SG.nrmFin, H, eps); quantForGemvCol(SG.sxnOff, H, 0);
    dispatchChunk(['lm_head.weight'], [0], 0, 1, 1);                              // single-column lm_head + distributed argmax
    let bi = 0, bv = -Infinity;
    for (let w = 0; w <= Wn; w++) { const v = argValV[w]; if (v > bv) { bv = v; bi = argIdxV[w]; } }
    return bi;
  }
  // SELF-TEST: prefill the same 4 tokens at positions 0..3 sequentially, then
  // batched, and compare the resulting next-token. Both write the same KV slots,
  // so the batched path must reproduce the sequential result exactly.
  function _selfTestBatch(tokenIds) {
    const t0 = _now();
    let seqLast = 0;
    for (let i = 0; i < 4; i++) seqLast = forwardChunk(tokenIds[i], i);
    const tSeq = _now() - t0;
    const t1 = _now();
    const bat = forwardChunkN(tokenIds, 0);
    const tBat = _now() - t1;
    return { seqLast, bat, ok: seqLast === bat, msSeq: +tSeq.toFixed(1), msBat: +tBat.toFixed(1), speedup: +(tSeq / tBat).toFixed(3) };
  }
  // Chunked-mode token forward: glue on the MS shared instance (zero copies), chunk-stolen
  // gemv with residual fold + distributed argmax, work-stolen shared-KV attention.
  function forwardChunk(tokenId, pos) {
    const { H, hd, nH, nKV, L, I, eps, theta } = CFG, half = hd / 2;
    const PF = globalThis.__pf, _n = PF ? _now : null;
    let tg = 0, ta = 0, t0 = PF ? _n() : 0;
    for (let i = 0; i < H; i++) sxV[i] = f16f(embedF16[tokenId * H + i]);
    for (let i = 0; i < half; i++) { const a = pos * Math.pow(theta, -(2 * i) / hd); sabCos[i] = Math.cos(a); sabSin[i] = Math.sin(a); }
    const Nq = nH * hd, Nk = nKV * hd;
    if (globalThis.__v3) {   // OPT-IN: measured parity-at-best vs V2 (worse with hot spin)
      // V3 MEGA-LAYER: one dispatch per layer (JOBTYPE=7). Main participates in every
      // work-stolen phase and does the µs glue at the in-job gaps (GLUEGEN bumps).
      const I2 = CFG.I;
      for (let l = 0; l < L; l++) {
        const M = megaMap[l];
        MS.rmsnorm(SG.sxnOff, SG.sxOff, SG.nrmIn + l * H * 4, H, eps);
        quantForGemv(SG.sxnOff, H);
        Atomics.store(ctrl, LAYER, l); Atomics.store(ctrl, BASEPOS, pos); Atomics.store(ctrl, NMAT, 1);
        Atomics.store(ctrl, CUR_A, 0); Atomics.store(ctrl, CUR_B, 0); Atomics.store(ctrl, CUR_C, 0); Atomics.store(ctrl, CUR_D, 0); Atomics.store(ctrl, CUR_E, 0);
        Atomics.store(ctrl, JOBTYPE, 7);
        const g = Atomics.add(ctrl, GEN, 1) + 1;
        Atomics.notify(ctrl, GEN);
        mainClaim([M.q, M.k, M.v], [0, Nq, Nq + Nk], CUR_A, 0, 0);
        pbarM('A' + l);
        mainAttnUnits(l, pos, CUR_B);
        pbarM('B' + l);
        sxnV.set(swAttnV); quantForGemv(SG.sxnOff, H); glueBump();
        mainClaim([M.o], [0], CUR_C, 1, 0);
        pbarM('C' + l);
        MS.rmsnorm(SG.sxnOff, SG.sxOff, SG.nrmPost + l * H * 4, H, eps);
        quantForGemv(SG.sxnOff, H); glueBump();
        mainClaim([M.gate, M.up], [0, I2], CUR_D, 0, 0);
        pbarM('D' + l);
        if (globalThis.__jsSwiglu) { for (let i = 0; i < I2; i++) { const g2 = swSwiGV[i]; swF32[(SG.sswiOff >> 2) + i] = (g2 / (1 + Math.exp(-g2))) * swSwiUV[i]; } }
        else MS.swiglu(SG.sswiOff, swOutOff, swOutOff + I2 * 4, I2);
        quantForGemv(SG.sswiOff, I2); glueBump();
        mainClaim([M.down], [0], CUR_E, 1, 0);
        waitBarrierTO(g, 'mega7-' + l);
      }
      MS.rmsnorm(SG.sxnOff, SG.sxOff, SG.nrmFin, H, eps);
      quantForGemv(SG.sxnOff, H);
      dispatchChunk(['lm_head.weight'], [0], 0, 1);
      let bi3 = 0, bv3 = -Infinity;
      for (let w = 0; w <= Wn; w++) { const v = argValV[w]; if (v > bv3) { bv3 = v; bi3 = argIdxV[w]; } }
      return bi3;
    }
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      MS.rmsnorm(SG.sxnOff, SG.sxOff, SG.nrmIn + l * H * 4, H, eps);
      quantForGemv(SG.sxnOff, H);
      let t1 = PF ? _n() : 0;
      dispatchChunk([p + 'self_attn.q_proj.weight', p + 'self_attn.k_proj.weight', p + 'self_attn.v_proj.weight'], [0, Nq, Nq + Nk], 0, 0);
      if (PF) { const t2 = _n(); tg += t2 - t1; t1 = t2; }
      attnChunk(l, pos);
      if (PF) { const t2 = _n(); ta += t2 - t1; }
      sxnV.set(swAttnV);   // attn (shared) → xn slot as o_proj quant input
      quantForGemv(SG.sxnOff, H);
      if (PF) t1 = _n();
      dispatchChunk([p + 'self_attn.o_proj.weight'], [0], 1, 0);
      if (PF) tg += _n() - t1;
      MS.rmsnorm(SG.sxnOff, SG.sxOff, SG.nrmPost + l * H * 4, H, eps);
      quantForGemv(SG.sxnOff, H);
      if (PF) t1 = _n();
      dispatchChunk([p + 'mlp.gate_proj.weight', p + 'mlp.up_proj.weight'], [0, I], 0, 0);
      if (PF) tg += _n() - t1;
      if (globalThis.__jsSwiglu) { for (let i = 0; i < I; i++) { const g2 = swSwiGV[i]; swF32[(SG.sswiOff >> 2) + i] = (g2 / (1 + Math.exp(-g2))) * swSwiUV[i]; } }
      else MS.swiglu(SG.sswiOff, swOutOff, swOutOff + I * 4, I);
      quantForGemv(SG.sswiOff, I);
      if (PF) t1 = _n();
      dispatchChunk([p + 'mlp.down_proj.weight'], [0], 1, 0);
      if (PF) tg += _n() - t1;
    }
    MS.rmsnorm(SG.sxnOff, SG.sxOff, SG.nrmFin, H, eps);
    quantForGemv(SG.sxnOff, H);
    if (PF) { var tl0 = _n(); }
    dispatchChunk(['lm_head.weight'], [0], 0, 1);
    if (PF) { PF.gemv = (PF.gemv || 0) + tg; PF.lm = (PF.lm || 0) + (_n() - tl0); PF.attn = (PF.attn || 0) + ta; PF.tok = (PF.tok || 0) + (_n() - t0); PF.n = (PF.n || 0) + 1; }
    let bi = 0, bv = -Infinity;
    for (let w = 0; w <= Wn; w++) { const v = argValV[w]; if (v > bv) { bv = v; bi = argIdxV[w]; } }
    return bi;
  }
  // Threaded GQA attention: q (post qk-norm+rope) is in sabQ, K/V history in sabKV[layer];
  // workers split the nH heads, each computes score·softmax·(Σ w·V) → sabAttn. Read into `out`.
  // FUSED qkv-gemv + attention in ONE dispatch (JOBTYPE=2, single-token). Quantizes B.xn, workers
  // compute their q/k/v row-slices → sabOut, hit a WORKER-SIDE spin-barrier (PHASE, no main
  // round-trip), then read q/k/v straight from sabOut and run attention → sabAttn → B.attn.
  // Removes one main↔worker round-trip (28 wakes/token) + the q/k/v gather/scatter.
  function qkvAttnFused(l, pos) {
    const p = `model.layers.${l}.`;
    const names = [p + 'self_attn.q_proj.weight', p + 'self_attn.k_proj.weight', p + 'self_attn.v_proj.weight'];
    const K = mats[names[0]].K;
    const t0 = P ? _now() : 0;
    const asc = MW.quantize(qDstOff, qXsumOff, B.xn.__off, K);
    sabAct.set(qDstI8.subarray(0, K)); sabXsum.set(qXsumI32.subarray(0, K / 64)); sabAsc[0] = asc;
    let off = 0; const offs = [];
    for (let i = 0; i < 3; i++) { offs.push(off); Atomics.store(ctrl, MATID0 + i, mats[names[i]].mid); Atomics.store(ctrl, OUTOFF0 + i, off); off += mats[names[i]].N; }
    Atomics.store(ctrl, KK, K); Atomics.store(ctrl, NMAT, 3); Atomics.store(ctrl, JOBTYPE, 2);
    Atomics.store(ctrl, LAYER, l); Atomics.store(ctrl, BASEPOS, pos); Atomics.store(ctrl, BATCH, 1); Atomics.store(ctrl, PHASE, 0);
    const t1 = P ? _now() : 0;
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    if (mainOn) {   // phase 1: main computes its qkv row-slices, then signals the worker-side barrier
      for (let i = 0; i < 3; i++) { const m = mainMeta[mats[names[i]].mid]; if (!m || m.rows <= 0) continue;
        MW.gemv_tern(mainOutOff, m.codesOff, m.scalesOff, qDstOff, qXsumOff, m.rows, m.K);
        const base = offs[i] + m.r0; for (let j = 0; j < m.rows; j++) sabOut[base + j] = mainOutV[j] * asc; }
      Atomics.add(ctrl, PHASE, 1);
    }
    waitBarrier(g);
    B.attn.set(sabAttn.subarray(0, CFG.nH * CFG.hd));
    if (P) { P.quant += t1 - t0; P.barrier += (_now() - t1); P.attn = (P.attn || 0) + 0; P.n++; }
  }
  // Batched over B columns at base position p0 (single-token: p0=pos, B=1). Workers store B k/v
  // and compute B causal queries. Outputs B columns into sabAttn (col-strided nH*hd).
  function attnThreaded(l, p0, B) {
    const t1 = P ? _now() : 0;
    Atomics.store(ctrl, JOBTYPE, 1); Atomics.store(ctrl, LAYER, l); Atomics.store(ctrl, BASEPOS, p0); Atomics.store(ctrl, BATCH, B);
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    waitBarrier(g);
    if (P) { P.attn = (P.attn || 0) + (_now() - t1); }
  }

  // JS rmsnorm — used only for the small per-head qk-norm (over hd=128, on the non-wasm B.q/B.k).
  // The big H-sized norms go through MW.rmsnorm (SIMD).
  function rmsnorm(x, w, n, out) { let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i]; const inv = 1 / Math.sqrt(s / n + CFG.eps); for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i]; return out; }
  // rope with cos/sin PRECOMPUTED once per token (B.cosT/B.sinT) — they depend only on (pos,i),
  // identical across all 24 q/k heads, so computing them once kills 23/24 of the transcendentals.
  function ropeC(v, off) { const hd = CFG.hd, half = hd / 2, cT = B.cosT, sT = B.sinT; for (let i = 0; i < half; i++) { const c = cT[i], s = sT[i], x0 = v[off + i], x1 = v[off + i + half]; v[off + i] = x0 * c - x1 * s; v[off + i + half] = x1 * c + x0 * s; } }

  let B = null, NB = null;
  function ensureBufs() { if (!B) throw new Error('CPUEngineMT: call load() before forward()'); }

  // quantize column c of a batched wasm-backed buffer `src` (stride K) into sabAct[c]/sabXsum[c]/sabAsc[c]
  function quantCol(src, c, K) {
    const asc = MW.quantize(qDstOff, qXsumOff, src.__off + c * src.stride * 4, K);
    sabAct.set(qDstI8.subarray(0, K), c * K);
    sabXsum.set(qXsumI32.subarray(0, K / 64), c * (K / 64));
    sabAsc[c] = asc;
  }
  // Batched forward over B tokens at positions p0..p0+B-1 → returns [argmax per position].
  // Reuses every kernel; matmuls/attention share ONE barrier per B tokens (barrier amortized).
  function forwardN(tokenIds, p0) {
    const { H, hd, nH, nKV, L, I, eps, theta, vocab } = CFG; ensureBufs();
    const Bn = tokenIds.length, half = hd / 2, N = NB;
    for (let c = 0; c < Bn; c++) { const bx = N.x, o = c * H, tid = tokenIds[c]; for (let i = 0; i < H; i++) bx[o + i] = f16f(embedF16[tid * H + i]); }
    // rope cos/sin for each column position
    for (let c = 0; c < Bn; c++) { const pos = p0 + c, cb = c * half; for (let i = 0; i < half; i++) { const a = pos * Math.pow(theta, -(2 * i) / hd); sabCos[cb + i] = Math.cos(a); sabSin[cb + i] = Math.sin(a); } }
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      for (let c = 0; c < Bn; c++) MW.rmsnorm(N.xn.__off + c * H * 4, N.x.__off + c * H * 4, nrmOff[p + 'input_layernorm.weight'], H, eps);
      for (let c = 0; c < Bn; c++) quantCol(N.xn, c, H);
      let offs = dispatchMM([p + 'self_attn.q_proj.weight', p + 'self_attn.k_proj.weight', p + 'self_attn.v_proj.weight'], Bn);
      const qN = mats[p + 'self_attn.q_proj.weight'].N, kN = mats[p + 'self_attn.k_proj.weight'].N, vN = mats[p + 'self_attn.v_proj.weight'].N;
      for (let c = 0; c < Bn; c++) {   // gather raw q,k,v per column → SAB batched q/kcur/vcur
        const cb = c * vocab;
        sabQ.set(sabOut.subarray(cb + offs[0], cb + offs[0] + qN), c * nH * hd);
        sabKcur.set(sabOut.subarray(cb + offs[1], cb + offs[1] + kN), c * nKV * hd);
        sabVcur.set(sabOut.subarray(cb + offs[2], cb + offs[2] + vN), c * nKV * hd);
      }
      attnThreaded(l, p0, Bn);
      for (let c = 0; c < Bn; c++) N.attn.set(sabAttn.subarray(c * nH * hd, c * nH * hd + nH * hd), c * nH * hd);
      for (let c = 0; c < Bn; c++) quantCol(N.attn, c, nH * hd);
      offs = dispatchMM([p + 'self_attn.o_proj.weight'], Bn);
      for (let c = 0; c < Bn; c++) { const xo = c * H, so = c * vocab; const x = N.x; for (let i = 0; i < H; i++) x[xo + i] += sabOut[so + i]; }
      for (let c = 0; c < Bn; c++) MW.rmsnorm(N.xn2.__off + c * H * 4, N.x.__off + c * H * 4, nrmOff[p + 'post_attention_layernorm.weight'], H, eps);
      for (let c = 0; c < Bn; c++) quantCol(N.xn2, c, H);
      offs = dispatchMM([p + 'mlp.gate_proj.weight', p + 'mlp.up_proj.weight'], Bn);
      const gN = mats[p + 'mlp.gate_proj.weight'].N;
      for (let c = 0; c < Bn; c++) { const so = c * vocab, wo = c * I, swi = N.swi; for (let i = 0; i < I; i++) { const g = sabOut[so + offs[0] + i]; swi[wo + i] = (g / (1 + Math.exp(-g))) * sabOut[so + offs[1] + i]; } }
      for (let c = 0; c < Bn; c++) quantCol(N.swi, c, I);
      offs = dispatchMM([p + 'mlp.down_proj.weight'], Bn);
      for (let c = 0; c < Bn; c++) { const xo = c * H, so = c * vocab, x = N.x; for (let i = 0; i < H; i++) x[xo + i] += sabOut[so + i]; }
    }
    for (let c = 0; c < Bn; c++) MW.rmsnorm(N.xf.__off + c * H * 4, N.x.__off + c * H * 4, nrmOff['model.norm.weight'], H, eps);
    for (let c = 0; c < Bn; c++) quantCol(N.xf, c, H);
    dispatchMM(['lm_head.weight'], Bn);
    const out = [];
    for (let c = 0; c < Bn; c++) { const so = c * vocab; let bi = 0, bv = -Infinity; for (let i = 0; i < vocab; i++) { const v = sabOut[so + i]; if (v > bv) { bv = v; bi = i; } } out.push(bi); }
    return out;
  }

  function forward(tokenId, pos, kv) {
    const { H, hd, nH, nKV, L, I, eps, theta } = CFG; ensureBufs(); const x = B.x;
    for (let i = 0; i < H; i++) x[i] = f16f(embedF16[tokenId * H + i]);
    // precompute rope cos/sin for this position (shared by all heads); publish to SAB so
    // workers can apply qk-norm + rope on their own heads.
    { const half = hd / 2; for (let i = 0; i < half; i++) { const a = pos * Math.pow(theta, -(2 * i) / hd); B.cosT[i] = Math.cos(a); B.sinT[i] = Math.sin(a); } }
    sabCos.set(B.cosT); sabSin.set(B.sinT);
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      MW.rmsnorm(B.xn.__off, x.__off, nrmOff[p + 'input_layernorm.weight'], H, eps);
      if (globalThis.__fuse) {
        // OPT-IN (default OFF): fused qkv+attn via worker-side spin-barrier. MEASURED SLOWER
        // (8.7 vs 11.8 tps) — the spin-barrier hot-spins and kills P-core turbo, costing more
        // than the round-trip it saves. Kept only as the documented negative result.
        qkvAttnFused(l, pos);
      } else {
        // DEFAULT: separate qkv-gemv + attention dispatches — each parks (turbo recovers).
        mmBatch([p + 'self_attn.q_proj.weight', p + 'self_attn.k_proj.weight', p + 'self_attn.v_proj.weight'], B.xn, [B.q, B.k, B.v]);
        sabQ.set(B.q); sabKcur.set(B.k); sabVcur.set(B.v);
        attnThreaded(l, pos, 1);
        B.attn.set(sabAttn.subarray(0, nH * hd));
      }
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
  // MEGA-token forward: ONE dispatch for the whole token. Main writes embed(x) + rope cos/sin,
  // bumps GEN once, spins on DONE flags (as every barrier already does), reduces the workers'
  // per-slice argmax candidates. Requires the pool loaded with __mainCompute=false (workers
  // hold ALL matrix rows — main's T4 slice would otherwise be missing from the forward).
  function forwardTok(tokenId, pos) {
    ensureBufs();
    if (chunkOn) return forwardChunk(tokenId, pos);
    if (mainOn) throw new Error('forwardTok requires load with __mainCompute=false');
    const { H, hd, theta } = CFG, half = hd / 2;
    for (let i = 0; i < H; i++) sabX[i] = f16f(embedF16[tokenId * H + i]);
    for (let i = 0; i < half; i++) { const a = pos * Math.pow(theta, -(2 * i) / hd); sabCos[i] = Math.cos(a); sabSin[i] = Math.sin(a); }
    Atomics.store(ctrl, JOBTYPE, 3); Atomics.store(ctrl, BASEPOS, pos); Atomics.store(ctrl, BATCH, 1);
    Atomics.store(ctrl, NMAT, 1); Atomics.store(ctrl, PBCNT, 0);
    const g = Atomics.add(ctrl, GEN, 1) + 1;
    Atomics.notify(ctrl, GEN);
    // bounded wait + stall diagnostics (a hung worker would otherwise freeze the page silently)
    { const t0 = performance.now();
      for (let w = 0; w < Wn; w++) {
        const s = DONEBASE + w * DONESTRIDE;
        while (Atomics.load(ctrl, s) !== g) {
          if (performance.now() - t0 > 8000) {
            const prog = [], done = [];
            for (let x2 = 0; x2 < Wn; x2++) { prog.push(Atomics.load(ctrl, DONEBASE + x2 * DONESTRIDE + 2)); done.push(Atomics.load(ctrl, DONEBASE + x2 * DONESTRIDE)); }
            throw new Error('mega stall: gen=' + g + ' done=' + JSON.stringify(done) + ' prog(l*8+p)=' + JSON.stringify(prog) + ' pbcnt=' + Atomics.load(ctrl, PBCNT) + ' pbgen=' + Atomics.load(ctrl, PBGEN));
          }
        }
      }
    }
    let bi = 0, bv = -Infinity;
    for (let w = 0; w < Wn; w++) { const v = argValV[w]; if (v > bv) { bv = v; bi = argIdxV[w]; } }
    return bi;
  }
  // prompt-lookup: most-recent earlier occurrence of the last `ng` tokens → the up-to-K tokens after it
  function plookup(seq, ng, K) {
    const n = seq.length; if (n < ng + 1) return [];
    for (let s = n - ng - 1; s >= 0; s--) {
      let m = true; for (let j = 0; j < ng; j++) if (seq[s + j] !== seq[n - ng + j]) { m = false; break; }
      if (m) { const d = []; for (let j = 0; j < K && s + ng + j < n; j++) d.push(seq[s + ng + j]); return d.length ? d : []; }
    }
    return [];
  }
  // Greedy generate with prompt-lookup speculative decode. onTok(tokenId) per accepted token.
  // Verification makes output BIT-IDENTICAL to plain greedy — speculation only changes speed.
  function generateSpec(promptIds, maxNew, onTok, eosSet) {
    const kv = newKV(); const seq = promptIds.slice(); const eos = eosSet || new Set();
    let pos = 0, last = 0;
    for (let i = 0; i < seq.length; i++) { const lg = forward(seq[i], pos++, kv); if (i === seq.length - 1) last = argmax(lg); }
    let produced = 0; const emit = (t) => { onTok(t); seq.push(t); produced++; };
    if (eos.has(last)) return; emit(last);
    const NG = 2, K = BMAX - 1;
    let drafted = 0, accepted = 0;   // stats
    while (produced < maxNew && !eos.has(last)) {
      const draft = plookup(seq, NG, K);
      if (!draft.length) { const lg = forward(last, pos++, kv); last = argmax(lg); if (!eos.has(last)) emit(last); else { emit(last); break; } continue; }
      const cols = [last, ...draft]; drafted += draft.length;
      const out = forwardN(cols, pos);           // out[i] = greedy token after position pos+i
      // accept longest prefix where the draft matched the model's greedy prediction
      let acc = 0; while (acc < draft.length && out[acc] === draft[acc]) acc++;
      accepted += acc;
      for (let i = 0; i < acc; i++) { last = draft[i]; if (eos.has(last)) { emit(last); pos += (i + 1); return; } emit(last); }
      pos += acc;                                 // positions pos..pos+acc now hold accepted tokens
      last = out[acc]; pos += 1;                  // the correction/bonus token (position pos+acc)
      emit(last);
      if (produced >= maxNew) break;
    }
    return { drafted, accepted };
  }
  function stop() { Atomics.store(ctrl, NMAT, -1); Atomics.add(ctrl, GEN, 1); Atomics.notify(ctrl, GEN); setTimeout(() => workers.forEach(w => w.terminate()), 50); }

  const setSpin = (v) => { if (ctrl) Atomics.store(ctrl, SPIN, v | 0); };
  // per-worker gemv busy-µs accumulators (slot DONEBASE+w*STRIDE+1) — load-balance probe
  const busyReset = () => { for (let w = 0; w < Wn; w++) Atomics.store(ctrl, DONEBASE + w * DONESTRIDE + 1, 0); };
  const busyTimes = () => { const a = []; for (let w = 0; w < Wn; w++) a.push(Atomics.load(ctrl, DONEBASE + w * DONESTRIDE + 1)); return a; };
  // mega debug: per-worker pbar wait-µs (+3) and park-fallback count (+4)
  const pbarReset = () => { for (let w = 0; w < Wn; w++) { Atomics.store(ctrl, DONEBASE + w * DONESTRIDE + 3, 0); Atomics.store(ctrl, DONEBASE + w * DONESTRIDE + 4, 0); } };
  const pbarStats = () => { const waitUs = [], parks = []; for (let w = 0; w < Wn; w++) { waitUs.push(Atomics.load(ctrl, DONEBASE + w * DONESTRIDE + 3)); parks.push(Atomics.load(ctrl, DONEBASE + w * DONESTRIDE + 4)); } return { waitUs, parks }; };
  return { load, forward, forwardTok, forwardN, forwardChunkN, _selfTestBatch, generateSpec, plookup, newKV, argmax, stop, CFG, _mats: mats, profReset, profGet, setSpin, busyReset, busyTimes, pbarReset, pbarStats };
})();
if (typeof window !== 'undefined') window.CPUEngineMT = CPUEngineMT;
if (typeof globalThis !== 'undefined') globalThis.CPUEngineMT = CPUEngineMT;
