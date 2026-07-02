// sandpie/modules/webgpu-lfm25.js — LFM2.5-8B-A1B (Lfm2MoeForCausalLM) port for the
// hand-written WebGPU engine. PHASE A: the two architecture-specific kernels + self-tests.
//
// WHY THIS MODEL: MoE with 8.3B total / 1.5B ACTIVE params — decode reads only the active
// experts, so it runs at ~1.5B-dense cost with far more capability; unified iGPU memory
// holds all experts with no swap penalty. Native tool calling. The engine work is friendly:
// 18/24 layers are double-gated SHORT CONV blocks whose entire decode state is H×(L-1)=
// 2048×2 floats (NO KV cache!), only 6 layers are GQA attention (our existing streaming
// int8-KV kernels apply, with 4× fewer layers to pay for), and experts ship as FUSED
// gate_up tensors matching our fused gateup kernel shape.
//
// ARCHITECTURE (from config.json + transformers modeling_lfm2_moe.py, verified 2026-07-02):
//   24 layers; layer_types: attention at [2,6,10,14,18,21], conv elsewhere
//   hidden 2048 · 32 q-heads × 64 · 8 kv-heads · rope_theta 5e6 · rms_norm_eps 1e-5
//   conv: in_proj [3H,H] → (B,C,x) chunks; y = C ⊙ depthwise_causal_conv3(B ⊙ x); out_proj [H,H]
//         no activation, no conv bias; decode state = last L-1 columns of B⊙x per channel
//   attn: q/k/v/out_proj + q_layernorm/k_layernorm (RMSNorm) + rotate_half RoPE
//   FFN:  layers 0-1 dense SwiGLU (I=7168); layers 2-23 MoE — router `gate` [32,H]:
//         sigmoid(logits) + expert_bias → top-4 → weights normalized to sum 1 (norm_topk_prob),
//         routed_scaling 1.0; experts fused gate_up_proj [32, 2·1792, H], down_proj [32, H, 1792]
//   embedding_norm final; TIED embeddings; vocab 128000 (lm_head GEMV → big-N gemvQ NR4 route)
//
// PHASE PLAN:
//   A (this file, now): shortConv kernel (batched T with carry state, double-buffered) +
//     router top-k kernel — the only NEW math. Self-tests vs CPU refs incl. state continuity.
//   B: loader (16.9GB bf16 safetensors → streamed int4 quant cache; ~4.7GB — mind the Cache
//     API quota), tokenizer (new BPE + chat template), projections/attention reuse.
//   C: forward graph + MoE dispatch (predicated expert GEMVs: all-expert grid early-exits on
//     the GPU-resident router output — no CPU readback), generate/runConversation contract.

const SandpieLfm25 = (function () {
  'use strict';

  const E = (typeof window !== 'undefined' ? window : self).SandpieWebGPU;
  const U = (typeof GPUBufferUsage !== 'undefined') ? GPUBufferUsage : { STORAGE: 0x80, COPY_DST: 0x8, COPY_SRC: 0x4, UNIFORM: 0x40, MAP_READ: 0x1 };

  const CONFIG = {
    numLayers: 24, hidden: 2048, nHeads: 32, nKvHeads: 8, headDim: 64,
    convL: 3, denseLayers: 2, denseIntermediate: 7168,
    nExperts: 32, topK: 4, expertIntermediate: 1792,
    vocab: 128000, ropeTheta: 5000000, rmsEps: 1e-5, tieEmbeddings: true,
    attnLayers: [2, 6, 10, 14, 18, 21],
    bos: 124894, eos: 124900,
  };

  let _uPool = [], _uIdx = 0;
  function uniformReset() { _uIdx = 0; }
  function uniform(arr) {
    let buf = _uPool[_uIdx];
    if (!buf) { buf = E.createBuffer(32, U.UNIFORM | U.COPY_DST, 'lu' + _uIdx); _uPool[_uIdx] = buf; }
    E.device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset || 0, arr.byteLength);
    _uIdx++;
    return buf;
  }

  // ============================================================
  // Kernel 1 — double-gated short causal conv (the LFM2 mixer).
  // BCx[T, 3H] (in_proj output; chunks B|C|x) + stateIn[H, L-1] →
  //   y[T,H] = C ⊙ conv3(B ⊙ x)   (depthwise, causal, weights w[H,L], no bias/activation)
  //   stateOut[H, L-1] = last L-1 columns of B⊙x  (the whole decode "cache"!)
  // State is DOUBLE-BUFFERED (read stateIn, write stateOut, caller swaps): with T>1 the
  // threads that read the carry (t < L-1) and the thread that writes it (t = T-1) are in
  // different workgroups — in-place update would race. useState=0 zero-pads (fresh prefill).
  // ============================================================
  const CONV_WGSL = `
enable f16;
struct P { T:u32, H:u32, L:u32, useState:u32 };
@group(0) @binding(0) var<storage, read>       bcx      : array<f32>;   // [T, 3H]
@group(0) @binding(1) var<storage, read>       w        : array<f16>;   // [H, L] depthwise taps
@group(0) @binding(2) var<storage, read>       stateIn  : array<f32>;   // [H, L-1]
@group(0) @binding(3) var<storage, read_write> stateOut : array<f32>;   // [H, L-1]
@group(0) @binding(4) var<storage, read_write> y        : array<f32>;   // [T, H]
@group(0) @binding(5) var<uniform>             p        : P;
fn bx_at(t:i32, h:u32, H:u32, Lm1:i32, useState:u32) -> f32 {
  if (t >= 0) { let r = u32(t)*3u*H; return bcx[r + h] * bcx[r + 2u*H + h]; }   // B[t,h]*x[t,h]
  if (useState == 0u) { return 0.0; }
  return stateIn[h*u32(Lm1) + u32(t + Lm1)];   // t in [-Lm1, -1] → carry column
}
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let H = p.H; let T = p.T; let L = p.L; let Lm1 = i32(L - 1u);
  let i = gid.x; if (i >= T*H) { return; }
  let t = i32(i / H); let h = i % H;
  var acc = 0.0;
  for (var k:u32 = 0u; k < L; k = k + 1u) {     // causal: y(t) = Σ_k w[h,k]·bx(t-(L-1)+k)
    acc = acc + f32(w[h*L + k]) * bx_at(t - Lm1 + i32(k), h, H, Lm1, p.useState);
  }
  let r = u32(t)*3u*H;
  y[u32(t)*H + h] = bcx[r + H + h] * acc;       // C[t,h] ⊙ conv
  if (u32(t) == T - 1u) {                        // last-token thread persists the carry
    for (var c:i32 = 0; c < Lm1; c = c + 1) {
      stateOut[h*u32(Lm1) + u32(c)] = bx_at(i32(T) - Lm1 + c, h, H, Lm1, p.useState);
    }
  }
}`;
  function shortConv(bcxBuf, wBuf, stateInBuf, stateOutBuf, yBuf, T, H, L, useState) {
    const p = uniform(new Uint32Array([T, H, L, useState ? 1 : 0]));
    const pipe = E.getPipeline('lfm25.conv', CONV_WGSL);
    const n = Math.ceil((T * H) / 64), gx = Math.min(n, 65535), gy = Math.ceil(n / gx);
    return E.dispatch(pipe, [bcxBuf, wBuf, stateInBuf, stateOutBuf, yBuf, p], [gx, gy, 1]);
  }

  // ============================================================
  // Kernel 2 — MoE router (Lfm2MoeTopKRouter). logits[T, nE] →
  //   scores = sigmoid(logits); sel = top-k of (scores + expert_bias)  [bias affects SELECTION only]
  //   weights = scores[sel] / (Σ scores[sel] + 1e-6)   (norm_topk_prob) × routed_scaling(1.0)
  // One workgroup per token; nE=32 is tiny — thread 0 does the serial top-k (selection sort,
  // k=4×32 comparisons). Output idx[T,K] (u32) + wt[T,K] (f32) stays GPU-resident: Phase C's
  // expert GEMVs read it directly (predicated early-exit), no CPU readback in the loop.
  // ============================================================
  const ROUTER_WGSL = `
struct P { T:u32, nE:u32, K:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       logits : array<f32>;   // [T, nE]
@group(0) @binding(1) var<storage, read>       bias   : array<f32>;   // [nE] expert_bias
@group(0) @binding(2) var<storage, read_write> idx    : array<u32>;   // [T, K]
@group(0) @binding(3) var<storage, read_write> wt     : array<f32>;   // [T, K]
@group(0) @binding(4) var<uniform>             p      : P;
var<workgroup> sc  : array<f32, 64>;   // sigmoid scores (nE ≤ 64)
var<workgroup> sel : array<f32, 64>;   // selection scores (score + bias)
@compute @workgroup_size(64,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let t = wg.x; if (t >= p.T) { return; }
  let e = lid.x;
  if (e < p.nE) {
    let s = 1.0 / (1.0 + exp(-logits[t*p.nE + e]));
    sc[e] = s; sel[e] = s + bias[e];
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var sum = 0.0;
    for (var k:u32 = 0u; k < p.K; k = k + 1u) {
      var best = 0u; var bv = -1.0e30;
      for (var j:u32 = 0u; j < p.nE; j = j + 1u) { if (sel[j] > bv) { bv = sel[j]; best = j; } }
      sel[best] = -2.0e30;
      idx[t*p.K + k] = best;
      wt[t*p.K + k] = sc[best];
      sum = sum + sc[best];
    }
    for (var k:u32 = 0u; k < p.K; k = k + 1u) { wt[t*p.K + k] = wt[t*p.K + k] / (sum + 1e-6); }
  }
}`;
  function router(logitsBuf, biasBuf, idxBuf, wtBuf, T, nE, K) {
    const p = uniform(new Uint32Array([T, nE, K, 0]));
    const pipe = E.getPipeline('lfm25.router', ROUTER_WGSL);
    return E.dispatch(pipe, [logitsBuf, biasBuf, idxBuf, wtBuf, p], [T, 1, 1]);
  }

  // ============================================================
  // Self-tests vs CPU references (no model download needed).
  // ============================================================
  const ST = () => U.STORAGE | U.COPY_DST | U.COPY_SRC;
  function f32buf(a) { return E.uploadF32(a instanceof Float32Array ? a : new Float32Array(a), ST()); }
  function f16buf(a) { const f = a instanceof Float32Array ? a : new Float32Array(a); const bits = E.f32ToF16(f); const b = E.createBuffer(f.length * 2, ST(), 'lw16'); E.device().queue.writeBuffer(b, 0, bits); return b; }
  const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

  async function selfTestKernels() {
    await E.init();
    const out = [];
    const check = (name, err, tol) => out.push({ name, ok: err < (tol || 1e-3), err });

    // CPU reference for the conv mixer
    const convRef = (bcx, w, state, T, H, L, useState) => {
      const y = new Float32Array(T * H), Lm1 = L - 1;
      const bx = (t, h) => t >= 0 ? bcx[t * 3 * H + h] * bcx[t * 3 * H + 2 * H + h] : (useState ? state[h * Lm1 + (t + Lm1)] : 0);
      for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) {
        let acc = 0; for (let k = 0; k < L; k++) acc += w[h * L + k] * bx(t - Lm1 + k, h);
        y[t * H + h] = bcx[t * 3 * H + H + h] * acc;
      }
      const st = new Float32Array(H * Lm1);
      for (let h = 0; h < H; h++) for (let c = 0; c < Lm1; c++) st[h * Lm1 + c] = bx(T - Lm1 + c, h);
      return { y, st };
    };

    // --- shortConv: fresh prefill (no state) ---
    {
      const T = 7, H = 96, L = 3, Lm1 = 2;
      const bcx = new Float32Array(T * 3 * H); for (let i = 0; i < bcx.length; i++) bcx[i] = Math.sin(i * 0.13);
      const w = new Float32Array(H * L); for (let i = 0; i < w.length; i++) w[i] = 0.3 + 0.5 * Math.cos(i * 0.07);
      const wR = (() => { const b = E.f32ToF16(w); const o = new Float32Array(w.length); for (let i = 0; i < w.length; i++) { o[i] = w[i]; } return o; })();
      const ref = convRef(bcx, w, null, T, H, L, 0);
      const bb = f32buf(bcx), wb = f16buf(w), s0 = f32buf(new Float32Array(H * Lm1)), s1 = E.createBuffer(H * Lm1 * 4, ST(), 's1'), yb = E.createBuffer(T * H * 4, ST(), 'y');
      await shortConv(bb, wb, s0, s1, yb, T, H, L, false);
      const gotY = await E.readF32(yb, T * H), gotS = await E.readF32(s1, H * Lm1);
      check('lfm25 conv (prefill)', Math.max(maxAbs(gotY, ref.y), maxAbs(gotS, ref.st)), 5e-3);   // f16 taps
      [bb, wb, s0, s1, yb].forEach(b => b.destroy());
    }
    // --- shortConv: state continuity — one T=6 pass ≡ 6 chained T=1 passes ---
    {
      const T = 6, H = 64, L = 3, Lm1 = 2;
      const bcx = new Float32Array(T * 3 * H); for (let i = 0; i < bcx.length; i++) bcx[i] = Math.cos(i * 0.11) * (1 + (i % 5) * 0.1);
      const w = new Float32Array(H * L); for (let i = 0; i < w.length; i++) w[i] = 0.2 + 0.4 * Math.sin(i * 0.05);
      const full = convRef(bcx, w, null, T, H, L, 0);
      const wb = f16buf(w);
      let sIn = f32buf(new Float32Array(H * Lm1)), sOut = E.createBuffer(H * Lm1 * 4, ST(), 'so');
      const gotY = new Float32Array(T * H);
      for (let t = 0; t < T; t++) {
        const one = bcx.slice(t * 3 * H, (t + 1) * 3 * H);
        const bb = f32buf(one), yb = E.createBuffer(H * 4, ST(), 'y1');
        await shortConv(bb, wb, sIn, sOut, yb, 1, H, L, t > 0);
        gotY.set(await E.readF32(yb, H), t * H);
        [bb, yb].forEach(b => b.destroy());
        const tmp = sIn; sIn = sOut; sOut = tmp;   // double-buffer swap (as the forward will)
      }
      check('lfm25 conv (decode state continuity)', maxAbs(gotY, full.y), 5e-3);
      [wb, sIn, sOut].forEach(b => b.destroy());
    }
    // --- router: sigmoid + bias-selected top-4 + normalized weights ---
    {
      const T = 3, nE = 32, K = 4;
      const logits = new Float32Array(T * nE); for (let i = 0; i < logits.length; i++) logits[i] = Math.sin(i * 0.37) * 3;
      const bias = new Float32Array(nE); for (let i = 0; i < nE; i++) bias[i] = Math.cos(i * 0.9) * 0.05;
      // CPU ref
      const refIdx = new Uint32Array(T * K), refW = new Float32Array(T * K);
      for (let t = 0; t < T; t++) {
        const sc = [], sel = [];
        for (let e = 0; e < nE; e++) { const s = 1 / (1 + Math.exp(-logits[t * nE + e])); sc.push(s); sel.push(s + bias[e]); }
        let sum = 0;
        for (let k = 0; k < K; k++) {
          let best = 0, bv = -Infinity;
          for (let j = 0; j < nE; j++) if (sel[j] > bv) { bv = sel[j]; best = j; }
          sel[best] = -Infinity; refIdx[t * K + k] = best; refW[t * K + k] = sc[best]; sum += sc[best];
        }
        for (let k = 0; k < K; k++) refW[t * K + k] /= (sum + 1e-6);
      }
      const lb = f32buf(logits), bb = f32buf(bias), ib = E.createBuffer(T * K * 4, ST(), 'ri'), wb2 = E.createBuffer(T * K * 4, ST(), 'rw');
      await router(lb, bb, ib, wb2, T, nE, K);
      const gotIraw = await E.readF32(ib, T * K); const gotI = new Uint32Array(gotIraw.buffer);
      const gotW = await E.readF32(wb2, T * K);
      let idxErr = 0; for (let i = 0; i < T * K; i++) if (gotI[i] !== refIdx[i]) idxErr++;
      check('lfm25 router (top-4 indices)', idxErr);
      check('lfm25 router (normalized weights)', maxAbs(gotW, refW), 1e-5);
      [lb, bb, ib, wb2].forEach(b => b.destroy());
    }
    // --- MoE block: router → 2 experts → weighted combine (full dispatch composition) ---
    {
      const H = 64, nE = 8, K = 2, eI = 32;
      const rnd = (n, s) => { const a = new Float32Array(n); let x = s; for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; a[i] = (x / 0x3fffffff - 1) * 0.5; } return a; };
      // f32 → int4 rec (+ dequantized copy so the CPU ref matches GPU dequant exactly)
      const mk4 = (f32, N, Kd) => {
        const u16 = new Uint16Array(N * Kd), t = new Float32Array(1), ti = new Uint32Array(t.buffer);
        for (let i = 0; i < N * Kd; i++) { t[0] = f32[i]; u16[i] = ti[0] >>> 16; }
        const { pack, scales } = _quantInt4(u16, N, Kd);
        const pb = E.createBuffer(pack.byteLength, ST(), 'tp'), sb = E.createBuffer(scales.byteLength, ST(), 'ts');
        E.device().queue.writeBuffer(pb, 0, pack); E.device().queue.writeBuffer(sb, 0, scales);
        const deq = new Float32Array(N * Kd);
        for (let n = 0; n < N; n++) for (let g = 0; g < Kd / QGROUP; g++) { const sc = _f16ToF32s(scales[n * (Kd / QGROUP) + g]); for (let j = 0; j < QGROUP; j++) { const kk = g * QGROUP + j; const q = (pack[n * (Kd / 8) + (kk >> 3)] >>> (4 * (kk & 7))) & 0xF; deq[n * Kd + kk] = (q - 8) * sc; } }
        return { rec: { pack: pb, scales: sb, N, K: Kd, int4: true }, deq };
      };
      const normed = rnd(H, 7);
      const gate = mk4(rnd(nE * H, 11), nE, H);
      const bias = rnd(nE, 13);
      const experts = []; for (let e = 0; e < nE; e++) experts.push({ w1: mk4(rnd(eI * H, e * 3 + 1), eI, H), w3: mk4(rnd(eI * H, e * 3 + 2), eI, H), w2: mk4(rnd(H * eI, e * 3 + 3), H, eI) });
      const nb = f32buf(normed), bb = f32buf(bias);
      const rl = E.createBuffer(nE * 4, ST(), 'rl'), ri = E.createBuffer(K * 4, ST(), 'ri'), rw = E.createBuffer(K * 4, ST(), 'rw');
      const g_ = E.createBuffer(eI * 4, ST(), 'g'), u_ = E.createBuffer(eI * 4, ST(), 'u'), sw = E.createBuffer(eI * 4, ST(), 's'), eo = E.createBuffer(H * 4, ST(), 'eo'), xo = f32buf(new Float32Array(H));
      await matvecQ(nb, gate.rec, rl, 1, nE, H);
      await router(rl, bb, ri, rw, 1, nE, K);
      await E.device().queue.onSubmittedWorkDone();
      const idx = new Uint32Array((await E.readF32(ri, K)).buffer), wt = await E.readF32(rw, K);
      for (let k = 0; k < K; k++) { const e = idx[k]; await matvecQ(nb, experts[e].w1.rec, g_, 1, eI, H); await matvecQ(nb, experts[e].w3.rec, u_, 1, eI, H); await swiglu(g_, u_, sw, eI); await matvecQ(sw, experts[e].w2.rec, eo, 1, H, eI); await axpy(xo, eo, H, wt[k]); }
      const got = await E.readF32(xo, H);
      // CPU ref from the GPU's routing (router selection itself is covered by the router test)
      const ref = new Float32Array(H);
      for (let k = 0; k < K; k++) {
        const e = idx[k], a = new Float32Array(eI);
        for (let n = 0; n < eI; n++) { let d1 = 0, d3 = 0; for (let i = 0; i < H; i++) { d1 += normed[i] * experts[e].w1.deq[n * H + i]; d3 += normed[i] * experts[e].w3.deq[n * H + i]; } a[n] = (d1 / (1 + Math.exp(-d1))) * d3; }
        for (let h = 0; h < H; h++) { let o = 0; for (let n = 0; n < eI; n++) o += a[n] * experts[e].w2.deq[h * eI + n]; ref[h] += wt[k] * o; }
      }
      check('lfm25 MoE block (route+experts+combine)', maxAbs(got, ref), 3e-3);
      [nb, bb, rl, ri, rw, g_, u_, sw, eo, xo].forEach(b => b.destroy());
      [gate, ...experts.flatMap(e => [e.w1, e.w3, e.w2])].forEach(o => { o.rec.pack.destroy(); o.rec.scales.destroy(); });
    }
    // --- GPU-resident indexed MoE dispatch (packed experts, no readback) vs CPU ref ---
    {
      const H = 64, nE = 8, K = 2, eI = 32, wpr = (Kd) => Kd / 8, gpr = (Kd) => Kd / QGROUP;
      const rnd = (n, s) => { const a = new Float32Array(n); let x = s; for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; a[i] = (x / 0x3fffffff - 1) * 0.5; } return a; };
      // f32 -> {pack, scales, deq} CPU arrays (for packing + dequant ref)
      const q4 = (f32, N, Kd) => {
        const u16 = new Uint16Array(N * Kd), t = new Float32Array(1), ti = new Uint32Array(t.buffer);
        for (let i = 0; i < N * Kd; i++) { t[0] = f32[i]; u16[i] = ti[0] >>> 16; }
        const { pack, scales } = _quantInt4(u16, N, Kd);
        const deq = new Float32Array(N * Kd);
        for (let n = 0; n < N; n++) for (let g = 0; g < gpr(Kd); g++) { const sc = _f16ToF32s(scales[n * gpr(Kd) + g]); for (let j = 0; j < QGROUP; j++) { const kk = g * QGROUP + j; const qv = (pack[n * wpr(Kd) + (kk >> 3)] >>> (4 * (kk & 7))) & 0xF; deq[n * Kd + kk] = (qv - 8) * sc; } }
        return { pack, scales, deq };
      };
      const mkBuf = (arr) => { const b = E.createBuffer(arr.byteLength, ST(), 'pk'); E.device().queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength); return b; };
      const normed = rnd(H, 7), bias = rnd(nE, 13);
      const gate = q4(rnd(nE * H, 11), nE, H);
      // per-expert weights + CPU-concatenated packed buffers
      const ew = { w1: [], w3: [], w2: [] };
      const packed = {};
      for (const [wt, rows, Kd] of [['w1', eI, H], ['w3', eI, H], ['w2', H, eI]]) {
        const pk = new Uint32Array(nE * rows * wpr(Kd)), scc = new Uint16Array(nE * rows * gpr(Kd));
        for (let e = 0; e < nE; e++) { const q = q4(rnd(rows * Kd, e * 7 + (wt === 'w1' ? 1 : wt === 'w3' ? 2 : 3)), rows, Kd); ew[wt].push(q); pk.set(q.pack, e * rows * wpr(Kd)); scc.set(q.scales, e * rows * gpr(Kd)); }
        packed[wt] = { pack: mkBuf(pk), scales: mkBuf(scc), N: nE * rows, K: Kd };
      }
      const nb = f32buf(normed), bb = f32buf(bias), gb = mkBuf(gate.pack), gs = mkBuf(gate.scales);
      const rl = E.createBuffer(nE * 4, ST(), 'rl'), ri = E.createBuffer(K * 4, ST(), 'ri'), rw = E.createBuffer(K * 4, ST(), 'rw');
      const mg = E.createBuffer(K * eI * 4, ST(), 'mg'), mu = E.createBuffer(K * eI * 4, ST(), 'mu'), ma = E.createBuffer(K * eI * 4, ST(), 'ma'), mo = E.createBuffer(K * H * 4, ST(), 'mo'), xo = f32buf(new Float32Array(H));
      await matvecQ(nb, { pack: gb, scales: gs }, rl, 1, nE, H);
      await router(rl, bb, ri, rw, 1, nE, K);
      await idxGemv(nb, packed.w1, ri, mg, K, eI, H, 0);
      await idxGemv(nb, packed.w3, ri, mu, K, eI, H, 0);
      await swiglu(mg, mu, ma, K * eI);
      await idxGemv(ma, packed.w2, ri, mo, K, H, eI, eI);
      await moeCombine(xo, mo, rw, H, K);
      await E.device().queue.onSubmittedWorkDone();
      const got = await E.readF32(xo, H);
      const idx = new Uint32Array((await E.readF32(ri, K)).buffer), wt = await E.readF32(rw, K);
      const ref = new Float32Array(H);
      for (let k = 0; k < K; k++) {
        const e = idx[k], a = new Float32Array(eI);
        for (let n = 0; n < eI; n++) { let d1 = 0, d3 = 0; for (let i = 0; i < H; i++) { d1 += normed[i] * ew.w1[e].deq[n * H + i]; d3 += normed[i] * ew.w3[e].deq[n * H + i]; } a[n] = (d1 / (1 + Math.exp(-d1))) * d3; }
        for (let h = 0; h < H; h++) { let o = 0; for (let n = 0; n < eI; n++) o += a[n] * ew.w2[e].deq[h * eI + n]; ref[h] += wt[k] * o; }
      }
      check('lfm25 MoE indexed dispatch (packed, no readback)', maxAbs(got, ref), 3e-3);
      [nb, bb, gb, gs, rl, ri, rw, mg, mu, ma, mo, xo, packed.w1.pack, packed.w1.scales, packed.w3.pack, packed.w3.scales, packed.w2.pack, packed.w2.scales].forEach(b => b.destroy());
    }
    return out;
  }

  // ============================================================
  // PHASE B — tokenizer + streaming weight loader.
  // ============================================================
  const MODELS = {
    // Bring-up model: same generation/architecture family, dense FFN, small download.
    '350M': {
      root: 'https://huggingface.co/LiquidAI/LFM2.5-350M/resolve/main/',
      cfg: { numLayers: 16, hidden: 1024, nHeads: 16, nKvHeads: 8, headDim: 64, convL: 3, vocab: 65536, ropeTheta: 1000000, rmsEps: 1e-5, attnLayers: [2, 5, 8, 10, 12, 14], moe: false, bos: 1, eos: 7 },
    },
    // The target. moe fields per config.json (verified 2026-07-02).
    '8B-A1B': {
      root: 'https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/resolve/main/',
      cfg: { numLayers: 24, hidden: 2048, nHeads: 32, nKvHeads: 8, headDim: 64, convL: 3, vocab: 128000, ropeTheta: 5000000, rmsEps: 1e-5, attnLayers: [2, 6, 10, 14, 18, 21], moe: true, nExperts: 32, topK: 4, expertI: 1792, denseLayers: 2, denseI: 7168, bos: 124894, eos: 124900 },
    },
  };
  let _variant = null, _cfg = null, _weights = null, _loaded = false;

  // ---- tokenizer: byte-level BPE with the pre_tokenizer regex extracted from
  // tokenizer.json at load (LFM2's split pattern differs from Qwen's; the (?i:…)
  // contraction group — the one construct JS regexes lack — is expanded manually).
  function buildByteMaps() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const byteEnc = new Array(256), byteDec = {};
    for (let i = 0; i < bs.length; i++) { const ch = String.fromCharCode(cs[i]); byteEnc[bs[i]] = ch; byteDec[ch] = bs[i]; }
    return { byteEnc, byteDec };
  }
  function _jsRegexFrom(hfPattern) {
    // The only HF-regex constructs JS lacks in practice: (?i:…) groups. Expand the
    // standard contraction group; anything else unsupported → fall back to GPT-4-style.
    let p = hfPattern.replace(/\(\?i:('s\|'t\|'re\|'ve\|'m\|'ll\|'d)\)/i,
      "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])");
    try { return new RegExp(p, 'gu'); } catch (_) {
      return /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
    }
  }
  const TOK = (function () {
    let vocab = null, idToTok = null, bpeRanks = null, byteEnc = null, byteDec = null, ready = false, pretokRe = null;
    let imStart = -1, imEnd = -1, bosId = -1;
    const _addedIds = new Set();
    const enc = new TextEncoder(), dec = new TextDecoder();
    async function load(root) {
      if (ready) return;
      const j = await (await fetch(root + 'tokenizer.json')).json();
      vocab = j.model.vocab;
      idToTok = {}; for (const k in vocab) idToTok[vocab[k]] = k;
      for (const a of (j.added_tokens || [])) { vocab[a.content] = a.id; idToTok[a.id] = a.content; _addedIds.add(a.id); }
      bpeRanks = new Map();
      const merges = j.model.merges || [];
      for (let i = 0; i < merges.length; i++) { const m = merges[i]; bpeRanks.set(Array.isArray(m) ? (m[0] + ' ' + m[1]) : m, i); }
      // pre_tokenizer: find the Split regex wherever it sits (top-level or Sequence)
      let pat = null;
      const scan = (pt) => { if (!pt) return; if (pt.pattern && pt.pattern.Regex) pat = pt.pattern.Regex; (pt.pretokenizers || []).forEach(scan); };
      scan(j.pre_tokenizer);
      pretokRe = pat ? _jsRegexFrom(pat) : _jsRegexFrom('');
      ({ byteEnc, byteDec } = buildByteMaps());
      imStart = vocab['<|im_start|>'] ?? -1; imEnd = vocab['<|im_end|>'] ?? -1;
      bosId = vocab['<|startoftext|>'] ?? _cfg.bos;
      ready = true;
    }
    function bpe(piece) {
      let word = Array.from(piece);
      if (word.length < 2) return word;
      for (;;) {
        let bestRank = Infinity, bestI = -1;
        for (let i = 0; i < word.length - 1; i++) { const r = bpeRanks.get(word[i] + ' ' + word[i + 1]); if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; } }
        if (bestI < 0) break;
        word = word.slice(0, bestI).concat(word[bestI] + word[bestI + 1], word.slice(bestI + 2));
      }
      return word;
    }
    function encodeText(text) {
      const ids = [];
      for (const piece of (text.match(pretokRe) || [])) {
        const bytes = enc.encode(piece);
        let s = ''; for (const b of bytes) s += byteEnc[b];
        for (const sub of bpe(s)) { const id = vocab[sub]; if (id !== undefined) ids.push(id); }
      }
      return ids;
    }
    function encodeChat(messages, { addGenerationPrompt = true } = {}) {
      const ids = [bosId];   // <|startoftext|>
      for (const m of messages) {
        ids.push(imStart); ids.push(...encodeText(m.role + '\n' + (typeof m.content === 'string' ? m.content : ''))); ids.push(imEnd); ids.push(...encodeText('\n'));
      }
      if (addGenerationPrompt) { ids.push(imStart); ids.push(...encodeText('assistant\n')); }
      return ids;
    }
    function decode(ids) {
      let s = ''; for (const id of ids) { const t = idToTok[id]; if (t !== undefined) s += t; }
      const bytes = []; for (const ch of s) { const b = byteDec[ch]; if (b !== undefined) bytes.push(b); }
      return dec.decode(new Uint8Array(bytes));
    }
    return { load, encodeText, encodeChat, decode, isReady: () => ready, specialIds: () => _addedIds, imEnd: () => imEnd };
  })();

  // ---- streaming single-pass loader: safetensors header via Range request, then one
  // sequential body stream; each tensor is quantized/converted the moment its bytes are
  // complete and the raw bytes are DROPPED — peak host RAM ≈ the largest single tensor
  // (embed: ~525MB on the 8B) regardless of file size (16.9GB single-file works).
  const QGROUP = 32;
  const _f32a = new Float32Array(1), _u32a = new Uint32Array(_f32a.buffer);
  function _f32ToF16(v) { _f32a[0] = v; const x = _u32a[0]; const sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff, mant = x & 0x7fffff; if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00); let e = exp - 127 + 15; if (e >= 31) return sign | 0x7c00; if (e <= 0) { if (e < -10) return sign; const m = mant | 0x800000, sh = 14 - e; let h = m >>> sh; if ((m >>> (sh - 1)) & 1) h += 1; return sign | h; } let h = (e << 10) | (mant >>> 13); if ((mant >>> 12) & 1) h += 1; return sign | h; }
  function _f16ToF32s(h) { const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff; if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024); if (e === 31) return f ? NaN : (s ? -Infinity : Infinity); return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024); }
  function _bf16ToF16bits(u16) { const out = new Uint16Array(u16.length); const t = new Float32Array(1), ti = new Uint32Array(t.buffer); for (let i = 0; i < u16.length; i++) { ti[0] = u16[i] << 16; out[i] = _f32ToF16(t[0]); } return out; }
  // group-wise symmetric int4 along K; rows = product of leading dims (3D expert tensors
  // [E,N,K] quantize as (E·N, K) — per-expert rows are contiguous, so Phase C's expert
  // GEMV just offsets rows by e·N).
  function _quantInt4(u16, rows, K) {
    const wpr = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(rows * wpr), scales = new Uint16Array(rows * gpr);
    const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
    for (let n = 0; n < rows; n++) {
      const rU = n * K, rP = n * wpr, rS = n * gpr;
      for (let g = 0; g < gpr; g++) {
        let maxabs = 0;
        for (let j = 0; j < QGROUP; j++) { ti[0] = u16[rU + g * QGROUP + j] << 16; const v = Math.abs(t[0]); if (v > maxabs) maxabs = v; }
        const sBits = _f32ToF16(maxabs > 0 ? maxabs / 7 : 1e-8); scales[rS + g] = sBits;
        const inv = 1 / _f16ToF32s(sBits);
        for (let j = 0; j < QGROUP; j++) {
          const k = g * QGROUP + j; ti[0] = u16[rU + k] << 16;
          let q = Math.round(t[0] * inv); if (q < -8) q = -8; else if (q > 7) q = 7;
          pack[rP + (k >> 3)] |= ((q + 8) & 0xF) << (4 * (k & 7));
        }
      }
    }
    return { pack, scales };
  }
  // Quant plan by tensor name (lfm2 + lfm2_moe families).
  const _isInt4 = (name) => /(_proj|\.w[123]|feed_forward\.gate)\.weight$/.test(name);   // router gate int4 too (reuses matvecQ)
  const _isF32 = (name) => /expert_bias/.test(name);

  async function _uploadTensor(name, info, raw, sink) {
    const numel = info.shape.reduce((a, b) => a * b, 1);
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, numel);
    const put = (buf, arr) => E.device().queue.writeBuffer(buf, 0, arr.buffer === undefined ? arr : new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    if (_isInt4(name) && info.shape.length >= 2 && (info.shape[info.shape.length - 1] % QGROUP) === 0) {
      const K = info.shape[info.shape.length - 1], rows = numel / K;
      const { pack, scales } = _quantInt4(u16, rows, K);
      const packBuf = E.createBuffer(pack.byteLength, ST(), name + '.pack');
      const scBuf = E.createBuffer(scales.byteLength, ST(), name + '.sc');
      put(packBuf, pack); put(scBuf, scales);
      _weights[name] = { pack: packBuf, scales: scBuf, N: rows, K, int4: true, shape: info.shape };
      if (sink) { await sink.add(name, 'pack', pack, { kind: 'int4', shape: info.shape, N: rows, K }); await sink.add(name, 'scales', scales, { kind: 'int4', shape: info.shape, N: rows, K }); }
    } else if (_isF32(name)) {
      const f32 = new Float32Array(numel); const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
      for (let i = 0; i < numel; i++) { ti[0] = u16[i] << 16; f32[i] = t[0]; }
      const buf = E.createBuffer(f32.byteLength, ST(), name);
      put(buf, f32);
      _weights[name] = { buf, f32: true, shape: info.shape };
      if (sink) await sink.add(name, 'buf', f32, { kind: 'f32', shape: info.shape });
    } else {
      const bits = info.dtype === 'BF16' ? _bf16ToF16bits(u16) : u16;
      const buf = E.createBuffer(numel * 2, ST(), name);
      put(buf, bits);
      _weights[name] = { buf, shape: info.shape };
      if (sink) await sink.add(name, 'buf', bits, { kind: 'f16', shape: info.shape });
    }
    // TIED lm_head: the embedding also serves the vocab GEMV — make its int4 twin now,
    // while the raw bf16 is still in hand (it is never resident again after this call).
    if (name === 'model.embed_tokens.weight' && !_weights['lm_head.weight']) {
      const K = info.shape[1], rows = info.shape[0];
      const { pack, scales } = _quantInt4(u16, rows, K);
      const packBuf = E.createBuffer(pack.byteLength, ST(), 'lm_head.pack');
      const scBuf = E.createBuffer(scales.byteLength, ST(), 'lm_head.sc');
      put(packBuf, pack); put(scBuf, scales);
      _weights['lm_head.weight'] = { pack: packBuf, scales: scBuf, N: rows, K, int4: true, shape: info.shape };
      if (sink) { await sink.add('lm_head.weight', 'pack', pack, { kind: 'int4', shape: info.shape, N: rows, K }); await sink.add('lm_head.weight', 'scales', scales, { kind: 'int4', shape: info.shape, N: rows, K }); }
    }
  }

  // ---- persistent quantized-weights cache -------------------------------------------
  // Same design as the qwen3 engine's (the pattern the user was right to insist on):
  // GPU-READY bytes stream into fixed 64MiB Cache Storage chunks; a manifest of 4-byte-
  // aligned segments is written LAST as the commit point (a torn write = cache miss, never
  // corruption). Reload: allocate buffers from the manifest and writeBuffer chunk slices
  // straight into them — no download, no re-quantize, peak host RAM = one chunk.
  // Namespaced under /lfm25/ so it coexists with the qwen3 cache in the same Cache bucket.
  const QC_NAME = 'sandpie-webgpu-quant';
  const QC_VER = 1;
  const QC_CHUNK = 64 * 1024 * 1024;
  const _qcUrl = (variant, part) => 'https://sandpie.quant/lfm25/v' + QC_VER + '/' + variant + '/' + part;
  function _makeSink(variant) {
    let cache = null, buf = new Uint8Array(QC_CHUNK), used = 0, chunkIdx = 0, globalOff = 0, dead = false;
    const segs = [];
    const flush = async () => {
      if (!used) return;
      await cache.put(_qcUrl(variant, 'c' + chunkIdx), new Response(buf.subarray(0, used)));
      chunkIdx++; buf = new Uint8Array(QC_CHUNK); used = 0;
    };
    return {
      async open() {
        try { cache = await caches.open(QC_NAME); await cache.delete(_qcUrl(variant, 'manifest')); return true; } catch (_) { dead = true; return false; }
      },
      async add(name, part, arr, meta) {
        if (dead) return;
        try {
          const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
          const pad = (4 - (globalOff % 4)) % 4;                       // 4-align every segment start
          for (let p = 0; p < pad; p++) { if (used === QC_CHUNK) await flush(); buf[used++] = 0; globalOff++; }
          const lenPad = (4 - (bytes.byteLength % 4)) % 4;             // …and its length (writeBuffer size must be %4)
          segs.push({ name, part, off: globalOff, len: bytes.byteLength + lenPad, ...meta });
          let src = 0;
          while (src < bytes.byteLength) {
            if (used === QC_CHUNK) await flush();
            const n = Math.min(QC_CHUNK - used, bytes.byteLength - src);
            buf.set(bytes.subarray(src, src + n), used); used += n; src += n; globalOff += n;
          }
          for (let p = 0; p < lenPad; p++) { if (used === QC_CHUNK) await flush(); buf[used++] = 0; globalOff++; }
        } catch (e) { dead = true; try { console.warn('[lfm25] quant-cache write failed (quota?) — continuing uncached', e); } catch (_) {} }
      },
      async finish() {
        if (dead) return false;
        try {
          await flush();
          const manifest = { ver: QC_VER, qgroup: QGROUP, chunkSize: QC_CHUNK, nChunks: chunkIdx, totalBytes: globalOff, segs };
          await cache.put(_qcUrl(variant, 'manifest'), new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } }));
          return true;
        } catch (_) { return false; }
      },
    };
  }
  async function _readQuantCache(variant, onProgress) {
    let cache; try { cache = await caches.open(QC_NAME); } catch (_) { return false; }
    const mResp = await cache.match(_qcUrl(variant, 'manifest'));
    if (!mResp) return false;
    let m; try { m = await mResp.json(); } catch (_) { return false; }
    if (!m || m.ver !== QC_VER || m.qgroup !== QGROUP || m.chunkSize !== QC_CHUNK) return false;
    // allocate every GPU buffer up front from the manifest
    const bySeg = [];
    for (const s of m.segs) {
      const rec = _weights[s.name] || (_weights[s.name] = s.kind === 'int4' ? { N: s.N, K: s.K, int4: true, shape: s.shape } : { shape: s.shape, ...(s.kind === 'f32' ? { f32: true } : {}) });
      const buf = E.createBuffer(s.len, ST(), s.name + '.' + s.part);
      if (s.part === 'pack') rec.pack = buf; else if (s.part === 'scales') rec.scales = buf; else rec.buf = buf;
      bySeg.push({ ...s, buf });
    }
    for (let ci = 0; ci < m.nChunks; ci++) {   // stream chunks → writeBuffer slices
      const resp = await cache.match(_qcUrl(variant, 'c' + ci));
      if (!resp) return false;                  // evicted mid-set → treat as full miss
      const bytes = new Uint8Array(await resp.arrayBuffer());
      const cStart = ci * QC_CHUNK, cEnd = cStart + bytes.byteLength;
      for (const s of bySeg) {
        if (s.off + s.len <= cStart || s.off >= cEnd) continue;
        const b = Math.max(s.off, cStart), e = Math.min(s.off + s.len, cEnd);
        E.device().queue.writeBuffer(s.buf, b - s.off, bytes.buffer, b - cStart, e - b);
      }
      onProgress && onProgress({ phase: 'cache', pct: Math.round((ci + 1) / m.nChunks * 100) });
    }
    return true;
  }

  async function _streamWeights(url, onProgress, sink) {
    // header: first 8 bytes = u64 header length, then the JSON header (Range requests)
    const h8 = await (await fetch(url, { headers: { Range: 'bytes=0-7' } })).arrayBuffer();
    const headerLen = Number(new DataView(h8).getBigUint64(0, true));
    const hResp = await fetch(url, { headers: { Range: 'bytes=8-' + (7 + headerLen) } });
    const header = JSON.parse(new TextDecoder().decode(await hResp.arrayBuffer()));
    const dataStart = 8 + headerLen;
    const tensors = Object.keys(header).filter(n => n !== '__metadata__')
      .map(n => ({ name: n, info: header[n], begin: header[n].data_offsets[0], end: header[n].data_offsets[1] }))
      .sort((a, b) => a.begin - b.begin);
    // one sequential body stream; rolling chunk window, tensors consumed in offset order
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('download failed: HTTP ' + resp.status);
    const total = +(resp.headers.get('content-length') || 0);
    const reader = resp.body.getReader();
    let chunks = [], winStart = 0, recv = 0, ti = 0;   // winStart = absolute offset of chunks[0] RELATIVE TO dataStart
    const have = () => recv - winStart;
    const takeRange = (begin, end) => {   // begin/end relative to dataStart, within the window
      const out = new Uint8Array(end - begin); let w = 0, off = winStart;
      for (const c of chunks) { const cEnd = off + c.length;
        if (cEnd > begin && off < end) { const s = Math.max(begin - off, 0), e = Math.min(end - off, c.length); out.set(c.subarray(s, e), w); w += e - s; }
        off = cEnd; }
      return out;
    };
    const dropTo = (abs) => { let off = winStart;
      while (chunks.length && off + chunks[0].length <= abs) { off += chunks[0].length; chunks.shift(); }
      winStart = off; };
    let skippedHeader = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (value) {
        if (!skippedHeader) {   // the body stream re-delivers header bytes — skip them
          const priorRecv = recv; recv += value.length;
          if (recv <= dataStart) continue;
          const cut = Math.max(dataStart - priorRecv, 0);
          chunks.push(value.subarray(cut)); winStart = 0; recv = recv - dataStart; skippedHeader = true;
        } else { chunks.push(value); recv += value.length; }
        while (ti < tensors.length && tensors[ti].end <= recv) {
          const t = tensors[ti];
          await _uploadTensor(t.name, t.info, takeRange(t.begin, t.end), sink);
          ti++; dropTo(ti < tensors.length ? tensors[ti].begin : recv);
          onProgress && onProgress({ phase: 'parse', pct: Math.round(ti / tensors.length * 100) });
        }
        if (total && onProgress) onProgress({ phase: 'download', pct: Math.round((recv + dataStart) / total * 100) });
      }
      if (done) break;
    }
    if (ti < tensors.length) throw new Error('stream ended early: ' + tensors[ti].name);
  }

  // Pack each MoE layer's 32 per-expert int4 tensors into ONE buffer per weight type, GPU-side
  // (copyBuffer from the already-loaded per-expert buffers — NO re-quantize, NO re-download, so
  // the per-tensor quant cache format is untouched). Enables the indexed GPU-resident expert
  // GEMV (idxGemv) that removes the per-layer router readback. One layer at a time so the
  // transient footprint is ~1 layer's experts (~350MB), not the whole model doubled.
  async function packMoE(onProgress) {
    const C = _cfg, eI = C.expertI, H = C.hidden, nE = C.nExperts;
    const specs = [['w1', eI * H / 2, eI * (H / QGROUP) * 2, eI, H], ['w3', eI * H / 2, eI * (H / QGROUP) * 2, eI, H], ['w2', H * eI / 2, H * (eI / QGROUP) * 2, H, eI]];
    for (let l = C.denseLayers; l < C.numLayers; l++) {
      const p = 'model.layers.' + l + '.feed_forward.';
      for (const [wt, pb, sb, rows, K] of specs) {
        const packBuf = E.createBuffer(pb * nE, ST(), 'moe' + l + wt + 'p');
        const scBuf = E.createBuffer(sb * nE, ST(), 'moe' + l + wt + 's');
        E.beginBatch();
        for (let e = 0; e < nE; e++) { const r = _weights[p + 'experts.' + e + '.' + wt + '.weight']; E.copyBuffer(r.pack, 0, packBuf, e * pb, pb); E.copyBuffer(r.scales, 0, scBuf, e * sb, sb); }
        await E.endBatch();
        for (let e = 0; e < nE; e++) { const key = p + 'experts.' + e + '.' + wt + '.weight'; const r = _weights[key]; try { r.pack.destroy(); r.scales.destroy(); } catch (_) {} delete _weights[key]; }
        _weights['moe.' + l + '.' + wt] = { pack: packBuf, scales: scBuf, N: rows * nE, K, int4: true, rowsPerExpert: rows };
      }
      onProgress && onProgress({ phase: 'pack', pct: Math.round((l - C.denseLayers + 1) / (C.numLayers - C.denseLayers) * 100) });
    }
  }

  async function loadModel({ variant = '350M', onProgress } = {}) {
    if (_loaded && _variant === variant) return;
    const m = MODELS[variant]; if (!m) throw new Error('unknown LFM2.5 variant ' + variant);
    _variant = variant; _cfg = m.cfg; _weights = {};
    await E.init();
    await TOK.load(m.root);
    onProgress && onProgress({ phase: 'tokenizer', pct: 100 });
    // FAST PATH: quantized-weights cache (skips download AND re-quantize)
    let hit = false;
    try { hit = await _readQuantCache(variant, onProgress); } catch (e) { try { console.warn('[lfm25] cache read failed — falling back to download', e); } catch (_) {} _weights = {}; hit = false; }
    if (!hit) {
      _weights = {};
      const sink = _makeSink(variant);
      const sinkOk = await sink.open();
      await _streamWeights(m.root + 'model.safetensors', onProgress, sinkOk ? sink : null);
      if (sinkOk) { const committed = await sink.finish(); try { console.log('[lfm25] quant cache ' + (committed ? 'written' : 'NOT written (quota?)')); } catch (_) {} }
    }
    // GPU-resident MoE: pack experts (from cache OR fresh — both leave per-expert buffers in _weights).
    if (_cfg.moe && !globalThis.__noMoePack) { await packMoE(onProgress); try { console.log('[lfm25] experts packed for GPU-resident dispatch'); } catch (_) {} }
    _loaded = true;
  }
  function unload() {
    try { if (_weights) for (const k in _weights) { const w = _weights[k]; for (const p of ['buf', 'pack', 'scales']) if (w[p] && w[p].destroy) try { w[p].destroy(); } catch (_) {} } } catch (_) {}
    _weights = null; _loaded = false; _variant = null;
  }
  function inventory() {
    if (!_weights) return null;
    const inv = { int4: 0, f16: 0, f32: 0, tensors: 0, int4Bytes: 0, f16Bytes: 0, names: [] };
    for (const k in _weights) {
      const w = _weights[k]; inv.tensors++;
      if (w.int4) { inv.int4++; inv.int4Bytes += w.N * w.K / 2 + w.N * w.K / QGROUP * 2; }
      else if (w.f32) inv.f32++;
      else { inv.f16++; inv.f16Bytes += w.shape.reduce((a, b) => a * b, 1) * 2; }
      if (inv.names.length < 400) inv.names.push(k + (w.int4 ? ' [int4 ' + w.N + 'x' + w.K + ']' : ' ' + JSON.stringify(w.shape)));
    }
    return inv;
  }

  // ============================================================
  // PHASE C — forward graph + generate (bring-up: correctness first).
  // Kernels adapted from the proven qwen3 set: rmsnorm, embed gather, batched int4
  // matvec (T ≤ 32; prefill runs in 32-token chunks), per-head norm + rotate_half RoPE
  // (LFM2's q/k_layernorm is [head_dim] — same contract), f32 flash attention (used for
  // BOTH prefill and decode for now), SwiGLU. Greedy decode with CPU argmax readback per
  // token. Deliberately unoptimized: the GPU-argmax chain / int8 KV / DP4A come after
  // the 350M speaks coherent text.
  // ============================================================
  const MATVEC_WG = 64, MATVEC_MAXT = 32;
  const MATVECQ_WGSL = `
enable f16;
enable subgroups;
struct D { T:u32, N:u32, K:u32, tBase:u32, acc:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${MATVEC_MAXT * MATVEC_WG}>;
@compute @workgroup_size(${MATVEC_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
  let n = wg.x + wg.y * nwg.x;
  if (n >= d.N) { return; }
  let words = d.K / 8u; let wbase = n*words;
  let gpr = d.K / ${QGROUP}u; let sbase = n*gpr;
  let K4 = d.K / 4u; let T = d.T;
  var acc : array<f32, ${MATVEC_MAXT}>;
  for (var t:u32=0u; t<${MATVEC_MAXT}u; t=t+1u) { acc[t] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let p = W[wbase + w];
    let s = f32(sc[sbase + (w*8u)/${QGROUP}u]);
    let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
    let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
    let wa = vec4<f32>(lo.x,hi.x,lo.y,hi.y); let wb_ = vec4<f32>(lo.z,hi.z,lo.w,hi.w);
    for (var t:u32=0u; t<T; t=t+1u) {
      let base = t*K4;
      acc[t] = acc[t] + s*( dot(wa, x[base + 2u*w]) + dot(wb_, x[base + 2u*w + 1u]) );
    }
    w = w + ${MATVEC_WG}u;
  }
  let sgIdx = lid.x / sgs;
  for (var t:u32=0u; t<T; t=t+1u) {
    let ssum = subgroupAdd(acc[t]);
    if (sgi == 0u) { part[t*${MATVEC_WG}u + sgIdx] = ssum; }
  }
  workgroupBarrier();
  if (lid.x < T) {
    let t = lid.x; let nsg=(${MATVEC_WG}u+sgs-1u)/sgs; var tot:f32=0.0;
    for (var i:u32=0u; i<nsg; i=i+1u) { tot = tot + part[t*${MATVEC_WG}u + i]; }
    let idx = t*d.N + n;
    y[idx] = select(0.0, y[idx], d.acc != 0u) + tot;
  }
}`;
  function matvecQ(xBuf, rec, yBuf, T, N, K, acc) {
    const d = uniform(new Uint32Array([T, N, K, 0, acc ? 1 : 0, 0, 0, 0]));
    const pipe = E.getPipeline('lfm25.matvecQ', MATVECQ_WGSL);
    const gx = Math.min(N, 65535), gy = Math.ceil(N / gx);
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, yBuf, d], [gx, gy, 1]);
  }

  const RMSNORM_WGSL = `
enable f16;
struct P { T:u32, H:u32, eps:f32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       w : array<f16>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;
var<workgroup> red : array<f32, 256>;
@compute @workgroup_size(256,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let t = wg.x; let H = p.H; let base = t*H;
  var ss = 0.0;
  var i = lid.x; loop { if (i >= H) { break; } let v = x[base+i]; ss = ss + v*v; i = i + 256u; }
  red[lid.x] = ss; workgroupBarrier();
  var st = 128u; loop { if (st==0u) { break; } if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x+st]; } workgroupBarrier(); st = st/2u; }
  let inv = inverseSqrt(red[0]/f32(H) + p.eps);
  i = lid.x; loop { if (i >= H) { break; } y[base+i] = x[base+i]*inv*f32(w[i]); i = i + 256u; }
}`;
  function rmsnorm(xBuf, wBuf, yBuf, T, H, eps) {
    const u = new Uint32Array(4); new DataView(u.buffer).setUint32(0, T, true); new DataView(u.buffer).setUint32(4, H, true); new DataView(u.buffer).setFloat32(8, eps, true);
    return E.dispatch(E.getPipeline('lfm25.rmsnorm', RMSNORM_WGSL), [xBuf, wBuf, yBuf, uniform(u)], [T, 1, 1]);
  }

  const EMBED_WGSL = `
enable f16;
struct P { T:u32, H:u32, idOff:u32, _b:u32 };   // reads ids[idOff + t] — chain mode passes tokHist + posBase
@group(0) @binding(0) var<storage, read>       ids   : array<u32>;
@group(0) @binding(1) var<storage, read>       embed : array<f16>;
@group(0) @binding(2) var<storage, read_write> y     : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let idx = gid.y*(nwg.x*64u)+gid.x; let total = p.T*p.H; if (idx >= total) { return; }
  let t = idx/p.H; let h = idx%p.H;
  y[idx] = f32(embed[ids[p.idOff + t]*p.H + h]);
}`;
  function embedGather(idsBuf, embBuf, yBuf, T, H, idOff) {
    const p = uniform(new Uint32Array([T, H, idOff || 0, 0]));
    const n = Math.ceil((T * H) / 64), gx = Math.min(n, 65535), gy = Math.ceil(n / gx);
    return E.dispatch(E.getPipeline('lfm25.embed', EMBED_WGSL), [idsBuf, embBuf, yBuf, p], [gx, gy, 1]);
  }

  const ROPEQK_WGSL = `
enable f16;
struct P { T:u32, nH:u32, hd:u32, posBase:u32, theta:f32, eps:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read>       normW: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
var<workgroup> red : array<f32, 128>;
var<workgroup> nrm : array<f32, 128>;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let hd = p.hd; let j = lid.x;
  let unit = wg.x; let t = unit/p.nH; let head = unit%p.nH;
  let base = t*(p.nH*hd) + head*hd;
  var v:f32 = 0.0; if (j < hd) { v = inp[base+j]; }
  red[j] = select(0.0, v*v, j < hd); workgroupBarrier();
  var stride = 64u;
  loop { if (stride==0u) { break; } if (j < stride) { red[j] = red[j] + red[j+stride]; } workgroupBarrier(); stride = stride/2u; }
  let inv = inverseSqrt(red[0]/f32(hd) + p.eps);
  if (j < hd) { nrm[j] = v*inv*f32(normW[j]); }
  workgroupBarrier();
  if (j >= hd) { return; }
  let half = hd/2u;
  let pos = f32(p.posBase + t);
  let freqIdx = select(j-half, j, j<half);
  let ang = pos * pow(p.theta, -2.0*f32(freqIdx)/f32(hd));
  let c = cos(ang); let s = sin(ang);
  let partner = select(nrm[j-half], nrm[j+half], j<half);
  let rot = select(partner, -partner, j<half);
  outp[base+j] = nrm[j]*c + rot*s;
}`;
  function ropeQK(inBuf, normWBuf, outBuf, T, nH, hd, posBase, theta, eps) {
    const u = new Uint32Array(8); const dv = new DataView(u.buffer);
    dv.setUint32(0, T, true); dv.setUint32(4, nH, true); dv.setUint32(8, hd, true);
    dv.setUint32(12, posBase, true); dv.setFloat32(16, theta, true); dv.setFloat32(20, eps, true);
    return E.dispatch(E.getPipeline('lfm25.ropeqk', ROPEQK_WGSL), [inBuf, normWBuf, outBuf, uniform(u)], [T * nH, 1, 1]);
  }

  // f32 tiled flash attention (QT=16 query block × KT=8 key tiles, online softmax) —
  // used for prefill AND decode in the bring-up (decode wastes 15/16 of the block; the
  // dedicated T=1 kernel is a later port).
  const AQT = 16, AKT = 8, AHD4 = 32, AWG = 128;
  const ATTN_WGSL = `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       K : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       V : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             p : P;
const QT=${AQT}u; const KT=${AKT}u; const HD4=${AHD4}u; const WG=${AWG}u;
var<workgroup> qsh : array<vec4<f32>, QT*HD4>;
var<workgroup> ksh : array<vec4<f32>, KT*HD4>;
var<workgroup> vsh : array<vec4<f32>, KT*HD4>;
var<workgroup> acc : array<vec4<f32>, QT*HD4>;
var<workgroup> scr : array<f32, QT*KT>;
var<workgroup> msh : array<f32, QT>;
var<workgroup> lsh : array<f32, QT>;
var<workgroup> csh : array<f32, QT>;
@compute @workgroup_size(${AWG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>) {
  let tid = lidv.x;
  let hd4 = p.hd / 4u;
  let nqb = (p.T + QT - 1u)/QT;
  let blk = wg.x + wg.y*nwg.x;
  let hq = blk / nqb;
  if (hq >= p.nHq) { return; }
  let qbase = (blk % nqb) * QT;
  let grp = p.nHq / p.nKv; let hk = hq / grp;
  let kv4 = (p.nKv*p.hd)/4u; let qhs4 = (p.nHq*p.hd)/4u; let hkoff = hk*hd4; let hqoff = hq*hd4;
  let scale = 1.0/sqrt(f32(p.hd));
  let qmax = (p.S - p.T) + min(qbase + QT, p.T) - 1u;
  var e = tid;
  loop { if (e >= QT*hd4) { break; }
    let qi = e/hd4; let d4 = e%hd4; let gq = qbase+qi;
    qsh[qi*HD4+d4] = select(vec4<f32>(0.0), Q[gq*qhs4 + hqoff + d4], gq < p.T);
    acc[qi*HD4+d4] = vec4<f32>(0.0);
    e = e + WG;
  }
  if (tid < QT) { msh[tid] = -3.0e38; lsh[tid] = 0.0; }
  workgroupBarrier();
  var k0 = 0u;
  loop {
    if (k0 >= p.S || k0 > qmax) { break; }
    e = tid;
    loop { if (e >= KT*hd4) { break; }
      let kj = e/hd4; let d4 = e%hd4; let gk = k0+kj; let ok = gk < p.S;
      ksh[kj*HD4+d4] = select(vec4<f32>(0.0), K[gk*kv4 + hkoff + d4], ok);
      vsh[kj*HD4+d4] = select(vec4<f32>(0.0), V[gk*kv4 + hkoff + d4], ok);
      e = e + WG;
    }
    workgroupBarrier();
    { let qi = tid / KT; let kj = tid % KT;
      var s4 = vec4<f32>(0.0);
      for (var i4=0u;i4<hd4;i4=i4+1u){ s4 = s4 + qsh[qi*HD4+i4]*ksh[kj*HD4+i4]; }
      let dot = s4.x + s4.y + s4.z + s4.w;
      let gq = qbase+qi; let gk = k0+kj; let gqpos = (p.S - p.T) + gq;
      let valid = (gq < p.T) && (gk < p.S) && (gk <= gqpos);
      scr[tid] = select(-3.0e38, dot*scale, valid);
    }
    workgroupBarrier();
    if (tid < QT) {
      let qi = tid;
      var tm = -3.0e38;
      for (var kj=0u;kj<KT;kj=kj+1u){ tm = max(tm, scr[qi*KT+kj]); }
      let mnew = max(msh[qi], tm);
      let corr = exp(msh[qi] - mnew);
      var sum = 0.0;
      for (var kj=0u;kj<KT;kj=kj+1u){
        let pw = select(0.0, exp(scr[qi*KT+kj]-mnew), scr[qi*KT+kj] > -3.0e37);
        scr[qi*KT+kj] = pw; sum = sum + pw;
      }
      lsh[qi] = lsh[qi]*corr + sum; msh[qi] = mnew; csh[qi] = corr;
    }
    workgroupBarrier();
    e = tid;
    loop { if (e >= QT*hd4) { break; }
      let qi = e/hd4; let d4 = e%hd4;
      var a = acc[qi*HD4+d4]*csh[qi];
      for (var kj=0u;kj<KT;kj=kj+1u){ a = a + scr[qi*KT+kj]*vsh[kj*HD4+d4]; }
      acc[qi*HD4+d4] = a;
      e = e + WG;
    }
    workgroupBarrier();
    k0 = k0 + KT;
  }
  e = tid;
  loop { if (e >= QT*hd4) { break; }
    let qi = e/hd4; let d4 = e%hd4; let gq = qbase+qi;
    if (gq < p.T) { O[gq*qhs4 + hqoff + d4] = acc[qi*HD4+d4] / lsh[qi]; }
    e = e + WG;
  }
}`;
  function attention(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd) {
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    const pipe = E.getPipeline('lfm25.attn', ATTN_WGSL);
    const blocks = nHq * Math.ceil(T / AQT);
    const gx = Math.min(blocks, 65535), gy = Math.ceil(blocks / gx);
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
  }

  const SWIGLU_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       g : array<f32>;
@group(0) @binding(1) var<storage, read>       u : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i = gid.y*(nwg.x*64u)+gid.x; if (i >= p.n) { return; }
  let gv = g[i];
  y[i] = (gv / (1.0 + exp(-gv))) * u[i];
}`;
  function swiglu(gBuf, uBuf, yBuf, n) {
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    const c = Math.ceil(n / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('lfm25.swiglu', SWIGLU_WGSL), [gBuf, uBuf, yBuf, p], [gx, gy, 1]);
  }

  // scaled residual add: x[i] += scale * v[i]  (MoE expert-output accumulation)
  const AXPY_WGSL = `
struct P { n:u32, scale:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read_write> x : array<f32>;
@group(0) @binding(1) var<storage, read>       v : array<f32>;
@group(0) @binding(2) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i = gid.y*(nwg.x*64u)+gid.x; if (i >= p.n) { return; }
  x[i] = x[i] + p.scale * v[i];
}`;
  function axpy(xBuf, vBuf, n, scale) {
    const u = new Uint32Array(4); const dv = new DataView(u.buffer); dv.setUint32(0, n, true); dv.setFloat32(4, scale, true);
    const c = Math.ceil(n / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('lfm25.axpy', AXPY_WGSL), [xBuf, vBuf, uniform(u)], [gx, gy, 1]);
  }

  // ---- GPU-resident indexed expert GEMV (kills the per-layer router readback) --------
  // int4 GEMV over PACKED experts [nE*N, Kc]: workgroup (rowBlock, k) reads idx[k] on the
  // GPU → runs expert e=idx[k]'s block (rows e*N..), input at k*inStride (0 for w1/w3 which
  // share `normed`; eI for w2 whose input is per-expert act[k]), output at k*N. No readback:
  // the whole MoE block stays batched. GEMVQ_NR rows/workgroup, subgroupAdd reduction.
  const GEMV_WG = 64, GEMVQ_NR = 8;
  const IDXGEMV_WGSL = `
enable f16;
enable subgroups;
struct D { N:u32, Kc:u32, inStride:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W   : array<u32>;
@group(0) @binding(2) var<storage, read>       sc  : array<f16>;
@group(0) @binding(3) var<storage, read>       idx : array<u32>;
@group(0) @binding(4) var<storage, read_write> y   : array<f32>;
@group(0) @binding(5) var<uniform>             d   : D;
var<workgroup> part : array<f32, ${GEMVQ_NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
  let k = wg.y;                              // selected-expert slot
  let e = idx[k];
  let nBase = wg.x * ${GEMVQ_NR}u;           // first output row (within the expert's N rows)
  let words = d.Kc / 8u; let gpr = d.Kc / ${QGROUP}u;
  let xb = (k * d.inStride) / 4u;            // per-k input base (vec4 units)
  var acc : array<f32, ${GEMVQ_NR}>;
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = x[xb + 2u*w]; let xc = x[xb + 2u*w + 1u];
    let grp = (w*8u)/${QGROUP}u;
    for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) {
      let n = nBase + r; if (n >= d.N) { continue; }
      let row = e*d.N + n;                    // expert e's row block in the packed tensor
      let p = W[row*words + w];
      let s = f32(sc[row*gpr + grp]);
      let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      acc[r] = acc[r] + s*( dot(vec4<f32>(lo.x,hi.x,lo.y,hi.y), xa) + dot(vec4<f32>(lo.z,hi.z,lo.w,hi.w), xc) );
    }
    w = w + ${GEMV_WG}u;
  }
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) {
    let ss = subgroupAdd(acc[r]);
    if (sgi == 0u) { part[r*${GEMV_WG}u + sgIdx] = ss; }
  }
  workgroupBarrier();
  if (lid.x < ${GEMVQ_NR}u) {
    let n = nBase + lid.x;
    if (n < d.N) {
      let nsg=(${GEMV_WG}u+sgs-1u)/sgs; var t:f32=0.0;
      for(var i:u32=0u;i<nsg;i=i+1u){ t = t + part[lid.x*${GEMV_WG}u + i]; }
      y[k*d.N + n] = t;
    }
  }
}`;
  function idxGemv(xBuf, rec, idxBuf, yBuf, topK, N, Kc, inStride) {
    const d = uniform(new Uint32Array([N, Kc, inStride, 0]));
    const pipe = E.getPipeline('lfm25.idxGemv', IDXGEMV_WGSL);
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, idxBuf, yBuf, d], [Math.ceil(N / GEMVQ_NR), topK, 1]);
  }
  // combine: x[h] += Σ_k wt[k] * o[k*H + h]   (weighted expert sum into the residual)
  const MOECOMBINE_WGSL = `
struct D { H:u32, K:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read_write> x  : array<f32>;
@group(0) @binding(1) var<storage, read>       o  : array<f32>;
@group(0) @binding(2) var<storage, read>       wt : array<f32>;
@group(0) @binding(3) var<uniform>             d  : D;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let h = gid.y*(nwg.x*64u)+gid.x; if (h >= d.H) { return; }
  var s = 0.0;
  for (var k:u32=0u; k<d.K; k=k+1u) { s = s + wt[k]*o[k*d.H + h]; }
  x[h] = x[h] + s;
}`;
  function moeCombine(xBuf, oBuf, wtBuf, H, K) {
    const d = uniform(new Uint32Array([H, K, 0, 0]));
    const c = Math.ceil(H / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('lfm25.moeCombine', MOECOMBINE_WGSL), [xBuf, oBuf, wtBuf, d], [gx, gy, 1]);
  }

  // ---- two-stage GPU argmax → token history (enables GPU-resident decode chaining) ---
  // Stage 1: AMAX_WGS workgroups reduce interleaved logit slices → per-WG (max,idx). Stage 2:
  // one workgroup folds them and writes the token id into tokHist[outPos] on the GPU — so the
  // NEXT forward's embed can read it without any CPU readback. This is what lets GEN_BATCH
  // decode tokens chain in one submit (the throughput win: 1 readback per batch, not per token).
  const ARGMAX_WG = 256, AMAX_WGS = 64;
  const AMAX_P1_WGSL = `
struct P { n:u32, outPos:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> pV : array<f32>;
@group(0) @binding(2) var<storage, read_write> pI : array<u32>;
@group(0) @binding(3) var<uniform>             p  : P;
var<workgroup> sv : array<f32, ${ARGMAX_WG}>;
var<workgroup> si : array<u32, ${ARGMAX_WG}>;
@compute @workgroup_size(${ARGMAX_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  var bv=-3.0e38; var bi=0u;
  var i = wg.x*${ARGMAX_WG}u + lid.x;
  loop { if (i>=p.n) {break;} let v=logits[i]; if (v>bv){bv=v;bi=i;} i=i+${AMAX_WGS * ARGMAX_WG}u; }
  sv[lid.x]=bv; si[lid.x]=bi; workgroupBarrier();
  var s=${ARGMAX_WG}u/2u;
  loop { if(s==0u){break;} if(lid.x<s){ if(sv[lid.x+s]>sv[lid.x]){sv[lid.x]=sv[lid.x+s];si[lid.x]=si[lid.x+s];} } workgroupBarrier(); s=s/2u; }
  if (lid.x==0u){ pV[wg.x]=sv[0]; pI[wg.x]=si[0]; }
}`;
  const AMAX_P2_WGSL = `
struct P { n:u32, outPos:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       pV : array<f32>;
@group(0) @binding(1) var<storage, read>       pI : array<u32>;
@group(0) @binding(2) var<storage, read_write> tok: array<u32>;
@group(0) @binding(3) var<uniform>             p  : P;
var<workgroup> sv : array<f32, ${AMAX_WGS}>;
var<workgroup> si : array<u32, ${AMAX_WGS}>;
@compute @workgroup_size(${AMAX_WGS},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>){
  sv[lid.x]=pV[lid.x]; si[lid.x]=pI[lid.x]; workgroupBarrier();
  var s=${AMAX_WGS}u/2u;
  loop { if(s==0u){break;} if(lid.x<s){ if(sv[lid.x+s]>sv[lid.x]){sv[lid.x]=sv[lid.x+s];si[lid.x]=si[lid.x+s];} } workgroupBarrier(); s=s/2u; }
  if (lid.x==0u){ tok[p.outPos]=si[0]; }
}`;
  let _amaxV = null, _amaxI = null;
  function argmaxKernel(logitsBuf, tokHistBuf, N, outPos) {
    if (!_amaxV) { _amaxV = E.createBuffer(AMAX_WGS * 4, ST(), 'amV'); _amaxI = E.createBuffer(AMAX_WGS * 4, ST(), 'amI'); }
    const p = uniform(new Uint32Array([N, outPos, 0, 0]));
    E.dispatch(E.getPipeline('lfm25.amaxP1', AMAX_P1_WGSL), [logitsBuf, _amaxV, _amaxI, p], [AMAX_WGS, 1, 1]);
    return E.dispatch(E.getPipeline('lfm25.amaxP2', AMAX_P2_WGSL), [_amaxV, _amaxI, tokHistBuf, p], [1, 1, 1]);
  }

  // ---- forward state --------------------------------------------------------------
  const MAX_SEQ = 2048;   // bring-up context
  let _scr = null, _kv = null, _conv = null, _idsBuf = null;
  function _ensureState() {
    if (_scr) return;
    const C = _cfg, H = C.hidden, T = MATVEC_MAXT;
    const mk = (n, l) => E.createBuffer(n * 4, ST(), l);
    // w1 shapes are auto-adjusted — read the real intermediate from the weights
    const I = _weights['model.layers.0.feed_forward.w1.weight'].N;
    _cfg.ffI = I;
    _scr = {
      x: mk(T * H, 'x'), normed: mk(T * H, 'nrm'),
      bcx: mk(T * 3 * H, 'bcx'), convy: mk(T * H, 'cy'),
      q: mk(T * C.nHeads * C.headDim, 'q'), k: mk(T * C.nKvHeads * C.headDim, 'k'), v: mk(T * C.nKvHeads * C.headDim, 'v'),
      qr: mk(T * C.nHeads * C.headDim, 'qr'), kr: mk(T * C.nKvHeads * C.headDim, 'kr'),
      attn: mk(T * C.nHeads * C.headDim, 'at'),
      gate: mk(T * I, 'g'), up: mk(T * I, 'u'), swi: mk(T * I, 's'),
      last: mk(H, 'last'), logits: mk(C.vocab, 'lg'), tokHist: E.createBuffer(MAX_SEQ * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'th'),
      // MoE scratch (harmless on dense models): expert output [H], router logits [nE], top-k idx/wt,
      // and the GPU-resident indexed-dispatch buffers [K,eI]/[K,H]
      eout: mk(H, 'eo'), rlogits: mk(Math.max(C.nExperts || 1, 1), 'rl'), ridx: mk(Math.max(C.topK || 1, 1), 'ri'), rwt: mk(Math.max(C.topK || 1, 1), 'rw'),
      moeGate: mk(Math.max((C.topK || 1) * (C.expertI || 1), 1), 'mg'), moeUp: mk(Math.max((C.topK || 1) * (C.expertI || 1), 1), 'mu'),
      moeAct: mk(Math.max((C.topK || 1) * (C.expertI || 1), 1), 'ma'), moeOut: mk(Math.max((C.topK || 1) * H, 1), 'mo'),
    };
    _kv = {}; _conv = {};
    for (let l = 0; l < C.numLayers; l++) {
      if (C.attnLayers.includes(l)) _kv[l] = { k: mk(MAX_SEQ * C.nKvHeads * C.headDim, 'k' + l), v: mk(MAX_SEQ * C.nKvHeads * C.headDim, 'v' + l) };
      else _conv[l] = { a: mk(C.hidden * (C.convL - 1), 'ca' + l), b: mk(C.hidden * (C.convL - 1), 'cb' + l), cur: 0 };
    }
  }
  function _freeState() {
    const kill = (o) => { for (const k in o) { const v = o[k]; if (v && v.destroy) try { v.destroy(); } catch (_) {} else if (v && typeof v === 'object') kill(v); } };
    if (_scr) kill(_scr); if (_kv) kill(_kv); if (_conv) kill(_conv);
    if (_idsBuf) try { _idsBuf.destroy(); } catch (_) {}
    try { if (_amaxV) _amaxV.destroy(); if (_amaxI) _amaxI.destroy(); } catch (_) {}
    _scr = null; _kv = null; _conv = null; _idsBuf = null; _amaxV = null; _amaxI = null;
  }

  // One forward over T tokens at absolute positions [posBase, posBase+T).
  // opts.chain: embed reads the GPU token history at posBase (no CPU ids) — for chained decode.
  // opts.argmax: GPU-argmax the final logits into tokHist[posBase+T] (so the next chained
  //   forward's embed reads it) instead of leaving logits for a CPU readback.
  // opts.batched: caller owns beginBatch/endBatch (lets many forwards chain in one submit).
  async function forward(idsArr, posBase, opts) {
    opts = opts || {};
    const C = _cfg, H = C.hidden, T = opts.chain ? 1 : idsArr.length, S = posBase + T;
    const W = (n) => _weights[n];
    _ensureState();
    const s = _scr;
    if (!opts.chain) {
      if (!_idsBuf) _idsBuf = E.createBuffer(MATVEC_MAXT * 4, U.STORAGE | U.COPY_DST, 'ids');
      E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(idsArr));
    }
    uniformReset();
    if (!opts.batched) E.beginBatch();
    if (opts.chain) await embedGather(s.tokHist, W('model.embed_tokens.weight').buf, s.x, T, H, posBase);
    else await embedGather(_idsBuf, W('model.embed_tokens.weight').buf, s.x, T, H, 0);
    for (let l = 0; l < C.numLayers; l++) {
      const p = 'model.layers.' + l + '.';
      await rmsnorm(s.x, W(p + 'operator_norm.weight').buf, s.normed, T, H, C.rmsEps);
      if (_conv[l]) {   // conv mixer
        await matvecQ(s.normed, W(p + 'conv.in_proj.weight'), s.bcx, T, 3 * H, H);
        const st = _conv[l];
        const sIn = st.cur === 0 ? st.a : st.b, sOut = st.cur === 0 ? st.b : st.a;
        await shortConv(s.bcx, W(p + 'conv.conv.weight').buf, sIn, sOut, s.convy, T, H, C.convL, posBase > 0);
        st.cur ^= 1;
        await matvecQ(s.convy, W(p + 'conv.out_proj.weight'), s.x, T, H, H, true);   // + residual
      } else {          // GQA attention
        const nHq = C.nHeads, nKv = C.nKvHeads, hd = C.headDim;
        await matvecQ(s.normed, W(p + 'self_attn.q_proj.weight'), s.q, T, nHq * hd, H);
        await matvecQ(s.normed, W(p + 'self_attn.k_proj.weight'), s.k, T, nKv * hd, H);
        await matvecQ(s.normed, W(p + 'self_attn.v_proj.weight'), s.v, T, nKv * hd, H);
        await ropeQK(s.q, W(p + 'self_attn.q_layernorm.weight').buf, s.qr, T, nHq, hd, posBase, C.ropeTheta, C.rmsEps);
        await ropeQK(s.k, W(p + 'self_attn.k_layernorm.weight').buf, s.kr, T, nKv, hd, posBase, C.ropeTheta, C.rmsEps);
        E.copyBuffer(s.kr, 0, _kv[l].k, posBase * nKv * hd * 4, T * nKv * hd * 4);
        E.copyBuffer(s.v, 0, _kv[l].v, posBase * nKv * hd * 4, T * nKv * hd * 4);
        await attention(s.qr, _kv[l].k, _kv[l].v, s.attn, T, S, nHq, nKv, hd);
        await matvecQ(s.attn, W(p + 'self_attn.out_proj.weight'), s.x, T, H, nHq * hd, true);   // + residual
      }
      await rmsnorm(s.x, W(p + 'ffn_norm.weight').buf, s.normed, T, H, C.rmsEps);
      if (C.moe && l >= C.denseLayers) {
        // Mixture-of-experts FFN (T=1). BRING-UP dispatch: compute the router on-GPU, read
        // back the 4 chosen expert ids + weights, then run those 4 standard SwiGLU experts
        // and weight-sum them into the residual. Each expert reuses the dense matvecQ/swiglu
        // path (the checkpoint stores per-expert w1/w2/w3, not a fused tensor). The per-layer
        // readback is the correctness-first shortcut; the GPU-resident indexed dispatch over
        // packed expert tensors (no readback) is the queued optimization.
        const nE = C.nExperts, K = C.topK, eI = C.expertI;
        await matvecQ(s.normed, W(p + 'feed_forward.gate.weight'), s.rlogits, 1, nE, H);   // router logits
        await router(s.rlogits, W(p + 'feed_forward.expert_bias').buf, s.ridx, s.rwt, 1, nE, K);
        if (_weights['moe.' + l + '.w1']) {
          // GPU-RESIDENT indexed dispatch — no readback, whole block stays batched.
          const w1 = W('moe.' + l + '.w1'), w3 = W('moe.' + l + '.w3'), w2 = W('moe.' + l + '.w2');
          await idxGemv(s.normed, w1, s.ridx, s.moeGate, K, eI, H, 0);        // [K, eI]
          await idxGemv(s.normed, w3, s.ridx, s.moeUp, K, eI, H, 0);          // [K, eI]
          await swiglu(s.moeGate, s.moeUp, s.moeAct, K * eI);
          await idxGemv(s.moeAct, w2, s.ridx, s.moeOut, K, H, eI, eI);        // [K, H], per-k input stride eI
          await moeCombine(s.x, s.moeOut, s.rwt, H, K);                       // x += Σ_k wt[k]·out[k]
        } else {
          // FALLBACK (__noMoePack): readback the routing, run the per-expert buffers.
          await E.endBatch();
          const idx = new Uint32Array((await E.readF32(s.ridx, K)).buffer);
          const wt = await E.readF32(s.rwt, K);
          E.beginBatch();
          for (let k = 0; k < K; k++) {
            const ep = p + 'feed_forward.experts.' + idx[k] + '.';
            await matvecQ(s.normed, W(ep + 'w1.weight'), s.gate, 1, eI, H);
            await matvecQ(s.normed, W(ep + 'w3.weight'), s.up, 1, eI, H);
            await swiglu(s.gate, s.up, s.swi, eI);
            await matvecQ(s.swi, W(ep + 'w2.weight'), s.eout, 1, H, eI);
            await axpy(s.x, s.eout, H, wt[k]);
          }
        }
      } else {
        const I = C.ffI;
        await matvecQ(s.normed, W(p + 'feed_forward.w1.weight'), s.gate, T, I, H);
        await matvecQ(s.normed, W(p + 'feed_forward.w3.weight'), s.up, T, I, H);
        await swiglu(s.gate, s.up, s.swi, T * I);
        await matvecQ(s.swi, W(p + 'feed_forward.w2.weight'), s.x, T, H, I, true);              // + residual
      }
    }
    await rmsnorm(s.x, W('model.embedding_norm.weight').buf, s.normed, T, H, C.rmsEps);
    E.copyBuffer(s.normed, (T - 1) * H * 4, s.last, 0, H * 4);
    await matvecQ(s.last, W('lm_head.weight'), s.logits, 1, C.vocab, H);
    if (opts.argmax) await argmaxKernel(s.logits, s.tokHist, C.vocab, posBase + T);   // GPU argmax → tokHist (no CPU roundtrip)
    if (!opts.batched) await E.endBatch();
  }

  async function readU32Range(buf, idx, count) {
    const bytes = count * 4;
    const staging = E.createBuffer(bytes, U.COPY_DST | U.MAP_READ, 'rd');
    const enc = E.device().createCommandEncoder();
    enc.copyBufferToBuffer(buf, idx * 4, staging, 0, bytes);
    E.device().queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return out;
  }

  const GEN_BATCH = 8;   // decode tokens chained GPU-resident per submit (1 readback per batch)
  async function generate(prompt, { maxTokens = 64, onToken, signal } = {}) {
    if (!_loaded) throw new Error('loadModel first');
    const C = _cfg;
    const ids = TOK.encodeChat([{ role: 'user', content: prompt }]);
    if (ids.length + maxTokens + 2 > MAX_SEQ) throw new Error('prompt too long for bring-up MAX_SEQ');
    // chained decode needs the GPU-resident path (no mid-forward readback); the __noMoePack
    // fallback can't chain, so it uses per-token CPU argmax.
    const chainable = !(C.moe && globalThis.__noMoePack);
    let pos = 0;
    const CH = C.moe ? 1 : MATVEC_MAXT;   // MoE runs T=1 (per-token routing); dense chunks at 32
    for (let off = 0; off < ids.length; off += CH) {   // prefill; last forward GPU-argmaxes → tokHist[L]
      if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
      const chunk = ids.slice(off, Math.min(off + CH, ids.length));
      await forward(chunk, pos, { argmax: chainable && (off + CH >= ids.length) });
      pos += chunk.length;
    }
    const outIds = [], imEnd = TOK.imEnd();
    if (chainable) {
      // GPU-RESIDENT chained decode: GEN_BATCH forwards in one submit (embed←tokHist, argmax→tokHist),
      // ONE readback per batch. Removes the per-token CPU roundtrip that starved GPU submit windows.
      let _fwdMs = 0, _rbMs = 0;
      // The FIRST token was argmax'd by the last prefill forward into tokHist[L]; emit it, then
      // the chained forwards consume it as input and produce tokHist[L+1..].
      let stopped = false;
      { const f = (await readU32Range(_scr.tokHist, ids.length, 1))[0];
        if (f === C.eos || f === imEnd) stopped = true;
        else { outIds.push(f); if (onToken) { try { onToken(TOK.decode([f]), f); } catch (_) {} } } }
      while (!stopped && outIds.length < maxTokens) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const K = Math.min(GEN_BATCH, maxTokens - outIds.length, MAX_SEQ - 1 - pos);
        if (K <= 0) break;
        const tf = performance.now();
        E.beginBatch();
        for (let k = 0; k < K; k++) await forward(null, pos + k, { chain: true, argmax: true, batched: true });
        await E.endBatch();
        _fwdMs += performance.now() - tf;
        const tr = performance.now();
        const toks = await readU32Range(_scr.tokHist, pos + 1, K);   // the K tokens argmax'd this batch
        _rbMs += performance.now() - tr;
        for (let k = 0; k < K; k++) { const t = toks[k]; if (t === C.eos || t === imEnd) { stopped = true; break; } outIds.push(t); if (onToken) { try { onToken(TOK.decode(outIds.slice(-4)).slice(-24), t); } catch (_) {} } }
        pos += K;
        if (stopped) break;
      }
      _lastProf = { mode: 'chained', batch: GEN_BATCH, forward_ms_per_tok: +(_fwdMs / Math.max(1, outIds.length)).toFixed(1), readback_ms_per_tok: +(_rbMs / Math.max(1, outIds.length)).toFixed(1) };
    } else {
      let _amMs = 0, _fwdMs = 0;   // fallback: per-token CPU argmax (readback path)
      for (let i = 0; i < maxTokens; i++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const ta = performance.now();
        const logits = await E.readF32(_scr.logits, C.vocab);
        let best = 0, bv = -Infinity;
        for (let j = 0; j < C.vocab; j++) if (logits[j] > bv) { bv = logits[j]; best = j; }
        _amMs += performance.now() - ta;
        if (best === C.eos || best === imEnd) break;
        outIds.push(best);
        if (onToken) { try { onToken(TOK.decode(outIds.slice(-4)).slice(-24), best); } catch (_) {} }
        const tf = performance.now();
        await forward([best], pos); pos++;
        _fwdMs += performance.now() - tf;
      }
      _lastProf = { mode: 'cpu-argmax', argmax_ms_per_tok: +(_amMs / Math.max(1, outIds.length)).toFixed(1), forward_ms_per_tok: +(_fwdMs / Math.max(1, outIds.length)).toFixed(1) };
    }
    try { console.log('[lfm25 prof] ' + JSON.stringify(_lastProf)); } catch (_) {}
    return TOK.decode(outIds);
  }
  let _lastProf = null;

  return { CONFIG, MODELS, shortConv, router, selfTestKernels, loadModel, unload: () => { _freeState(); unload(); }, inventory, TOK, isLoaded: () => _loaded, variant: () => _variant, generate, forward, lastProf: () => _lastProf };
})();

if (typeof window !== 'undefined') window.SandpieLfm25 = SandpieLfm25;
if (typeof self !== 'undefined') self.SandpieLfm25 = SandpieLfm25;
