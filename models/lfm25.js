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
    return out;
  }

  return { CONFIG, shortConv, router, selfTestKernels };
})();

if (typeof window !== 'undefined') window.SandpieLfm25 = SandpieLfm25;
if (typeof self !== 'undefined') self.SandpieLfm25 = SandpieLfm25;
