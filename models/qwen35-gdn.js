// sandpie/modules/webgpu-gdn.js — Qwen3.5-0.8B (Qwen3_5ForConditionalGeneration, text-only)
// port for the hand-written WebGPU engine. PHASE A: the Gated-DeltaNet-specific kernels +
// self-tests. Forward graph / loader / tokenizer follow in later phases (LFM2.5 pattern).
//
// WHY THIS MODEL: best sub-1B model as of 2026-07 (AA Intelligence Index 9 vs 6.5 for
// Qwen3-0.6B), Apache 2.0, 262K context. 18 of 24 layers are Gated DeltaNet with O(1)
// per-token recurrent state (16×128×128 f32 = 1MB/layer, NO growing KV) — only 6 layers
// are gated GQA attention. Constant-memory attention is exactly what mobile wants.
//
// ARCHITECTURE (config.json + transformers modeling_qwen3_next.py, verified 2026-07-20):
//   24 layers; FULL ATTENTION at [3,7,11,15,19,23], Gated DeltaNet elsewhere
//   hidden 1024 · vocab 248320 (tied) · rope_theta 1e7 · rms_eps 1e-6 · SwiGLU I=3584
//   attn: 8 q-heads / 2 kv-heads × headDim 256, q/k RMSNorm, OUTPUT GATE (silu)
//   GDN:  16 qk-heads × dk 128, 16 v-heads × dv 128, conv1d L=4 (depthwise, no bias, SiLU)
//         over concatenated q|k|v (conv dim 6144); β = sigmoid(b);
//         decay per step g = exp(−exp(A_log)·softplus(a + dt_bias))  (a,b from in_proj_ba)
//         recurrence (per head, S ∈ R^{dk×dv}):
//           S ← g_t·S;  kv = Sᵀk_t;  Δ = β_t(v_t − kv);  S ← S + k_t Δᵀ;  o_t = Sᵀq_t
//         q,k L2-normalized per head (eps 1e-6) inside the kernel; q scaled 1/√dk
//         output: RMSNorm(o)·weight  THEN  ·silu(z)   ("Norm before gate" — NOT Mamba order)
//
// LAYOUTS (verified against the actual checkpoint header 2026-07-20): unlike Qwen3-Next,
// the Qwen3.5 checkpoint ships in_proj_qkv [6144,H] (plain contiguous q|k|v — the split
// is torch.split, NOT per-head interleaved), plus separate in_proj_z/in_proj_b/in_proj_a.
// Kernels assume exactly that layout; NO permutation needed on the GDN side. The one
// interleaved tensor is attention q_proj [4096,H] = per-head [q(256)|gate(256)]×8 — the
// loader permutes its rows to [all q | all gate] once (gate = sigmoid, applied to attn
// out before o_proj). RoPE is PARTIAL: rotary factor 0.25 → first 64 of 256 dims (mrope
// collapses to standard RoPE for text-only). A_log + linear_attn.norm ship as F32.
//
// PHASE PLAN:
//   A (this file, now): gdnConv, gdnGates, gdnQkNorm, gdnDelta, gatedRmsNorm + self-tests
//     vs CPU refs incl. ≥512-token state continuity (numerical-drift guard).
//   B: loader (streamed safetensors → quant cache; skip model.visual.*; row-permute
//     in_proj_qkvz/in_proj_ba; A_log/dt_bias/conv taps unquantized), tokenizer (vocab 248320).
//   C: forward graph (GDN block | gated GQA attn headDim-256 | SwiGLU), generate/_streamIds/
//     runConversation matching the SandpieQwen3 contract, grammar + logprob hooks.
//   D: worker engineFor routing, picker entry, embed sharding (248320-vocab > Adreno bind cap).

const SandpieQwen35 = (function () {
  'use strict';

  const E = (typeof window !== 'undefined' ? window : self).SandpieWebGPU;
  const U = (typeof GPUBufferUsage !== 'undefined') ? GPUBufferUsage : { STORAGE: 0x80, COPY_DST: 0x8, COPY_SRC: 0x4, UNIFORM: 0x40, MAP_READ: 0x1 };

  const CONFIG = {
    numLayers: 24, hidden: 1024, vocab: 248320,
    attnLayers: [3, 7, 11, 15, 19, 23],
    nHeads: 8, nKvHeads: 2, headDim: 256, ropeTheta: 10000000, rmsEps: 1e-6,
    intermediate: 3584, tieEmbeddings: true,
    gdn: { nHeads: 16, dk: 128, dv: 128, convL: 4 },   // convDim = 16*(128+128)+16*128 = 6144
  };
  const GDN_CONV_DIM = CONFIG.gdn.nHeads * (CONFIG.gdn.dk * 2 + CONFIG.gdn.dv);   // q|k|v = 6144

  let _uPool = [], _uIdx = 0;
  function uniformReset() { _uIdx = 0; }
  function uniform(arr) {
    let buf = _uPool[_uIdx];
    if (!buf) { buf = E.createBuffer(32, U.UNIFORM | U.COPY_DST, 'gu' + _uIdx); _uPool[_uIdx] = buf; }
    E.device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset || 0, arr.byteLength);
    _uIdx++;
    return buf;
  }

  // ============================================================
  // Kernel 1 — depthwise causal conv1d (L=4) + SiLU over the packed q|k|v
  // projection output. qkv[T, C] + stateIn[C, L-1] →
  //   y[T,C] = silu(Σ_k w[c,k]·x(t−(L−1)+k, c))     (no bias, no gating)
  //   stateOut[C, L-1] = last L-1 RAW inputs (pre-activation) — the decode carry.
  // Double-buffered state (read stateIn / write stateOut, caller swaps) — same
  // race-avoidance as LFM2.5's conv: with T>1 the carry readers and the writer
  // are in different workgroups. useState=0 zero-pads (fresh prefill).
  // ============================================================
  const CONV_WGSL = `
enable f16;
struct P { T:u32, C:u32, L:u32, useState:u32 };
@group(0) @binding(0) var<storage, read>       x        : array<f32>;   // [T, C]
@group(0) @binding(1) var<storage, read>       w        : array<f16>;   // [C, L] depthwise taps
@group(0) @binding(2) var<storage, read>       stateIn  : array<f32>;   // [C, L-1]
@group(0) @binding(3) var<storage, read_write> stateOut : array<f32>;   // [C, L-1]
@group(0) @binding(4) var<storage, read_write> y        : array<f32>;   // [T, C]
@group(0) @binding(5) var<uniform>             p        : P;
fn x_at(t:i32, c:u32, C:u32, Lm1:i32, useState:u32) -> f32 {
  if (t >= 0) { return x[u32(t)*C + c]; }
  if (useState == 0u) { return 0.0; }
  return stateIn[c*u32(Lm1) + u32(t + Lm1)];
}
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let C = p.C; let T = p.T; let L = p.L; let Lm1 = i32(L - 1u);
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= T*C) { return; }
  let t = i32(i / C); let c = i % C;
  var acc = 0.0;
  for (var k:u32 = 0u; k < L; k = k + 1u) {
    acc = acc + f32(w[c*L + k]) * x_at(t - Lm1 + i32(k), c, C, Lm1, p.useState);
  }
  y[u32(t)*C + c] = acc / (1.0 + exp(-acc));           // SiLU
  if (u32(t) == T - 1u) {                               // last-token thread persists the carry
    for (var q:i32 = 0; q < Lm1; q = q + 1) {
      stateOut[c*u32(Lm1) + u32(q)] = x_at(i32(T) - Lm1 + q, c, C, Lm1, p.useState);
    }
  }
}`;
  function gdnConv(xBuf, wBuf, stateInBuf, stateOutBuf, yBuf, T, C, L, useState) {
    const p = uniform(new Uint32Array([T, C, L, useState ? 1 : 0]));
    const pipe = E.getPipeline('gdn.conv', CONV_WGSL);
    const n = Math.ceil((T * C) / 64), gx = Math.min(n, 65535), gy = Math.ceil(n / gx);
    return E.dispatch(pipe, [xBuf, wBuf, stateInBuf, stateOutBuf, yBuf, p], [gx, gy, 1]);
  }

  // ============================================================
  // Kernel 2 — per-token gates. ba[T, 2V] (contiguous b|a — loader permuted) →
  //   beta[T,V] = sigmoid(b);  gexp[T,V] = exp(−exp(A_log[h])·softplus(a + dt_bias[h]))
  // gexp is the per-step multiplicative state decay (always in (0,1]).
  // ============================================================
  const GATES_WGSL = `
struct P { T:u32, V:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       ba      : array<f32>;   // [T, 2V] = b|a
@group(0) @binding(1) var<storage, read>       alog    : array<f32>;   // [V]
@group(0) @binding(2) var<storage, read>       dtbias  : array<f32>;   // [V]
@group(0) @binding(3) var<storage, read_write> beta    : array<f32>;   // [T, V]
@group(0) @binding(4) var<storage, read_write> gexp    : array<f32>;   // [T, V]
@group(0) @binding(5) var<uniform>             p       : P;
fn softplus(x:f32) -> f32 { if (x > 20.0) { return x; } return log(1.0 + exp(x)); }
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x; let V = p.V;
  if (i >= p.T*V) { return; }
  let t = i / V; let h = i % V;
  beta[i] = 1.0 / (1.0 + exp(-ba[t*2u*V + h]));
  let a = ba[t*2u*V + V + h];
  gexp[i] = exp(-exp(alog[h]) * softplus(a + dtbias[h]));
}`;
  function gdnGates(baBuf, alogBuf, dtbiasBuf, betaBuf, gexpBuf, T, V) {
    const p = uniform(new Uint32Array([T, V, 0, 0]));
    const pipe = E.getPipeline('gdn.gates', GATES_WGSL);
    return E.dispatch(pipe, [baBuf, alogBuf, dtbiasBuf, betaBuf, gexpBuf, p], [Math.ceil((T * V) / 64), 1, 1]);
  }

  // ============================================================
  // Kernel 3 — per-head L2 normalization of q and k (IN-PLACE on the conv
  // output, which is laid out [T, q(2048)|k(2048)|v(2048)]). One workgroup per
  // (t, head, {q|k}); 128 threads reduce Σx². q additionally scaled 1/√dk.
  //   x ← x · rsqrt(Σx² + 1e-6) · (isQ ? 1/√dk : 1)
  // v is untouched. eps and placement match torch_recurrent_gated_delta_rule.
  // ============================================================
  const QKNORM_WGSL = `
struct P { T:u32, H:u32, D:u32, C:u32 };
@group(0) @binding(0) var<storage, read_write> x : array<f32>;   // [T, C] conv output
@group(0) @binding(1) var<uniform>             p : P;
var<workgroup> red : array<f32, 128>;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let D = p.D; let H = p.H;
  let g = wg.x;                       // g in [0, T*H*2): (t, head, q-or-k)
  let t = g / (H*2u); let r = g % (H*2u);
  let isQ = r < H;
  let h = select(r - H, r, isQ);
  let base = t*p.C + select(H*D, 0u, isQ) + h*D;   // q at offset 0, k at H*D
  let i = lid.x;
  let v = x[base + i];
  red[i] = v*v;
  workgroupBarrier();
  for (var s:u32 = 64u; s > 0u; s = s >> 1u) {
    if (i < s) { red[i] = red[i] + red[i + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] + 1e-6);
  let sc = select(1.0, inverseSqrt(f32(D)), isQ);
  x[base + i] = v * inv * sc;
}`;
  function gdnQkNorm(xBuf, T, H, D, C) {
    const p = uniform(new Uint32Array([T, H, D, C]));
    const pipe = E.getPipeline('gdn.qknorm', QKNORM_WGSL);
    return E.dispatch(pipe, [xBuf, p], [T * H * 2, 1, 1]);
  }

  // ============================================================
  // Kernel 4 — the gated delta rule recurrence. One workgroup per (head, j∈dv);
  // thread i owns state element S[h, i, j] IN A REGISTER across the whole T loop
  // (zero state traffic inside the chunk). Per step t:
  //   S ← S·gexp_t ;  kv = Σ_i S[i,j]·k_t[i]  (wg reduction) ;
  //   Δ_j = β_t·(v_t[j] − kv) ;  S[i,j] += k_t[i]·Δ_j ;
  //   o_t[h,j] = Σ_i S[i,j]·q_t[i]            (wg reduction)
  // Exactly torch_recurrent_gated_delta_rule's order (decay BEFORE the kv read).
  // State double-buffered (stateIn → registers → stateOut; caller swaps).
  // T=1 decode is the same kernel. qkv layout: [T, q|k|v] with H*D each.
  // ============================================================
  const DELTA_WGSL = `
struct P { T:u32, H:u32, D:u32, useState:u32 };
@group(0) @binding(0) var<storage, read>       qkv      : array<f32>;   // [T, 3*H*D] normed q|k|v
@group(0) @binding(1) var<storage, read>       beta     : array<f32>;   // [T, H]
@group(0) @binding(2) var<storage, read>       gexp     : array<f32>;   // [T, H]
@group(0) @binding(3) var<storage, read>       stateIn  : array<f32>;   // [H, D, D] (i-major: [h][i][j])
@group(0) @binding(4) var<storage, read_write> stateOut : array<f32>;   // [H, D, D]
@group(0) @binding(5) var<storage, read_write> o        : array<f32>;   // [T, H*D]
@group(0) @binding(6) var<uniform>             p        : P;
var<workgroup> red : array<f32, 128>;
var<workgroup> kvm : f32;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let D = p.D; let H = p.H; let C = 3u*H*D;
  let h = wg.x / D; let j = wg.x % D;
  let i = lid.x;
  var s = 0.0;
  if (p.useState != 0u) { s = stateIn[(h*D + i)*D + j]; }
  for (var t:u32 = 0u; t < p.T; t = t + 1u) {
    let row = t*C;
    let kt = qkv[row + H*D + h*D + i];
    s = s * gexp[t*H + h];
    red[i] = s * kt;
    workgroupBarrier();
    for (var r:u32 = 64u; r > 0u; r = r >> 1u) {
      if (i < r) { red[i] = red[i] + red[i + r]; }
      workgroupBarrier();
    }
    if (i == 0u) { kvm = red[0]; }
    workgroupBarrier();
    let delta = beta[t*H + h] * (qkv[row + 2u*H*D + h*D + j] - kvm);
    s = s + kt * delta;
    let qt = qkv[row + h*D + i];
    red[i] = s * qt;
    workgroupBarrier();
    for (var r:u32 = 64u; r > 0u; r = r >> 1u) {
      if (i < r) { red[i] = red[i] + red[i + r]; }
      workgroupBarrier();
    }
    if (i == 0u) { o[t*H*D + h*D + j] = red[0]; }
    workgroupBarrier();
  }
  stateOut[(h*D + i)*D + j] = s;
}`;
  function gdnDelta(qkvBuf, betaBuf, gexpBuf, stateInBuf, stateOutBuf, oBuf, T, H, D, useState) {
    const p = uniform(new Uint32Array([T, H, D, useState ? 1 : 0]));
    const pipe = E.getPipeline('gdn.delta', DELTA_WGSL);
    return E.dispatch(pipe, [qkvBuf, betaBuf, gexpBuf, stateInBuf, stateOutBuf, oBuf, p], [H * D, 1, 1]);
  }

  // ============================================================
  // Kernel 5 — gated RMSNorm (Qwen3NextRMSNormGated — "Norm before gate"):
  //   out[t,h,:] = (o / rms(o)) · weight · silu(z)     per (t, head) over dv
  // weight is [dv] (shared across heads). z laid out [T, H*dv] (loader-permuted).
  // NOTE: norm FIRST, then gate — this is NOT the Mamba2 order.
  // ============================================================
  const GNORM_WGSL = `
enable f16;
struct P { T:u32, H:u32, D:u32, eps:f32 };
@group(0) @binding(0) var<storage, read>       o   : array<f32>;   // [T, H*D]
@group(0) @binding(1) var<storage, read>       z   : array<f32>;   // [T, H*D]
@group(0) @binding(2) var<storage, read>       w   : array<f16>;   // [D]
@group(0) @binding(3) var<storage, read_write> out : array<f32>;   // [T, H*D]
@group(0) @binding(4) var<uniform>             p   : P;
var<workgroup> red : array<f32, 128>;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let D = p.D;
  let t = wg.x / p.H; let h = wg.x % p.H;
  let base = t*p.H*D + h*D;
  let i = lid.x;
  let v = o[base + i];
  red[i] = v*v;
  workgroupBarrier();
  for (var s:u32 = 64u; s > 0u; s = s >> 1u) {
    if (i < s) { red[i] = red[i] + red[i + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(D) + p.eps);
  let zg = z[base + i];
  out[base + i] = v * inv * f32(w[i]) * (zg / (1.0 + exp(-zg)));
}`;
  function gatedRmsNorm(oBuf, zBuf, wBuf, outBuf, T, H, D, eps) {
    const pb = new ArrayBuffer(16); const dv = new DataView(pb);
    dv.setUint32(0, T, true); dv.setUint32(4, H, true); dv.setUint32(8, D, true); dv.setFloat32(12, eps, true);
    const p = uniform(new Uint32Array(pb));
    const pipe = E.getPipeline('gdn.gnorm', GNORM_WGSL);
    return E.dispatch(pipe, [oBuf, zBuf, wBuf, outBuf, p], [T * H, 1, 1]);
  }

  // ============================================================
  // Self-tests vs CPU references (no model download needed).
  // ============================================================
  const ST = () => U.STORAGE | U.COPY_DST | U.COPY_SRC;
  function f32buf(a) { return E.uploadF32(a instanceof Float32Array ? a : new Float32Array(a), ST()); }
  function f16buf(a) { const f = a instanceof Float32Array ? a : new Float32Array(a); const bits = E.f32ToF16(f); const b = E.createBuffer(f.length * 2, ST(), 'gw16'); E.device().queue.writeBuffer(b, 0, bits); return b; }
  const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
  const silu = (x) => x / (1 + Math.exp(-x));
  const softplus = (x) => x > 20 ? x : Math.log(1 + Math.exp(x));

  // CPU reference: conv (raw carry, SiLU output)
  function convRef(x, w, state, T, C, L, useState) {
    const y = new Float32Array(T * C), Lm1 = L - 1;
    const at = (t, c) => t >= 0 ? x[t * C + c] : (useState ? state[c * Lm1 + (t + Lm1)] : 0);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) {
      let acc = 0; for (let k = 0; k < L; k++) acc += w[c * L + k] * at(t - Lm1 + k, c);
      y[t * C + c] = silu(acc);
    }
    const st = new Float32Array(C * Lm1);
    for (let c = 0; c < C; c++) for (let q = 0; q < Lm1; q++) st[c * Lm1 + q] = at(T - Lm1 + q, c);
    return { y, st };
  }
  // CPU reference: full GDN chain from NORMED q,k,v — the gated delta rule.
  function deltaRef(qkv, beta, gexp, stateIn, T, H, D, useState) {
    const S = stateIn ? Float32Array.from(stateIn) : new Float32Array(H * D * D);
    const o = new Float32Array(T * H * D);
    for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) {
      const q0 = t * 3 * H * D + h * D, k0 = q0 + H * D, v0 = q0 + 2 * H * D;
      const g = gexp[t * H + h], b = beta[t * H + h];
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) S[(h * D + i) * D + j] *= g;
      for (let j = 0; j < D; j++) {
        let kv = 0; for (let i = 0; i < D; i++) kv += S[(h * D + i) * D + j] * qkv[k0 + i];
        const delta = b * (qkv[v0 + j] - kv);
        for (let i = 0; i < D; i++) S[(h * D + i) * D + j] += qkv[k0 + i] * delta;
      }
      for (let j = 0; j < D; j++) {
        let acc = 0; for (let i = 0; i < D; i++) acc += S[(h * D + i) * D + j] * qkv[q0 + i];
        o[t * H * D + h * D + j] = acc;
      }
    }
    return { o, S };
  }
  // CPU reference: per-head l2 norm of q,k in-place (q scaled 1/sqrt(D))
  function qkNormRef(x, T, H, D, C) {
    for (let t = 0; t < T; t++) for (let r = 0; r < 2 * H; r++) {
      const isQ = r < H, h = isQ ? r : r - H;
      const base = t * C + (isQ ? 0 : H * D) + h * D;
      let ss = 0; for (let i = 0; i < D; i++) ss += x[base + i] * x[base + i];
      const inv = 1 / Math.sqrt(ss + 1e-6), sc = isQ ? 1 / Math.sqrt(D) : 1;
      for (let i = 0; i < D; i++) x[base + i] *= inv * sc;
    }
  }

  async function selfTestKernels() {
    await E.init();
    const out = [];
    const check = (name, err, tol) => out.push({ name, ok: err < (tol || 1e-3), err });

    // --- gdnConv: fresh prefill + state continuity (T=6 ≡ 6×T=1) ---
    {
      const T = 7, C = 96, L = 4, Lm1 = 3;
      const x = new Float32Array(T * C); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.13);
      const w = new Float32Array(C * L); for (let i = 0; i < w.length; i++) w[i] = 0.3 + 0.5 * Math.cos(i * 0.07);
      const ref = convRef(x, w, null, T, C, L, 0);
      const xb = f32buf(x), wb = f16buf(w), s0 = f32buf(new Float32Array(C * Lm1)), s1 = E.createBuffer(C * Lm1 * 4, ST(), 's1'), yb = E.createBuffer(T * C * 4, ST(), 'y');
      await gdnConv(xb, wb, s0, s1, yb, T, C, L, false);
      check('gdn conv (prefill)', Math.max(maxAbs(await E.readF32(yb, T * C), ref.y), maxAbs(await E.readF32(s1, C * Lm1), ref.st)), 5e-3);
      [xb, wb, s0, s1, yb].forEach(b => b.destroy());
    }
    {
      const T = 6, C = 64, L = 4, Lm1 = 3;
      const x = new Float32Array(T * C); for (let i = 0; i < x.length; i++) x[i] = Math.cos(i * 0.11) * (1 + (i % 5) * 0.1);
      const w = new Float32Array(C * L); for (let i = 0; i < w.length; i++) w[i] = 0.2 + 0.4 * Math.sin(i * 0.05);
      const full = convRef(x, w, null, T, C, L, 0);
      const wb = f16buf(w);
      let sIn = f32buf(new Float32Array(C * Lm1)), sOut = E.createBuffer(C * Lm1 * 4, ST(), 'so');
      const gotY = new Float32Array(T * C);
      for (let t = 0; t < T; t++) {
        const xb = f32buf(x.slice(t * C, (t + 1) * C)), yb = E.createBuffer(C * 4, ST(), 'y1');
        await gdnConv(xb, wb, sIn, sOut, yb, 1, C, L, t > 0);
        gotY.set(await E.readF32(yb, C), t * C);
        [xb, yb].forEach(b => b.destroy());
        const tmp = sIn; sIn = sOut; sOut = tmp;
      }
      check('gdn conv (decode state continuity)', maxAbs(gotY, full.y), 5e-3);
      [wb, sIn, sOut].forEach(b => b.destroy());
    }

    // --- gdnGates: beta/gexp formulas ---
    {
      const T = 5, V = 16;
      const ba = new Float32Array(T * 2 * V); for (let i = 0; i < ba.length; i++) ba[i] = Math.sin(i * 0.31) * 2;
      const alog = new Float32Array(V); for (let i = 0; i < V; i++) alog[i] = Math.log(0.5 + i * 0.4);
      const dtb = new Float32Array(V); dtb.fill(1);
      const refB = new Float32Array(T * V), refG = new Float32Array(T * V);
      for (let t = 0; t < T; t++) for (let h = 0; h < V; h++) {
        refB[t * V + h] = 1 / (1 + Math.exp(-ba[t * 2 * V + h]));
        refG[t * V + h] = Math.exp(-Math.exp(alog[h]) * softplus(ba[t * 2 * V + V + h] + dtb[h]));
      }
      const bab = f32buf(ba), ab = f32buf(alog), db = f32buf(dtb), bb = E.createBuffer(T * V * 4, ST(), 'b'), gb = E.createBuffer(T * V * 4, ST(), 'g');
      await gdnGates(bab, ab, db, bb, gb, T, V);
      check('gdn gates (beta)', maxAbs(await E.readF32(bb, T * V), refB), 1e-5);
      check('gdn gates (gexp)', maxAbs(await E.readF32(gb, T * V), refG), 1e-5);
      [bab, ab, db, bb, gb].forEach(b => b.destroy());
    }

    // --- gdnQkNorm: in-place per-head l2 norm, q scaled ---
    {
      const T = 3, H = 4, D = 128, C = 3 * H * D;
      const x = new Float32Array(T * C); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.17) * 2;
      const ref = Float32Array.from(x); qkNormRef(ref, T, H, D, C);
      const xb = f32buf(x);
      await gdnQkNorm(xb, T, H, D, C);
      check('gdn qknorm (l2 + q-scale, v untouched)', maxAbs(await E.readF32(xb, T * C), ref), 1e-5);
      xb.destroy();
    }

    // --- gdnDelta: prefill vs CPU ref ---
    {
      const T = 9, H = 2, D = 128;
      const qkv = new Float32Array(T * 3 * H * D); for (let i = 0; i < qkv.length; i++) qkv[i] = Math.sin(i * 0.113) * 0.5;
      qkNormRef(qkv, T, H, D, 3 * H * D);
      const beta = new Float32Array(T * H), gexp = new Float32Array(T * H);
      for (let i = 0; i < T * H; i++) { beta[i] = 0.3 + 0.6 * Math.abs(Math.sin(i)); gexp[i] = 0.85 + 0.14 * Math.abs(Math.cos(i * 0.7)); }
      const ref = deltaRef(qkv, beta, gexp, null, T, H, D, 0);
      const qb = f32buf(qkv), bb = f32buf(beta), gb = f32buf(gexp);
      const s0 = f32buf(new Float32Array(H * D * D)), s1 = E.createBuffer(H * D * D * 4, ST(), 'ds1'), ob = E.createBuffer(T * H * D * 4, ST(), 'do');
      await gdnDelta(qb, bb, gb, s0, s1, ob, T, H, D, false);
      check('gdn delta (prefill o)', maxAbs(await E.readF32(ob, T * H * D), ref.o), 2e-3);
      check('gdn delta (prefill state)', maxAbs(await E.readF32(s1, H * D * D), ref.S), 2e-3);
      [qb, bb, gb, s0, s1, ob].forEach(b => b.destroy());
    }

    // --- gdnDelta: LONG state continuity — 512 tokens as 8×64-chunks ≡ one T=512 CPU pass.
    // This is the numerical-drift guard: recurrent f32 state must not diverge across
    // chunk boundaries the way the real prefill (chunk 256) will drive it. ---
    {
      const T = 512, CH = 64, H = 1, D = 128;
      const qkv = new Float32Array(T * 3 * H * D); for (let i = 0; i < qkv.length; i++) qkv[i] = Math.sin(i * 0.0501) * 0.5 + 0.1 * Math.cos(i * 0.013);
      qkNormRef(qkv, T, H, D, 3 * H * D);
      const beta = new Float32Array(T * H), gexp = new Float32Array(T * H);
      for (let i = 0; i < T * H; i++) { beta[i] = 0.2 + 0.7 * Math.abs(Math.sin(i * 0.3)); gexp[i] = 0.9 + 0.099 * Math.abs(Math.cos(i * 0.11)); }
      const ref = deltaRef(qkv, beta, gexp, null, T, H, D, 0);
      const rowC = 3 * H * D;
      let sIn = f32buf(new Float32Array(H * D * D)), sOut = E.createBuffer(H * D * D * 4, ST(), 'ls');
      const gotO = new Float32Array(T * H * D);
      for (let c = 0; c < T / CH; c++) {
        const qb = f32buf(qkv.slice(c * CH * rowC, (c + 1) * CH * rowC));
        const bb = f32buf(beta.slice(c * CH * H, (c + 1) * CH * H));
        const gb = f32buf(gexp.slice(c * CH * H, (c + 1) * CH * H));
        const ob = E.createBuffer(CH * H * D * 4, ST(), 'lo');
        await gdnDelta(qb, bb, gb, sIn, sOut, ob, CH, H, D, c > 0);
        gotO.set(await E.readF32(ob, CH * H * D), c * CH * H * D);
        [qb, bb, gb, ob].forEach(b => b.destroy());
        const tmp = sIn; sIn = sOut; sOut = tmp;
      }
      check('gdn delta (512-tok chunked continuity)', maxAbs(gotO, ref.o), 5e-3);
      [sIn, sOut].forEach(b => b.destroy());
    }

    // --- gatedRmsNorm: norm-then-gate order ---
    {
      const T = 4, H = 3, D = 128, eps = 1e-6;
      const o = new Float32Array(T * H * D), z = new Float32Array(T * H * D), w = new Float32Array(D);
      for (let i = 0; i < o.length; i++) { o[i] = Math.sin(i * 0.21) * 3; z[i] = Math.cos(i * 0.17) * 2; }
      for (let i = 0; i < D; i++) w[i] = 0.8 + 0.4 * Math.sin(i * 0.09);
      const ref = new Float32Array(T * H * D);
      for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) {
        const base = t * H * D + h * D;
        let ss = 0; for (let i = 0; i < D; i++) ss += o[base + i] * o[base + i];
        const inv = 1 / Math.sqrt(ss / D + eps);
        for (let i = 0; i < D; i++) ref[base + i] = o[base + i] * inv * w[i] * silu(z[base + i]);
      }
      const ob = f32buf(o), zb = f32buf(z), wb = f16buf(w), rb = E.createBuffer(T * H * D * 4, ST(), 'gn');
      await gatedRmsNorm(ob, zb, wb, rb, T, H, D, eps);
      check('gdn gated rmsnorm (norm-before-gate)', maxAbs(await E.readF32(rb, T * H * D), ref), 5e-3);
      [ob, zb, wb, rb].forEach(b => b.destroy());
    }

    const fails = out.filter(r => !r.ok);
    console.table(out.map(r => ({ test: r.name, ok: r.ok ? '✓' : '✗ FAIL', maxErr: r.err.toExponential(2) })));
    return { ok: fails.length === 0, results: out };
  }

  // ============================================================
  // PHASE B — tokenizer + streaming weight loader.
  // Adapted from webgpu-lfm25.js (same design, minus MoE/expert-streaming/folder
  // store — this model is a single 1.75GB file, ~0.95GB resident after quant).
  // ============================================================
  const MODELS = {
    '0.8B': {
      root: 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/',
      file: 'model.safetensors-00001-of-00001.safetensors',
      cfg: { ...CONFIG, eos: 248044 },
    },
  };
  let _variant = null, _cfg = null, _weights = null, _loaded = false;

  // ---- tokenizer: byte-level BPE, pre_tokenizer regex extracted from tokenizer.json
  // (same generic machinery as lfm25; Qwen chat template — NO BOS token).
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
    let p = hfPattern.replace(/\(\?i:('s\|'t\|'re\|'ve\|'m\|'ll\|'d)\)/i,
      "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])");
    try { return new RegExp(p, 'gu'); } catch (_) {
      return /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
    }
  }
  const TOK = (function () {
    let vocab = null, idToTok = null, bpeRanks = null, byteEnc = null, byteDec = null, ready = false, pretokRe = null;
    let imStart = -1, imEnd = -1;
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
      let pat = null;
      const scan = (pt) => { if (!pt) return; if (pt.pattern && pt.pattern.Regex) pat = pt.pattern.Regex; (pt.pretokenizers || []).forEach(scan); };
      scan(j.pre_tokenizer);
      pretokRe = pat ? _jsRegexFrom(pat) : _jsRegexFrom('');
      ({ byteEnc, byteDec } = buildByteMaps());
      imStart = vocab['<|im_start|>'] ?? -1; imEnd = vocab['<|im_end|>'] ?? -1;
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
    // Qwen chat format: no BOS; <|im_start|>role\n…<|im_end|>\n per message.
    // Phase C replaces this with the full template (system/tools/<think> handling)
    // mirroring webgpu-qwen3.js's encodeChat.
    function encodeChat(messages, { addGenerationPrompt = true } = {}) {
      const ids = [];
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
    return { load, encodeText, encodeChat, decode, isReady: () => ready, specialIds: () => _addedIds, imEnd: () => imEnd, id: (t) => vocab ? vocab[t] : undefined };
  })();

  // ---- streaming single-pass loader (lfm25 design): header via Range, one sequential
  // body stream, each tensor quantized the moment its bytes complete, raw bytes dropped.
  const QGROUP = 32;
  const _f32a = new Float32Array(1), _u32a = new Uint32Array(_f32a.buffer);
  function _f32ToF16(v) { _f32a[0] = v; const x = _u32a[0]; const sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff, mant = x & 0x7fffff; if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00); let e = exp - 127 + 15; if (e >= 31) return sign | 0x7c00; if (e <= 0) { if (e < -10) return sign; const m = mant | 0x800000, sh = 14 - e; let h = m >>> sh; if ((m >>> (sh - 1)) & 1) h += 1; return sign | h; } let h = (e << 10) | (mant >>> 13); if ((mant >>> 12) & 1) h += 1; return sign | h; }
  function _f16ToF32s(h) { const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff; if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024); if (e === 31) return f ? NaN : (s ? -Infinity : Infinity); return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024); }
  function _bf16ToF16bits(u16) { const out = new Uint16Array(u16.length); const t = new Float32Array(1), ti = new Uint32Array(t.buffer); for (let i = 0; i < u16.length; i++) { ti[0] = u16[i] << 16; out[i] = _f32ToF16(t[0]); } return out; }
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
  // Quant plan by tensor name.
  const _skip = (n) => /^model\.visual\./.test(n) || /^mtp\./.test(n);   // vision tower + MTP head: never loaded
  const _isInt4 = (n) => /(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|out_proj|in_proj_qkv|in_proj_z)\.weight$/.test(n);
  const _isF32T = (n) => /(A_log|dt_bias|linear_attn\.norm\.weight)$/.test(n);   // small GDN params: exact
  const EMBED = 'model.language_model.embed_tokens.weight';

  // q_proj rows are per-head interleaved [q(256)|gate(256)]×8 (torch.chunk on the
  // reshaped [..., head, 2*hd] output). Permute rows → [all q | all gate] once here
  // so Phase C's projections are two contiguous halves. Zero runtime cost.
  function _permuteQGate(u16, rows, K, headDim) {
    const out = new Uint16Array(u16.length);
    const nH = rows / (2 * headDim);
    for (let h = 0; h < nH; h++) for (let d = 0; d < headDim; d++) {
      out.set(u16.subarray((h * 2 * headDim + d) * K, (h * 2 * headDim + d + 1) * K), (h * headDim + d) * K);                       // q half
      out.set(u16.subarray((h * 2 * headDim + headDim + d) * K, (h * 2 * headDim + headDim + d + 1) * K), (rows / 2 + h * headDim + d) * K);   // gate half
    }
    return out;
  }

  async function _uploadTensor(name, info, raw, sink) {
    if (_skip(name)) return;
    const numel = info.shape.reduce((a, b) => a * b, 1);
    const put = (buf, arr) => E.device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    // F32 tensors (A_log, linear_attn.norm) arrive as raw f32 — no conversion.
    if (info.dtype === 'F32') {
      const f32 = new Float32Array(raw.buffer, raw.byteOffset, numel).slice();
      const buf = E.createBuffer(f32.byteLength, ST(), name);
      put(buf, f32);
      _weights[name] = { buf, f32: true, shape: info.shape };
      if (sink) await sink.add(name, 'buf', f32, { kind: 'f32', shape: info.shape });
      return;
    }
    let u16 = new Uint16Array(raw.buffer, raw.byteOffset, numel);
    if (/self_attn\.q_proj\.weight$/.test(name)) u16 = _permuteQGate(u16, info.shape[0], info.shape[1], _cfg.headDim);
    if (_isInt4(name) && info.shape.length >= 2 && (info.shape[info.shape.length - 1] % QGROUP) === 0) {
      const K = info.shape[info.shape.length - 1], rows = numel / K;
      const { pack, scales } = _quantInt4(u16, rows, K);
      const packBuf = E.createBuffer(pack.byteLength, ST(), name + '.pack');
      const scBuf = E.createBuffer(scales.byteLength, ST(), name + '.sc');
      put(packBuf, pack); put(scBuf, scales);
      _weights[name] = { pack: packBuf, scales: scBuf, N: rows, K, int4: true, shape: info.shape };
      if (sink) { await sink.add(name, 'pack', pack, { kind: 'int4', shape: info.shape, N: rows, K }); await sink.add(name, 'scales', scales, { kind: 'int4', shape: info.shape, N: rows, K }); }
    } else if (_isF32T(name)) {
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
    // TIED lm_head: int4 twin of the embedding for the vocab GEMV, made while the
    // raw bf16 is still in hand (it is never resident again after this call).
    if (name === EMBED && !_weights['lm_head.weight']) {
      const K = info.shape[1], rows = info.shape[0];
      const { pack, scales } = _quantInt4(u16, rows, K);
      const packBuf = E.createBuffer(pack.byteLength, ST(), 'lm_head.pack');
      const scBuf = E.createBuffer(scales.byteLength, ST(), 'lm_head.sc');
      put(packBuf, pack); put(scBuf, scales);
      _weights['lm_head.weight'] = { pack: packBuf, scales: scBuf, N: rows, K, int4: true, shape: info.shape };
      if (sink) { await sink.add('lm_head.weight', 'pack', pack, { kind: 'int4', shape: info.shape, N: rows, K }); await sink.add('lm_head.weight', 'scales', scales, { kind: 'int4', shape: info.shape, N: rows, K }); }
    }
  }

  // ---- persistent quantized-weights cache (lfm25/qwen3 design; namespaced /gdn/).
  // GPU-ready bytes in fixed 64MiB chunks; manifest written LAST = commit point.
  const QC_NAME = 'sandpie-webgpu-quant';
  const QC_VER = 1;
  const QC_CHUNK = 64 * 1024 * 1024;
  const _qcUrl = (variant, part) => 'https://sandpie.quant/gdn/v' + QC_VER + '/' + variant + '/' + part;
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
          const pad = (4 - (globalOff % 4)) % 4;
          for (let p = 0; p < pad; p++) { if (used === QC_CHUNK) await flush(); buf[used++] = 0; globalOff++; }
          const lenPad = (4 - (bytes.byteLength % 4)) % 4;
          segs.push({ name, part, off: globalOff, len: bytes.byteLength + lenPad, ...meta });
          let src = 0;
          while (src < bytes.byteLength) {
            if (used === QC_CHUNK) await flush();
            const n = Math.min(QC_CHUNK - used, bytes.byteLength - src);
            buf.set(bytes.subarray(src, src + n), used); used += n; src += n; globalOff += n;
          }
          for (let p = 0; p < lenPad; p++) { if (used === QC_CHUNK) await flush(); buf[used++] = 0; globalOff++; }
        } catch (e) { dead = true; try { console.warn('[gdn] quant-cache write failed (quota?) — continuing uncached', e); } catch (_) {} }
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
    const bySeg = [];
    for (const s of m.segs) {
      const rec = _weights[s.name] || (_weights[s.name] = s.kind === 'int4' ? { N: s.N, K: s.K, int4: true, shape: s.shape } : { shape: s.shape, ...(s.kind === 'f32' ? { f32: true } : {}) });
      const buf = E.createBuffer(s.len, ST(), s.name + '.' + s.part);
      if (s.part === 'pack') rec.pack = buf; else if (s.part === 'scales') rec.scales = buf; else rec.buf = buf;
      bySeg.push({ ...s, buf });
    }
    for (let ci = 0; ci < m.nChunks; ci++) {
      const resp = await cache.match(_qcUrl(variant, 'c' + ci));
      if (!resp) return false;
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
    const h8 = await (await fetch(url, { headers: { Range: 'bytes=0-7' } })).arrayBuffer();
    const headerLen = Number(new DataView(h8).getBigUint64(0, true));
    const hResp = await fetch(url, { headers: { Range: 'bytes=8-' + (7 + headerLen) } });
    const header = JSON.parse(new TextDecoder().decode(await hResp.arrayBuffer()));
    const dataStart = 8 + headerLen;
    const tensors = Object.keys(header).filter(n => n !== '__metadata__')
      .map(n => ({ name: n, info: header[n], begin: header[n].data_offsets[0], end: header[n].data_offsets[1] }))
      .sort((a, b) => a.begin - b.begin);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('download failed: HTTP ' + resp.status);
    const total = +(resp.headers.get('content-length') || 0);
    const reader = resp.body.getReader();
    let chunks = [], winStart = 0, recv = 0, ti = 0;
    const takeRange = (begin, end) => {
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
        if (!skippedHeader) {
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

  async function loadModel({ variant = '0.8B', onProgress } = {}) {
    if (_loaded && _variant === variant) return;
    const m = MODELS[variant]; if (!m) throw new Error('unknown Qwen3.5 variant ' + variant);
    // Free the previous variant's buffers first (lfm25 lesson: orphaned buffers → GPU OOM).
    if (_loaded || _weights) {
      try { await E.device().queue.onSubmittedWorkDone(); } catch (_) {}
      unload();
    }
    _variant = variant; _cfg = m.cfg; _weights = {};
    await E.init();
    await TOK.load(m.root);
    onProgress && onProgress({ phase: 'tokenizer', pct: 100 });
    let hit = false;
    try { hit = await _readQuantCache(variant, onProgress); } catch (e) { try { console.warn('[gdn] cache read failed — falling back to download', e); } catch (_) {} _weights = {}; hit = false; }
    if (!hit) {
      _weights = {};
      const sink = _makeSink(variant);
      const sinkOk = await sink.open();
      await _streamWeights(m.root + m.file, onProgress, sinkOk ? sink : null);
      if (sinkOk) { const committed = await sink.finish(); try { console.log('[gdn] quant cache ' + (committed ? 'written' : 'NOT written (quota?)')); } catch (_) {} }
    }
    _loaded = true;
  }
  function unload() {
    try { if (_weights) for (const k in _weights) { const w = _weights[k]; for (const p of ['buf', 'pack', 'scales']) if (w[p] && w[p].destroy) try { w[p].destroy(); } catch (_) {} } } catch (_) {}
    _weights = null; _loaded = false; _variant = null;
  }

  return {
    CONFIG, GDN_CONV_DIM, MODELS,
    gdnConv, gdnGates, gdnQkNorm, gdnDelta, gatedRmsNorm,
    loadModel, unload, TOK,
    _selfTest: selfTestKernels,
    _weightsInfo: () => { const o = {}; let bytes = 0; for (const k in (_weights || {})) { const w = _weights[k]; o[k] = { int4: !!w.int4, f32: !!w.f32, shape: w.shape }; } return { count: Object.keys(o).length, tensors: o }; },
  };
})();

if (typeof self !== 'undefined') self.SandpieQwen35 = SandpieQwen35;
if (typeof window !== 'undefined') window.SandpieQwen35 = SandpieQwen35;
