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
    nHeads: 8, nKvHeads: 2, headDim: 256, rotDim: 64, ropeTheta: 10000000, rmsEps: 1e-6,
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
@group(0) @binding(0) var<storage, read>       b       : array<f32>;   // [T, V] (in_proj_b output)
@group(0) @binding(1) var<storage, read>       a       : array<f32>;   // [T, V] (in_proj_a output)
@group(0) @binding(2) var<storage, read>       alog    : array<f32>;   // [V]
@group(0) @binding(3) var<storage, read>       dtbias  : array<f32>;   // [V]
@group(0) @binding(4) var<storage, read_write> beta    : array<f32>;   // [T, V]
@group(0) @binding(5) var<storage, read_write> gexp    : array<f32>;   // [T, V]
@group(0) @binding(6) var<uniform>             p       : P;
fn softplus(x:f32) -> f32 { if (x > 20.0) { return x; } return log(1.0 + exp(x)); }
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x; let V = p.V;
  if (i >= p.T*V) { return; }
  let h = i % V;
  beta[i] = 1.0 / (1.0 + exp(-b[i]));
  gexp[i] = exp(-exp(alog[h]) * softplus(a[i] + dtbias[h]));
}`;
  function gdnGates(bBuf, aBuf, alogBuf, dtbiasBuf, betaBuf, gexpBuf, T, V) {
    const p = uniform(new Uint32Array([T, V, 0, 0]));
    const pipe = E.getPipeline('gdn.gates', GATES_WGSL);
    return E.dispatch(pipe, [bBuf, aBuf, alogBuf, dtbiasBuf, betaBuf, gexpBuf, p], [Math.ceil((T * V) / 64), 1, 1]);
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
@group(0) @binding(0) var<storage, read>       qkv   : array<f32>;   // [T, 3*H*D] normed q|k|v
@group(0) @binding(1) var<storage, read>       beta  : array<f32>;   // [T, H]
@group(0) @binding(2) var<storage, read>       gexp  : array<f32>;   // [T, H]
@group(0) @binding(3) var<storage, read_write> state : array<f32>;   // [H, D, D] (i-major) — SINGLE
   // buffer, in-place: each element is read once at entry and written once at exit by
   // exactly ONE thread (no cross-thread state access), so no double-buffer is needed —
   // and a separate read binding of the same buffer would be illegal binding aliasing.
@group(0) @binding(4) var<storage, read_write> o     : array<f32>;   // [T, H*D]
@group(0) @binding(5) var<uniform>             p     : P;
var<workgroup> red : array<f32, 128>;
var<workgroup> kvm : f32;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let D = p.D; let H = p.H; let C = 3u*H*D;
  let h = wg.x / D; let j = wg.x % D;
  let i = lid.x;
  var s = 0.0;
  if (p.useState != 0u) { s = state[(h*D + i)*D + j]; }
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
  state[(h*D + i)*D + j] = s;
}`;
  function gdnDelta(qkvBuf, betaBuf, gexpBuf, stateBuf, oBuf, T, H, D, useState) {
    const p = uniform(new Uint32Array([T, H, D, useState ? 1 : 0]));
    const pipe = E.getPipeline('gdn.delta', DELTA_WGSL);
    return E.dispatch(pipe, [qkvBuf, betaBuf, gexpBuf, stateBuf, oBuf, p], [H * D, 1, 1]);
  }

  // ============================================================
  // Kernel 5 — gated RMSNorm (Qwen3NextRMSNormGated — "Norm before gate"):
  //   out[t,h,:] = (o / rms(o)) · weight · silu(z)     per (t, head) over dv
  // weight is [dv] (shared across heads). z laid out [T, H*dv] (loader-permuted).
  // NOTE: norm FIRST, then gate — this is NOT the Mamba2 order.
  // ============================================================
  const GNORM_WGSL = `
struct P { T:u32, H:u32, D:u32, eps:f32 };
@group(0) @binding(0) var<storage, read>       o   : array<f32>;   // [T, H*D]
@group(0) @binding(1) var<storage, read>       z   : array<f32>;   // [T, H*D]
@group(0) @binding(2) var<storage, read>       w   : array<f32>;   // [D] (ships F32 in the checkpoint — kept exact)
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
  out[base + i] = v * inv * w[i] * (zg / (1.0 + exp(-zg)));
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
      const bIn = new Float32Array(T * V), aIn = new Float32Array(T * V);
      for (let i = 0; i < T * V; i++) { bIn[i] = Math.sin(i * 0.31) * 2; aIn[i] = Math.cos(i * 0.41) * 2; }
      const alog = new Float32Array(V); for (let i = 0; i < V; i++) alog[i] = Math.log(0.5 + i * 0.4);
      const dtb = new Float32Array(V); dtb.fill(1);
      const refB = new Float32Array(T * V), refG = new Float32Array(T * V);
      for (let i = 0; i < T * V; i++) {
        refB[i] = 1 / (1 + Math.exp(-bIn[i]));
        refG[i] = Math.exp(-Math.exp(alog[i % V]) * softplus(aIn[i] + dtb[i % V]));
      }
      const b0 = f32buf(bIn), a0 = f32buf(aIn), ab = f32buf(alog), db = f32buf(dtb), bb = E.createBuffer(T * V * 4, ST(), 'b'), gb = E.createBuffer(T * V * 4, ST(), 'g');
      await gdnGates(b0, a0, ab, db, bb, gb, T, V);
      check('gdn gates (beta)', maxAbs(await E.readF32(bb, T * V), refB), 1e-5);
      check('gdn gates (gexp)', maxAbs(await E.readF32(gb, T * V), refG), 1e-5);
      [b0, a0, ab, db, bb, gb].forEach(b => b.destroy());
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
      const sb = E.createBuffer(H * D * D * 4, ST(), 'ds'), ob = E.createBuffer(T * H * D * 4, ST(), 'do');
      await gdnDelta(qb, bb, gb, sb, ob, T, H, D, false);
      check('gdn delta (prefill o)', maxAbs(await E.readF32(ob, T * H * D), ref.o), 2e-3);
      check('gdn delta (prefill state)', maxAbs(await E.readF32(sb, H * D * D), ref.S), 2e-3);
      [qb, bb, gb, sb, ob].forEach(b => b.destroy());
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
      const sb = E.createBuffer(H * D * D * 4, ST(), 'ls');   // single in-place state, as the forward uses it
      const gotO = new Float32Array(T * H * D);
      for (let c = 0; c < T / CH; c++) {
        const qb = f32buf(qkv.slice(c * CH * rowC, (c + 1) * CH * rowC));
        const bb = f32buf(beta.slice(c * CH * H, (c + 1) * CH * H));
        const gb = f32buf(gexp.slice(c * CH * H, (c + 1) * CH * H));
        const ob = E.createBuffer(CH * H * D * 4, ST(), 'lo');
        await gdnDelta(qb, bb, gb, sb, ob, CH, H, D, c > 0);
        gotO.set(await E.readF32(ob, CH * H * D), c * CH * H * D);
        [qb, bb, gb, ob].forEach(b => b.destroy());
      }
      check('gdn delta (512-tok chunked continuity)', maxAbs(gotO, ref.o), 5e-3);
      sb.destroy();
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
      const ob = f32buf(o), zb = f32buf(z), wb = f32buf(w), rb = E.createBuffer(T * H * D * 4, ST(), 'gn');
      await gatedRmsNorm(ob, zb, wb, rb, T, H, D, eps);
      check('gdn gated rmsnorm (norm-before-gate)', maxAbs(await E.readF32(rb, T * H * D), ref), 1e-4);
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
  // reshaped [..., head, 2*hd] output). SPLIT into two separate weight tensors at
  // load — q rows must be contiguous [T, nH*hd] for ropeQK (a single permuted
  // tensor would give q a T-row stride of 2*nH*hd). Zero runtime cost.
  function _splitQGate(u16, rows, K, headDim) {
    const half = rows / 2;
    const q = new Uint16Array(half * K), g = new Uint16Array(half * K);
    const nH = rows / (2 * headDim);
    for (let h = 0; h < nH; h++) for (let d = 0; d < headDim; d++) {
      q.set(u16.subarray((h * 2 * headDim + d) * K, (h * 2 * headDim + d + 1) * K), (h * headDim + d) * K);
      g.set(u16.subarray((h * 2 * headDim + headDim + d) * K, (h * 2 * headDim + headDim + d + 1) * K), (h * headDim + d) * K);
    }
    return { q, g };
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
    if (/self_attn\.q_proj\.weight$/.test(name)) {
      // split fused [q|gate]-per-head projection into two int4 tensors
      const rows = info.shape[0], K = info.shape[1], half = rows / 2;
      const { q, g } = _splitQGate(u16, rows, K, _cfg.headDim);
      const store = async (nm, u) => {
        const { pack, scales } = _quantInt4(u, half, K);
        const packBuf = E.createBuffer(pack.byteLength, ST(), nm + '.pack');
        const scBuf = E.createBuffer(scales.byteLength, ST(), nm + '.sc');
        put(packBuf, pack); put(scBuf, scales);
        _weights[nm] = { pack: packBuf, scales: scBuf, N: half, K, int4: true, shape: [half, K] };
        if (sink) { await sink.add(nm, 'pack', pack, { kind: 'int4', shape: [half, K], N: half, K }); await sink.add(nm, 'scales', scales, { kind: 'int4', shape: [half, K], N: half, K }); }
      };
      await store(name, q);
      await store(name.replace('q_proj', 'out_gate'), g);
      return;
    }
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
      let bits = info.dtype === 'BF16' ? _bf16ToF16bits(u16) : u16;
      // ZERO-CENTERED RMSNorm (Qwen3_5RMSNorm): checkpoint stores weight-1 (zeros-init),
      // forward multiplies by (1.0 + weight). Bake the +1 in at load so the plain
      // rmsnorm/ropeQK kernels stay unchanged. Does NOT apply to linear_attn.norm
      // (Qwen3_5RMSNormGated is ones-init, plain multiply).
      if (/(input_layernorm|post_attention_layernorm|q_norm|k_norm|language_model\.norm)\.weight$/.test(name)) {
        const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
        const f = new Float32Array(numel);
        for (let i = 0; i < numel; i++) { ti[0] = u16[i] << 16; f[i] = t[0] + 1.0; }   // u16 is BF16 here
        bits = E.f32ToF16(f);
      }
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
  const QC_VER = 3;   // v3: zero-centered RMSNorm weights stored as (1+w); v2: q_proj split q/out_gate
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

  // ============================================================
  // PHASE C (prep) — forward primitives, ported from webgpu-lfm25.js.
  // matvecQ / gemvQ8 / DP4A GEMM / rmsnorm / embed / swiglu / axpy are VERBATIM
  // copies (battle-tested in lfm25+qwen3) with 'gdn.' pipeline keys — not re-tested
  // here. ROPEQK and ATTN are ADAPTED (headDim 256 + partial rotary) and get tests.
  // ============================================================
  const MATVEC_WG = 64, MATVEC_MAXT = 32;
  function wgReduceWGSL(CNT, WG) {
    return `
  for (var _r:u32=0u; _r<${CNT}u; _r=_r+1u) { part[_r*${WG}u + lid.x] = acc[_r]; }
  workgroupBarrier();
  for (var _st:u32=${WG >> 1}u; _st>0u; _st=_st>>1u) {
    if (lid.x < _st) { for (var _r:u32=0u; _r<${CNT}u; _r=_r+1u) { part[_r*${WG}u + lid.x] = part[_r*${WG}u + lid.x] + part[_r*${WG}u + lid.x + _st]; } }
    workgroupBarrier();
  }`;
  }
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
    const pipe = E.getPipeline('gdn.matvecQ', MATVECQ_WGSL);
    const gx = Math.min(N, 65535), gy = Math.ceil(N / gx);
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, yBuf, d], [gx, gy, 1]);
  }
  let _gemvNR = 4, _gemvWG = 32;
  const _NR = () => Math.max(1, (globalThis.__gemvNR | 0) || _gemvNR);
  const _WG = () => Math.max(32, (globalThis.__gemvWG | 0) || _gemvWG);
  function gemvQ8Wgsl(NR, GEMV_WG) { return `
enable f16;
struct D { N:u32, K:u32, acc:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>) {
  let nBase = (wg.x + wg.y*nwg.x) * ${NR}u;
  let words = d.K / 8u; let gpr = d.K / ${QGROUP}u;
  var acc : array<f32, ${NR}>;
  for (var r:u32=0u; r<${NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = x[2u*w]; let xc = x[2u*w + 1u];
    let grp = (w*8u)/${QGROUP}u;
    for (var r:u32=0u; r<${NR}u; r=r+1u) {
      let n = nBase + r; if (n >= d.N) { continue; }
      let p = W[n*words + w];
      let s = f32(sc[n*gpr + grp]);
      let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      acc[r] = acc[r] + s*( dot(vec4<f32>(lo.x,hi.x,lo.y,hi.y), xa) + dot(vec4<f32>(lo.z,hi.z,lo.w,hi.w), xc) );
    }
    w = w + ${GEMV_WG}u;
  }
${wgReduceWGSL(NR, GEMV_WG)}
  if (lid.x < ${NR}u) {
    let n = nBase + lid.x;
    if (n < d.N) { y[n] = select(0.0, y[n], d.acc != 0u) + part[lid.x*${GEMV_WG}u + 0u]; }
  }
}`; }
  function gemvQ8(xBuf, rec, yBuf, N, K, acc) {
    const NR = _NR(), WG = _WG();
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const pipe = E.getPipeline('gdn.gemvQ8.' + NR + '.' + WG, gemvQ8Wgsl(NR, WG));
    const nWG = Math.ceil(N / NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, yBuf, d], [gx, gy, 1]);
  }

  // ---- DP4A int8 GEMM for T>1 prefill (qwen3 d8db4ed recipe, via lfm25) ----
  const GEMMQ_BM = 64, GEMMQ_BN = 64, GEMMQ_TM = 4, GEMMQ_TN = 4;
  const QUANTQ8T_WGSL = `
struct Q { K:u32, gpr:u32, ng:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<f32>;
@group(0) @binding(1) var<storage, read_write> xq : array<u32>;
@group(0) @binding(2) var<storage, read_write> xs : array<f32>;
@group(0) @binding(3) var<uniform>             q  : Q;
var<workgroup> msh : array<f32, ${QGROUP}>;
var<workgroup> qsh : array<i32, ${QGROUP}>;
@compute @workgroup_size(${QGROUP},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let gg = wg.x + wg.y*nwg.x; if (gg >= q.ng) { return; }
  let t = gg / q.gpr; let g = gg % q.gpr;
  let lid = lidv.x; let val = x[t*q.K + g*${QGROUP}u + lid];
  msh[lid] = abs(val); workgroupBarrier();
  var st = ${QGROUP / 2}u; loop { if(st==0u){break;} if(lid<st){ msh[lid]=max(msh[lid],msh[lid+st]); } workgroupBarrier(); st=st/2u; }
  let scale = msh[0] / 127.0; let inv = select(0.0, 1.0/scale, scale > 0.0);
  if (lid==0u) { xs[gg] = scale; }
  qsh[lid] = clamp(i32(round(val*inv)), -127, 127); workgroupBarrier();
  if (lid < ${QGROUP / 4}u) {
    xq[gg*${QGROUP / 4}u + lid] = pack4xI8(vec4<i32>(qsh[lid*4u], qsh[lid*4u+1u], qsh[lid*4u+2u], qsh[lid*4u+3u]));
  }
}`;
  function gemmdp4Wgsl() {
    const BM = GEMMQ_BM, BN = GEMMQ_BN, BK = QGROUP, TM = GEMMQ_TM, TN = GEMMQ_TN, BK4 = BK / 4;
    const NTH = (BM / TM) * (BN / TN), RN = BN / TN, TILEA = BM * BK4, TILEB = BN * BK4;
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read>       xs : array<f32>;
@group(0) @binding(4) var<storage, read_write> Y  : array<f32>;
@group(0) @binding(5) var<uniform>             d  : D;
var<workgroup> As : array<u32, ${TILEA}>;
var<workgroup> Bs : array<u32, ${TILEB}>;
@compute @workgroup_size(${NTH}, 1, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let lx = lid.x; let tN = lx % ${RN}u; let tM = lx / ${RN}u;
  let mBase = wg.y*${BM}u; let nBase = wg.x*${BN}u;
  let WPR = d.K/8u; let gpr = d.K/${QGROUP}u; let K4 = d.K/4u; let nTiles = d.K/${BK}u;
`;
    for (let r = 0; r < TM * TN; r++) s += `  var f${r}:f32=0.0;\n`;
    s += `  for (var kt:u32=0u; kt<nTiles; kt=kt+1u) {
    let k0 = kt*${BK}u;
    for (var r:u32=0u; r<${TILEA / NTH}u; r=r+1u) {
      let idx = lx + r*${NTH}u; let lt = idx/${BK4}u; let kk4 = idx%${BK4}u; let gt = mBase+lt;
      As[idx] = select(0u, xq[gt*K4 + k0/4u + kk4], gt<d.T);
    }
    for (var r:u32=0u; r<${TILEB / NTH}u; r=r+1u) {
      let idx = lx + r*${NTH}u; let ln = idx/${BK4}u; let kk4 = idx%${BK4}u; let gn = nBase+ln;
      var packed = 0u;
      if (gn < d.N) {
        let kk = k0 + kk4*4u; let word = W[gn*WPR + (kk>>3u)]; let b = (kk & 7u);
        let n0 = i32((word >> (4u*(b+0u))) & 0xFu) - 8;
        let n1 = i32((word >> (4u*(b+1u))) & 0xFu) - 8;
        let n2 = i32((word >> (4u*(b+2u))) & 0xFu) - 8;
        let n3 = i32((word >> (4u*(b+3u))) & 0xFu) - 8;
        packed = pack4xI8(vec4<i32>(n0,n1,n2,n3));
      }
      Bs[idx] = packed;
    }
    workgroupBarrier();
`;
    for (let r = 0; r < TM * TN; r++) s += `    var i${r}:i32=0;\n`;
    s += `    for (var cc:u32=0u; cc<${BK4}u; cc=cc+1u) {
`;
    for (let i = 0; i < TM; i++) s += `      let a${i} = As[(tM*${TM}u+${i}u)*${BK4}u + cc];\n`;
    for (let j = 0; j < TN; j++) s += `      let b${j} = Bs[(tN*${TN}u+${j}u)*${BK4}u + cc];\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `      i${i * TN + j} = i${i * TN + j} + dot4I8Packed(a${i}, b${j});\n`;
    s += `    }
`;
    for (let i = 0; i < TM; i++) s += `    let xs${i} = select(0.0, xs[(mBase+tM*${TM}u+${i}u)*gpr + kt], (mBase+tM*${TM}u+${i}u)<d.T);\n`;
    for (let j = 0; j < TN; j++) s += `    let ws${j} = select(0.0, f32(sc[(nBase+tN*${TN}u+${j}u)*gpr + kt]), (nBase+tN*${TN}u+${j}u)<d.N);\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `    f${i * TN + j} = f${i * TN + j} + f32(i${i * TN + j}) * xs${i} * ws${j};\n`;
    s += `    workgroupBarrier();
  }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  { let gm=mBase+tM*${TM}u+${i}u; let gn=nBase+tN*${TN}u+${j}u; if (gm<d.T && gn<d.N) { let idx=gm*d.N+gn; Y[idx]=select(0.0,Y[idx],d.acc!=0u)+f${i * TN + j}; } }\n`;
    s += `}`;
    return s;
  }
  let _dp4g = null;
  function ensureDp4G(T, K) {
    const need = T * K;
    if (_dp4g && _dp4g.cap >= need) return;
    if (_dp4g) { try { _dp4g.xq.destroy(); _dp4g.xs.destroy(); } catch (_) {} }
    _dp4g = { cap: need, xq: E.createBuffer(need, ST(), 'gxq'), xs: E.createBuffer((need / QGROUP) * 4, ST(), 'gxs') };
  }
  function gemmDP4A(xBuf, wrec, yBuf, T, N, K, acc) {
    // K floored to 4096 so scratch is constant-size across a forward (K spans 1024..3584 here).
    ensureDp4G(Math.max(T, MATVEC_MAXT), Math.max(K, 4096));
    const gpr = K / QGROUP, ng = T * gpr;
    const qp = E.getPipeline('gdn.quantq8t', QUANTQ8T_WGSL);
    const qd = uniform(new Uint32Array([K, gpr, ng, 0]));
    const qgx = Math.min(ng, 65535), qgy = Math.ceil(ng / qgx);
    E.dispatch(qp, [xBuf, _dp4g.xq, _dp4g.xs, qd], [qgx, qgy, 1]);
    const pipe = E.getPipeline('gdn.gemmDP4', gemmdp4Wgsl());
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    return E.dispatch(pipe, [_dp4g.xq, wrec.pack, wrec.scales, _dp4g.xs, yBuf, d], [Math.ceil(N / GEMMQ_BN), Math.ceil(T / GEMMQ_BM), 1]);
  }
  // T=1 → 4-row f32-dequant GEMV; T>1 → DP4A GEMM (or batched matvecQ fallback).
  function mv(xBuf, rec, yBuf, T, N, K, acc) {
    if (T !== 1) return (globalThis.__noDp4Gemm ? matvecQ : gemmDP4A)(xBuf, rec, yBuf, T, N, K, acc);
    return gemvQ8(xBuf, rec, yBuf, N, K, acc);
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
    return E.dispatch(E.getPipeline('gdn.rmsnorm', RMSNORM_WGSL), [xBuf, wBuf, yBuf, uniform(u)], [T, 1, 1]);
  }

  const EMBED_WGSL = `
enable f16;
struct P { T:u32, H:u32, idOff:u32, _b:u32 };
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
    return E.dispatch(E.getPipeline('gdn.embed', EMBED_WGSL), [idsBuf, embBuf, yBuf, p], [gx, gy, 1]);
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
    return E.dispatch(E.getPipeline('gdn.swiglu', SWIGLU_WGSL), [gBuf, uBuf, yBuf, p], [gx, gy, 1]);
  }

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
    return E.dispatch(E.getPipeline('gdn.axpy', AXPY_WGSL), [xBuf, vBuf, uniform(u)], [gx, gy, 1]);
  }

  // ---- ADAPTED: fused per-head RMSNorm + PARTIAL RoPE (rotary factor 0.25 →
  // rope only the first rotDim=64 of hd=256 dims; norm covers ALL hd dims).
  // One workgroup per (t, head); 256 threads (hd ≤ 256). mrope collapses to
  // standard RoPE for text-only (all position ids equal).
  const ROPEQK_WGSL = `
enable f16;
struct P { T:u32, nH:u32, hd:u32, posBase:u32, theta:f32, eps:f32, rotDim:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read>       normW: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
var<workgroup> red : array<f32, 256>;
var<workgroup> nrm : array<f32, 256>;
@compute @workgroup_size(256,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let hd = p.hd; let j = lid.x;
  let unit = wg.x; let t = unit/p.nH; let head = unit%p.nH;
  let base = t*(p.nH*hd) + head*hd;
  var v:f32 = 0.0; if (j < hd) { v = inp[base+j]; }
  red[j] = select(0.0, v*v, j < hd); workgroupBarrier();
  var stride = 128u;
  loop { if (stride==0u) { break; } if (j < stride) { red[j] = red[j] + red[j+stride]; } workgroupBarrier(); stride = stride/2u; }
  let inv = inverseSqrt(red[0]/f32(hd) + p.eps);
  if (j < hd) { nrm[j] = v*inv*f32(normW[j]); }
  workgroupBarrier();
  if (j >= hd) { return; }
  if (j >= p.rotDim) { outp[base+j] = nrm[j]; return; }   // pass-through dims
  let half = p.rotDim/2u;
  let pos = f32(p.posBase + t);
  let freqIdx = select(j-half, j, j<half);
  let ang = pos * pow(p.theta, -2.0*f32(freqIdx)/f32(p.rotDim));
  let c = cos(ang); let s = sin(ang);
  let partner = select(nrm[j-half], nrm[j+half], j<half);
  let rot = select(partner, -partner, j<half);
  outp[base+j] = nrm[j]*c + rot*s;
}`;
  function ropeQK(inBuf, normWBuf, outBuf, T, nH, hd, posBase, theta, eps, rotDim) {
    const u = new Uint32Array(8); const dv = new DataView(u.buffer);
    dv.setUint32(0, T, true); dv.setUint32(4, nH, true); dv.setUint32(8, hd, true);
    dv.setUint32(12, posBase, true); dv.setFloat32(16, theta, true); dv.setFloat32(20, eps, true);
    dv.setUint32(24, rotDim || hd, true);
    return E.dispatch(E.getPipeline('gdn.ropeqk', ROPEQK_WGSL), [inBuf, normWBuf, outBuf, uniform(u)], [T * nH, 1, 1]);
  }

  // ---- ADAPTED: f32 tiled flash attention at hd=256. Tiles QT=8/KT=4/HD4=64 keep
  // workgroup memory at ~24.2KB (lfm25's 16/8/32 tiling would need 48KB at hd=256).
  // Score threads guarded (QT*KT=32 < WG=128 — lfm25 had exactly WG pairs).
  const AQT = 8, AKT = 4, AHD4 = 64, AWG = 128;
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
    if (tid < QT*KT) {
      let qi = tid / KT; let kj = tid % KT;
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
    const pipe = E.getPipeline('gdn.attn', ATTN_WGSL);
    const blocks = nHq * Math.ceil(T / AQT);
    const gx = Math.min(blocks, 65535), gy = Math.ceil(blocks / gx);
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
  }

  // ---- NEW: sigmoid output gate (gated attention): x[i] *= sigmoid(g[i]).
  // Applied to the attention output before o_proj (Qwen3NextAttention).
  const MULSIG_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read_write> x : array<f32>;
@group(0) @binding(1) var<storage, read>       g : array<f32>;
@group(0) @binding(2) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i = gid.y*(nwg.x*64u)+gid.x; if (i >= p.n) { return; }
  x[i] = x[i] / (1.0 + exp(-g[i]));
}`;
  function mulSigmoid(xBuf, gBuf, n) {
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    const c = Math.ceil(n / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('gdn.mulsig', MULSIG_WGSL), [xBuf, gBuf, p], [gx, gy, 1]);
  }

  // Self-tests for the ADAPTED kernels (ropeQK partial + attention hd=256 + mulSigmoid).
  async function selfTestFwdPrims() {
    await E.init();
    const out = [];
    const check = (name, err, tol) => out.push({ name, ok: err < (tol || 1e-3), err });

    // ropeQK: per-head norm over 256, rope on first 64 only
    {
      const T = 3, nH = 2, hd = 256, rotDim = 64, theta = 1e7, eps = 1e-6, posBase = 5;
      const x = new Float32Array(T * nH * hd); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.037) * 2;
      const w = new Float32Array(hd); for (let i = 0; i < hd; i++) w[i] = 0.9 + 0.2 * Math.cos(i * 0.11);
      const wf16r = (() => { const o = new Float32Array(hd); const bits = E.f32ToF16(w); for (let i = 0; i < hd; i++) o[i] = _f16ToF32s(bits[i]); return o; })();
      const ref = new Float32Array(T * nH * hd);
      for (let t = 0; t < T; t++) for (let h = 0; h < nH; h++) {
        const base = (t * nH + h) * hd;
        let ss = 0; for (let j = 0; j < hd; j++) ss += x[base + j] * x[base + j];
        const inv = 1 / Math.sqrt(ss / hd + eps);
        const nv = new Float32Array(hd); for (let j = 0; j < hd; j++) nv[j] = x[base + j] * inv * wf16r[j];
        const half = rotDim / 2, pos = posBase + t;
        for (let j = 0; j < hd; j++) {
          if (j >= rotDim) { ref[base + j] = nv[j]; continue; }
          const fi = j < half ? j : j - half;
          const ang = pos * Math.pow(theta, -2 * fi / rotDim);
          const c = Math.cos(ang), s = Math.sin(ang);
          const rot = j < half ? -nv[j + half] : nv[j - half];
          ref[base + j] = nv[j] * c + rot * s;
        }
      }
      const xb = f32buf(x), wb = f16buf(w), ob = E.createBuffer(x.byteLength, ST(), 'ro');
      await ropeQK(xb, wb, ob, T, nH, hd, posBase, theta, eps, rotDim);
      check('gdn ropeQK (norm256 + partial rope64)', maxAbs(await E.readF32(ob, x.length), ref), 2e-3);
      [xb, wb, ob].forEach(b => b.destroy());
    }

    // attention hd=256, GQA 8q/2kv, causal, vs CPU ref
    {
      const T = 5, S = 9, nHq = 4, nKv = 2, hd = 256;
      const q = new Float32Array(T * nHq * hd), k = new Float32Array(S * nKv * hd), v = new Float32Array(S * nKv * hd);
      for (let i = 0; i < q.length; i++) q[i] = Math.sin(i * 0.0173) * 0.5;
      for (let i = 0; i < k.length; i++) k[i] = Math.cos(i * 0.0119) * 0.5;
      for (let i = 0; i < v.length; i++) v[i] = Math.sin(i * 0.0231 + 1) * 0.5;
      const ref = new Float32Array(T * nHq * hd);
      const grp = nHq / nKv, scale = 1 / Math.sqrt(hd);
      for (let t = 0; t < T; t++) for (let h = 0; h < nHq; h++) {
        const hk = Math.floor(h / grp), qpos = (S - T) + t;
        const sc = [];
        for (let s = 0; s <= qpos; s++) {
          let d = 0; for (let j = 0; j < hd; j++) d += q[(t * nHq + h) * hd + j] * k[(s * nKv + hk) * hd + j];
          sc.push(d * scale);
        }
        const mx = Math.max(...sc); let sum = 0;
        for (let s = 0; s < sc.length; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
        for (let j = 0; j < hd; j++) {
          let o = 0; for (let s = 0; s < sc.length; s++) o += sc[s] * v[(s * nKv + hk) * hd + j];
          ref[(t * nHq + h) * hd + j] = o / sum;
        }
      }
      const qb = f32buf(q), kb = f32buf(k), vb = f32buf(v), ob = E.createBuffer(T * nHq * hd * 4, ST(), 'ao');
      await attention(qb, kb, vb, ob, T, S, nHq, nKv, hd);
      check('gdn attention (hd256 GQA causal)', maxAbs(await E.readF32(ob, T * nHq * hd), ref), 2e-3);
      [qb, kb, vb, ob].forEach(b => b.destroy());
    }

    // mulSigmoid
    {
      const n = 300;
      const x = new Float32Array(n), g = new Float32Array(n);
      for (let i = 0; i < n; i++) { x[i] = Math.sin(i) * 2; g[i] = Math.cos(i * 0.7) * 3; }
      const ref = new Float32Array(n); for (let i = 0; i < n; i++) ref[i] = x[i] / (1 + Math.exp(-g[i]));
      const xb = f32buf(x), gb = f32buf(g);
      await mulSigmoid(xb, gb, n);
      check('gdn mulSigmoid (attn output gate)', maxAbs(await E.readF32(xb, n), ref), 1e-5);
      [xb, gb].forEach(b => b.destroy());
    }

    const fails = out.filter(r => !r.ok);
    console.table(out.map(r => ({ test: r.name, ok: r.ok ? '✓' : '✗ FAIL', maxErr: r.err.toExponential(2) })));
    return { ok: fails.length === 0, results: out };
  }

  // ============================================================
  // PHASE C — forward graph + generate (lfm25 structure).
  // ============================================================
  // Tiny f16-weight matvec for in_proj_b / in_proj_a ([16,1024] — kept f16 for
  // exact decay/beta gates; the int4 GEMVs only serve quantized recs).
  const MVF16_WGSL = `
enable f16;
struct D { T:u32, N:u32, K:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       W : array<f16>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> red : array<f32, 64>;
@compute @workgroup_size(64,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let t = wg.x / d.N; let n = wg.x % d.N;
  var acc = 0.0;
  var k = lid.x;
  loop { if (k >= d.K) { break; } acc = acc + f32(W[n*d.K + k]) * x[t*d.K + k]; k = k + 64u; }
  red[lid.x] = acc; workgroupBarrier();
  var s = 32u; loop { if (s==0u) { break; } if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x+s]; } workgroupBarrier(); s = s/2u; }
  if (lid.x == 0u) { y[t*d.N + n] = red[0]; }
}`;
  function mvF16(xBuf, wBuf, yBuf, T, N, K) {
    const d = uniform(new Uint32Array([T, N, K, 0]));
    return E.dispatch(E.getPipeline('gdn.mvf16', MVF16_WGSL), [xBuf, wBuf, yBuf, d], [T * N, 1, 1]);
  }

  // GPU argmax logits → tokHist[outPos] (lfm25 two-pass, verbatim).
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
    E.dispatch(E.getPipeline('gdn.amaxP1', AMAX_P1_WGSL), [logitsBuf, _amaxV, _amaxI, p], [AMAX_WGS, 1, 1]);
    return E.dispatch(E.getPipeline('gdn.amaxP2', AMAX_P2_WGSL), [_amaxV, _amaxI, tokHistBuf, p], [1, 1, 1]);
  }

  // ---- forward state ----------------------------------------------------
  // KV only on the 6 attention layers; GDN layers carry (a) a conv tail
  // [convDim, L-1] ×2 (double-buffered — carry readers/writer are in different
  // workgroups) and (b) the delta state [16,128,128] f32, SINGLE-buffered:
  // gdnDelta reads each element into a register once at entry and writes it
  // once at exit, and each element is owned by exactly one thread → in-place
  // is race-free (double-buffering would cost 8MB/layer for nothing).
  let MAX_SEQ = 8192;
  let PREFILL_T = 128;   // prefill chunk (DP4A GEMM path); GDN recurrence loops T inside its kernel
  let _scr = null, _kv = null, _gdnState = null, _idsBuf = null;
  function _ensureState() {
    if (_scr) return;
    const C = _cfg, H = C.hidden, T = Math.max(MATVEC_MAXT, PREFILL_T);
    const G = C.gdn, CD = GDN_CONV_DIM, HV = G.nHeads * G.dv;   // 2048
    const mk = (n, l) => E.createBuffer(n * 4, ST(), l);
    _scr = {
      x: mk(T * H, 'x'), normed: mk(T * H, 'nrm'),
      qkv: mk(T * CD, 'qkv'), qkvc: mk(T * CD, 'qkvc'),
      bproj: mk(T * G.nHeads, 'bp'), aproj: mk(T * G.nHeads, 'ap'),
      beta: mk(T * G.nHeads, 'be'), gexp: mk(T * G.nHeads, 'ge'),
      gdnO: mk(T * HV, 'go'), z: mk(T * HV, 'z'), gnorm: mk(T * HV, 'gn'),
      q: mk(T * C.nHeads * C.headDim, 'q'), k: mk(T * C.nKvHeads * C.headDim, 'k'), v: mk(T * C.nKvHeads * C.headDim, 'v'),
      qr: mk(T * C.nHeads * C.headDim, 'qr'), kr: mk(T * C.nKvHeads * C.headDim, 'kr'),
      attn: mk(T * C.nHeads * C.headDim, 'at'), agate: mk(T * C.nHeads * C.headDim, 'ag'),
      gate: mk(T * C.intermediate, 'g'), up: mk(T * C.intermediate, 'u'), swi: mk(T * C.intermediate, 's'),
      last: mk(H, 'last'), logits: mk(C.vocab, 'lg'),
      tokHist: E.createBuffer(MAX_SEQ * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'th'),
    };
    _kv = {}; _gdnState = {};
    for (let l = 0; l < C.numLayers; l++) {
      if (C.attnLayers.includes(l)) _kv[l] = { k: mk(MAX_SEQ * C.nKvHeads * C.headDim, 'k' + l), v: mk(MAX_SEQ * C.nKvHeads * C.headDim, 'v' + l) };
      else _gdnState[l] = {
        convA: mk(CD * (G.convL - 1), 'ca' + l), convB: mk(CD * (G.convL - 1), 'cb' + l), cur: 0,
        S: mk(G.nHeads * G.dk * G.dv, 'S' + l),
      };
    }
  }
  function _freeState() {
    const kill = (o) => { for (const k in o) { const v = o[k]; if (v && v.destroy) try { v.destroy(); } catch (_) {} else if (v && typeof v === 'object') kill(v); } };
    if (_scr) kill(_scr); if (_kv) kill(_kv); if (_gdnState) kill(_gdnState);
    if (_idsBuf) try { _idsBuf.destroy(); } catch (_) {}
    try { if (_amaxV) { _amaxV.destroy(); _amaxI.destroy(); } } catch (_) {}
    try { if (_dp4g) { _dp4g.xq.destroy(); _dp4g.xs.destroy(); } } catch (_) {}
    _scr = null; _kv = null; _gdnState = null; _idsBuf = null; _amaxV = null; _amaxI = null; _dp4g = null;
  }

  // One forward over T tokens at absolute positions [posBase, posBase+T).
  // opts.chain: embed reads tokHist[posBase] (GPU-resident decode chaining);
  // opts.argmax: GPU-argmax final logits → tokHist[posBase+T]; opts.batched:
  // caller owns beginBatch/endBatch.
  async function forward(idsArr, posBase, opts) {
    opts = opts || {};
    const C = _cfg, H = C.hidden, T = opts.chain ? 1 : idsArr.length, S = posBase + T;
    const G = C.gdn, CD = GDN_CONV_DIM, HV = G.nHeads * G.dv;
    const W = (n) => _weights[n];
    _ensureState();
    const s = _scr;
    if (!opts.chain) {
      if (!_idsBuf) _idsBuf = E.createBuffer(Math.max(MATVEC_MAXT, PREFILL_T) * 4, U.STORAGE | U.COPY_DST, 'ids');
      E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(idsArr));
    }
    uniformReset();
    // (The earlier "flush>4 corrupts" finding was a red herring: a stale 7-arg
    // gdnDelta call passed a GPUBuffer as T→0, freezing s.gdnO at whatever the
    // last diagnostic probe left in it — run-order-dependent, mimicking a batching
    // race. Default flush restored; __gdnFlush kept as a debug override.)
    if (!opts.batched) E.beginBatch((globalThis.__gdnFlush | 0) || undefined);
    const P = (l) => 'model.language_model.layers.' + l + '.';
    if (opts.chain) await embedGather(s.tokHist, W(EMBED).buf, s.x, T, H, posBase);
    else await embedGather(_idsBuf, W(EMBED).buf, s.x, T, H, 0);
    const _stopL = opts.stopLayer !== undefined ? opts.stopLayer : C.numLayers;
    for (let l = 0; l < _stopL; l++) {
      const p = P(l);
      await rmsnorm(s.x, W(p + 'input_layernorm.weight').buf, s.normed, T, H, C.rmsEps);
      if (_gdnState[l]) {   // Gated DeltaNet block
        const st = _gdnState[l];
        await mv(s.normed, W(p + 'linear_attn.in_proj_qkv.weight'), s.qkv, T, CD, H);
        const cIn = st.cur === 0 ? st.convA : st.convB, cOut = st.cur === 0 ? st.convB : st.convA;
        await gdnConv(s.qkv, W(p + 'linear_attn.conv1d.weight').buf, cIn, cOut, s.qkvc, T, CD, G.convL, posBase > 0);
        st.cur ^= 1;
        await gdnQkNorm(s.qkvc, T, G.nHeads, G.dk, CD);
        await mvF16(s.normed, W(p + 'linear_attn.in_proj_b.weight').buf, s.bproj, T, G.nHeads, H);
        await mvF16(s.normed, W(p + 'linear_attn.in_proj_a.weight').buf, s.aproj, T, G.nHeads, H);
        await gdnGates(s.bproj, s.aproj, W(p + 'linear_attn.A_log').buf, W(p + 'linear_attn.dt_bias').buf, s.beta, s.gexp, T, G.nHeads);
        await gdnDelta(s.qkvc, s.beta, s.gexp, st.S, s.gdnO, T, G.nHeads, G.dk, posBase > 0);
        await mv(s.normed, W(p + 'linear_attn.in_proj_z.weight'), s.z, T, HV, H);
        await gatedRmsNorm(s.gdnO, s.z, W(p + 'linear_attn.norm.weight').buf, s.gnorm, T, G.nHeads, G.dv, C.rmsEps);
        await mv(s.gnorm, W(p + 'linear_attn.out_proj.weight'), s.x, T, H, HV, true);   // + residual
      } else {              // gated GQA attention
        const nHq = C.nHeads, nKv = C.nKvHeads, hd = C.headDim;
        await mv(s.normed, W(p + 'self_attn.q_proj.weight'), s.q, T, nHq * hd, H);
        await mv(s.normed, W(p + 'self_attn.out_gate.weight'), s.agate, T, nHq * hd, H);
        await mv(s.normed, W(p + 'self_attn.k_proj.weight'), s.k, T, nKv * hd, H);
        await mv(s.normed, W(p + 'self_attn.v_proj.weight'), s.v, T, nKv * hd, H);
        await ropeQK(s.q, W(p + 'self_attn.q_norm.weight').buf, s.qr, T, nHq, hd, posBase, C.ropeTheta, C.rmsEps, C.rotDim);
        await ropeQK(s.k, W(p + 'self_attn.k_norm.weight').buf, s.kr, T, nKv, hd, posBase, C.ropeTheta, C.rmsEps, C.rotDim);
        E.copyBuffer(s.kr, 0, _kv[l].k, posBase * nKv * hd * 4, T * nKv * hd * 4);
        E.copyBuffer(s.v, 0, _kv[l].v, posBase * nKv * hd * 4, T * nKv * hd * 4);
        await attention(s.qr, _kv[l].k, _kv[l].v, s.attn, T, S, nHq, nKv, hd);
        await mulSigmoid(s.attn, s.agate, T * nHq * hd);
        await mv(s.attn, W(p + 'self_attn.o_proj.weight'), s.x, T, H, nHq * hd, true);   // + residual
      }
      await rmsnorm(s.x, W(p + 'post_attention_layernorm.weight').buf, s.normed, T, H, C.rmsEps);
      const I = C.intermediate;
      await mv(s.normed, W(p + 'mlp.gate_proj.weight'), s.gate, T, I, H);
      await mv(s.normed, W(p + 'mlp.up_proj.weight'), s.up, T, I, H);
      await swiglu(s.gate, s.up, s.swi, T * I);
      await mv(s.swi, W(p + 'mlp.down_proj.weight'), s.x, T, H, I, true);                // + residual
      if (opts.syncLayers) { await E.device().queue.onSubmittedWorkDone(); }             // debug knob
    }
    await rmsnorm(s.x, W('model.language_model.norm.weight').buf, s.normed, T, H, C.rmsEps);
    E.copyBuffer(s.normed, (T - 1) * H * 4, s.last, 0, H * 4);
    await mv(s.last, W('lm_head.weight'), s.logits, 1, C.vocab, H);
    if (opts.argmax) await argmaxKernel(s.logits, s.tokHist, C.vocab, posBase + T);
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

  const GEN_BATCH = 8, PIPE_DEPTH = 3;
  // Deep-pipelined greedy decode (lfm25 design): GEN_BATCH chained forwards per
  // submit (GPU argmax → tokHist, no per-token readback), PIPE_DEPTH batches in
  // flight so CPU-encode, GPU-compute and readback overlap.
  async function _pipeDecode(startPos, maxTokens, emitTok, signal) {
    let pos = startPos, submitted = 0;
    const submitBatch = async () => {
      if (signal && signal.aborted) return null;
      const K = Math.min(GEN_BATCH, maxTokens - submitted, MAX_SEQ - 1 - pos);
      if (K <= 0) return null;
      const base = pos;
      E.beginBatch((globalThis.__gdnFlush | 0) || undefined);
      for (let k = 0; k < K; k++) await forward(null, base + k, { chain: true, argmax: true, batched: true });
      E.endBatch();
      pos += K; submitted += K;
      return { read: readU32Range(_scr.tokHist, base + 1, K), K };
    };
    const inflight = [];
    while (inflight.length < PIPE_DEPTH) { const b = await submitBatch(); if (!b) break; inflight.push(b); }
    while (inflight.length) {
      const cur = inflight.shift();
      const toks = await cur.read;
      let stop = false;
      for (let k = 0; k < cur.K; k++) { if (!emitTok(toks[k])) { stop = true; break; } }
      if (stop) break;
      const b = await submitBatch(); if (b) inflight.push(b);
    }
  }

  async function generate(prompt, { maxTokens = 64, onToken, signal } = {}) {
    if (!_loaded) throw new Error('loadModel first');
    const C = _cfg;
    const ids = Array.isArray(prompt) ? TOK.encodeChat(prompt) : TOK.encodeChat([{ role: 'user', content: prompt }]);
    if (ids.length + maxTokens + 2 > MAX_SEQ) throw new Error('prompt too long: ' + ids.length + ' + ' + maxTokens + ' > ' + MAX_SEQ);
    _ensureState();
    E.device().queue.writeBuffer(_scr.tokHist, 0, new Uint32Array(ids));
    // NOTE: GDN state persists across calls — a fresh generation must start clean.
    // posBase===0 tells gdnConv/gdnDelta useState=0 (zero-init), so no explicit clear needed.
    let pos = 0;
    const t0p = performance.now();
    const CH = PREFILL_T;
    for (let off = 0; off < ids.length; off += CH) {
      if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
      const chunk = ids.slice(off, Math.min(off + CH, ids.length));
      await forward(chunk, pos, { argmax: off + CH >= ids.length });
      pos += chunk.length;
    }
    const prefillMs = performance.now() - t0p;
    const outIds = [], imEnd = TOK.imEnd();
    const t0 = performance.now();
    let stopped = false;
    { const f = (await readU32Range(_scr.tokHist, ids.length, 1))[0];
      if (f === C.eos || f === imEnd) stopped = true;
      else { outIds.push(f); if (onToken) { try { onToken(TOK.decode([f]), f); } catch (_) {} } } }
    if (!stopped) {
      await _pipeDecode(pos, maxTokens, (t) => {
        if (t === C.eos || t === imEnd) return false;
        outIds.push(t); if (onToken) { try { onToken(TOK.decode(outIds.slice(-4)).slice(-24), t); } catch (_) {} }
        return outIds.length < maxTokens;
      }, signal);
    }
    _lastProf = {
      prefillToks: ids.length, prefillTps: +(ids.length / Math.max(1e-3, prefillMs / 1000)).toFixed(1),
      decodeToks: outIds.length, decodeTps: +(outIds.length / Math.max(1e-3, (performance.now() - t0) / 1000)).toFixed(2),
    };
    try { console.log('[gdn prof] ' + JSON.stringify(_lastProf)); } catch (_) {}
    return TOK.decode(outIds);
  }
  let _lastProf = null;

  // DIAGNOSTIC: run one non-chained forward over `text` tokens and report stats of
  // s.x (residual stream) after embed and after EVERY layer, plus final logits stats.
  // The all-"!" bisector: find the first stage where activations go zero/NaN/flat.
  async function _stageProbe(text) {
    if (!_loaded) throw new Error('loadModel first');
    const C = _cfg, H = C.hidden;
    const ids = TOK.encodeText(text || 'The capital of France is');
    const T = ids.length;
    _ensureState();
    const stats = async (buf, n, label) => {
      const a = await E.readF32(buf, n);
      let mn = Infinity, mx = -Infinity, sum = 0, nan = 0, zero = 0;
      for (let i = 0; i < n; i++) { const v = a[i]; if (Number.isNaN(v)) { nan++; continue; } if (v === 0) zero++; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
      return { label, min: +mn.toFixed(4), max: +mx.toFixed(4), mean: +(sum / n).toFixed(6), nan, zeroPct: +(zero / n * 100).toFixed(1) };
    };
    const out = [];
    const s = _scr;
    if (!_idsBuf) _idsBuf = E.createBuffer(Math.max(MATVEC_MAXT, PREFILL_T) * 4, U.STORAGE | U.COPY_DST, 'ids');
    E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(ids));
    uniformReset();
    await embedGather(_idsBuf, _weights[EMBED].buf, s.x, T, H, 0);
    out.push(await stats(s.x, T * H, 'embed'));
    const G = C.gdn, CD = GDN_CONV_DIM, HV = G.nHeads * G.dv;
    const W = (n) => _weights[n];
    for (let l = 0; l < C.numLayers; l++) {
      const p = 'model.language_model.layers.' + l + '.';
      uniformReset();
      E.beginBatch();
      await rmsnorm(s.x, W(p + 'input_layernorm.weight').buf, s.normed, T, H, C.rmsEps);
      if (_gdnState[l]) {
        const st = _gdnState[l];
        await mv(s.normed, W(p + 'linear_attn.in_proj_qkv.weight'), s.qkv, T, CD, H);
        const cIn = st.cur === 0 ? st.convA : st.convB, cOut = st.cur === 0 ? st.convB : st.convA;
        await gdnConv(s.qkv, W(p + 'linear_attn.conv1d.weight').buf, cIn, cOut, s.qkvc, T, CD, G.convL, false);
        st.cur ^= 1;
        await gdnQkNorm(s.qkvc, T, G.nHeads, G.dk, CD);
        await mvF16(s.normed, W(p + 'linear_attn.in_proj_b.weight').buf, s.bproj, T, G.nHeads, H);
        await mvF16(s.normed, W(p + 'linear_attn.in_proj_a.weight').buf, s.aproj, T, G.nHeads, H);
        await gdnGates(s.bproj, s.aproj, W(p + 'linear_attn.A_log').buf, W(p + 'linear_attn.dt_bias').buf, s.beta, s.gexp, T, G.nHeads);
        await gdnDelta(s.qkvc, s.beta, s.gexp, st.S, s.gdnO, T, G.nHeads, G.dk, false);
        await mv(s.normed, W(p + 'linear_attn.in_proj_z.weight'), s.z, T, HV, H);
        await gatedRmsNorm(s.gdnO, s.z, W(p + 'linear_attn.norm.weight').buf, s.gnorm, T, G.nHeads, G.dv, C.rmsEps);
        await mv(s.gnorm, W(p + 'linear_attn.out_proj.weight'), s.x, T, H, HV, true);
      } else {
        const nHq = C.nHeads, nKv = C.nKvHeads, hd = C.headDim;
        await mv(s.normed, W(p + 'self_attn.q_proj.weight'), s.q, T, nHq * hd, H);
        await mv(s.normed, W(p + 'self_attn.out_gate.weight'), s.agate, T, nHq * hd, H);
        await mv(s.normed, W(p + 'self_attn.k_proj.weight'), s.k, T, nKv * hd, H);
        await mv(s.normed, W(p + 'self_attn.v_proj.weight'), s.v, T, nKv * hd, H);
        await ropeQK(s.q, W(p + 'self_attn.q_norm.weight').buf, s.qr, T, nHq, hd, 0, C.ropeTheta, C.rmsEps, C.rotDim);
        await ropeQK(s.k, W(p + 'self_attn.k_norm.weight').buf, s.kr, T, nKv, hd, 0, C.ropeTheta, C.rmsEps, C.rotDim);
        E.copyBuffer(s.kr, 0, _kv[l].k, 0, T * nKv * hd * 4);
        E.copyBuffer(s.v, 0, _kv[l].v, 0, T * nKv * hd * 4);
        await attention(s.qr, _kv[l].k, _kv[l].v, s.attn, T, T, nHq, nKv, hd);
        await mulSigmoid(s.attn, s.agate, T * nHq * hd);
        await mv(s.attn, W(p + 'self_attn.o_proj.weight'), s.x, T, H, nHq * hd, true);
      }
      await rmsnorm(s.x, W(p + 'post_attention_layernorm.weight').buf, s.normed, T, H, C.rmsEps);
      const I = C.intermediate;
      await mv(s.normed, W(p + 'mlp.gate_proj.weight'), s.gate, T, I, H);
      await mv(s.normed, W(p + 'mlp.up_proj.weight'), s.up, T, I, H);
      await swiglu(s.gate, s.up, s.swi, T * I);
      await mv(s.swi, W(p + 'mlp.down_proj.weight'), s.x, T, H, I, true);
      await E.endBatch();
      out.push(await stats(s.x, T * H, 'L' + l + (_gdnState[l] ? ' gdn' : ' attn')));
    }
    uniformReset();
    E.beginBatch();
    await rmsnorm(s.x, W('model.language_model.norm.weight').buf, s.normed, T, H, C.rmsEps);
    E.copyBuffer(s.normed, (T - 1) * H * 4, s.last, 0, H * 4);
    await mv(s.last, W('lm_head.weight'), s.logits, 1, C.vocab, H);
    await E.endBatch();
    out.push(await stats(s.logits, C.vocab, 'logits'));
    const lg = await E.readF32(s.logits, C.vocab);
    let best = 0, bv = -Infinity; for (let i = 0; i < C.vocab; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
    return { T, ids: ids.slice(0, 8), argmax: best, argmaxTok: TOK.decode([best]), stages: out };
  }

  // Layer-0 GDN substage stats — mirror of the torch hf_l0.py dump.
  async function _l0Probe(text) {
    if (!_loaded) throw new Error('loadModel first');
    const C = _cfg, H = C.hidden, G = C.gdn, CD = GDN_CONV_DIM, HV = G.nHeads * G.dv;
    const ids = TOK.encodeText(text || 'The capital of France is');
    const T = ids.length;
    _ensureState();
    const s = _scr, W = (n) => _weights[n], p = 'model.language_model.layers.0.';
    const st = async (buf, n, label) => {
      const a = await E.readF32(buf, n);
      let mn = Infinity, mx = -Infinity, sum = 0;
      for (let i = 0; i < n; i++) { const v = a[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
      return { label, min: +mn.toFixed(4), max: +mx.toFixed(4), mean: +(sum / n).toFixed(6) };
    };
    const out = [];
    if (!_idsBuf) _idsBuf = E.createBuffer(Math.max(MATVEC_MAXT, PREFILL_T) * 4, U.STORAGE | U.COPY_DST, 'ids');
    E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(ids));
    uniformReset();
    await embedGather(_idsBuf, W(EMBED).buf, s.x, T, H, 0);
    await rmsnorm(s.x, W(p + 'input_layernorm.weight').buf, s.normed, T, H, C.rmsEps);
    out.push(await st(s.normed, T * H, 'normed'));
    await mv(s.normed, W(p + 'linear_attn.in_proj_qkv.weight'), s.qkv, T, CD, H);
    out.push(await st(s.qkv, T * CD, 'qkv_proj'));
    const g0 = _gdnState[0];
    await gdnConv(s.qkv, W(p + 'linear_attn.conv1d.weight').buf, g0.convA, g0.convB, s.qkvc, T, CD, G.convL, false);
    out.push(await st(s.qkvc, T * CD, 'conv_silu'));
    await gdnQkNorm(s.qkvc, T, G.nHeads, G.dk, CD);
    // slice stats: q rows [0,2048), k [2048,4096), v [4096,6144) per token — read full and split on CPU
    {
      const a = await E.readF32(s.qkvc, T * CD);
      const seg = (off, len, label) => { let mn = 1e9, mx = -1e9, sum = 0, n = 0; for (let t = 0; t < T; t++) for (let i = 0; i < len; i++) { const v = a[t * CD + off + i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; n++; } return { label, min: +mn.toFixed(4), max: +mx.toFixed(4), mean: +(sum / n).toFixed(6) }; };
      out.push(seg(0, 2048, 'q_normed_scaled'), seg(2048, 2048, 'k_normed'), seg(4096, 2048, 'v'));
    }
    await mvF16(s.normed, W(p + 'linear_attn.in_proj_b.weight').buf, s.bproj, T, G.nHeads, H);
    await mvF16(s.normed, W(p + 'linear_attn.in_proj_a.weight').buf, s.aproj, T, G.nHeads, H);
    await gdnGates(s.bproj, s.aproj, W(p + 'linear_attn.A_log').buf, W(p + 'linear_attn.dt_bias').buf, s.beta, s.gexp, T, G.nHeads);
    out.push(await st(s.beta, T * G.nHeads, 'beta'), await st(s.gexp, T * G.nHeads, 'gexp'));
    await gdnDelta(s.qkvc, s.beta, s.gexp, g0.S, s.gdnO, T, G.nHeads, G.dk, false);
    out.push(await st(s.gdnO, T * HV, 'delta_o'));
    await mv(s.normed, W(p + 'linear_attn.in_proj_z.weight'), s.z, T, HV, H);
    await gatedRmsNorm(s.gdnO, s.z, W(p + 'linear_attn.norm.weight').buf, s.gnorm, T, G.nHeads, G.dv, C.rmsEps);
    out.push(await st(s.gnorm, T * HV, 'gated_norm'));
    // out_proj WITHOUT residual (separate buffer) then with
    await mv(s.gnorm, W(p + 'linear_attn.out_proj.weight'), s.attn, T, H, HV, false);
    out.push(await st(s.attn, T * H, 'out_proj'));
    await axpy(s.x, s.attn, T * H, 1);
    out.push(await st(s.x, T * H, 'after_residual'));
    return out;
  }

  const _readTok = (idx, count) => readU32Range(_scr.tokHist, idx, count || 1);
  const _writeTokHist = (ids) => { _ensureState(); E.device().queue.writeBuffer(_scr.tokHist, 0, new Uint32Array(ids)); };
  const _readLast = () => E.readF32(_scr.last, _cfg.hidden);
  const _readScr = (name, n) => E.readF32(_scr[name], n);
  const _readLogits = (n) => E.readF32(_scr.logits, n || _cfg.vocab);
  // Isolated argmax test: synthetic logits with a known peak → run the real kernel
  // against the real tokHist → read back.
  async function _testArgmax(peak) {
    _ensureState();
    const N = _cfg.vocab;
    const lg = new Float32Array(N); for (let i = 0; i < N; i++) lg[i] = -10 + Math.sin(i * 0.1);
    lg[peak] = 42;
    E.device().queue.writeBuffer(_scr.logits, 0, lg);
    uniformReset();
    E.beginBatch();
    await argmaxKernel(_scr.logits, _scr.tokHist, N, 3);
    await E.endBatch();
    return (await readU32Range(_scr.tokHist, 3, 1))[0];
  }

  return {
    CONFIG, GDN_CONV_DIM, MODELS, _stageProbe, _l0Probe, _readTok, _writeTokHist, _readLast, _readLogits, _readScr, _testArgmax,
    gdnConv, gdnGates, gdnQkNorm, gdnDelta, gatedRmsNorm,
    loadModel, unload, TOK, generate, forward,
    lastProf: () => _lastProf,
    _fwd: { mv, matvecQ, gemvQ8, gemmDP4A, rmsnorm, embedGather, swiglu, axpy, ropeQK, attention, mulSigmoid, mvF16, argmaxKernel },
    _selfTest: selfTestKernels,
    _selfTestFwd: selfTestFwdPrims,
    _weightsInfo: () => { const o = {}; let bytes = 0; for (const k in (_weights || {})) { const w = _weights[k]; o[k] = { int4: !!w.int4, f32: !!w.f32, shape: w.shape }; } return { count: Object.keys(o).length, tensors: o }; },
  };
})();

if (typeof self !== 'undefined') self.SandpieQwen35 = SandpieQwen35;
if (typeof window !== 'undefined') window.SandpieQwen35 = SandpieQwen35;
