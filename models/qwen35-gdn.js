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
// LOADER CONTRACT (phase B): HF's in_proj_qkvz / in_proj_ba outputs are INTERLEAVED per
// head ([q_h|k_h|v_h|z_h] × 16, [b_h|a_h] × 16). The loader must PERMUTE THE WEIGHT ROWS
// once so projections emit contiguous q|k|v|z and b|a — kernels below assume that layout
// and pay zero runtime cost for it.
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

  return {
    CONFIG, GDN_CONV_DIM,
    gdnConv, gdnGates, gdnQkNorm, gdnDelta, gatedRmsNorm,
    _selfTest: selfTestKernels,
  };
})();

if (typeof self !== 'undefined') self.SandpieQwen35 = SandpieQwen35;
if (typeof window !== 'undefined') window.SandpieQwen35 = SandpieQwen35;
