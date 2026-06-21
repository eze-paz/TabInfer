// sandpie/modules/webgpu-qwen35.js — Qwen3.5 model layer (FORK of webgpu-qwen3.js).
// Depends on window.SandpieWebGPU (engine core). Reuses Qwen3's int4 GEMV/matvec,
// rmsnorm, embed, swiglu, gateUpSilu, argmax, loader/tokenizer scaffold,
// batch-generate, and the FULL-ATTENTION kernel — see [[project_sandpie-webgpu-engine]].
//
// ⚠️ SCAFFOLD — DOES NOT GENERATE YET. Qwen3.5 is NOT a plain transformer:
// it's a HYBRID. Of 24 layers, every 4th is full attention (6 total); the other
// 18 are gated-DeltaNet LINEAR attention (Mamba2-style recurrence + causal conv1d).
// The DeltaNet path needs novel WGSL kernels that don't exist yet (see deltaNet()).
//
// Qwen3.5-0.8B (verified from config.json + safetensors.index.json, model_type
// qwen3_5_text under multimodal Qwen3_5ForConditionalGeneration):
//   24 layers, layer_types = 3×linear_attention then 1×full_attention, ×6
//   (full_attention_interval=4 → layers 3,7,11,15,19,23 are full attention).
//   hidden 1024 · intermediate 3584 · vocab 248320 · tied embeds · RMSNorm 1e-6 ·
//   SwiGLU(silu). Weights under prefix `model.language_model.`; single shard
//   `model.safetensors-00001-of-00001.safetensors`. Vision tower (model.visual.*)
//   and MTP head (mtp.*) are SKIPPED (text-only chat).
//   FULL-ATTENTION layer: GQA 8 q-heads / 2 kv-heads, head_dim 256, per-head
//     q_norm/k_norm; self_attn.{q,k,v,o}_proj. (NB head_dim 256 ≠ Qwen3's 128 —
//     the attention kernel's hardcoded hd=128 / WG=128 must be parameterized.)
//   LINEAR-ATTENTION layer (gated DeltaNet): linear_attn.{in_proj_qkv, in_proj_a,
//     in_proj_b, in_proj_z, conv1d (kernel=4, causal), A_log, dt_bias, norm,
//     out_proj}; 16 value/key heads, key/value head_dim 128. NO WebGPU kernel
//     exists for this recurrence — it is the hard, net-new work.

const SandpieQwen35 = (function () {
  'use strict';

  const E = (typeof window !== 'undefined') ? window.SandpieWebGPU : null;

  // layer_types: false = linear_attention (DeltaNet), true = full_attention.
  // full_attention_interval=4 → every 4th layer (index%4==3) is full attention.
  const LAYER_FULL_ATTN = Array.from({ length: 24 }, (_, i) => (i % 4) === 3);

  // Qwen3.5 variants — architecturally identical except hidden_size & intermediate_size
  // (same 24 layers, head_dim 256, DeltaNet 16×128, vocab 248320, rotary). So switching
  // models is purely a config + URL change; no kernel changes.
  // repo = full HF "author/name"; file = the safetensors shard; compressed = pre-quantized
  // compressed-tensors (pack-quantized int4, group 32, symmetric — matches our scheme).
  const VARIANTS = {
    '0.8B':   { hidden: 1024, intermediate: 3584, repo: 'Qwen/Qwen3.5-0.8B', file: 'model.safetensors-00001-of-00001.safetensors' },
    '2B':     { hidden: 2048, intermediate: 6144, repo: 'Qwen/Qwen3.5-2B',   file: 'model.safetensors-00001-of-00001.safetensors' },
    '2B-AWQ': { hidden: 2048, intermediate: 6144, repo: 'cyankiwi/Qwen3.5-2B-AWQ-4bit', file: 'model-00001-of-00001.safetensors', compressed: true },
  };
  let _variant = '0.8B';
  const CONFIG = {
    numLayers: 24, hidden: 1024,
    // full-attention layers:
    nHeads: 8, nKvHeads: 2, headDim: 256,
    // gated-DeltaNet (linear-attention) layers:
    deltaHeads: 16, deltaKeyDim: 128, deltaValDim: 128, convKernel: 4,
    intermediate: 3584, vocab: 248320,
    ropeTheta: 10000000, rotaryDim: 64,   // partial_rotary_factor 0.25 × head_dim 256
    rmsEps: 1e-6,
    tieEmbeddings: true,
    layerFullAttn: LAYER_FULL_ATTN,
    weightPrefix: 'model.language_model.',
  };

  const U = GPUBufferUsage;
  const ST = () => (U.STORAGE | U.COPY_DST | U.COPY_SRC);

  // Pooled uniform buffers. Each uniform() call in a forward gets a STABLE buffer
  // (indexed by call order, reset per forward via uniformReset()) whose contents
  // are rewritten each token — so the engine's bind-group cache keeps hitting
  // instead of rebuilding ~250 bind groups/token. Pool buffers are a fixed 32B
  // (covers every kernel's uniform). Outside a forward (self-tests) the index just
  // keeps growing — still correct, just allocates a few extra pool slots once.
  let _uPool = [], _uIdx = 0;
  function uniformReset() { _uIdx = 0; }
  function uniform(arr) {
    let buf = _uPool[_uIdx];
    if (!buf) { buf = E.createBuffer(32, U.UNIFORM | U.COPY_DST, 'u' + _uIdx); _uPool[_uIdx] = buf; }
    E.device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset || 0, arr.byteLength);
    _uIdx++;
    return buf;
  }

  // ============================================================
  // Kernel 1 — RMSNorm.  y[T,H] = x / sqrt(mean(x²)+eps) * w[H]
  // One workgroup per row; WG_H threads cooperatively reduce over H.
  // ============================================================
  const WG_H = 256;
  const RMSNORM_WGSL = `
enable f16;
struct P { T:u32, H:u32, eps:f32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       w : array<f16>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;
var<workgroup> red : array<f32, ${WG_H}>;
@compute @workgroup_size(${WG_H},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let row = wg.x; let H = p.H; let base = row*H;
  var s : f32 = 0.0;
  var i = lid.x;
  loop { if (i >= H) { break; } let v = x[base+i]; s = s + v*v; i = i + ${WG_H}u; }
  red[lid.x] = s; workgroupBarrier();
  var stride = ${WG_H}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){red[lid.x]=red[lid.x]+red[lid.x+stride];} workgroupBarrier(); stride=stride/2u; }
  let inv = inverseSqrt(red[0]/f32(H) + p.eps);
  i = lid.x;
  loop { if (i >= H) { break; } y[base+i] = x[base+i]*inv*(1.0 + f32(w[i])); i = i + ${WG_H}u; }   // Qwen3.5 RMSNorm: *(1+weight)
}`;
  function rmsnorm(xBuf, wBuf, yBuf, T, H, eps) {
    const pipe = E.getPipeline('q3.rmsnorm', RMSNORM_WGSL);
    // pack T,H (u32) + eps (f32); pooled uniform (stable buffer for bind-group cache)
    const u = new Uint32Array(4); const du = new DataView(u.buffer);
    du.setUint32(0, T, true); du.setUint32(4, H, true); du.setFloat32(8, eps, true);
    const p = uniform(u);
    return E.dispatch(pipe, [xBuf, wBuf, yBuf, p], [T, 1, 1]);
  }

  // ============================================================
  // Kernel 2 — Linear (transposed weight).  y[T,N] = x[T,K] · W[N,K]ᵀ
  // W is stored [out=N, in=K] row-major (safetensors nn.Linear layout), so
  // y[t,n] = Σ_k x[t,k]·W[n,k]. Tiled 16×16; reads W transposed in-place (no
  // explicit transpose of the big weight matrices).
  // ============================================================
  const TILE = 16;
  const LINEAR_WGSL = `
enable f16;
struct D { T:u32, N:u32, K:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       W : array<f16>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> tX : array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> tW : array<array<f32, ${TILE}>, ${TILE}>;
@compute @workgroup_size(${TILE},${TILE},1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let row=gid.y; let col=gid.x; let T=d.T; let N=d.N; let K=d.K;
  var acc:f32=0.0;
  let nT=(K+${TILE}u-1u)/${TILE}u;
  for (var t:u32=0u; t<nT; t=t+1u){
    let xCol=t*${TILE}u+lid.x;
    let wCol=t*${TILE}u+lid.y;   // index into K dim of W[col, K]
    tX[lid.y][lid.x]=select(0.0, x[row*K+xCol], (row<T)&&(xCol<K));
    tW[lid.y][lid.x]=select(0.0, f32(W[col*K+wCol]), (col<N)&&(wCol<K));
    workgroupBarrier();
    for (var k:u32=0u;k<${TILE}u;k=k+1u){ acc=acc+tX[lid.y][k]*tW[k][lid.x]; }
    workgroupBarrier();
  }
  if((row<T)&&(col<N)){ y[row*N+col]=acc; }
}`;
  function linearT(xBuf, wBuf, yBuf, T, N, K) {
    const pipe = E.getPipeline('q3.linearT', LINEAR_WGSL);
    const d = uniform(new Uint32Array([T, N, K, 0]));
    return E.dispatch(pipe, [xBuf, wBuf, yBuf, d], [Math.ceil(N/TILE), Math.ceil(T/TILE), 1]);
  }

  // ----- Decode GEMV: y[N] = x[K] · W[N,K]ᵀ  (the T==1 fast path) -----
  // Mirrors the reference engine's separate M==1 path. The tiled GEMM wastes
  // ~94% of its threads at T=1 (16-row tile, 1 valid row); this gives every
  // thread useful reduction work: one workgroup per output row n, WG threads
  // stride K and reduce. 2D workgroup grid because N (151936 for lm_head)
  // exceeds the 65535 per-dimension dispatch limit.
  // Vectorized + shared-mem-reduced + N_ROWS reuse. Reads weights as vec4<f16> and
  // x as vec4<f32> (4 elems/load), reduces partials with a workgroup tree reduction
  // (portable — subgroupAdd broke on mobile GPUs), and each
  // workgroup computes GEMV_NR output rows — the activation chunk x[c] is read ONCE
  // and reused across all NR rows (fewer workgroups, the activation read amortized
  // NR×). Requires K % 4 == 0 (all Qwen3 matrices). N need not divide NR (row<N
  // guards; out-of-range weight reads are bounds-checked to 0 and never written).
  const GEMV_WG = 64;
  const GEMV_NR = 4;   // output rows per workgroup
  const GEMV_WGSL = `
enable f16;
struct D { N:u32, K:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       x : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> part : array<f32, ${GEMV_NR * GEMV_WG}>;   // part[r*WG + lid] — shared-mem tree reduction
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>) {
  let rowBase = (wg.x + wg.y * nwg.x) * ${GEMV_NR}u;
  if (rowBase >= d.N) { return; }
  let K4 = d.K / 4u;
  var acc : array<f32, ${GEMV_NR}>;
  for (var r:u32=0u; r<${GEMV_NR}u; r=r+1u) { acc[r] = 0.0; }
  var c = lid.x;
  loop {
    if (c >= K4) { break; }
    let xv = x[c];                                   // activation chunk — read once
    for (var r:u32=0u; r<${GEMV_NR}u; r=r+1u) {
      acc[r] = acc[r] + dot(xv, vec4<f32>(W[(rowBase+r)*K4 + c]));
    }
    c = c + ${GEMV_WG}u;
  }
  for (var r:u32=0u; r<${GEMV_NR}u; r=r+1u) { part[r*${GEMV_WG}u + lid.x] = acc[r]; }
  workgroupBarrier();
  var stride = ${GEMV_WG}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){ for(var r:u32=0u;r<${GEMV_NR}u;r=r+1u){ part[r*${GEMV_WG}u+lid.x] = part[r*${GEMV_WG}u+lid.x] + part[r*${GEMV_WG}u+lid.x+stride]; } } workgroupBarrier(); stride=stride/2u; }
  if (lid.x < ${GEMV_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.N) { y[row] = part[lid.x*${GEMV_WG}u]; }
  }
}`;
  function gemv(xBuf, wBuf, yBuf, N, K) {
    const pipe = E.getPipeline('q3.gemv', GEMV_WGSL);
    const d = uniform(new Uint32Array([N, K, 0, 0]));
    const nWG = Math.ceil(N / GEMV_NR);
    const gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [xBuf, wBuf, yBuf, d], [gx, gy, 1]);
  }
  // ----- Batched matvec for small T (prefill / speculative verify) -----
  // y[T,N] = x[T,K]·W[N,K]ᵀ. One workgroup per output row n; each WEIGHT row is
  // read ONCE and reused across all T tokens (the T-analog of N_ROWS) → a T-token
  // forward reads the weights once, like decode. vec4 loads + shared-mem reduction. The old
  // tiled GEMM read weights per 16-tile and was ~3.7× slower at T=8 (measured).
  // Handles T ≤ MATVEC_MAXT in one dispatch; K % 4 == 0.
  const MATVEC_WG = 64;
  const MATVEC_MAXT = 16;
  const MATVEC_WGSL = `
enable f16;
struct D { T:u32, N:u32, K:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<vec4<f32>>;   // [T, K/4]
@group(0) @binding(1) var<storage, read>       W : array<vec4<f16>>;   // [N, K/4]
@group(0) @binding(2) var<storage, read_write> y : array<f32>;          // [T, N]
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> part : array<f32, ${MATVEC_MAXT * MATVEC_WG}>;  // part[t*WG + lid] — shared-mem tree reduction
@compute @workgroup_size(${MATVEC_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>) {
  let n = wg.x + wg.y * nwg.x;
  if (n >= d.N) { return; }
  let K4 = d.K / 4u; let wbase = n*K4; let T = d.T;
  var acc : array<f32, ${MATVEC_MAXT}>;
  for (var t:u32=0u; t<${MATVEC_MAXT}u; t=t+1u) { acc[t] = 0.0; }
  var c = lid.x;
  loop {
    if (c >= K4) { break; }
    let wv = vec4<f32>(W[wbase+c]);                 // weight chunk — read once
    for (var t:u32=0u; t<T; t=t+1u) { acc[t] = acc[t] + dot(x[t*K4 + c], wv); }
    c = c + ${MATVEC_WG}u;
  }
  for (var t:u32=0u; t<T; t=t+1u) { part[t*${MATVEC_WG}u + lid.x] = acc[t]; }
  workgroupBarrier();
  var stride = ${MATVEC_WG}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){ for(var t:u32=0u;t<T;t=t+1u){ part[t*${MATVEC_WG}u+lid.x] = part[t*${MATVEC_WG}u+lid.x] + part[t*${MATVEC_WG}u+lid.x+stride]; } } workgroupBarrier(); stride=stride/2u; }
  if (lid.x < T) { y[lid.x*d.N + n] = part[lid.x*${MATVEC_WG}u]; }
}`;
  function matvecT(xBuf, wBuf, yBuf, T, N, K) {
    const pipe = E.getPipeline('q3.matvecT', MATVEC_WGSL);
    const d = uniform(new Uint32Array([T, N, K, 0]));
    const gx = Math.min(N, 65535), gy = Math.ceil(N / gx);
    return E.dispatch(pipe, [xBuf, wBuf, yBuf, d], [gx, gy, 1]);
  }

  // ============================================================
  // INT4 weight path — group-wise symmetric int4 (G=QGROUP) along K.
  // Weights packed 8 nibbles/u32 (nibble for k=8w+i in bits [4i..4i+3]); one f16
  // scale per group. Dequant: w ≈ (nibble-8)*scale. 4× less weight bandwidth than
  // f16 → the lever for the bandwidth-bound GEMV (proven). Used for all projection
  // + lm_head matrices; embed/norms stay f16.  Requires K % 32 == 0.
  // ============================================================
  const QGROUP = 32;

  // ---- INT4 decode GEMV (T=1), GEMVQ_NR rows/workgroup (activation reused) ----
  const GEMVQ_NR = 8;
  // Reduction is capability-conditional: subgroupAdd on desktop GPUs that support
  // subgroups (faster), portable shared-mem tree reduction elsewhere (mobile).
  const gemvqWgsl = (sub) => `
enable f16;${sub ? '\nenable subgroups;' : ''}
struct D { N:u32, K:u32, acc:u32, _b:u32 };   // acc=1 → y[n] += result (fused residual)
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;   // [K/4]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;          // [N*K/8] packed nibbles
@group(0) @binding(2) var<storage, read>       sc : array<f16>;          // [N*K/QGROUP] scales
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;          // [N]
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${GEMVQ_NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>${sub ? ',\n        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32' : ''}) {
  let rowBase = (wg.x + wg.y * nwg.x) * ${GEMVQ_NR}u;
  if (rowBase >= d.N) { return; }
  let words = d.K / 8u; let gpr = d.K / ${QGROUP}u;
  var acc : array<f32, ${GEMVQ_NR}>;
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = x[2u*w]; let xb = x[2u*w + 1u];      // activation chunk — read once, reused across rows
    let grp = (w*8u)/${QGROUP}u;
    for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) {
      let row = rowBase + r;
      let p = W[row*words + w];
      let s = f32(sc[row*gpr + grp]);
      let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      acc[r] = acc[r] + s*( dot(vec4<f32>(lo.x,hi.x,lo.y,hi.y), xa) + dot(vec4<f32>(lo.z,hi.z,lo.w,hi.w), xb) );
    }
    w = w + ${GEMV_WG}u;
  }
` + (sub ? `
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) { let ss = subgroupAdd(acc[r]); if (sgi == 0u) { part[r*${GEMV_WG}u + sgIdx] = ss; } }
  workgroupBarrier();
  if (lid.x < ${GEMVQ_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.N) {
      let nsg=(${GEMV_WG}u+sgs-1u)/sgs; var t:f32=0.0;
      for(var i:u32=0u;i<nsg;i=i+1u){ t = t + part[lid.x*${GEMV_WG}u + i]; }
      y[row] = select(0.0, y[row], d.acc != 0u) + t;
    }
  }
}` : `
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) { part[r*${GEMV_WG}u + lid.x] = acc[r]; }
  workgroupBarrier();
  var stride = ${GEMV_WG}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){ for(var r:u32=0u;r<${GEMVQ_NR}u;r=r+1u){ part[r*${GEMV_WG}u+lid.x] = part[r*${GEMV_WG}u+lid.x] + part[r*${GEMV_WG}u+lid.x+stride]; } } workgroupBarrier(); stride=stride/2u; }
  if (lid.x < ${GEMVQ_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.N) { y[row] = select(0.0, y[row], d.acc != 0u) + part[lid.x*${GEMV_WG}u]; }
  }
}`);
  function gemvQ(xBuf, packBuf, scBuf, yBuf, N, K, acc) {
    const sub = _useSub();
    const pipe = E.getPipeline(sub ? 'q3.gemvQ.sub' : 'q3.gemvQ', gemvqWgsl(sub));
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const nWG = Math.ceil(N / GEMVQ_NR);
    const gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [xBuf, packBuf, scBuf, yBuf, d], [gx, gy, 1]);
  }

  // ---- FUSED int4 gate+up+SwiGLU (T=1): swi[i] = silu(gate·x)*(up·x) ----
  // 3 decode passes (gate gemv, up gemv, swiglu) → 1. Reads x once per chunk and
  // dequant-dots it against BOTH gate and up weights; one fewer pass barrier ×2
  // per layer (the matmuls) plus the swiglu pass removed.
  const GUSQ_NR = 4;
  const gateupqWgsl = (sub) => `
enable f16;${sub ? '\nenable subgroups;' : ''}
struct D { I:u32, H:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;   // [H/4]
@group(0) @binding(1) var<storage, read>       gW : array<u32>;
@group(0) @binding(2) var<storage, read>       gS : array<f16>;
@group(0) @binding(3) var<storage, read>       uW : array<u32>;
@group(0) @binding(4) var<storage, read>       uS : array<f16>;
@group(0) @binding(5) var<storage, read_write> swi: array<f32>;          // [I]
@group(0) @binding(6) var<uniform>             d  : D;
var<workgroup> pg : array<f32, ${GUSQ_NR * GEMV_WG}>;
var<workgroup> pu : array<f32, ${GUSQ_NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>${sub ? ',\n        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32' : ''}) {
  let rowBase = (wg.x + wg.y * nwg.x) * ${GUSQ_NR}u;
  if (rowBase >= d.I) { return; }
  let words = d.H / 8u; let gpr = d.H / ${QGROUP}u;
  var ga : array<f32, ${GUSQ_NR}>; var ua : array<f32, ${GUSQ_NR}>;
  for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) { ga[r]=0.0; ua[r]=0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = x[2u*w]; let xb = x[2u*w + 1u];
    let grp = (w*8u)/${QGROUP}u;
    for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) {
      let row = rowBase + r; let wi = row*words + w; let si = row*gpr + grp;
      let gp = gW[wi]; let gsc = f32(gS[si]);
      let glo = vec4<f32>(unpack4xU8(gp & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let ghi = vec4<f32>(unpack4xU8((gp >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      ga[r] = ga[r] + gsc*( dot(vec4<f32>(glo.x,ghi.x,glo.y,ghi.y), xa) + dot(vec4<f32>(glo.z,ghi.z,glo.w,ghi.w), xb) );
      let up = uW[wi]; let usc = f32(uS[si]);
      let ulo = vec4<f32>(unpack4xU8(up & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let uhi = vec4<f32>(unpack4xU8((up >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      ua[r] = ua[r] + usc*( dot(vec4<f32>(ulo.x,uhi.x,ulo.y,uhi.y), xa) + dot(vec4<f32>(ulo.z,uhi.z,ulo.w,uhi.w), xb) );
    }
    w = w + ${GEMV_WG}u;
  }
` + (sub ? `
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) { let g=subgroupAdd(ga[r]); let u=subgroupAdd(ua[r]); if (sgi==0u) { pg[r*${GEMV_WG}u + sgIdx]=g; pu[r*${GEMV_WG}u + sgIdx]=u; } }
  workgroupBarrier();
  if (lid.x < ${GUSQ_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.I) {
      let nsg=(${GEMV_WG}u+sgs-1u)/sgs; var g:f32=0.0; var u:f32=0.0;
      for(var i:u32=0u;i<nsg;i=i+1u){ g=g+pg[lid.x*${GEMV_WG}u+i]; u=u+pu[lid.x*${GEMV_WG}u+i]; }
      let silu = g / (1.0 + exp(-g));
      swi[row] = silu * u;
    }
  }
}` : `
  for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) { pg[r*${GEMV_WG}u + lid.x] = ga[r]; pu[r*${GEMV_WG}u + lid.x] = ua[r]; }
  workgroupBarrier();
  var stride = ${GEMV_WG}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){ for(var r:u32=0u;r<${GUSQ_NR}u;r=r+1u){ pg[r*${GEMV_WG}u+lid.x]=pg[r*${GEMV_WG}u+lid.x]+pg[r*${GEMV_WG}u+lid.x+stride]; pu[r*${GEMV_WG}u+lid.x]=pu[r*${GEMV_WG}u+lid.x]+pu[r*${GEMV_WG}u+lid.x+stride]; } } workgroupBarrier(); stride=stride/2u; }
  if (lid.x < ${GUSQ_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.I) {
      let g = pg[lid.x*${GEMV_WG}u]; let u = pu[lid.x*${GEMV_WG}u];
      let silu = g / (1.0 + exp(-g));
      swi[row] = silu * u;
    }
  }
}`);
  function gateUpSiluQ(xBuf, gRec, uRec, swiBuf, I, H) {
    const sub = _useSub();
    const pipe = E.getPipeline(sub ? 'q3.gateupQ.sub' : 'q3.gateupQ', gateupqWgsl(sub));
    const d = uniform(new Uint32Array([I, H, 0, 0]));
    const nWG = Math.ceil(I / GUSQ_NR);
    const gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [xBuf, gRec.pack, gRec.scales, uRec.pack, uRec.scales, swiBuf, d], [gx, gy, 1]);
  }

  // ============================================================
  // Qwen3.5 — Gated DeltaNet recurrent step (decode, 1 token). THE crux kernel.
  // Per head: S[key][val] state. Inputs q,k (L2-normed), v, and per-head expg
  // (decay = exp(-exp(A_log)*softplus(a+dt_bias))) + beta (sigmoid(b)). Exact ref
  // (HF Qwen3NextGatedDeltaNet torch_recurrent_gated_delta_rule):
  //   S = S*expg ; kv[val]=Σ_key S[key][val]*k[key] ; δ[val]=(v[val]-kv[val])*beta
  //   S[key][val] += k[key]*δ[val] ; out[val]=Σ_key S[key][val]*q[key]
  // STATE STORED TRANSPOSED St[head][val][key] so thread=val owns a contiguous row.
  // One workgroup per head; DELTA_DIM threads (val index). g/beta/L2-norm computed
  // upstream (separate kernels, TODO); this kernel is the recurrence proper.
  // ============================================================
  const DELTA_DIM = 128;   // key_head_dim == value_head_dim
  // Gated-DeltaNet recurrence over a CHUNK of T tokens in ONE dispatch (loops t INSIDE the
  // kernel) — collapses the prefill's per-token deltaRecur loop (T×layers tiny dispatches,
  // the post-GEMM bottleneck) to one dispatch/layer. The recurrence stays sequential (state
  // S in the persistent buffer), but each thread owns its own S columns (sbase + kk*dim,
  // stride dim) so the per-t S read-after-write needs no barrier; only the shared k/q tiles
  // do. T=1 (decode) → the original single-token step.
  // S-IN-SHARED chunk recurrence: the state (128×128/head) was streamed from GLOBAL every
  // token (~256KB/head/token = the post-GEMM bottleneck). Now the chunk loads its head's S
  // into SHARED once (f16, 32KB = full SLM), runs the sequential recurrence with S in SLM,
  // and writes S back once → S global traffic drops ~T×. Thread vi owns column vi of S end-
  // to-end (no other thread touches it) → ZERO barriers in the token loop. kv/out accumulate
  // in f32; only the stored state is f16 (the per-head decay expg<1 forgets old tokens, so
  // f16 rounding doesn't accumulate unboundedly). k/q read from global (128-way broadcast).
  const DELTA_WGSL = `
enable f16;
struct P { nHeads:u32, dim:u32, T:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       q  : array<f32>;
@group(0) @binding(1) var<storage, read>       k  : array<f32>;
@group(0) @binding(2) var<storage, read>       v  : array<f32>;
@group(0) @binding(3) var<storage, read>       gb : array<f32>;
@group(0) @binding(4) var<storage, read_write> S  : array<f32>;   // [nHeads*dim*dim] [head][key][val]
@group(0) @binding(5) var<storage, read_write> outv : array<f32>;
@group(0) @binding(6) var<uniform>             p  : P;
var<workgroup> Ss : array<f16, ${DELTA_DIM * DELTA_DIM}>;   // [key][val] f16 (32KB = full SLM)
@compute @workgroup_size(${DELTA_DIM},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let h = wg.x; let vi = lid.x; let dim = p.dim;
  let gbase = h*dim*dim + vi;            // global S column vi for this head
  for (var kk:u32=0u; kk<dim; kk=kk+1u) { Ss[kk*dim+vi] = f16(S[gbase + kk*dim]); }   // load column → SLM
  for (var t:u32=0u; t<p.T; t=t+1u) {    // no barriers: thread vi owns column vi exclusively
    let th = t*p.nHeads + h; let base = th*dim;
    let expg = f16(gb[th*2u]); let beta = gb[th*2u + 1u];
    var kv : f32 = 0.0;
    for (var kk:u32=0u; kk<dim; kk=kk+1u) { let idx = kk*dim+vi; let s = Ss[idx]*expg; Ss[idx] = s; kv = kv + f32(s)*k[base+kk]; }
    let delta = (v[base+vi] - kv) * beta;
    var o : f32 = 0.0;
    for (var kk:u32=0u; kk<dim; kk=kk+1u) { let idx = kk*dim+vi; let s = Ss[idx] + f16(k[base+kk]*delta); Ss[idx] = s; o = o + f32(s)*q[base+kk]; }
    outv[base+vi] = o;
  }
  for (var kk:u32=0u; kk<dim; kk=kk+1u) { S[gbase + kk*dim] = f32(Ss[kk*dim+vi]); }   // write column back
}`;
  function deltaRecur(qBuf, kBuf, vBuf, gbBuf, SBuf, outBuf, nHeads, dim, T) {
    const pipe = E.getPipeline('q35.delta', DELTA_WGSL);
    const p = uniform(new Uint32Array([nHeads, dim, T || 1, 0]));
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, gbBuf, SBuf, outBuf, p], [nHeads, 1, 1]);
  }

  // ============================================================
  // CAUSAL DEPTHWISE conv1d (kernel=4) + SiLU, decode step (1 token).
  // Mixed q|k|v are conv'd along the sequence before the recurrence.
  // PyTorch causal Conv1d (left-pad K-1, groups=conv_dim) at the last pos:
  //   out[c] = silu( Σ_{j=0..K-1} w[c][j] * window[j] (+ bias[c]) )
  //   window = [x_{t-3}, x_{t-2}, x_{t-1}, x_t]   (newest last)
  // State = last K-1 inputs per channel, kept across decode steps (per layer);
  // this kernel ALSO advances the state (shift left, append x_t). One thread/ch.
  // ============================================================
  const CONV_K = 4;
  const CONV_WGSL = `
struct P { convDim:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       x     : array<f32>;   // [convDim]   current input x_t
@group(0) @binding(1) var<storage, read>       w     : array<f32>;   // [convDim*K] weights, row-major [c][j], j newest last
@group(0) @binding(2) var<storage, read>       bias  : array<f32>;   // [convDim]   (zeros if no bias)
@group(0) @binding(3) var<storage, read_write> state : array<f32>;   // [convDim*(K-1)] last K-1 inputs, oldest first
@group(0) @binding(4) var<storage, read_write> outv  : array<f32>;   // [convDim]
@group(0) @binding(5) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let c = gid.x; if (c >= p.convDim) { return; }
  let K = ${CONV_K}u; let Km1 = K - 1u;
  let sbase = c*Km1; let wbase = c*K;
  // window[0..K-2] = state (oldest..newest), window[K-1] = x_t
  var acc : f32 = bias[c];
  for (var j:u32=0u; j<Km1; j=j+1u) { acc = acc + w[wbase+j]*state[sbase+j]; }
  let xt = x[c];
  acc = acc + w[wbase+Km1]*xt;
  // SiLU
  outv[c] = acc / (1.0 + exp(-acc));
  // advance state: shift left, append x_t
  for (var j:u32=0u; j<Km1-1u; j=j+1u) { state[sbase+j] = state[sbase+j+1u]; }
  state[sbase+Km1-1u] = xt;
}`;
  function conv1dDecode(xBuf, wBuf, biasBuf, stateBuf, outBuf, convDim) {
    const pipe = E.getPipeline('q35.conv', CONV_WGSL);
    const p = uniform(new Uint32Array([convDim, 0, 0, 0]));
    return E.dispatch(pipe, [xBuf, wBuf, biasBuf, stateBuf, outBuf, p], [Math.ceil(convDim / 64), 1, 1]);
  }
  // BATCHED causal conv over T tokens (prefill): one thread per channel walks t=0..T-1
  // with a sliding window seeded from `state` (the K-1 pre-chunk inputs), and writes the
  // final K-1 inputs back to `state` so the following decode continues seamlessly.
  const CONV_PREFILL_WGSL = `
struct P { convDim:u32, T:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       x     : array<f32>;   // [T*convDim] inputs ([t][c])
@group(0) @binding(1) var<storage, read>       w     : array<f32>;   // [convDim*K]
@group(0) @binding(2) var<storage, read>       bias  : array<f32>;   // [convDim]
@group(0) @binding(3) var<storage, read_write> state : array<f32>;   // [convDim*(K-1)] oldest first
@group(0) @binding(4) var<storage, read_write> outv  : array<f32>;   // [T*convDim]
@group(0) @binding(5) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let c = gid.x; if (c >= p.convDim) { return; }
  let K = ${CONV_K}u; let Km1 = K - 1u; let sbase = c*Km1; let wbase = c*K;
  var w0 = state[sbase]; var w1 = state[sbase+1u]; var w2 = state[sbase+2u];   // K-1==3, oldest..newest
  for (var t:u32=0u; t<p.T; t=t+1u) {
    let xt = x[t*p.convDim + c];
    let acc = bias[c] + w[wbase]*w0 + w[wbase+1u]*w1 + w[wbase+2u]*w2 + w[wbase+3u]*xt;
    outv[t*p.convDim + c] = acc / (1.0 + exp(-acc));   // SiLU
    w0 = w1; w1 = w2; w2 = xt;                          // shift window
  }
  state[sbase] = w0; state[sbase+1u] = w1; state[sbase+2u] = w2;   // carry for next chunk/decode
}`;
  function conv1dPrefill(xBuf, wBuf, biasBuf, stateBuf, outBuf, convDim, T) {
    const pipe = E.getPipeline('q35.convpf', CONV_PREFILL_WGSL);
    const p = uniform(new Uint32Array([convDim, T, 0, 0]));
    return E.dispatch(pipe, [xBuf, wBuf, biasBuf, stateBuf, outBuf, p], [Math.ceil(convDim / 64), 1, 1]);
  }

  // ============================================================
  // Per-head L2 normalize: out[h][i] = x[h][i] / sqrt(Σ_i x[h][i]^2 + eps).
  // q and k are L2-normed per head before the DeltaNet recurrence.
  // One workgroup per head, ${DELTA_DIM} threads, shared-mem sum reduction.
  // ============================================================
  // T-aware: wg.x = head, wg.y = token; buffers laid out [token][head][dim]. T=1 → identical.
  const L2_WGSL = `
struct P { nHeads:u32, dim:u32, eps:f32, scale:f32 };
@group(0) @binding(0) var<storage, read>       x   : array<f32>;
@group(0) @binding(1) var<storage, read_write> outv: array<f32>;
@group(0) @binding(2) var<uniform>             p   : P;
var<workgroup> red : array<f32, ${DELTA_DIM}>;
@compute @workgroup_size(${DELTA_DIM},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let h = wg.x; let i = lid.x; let dim = p.dim; let base = (wg.y*p.nHeads + h)*dim;
  let v = select(0.0, x[base+i], i < dim);
  red[i] = v*v;
  workgroupBarrier();
  var stride = ${DELTA_DIM}u / 2u;
  loop { if (stride == 0u) { break; }
    if (i < stride) { red[i] = red[i] + red[i+stride]; }
    workgroupBarrier(); stride = stride / 2u; }
  if (i >= dim) { return; }
  let inv = (1.0 / sqrt(red[0] + p.eps)) * p.scale;
  outv[base+i] = x[base+i] * inv;
}`;
  function l2normHeads(inBuf, outBuf, nHeads, dim, eps, scale, T) {
    const pipe = E.getPipeline('q35.l2', L2_WGSL);
    const u = new Uint32Array(4); const du = new DataView(u.buffer);
    du.setUint32(0, nHeads, true); du.setUint32(4, dim, true);
    du.setFloat32(8, eps, true); du.setFloat32(12, scale == null ? 1.0 : scale, true);
    const p = uniform(u);
    return E.dispatch(pipe, [inBuf, outBuf, p], [nHeads, T || 1, 1]);
  }

  // ============================================================
  // g/beta precompute (per v-head). HF Qwen3NextGatedDeltaNet:
  //   beta = sigmoid(b)
  //   g    = -exp(A_log) * softplus(a + dt_bias)   ;  state decay = exp(g)
  // Output gb[h*2]=expg=exp(g), gb[h*2+1]=beta. One thread per head.
  // softplus(x) = log(1+exp(x)), numerically stable via max(x,0)+log1p(exp(-|x|)).
  // ============================================================
    // T-aware: gid.y = token. a,b are per-token [T,nHeads]; A_log,dt_bias per-head [nHeads];
  // gb out [T, nHeads*2]. T=1 → identical.
const GBETA_WGSL = `
struct P { nHeads:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       a      : array<f32>;   // [T*nHeads] in_proj_a
@group(0) @binding(1) var<storage, read>       b      : array<f32>;   // [T*nHeads] in_proj_b
@group(0) @binding(2) var<storage, read>       A_log  : array<f32>;   // [nHeads]
@group(0) @binding(3) var<storage, read>       dt_bias: array<f32>;   // [nHeads]
@group(0) @binding(4) var<storage, read_write> gb     : array<f32>;   // [T*nHeads*2]
@group(0) @binding(5) var<uniform>             p      : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let h = gid.x; if (h >= p.nHeads) { return; }
  let th = gid.y*p.nHeads + h;
  let beta = 1.0 / (1.0 + exp(-b[th]));
  let z = a[th] + dt_bias[h];
  let sp = max(z, 0.0) + log(1.0 + exp(-abs(z)));   // softplus, stable
  let g = -exp(A_log[h]) * sp;
  gb[th*2u]      = exp(g);
  gb[th*2u + 1u] = beta;
}`;
  function gbeta(aBuf, bBuf, ALogBuf, dtBiasBuf, gbBuf, nHeads, T) {
    const pipe = E.getPipeline('q35.gbeta', GBETA_WGSL);
    const p = uniform(new Uint32Array([nHeads, 0, 0, 0]));
    return E.dispatch(pipe, [aBuf, bBuf, ALogBuf, dtBiasBuf, gbBuf, p], [Math.ceil(nHeads / 64), T || 1, 1]);
  }

  // ============================================================
  // Gated RMSNorm (Qwen3NextRMSNormGated), per-head over value head_dim.
  // EXACT HF order: variance from the UN-gated x, normalize, weight, THEN silu(z):
  //   v = mean(x²) ; out = x * rsqrt(v+eps) * weight * silu(z)
  // weight is [dim], shared across heads. 1 workgroup/head, ${DELTA_DIM} threads.
  // ============================================================
  const GRMS_WGSL = `
enable f16;
struct P { nHeads:u32, dim:u32, eps:f32, _a:u32 };
@group(0) @binding(0) var<storage, read>       x   : array<f32>;   // [nHeads*dim]
@group(0) @binding(1) var<storage, read>       z   : array<f32>;   // [nHeads*dim] gate
@group(0) @binding(2) var<storage, read>       w   : array<f16>;   // [dim] shared weight
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;   // [nHeads*dim]
@group(0) @binding(4) var<uniform>             p   : P;
var<workgroup> red : array<f32, ${DELTA_DIM}>;
@compute @workgroup_size(${DELTA_DIM},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let h = wg.x; let i = lid.x; let dim = p.dim; let base = (wg.y*p.nHeads + h)*dim;   // wg.y = token
  let xv = select(0.0, x[base+i], i < dim);
  red[i] = xv*xv;                       // variance from UN-gated x
  workgroupBarrier();
  var stride = ${DELTA_DIM}u / 2u;
  loop { if (stride == 0u) { break; }
    if (i < stride) { red[i] = red[i] + red[i+stride]; }
    workgroupBarrier(); stride = stride / 2u; }
  if (i >= dim) { return; }
  let inv = inverseSqrt(red[0] / f32(dim) + p.eps);
  let zg = z[base+i]; let sz = zg / (1.0 + exp(-zg));   // silu(z), applied LAST
  outv[base+i] = xv * inv * f32(w[i]) * sz;
}`;
  function gatedRMSNorm(xBuf, zBuf, wBuf, outBuf, nHeads, dim, eps, T) {
    const pipe = E.getPipeline('q35.grms', GRMS_WGSL);
    const u = new Uint32Array(4); const du = new DataView(u.buffer);
    du.setUint32(0, nHeads, true); du.setUint32(4, dim, true); du.setFloat32(8, eps, true);
    const p = uniform(u);
    return E.dispatch(pipe, [xBuf, zBuf, wBuf, outBuf, p], [nHeads, T || 1, 1]);
  }

  // ---- INT4 batched matvec (T tokens, weight row unpacked once, reused) ----
  const MATVECQ_WGSL = `
enable f16;
struct D { T:u32, N:u32, K:u32, tBase:u32, acc:u32, _p0:u32, _p1:u32, _p2:u32 };  // acc=1 → y += result
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;   // [(tBase+t)*K/4 + ...]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;          // [Tfull*N]
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${MATVEC_MAXT * MATVEC_WG}>;
@compute @workgroup_size(${MATVEC_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>) {
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
      let base = (d.tBase + t)*K4;
      acc[t] = acc[t] + s*( dot(wa, x[base + 2u*w]) + dot(wb_, x[base + 2u*w + 1u]) );
    }
    w = w + ${MATVEC_WG}u;
  }
  for (var t:u32=0u; t<T; t=t+1u) { part[t*${MATVEC_WG}u + lid.x] = acc[t]; }
  workgroupBarrier();
  var stride = ${MATVEC_WG}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){ for(var t:u32=0u;t<T;t=t+1u){ part[t*${MATVEC_WG}u+lid.x]=part[t*${MATVEC_WG}u+lid.x]+part[t*${MATVEC_WG}u+lid.x+stride]; } } workgroupBarrier(); stride=stride/2u; }
  if (lid.x < T) {
    let t = lid.x;
    let idx = (d.tBase + t)*d.N + n;
    y[idx] = select(0.0, y[idx], d.acc != 0u) + part[t*${MATVEC_WG}u];
  }
}`;
  function matvecQ(xBuf, packBuf, scBuf, yBuf, T, N, K, tBase, acc) {
    const pipe = E.getPipeline('q3.matvecQ', MATVECQ_WGSL);
    const d = uniform(new Uint32Array([T, N, K, tBase || 0, acc ? 1 : 0, 0, 0, 0]));
    const gx = Math.min(N, 65535), gy = Math.ceil(N / gx);
    return E.dispatch(pipe, [xBuf, packBuf, scBuf, yBuf, d], [gx, gy, 1]);
  }

  // ---- REGISTER-BLOCKED + SHARED-MEM int4 GEMM (prefill) — Y[T,N] = X[T,K]·dq(W)ᵀ ----
  // A BM×BN output tile per workgroup, BK-deep. Each k-step stages a BM-token × BK and a
  // BN-row × BK tile into shared memory (weights dequantized ONCE on load → no redundant
  // global loads), then each thread accumulates a TM×TN output block FROM the shared tiles
  // in registers. BK=QGROUP=32 → only K/32 barriers (the BK=8 register-block attempt died on
  // K/8 barriers) AND one scale per row per k-tile. Reading TM+TN shared values for TM*TN
  // FMAs = ~2 FLOP/shared-read (vs 0.5 for one-output) → higher FLOP efficiency. Codegen-
  // unrolled for static registers. (BM*BN/(TM*TN) threads.)
  // vec4 along K: shared tiles are vec4<f32> (16-byte loads), inner loop uses dot() so each
  // step is 4 FMAs/instruction over BK4=BK/4 vec4s (vs BK scalar steps). X read as vec4.
  const GEMMQ_BM = 64, GEMMQ_BN = 64, GEMMQ_BK = QGROUP, GEMMQ_TM = 4, GEMMQ_TN = 4;
  const gemmqWgsl = (() => {
    const BM = GEMMQ_BM, BN = GEMMQ_BN, BK = GEMMQ_BK, TM = GEMMQ_TM, TN = GEMMQ_TN, BK4 = BK / 4;
    const NTH = (BM / TM) * (BN / TN), TILEA4 = BM * BK4, TILEB4 = BN * BK4, RN = BN / TN;
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var<storage, read>       X  : array<vec4<f32>>;   // [T*K/4]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;          // [N*K/8] packed nibbles (q+8)
@group(0) @binding(2) var<storage, read>       sc : array<f16>;          // [N*K/${QGROUP}]
@group(0) @binding(3) var<storage, read_write> Y  : array<f32>;          // [T*N]
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> As : array<vec4<f16>, ${TILEA4}>;   // [BM][BK4] (f16: half SLM bytes/occupancy)
var<workgroup> Bs : array<vec4<f16>, ${TILEB4}>;   // [BN][BK4] dequantized
@compute @workgroup_size(${NTH}, 1, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let lx = lid.x; let tN = lx % ${RN}u; let tM = lx / ${RN}u;
  let mBase = wg.y*${BM}u; let nBase = wg.x*${BN}u;
  let WPR = d.K/8u; let gpr = d.K/${QGROUP}u; let K4 = d.K/4u; let nTiles = (d.K + ${BK}u - 1u)/${BK}u;
`;
    for (let r = 0; r < TM * TN; r++) s += `  var acc${r}:f32=0.0;\n`;
    s += `  for (var kt:u32=0u; kt<nTiles; kt=kt+1u) {
    let k0 = kt*${BK}u;
    for (var r:u32=0u; r<${TILEA4 / NTH}u; r=r+1u) {
      let idx = lx + r*${NTH}u; let lt = idx/${BK4}u; let kk4 = idx%${BK4}u; let gt = mBase+lt;
      As[idx] = select(vec4<f16>(0.0), vec4<f16>(X[gt*K4 + k0/4u + kk4]), gt<d.T);
    }
    for (var r:u32=0u; r<${TILEB4 / NTH}u; r=r+1u) {
      let idx = lx + r*${NTH}u; let ln = idx/${BK4}u; let kk4 = idx%${BK4}u; let gn = nBase+ln;
      var v:vec4<f32> = vec4<f32>(0.0);
      if (gn<d.N) {
        let word = W[gn*WPR + k0/8u + kk4/2u];
        let lo = vec4<f32>(unpack4xU8(word & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
        let hi = vec4<f32>(unpack4xU8((word >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
        let sv = f32(sc[gn*gpr + kt]);
        if ((kk4 & 1u) == 0u) { v = vec4<f32>(lo.x,hi.x,lo.y,hi.y) * sv; } else { v = vec4<f32>(lo.z,hi.z,lo.w,hi.w) * sv; }
      }
      Bs[idx] = vec4<f16>(v);
    }
    workgroupBarrier();
    for (var kk4:u32=0u; kk4<${BK4}u; kk4=kk4+1u) {
`;
    for (let i = 0; i < TM; i++) s += `      let a${i} = As[(tM*${TM}u + ${i}u)*${BK4}u + kk4];\n`;
    for (let j = 0; j < TN; j++) s += `      let b${j} = Bs[(tN*${TN}u + ${j}u)*${BK4}u + kk4];\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `      acc${i * TN + j} = acc${i * TN + j} + dot(vec4<f32>(a${i}), vec4<f32>(b${j}));\n`;
    s += `    }
    workgroupBarrier();
  }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  { let gm=mBase+tM*${TM}u+${i}u; let gn=nBase+tN*${TN}u+${j}u; if (gm<d.T && gn<d.N) { let idx=gm*d.N+gn; Y[idx]=select(0.0,Y[idx],d.acc!=0u)+acc${i * TN + j}; } }\n`;
    s += `}`;
    return s;
  })();
  function gemmQ(xBuf, wrec, yBuf, T, N, K, acc) {
    const pipe = E.getPipeline('q35.gemmQ', gemmqWgsl);
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    return E.dispatch(pipe, [xBuf, wrec.pack, wrec.scales, yBuf, d], [Math.ceil(N / GEMMQ_BN), Math.ceil(T / GEMMQ_BM), 1]);
  }

  // Router for the int4 weight path. wrec = { pack, scales, N, K }. acc=true →
  // y += result (fused residual add, saves a separate addInPlace pass).
  async function linearQ(xBuf, wrec, yBuf, T, N, K, acc) {
    if (T === 1) return gemvQ(xBuf, wrec.pack, wrec.scales, yBuf, N, K, acc);
    return gemmQ(xBuf, wrec, yBuf, T, N, K, acc);   // batched prefill → one tiled-GEMM dispatch
  }

  // (f16 path below — gemv/matvecT/linearT — retained for the f16 self-tests; the
  // live forward now uses the int4 linearQ for all projection + lm_head matrices.)
  function linear(xBuf, wBuf, yBuf, T, N, K) {
    if (T === 1) return gemv(xBuf, wBuf, yBuf, N, K);
    if (T <= MATVEC_MAXT) return matvecT(xBuf, wBuf, yBuf, T, N, K);
    return linearT(xBuf, wBuf, yBuf, T, N, K);
  }

  // ============================================================
  // Kernel 3 — Embedding gather.  y[T,H] = embed[ids[t], :]
  // ============================================================
  const EMBED_WGSL = `
enable f16;
struct P { T:u32, H:u32, idOff:u32, _b:u32 };   // reads ids[idOff + t] (decode: idOff=posBase into the GPU token history)
@group(0) @binding(0) var<storage, read>       ids   : array<u32>;
@group(0) @binding(1) var<storage, read>       embed : array<f16>;
@group(0) @binding(2) var<storage, read_write> y     : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; let total=p.T*p.H; if(idx>=total){return;}
  let t=idx/p.H; let h=idx%p.H;
  y[idx]=f32(embed[ids[p.idOff + t]*p.H + h]);
}`;
  function embedGather(idsBuf, embedBuf, yBuf, T, H, idOff) {
    const pipe = E.getPipeline('q3.embed', EMBED_WGSL);
    const p = uniform(new Uint32Array([T, H, idOff || 0, 0]));
    return E.dispatch(pipe, [idsBuf, embedBuf, yBuf, p], [Math.ceil((T*H)/64), 1, 1]);
  }
  // int4 embedding gather — dequantizes row ids[idOff+t] from the tied int4 embed
  // (so we don't keep a separate f16 embed copy; the same int4 weight also feeds
  // the lm_head). Saves ~0.5GB (0.8B) / ~1GB (2B) of GPU memory.
  const EMBEDQ_WGSL = `
enable f16;
struct P { T:u32, H:u32, idOff:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       ids : array<u32>;
@group(0) @binding(1) var<storage, read>       W   : array<u32>;     // [vocab*H/8] packed nibbles
@group(0) @binding(2) var<storage, read>       sc  : array<f16>;     // [vocab*H/QGROUP] scales
@group(0) @binding(3) var<storage, read_write> y   : array<f32>;
@group(0) @binding(4) var<uniform>             p   : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; let total=p.T*p.H; if(idx>=total){return;}
  let t=idx/p.H; let h=idx%p.H;
  let id=ids[p.idOff + t];
  let words=p.H/8u; let gpr=p.H/${QGROUP}u;
  let word=W[id*words + h/8u];
  let nib=(word >> (4u*(h%8u))) & 0xFu;
  let s=f32(sc[id*gpr + h/${QGROUP}u]);
  y[idx]=(f32(nib)-8.0)*s;
}`;
  function embedGatherQ(idsBuf, packBuf, scBuf, yBuf, T, H, idOff) {
    const pipe = E.getPipeline('q35.embedQ', EMBEDQ_WGSL);
    const p = uniform(new Uint32Array([T, H, idOff || 0, 0]));
    return E.dispatch(pipe, [idsBuf, packBuf, scBuf, yBuf, p], [Math.ceil((T*H)/64), 1, 1]);
  }

  // ============================================================
  // Kernel 4 — RoPE + per-head QK-norm (fused), for one tensor (q or k).
  // in[T, nH*hd] → out[T, nH*hd]. Per (t,head): RMSNorm over hd * normW[hd],
  // then RoPE (full rotary, rotate_half) at absolute position posBase+t.
  // One workgroup per (t,head); hd threads.
  // ============================================================
  const ROPEQK_WGSL = `
enable f16;
struct P { T:u32, nH:u32, hd:u32, posBase:u32, theta:f32, eps:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read>       normW: array<f16>;
@group(0) @binding(2) var<storage, read_write> out  : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
var<workgroup> red : array<f32, 128>;
var<workgroup> nrm : array<f32, 128>;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let hd=p.hd; let j=lid.x;
  let unit=wg.x;            // 0 .. T*nH-1
  let t=unit/p.nH; let head=unit%p.nH;
  let base=t*(p.nH*hd) + head*hd;
  // RMSNorm over hd
  var v:f32=0.0; if(j<hd){ v=inp[base+j]; }
  red[j]=select(0.0, v*v, j<hd); workgroupBarrier();
  var stride=64u;
  loop{ if(stride==0u){break;} if(j<stride){red[j]=red[j]+red[j+stride];} workgroupBarrier(); stride=stride/2u; }
  let inv=inverseSqrt(red[0]/f32(hd)+p.eps);
  if(j<hd){ nrm[j]=v*inv*f32(normW[j]); }
  workgroupBarrier();
  if(j>=hd){ return; }
  // RoPE rotate_half: pair j with j±hd/2
  let half=hd/2u;
  let pos=f32(p.posBase + t);
  let freqIdx = select(j-half, j, j<half);              // index into [0,half)
  let invFreq = pow(p.theta, -2.0*f32(freqIdx)/f32(hd));
  let ang = pos*invFreq;
  let c=cos(ang); let s=sin(ang);
  let xj=nrm[j];
  let partner = select(nrm[j-half], nrm[j+half], j<half);
  // j<half: out = x*cos - partner*sin ; j>=half: out = x*cos + partner*sin
  let rot = select(partner, -partner, j<half);
  out[base+j] = xj*c + rot*s;
}`;
  function ropeQK(inBuf, normWBuf, outBuf, T, nH, hd, posBase, theta, eps) {
    // outBuf MUST differ from inBuf — aliasing read + read_write bindings to one
    // buffer is WebGPU UB (miscompiles on Intel: produces wrong values silently).
    if (inBuf === outBuf) throw new Error('ropeQK: in-place not allowed (use a separate output buffer)');
    const pipe = E.getPipeline('q3.ropeqk', ROPEQK_WGSL);
    const u = new Uint32Array(8); const dv = new DataView(u.buffer);
    dv.setUint32(0,T,true); dv.setUint32(4,nH,true); dv.setUint32(8,hd,true);
    dv.setUint32(12,posBase,true); dv.setFloat32(16,theta,true); dv.setFloat32(20,eps,true);
    const p = uniform(u);   // pooled (stable buffer for bind-group cache)
    return E.dispatch(pipe, [inBuf, normWBuf, outBuf, p], [T*nH, 1, 1]);
  }

  // ============================================================
  // Qwen3.5 full-attention RoPE + per-head QK-norm (head_dim 256, PARTIAL rotary).
  // - per-head RMSNorm over the full hd (256), * normW[hd]
  // - rotate only the first rotDim (64) dims (NeoX half-split within the 64-block);
  //   dims [rotDim,hd) pass through. theta=1e7. (mRoPE reduces to standard RoPE for
  //   text-only since all 3 position components equal the text position.)
  // - input head stride `inStride` lets q read from the gated q_proj output
  //   ([nH, 2*hd] interleaved query|gate → take the first hd) while k uses inStride=hd.
  // One workgroup per (t,head), WG_HD threads (= hd).
  // ============================================================
  const WG_HD = 256;
  const ROPEQK35_WGSL = `
enable f16;
struct P { T:u32, nH:u32, hd:u32, inStride:u32, rotDim:u32, posBase:u32, theta:f32, eps:f32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read>       normW: array<f16>;
@group(0) @binding(2) var<storage, read_write> out  : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
var<workgroup> red : array<f32, ${WG_HD}>;
var<workgroup> nrm : array<f32, ${WG_HD}>;
@compute @workgroup_size(${WG_HD},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let hd=p.hd; let j=lid.x;
  let unit=wg.x; let t=unit/p.nH; let head=unit%p.nH;
  let baseIn  = t*(p.nH*p.inStride) + head*p.inStride;
  let baseOut = t*(p.nH*hd) + head*hd;
  var v:f32=0.0; if(j<hd){ v=inp[baseIn+j]; }
  red[j]=select(0.0, v*v, j<hd); workgroupBarrier();
  var stride=${WG_HD}u/2u;
  loop{ if(stride==0u){break;} if(j<stride){red[j]=red[j]+red[j+stride];} workgroupBarrier(); stride=stride/2u; }
  let inv=inverseSqrt(red[0]/f32(hd)+p.eps);
  if(j<hd){ nrm[j]=v*inv*(1.0 + f32(normW[j])); }   // Qwen3.5 q/k-norm: *(1+weight)
  workgroupBarrier();
  if(j>=hd){ return; }
  let rotDim=p.rotDim;
  if(j>=rotDim){ out[baseOut+j]=nrm[j]; return; }   // pass-through (no rotation)
  let half=rotDim/2u;
  let pos=f32(p.posBase+t);
  let freqIdx = select(j-half, j, j<half);
  let invFreq = pow(p.theta, -2.0*f32(freqIdx)/f32(rotDim));
  let ang=pos*invFreq; let c=cos(ang); let s=sin(ang);
  let xj=nrm[j];
  let partner = select(nrm[j-half], nrm[j+half], j<half);
  let rot = select(partner, -partner, j<half);
  out[baseOut+j] = xj*c + rot*s;
}`;
  function ropeQKNorm35(inBuf, normWBuf, outBuf, T, nH, hd, inStride, rotDim, posBase, theta, eps) {
    if (inBuf === outBuf) throw new Error('ropeQKNorm35: in-place not allowed');
    const pipe = E.getPipeline('q35.ropeqk', ROPEQK35_WGSL);
    const u = new Uint32Array(8); const dv = new DataView(u.buffer);
    dv.setUint32(0,T,true); dv.setUint32(4,nH,true); dv.setUint32(8,hd,true); dv.setUint32(12,inStride,true);
    dv.setUint32(16,rotDim,true); dv.setUint32(20,posBase,true); dv.setFloat32(24,theta,true); dv.setFloat32(28,eps,true);
    const p = uniform(u);
    return E.dispatch(pipe, [inBuf, normWBuf, outBuf, p], [T*nH, 1, 1]);
  }

  // ============================================================
  // Qwen3.5 attention output gate. q_proj output is [nH, 2*hd] = query|gate
  // interleaved per head; after attention, out *= sigmoid(gate). This kernel reads
  // the gate half straight from qproj (stride 2*hd, offset hd) and applies it.
  //   gated[h*hd+d] = attn[h*hd+d] * sigmoid(qproj[h*2*hd + hd + d])
  // ============================================================
    // T-aware: index covers [T*nH*hd]; attn/out laid [T,nH*hd], qproj laid [T,nH*2*hd].
const GATE35_WGSL = `
struct P { nH:u32, hd:u32, T:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       attn  : array<f32>;   // [T*nH*hd]
@group(0) @binding(1) var<storage, read>       qproj : array<f32>;   // [T*nH*2*hd]
@group(0) @binding(2) var<storage, read_write> outv  : array<f32>;   // [T*nH*hd]
@group(0) @binding(3) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x; let per=p.nH*p.hd; let total=p.T*per; if(i>=total){return;}
  let t=i/per; let r=i%per; let h=r/p.hd; let d=r%p.hd;
  let g=qproj[t*p.nH*2u*p.hd + h*2u*p.hd + p.hd + d];
  outv[i] = attn[i] * (1.0/(1.0+exp(-g)));
}`;
  function applyGate35(attnBuf, qprojBuf, outBuf, nH, hd, T) {
    const pipe = E.getPipeline('q35.gate', GATE35_WGSL);
    const p = uniform(new Uint32Array([nH, hd, T || 1, 0]));
    return E.dispatch(pipe, [attnBuf, qprojBuf, outBuf, p], [Math.ceil((T||1)*nH*hd/64), 1, 1]);
  }

  // ============================================================
  // Kernel 5 — GQA causal attention.
  // Q[T, nHq*hd], K[S, nKv*hd], V[S, nKv*hd] → O[T, nHq*hd].
  // q-head h uses kv-head h/(nHq/nKv). Causal: key s attends iff s <= (S-T)+t
  // (so decode with T=1,S=cacheLen attends all; prefill T=S is lower-triangular).
  //
  // PARALLEL design (the old one ran 1 THREAD per (t,head) — 16 threads for decode,
  // ~2% iGPU occupancy, 57% of decode time). Now: ONE WORKGROUP per (t,head) with
  // ATTN_WG threads cooperating —
  //   • load the query row into shared memory
  //   • scores: threads stride the keys, each computes dot(q,k_s)*scale → shared sc[]
  //   • softmax: parallel max-reduce, then exp + parallel sum-reduce
  //   • output: hd threads (one per dim) each sum over keys → O[d]
  // Scores live in shared mem (capacity ATTN_MAXK = MAX_SEQ). hd ≤ ATTN_WG.
  // ============================================================
  const ATTN_WG = 256;      // ≥ head_dim (Qwen3.5 full-attn hd=256); one thread per output dim
  const ATTN_MAXK = 4096;   // = MAX_SEQ; scores buffer size in shared memory (16KB f32, < 32KB SLM)
  const ATTN_WGSL = `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<f32>;
@group(0) @binding(1) var<storage, read>       K : array<f32>;
@group(0) @binding(2) var<storage, read>       V : array<f32>;
@group(0) @binding(3) var<storage, read_write> O : array<f32>;
@group(0) @binding(4) var<uniform>             p : P;
var<workgroup> qsh : array<f32, ${ATTN_WG}>;   // query row (hd ≤ ATTN_WG)
var<workgroup> sc  : array<f32, ${ATTN_MAXK}>; // scores / probabilities per key
var<workgroup> red : array<f32, ${ATTN_WG}>;   // reduction scratch
@compute @workgroup_size(${ATTN_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>) {
  let lid = lidv.x;
  let unit = wg.x;                       // 0 .. T*nHq-1
  let t = unit / p.nHq; let hq = unit % p.nHq;
  let hd = p.hd; let grp = p.nHq / p.nKv; let hk = hq / grp;
  let qb = t*(p.nHq*hd) + hq*hd;
  let kvstride = p.nKv*hd;
  let scale = 1.0/sqrt(f32(hd));
  let last = (p.S - p.T) + t;            // inclusive last key
  // load query row
  if (lid < hd) { qsh[lid] = Q[qb+lid]; }
  workgroupBarrier();
  // scores: each thread handles a strided subset of keys
  var key = lid;
  loop {
    if (key > last) { break; }
    let kb = key*kvstride + hk*hd;
    var dot : f32 = 0.0;
    for (var i:u32=0u; i<hd; i=i+1u) { dot = dot + qsh[i]*K[kb+i]; }
    sc[key] = dot*scale;
    key = key + ${ATTN_WG}u;
  }
  workgroupBarrier();
  // max-reduce over sc[0..last]
  var lmax : f32 = -3.0e38;
  key = lid; loop { if (key > last) { break; } lmax = max(lmax, sc[key]); key = key + ${ATTN_WG}u; }
  red[lid] = lmax; workgroupBarrier();
  var st = ${ATTN_WG}u/2u;
  loop { if (st==0u){break;} if (lid<st){ red[lid]=max(red[lid],red[lid+st]); } workgroupBarrier(); st=st/2u; }
  let m = red[0]; workgroupBarrier();
  // exp + sum-reduce
  var lsum : f32 = 0.0;
  key = lid; loop { if (key > last) { break; } let e = exp(sc[key]-m); sc[key]=e; lsum=lsum+e; key=key+${ATTN_WG}u; }
  red[lid] = lsum; workgroupBarrier();
  st = ${ATTN_WG}u/2u;
  loop { if (st==0u){break;} if (lid<st){ red[lid]=red[lid]+red[lid+st]; } workgroupBarrier(); st=st/2u; }
  let denom = red[0]; workgroupBarrier();
  // output: one thread per dim sums prob*V over all keys
  if (lid < hd) {
    var acc : f32 = 0.0;
    for (var s:u32=0u; s<=last; s=s+1u) { acc = acc + sc[s]*V[s*kvstride + hk*hd + lid]; }
    O[qb+lid] = acc/denom;
  }
}`;
  function attention(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd) {
    const pipe = E.getPipeline('q3.attn', ATTN_WGSL);
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [T * nHq, 1, 1]);   // one workgroup per (t,head)
  }

  // ============================================================
  // Kernel 6 — SwiGLU.  y[T,I] = silu(gate[T,I]) * up[T,I]
  // ============================================================
  const SWIGLU_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       gate : array<f32>;
@group(0) @binding(1) var<storage, read>       up   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y    : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x; if(i>=p.n){return;}
  let g=gate[i]; let silu=g/(1.0+exp(-g)); y[i]=silu*up[i];
}`;
  function swiglu(gateBuf, upBuf, yBuf, n) {
    const pipe = E.getPipeline('q3.swiglu', SWIGLU_WGSL);
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    return E.dispatch(pipe, [gateBuf, upBuf, yBuf, p], [Math.ceil(n/64), 1, 1]);
  }


  // ============================================================
  // Kernel 7 — Residual add (in place).  a[n] += b[n]
  // ============================================================
  const ADD_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read_write> a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){ let i=gid.x; if(i>=p.n){return;} a[i]=a[i]+b[i]; }`;
  function addInPlace(aBuf, bBuf, n) {
    const pipe = E.getPipeline('q3.add', ADD_WGSL);
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    return E.dispatch(pipe, [aBuf, bBuf, p], [Math.ceil(n/64), 1, 1]);
  }

  // ============================================================
  // Kernel 8 — ArgMax over logits[N] → out[0] = index of max (greedy sampling).
  // Single workgroup, WG threads stride N tracking (max,idx), then a shared-mem
  // reduction. Lets us read back 4 bytes/token instead of the full 600KB logits.
  // Tie-break differs trivially from JS (rare; irrelevant to coherence).
  // ============================================================
  const ARGMAX_WG = 256;
  const ARGMAX_WGSL = `
struct P { n:u32, outPos:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> outIdx : array<u32>;
@group(0) @binding(2) var<uniform>             p      : P;
var<workgroup> sv : array<f32, ${ARGMAX_WG}>;
var<workgroup> si : array<u32, ${ARGMAX_WG}>;
@compute @workgroup_size(${ARGMAX_WG},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>){
  var bv : f32 = -3.0e38; var bi : u32 = 0u;
  var i = lid.x;
  loop { if (i >= p.n) { break; } let v = logits[i]; if (v > bv) { bv = v; bi = i; } i = i + ${ARGMAX_WG}u; }
  sv[lid.x] = bv; si[lid.x] = bi; workgroupBarrier();
  var s = ${ARGMAX_WG}u/2u;
  loop { if (s==0u) { break; } if (lid.x < s) { if (sv[lid.x+s] > sv[lid.x]) { sv[lid.x]=sv[lid.x+s]; si[lid.x]=si[lid.x+s]; } } workgroupBarrier(); s=s/2u; }
  if (lid.x==0u) { outIdx[p.outPos] = si[0]; }   // write into the GPU token history at outPos (next position)
}`;
  function argmaxKernel(logitsBuf, outBuf, N, outPos) {
    const pipe = E.getPipeline('q3.argmax', ARGMAX_WGSL);
    const p = uniform(new Uint32Array([N, outPos || 0, 0, 0]));
    return E.dispatch(pipe, [logitsBuf, outBuf, p], [1, 1, 1]);
  }

  // ============================================================
  // Self-tests — each kernel vs a CPU reference. Returns {name, ok, err}[].
  // ============================================================
  function f32buf(arr) { return E.uploadF32(arr instanceof Float32Array ? arr : new Float32Array(arr), ST()); }
  function u32buf(arr) { const a = arr instanceof Uint32Array ? arr : new Uint32Array(arr); const b = E.createBuffer(a.byteLength, ST(), 'ids'); E.device().queue.writeBuffer(b,0,a.buffer,a.byteOffset,a.byteLength); return b; }
  // f16 weight upload (weights are f16 at runtime) + the matching f16-rounded
  // values for the CPU reference, so tests isolate kernel logic from f16 rounding.
  function f16buf(arr) { const a = arr instanceof Float32Array ? arr : new Float32Array(arr); const bits = f32ToF16bits(a); const b = E.createBuffer(a.length*2, ST(), 'w16'); E.device().queue.writeBuffer(b,0,bits); return b; }
  function roundF16(arr) { return f16ToF32(f32ToF16bits(arr instanceof Float32Array ? arr : new Float32Array(arr))); }
  const maxAbs = (a, b) => { let m = 0; for (let i=0;i<a.length;i++) m=Math.max(m, Math.abs(a[i]-b[i])); return m; };

  async function selfTestKernels() {
    await E.init();
    const out = [];
    const check = (name, err, tol) => out.push({ name, ok: err < (tol||1e-3), err });

    // --- rmsnorm ---
    {
      const T=3, H=512, eps=1e-6;
      const x=new Float32Array(T*H), w=new Float32Array(H);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.07);
      for(let i=0;i<H;i++)w[i]=0.5+0.5*Math.cos(i*0.03);
      const y=new Float32Array(T*H); const wR=roundF16(w);
      const xb=f32buf(x), wb=f16buf(w), yb=E.createBuffer(T*H*4, ST(),'y');
      await rmsnorm(xb,wb,yb,T,H,eps);
      const got=await E.readF32(yb,T*H);
      for(let t=0;t<T;t++){let ss=0;for(let i=0;i<H;i++)ss+=x[t*H+i]**2;const inv=1/Math.sqrt(ss/H+eps);for(let i=0;i<H;i++)y[t*H+i]=x[t*H+i]*inv*(1+wR[i]);}
      check('rmsnorm', maxAbs(got,y));
      [xb,wb,yb].forEach(b=>b.destroy());
    }
    // --- linearT ---
    {
      const T=5,N=7,K=64;
      const x=new Float32Array(T*K),W=new Float32Array(N*K);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.11);
      for(let i=0;i<W.length;i++)W[i]=Math.cos(i*0.05);
      const WR=roundF16(W);
      const xb=f32buf(x),wb=f16buf(W),yb=E.createBuffer(T*N*4,ST(),'y');
      await linearT(xb,wb,yb,T,N,K);
      const got=await E.readF32(yb,T*N);
      const y=new Float32Array(T*N);
      for(let t=0;t<T;t++)for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[t*K+k]*WR[n*K+k];y[t*N+n]=a;}
      check('linearT', maxAbs(got,y));
      [xb,wb,yb].forEach(b=>b.destroy());
    }
    // --- embedGather ---
    {
      const T=4,H=32,V=20;
      const embed=new Float32Array(V*H); for(let i=0;i<embed.length;i++)embed[i]=i*0.01;
      const ids=new Uint32Array([3,0,19,7]);
      const eR=roundF16(embed);
      const ib=u32buf(ids),eb=f16buf(embed),yb=E.createBuffer(T*H*4,ST(),'y');
      await embedGather(ib,eb,yb,T,H);
      const got=await E.readF32(yb,T*H);
      const y=new Float32Array(T*H);
      for(let t=0;t<T;t++)for(let h=0;h<H;h++)y[t*H+h]=eR[ids[t]*H+h];
      check('embedGather', maxAbs(got,y));
      [ib,eb,yb].forEach(b=>b.destroy());
    }
    // --- ropeQK (rope + qk-norm) ---
    {
      const T=2,nH=2,hd=8,theta=10000,eps=1e-6,posBase=1;
      const inp=new Float32Array(T*nH*hd),nw=new Float32Array(hd);
      for(let i=0;i<inp.length;i++)inp[i]=Math.sin(i*0.3);
      for(let i=0;i<hd;i++)nw[i]=0.7+0.1*i;
      const nwR=roundF16(nw);
      const ib=f32buf(inp),nb=f16buf(nw),ob=E.createBuffer(inp.length*4,ST(),'o');
      await ropeQK(ib,nb,ob,T,nH,hd,posBase,theta,eps);
      const got=await E.readF32(ob,inp.length);
      // CPU ref
      const y=new Float32Array(inp.length); const half=hd/2;
      for(let t=0;t<T;t++)for(let h=0;h<nH;h++){
        const base=t*(nH*hd)+h*hd; let ss=0; for(let j=0;j<hd;j++)ss+=inp[base+j]**2; const inv=1/Math.sqrt(ss/hd+eps);
        const nrm=new Float32Array(hd); for(let j=0;j<hd;j++)nrm[j]=inp[base+j]*inv*nwR[j];
        const pos=posBase+t;
        for(let j=0;j<hd;j++){const fi=j<half?j:j-half;const ang=pos*Math.pow(theta,-2*fi/hd);const c=Math.cos(ang),s=Math.sin(ang);
          const partner=j<half?nrm[j+half]:nrm[j-half];const rot=j<half?-partner:partner;y[base+j]=nrm[j]*c+rot*s;}
      }
      check('ropeQK', maxAbs(got,y), 2e-3);
      [ib,nb,ob].forEach(b=>b.destroy());
    }
    // --- attention (GQA causal) ---
    {
      const T=3,S=3,nHq=4,nKv=2,hd=8;
      const Q=new Float32Array(T*nHq*hd),Kk=new Float32Array(S*nKv*hd),Vv=new Float32Array(S*nKv*hd);
      for(let i=0;i<Q.length;i++)Q[i]=Math.sin(i*0.2);
      for(let i=0;i<Kk.length;i++)Kk[i]=Math.cos(i*0.15);
      for(let i=0;i<Vv.length;i++)Vv[i]=Math.sin(i*0.09+1);
      const qb=f32buf(Q),kb=f32buf(Kk),vb=f32buf(Vv),ob=E.createBuffer(Q.length*4,ST(),'o');
      await attention(qb,kb,vb,ob,T,S,nHq,nKv,hd);
      const got=await E.readF32(ob,Q.length);
      const y=new Float32Array(Q.length); const grp=nHq/nKv; const scale=1/Math.sqrt(hd);
      for(let t=0;t<T;t++)for(let hq=0;hq<nHq;hq++){const hk=Math.floor(hq/grp);const qb2=t*(nHq*hd)+hq*hd;const last=(S-T)+t;
        let m=-1e38;for(let s=0;s<=last;s++){const kb2=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qb2+i]*Kk[kb2+i];d*=scale;if(d>m)m=d;}
        let den=0;const acc=new Float32Array(hd);for(let s=0;s<=last;s++){const kb2=s*(nKv*hd)+hk*hd;const vb2=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qb2+i]*Kk[kb2+i];const w=Math.exp(d*scale-m);den+=w;for(let i=0;i<hd;i++)acc[i]+=w*Vv[vb2+i];}
        for(let i=0;i<hd;i++)y[qb2+i]=acc[i]/den;}
      check('attention', maxAbs(got,y), 2e-3);
      [qb,kb,vb,ob].forEach(b=>b.destroy());
    }
    // --- swiglu ---
    {
      const n=300;const g=new Float32Array(n),u=new Float32Array(n);
      for(let i=0;i<n;i++){g[i]=Math.sin(i*0.1)*2;u[i]=Math.cos(i*0.07);}
      const gb=f32buf(g),ub=f32buf(u),yb=E.createBuffer(n*4,ST(),'y');
      await swiglu(gb,ub,yb,n);
      const got=await E.readF32(yb,n);const y=new Float32Array(n);
      for(let i=0;i<n;i++){const s=g[i]/(1+Math.exp(-g[i]));y[i]=s*u[i];}
      check('swiglu', maxAbs(got,y));
      [gb,ub,yb].forEach(b=>b.destroy());
    }
    // --- addInPlace ---
    {
      const n=257;const a=new Float32Array(n),b=new Float32Array(n);
      for(let i=0;i<n;i++){a[i]=i*0.5;b[i]=-i*0.2;}
      const ab=f32buf(a),bb=f32buf(b);
      await addInPlace(ab,bb,n);
      const got=await E.readF32(ab,n);const y=new Float32Array(n);for(let i=0;i<n;i++)y[i]=a[i]+b[i];
      check('addInPlace', maxAbs(got,y));
      [ab,bb].forEach(x=>x.destroy());
    }
    // --- gemv (decode T=1 path) vs CPU; also vs linearT for equivalence ---
    {
      const N=200,K=320;
      const x=new Float32Array(K),W=new Float32Array(N*K);
      for(let i=0;i<K;i++)x[i]=Math.sin(i*0.21);
      for(let i=0;i<W.length;i++)W[i]=Math.cos(i*0.013);
      const WR=roundF16(W);
      const xb=f32buf(x),wb=f16buf(W),yb=E.createBuffer(N*4,ST(),'y');
      await gemv(xb,wb,yb,N,K);
      const got=await E.readF32(yb,N);
      const y=new Float32Array(N);
      for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[k]*WR[n*K+k];y[n]=a;}
      check('gemv', maxAbs(got,y));
      [xb,wb,yb].forEach(b=>b.destroy());
    }
    // --- matvecT (batched small-T matmul) vs CPU ---
    {
      const T=8,N=130,K=256;
      const x=new Float32Array(T*K),W=new Float32Array(N*K);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.05);
      for(let i=0;i<W.length;i++)W[i]=Math.cos(i*0.013);
      const WR=roundF16(W);
      const xb=f32buf(x),wb=f16buf(W),yb=E.createBuffer(T*N*4,ST(),'y');
      await matvecT(xb,wb,yb,T,N,K);
      const got=await E.readF32(yb,T*N);
      const y=new Float32Array(T*N);
      for(let t=0;t<T;t++)for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[t*K+k]*WR[n*K+k];y[t*N+n]=a;}
      check('matvecT', maxAbs(got,y));
      [xb,wb,yb].forEach(b=>b.destroy());
    }
    // --- int4 quant helpers for the GEMV-Q / matvec-Q tests ---
    const f32ToBf16 = (a) => { const u=new Uint16Array(a.length); const t=new Float32Array(1),ti=new Uint32Array(t.buffer); for(let i=0;i<a.length;i++){t[0]=a[i];u[i]=ti[0]>>>16;} return u; };
    const dequantInt4 = (pack, scales, N, K) => { const G=QGROUP,wpr=K/8,gpr=K/G; const w=new Float32Array(N*K);
      for(let n=0;n<N;n++)for(let k=0;k<K;k++){const word=pack[n*wpr+(k>>3)];const nib=(word>>>(4*(k&7)))&0xF;const sc=f16ToF32scalar(scales[n*gpr+Math.floor(k/G)]);w[n*K+k]=(nib-8)*sc;} return w; };
    const qbuf = (u32) => { const b=E.createBuffer(u32.byteLength, ST(),'pk'); E.device().queue.writeBuffer(b,0,u32); return b; };
    const sbuf = (u16) => { const b=E.createBuffer(u16.byteLength, ST(),'scs'); E.device().queue.writeBuffer(b,0,u16); return b; };
    // --- gemvQ (int4 decode) vs CPU dequant ---
    {
      const N=130,K=256;
      const x=new Float32Array(K),Wf=new Float32Array(N*K);
      for(let i=0;i<K;i++)x[i]=Math.sin(i*0.2);
      for(let i=0;i<Wf.length;i++)Wf[i]=Math.cos(i*0.013);
      const {pack,scales}=quantizeInt4Bf16(f32ToBf16(Wf),N,K);
      const Wdq=dequantInt4(pack,scales,N,K);
      const xb=f32buf(x),pb=qbuf(pack),sb=sbuf(scales),yb=E.createBuffer(N*4,ST(),'y');
      await gemvQ(xb,pb,sb,yb,N,K);
      const got=await E.readF32(yb,N); const y=new Float32Array(N);
      for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[k]*Wdq[n*K+k];y[n]=a;}
      check('gemvQ', maxAbs(got,y), 1e-2);
      [xb,pb,sb,yb].forEach(b=>b.destroy());
    }
    // --- embedGatherQ (int4 embedding gather) vs CPU dequant ---
    {
      const V=20,H=64,T=4;
      const Ef=new Float32Array(V*H); for(let i=0;i<Ef.length;i++)Ef[i]=Math.sin(i*0.017);
      const {pack,scales}=quantizeInt4Bf16(f32ToBf16(Ef),V,H);
      const Edq=dequantInt4(pack,scales,V,H);
      const ids=new Uint32Array([3,0,19,7]);
      const ib=u32buf(ids),pb=qbuf(pack),sb=sbuf(scales),yb=E.createBuffer(T*H*4,ST(),'y');
      await embedGatherQ(ib,pb,sb,yb,T,H,0);
      const got=await E.readF32(yb,T*H); const y=new Float32Array(T*H);
      for(let t=0;t<T;t++)for(let h=0;h<H;h++)y[t*H+h]=Edq[ids[t]*H+h];
      check('embedGatherQ', maxAbs(got,y), 1e-3);
      [ib,pb,sb,yb].forEach(b=>b.destroy());
    }
    // --- matvecQ (int4 batched) vs CPU dequant ---
    {
      const T=5,N=96,K=128;
      const x=new Float32Array(T*K),Wf=new Float32Array(N*K);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.05);
      for(let i=0;i<Wf.length;i++)Wf[i]=Math.cos(i*0.017);
      const {pack,scales}=quantizeInt4Bf16(f32ToBf16(Wf),N,K);
      const Wdq=dequantInt4(pack,scales,N,K);
      const xb=f32buf(x),pb=qbuf(pack),sb=sbuf(scales),yb=E.createBuffer(T*N*4,ST(),'y');
      await matvecQ(xb,pb,sb,yb,T,N,K,0);
      const got=await E.readF32(yb,T*N); const y=new Float32Array(T*N);
      for(let t=0;t<T;t++)for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[t*K+k]*Wdq[n*K+k];y[t*N+n]=a;}
      check('matvecQ', maxAbs(got,y), 1e-2);
      [xb,pb,sb,yb].forEach(b=>b.destroy());
    }
    // --- gateUpSiluQ (fused int4 gate+up+silu, T=1) vs CPU dequant ---
    {
      const I=130,H=256;
      const x=new Float32Array(H),gf=new Float32Array(I*H),uf=new Float32Array(I*H);
      for(let i=0;i<H;i++)x[i]=Math.sin(i*0.2);
      for(let i=0;i<gf.length;i++){gf[i]=Math.cos(i*0.013);uf[i]=Math.sin(i*0.009);}
      const gq=quantizeInt4Bf16(f32ToBf16(gf),I,H), uq=quantizeInt4Bf16(f32ToBf16(uf),I,H);
      const gdq=dequantInt4(gq.pack,gq.scales,I,H), udq=dequantInt4(uq.pack,uq.scales,I,H);
      const xb=f32buf(x), gW=qbuf(gq.pack), gS=sbuf(gq.scales), uW=qbuf(uq.pack), uS=sbuf(uq.scales), yb=E.createBuffer(I*4,ST(),'y');
      await gateUpSiluQ(xb, {pack:gW,scales:gS}, {pack:uW,scales:uS}, yb, I, H);
      const got=await E.readF32(yb,I); const y=new Float32Array(I);
      for(let i=0;i<I;i++){let g=0,u=0;for(let k=0;k<H;k++){g+=x[k]*gdq[i*H+k];u+=x[k]*udq[i*H+k];}const silu=g/(1+Math.exp(-g));y[i]=silu*u;}
      check('gateUpSiluQ', maxAbs(got,y), 1e-2);
      [xb,gW,gS,uW,uS,yb].forEach(b=>b.destroy());
    }
    // --- argmax ---
    {
      const N=5000; const a=new Float32Array(N);
      for(let i=0;i<N;i++)a[i]=Math.sin(i*0.017); a[3712]=99.0;  // known max
      const ab=f32buf(a), ob=E.createBuffer(4, ST(),'oi');
      await argmaxKernel(ab,ob,N);
      const enc=E.device().createCommandEncoder(); const st=E.createBuffer(4,U.COPY_DST|U.MAP_READ,'s'); enc.copyBufferToBuffer(ob,0,st,0,4); E.device().queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ); const idx=new Uint32Array(st.getMappedRange())[0]; st.unmap(); st.destroy();
      check('argmax', idx===3712?0:1);
      [ab,ob].forEach(b=>b.destroy());
    }
    // --- deltaRecur (gated DeltaNet recurrent step) vs CPU reference ---
    {
      const H=3, dim=DELTA_DIM;              // heads, head_dim (=128, matches WG)
      const q=new Float32Array(H*dim), k=new Float32Array(H*dim), v=new Float32Array(H*dim);
      const gb=new Float32Array(H*2), S0=new Float32Array(H*dim*dim);
      for(let i=0;i<H*dim;i++){ q[i]=Math.sin(i*0.11); k[i]=Math.cos(i*0.07); v[i]=Math.sin(i*0.05+1); }
      for(let h=0;h<H;h++){ gb[h*2]=0.6+0.1*h; gb[h*2+1]=0.3+0.2*h; }   // expg, beta
      for(let i=0;i<S0.length;i++) S0[i]=Math.sin(i*0.013)*0.1;          // nonzero initial state
      // GPU
      const qb=f32buf(q),kb=f32buf(k),vb=f32buf(v),gbb=f32buf(gb);
      const Sb=f32buf(S0), ob=E.createBuffer(H*dim*4, ST(),'o');
      await deltaRecur(qb,kb,vb,gbb,Sb,ob,H,dim);
      const gotOut=await E.readF32(ob,H*dim); const gotS=await E.readF32(Sb,H*dim*dim);
      // CPU ref — state layout [head][key][val] (matches the coalesced GPU kernel), exact recurrence
      const St=Float32Array.from(S0); const cOut=new Float32Array(H*dim);
      for(let h=0;h<H;h++){ const expg=gb[h*2], beta=gb[h*2+1], base=h*dim;
        for(let vi=0;vi<dim;vi++){
          let kv=0; for(let kk=0;kk<dim;kk++){ const i=h*dim*dim+kk*dim+vi; const s=St[i]*expg; St[i]=s; kv+=s*k[base+kk]; }
          const delta=(v[base+vi]-kv)*beta;
          let o=0; for(let kk=0;kk<dim;kk++){ const i=h*dim*dim+kk*dim+vi; const s=St[i]+k[base+kk]*delta; St[i]=s; o+=s*q[base+kk]; }
          cOut[base+vi]=o;
        } }
      check('deltaRecur', Math.max(maxAbs(gotOut,cOut), maxAbs(gotS,St)), 1e-3);
      [qb,kb,vb,gbb,Sb,ob].forEach(b=>b.destroy());
    }
    // --- conv1dDecode (causal depthwise conv1d k=4 + silu + ring state) vs CPU ---
    {
      const C=10, K=CONV_K, T=5;             // channels, kernel, decode steps
      const w=new Float32Array(C*K), bias=new Float32Array(C);
      const xs=[]; for(let t=0;t<T;t++){ const x=new Float32Array(C); for(let c=0;c<C;c++) x[c]=Math.sin(t*0.7+c*0.3); xs.push(x); }
      for(let c=0;c<C;c++){ bias[c]=0.05*c-0.2; for(let j=0;j<K;j++) w[c*K+j]=Math.cos(c*0.4+j*0.9)*0.5; }
      const silu=a=>a/(1+Math.exp(-a));
      // CPU ref: full causal conv with left-pad K-1 zeros, take each step's output
      const cpuOut=[];
      for(let t=0;t<T;t++){ const o=new Float32Array(C);
        for(let c=0;c<C;c++){ let acc=bias[c];
          for(let j=0;j<K;j++){ const ti=t-(K-1)+j; const xv=ti>=0?xs[ti][c]:0; acc+=w[c*K+j]*xv; }
          o[c]=silu(acc); } cpuOut.push(o); }
      // GPU: step the kernel T times, state buffer starts at zero
      const wb=f32buf(w), bb=f32buf(bias);
      const stb=E.createBuffer(C*(K-1)*4, ST(),'st');  // zero-init
      const ob=E.createBuffer(C*4, ST(),'o');
      let maxe=0;
      for(let t=0;t<T;t++){ const xb=f32buf(xs[t]);
        await conv1dDecode(xb, wb, bb, stb, ob, C);
        const got=await E.readF32(ob, C);
        maxe=Math.max(maxe, maxAbs(got, cpuOut[t]));
        xb.destroy();
      }
      check('conv1dDecode', maxe, 1e-3);
      [wb,bb,stb,ob].forEach(b=>b.destroy());
    }
    // --- l2normHeads (per-head L2 normalize) vs CPU ---
    {
      const H=4, dim=DELTA_DIM, eps=1e-6;
      const x=new Float32Array(H*dim); for(let i=0;i<x.length;i++) x[i]=Math.sin(i*0.09)*(1+0.5*((i/dim)|0));
      const xb=f32buf(x), ob=E.createBuffer(H*dim*4, ST(),'o');
      await l2normHeads(xb, ob, H, dim, eps);
      const got=await E.readF32(ob, H*dim);
      const ref=new Float32Array(H*dim);
      for(let h=0;h<H;h++){ let ss=0; for(let i=0;i<dim;i++){ const v=x[h*dim+i]; ss+=v*v; }
        const inv=1/Math.sqrt(ss+eps); for(let i=0;i<dim;i++) ref[h*dim+i]=x[h*dim+i]*inv; }
      check('l2normHeads', maxAbs(got,ref), 1e-4);
      [xb,ob].forEach(b=>b.destroy());
    }
    // --- gbeta (decay + beta precompute) vs CPU ---
    {
      const H=16;
      const a=new Float32Array(H), b=new Float32Array(H), Al=new Float32Array(H), dtb=new Float32Array(H);
      for(let h=0;h<H;h++){ a[h]=Math.sin(h*0.3)*2; b[h]=Math.cos(h*0.2)*1.5; Al[h]=Math.sin(h*0.5)-0.5; dtb[h]=0.1*h-0.5; }
      const ab=f32buf(a),bb=f32buf(b),alb=f32buf(Al),dtbb=f32buf(dtb);
      const gbb=E.createBuffer(H*2*4, ST(),'gb');
      await gbeta(ab,bb,alb,dtbb,gbb,H);
      const got=await E.readF32(gbb, H*2);
      const sp=z=>Math.max(z,0)+Math.log(1+Math.exp(-Math.abs(z)));
      const ref=new Float32Array(H*2);
      for(let h=0;h<H;h++){ ref[h*2]=Math.exp(-Math.exp(Al[h])*sp(a[h]+dtb[h])); ref[h*2+1]=1/(1+Math.exp(-b[h])); }
      check('gbeta', maxAbs(got,ref), 1e-5);
      [ab,bb,alb,dtbb,gbb].forEach(x=>x.destroy());
    }
    // --- gatedRMSNorm (Qwen3NextRMSNormGated) vs CPU ---
    {
      const H=4, dim=DELTA_DIM, eps=1e-6;
      const x=new Float32Array(H*dim), z=new Float32Array(H*dim), w=new Float32Array(dim);
      for(let i=0;i<H*dim;i++){ x[i]=Math.sin(i*0.07)*2; z[i]=Math.cos(i*0.05); }
      for(let i=0;i<dim;i++) w[i]=0.8+0.4*Math.sin(i*0.11);
      const wR=roundF16(w);
      const xb=f32buf(x), zb=f32buf(z), wb=f16buf(w), ob=E.createBuffer(H*dim*4,ST(),'o');
      await gatedRMSNorm(xb, zb, wb, ob, H, dim, eps);
      const got=await E.readF32(ob, H*dim);
      const silu=a=>a/(1+Math.exp(-a));
      const ref=new Float32Array(H*dim);
      for(let h=0;h<H;h++){ let ss=0;
        for(let i=0;i<dim;i++){ const xv=x[h*dim+i]; ss+=xv*xv; }
        const inv=1/Math.sqrt(ss/dim+eps);
        for(let i=0;i<dim;i++) ref[h*dim+i]=x[h*dim+i]*inv*wR[i]*silu(z[h*dim+i]); }
      check('gatedRMSNorm', maxAbs(got,ref), 1e-2);
      [xb,zb,wb,ob].forEach(b=>b.destroy());
    }
    // --- ropeQKNorm35 (qnorm hd256 + partial rotary 64, gated input stride) vs CPU ---
    {
      const T=2, nH=2, hd=8, rotDim=4, inStride=2*hd, eps=1e-6, theta=1e7, posBase=3;
      const inp=new Float32Array(T*nH*inStride), nw=new Float32Array(hd);
      for(let i=0;i<inp.length;i++) inp[i]=Math.sin(i*0.13)*1.5;
      for(let i=0;i<hd;i++) nw[i]=0.7+0.3*Math.cos(i);
      const nwR=roundF16(nw);
      const ib=f32buf(inp), wb=f16buf(nw), ob=E.createBuffer(T*nH*hd*4,ST(),'o');
      await ropeQKNorm35(ib, wb, ob, T, nH, hd, inStride, rotDim, posBase, theta, eps);
      const got=await E.readF32(ob, T*nH*hd);
      // CPU ref
      const ref=new Float32Array(T*nH*hd); const half=rotDim/2;
      for(let t=0;t<T;t++) for(let h=0;h<nH;h++){
        const bi=t*nH*inStride+h*inStride, bo=t*nH*hd+h*hd;
        let ss=0; for(let j=0;j<hd;j++){ const v=inp[bi+j]; ss+=v*v; }
        const inv=1/Math.sqrt(ss/hd+eps);
        const nrm=new Float32Array(hd); for(let j=0;j<hd;j++) nrm[j]=inp[bi+j]*inv*(1+nwR[j]);
        const pos=posBase+t;
        for(let j=0;j<hd;j++){
          if(j>=rotDim){ ref[bo+j]=nrm[j]; continue; }
          const freqIdx = j<half ? j : j-half;
          const invF = Math.pow(theta, -2*freqIdx/rotDim);
          const ang=pos*invF, c=Math.cos(ang), s=Math.sin(ang);
          const partner = j<half ? nrm[j+half] : nrm[j-half];
          const rot = j<half ? -partner : partner;
          ref[bo+j]=nrm[j]*c+rot*s;
        }
      }
      check('ropeQKNorm35', maxAbs(got,ref), 1e-2);
      [ib,wb,ob].forEach(b=>b.destroy());
    }
    // --- applyGate35 (attn * sigmoid(gate-half of qproj)) vs CPU ---
    {
      const nH=3, hd=8;
      const attn=new Float32Array(nH*hd), qproj=new Float32Array(nH*2*hd);
      for(let i=0;i<attn.length;i++) attn[i]=Math.sin(i*0.2);
      for(let i=0;i<qproj.length;i++) qproj[i]=Math.cos(i*0.15);
      const ab=f32buf(attn), qb=f32buf(qproj), ob=E.createBuffer(nH*hd*4,ST(),'o');
      await applyGate35(ab, qb, ob, nH, hd);
      const got=await E.readF32(ob, nH*hd);
      const ref=new Float32Array(nH*hd);
      for(let h=0;h<nH;h++) for(let d=0;d<hd;d++){ const g=qproj[h*2*hd+hd+d]; ref[h*hd+d]=attn[h*hd+d]/(1+Math.exp(-g)); }
      check('applyGate35', maxAbs(got,ref), 1e-5);
      [ab,qb,ob].forEach(b=>b.destroy());
    }
    return out;
  }

  // ============================================================
  // Tokenizer — byte-level BPE (Qwen2Tokenizer), faithful to tokenizer.json:
  // pretokenizer regex (contractions expanded since JS lacks (?i:) groups) →
  // byte-level encode → rank-ordered BPE merges → vocab ids. Specials spliced
  // directly. Loaded from tokenizer.json (vocab + merges + added_tokens).
  // ============================================================
  // Qwen3.5 special ids (resolved from the loaded vocab by content in TOK.load;
  // these defaults are the known Qwen3.5-0.8B ids in case resolution is skipped).
  const SPECIAL = { endoftext: 248044, im_start: 248045, im_end: 248046, think: 248068, think_end: 248069 };
  // JS port of the Qwen pre_tokenizer Split regex ((?i:'s|…) expanded to case classes).
  const PRETOK_RE = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

  // GPT-2 byte↔unicode map (printable bytes map to themselves; the rest to a
  // private high range) so every byte becomes a visible char for BPE.
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

  const TOK = (function () {
    let vocab = null, idToTok = null, bpeRanks = null, byteEnc = null, byteDec = null, ready = false;
    const enc = new TextEncoder(), dec = new TextDecoder();

    async function load(root) {
      if (ready) return;
      const url = (root || 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/') + 'tokenizer.json';
      const j = await (await fetch(url)).json();
      vocab = j.model.vocab;                       // token string -> id
      idToTok = {}; for (const k in vocab) idToTok[vocab[k]] = k;
      for (const a of (j.added_tokens || [])) { vocab[a.content] = a.id; idToTok[a.id] = a.content; }
      // Resolve special-token ids from the actual vocab (robust across vocab sizes).
      const sp = (c, d) => (vocab[c] != null ? vocab[c] : d);
      SPECIAL.endoftext = sp('<|endoftext|>', SPECIAL.endoftext);
      SPECIAL.im_start  = sp('<|im_start|>',  SPECIAL.im_start);
      SPECIAL.im_end    = sp('<|im_end|>',    SPECIAL.im_end);
      SPECIAL.think     = sp('<think>',       SPECIAL.think);
      SPECIAL.think_end = sp('</think>',      SPECIAL.think_end);
      bpeRanks = new Map();
      const merges = j.model.merges || [];
      for (let i = 0; i < merges.length; i++) {
        const m = merges[i];
        const pair = Array.isArray(m) ? (m[0] + ' ' + m[1]) : m.replace(' ', ' ');
        bpeRanks.set(pair, i);
      }
      ({ byteEnc, byteDec } = buildByteMaps());
      ready = true;
    }

    // BPE-merge one byte-encoded piece (a string of byte-chars) into subword tokens.
    function bpe(piece) {
      let word = Array.from(piece);
      if (word.length < 2) return word;
      for (;;) {
        let best = null, bestRank = Infinity, bestI = -1;
        for (let i = 0; i < word.length - 1; i++) {
          const r = bpeRanks.get(word[i] + ' ' + word[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; bestI = i; }
        }
        if (best === null) break;
        const merged = word[bestI] + word[bestI + 1];
        word = word.slice(0, bestI).concat(merged, word.slice(bestI + 2));
      }
      return word;
    }

    // Encode raw text (no special-token handling) → id[].
    function encodeText(text) {
      const ids = [];
      const matches = text.match(PRETOK_RE) || [];
      for (const piece of matches) {
        const bytes = enc.encode(piece);
        let s = ''; for (const b of bytes) s += byteEnc[b];
        for (const sub of bpe(s)) {
          const id = vocab[sub];
          if (id !== undefined) ids.push(id);
          // (no byte_fallback in this tokenizer; unknown subwords shouldn't occur)
        }
      }
      return ids;
    }

    // Hermes-style tool preamble Qwen3.5 expects, appended to the system message.
    function toolsPreamble(tools) {
      let s = '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n' +
        'You are provided with function signatures within <tools></tools> XML tags:\n<tools>';
      for (const t of tools) s += '\n' + JSON.stringify(t);
      s += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within ' +
        '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>';
      return s;
    }

    // Build the Qwen3 ChatML prompt with special ids spliced in. With `tools`, injects
    // the Hermes tool block into the system turn, renders assistant tool_calls as
    // <tool_call>{json}</tool_call>, and groups consecutive tool results into one
    // <|im_start|>user … <tool_response>…</tool_response> turn (the Qwen template shape).
    function encodeChat(messages, { addGenerationPrompt = true, tools = null } = {}) {
      const ids = [];
      const seg = (role, content) => {
        ids.push(SPECIAL.im_start);
        ids.push(...encodeText(role + '\n' + content));
        ids.push(SPECIAL.im_end);
        ids.push(...encodeText('\n'));
      };
      let i = 0;
      const hasSys = messages.length && messages[0].role === 'system';
      const sysContent = hasSys ? (messages[0].content || '') : '';
      if (hasSys) i = 1;
      if (tools && tools.length) {
        seg('system', (sysContent ? sysContent + '\n\n' : '') + toolsPreamble(tools));
      } else if (hasSys) {
        seg('system', sysContent);
      }
      for (; i < messages.length; i++) {
        const m = messages[i];
        if (m.role === 'tool') {                                  // group consecutive tool results
          const parts = []; let j = i;
          while (j < messages.length && messages[j].role === 'tool') {
            parts.push('<tool_response>\n' + (messages[j].content || '') + '\n</tool_response>'); j++;
          }
          seg('user', parts.join('\n')); i = j - 1; continue;
        }
        if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length) {
          let body = m.content || '';
          for (const tc of m.tool_calls) {
            const name = (tc.function && tc.function.name) || tc.name || '';
            let a = (tc.function && tc.function.arguments != null) ? tc.function.arguments : tc.arguments;
            if (typeof a !== 'string') a = JSON.stringify(a || {});
            body += (body ? '\n' : '') + '<tool_call>\n{"name": "' + name + '", "arguments": ' + a + '}\n</tool_call>';
          }
          seg('assistant', body); continue;
        }
        seg(m.role, typeof m.content === 'string' ? m.content : '');
      }
      if (addGenerationPrompt) { ids.push(SPECIAL.im_start); ids.push(...encodeText('assistant\n')); }
      return ids;
    }

    function decode(ids) {
      let s = ''; for (const id of ids) { const t = idToTok[id]; if (t !== undefined) s += t; }
      // reverse byte-level: each char -> its byte (skip chars not in byteDec, e.g. specials)
      const bytes = [];
      for (const ch of s) { const b = byteDec[ch]; if (b !== undefined) bytes.push(b); }
      return dec.decode(new Uint8Array(bytes));
    }

    return { load, encodeText, encodeChat, decode, isReady: () => ready };
  })();

  // ============================================================
  // Weight loader — safetensors (bf16) → per-tensor f16 GPU buffers.
  // Weights are stored as f16 (half the bytes of f32): decode reads every weight
  // each token, so this ~halves memory bandwidth AND the GPU footprint (~3GB→1.5GB).
  // Kernels read array<f16> and convert to f32 for the math (activations stay f32).
  // bf16→f16 goes via f32 (different exponent widths). Cached in Cache Storage.
  // ============================================================
  let MODEL_ROOT = 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/';
  let MODEL_FILE = 'model.safetensors-00001-of-00001.safetensors';
  let _compressed = false;   // true → safetensors is pre-quantized compressed-tensors (weight_packed)
  // Select the active variant (call before loadModel). Mutates CONFIG dims + the model
  // URL + the OPFS cache key. If a different variant is already loaded, unload first.
  function selectModel(v) {
    if (!VARIANTS[v]) throw new Error('unknown variant ' + v);
    if (_loaded && v !== _variant) unload();
    _variant = v;
    CONFIG.hidden = VARIANTS[v].hidden;
    CONFIG.intermediate = VARIANTS[v].intermediate;
    // Optional self-hosting / dev override: localStorage 'q35_root_<variant>' (or a global
    // window.__Q35_ROOT[variant]) points the weights+tokenizer fetch at a mirror (e.g. a
    // localhost static server) instead of HuggingFace — avoids HF throttling and OPFS-eviction
    // re-downloads during iteration. The mirror must hold both MODEL_FILE and tokenizer.json.
    let override = null;
    try { override = (window.__Q35_ROOT && window.__Q35_ROOT[v]) || localStorage.getItem('q35_root_' + v); } catch (_) {}
    MODEL_ROOT = override || ('https://huggingface.co/' + VARIANTS[v].repo + '/resolve/main/');
    MODEL_FILE = VARIANTS[v].file;
    _compressed = !!VARIANTS[v].compressed;
  }
  let _weights = null;            // name -> { buf, shape, numel }  (buf holds f16)
  let _loaded = false;

  // f32 → f16 bits (round-to-nearest), handling normals, subnormals, overflow.
  const _f32a = new Float32Array(1), _u32a = new Uint32Array(_f32a.buffer);
  function f32ToF16(val) {
    _f32a[0] = val; const x = _u32a[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff; const mant = x & 0x7fffff;
    if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00);   // NaN / Inf
    let e = exp - 127 + 15;
    if (e >= 31) return sign | 0x7c00;                          // overflow → Inf
    if (e <= 0) {                                               // subnormal / zero
      if (e < -10) return sign;
      const m = mant | 0x800000; const shift = 14 - e;
      let half = m >>> shift;
      if ((m >>> (shift - 1)) & 1) half += 1;                  // round
      return sign | half;
    }
    let half = (e << 10) | (mant >>> 13);
    if ((mant >>> 12) & 1) half += 1;                          // round to nearest
    return sign | half;
  }
  function f16ToF32(u16) {
    const n = u16.length, out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = u16[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
      if (e === 0) out[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) out[i] = f ? NaN : (s ? -Infinity : Infinity);
      else out[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }
  // bf16 bits → f16 bits (via lossless bf16→f32, then rounded f32→f16).
  function bf16ToF16bits(u16) {
    const n = u16.length, out = new Uint16Array(n);
    const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
    for (let i = 0; i < n; i++) { ti[0] = u16[i] << 16; out[i] = f32ToF16(t[0]); }
    return out;
  }
  function f32ToF16bits(f32) {
    const out = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) out[i] = f32ToF16(f32[i]);
    return out;
  }
  function f16ToF32scalar(h) { const s=(h&0x8000)>>15,e=(h&0x7c00)>>10,f=h&0x03ff;
    if(e===0) return (s?-1:1)*Math.pow(2,-14)*(f/1024);
    if(e===31) return f?NaN:(s?-Infinity:Infinity);
    return (s?-1:1)*Math.pow(2,e-15)*(1+f/1024); }

  // Quantize a [N,K] bf16 tensor → group-wise symmetric int4 (G=QGROUP along K).
  // Returns { pack:Uint32Array(N*K/8), scales:Uint16Array(N*K/QGROUP f16) }.
  // nibble = clamp(round(w/scale),-8,7)+8 ; packed 8/u32 (k=8w+i → bits 4i).
  function quantizeInt4Bf16(u16, N, K) {
    const G = QGROUP, wpr = K / 8, gpr = K / G;
    const pack = new Uint32Array(N * wpr), scales = new Uint16Array(N * gpr);
    const tf = new Float32Array(1), ti = new Uint32Array(tf.buffer);
    const f = (idx) => { ti[0] = u16[idx] << 16; return tf[0]; };
    for (let n = 0; n < N; n++) {
      const rU = n * K, rP = n * wpr, rS = n * gpr;
      for (let g = 0; g < gpr; g++) {
        let maxabs = 0;
        for (let j = 0; j < G; j++) { const v = Math.abs(f(rU + g*G + j)); if (v > maxabs) maxabs = v; }
        const scale = maxabs > 0 ? maxabs / 7 : 1e-8;
        const sBits = f32ToF16(scale); scales[rS + g] = sBits;
        const inv = 1 / f16ToF32scalar(sBits);   // quantize against the STORED (f16) scale
        for (let j = 0; j < G; j++) {
          const k = g*G + j;
          let q = Math.round(f(rU + k) * inv); if (q < -8) q = -8; else if (q > 7) q = 7;
          pack[rP + (k >> 3)] |= ((q + 8) & 0xF) << (4 * (k & 7));
        }
      }
    }
    return { pack, scales };
  }
  // Big matrices → int4 (all have K%32==0). NB in_proj_a/b are tiny [16,1024] but
  // still K%32==0, so int4 is fine and keeps the path uniform.
  const QUANT_SUFFIX = ['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight',
    'gate_proj.weight','up_proj.weight','down_proj.weight',
    'in_proj_qkv.weight','in_proj_a.weight','in_proj_b.weight','in_proj_z.weight','out_proj.weight'];
  // DeltaNet projections: int4 (DELTA_F16=false) or f16 (true). int4 for production.
  const DELTA_F16 = false;
  const isDeltaProj = (name) => name.includes('linear_attn.in_proj') || name.includes('linear_attn.out_proj');
  const isQuantWeight = (name) => (DELTA_F16 && isDeltaProj(name)) ? false : QUANT_SUFFIX.some(s => name.endsWith(s));
  // Precision-sensitive small params kept as raw f32 (kernels read them as f32).
  const isF32Raw = (name) => name.endsWith('conv1d.weight') || name.endsWith('.A_log') || name.endsWith('.dt_bias');
  // Vision tower + multi-token-prediction head: not used for text generation.
  const isSkip = (name) => name.startsWith('model.visual.') || name.startsWith('mtp.');
  // bf16 bits → f32 (lossless: bf16 is the top 16 bits of f32).
  function bf16ToF32arr(u16) { const out = new Float32Array(u16.length); const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
    for (let i = 0; i < u16.length; i++) { ti[0] = u16[i] << 16; out[i] = t[0]; } return out; }

  const opfsFile = () => 'q35-' + _variant + '-model.safetensors';   // per-variant OPFS cache (GB-scale quota)
  const OPFS_SLICE = 256 * 1024 * 1024;   // read/return the cached file in ≤256MB pieces (never one GB-scale alloc)
  // Read the cached model back as an ARRAY OF CHUNKS (File.slice per piece). Avoids a single
  // 1.75GB+ ArrayBuffer alloc (which fails under memory pressure / on mobile). null if absent.
  async function opfsReadChunks(onProgress) {
    try {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle(opfsFile());   // throws if absent
      const f = await fh.getFile();
      if (f.size < 1e9) return null;                    // partial/corrupt
      const chunks = [];
      for (let off = 0; off < f.size; off += OPFS_SLICE) {
        const end = Math.min(off + OPFS_SLICE, f.size);
        chunks.push(new Uint8Array(await f.slice(off, end).arrayBuffer()));
        onProgress && onProgress({ phase: 'cache', pct: Math.round(end / f.size * 100) });
      }
      return chunks;
    } catch (_) { return null; }
  }
  // Stream the downloaded chunks straight to an OPFS file — NO 1.75GB concat buffer, and
  // delete any stale/partial file first so createWritable doesn't copy-on-write double it
  // (the QuotaExceededError culprit: 2×1.75GB > the ~3GB sandbox quota).
  // Stream the download STRAIGHT into OPFS (write-through), without holding the whole
  // model in the JS heap. That heap pressure (~1.75GB of chunks) is what was squeezing the
  // dynamic storage quota and quota-failing the write — a fresh 1.75GB OPFS file writes fine
  // when RAM is free. Returns true if fully cached. (Also fixes the mobile low-RAM failure.)
  async function downloadToOpfs(onProgress) {
    let root, writer;
    try {
      root = await navigator.storage.getDirectory();
      try { await root.removeEntry(opfsFile()); } catch (_) {}
      writer = await (await root.getFileHandle(opfsFile(), { create: true })).createWritable();
    } catch (_) { return false; }
    try {
      const resp = await fetch(MODEL_ROOT + MODEL_FILE);
      const total = +(resp.headers.get('content-length') || 0);
      const reader = resp.body.getReader(); let recv = 0;
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        await writer.write(value); recv += value.length;
        if (total) onProgress && onProgress({ phase: 'download', pct: Math.round(recv / total * 100), recv, total });
      }
      await writer.close();
      return true;
    } catch (e) {
      try { console.warn('[q35] OPFS write-through failed (', (e && e.message) || e, ') — will stream from RAM', e); } catch (_) {}
      try { await writer.close(); } catch (_) {}
      try { await root.removeEntry(opfsFile()); } catch (_) {}   // drop the partial
      return false;
    }
  }
  // Fallback: download into RAM chunks (uncached) when OPFS caching is impossible.
  async function downloadToRam(onProgress) {
    const resp = await fetch(MODEL_ROOT + MODEL_FILE);
    const total = +(resp.headers.get('content-length') || 0);
    const reader = resp.body.getReader(); const chunks = []; let recv = 0;
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      chunks.push(value); recv += value.length;
      if (total) onProgress && onProgress({ phase: 'download', pct: Math.round(recv / total * 100), recv, total });
    }
    return chunks;
  }
  // A byte source with readRange(start,len) → fresh Uint8Array, backed by an array of byte
  // chunks (cached file read back in slices, or raw download chunks). No GB-scale alloc.
  function srcFromChunks(chunks) {
    const offs = new Array(chunks.length + 1); offs[0] = 0;
    for (let i = 0; i < chunks.length; i++) offs[i + 1] = offs[i] + chunks[i].length;
    const find = (pos) => { let lo = 0, hi = chunks.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (offs[m + 1] <= pos) lo = m + 1; else hi = m; } return lo; };
    return {
      byteLength: offs[chunks.length],
      readRange(start, len) {
        const out = new Uint8Array(len); let written = 0, pos = start, ci = find(start);
        while (written < len) { const c = chunks[ci], inOff = pos - offs[ci], take = Math.min(c.length - inOff, len - written);
          out.set(c.subarray(inOff, inOff + take), written); written += take; pos += take; ci++; }
        return out;
      },
    };
  }
  async function fetchModelBytes(onProgress) {
    const cached = await opfsReadChunks(onProgress);
    if (cached) return srcFromChunks(cached);
    // First try write-through to OPFS (low peak RAM → the write actually fits the quota),
    // then serve THIS session by reading the cache back in slices. If caching is impossible
    // (quota/permission), fall back to an in-RAM download (uncached — re-downloads next time).
    if (await downloadToOpfs(onProgress)) {
      const back = await opfsReadChunks(onProgress);
      if (back) return srcFromChunks(back);
    }
    return srcFromChunks(await downloadToRam(onProgress));
  }

  // ---- Quantized-weights cache --------------------------------------------------
  // Skip the per-load bf16→int4 quantize (the ~40s "parse" phase, the dominant fixed load
  // cost once the model is local) by saving the int4 packs + scales + f16/f32 raws to a
  // compact OPFS blob (~0.3GB for 0.8B, ~5× smaller than the safetensors) and reloading them
  // straight into GPU buffers. Format: [u64 headerLen][JSON header][data]; header.parts give
  // byte offsets into the data section (each part 4-byte padded for writeBuffer).
  const QCACHE_VER = 1;
  const qcacheFile = () => 'q35-' + _variant + '-q4v' + QCACHE_VER + '.bin';
  const _u8 = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const _pad4 = (n) => (n + 3) & ~3;
  async function writeQuantCache(qc, onProgress) {
    let off = 0;
    const tensors = qc.map(t => {
      const arrs = t.kind === 'int4' ? [t.pack, t.scales] : [t.data];
      const parts = arrs.map(a => { const len = _pad4(a.byteLength); const p = { off, len }; off += len; return p; });
      return { name: t.name, kind: t.kind, N: t.N, K: t.K, shape: t.shape, numel: t.numel, parts };
    });
    const hjson = new TextEncoder().encode(JSON.stringify({ version: QCACHE_VER, variant: _variant, hidden: CONFIG.hidden, intermediate: CONFIG.intermediate, tensors }));
    let root, w;
    try {
      root = await navigator.storage.getDirectory();
      try { await root.removeEntry(qcacheFile()); } catch (_) {}
      w = await (await root.getFileHandle(qcacheFile(), { create: true })).createWritable();
      const hdr = new Uint8Array(8); new DataView(hdr.buffer).setBigUint64(0, BigInt(hjson.byteLength), true);
      await w.write(hdr); await w.write(hjson);
      const pad = new Uint8Array(4);
      for (const t of qc) {
        const arrs = t.kind === 'int4' ? [t.pack, t.scales] : [t.data];
        for (const a of arrs) { const u = _u8(a); await w.write(u); const r = _pad4(u.byteLength) - u.byteLength; if (r) await w.write(pad.subarray(0, r)); }
      }
      await w.close();
      onProgress && onProgress({ phase: 'qcache', pct: 100 });
      return true;
    } catch (e) { try { console.warn('[q35] quant-cache write failed', e); } catch (_) {} try { await w.close(); } catch (_) {} try { await root.removeEntry(qcacheFile()); } catch (_) {} return false; }
  }
  // Delete the large bf16 safetensors from OPFS (called once the compact quant cache is
  // written) so only the ~0.4GB quantized blob remains — the 1.7GB bf16 otherwise wastes
  // OPFS quota and risks evicting the quant cache it was used to build.
  async function deleteBf16Opfs() {
    try { const root = await navigator.storage.getDirectory(); await root.removeEntry(opfsFile()); return true; }
    catch (_) { return false; }
  }
  async function loadQuantCache(onProgress) {
    let chunks = null;
    try {
      const root = await navigator.storage.getDirectory();
      const f = await (await root.getFileHandle(qcacheFile())).getFile();
      if (f.size < 1e7) return false;
      chunks = [];
      for (let o = 0; o < f.size; o += OPFS_SLICE) { const e = Math.min(o + OPFS_SLICE, f.size); chunks.push(new Uint8Array(await f.slice(o, e).arrayBuffer())); onProgress && onProgress({ phase: 'cache', pct: Math.round(e / f.size * 100) }); }
    } catch (_) { return false; }
    try {
      const src = srcFromChunks(chunks);
      const hlen = Number(new DataView(src.readRange(0, 8).buffer).getBigUint64(0, true));
      const hdr = JSON.parse(dec_(src.readRange(8, hlen)));
      if (hdr.version !== QCACHE_VER || hdr.variant !== _variant || hdr.hidden !== CONFIG.hidden) return false;
      const dataStart = 8 + hlen;
      _weights = {};
      const mk = (p, label) => { const b = E.createBuffer(p.len, ST(), label); E.device().queue.writeBuffer(b, 0, src.readRange(dataStart + p.off, p.len)); return b; };
      for (let i = 0; i < hdr.tensors.length; i++) {
        const t = hdr.tensors[i];
        if (t.kind === 'int4') _weights[t.name] = { pack: mk(t.parts[0], t.name + '.pk'), scales: mk(t.parts[1], t.name + '.sc'), N: t.N, K: t.K, int4: true, shape: t.shape, numel: t.numel };
        else if (t.kind === 'f32') _weights[t.name] = { buf: mk(t.parts[0], t.name), f32: true, shape: t.shape, numel: t.numel };
        else _weights[t.name] = { buf: mk(t.parts[0], t.name), shape: t.shape, numel: t.numel };
        if ((i & 15) === 0) onProgress && onProgress({ phase: 'parse', pct: Math.round(i / hdr.tensors.length * 100) });
      }
      return true;
    } catch (e) { try { console.warn('[q35] quant-cache load failed', e); } catch (_) {} return false; }
  }

  // ---- Pre-built (remote) quantized weights ------------------------------------
  // The bf16→int4 quantize is a ~40s one-time CPU wait on a fresh browser (the "parse"
  // phase). It's avoidable entirely: the q4v blob written by writeQuantCache IS our exact
  // GPU format, so if a host serves a pre-built one we download THAT (~0.3GB for 0.8B, vs
  // the 1.75GB bf16 safetensors) straight into the OPFS quant-cache and skip both the big
  // download AND the quantize. Resolve the URL from (in order) a global, a localStorage
  // override, or the variant's q4vUrl; null → no remote, fall back to bf16+quantize.
  // Produce the file to host with exportQuantCache() (below) after one normal load.
  function q4vUrl() {
    try { return (window.__Q35_Q4V && window.__Q35_Q4V[_variant]) || localStorage.getItem('q35_q4v_' + _variant) || (VARIANTS[_variant] && VARIANTS[_variant].q4vUrl) || null; }
    catch (_) { return (VARIANTS[_variant] && VARIANTS[_variant].q4vUrl) || null; }
  }
  // Stream a pre-built q4v straight into the OPFS quant-cache file (write-through, no
  // GB-scale heap alloc — same approach as downloadToOpfs). Returns true if it landed;
  // loadQuantCache() then reads it back (and validates version/variant). Best-effort:
  // any failure (no URL, 404, quota) returns false and the caller falls back to bf16.
  async function fetchRemoteQuantCache(onProgress) {
    const url = q4vUrl();
    if (!url) return false;
    let root, writer;
    try {
      root = await navigator.storage.getDirectory();
      try { await root.removeEntry(qcacheFile()); } catch (_) {}
      writer = await (await root.getFileHandle(qcacheFile(), { create: true })).createWritable();
    } catch (_) { return false; }
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error('q4v fetch ' + resp.status);
      const total = +(resp.headers.get('content-length') || 0);
      const reader = resp.body.getReader(); let recv = 0;
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        await writer.write(value); recv += value.length;
        if (total) onProgress && onProgress({ phase: 'download', pct: Math.round(recv / total * 100), recv, total, prebuilt: true });
      }
      await writer.close();
      return true;
    } catch (e) {
      try { console.warn('[q35] prebuilt q4v fetch failed (', (e && e.message) || e, ') — falling back to bf16+quantize'); } catch (_) {}
      try { await writer.close(); } catch (_) {}
      try { await root.removeEntry(qcacheFile()); } catch (_) {}
      return false;
    }
  }
  // Save the OPFS quant-cache to a downloadable file so it can be hosted and served back
  // via q4vUrl() — i.e. how you produce the pre-built blob. Run a normal load once (which
  // writes the cache), then call SandpieQwen35.exportQuantCache() from the console.
  async function exportQuantCache() {
    const root = await navigator.storage.getDirectory();
    const f = await (await root.getFileHandle(qcacheFile())).getFile();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(f); a.download = qcacheFile();
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    return { file: qcacheFile(), bytes: f.size };
  }

  async function loadModel({ onProgress, variant } = {}) {
    if (variant) selectModel(variant);   // may unload a different already-loaded variant
    // Device-loss recovery (THE failure mode on weak iGPUs): a TDR/driver reset nulls the
    // engine's device + caps, but our _loaded stayed true — so without this check loadModel
    // would early-return and every forward would throw "engine not initialised". When caps
    // is gone, our GPU buffers are dead handles: drop them (unload) and do a full reload.
    // Fast, because the OPFS quant-cache is still warm (~3.5s, no re-quantize).
    if (_loaded && (!E.caps || !E.caps())) {
      try { console.warn('[q35] WebGPU device was lost — reloading the model on the new device.'); } catch (_) {}
      try { unload(); } catch (_) { _loaded = false; _weights = null; }
    }
    if (_loaded) return;
    await E.init();
    await probeSubgroups();   // mobile-safety: disable subgroups if this GPU computes them wrong
    await TOK.load(MODEL_ROOT);
    onProgress && onProgress({ phase: 'tokenizer', pct: 100 });
    // Fast path: quantized weights already cached → straight to GPU buffers (skip the
    // safetensors download/parse AND the bf16→int4 quantize).
    if (await loadQuantCache(onProgress)) { onProgress && onProgress({ phase: 'parse', pct: 100 }); _loaded = true; await warmup(onProgress); deleteBf16Opfs(); return; }   // drop any lingering bf16 (older caches kept both)
    // Next-fastest: a host-served pre-built q4v (skips the 1.75GB bf16 download AND the
    // ~40s quantize). Downloaded into the OPFS cache, then read back like a local hit.
    if (await fetchRemoteQuantCache(onProgress) && await loadQuantCache(onProgress)) {
      onProgress && onProgress({ phase: 'parse', pct: 100 }); _loaded = true; await warmup(onProgress); return;
    }
    const src = await fetchModelBytes(onProgress);
    onProgress && onProgress({ phase: 'parse', pct: 0 });
    const qc = [];   // collect quantized CPU arrays → written to the OPFS quant-cache after parse
    const headerLen = Number(new DataView(src.readRange(0, 8).buffer).getBigUint64(0, true));
    const header = JSON.parse(dec_(src.readRange(8, headerLen)));
    const dataStart = 8 + headerLen;
    _weights = {};
    const names = Object.keys(header).filter(n => n !== '__metadata__' && !isSkip(n));
    // Aligned byte-copy of a tensor's raw bytes → a typed array of the given ctor.
    const aligned = (raw, Ctor) => { const a = new Ctor(raw.byteLength / Ctor.BYTES_PER_ELEMENT); new Uint8Array(a.buffer).set(raw); return a; };
    const readT = (inf) => src.readRange(dataStart + inf.data_offsets[0], inf.data_offsets[1] - inf.data_offsets[0]);
    for (let i = 0; i < names.length; i++) {
      const name = names[i], info = header[name];
      // compressed-tensors: scale/shape are consumed alongside their weight_packed
      if (name.endsWith('.weight_scale') || name.endsWith('.weight_shape')) continue;
      if (name.endsWith('.weight_packed')) {
        // pre-quantized int4 (group 32, symmetric) — same scheme as ours, but two's-complement
        // nibbles; XOR 0x88888888 flips bit-3 of each nibble → our offset-binary (q+8) encoding.
        const base = name.slice(0, -'.weight_packed'.length);
        const N = info.shape[0], K = info.shape[1] * 8;   // weight_packed is [N, K/8]
        const pk = aligned(readT(info), Uint32Array);
        if (_awqXor) { for (let j = 0; j < pk.length; j++) pk[j] = pk[j] ^ 0x88888888; }
        const scF16 = bf16ToF16bits(aligned(readT(header[base + '.weight_scale']), Uint16Array));
        const packBuf = E.createBuffer(pk.byteLength, ST(), base + '.pk');
        const scBuf = E.createBuffer(scF16.byteLength, ST(), base + '.sc');
        E.device().queue.writeBuffer(packBuf, 0, pk);
        E.device().queue.writeBuffer(scBuf, 0, scF16);
        _weights[base + '.weight'] = { pack: packBuf, scales: scBuf, N, K, int4: true, shape: [N, K], numel: N * K };
        qc.push({ name: base + '.weight', kind: 'int4', N, K, shape: [N, K], numel: N * K, pack: pk, scales: scF16 });
        if ((i & 15) === 0) onProgress && onProgress({ phase: 'parse', pct: Math.round(i / names.length * 100) });
        continue;
      }
      const [begin, end] = info.data_offsets;
      const numel = info.shape.reduce((a, b) => a * b, 1);
      const raw = src.readRange(dataStart + begin, end - begin);
      if (isQuantWeight(name)) {
        if (info.dtype !== 'BF16') throw new Error('quant path expects BF16 for ' + name);
        const N = info.shape[0], K = info.shape[1];
        const { pack, scales } = quantizeInt4Bf16(aligned(raw, Uint16Array), N, K);
        const packBuf = E.createBuffer(pack.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, name + '.pack');
        const scBuf = E.createBuffer(scales.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, name + '.sc');
        E.device().queue.writeBuffer(packBuf, 0, pack);
        E.device().queue.writeBuffer(scBuf, 0, scales);
        _weights[name] = { pack: packBuf, scales: scBuf, N, K, int4: true, shape: info.shape, numel };
        qc.push({ name, kind: 'int4', N, K, shape: info.shape, numel, pack, scales });
      } else if (isF32Raw(name)) {
        let f32;
        if (info.dtype === 'BF16') f32 = bf16ToF32arr(aligned(raw, Uint16Array));
        else if (info.dtype === 'F32') f32 = aligned(raw, Float32Array);
        else throw new Error('f32-raw path expects BF16/F32 for ' + name);
        const buf = E.createBuffer(numel * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, name);
        E.device().queue.writeBuffer(buf, 0, f32);
        _weights[name] = { buf, f32: true, shape: info.shape, numel };
        qc.push({ name, kind: 'f32', shape: info.shape, numel, data: f32 });
      } else if (name.endsWith('embed_tokens.weight')) {
        // Tied embedding: store ONLY an int4 copy. It serves BOTH the embedding gather
        // (embedGatherQ dequantizes a row) AND the lm_head matmul. No f16 copy → saves
        // ~0.5GB (0.8B) / ~1GB (2B) of GPU memory and a redundant buffer.
        if (info.dtype !== 'BF16') throw new Error('embed expects BF16');
        const N = info.shape[0], K = info.shape[1];
        const { pack, scales } = quantizeInt4Bf16(aligned(raw, Uint16Array), N, K);
        const packBuf = E.createBuffer(pack.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'embed.pack');
        const scBuf = E.createBuffer(scales.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'embed.sc');
        E.device().queue.writeBuffer(packBuf, 0, pack);
        E.device().queue.writeBuffer(scBuf, 0, scales);
        _weights['__embed_int4'] = { pack: packBuf, scales: scBuf, N, K, int4: true, shape: info.shape, numel };
        qc.push({ name: '__embed_int4', kind: 'int4', N, K, shape: info.shape, numel, pack, scales });
      } else {
        let f16bits;
        if (info.dtype === 'BF16') f16bits = bf16ToF16bits(aligned(raw, Uint16Array));
        else if (info.dtype === 'F16') f16bits = aligned(raw, Uint16Array);
        else if (info.dtype === 'F32') f16bits = f32ToF16bits(aligned(raw, Float32Array));
        else throw new Error('unsupported dtype ' + info.dtype + ' for ' + name);
        const buf = E.createBuffer(numel * 2, U.STORAGE | U.COPY_DST | U.COPY_SRC, name);
        E.device().queue.writeBuffer(buf, 0, f16bits);
        _weights[name] = { buf, shape: info.shape, numel };
        qc.push({ name, kind: 'f16', shape: info.shape, numel, data: f16bits });
      }
      if ((i & 15) === 0) onProgress && onProgress({ phase: 'parse', pct: Math.round(i / names.length * 100) });
    }
    onProgress && onProgress({ phase: 'parse', pct: 100 });
    _loaded = true;
    await warmup(onProgress);                                     // compile all pipelines now, not on the first message
    // Save the compact quant cache, then DELETE the big bf16 safetensors — every future
    // load reads only the ~0.4GB quantized blob (no re-download, no re-quantize). Only delete
    // if the quant cache wrote OK (else keep the bf16 so we don't have to re-download).
    try { if (await writeQuantCache(qc, onProgress)) await deleteBf16Opfs(); } catch (_) {}
  }
  const _td = new TextDecoder();
  function dec_(u8) { return _td.decode(u8); }
  // Read an f16 GPU buffer back to a CPU Float32Array (debug only).
  async function readF16(buf, n) {
    const bytes = Math.ceil(n * 2 / 4) * 4;
    const staging = E.createBuffer(bytes, U.COPY_DST | U.MAP_READ, 'rd16');
    const enc = E.device().createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    E.device().queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(staging.getMappedRange().slice(0), 0, n);
    const out = f16ToF32(u16); staging.unmap(); staging.destroy();
    return out;
  }

  // ============================================================
  // Forward graph + KV cache + generate
  // ============================================================
  const MAX_SEQ = 4096;   // context + KV-cache + attn-score-buffer size; must hold prompt (tool schemas are large: 9 tools ≈ 2400 tokens) + generation. Prompts beyond this overflow → guarded in _streamIds.
  const LP = CONFIG.weightPrefix;        // 'model.language_model.'
  let _PERF = false, _perfData = null;   // CPU phase profiler (encode vs readback)
  let _DBGLAYERS = false, _layerDbg = null;   // per-layer hidden-state norm capture (debug)
  let _qScaleOn = true;     // apply 1/sqrt(dK) to q after l2norm (debug toggle)
  let _gnormEps = 1e-6;     // gated-RMSNorm eps (debug toggle)
  let _skipDelta = false, _skipAttn = false, _ropeMode = 'partial';   // debug isolation toggles
  let _swapAB = false, _noConv = false, _noL2 = false;   // debug delta-path toggles
  let _dbgLayer = -1, _dbgCap = {};   // snapshot one delta layer's intermediates (CPU-ref debug)
  let _swapQK = false, _deltaMaxLayer = 999;   // debug: swap q/k in recurrence; limit active delta layers
  let _subOverride = null;   // GEMV reduction: null=auto (subgroups if supported), true/false to force
  let _prefillSerial = false;   // debug A/B: force the old token-by-token (drain-every-token) prefill
  let _batchedPrefill = true;   // batched prefill via tiled int4 GEMM (gemmQ) — the prefill speedup path
  let _benchNoSys = false, _benchNoTools = false;   // bench: strip system / tool prompt in runConversation for a 1-to-1 vs the test harness
  let _awqXor = false;       // compressed-tensors packs offset-binary (q+8) = OUR format → read direct, no transcode.
                             // (Toggle exists only for hypothetical two's-complement repos.)
  function _useSub() { return _subOverride !== null ? _subOverride : !!(E.caps && E.caps() && E.caps().hasSubgroups); }

  // Some mobile GPUs ADVERTISE `subgroups` but compute subgroupAdd incorrectly — this was
  // the original mobile all-"!" bug, whose fix was "don't use subgroups." Trusting
  // caps().hasSubgroups (the capability-conditional GEMV path) silently re-introduces it on
  // exactly those devices. So before relying on subgroups, verify the subgroup GEMV path
  // numerically against the portable shared-mem path at load; on ANY mismatch (or error),
  // fall back to shared-mem. Keeps the desktop subgroup speedup where it's actually correct,
  // auto-disables it everywhere it isn't — no user-agent sniffing.
  let _subProbed = false;
  async function probeSubgroups() {
    if (_subProbed) return; _subProbed = true;
    if (_subOverride !== null) return;                                  // user forced a value — respect it
    if (!(E.caps && E.caps() && E.caps().hasSubgroups)) return;         // no subgroups → already shared-mem
    let bufs = [];
    try {
      const N = 40, K = 256;
      const x = new Float32Array(K); for (let i = 0; i < K; i++) x[i] = Math.sin(i * 0.2);
      const Wf = new Float32Array(N * K); for (let i = 0; i < Wf.length; i++) Wf[i] = Math.cos(i * 0.013);
      const u16 = new Uint16Array(Wf.length); const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
      for (let i = 0; i < Wf.length; i++) { t[0] = Wf[i]; u16[i] = ti[0] >>> 16; }   // f32 → bf16 bits
      const { pack, scales } = quantizeInt4Bf16(u16, N, K);
      const xb = f32buf(x);
      const pb = E.createBuffer(pack.byteLength, ST(), 'probe.pk'); E.device().queue.writeBuffer(pb, 0, pack);
      const sb = E.createBuffer(scales.byteLength, ST(), 'probe.sc'); E.device().queue.writeBuffer(sb, 0, scales);
      const yb = E.createBuffer(N * 4, ST(), 'probe.y');
      bufs = [xb, pb, sb, yb];
      _subOverride = true;  await gemvQ(xb, pb, sb, yb, N, K, false); const ySub = Array.from(await E.readF32(yb, N));
      _subOverride = false; await gemvQ(xb, pb, sb, yb, N, K, false); const yRef = Array.from(await E.readF32(yb, N));
      _subOverride = null;                                             // back to auto unless we disable below
      let maxErr = 0, ref = 0;
      for (let i = 0; i < N; i++) { maxErr = Math.max(maxErr, Math.abs(ySub[i] - yRef[i])); ref = Math.max(ref, Math.abs(yRef[i])); }
      const rel = maxErr / (ref || 1);
      if (rel > 1e-2) { _subOverride = false; console.warn('[q35] subgroup GEMV WRONG (rel ' + rel.toFixed(3) + ') — disabling subgroups, using portable shared-mem path'); }
      else console.log('[q35] subgroup GEMV verified (rel ' + rel.toExponential(1) + ') — keeping subgroup path');
    } catch (e) {
      _subOverride = false;                                            // anything goes wrong → safe path
      console.warn('[q35] subgroup probe failed — disabling subgroups:', (e && e.message) || e);
    } finally {
      for (const b of bufs) { try { b.destroy(); } catch (_) {} }
    }
  }
  let _kv = null;          // per full-attn layer: {k,v} sized MAX_SEQ ; null for delta layers
  let _convState = null;   // per delta layer: [convDim*(K-1)] causal-conv ring ; null for attn
  let _deltaS = null;      // per delta layer: [deltaHeads*valDim*keyDim] recurrent state
  let _scr = null;
  let _idsBuf = null, _idsCap = 0;
  let _tokHist = null;     // GPU token history: argmax of pos P writes [P+1].

  // Free all GPU buffers + reset state (so a different variant can load fresh).
  function unload() {
    const free = (b) => { if (b && b.destroy) b.destroy(); };
    if (_weights) { for (const k in _weights) { const w = _weights[k]; free(w.buf); free(w.pack); free(w.scales); } _weights = null; }
    if (_kv) { for (const l of _kv) { if (l) { free(l.k); free(l.v); } } _kv = null; }
    if (_convState) { for (const b of _convState) free(b); _convState = null; }
    if (_deltaS) { for (const b of _deltaS) free(b); _deltaS = null; }
    if (_snap) { for (const x of _snap.kv) { if (x) { free(x.k); free(x.v); } } for (const b of _snap.conv) free(b); for (const b of _snap.S) free(b); _snap = null; }
    _snapIds = null;
    if (_scr) { for (const k in _scr) free(_scr[k]); _scr = null; }
    if (_pscr) { for (const k in _pscr) free(_pscr[k]); _pscr = null; _pscrCap = 0; }   // prefill scratch
    if (_dbgCap) { for (const k in _dbgCap) free(_dbgCap[k]); _dbgCap = {}; }            // debug capture buffers
    // CRUCIAL for device-loss recovery: the pooled uniform buffers are device-bound. If
    // left dangling, after a reload uniform() hands back buffers from the DEAD device and
    // writeBuffer silently no-ops → kernels read garbage dims → "!!!!" output. Drop them so
    // they're recreated on the new device.
    for (const b of _uPool) free(b); _uPool = []; _uIdx = 0;
    free(_tokHist); _tokHist = null;
    free(_idsBuf); _idsBuf = null; _idsCap = 0;
    _loaded = false;
    _warmed = false;   // a fresh load must re-warm the pipelines (device/buffers were torn down)
  }

  function scrBuf(n, label) { return E.createBuffer(n * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, label); }
  function zeroBuf(buf, bytes) { const enc = E.device().createCommandEncoder(); enc.clearBuffer(buf, 0, bytes); E.device().queue.submit([enc.finish()]); }

  const CONV_DIM = 3 * CONFIG.deltaHeads * CONFIG.deltaKeyDim;   // q|k|v concatenated = 6144
  const DVAL = CONFIG.deltaHeads * CONFIG.deltaValDim;           // 2048

  // Persistent per-layer state (allocated once). Full-attn layers get a KV cache;
  // DeltaNet layers get a conv ring + a recurrent matrix state.
  function ensureState() {
    if (_kv) return;
    const C = CONFIG, Km1 = C.convKernel - 1;
    _kv = []; _convState = []; _deltaS = [];
    for (let l = 0; l < C.numLayers; l++) {
      if (C.layerFullAttn[l]) {
        const per = MAX_SEQ * C.nKvHeads * C.headDim;
        _kv.push({ k: scrBuf(per, 'k' + l), v: scrBuf(per, 'v' + l) });
        _convState.push(null); _deltaS.push(null);
      } else {
        _kv.push(null);
        _convState.push(scrBuf(CONV_DIM * Km1, 'conv' + l));
        _deltaS.push(scrBuf(C.deltaHeads * C.deltaValDim * C.deltaKeyDim, 'S' + l));
      }
    }
    _tokHist = E.createBuffer(MAX_SEQ * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'tokHist');
  }
  // Zero conv + recurrent state for a fresh sequence (KV cache need not be cleared:
  // attention only reads positions [0, pos]).
  function resetState() {
    ensureState();
    const C = CONFIG, Km1 = C.convKernel - 1;
    for (let l = 0; l < C.numLayers; l++) {
      if (!C.layerFullAttn[l]) {
        zeroBuf(_convState[l], CONV_DIM * Km1 * 4);
        zeroBuf(_deltaS[l], C.deltaHeads * C.deltaValDim * C.deltaKeyDim * 4);
      }
    }
    _snapIds = null;   // an explicit reset invalidates the prefix-cache snapshot
  }

  // ---- PREFIX CACHE -------------------------------------------------------------
  // Re-prefilling the (constant) system+tools prefix every turn is the dominant cost in
  // agent mode (9 tools ≈ 2400 tokens ≈ 11s). After each prompt prefill we SNAPSHOT the
  // full state (KV + conv ring + recurrent S) — taken BEFORE decode, so decode's batch
  // overshoot / re-tokenization of generated text can never corrupt it. When the next
  // turn's prompt EXTENDS the snapshotted prompt (it does: tools+history are a stable
  // prefix), we RESTORE the snapshot and prefill only the new tokens.
  let _snapIds = null, _snap = null;
  function ensureSnap() {
    if (_snap) return;
    const C = CONFIG, Km1 = C.convKernel - 1;
    _snap = { kv: [], conv: [], S: [] };
    for (let l = 0; l < C.numLayers; l++) {
      if (C.layerFullAttn[l]) { const per = MAX_SEQ * C.nKvHeads * C.headDim; _snap.kv.push({ k: scrBuf(per, 'sk' + l), v: scrBuf(per, 'sv' + l) }); _snap.conv.push(null); _snap.S.push(null); }
      else { _snap.kv.push(null); _snap.conv.push(scrBuf(CONV_DIM * Km1, 'sc' + l)); _snap.S.push(scrBuf(C.deltaHeads * C.deltaValDim * C.deltaKeyDim, 'sS' + l)); }
    }
  }
  // toSaved=true: live → snapshot (after prefill). false: snapshot → live (restore before reuse).
  function snapCopy(toSaved) {
    ensureSnap();
    const C = CONFIG, Km1 = C.convKernel - 1;
    const enc = E.device().createCommandEncoder();
    const cp = (live, saved, bytes) => { const [a, b] = toSaved ? [live, saved] : [saved, live]; enc.copyBufferToBuffer(a, 0, b, 0, bytes); };
    for (let l = 0; l < C.numLayers; l++) {
      if (C.layerFullAttn[l]) { const kb = MAX_SEQ * C.nKvHeads * C.headDim * 4; cp(_kv[l].k, _snap.kv[l].k, kb); cp(_kv[l].v, _snap.kv[l].v, kb); }
      else { cp(_convState[l], _snap.conv[l], CONV_DIM * Km1 * 4); cp(_deltaS[l], _snap.S[l], C.deltaHeads * C.deltaValDim * C.deltaKeyDim * 4); }
    }
    E.device().queue.submit([enc.finish()]);
  }
  // Scratch buffers (single-token decode; T=1 everywhere — prefill loops token-by-token
  // because the DeltaNet recurrence is inherently sequential).
  function ensureScratch() {
    if (_scr) return;
    const C = CONFIG, H = C.hidden, hd = C.headDim, nHq = C.nHeads, nKv = C.nKvHeads, dH = C.deltaHeads, dK = C.deltaKeyDim;
    _scr = {
      x: scrBuf(H, 'x'), normed: scrBuf(H, 'normed'),
      // full-attention
      qproj: scrBuf(nHq * 2 * hd, 'qproj'), kf: scrBuf(nKv * hd, 'kf'), vf: scrBuf(nKv * hd, 'vf'),
      qr: scrBuf(nHq * hd, 'qr'), kr: scrBuf(nKv * hd, 'kr'), attnO: scrBuf(nHq * hd, 'attnO'), gatedO: scrBuf(nHq * hd, 'gatedO'),
      // DeltaNet
      qkv: scrBuf(CONV_DIM, 'qkv'), qkvc: scrBuf(CONV_DIM, 'qkvc'),
      qd: scrBuf(DVAL, 'qd'), kd: scrBuf(DVAL, 'kd'), vd: scrBuf(DVAL, 'vd'),
      qn: scrBuf(DVAL, 'qn'), kn: scrBuf(DVAL, 'kn'),
      aD: scrBuf(dH, 'aD'), bD: scrBuf(dH, 'bD'), zD: scrBuf(DVAL, 'zD'),
      gb: scrBuf(dH * 2, 'gb'), core: scrBuf(DVAL, 'core'), gnorm: scrBuf(DVAL, 'gnorm'),
      dout: scrBuf(H, 'dout'),                  // delta out_proj output (then residual-added)
      convBias: scrBuf(CONV_DIM, 'convBias'),   // Qwen3.5 conv1d has NO bias → kept zero
      // MLP + head
      swi: scrBuf(C.intermediate, 'swi'),
      logits: scrBuf(C.vocab, 'logits'),
    };
    zeroBuf(_scr.convBias, CONV_DIM * 4);
  }
  function copyRange(src, dst, dstFloatOffset, floatCount) {
    E.copyBuffer(src, 0, dst, dstFloatOffset * 4, floatCount * 4);   // batch-aware
  }
  function setIds(arr) {
    if (!_idsBuf || _idsCap < arr.length) { if (_idsBuf) _idsBuf.destroy(); _idsBuf = E.createBuffer(Math.max(16, arr.length * 4), U.STORAGE | U.COPY_DST, 'ids'); _idsCap = arr.length; }
    E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(arr));
    return _idsBuf;
  }

  // Single-token forward at absolute position `pos`. Processes token `tokenId`,
  // updates all per-layer state (KV cache / conv ring / recurrent state), writes
  // the greedy next-token into _tokHist[pos+1], and returns it.
  // Hybrid: each layer is full_attention or gated-DeltaNet (config layer_types);
  // the post-attention RMSNorm + SwiGLU MLP block is identical for both.
  async function forward(tokenId, pos, opts) {
    const chain = !!(opts && opts.chain), submitOnly = !!(opts && opts.submitOnly);
    // Prefill tokens 0..L-2 only need to UPDATE state (KV/conv/delta); their logits are
    // never read. Skip the final norm + 248K-vocab lm_head + argmax for them — that matmul
    // is ~35% of a prefill token's GPU time (the lm_head dominates gemvQ).
    const noLmHead = !!(opts && opts.noLmHead);
    const _t0 = _PERF ? performance.now() : 0;
    const C = CONFIG, H = C.hidden, hd = C.headDim, nHq = C.nHeads, nKv = C.nKvHeads, I = C.intermediate;
    const dH = C.deltaHeads, dK = C.deltaKeyDim, dV = C.deltaValDim;
    const S = pos + 1, qScale = _qScaleOn ? (1 / Math.sqrt(dK)) : 1.0;
    ensureState(); ensureScratch();
    if (_DBGLAYERS) _layerDbg = [];
    const W   = (n) => _weights[LP + n].buf;   // f16/f32 raw buffer
    const Wq  = (n) => _weights[LP + n];        // int4 record
    const s = _scr;
    uniformReset();
    E.beginBatch();
    const embIds = chain ? _tokHist : setIds([tokenId]);
    const embOff = chain ? pos : 0;
    const emb = _weights['__embed_int4'];
    await embedGatherQ(embIds, emb.pack, emb.scales, s.x, 1, H, embOff);
    for (let l = 0; l < C.numLayers; l++) {
      const p = 'layers.' + l + '.';
      await rmsnorm(s.x, W(p + 'input_layernorm.weight'), s.normed, 1, H, C.rmsEps);
      if (C.layerFullAttn[l] && !_skipAttn) {
        const a = p + 'self_attn.';
        const rotDim = _ropeMode === 'none' ? 0 : (_ropeMode === 'full' ? hd : C.rotaryDim);
        await linearQ(s.normed, Wq(a + 'q_proj.weight'), s.qproj, 1, nHq * 2 * hd, H);   // [nHq, query|gate]
        await linearQ(s.normed, Wq(a + 'k_proj.weight'), s.kf, 1, nKv * hd, H);
        await linearQ(s.normed, Wq(a + 'v_proj.weight'), s.vf, 1, nKv * hd, H);
        // q reads the query-half of qproj (stride 2*hd); both get qnorm + partial RoPE.
        await ropeQKNorm35(s.qproj, W(a + 'q_norm.weight'), s.qr, 1, nHq, hd, 2 * hd, rotDim, pos, C.ropeTheta, C.rmsEps);
        await ropeQKNorm35(s.kf, W(a + 'k_norm.weight'), s.kr, 1, nKv, hd, hd, rotDim, pos, C.ropeTheta, C.rmsEps);
        copyRange(s.kr, _kv[l].k, pos * nKv * hd, nKv * hd);
        copyRange(s.vf, _kv[l].v, pos * nKv * hd, nKv * hd);
        await attention(s.qr, _kv[l].k, _kv[l].v, s.attnO, 1, S, nHq, nKv, hd);
        await applyGate35(s.attnO, s.qproj, s.gatedO, nHq, hd);                          // out *= sigmoid(gate)
        await linearQ(s.gatedO, Wq(a + 'o_proj.weight'), s.x, 1, H, nHq * hd, true);     // residual
      } else if (!C.layerFullAttn[l] && !_skipDelta && l <= _deltaMaxLayer) {
        const d = p + 'linear_attn.';
        // DeltaNet projections in f16 (gemv) — int4 noise corrupts the tiny recurrence signal.
        const dproj = (n, y, N) => DELTA_F16 ? gemv(s.normed, W(d + n), y, N, H) : linearQ(s.normed, Wq(d + n), y, 1, N, H);
        await dproj('in_proj_qkv.weight', s.qkv, CONV_DIM);
        if (_noConv) { E.copyBuffer(s.qkv, 0, s.qkvc, 0, CONV_DIM * 4); }                 // debug: bypass conv
        else await conv1dDecode(s.qkv, W(d + 'conv1d.weight'), s.convBias, _convState[l], s.qkvc, CONV_DIM);  // +silu, advances ring
        E.copyBuffer(s.qkvc, 0, s.qd, 0, DVAL * 4);          // split q|k|v (each 16×128)
        E.copyBuffer(s.qkvc, DVAL * 4, s.kd, 0, DVAL * 4);
        E.copyBuffer(s.qkvc, 2 * DVAL * 4, s.vd, 0, DVAL * 4);
        await dproj('in_proj_a.weight', s.aD, dH);
        await dproj('in_proj_b.weight', s.bD, dH);
        await dproj('in_proj_z.weight', s.zD, DVAL);
        if (_swapAB) await gbeta(s.bD, s.aD, W(d + 'A_log'), W(d + 'dt_bias'), s.gb, dH);  // debug: swap a/b
        else await gbeta(s.aD, s.bD, W(d + 'A_log'), W(d + 'dt_bias'), s.gb, dH);          // expg, beta
        let qIn = s.qn, kIn = s.kn;
        if (_noL2) { qIn = s.qd; kIn = s.kd; }                                            // debug: skip l2norm
        else { await l2normHeads(s.qd, s.qn, dH, dK, C.rmsEps, qScale);                   // q L2-normed then *1/sqrt(dK)
               await l2normHeads(s.kd, s.kn, dH, dK, C.rmsEps, 1.0); }
        if (_swapQK) await deltaRecur(kIn, qIn, s.vd, s.gb, _deltaS[l], s.core, dH, dK);   // debug: q/k swapped
        else await deltaRecur(qIn, kIn, s.vd, s.gb, _deltaS[l], s.core, dH, dK);           // recurrence (updates S)
        await gatedRMSNorm(s.core, s.zD, W(d + 'norm.weight'), s.gnorm, dH, dV, _gnormEps);
        if (l === _dbgLayer) {   // snapshot stage outputs for CPU-reference comparison
          const grab = (nm, buf, n) => { if (!_dbgCap[nm]) _dbgCap[nm] = scrBuf(n, 'cap_' + nm); E.copyBuffer(buf, 0, _dbgCap[nm], 0, n * 4); };
          grab('normed', s.normed, H); grab('qkv', s.qkv, CONV_DIM); grab('qkvc', s.qkvc, CONV_DIM);
          grab('qn', s.qn, DVAL); grab('kn', s.kn, DVAL); grab('vd', s.vd, DVAL);
          grab('aD', s.aD, dH); grab('bD', s.bD, dH); grab('zD', s.zD, DVAL);
          grab('gb', s.gb, dH * 2); grab('core', s.core, DVAL); grab('gnorm', s.gnorm, DVAL);
        }
        if (DELTA_F16) { await gemv(s.gnorm, W(d + 'out_proj.weight'), s.dout, H, DVAL); if (l === _dbgLayer) { if(!_dbgCap.dout) _dbgCap.dout=scrBuf(H,'cap_dout'); E.copyBuffer(s.dout,0,_dbgCap.dout,0,H*4);} await addInPlace(s.x, s.dout, H); }  // residual
        else await linearQ(s.gnorm, Wq(d + 'out_proj.weight'), s.x, 1, H, DVAL, true);    // residual
      }
      // shared post-attention RMSNorm + SwiGLU MLP
      await rmsnorm(s.x, W(p + 'post_attention_layernorm.weight'), s.normed, 1, H, C.rmsEps);
      await gateUpSiluQ(s.normed, Wq(p + 'mlp.gate_proj.weight'), Wq(p + 'mlp.up_proj.weight'), s.swi, I, H);
      await linearQ(s.swi, Wq(p + 'mlp.down_proj.weight'), s.x, 1, H, I, true);           // residual
      if (_DBGLAYERS) {
        await E.endBatch(); const xn = await E.readF32(s.x, H);
        let ss = 0, mx = 0; for (const v of xn) { ss += v * v; if (Math.abs(v) > mx) mx = Math.abs(v); }
        _layerDbg.push({ l, t: C.layerFullAttn[l] ? 'A' : 'D', xnorm: +Math.sqrt(ss).toFixed(2), xmax: +mx.toFixed(3) });
        E.beginBatch();
      }
    }
    if (!noLmHead) {
      await rmsnorm(s.x, W('norm.weight'), s.normed, 1, H, C.rmsEps);
      await linearQ(s.normed, _weights['__embed_int4'], s.logits, 1, C.vocab, H);         // tied lm_head (same int4 embed buffer)
      await argmaxKernel(s.logits, _tokHist, C.vocab, pos + 1);
    }
    const _t1 = _PERF ? performance.now() : 0;
    const drain = E.endBatch();
    if (submitOnly) return undefined;
    await drain;
    const _t2 = _PERF ? performance.now() : 0;
    if (noLmHead) return undefined;   // no logits computed — nothing to read
    const tok = await readU32At(_tokHist, pos + 1);
    if (_PERF) _perfData = { encode_ms: +(_t1 - _t0).toFixed(2), gpu_drain_ms: +(_t2 - _t1).toFixed(2), map_ms: +(performance.now() - _t2).toFixed(2) };
    return tok;
  }

  // Read 1 u32 from a GPU buffer at element index idx (waits for the queue).
  async function readU32At(buf, idx) {
    const staging = E.createBuffer(4, U.COPY_DST | U.MAP_READ, 'rdu32');
    const enc = E.device().createCommandEncoder();
    enc.copyBufferToBuffer(buf, (idx || 0) * 4, staging, 0, 4);
    E.device().queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(staging.getMappedRange())[0];
    staging.unmap(); staging.destroy();
    return v;
  }
  // Read `count` u32 starting at element idx — ONE mapAsync for the whole batch.
  async function readU32Range(buf, idx, count) {
    const bytes = count * 4;
    const staging = E.createBuffer(bytes, U.COPY_DST | U.MAP_READ, 'rdrange');
    const enc = E.device().createCommandEncoder();
    enc.copyBufferToBuffer(buf, idx * 4, staging, 0, bytes);
    E.device().queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return out;
  }
  // Debug: full logits readback (call right after a forward, before the next one).
  async function readLogits() { return E.readF32(_scr.logits, CONFIG.vocab); }

  // Greedy generate, PIPELINED + GPU-resident. The decode loop chains through the
  // GPU token history (argmax of pos P writes _tokHist[P+1]; embed at P+1 reads it),
  // so the CPU never round-trips the token mid-loop. We submit the next forward
  // (CPU encode, no drain wait) BEFORE reading the current token, so encode(N+1)
  // overlaps GPU-run(N) — un-doing the serialization batching introduced.
  const STOP = (t) => t === SPECIAL.im_end || t === SPECIAL.endoftext;
  let _genBatch = 8;   // tokens per GPU-resident decode batch (1 readback per batch); tunable for A/B

  // Core greedy decode over a prepared prompt id list. Prefill feeds the prompt
  // token-by-token (the DeltaNet recurrence is sequential), then a batched GPU-resident
  // decode loop chains GEN_BATCH forwards per readback. onToken(piece) gets each decoded
  // token. Caller must resetState() + loadModel() first. Returns the output token ids.
  // ============================================================
  // BATCHED PREFILL — process the whole prompt as GEMM over T tokens (each weight read
  // ONCE for all T), instead of L per-token GEMV forwards (each weight read L times). On
  // this bandwidth-bound iGPU that's the ~order-of-magnitude prefill win. The DeltaNet
  // recurrence stays sequential (a per-token deltaRecur loop) — everything else is batched.
  // ============================================================
  const SPLIT_WGSL = `
struct P { T:u32, dval:u32, convDim:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       qkvc : array<f32>;   // [T*convDim]  (q|k|v per token)
@group(0) @binding(1) var<storage, read_write> qd   : array<f32>;   // [T*dval]
@group(0) @binding(2) var<storage, read_write> kd   : array<f32>;   // [T*dval]
@group(0) @binding(3) var<storage, read_write> vd   : array<f32>;   // [T*dval]
@group(0) @binding(4) var<uniform>             p    : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x; let total=p.T*p.dval; if(i>=total){return;}
  let t=i/p.dval; let j=i%p.dval; let cb=t*p.convDim + j;
  qd[i]=qkvc[cb]; kd[i]=qkvc[cb+p.dval]; vd[i]=qkvc[cb+2u*p.dval];
}`;
  function splitQKV(qkvcBuf, qdBuf, kdBuf, vdBuf, T) {
    const pipe = E.getPipeline('q35.split', SPLIT_WGSL);
    const p = uniform(new Uint32Array([T, DVAL, CONV_DIM, 0]));
    return E.dispatch(pipe, [qkvcBuf, qdBuf, kdBuf, vdBuf, p], [Math.ceil(T * DVAL / 64), 1, 1]);
  }

  let _pscr = null, _pscrCap = 0;
  function ensurePScr(T) {
    if (_pscr && _pscrCap >= T) return;
    if (_pscr) for (const k in _pscr) { try { _pscr[k].destroy(); } catch (_) {} }
    const C = CONFIG, H = C.hidden, hd = C.headDim, nHq = C.nHeads, nKv = C.nKvHeads, I = C.intermediate, dH = C.deltaHeads;
    const b = (e, nm) => scrBuf(T * e, 'p_' + nm);
    _pscr = {
      x: b(H, 'x'), normed: b(H, 'normed'),
      qproj: b(nHq * 2 * hd, 'qproj'), kf: b(nKv * hd, 'kf'), vf: b(nKv * hd, 'vf'),
      qr: b(nHq * hd, 'qr'), kr: b(nKv * hd, 'kr'), attnO: b(nHq * hd, 'attnO'), gatedO: b(nHq * hd, 'gatedO'),
      qkv: b(CONV_DIM, 'qkv'), qkvc: b(CONV_DIM, 'qkvc'),
      qd: b(DVAL, 'qd'), kd: b(DVAL, 'kd'), vd: b(DVAL, 'vd'), qn: b(DVAL, 'qn'), kn: b(DVAL, 'kn'),
      aD: b(dH, 'aD'), bD: b(dH, 'bD'), zD: b(DVAL, 'zD'), gb: b(dH * 2, 'gb'), core: b(DVAL, 'core'), gnorm: b(DVAL, 'gnorm'),
      gate: b(I, 'gate'), up: b(I, 'up'), swi: b(I, 'swi'),
    };
    _pscrCap = T;
  }

  // One batched chunk of T tokens at absolute positions [posBase, posBase+T). State
  // (KV cache / conv ring / delta S) carries across chunks. needHead → also run the final
  // norm + lm_head + argmax on the LAST token and return the first generated token id.
  async function forwardChunk(ids, posBase, needHead) {
    const C = CONFIG, H = C.hidden, hd = C.headDim, nHq = C.nHeads, nKv = C.nKvHeads, I = C.intermediate;
    const dH = C.deltaHeads, dK = C.deltaKeyDim, dV = C.deltaValDim;
    const T = ids.length, qScale = _qScaleOn ? (1 / Math.sqrt(dK)) : 1.0;
    const rotDim = C.rotaryDim;
    ensureState(); ensureScratch(); ensurePScr(T);
    const W = (n) => _weights[LP + n].buf, Wq = (n) => _weights[LP + n];
    const ps = _pscr, s = _scr, emb = _weights['__embed_int4'];
    uniformReset();
    E.beginBatch();
    setIds(ids);
    await embedGatherQ(_idsBuf, emb.pack, emb.scales, ps.x, T, H, 0);
    for (let l = 0; l < C.numLayers; l++) {
      const p = 'layers.' + l + '.';
      await rmsnorm(ps.x, W(p + 'input_layernorm.weight'), ps.normed, T, H, C.rmsEps);
      if (C.layerFullAttn[l]) {
        const a = p + 'self_attn.';
        await linearQ(ps.normed, Wq(a + 'q_proj.weight'), ps.qproj, T, nHq * 2 * hd, H);
        await linearQ(ps.normed, Wq(a + 'k_proj.weight'), ps.kf, T, nKv * hd, H);
        await linearQ(ps.normed, Wq(a + 'v_proj.weight'), ps.vf, T, nKv * hd, H);
        await ropeQKNorm35(ps.qproj, W(a + 'q_norm.weight'), ps.qr, T, nHq, hd, 2 * hd, rotDim, posBase, C.ropeTheta, C.rmsEps);
        await ropeQKNorm35(ps.kf, W(a + 'k_norm.weight'), ps.kr, T, nKv, hd, hd, rotDim, posBase, C.ropeTheta, C.rmsEps);
        copyRange(ps.kr, _kv[l].k, posBase * nKv * hd, T * nKv * hd);
        copyRange(ps.vf, _kv[l].v, posBase * nKv * hd, T * nKv * hd);
        await attention(ps.qr, _kv[l].k, _kv[l].v, ps.attnO, T, posBase + T, nHq, nKv, hd);
        await applyGate35(ps.attnO, ps.qproj, ps.gatedO, nHq, hd, T);
        await linearQ(ps.gatedO, Wq(a + 'o_proj.weight'), ps.x, T, H, nHq * hd, true);
      } else {
        const d = p + 'linear_attn.';
        await linearQ(ps.normed, Wq(d + 'in_proj_qkv.weight'), ps.qkv, T, CONV_DIM, H);
        await conv1dPrefill(ps.qkv, W(d + 'conv1d.weight'), s.convBias, _convState[l], ps.qkvc, CONV_DIM, T);
        await splitQKV(ps.qkvc, ps.qd, ps.kd, ps.vd, T);
        await linearQ(ps.normed, Wq(d + 'in_proj_a.weight'), ps.aD, T, dH, H);
        await linearQ(ps.normed, Wq(d + 'in_proj_b.weight'), ps.bD, T, dH, H);
        await linearQ(ps.normed, Wq(d + 'in_proj_z.weight'), ps.zD, T, DVAL, H);
        await gbeta(ps.aD, ps.bD, W(d + 'A_log'), W(d + 'dt_bias'), ps.gb, dH, T);
        await l2normHeads(ps.qd, ps.qn, dH, dK, C.rmsEps, qScale, T);
        await l2normHeads(ps.kd, ps.kn, dH, dK, C.rmsEps, 1.0, T);
        await deltaRecur(ps.qn, ps.kn, ps.vd, ps.gb, _deltaS[l], ps.core, dH, dK, T);   // whole chunk, 1 dispatch (loops t internally)
        await gatedRMSNorm(ps.core, ps.zD, W(d + 'norm.weight'), ps.gnorm, dH, dV, _gnormEps, T);
        await linearQ(ps.gnorm, Wq(d + 'out_proj.weight'), ps.x, T, H, DVAL, true);
      }
      await rmsnorm(ps.x, W(p + 'post_attention_layernorm.weight'), ps.normed, T, H, C.rmsEps);
      await linearQ(ps.normed, Wq(p + 'mlp.gate_proj.weight'), ps.gate, T, I, H);
      await linearQ(ps.normed, Wq(p + 'mlp.up_proj.weight'), ps.up, T, I, H);
      await swiglu(ps.gate, ps.up, ps.swi, T * I);
      await linearQ(ps.swi, Wq(p + 'mlp.down_proj.weight'), ps.x, T, H, I, true);
    }
    if (needHead) {
      E.copyBuffer(ps.x, (T - 1) * H * 4, s.x, 0, H * 4);   // last token → T=1 head path
      await rmsnorm(s.x, W('norm.weight'), s.normed, 1, H, C.rmsEps);
      await linearQ(s.normed, emb, s.logits, 1, C.vocab, H);
      await argmaxKernel(s.logits, _tokHist, C.vocab, posBase + T);
    }
    await E.endBatch();
    return needHead ? await readU32At(_tokHist, posBase + T) : undefined;
  }

  const PCHUNK = 128;   // tokens per batched prefill chunk (bounds scratch memory + command-buffer size)
  // Batched prefill of ids[startPos..]; returns the first generated token id. startPos>0 =
  // prefix-cache reuse (the state for [0,startPos) was restored from the snapshot).
  async function forwardPrefill(ids, signal, startPos = 0) {
    const L = ids.length;
    let tok;
    for (let c = startPos; c < L; c += PCHUNK) {
      if (signal && signal.aborted) return undefined;
      const end = Math.min(c + PCHUNK, L);
      tok = await forwardChunk(ids.slice(c, end), c, end === L);   // last chunk computes the head
    }
    return tok;
  }

  // WARMUP — eliminate the cold-start shader-compile tax on the user's FIRST message.
  // Every kernel uses a constant pipeline cache key (q35.gemmQ, q3.gemvQ, …) with N/K/T
  // passed as uniforms, so there are only ~25 distinct pipelines for the whole model.
  // But createComputePipeline is SYNCHRONOUS (webgpu-engine getPipeline): the first time
  // each is dispatched, the driver compiles WGSL→native on the main thread — hundreds of
  // ms each on Intel iGPUs. Lazily, that ENTIRE compile bill lands on the first real
  // prefill (the warm bench never pays it — that's why turn-1 looked ~2× slower than the
  // 220 tok/s bench). Here we force every pipeline to compile NOW, during load, by running
  // one tiny prefill chunk (warms the batched GEMM, attention, DeltaNet, conv-prefill,
  // embed-gather, final-norm, lm_head + argmax) plus two decode steps (warms the decode
  // gemvQ + chained embed path). State touched by the dummy run is wiped by resetState().
  let _warmed = false;
  async function warmup(onProgress) {
    if (_warmed) return; _warmed = true;
    try {
      onProgress && onProgress({ phase: 'warmup', pct: 0 });
      ensureState(); ensureScratch();
      const T = 8, ids = []; for (let i = 0; i < T; i++) ids.push(i + 1);   // valid, < vocab
      await forwardChunk(ids, 0, true);            // compiles every prefill-path pipeline + head
      await forward(null, T, { chain: true, submitOnly: true });   // decode gemvQ + chained embed
      await forward(null, T + 1, { chain: true, submitOnly: true });
      await E.device().queue.onSubmittedWorkDone();   // ensure all compiles + dispatches flushed
      try { await readU32Range(_tokHist, T + 1, 1); } catch (_) {}   // warm the readback/staging path
      resetState();   // wipe conv/delta state the dummy run dirtied (KV is overwritten from pos 0 anyway); nulls the snapshot
      onProgress && onProgress({ phase: 'warmup', pct: 100 });
    } catch (e) { try { console.warn('[q35] warmup skipped:', (e && e.message) || e); } catch (_) {} }
  }

  const PF_BATCH = 16;   // prefill: submit this many forwards before draining (bounds queue depth)
  async function _streamIds(ids, { maxTokens = 256, onToken, signal } = {}) {
    const L = ids.length;
    // Hard guard: a prompt at/over the context length overflows the KV cache + attention
    // score buffer and silently produces garbage ("!"). Surface a clear error instead.
    // (Tool schemas are large — ~9 tools ≈ 2400 tokens — so this is reachable in agent mode.)
    if (L >= MAX_SEQ) throw new Error('prompt is ' + L + ' tokens but the WebGPU context is ' + MAX_SEQ + ' — reduce the number of tools or shorten the conversation.');
    // PREFIX CACHE: if this prompt extends the snapshotted one (tools+history are a stable
    // prefix every turn), restore that state and prefill only the new tail. Else reset + full.
    let startPos = 0;
    const canReuse = _snapIds && _snapIds.length >= 1 && _snapIds.length < L && (() => { for (let i = 0; i < _snapIds.length; i++) if (_snapIds[i] !== ids[i]) return false; return true; })();
    if (canReuse) { snapCopy(false); startPos = _snapIds.length; }   // restore snapshot, prefill from here
    else resetState();
    let tok = 0;
    const _tStart = (typeof performance !== 'undefined') ? performance.now() : 0;   // bench instrumentation
    if (_batchedPrefill && !_prefillSerial) {
      // Batched tiled-GEMM prefill of ids[startPos..].
      tok = await forwardPrefill(ids, signal, startPos);
      if (signal && signal.aborted) return [];
    } else {
      // Per-token fallback (A/B via _setPrefillSerial), pipelined from startPos.
      for (let i = startPos; i < L; i++) {
        if (signal && signal.aborted) return [];
        const last = i === L - 1;
        const sync = _prefillSerial || last || ((i + 1) % PF_BATCH === 0);
        const opts = { noLmHead: !last, submitOnly: !sync };
        if (sync) tok = await forward(ids[i], i, opts);
        else await forward(ids[i], i, opts);
      }
    }
    // Snapshot the post-prefill state (BEFORE decode) so the next turn can reuse this prefix.
    if (!signal || !signal.aborted) { try { snapCopy(true); _snapIds = ids.slice(0, L); } catch (_) { _snapIds = null; } }
    const _tFirst = (typeof performance !== 'undefined') ? performance.now() : 0;   // first token ready = bench's t0 (excludes prefill)
    const outIds = []; let pos = L;
    const emit = (t) => { if (STOP(t)) return false; outIds.push(t); if (onToken) { try { onToken(TOK.decode([t])); } catch (_) {} } return true; };
    if (!emit(tok)) { _recordDecodeStats(L, startPos, _tStart, _tFirst, _tFirst, outIds.length); return outIds; }
    while (outIds.length < maxTokens && pos + 1 < MAX_SEQ) {
      if (signal && signal.aborted) break;
      const K = Math.min(_genBatch, maxTokens - outIds.length, MAX_SEQ - 1 - pos);
      if (K <= 0) break;
      for (let k = 0; k < K; k++) await forward(null, pos + k, { chain: true, submitOnly: true });
      const toks = await readU32Range(_tokHist, pos + 1, K);   // one readback for the whole batch
      pos += K;
      let brk = false; for (let k = 0; k < K; k++) { if (!emit(toks[k])) { brk = true; break; } }
      if (brk) break;
    }
    _recordDecodeStats(L, startPos, _tStart, _tFirst, (typeof performance !== 'undefined') ? performance.now() : 0, outIds.length);
    return outIds;
  }
  // Decode-only throughput, measured the SAME way as the test harness (timer starts at the
  // first token, so prefill is excluded). Exposed via decodeStats(); logged when _benchLog.
  // This is the apples-to-apples number to compare against the bench HTML — the UI's "tok/s"
  // is TOTAL-time (load+prefill+decode) and is NOT comparable.
  let _decodeStats = null, _benchLog = true;
  function _recordDecodeStats(ctxLen, startPos, tStart, tFirst, tEnd, totalOut) {
    const prefillMs = tFirst - tStart;
    const decodeMs = tEnd - tFirst;
    const decodeTokens = Math.max(0, totalOut - 1);   // tokens after the first (the bench counts the same way)
    const decodeTokps = decodeMs > 0 ? (decodeTokens / (decodeMs / 1000)) : 0;
    const prefillTokens = ctxLen - startPos;
    _decodeStats = {
      ctxLen, prefillTokens, prefillCached: startPos,
      prefillMs: Math.round(prefillMs), prefillTokps: prefillMs > 0 ? +(prefillTokens / (prefillMs / 1000)).toFixed(1) : 0,
      decodeTokens, decodeMs: Math.round(decodeMs), decodeTokps: +decodeTokps.toFixed(1),
    };
    if (_benchLog) { try { console.log('[q35 bench] ctx=' + ctxLen + ' (prefill ' + prefillTokens + ' tok' + (startPos ? ', ' + startPos + ' cached' : '') + ' in ' + _decodeStats.prefillMs + 'ms = ' + _decodeStats.prefillTokps + ' tok/s) | DECODE ' + decodeTokens + ' tok in ' + _decodeStats.decodeMs + 'ms = ' + _decodeStats.decodeTokps + ' tok/s'); } catch (_) {} }
  }

  // Simple chat generate (single user turn). Returns the full decoded string.
  async function generate(prompt, { maxTokens = 64, onToken, signal, system } = {}) {
    await loadModel({});
    // (no resetState here — _streamIds reuses the prefix-cache snapshot when the prompt extends it)
    const msgs = [];
    if (system) msgs.push({ role: 'system', content: system });
    msgs.push({ role: 'user', content: prompt });
    const outIds = await _streamIds(TOK.encodeChat(msgs), { maxTokens, onToken, signal });
    return TOK.decode(outIds);
  }

  // Incremental splitter for Qwen3.5's "<think>…</think>answer" output: text inside the
  // think tags → onReason (live Thinking box), the rest → onContent. Holds an 8-char
  // tail so a tag split across token pieces is still detected.
  function makeThinkSplitter(onReason, onContent) {
    let buf = '', inThink = false; const acc = { content: '', reasoning: '' };
    const out = (text) => { if (!text) return; if (inThink) { acc.reasoning += text; onReason(text); } else { acc.content += text; onContent(text); } };
    const step = () => {
      for (;;) { const needle = inThink ? '</think>' : '<think>'; const idx = buf.indexOf(needle);
        if (idx === -1) break; out(buf.slice(0, idx)); buf = buf.slice(idx + needle.length); inThink = !inThink; }
      if (buf.length > 8) { out(buf.slice(0, buf.length - 8)); buf = buf.slice(buf.length - 8); }   // keep a tag-length guard tail
    };
    return { push(t) { buf += t; step(); }, flush() { out(buf); buf = ''; }, get content() { return acc.content; }, get reasoning() { return acc.reasoning; } };
  }

  // Streaming parser for a tool-calling round. Three states: normal text → onContent,
  // <think>…</think> → onReason, <tool_call>…</tool_call> → buffered into toolCalls[]
  // (NOT streamed — the JSON is machine payload, not user-visible). Holds a 12-char guard
  // tail (longest needle is "</tool_call>") so a tag split across token pieces is detected.
  function makeRoundParser(onReason, onContent) {
    let buf = '', state = 'normal', cur = '';   // state ∈ normal|think|tool
    const acc = { content: '', reasoning: '', toolCalls: [] };
    const NEEDLES = { normal: ['<think>', '<tool_call>'], think: ['</think>'], tool: ['</tool_call>'] };
    const out = (text) => {
      if (!text) return;
      if (state === 'think') { acc.reasoning += text; onReason(text); }
      else if (state === 'tool') { cur += text; }
      else { acc.content += text; onContent(text); }
    };
    const step = () => {
      for (;;) {
        let bi = -1, bn = null;
        for (const n of NEEDLES[state]) { const idx = buf.indexOf(n); if (idx !== -1 && (bi === -1 || idx < bi)) { bi = idx; bn = n; } }
        if (bi === -1) break;
        out(buf.slice(0, bi)); buf = buf.slice(bi + bn.length);
        if (bn === '<think>') state = 'think';
        else if (bn === '</think>') state = 'normal';
        else if (bn === '<tool_call>') { state = 'tool'; cur = ''; }
        else if (bn === '</tool_call>') { state = 'normal'; if (cur.trim()) acc.toolCalls.push(cur.trim()); cur = ''; }
      }
      if (buf.length > 12) { out(buf.slice(0, buf.length - 12)); buf = buf.slice(buf.length - 12); }
    };
    return {
      push(t) { buf += t; step(); },
      flush() { out(buf); buf = ''; if (state === 'tool' && cur.trim()) acc.toolCalls.push(cur.trim()); },
      get content() { return acc.content; }, get reasoning() { return acc.reasoning; }, get toolCalls() { return acc.toolCalls; },
    };
  }

  // ============================================================
  // Host contract — selectable backend in sandpie (see providers.js / conversations.js).
  // Chat + agentic tool-calling (Hermes-style <tool_call>); see runConversation.
  // ============================================================
  const DEFAULT_N_CTX = MAX_SEQ;
  const DEFAULT_MODELS = [
    { id: '0.8B',   modelId: '0.8B',   label: 'Qwen3.5-0.8B int4 (hybrid DeltaNet, ~1.75GB DL)' },
    { id: '2B-AWQ', modelId: '2B-AWQ', label: 'Qwen3.5-2B int4 (pre-quantized, ~2.5GB DL — recommended)' },
    { id: '2B',     modelId: '2B',     label: 'Qwen3.5-2B (bf16→int4 in-browser, ~4.5GB DL)' },
  ];
  // Delete the OPFS-cached model files (called by the Settings "clear local models" button).
  async function clearCache() {
    try { const root = await navigator.storage.getDirectory();
      for (const v of Object.keys(VARIANTS)) {
        try { await root.removeEntry('q35-' + v + '-model.safetensors'); } catch (_) {}
        try { await root.removeEntry('q35-' + v + '-q4v' + QCACHE_VER + '.bin'); } catch (_) {}   // quantized-weights cache
      } } catch (_) {}
  }

  // Page-side agent run (mirrors the wllama/litertlm/transformers.js contract). Streams
  // the same event protocol conversations.js expects. Supports the agentic tool loop:
  // stream a round → if the model emitted <tool_call>s, run them via ./sandpie-tool →
  // append results → re-prefill the whole history and stream again, up to MAX_ROUNDS.
  // (The small hybrid models are unreliable tool-callers — 2B is the realistic floor.)
  async function runConversation({ provider, messages, systemPrompt, tools, convId, signal }, emit) {
    // Single active local backend: free the OTHER local LLMs' GPU/WASM contexts first.
    try { await window.SandpieWllama?.unload?.(); } catch (_) {}
    try { await window.SandpieTransformersJS?.unload?.(); } catch (_) {}
    try { await window.SandpieLiteRTLM?.unload?.(); } catch (_) {}
    const variant = (provider && provider.endpoint) || '0.8B';
    const maxTokens = (provider && (provider.maxTokens | 0)) || 512;
    try {
      emit({ type: 'info', message: 'Loading Qwen3.5 (' + variant + ') locally (WebGPU)… first run downloads the weights.' });
      let lastPct = -1;
      await loadModel({ variant, onProgress: (p) => {
        if (!p) return;
        if (p.phase === 'download') { if (p.pct === lastPct) return; lastPct = p.pct; emit({ type: 'info', message: p.prebuilt ? `Downloading prequantized weights… ${p.pct}%` : `Downloading… ${p.pct}% (${(p.recv / 1e9).toFixed(2)}GB)` }); }
        else if (p.phase === 'cache') emit({ type: 'info', message: 'Loading from cache…' });
        else if (p.phase === 'parse') emit({ type: 'info', message: 'Preparing weights… ' + (p.pct || 0) + '%' });
        else if (p.phase === 'warmup') emit({ type: 'info', message: 'Warming up GPU…' });
      } });
    } catch (e) {
      emit({ type: 'info', message: null });
      if (e && e.name === 'AbortError') throw e;
      emit({ type: 'error', message: 'webgpu: ' + ((e && e.message) || e) }); emit({ type: 'agent_done' }); return;
    }

    const toolList = Array.isArray(tools) ? tools : [];
    // Flatten content to a string (arrays of text parts → joined) but keep tool_calls /
    // tool_call_id so the agentic loop can replay them through encodeChat.
    const norm = (m) => {
      let c = m.content;
      if (Array.isArray(c)) c = c.filter(p => p && p.type === 'text').map(p => p.text || '').join('\n');
      else if (c != null && typeof c !== 'string') c = String(c);
      const o = { role: m.role, content: c == null ? '' : c };
      if (m.tool_calls) o.tool_calls = m.tool_calls;
      if (m.tool_call_id) o.tool_call_id = m.tool_call_id;
      return o;
    };
    const sys = (systemPrompt && typeof systemPrompt === 'object') ? (systemPrompt.content || '') : (systemPrompt || '');
    const work = [];
    if (sys && !_benchNoSys) work.push({ role: 'system', content: sys });   // _benchNoSys: drop system prompt for a 1-to-1 vs the bench
    for (const m of (messages || [])) { if (m && m.role) work.push(norm(m)); }

    const benchTools = _benchNoTools ? [] : toolList;   // _benchNoTools: drop the (~2400-token) tool preamble
    const MAX_ROUNDS = benchTools.length ? 8 : 1;
    try {
      for (let round = 0; round < MAX_ROUNDS; round++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        // (no resetState — _streamIds prefix-cache reuses the stable system+tools+history prefix)
        emit({ type: 'round_start' });
        let firstTok = false;
        const clearInfo = () => { if (!firstTok) { firstTok = true; emit({ type: 'info', message: null }); } };
        const parser = makeRoundParser(
          (rz) => { clearInfo(); emit({ type: 'delta', delta: { reasoning: rz } }); },
          (ct) => { clearInfo(); emit({ type: 'delta', delta: { content: ct } }); },
        );
        const ids = TOK.encodeChat(work, { tools: benchTools.length ? benchTools : null });
        await _streamIds(ids, { maxTokens, signal, onToken: (piece) => parser.push(piece) });
        parser.flush();
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        emit({ type: 'info', message: null });

        const content = parser.content.replace(/^\s+/, '');
        const toolCalls = [];
        parser.toolCalls.forEach((raw, k) => {
          try { const o = JSON.parse(raw); if (o && o.name) toolCalls.push({ id: 'call_' + round + '_' + k, type: 'function', function: { name: o.name, arguments: JSON.stringify(o.arguments || {}) } }); } catch (_) {}
        });

        emit({ type: 'round_end', content });
        const asst = { role: 'assistant', content };
        if (toolCalls.length) asst.tool_calls = toolCalls;
        work.push(asst);
        emit({ type: 'message_added', message: asst });
        if (!toolCalls.length) break;   // no tools → turn complete

        for (const tc of toolCalls) {
          if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
          emit({ type: 'tool_started', tc });
          let args = {}; try { args = JSON.parse(tc.function.arguments || '{}'); } catch (_) {}
          let out;
          try {
            const res = await fetch('./sandpie-tool', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: tc.function.name, args, conversation_file_name: convId }), signal,
            });
            out = res.ok ? await res.json() : { result: 'Error: tool endpoint ' + res.status + ' — service worker not ready (reload once).' };
          } catch (e) {
            if (e && e.name === 'AbortError') throw e;
            out = { result: 'Error: ' + ((e && e.message) || e) };
          }
          const toolResult = (out && out.result != null) ? out.result : '';
          emit({ type: 'tool_result', id: tc.id, result: toolResult, artifacts: out && out.artifacts });
          const toolMsg = { role: 'tool', tool_call_id: tc.id, content: toolResult };
          work.push(toolMsg);
          emit({ type: 'message_added', message: toolMsg });
        }
      }
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      emit({ type: 'info', message: null });
      emit({ type: 'error', message: 'webgpu: ' + ((e && e.message) || e) });
    }
    emit({ type: 'agent_done' });
  }

  return {
    CONFIG,
    rmsnorm, linearT, gemv, linear, embedGather, ropeQK, attention, swiglu, addInPlace,
    selfTestKernels,
    TOK, loadModel, forward, generate, readLogits, isLoaded: () => _loaded,
    selectModel, unload, variant: () => _variant, VARIANTS,
    exportQuantCache,   // save the OPFS q4v to a file → host it → set q35_q4v_<variant> to skip quantize for everyone
    warmup,             // force pipeline compilation (normally auto-run by loadModel)
    // Bench: strip system/tool prompt in runConversation so the app path matches the test
    // harness 1-to-1, + decode-only throughput (the UI tok/s is total-time, not comparable).
    decodeStats: () => _decodeStats,
    _setBench: (o) => { if (o && 'noSys' in o) _benchNoSys = !!o.noSys; if (o && 'noTools' in o) _benchNoTools = !!o.noTools; if (o && 'log' in o) _benchLog = !!o.log; return { noSys: _benchNoSys, noTools: _benchNoTools, log: _benchLog }; },
    // host contract (sandpie backend):
    DEFAULT_MODELS, DEFAULT_N_CTX, runConversation, clearCache,
    _setMatvec: (b) => { _USE_MATVEC = !!b; },
    _setPerf: (b) => { _PERF = !!b; }, _perf: () => _perfData,
    __reset: () => resetState(),
    _setDbgLayers: (b) => { _DBGLAYERS = !!b; }, _layerDbg: () => _layerDbg,
    _setQScale: (b) => { _qScaleOn = !!b; }, _setGnormEps: (v) => { _gnormEps = v; },
    _setSkip: (delta, attn) => { _skipDelta = !!delta; _skipAttn = !!attn; }, _setRope: (m) => { _ropeMode = m; },
    _setSwapAB: (b) => { _swapAB = !!b; }, _setNoConv: (b) => { _noConv = !!b; }, _setNoL2: (b) => { _noL2 = !!b; },
    _setDbgLayer: (l) => { _dbgLayer = l; }, _dbgCapRead: async (nm, n) => _dbgCap[nm] ? Array.from(await E.readF32(_dbgCap[nm], n)) : null,
    _setSwapQK: (b) => { _swapQK = !!b; }, _setDeltaMaxLayer: (l) => { _deltaMaxLayer = l; },
    _setSub: (b) => { _subOverride = b; }, _caps: () => (E.caps ? E.caps() : null),
    _setPrefillSerial: (b) => { _prefillSerial = !!b; },
    _setBatchedPrefill: (b) => { _batchedPrefill = !!b; },
    _probeSubgroups: async () => { _subProbed = false; await E.init(); await probeSubgroups(); return { subOverride: _subOverride, useSub: _useSub() }; },
    _setGenBatch: (k) => { _genBatch = k; }, _setAwqXor: (b) => { _awqXor = !!b; },
    _weightFull: async (name, n) => { const w = _weights[CONFIG.weightPrefix + name]; if (!w) return null; return w.f32 ? Array.from(await E.readF32(w.buf, n)) : Array.from(await readF16(w.buf, n)); },
    _dbgScr: async (name, n) => E.readF32(_scr[name], n || 64),
    _dbgState: async (l, n) => E.readF32(_deltaS[l], n || 64),
    _weightStat: async (name, n) => { const w = _weights[CONFIG.weightPrefix + name]; if (!w) return 'missing'; if (w.f32) return Array.from(await E.readF32(w.buf, n||8)); if (w.int4) return 'int4 ' + JSON.stringify(w.shape); return Array.from(await readF16(w.buf, n||8)); },
    _dbg: {
      weight: async (name, n) => readF16(_weights[name].buf, n || _weights[name].numel),
      weightInfo: (name) => ({ shape: _weights[name].shape, numel: _weights[name].numel }),
      names: () => Object.keys(_weights || {}),
    },
  };
})();

if (typeof window !== 'undefined') window.SandpieQwen35 = SandpieQwen35;
