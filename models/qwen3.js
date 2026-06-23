// sandpie/modules/webgpu-qwen3.js — Qwen3 model layer for the hand-written
// WebGPU engine. Depends on window.SandpieWebGPU (the generic core in
// webgpu-engine.js) for GPU init, buffers, pipeline cache and dispatch.
//
// This file owns the MODEL-SPECIFIC kernels and (later) the forward graph,
// weight loader and tokenizer. See [[project_sandpie-webgpu-engine]].
//
// STATUS: kernel set + per-kernel CPU-reference self-tests. NO forward graph /
// loader / tokenizer yet — those assemble once every kernel here is green.
// Everything is f32 for first-correctness; f16/subgroup fusion comes after we
// produce a correct token.
//
// Qwen3-0.6B (verified from config.json + safetensors header):
//   28 layers · hidden 1024 · 16 q-heads / 8 kv-heads · head_dim 128 (explicit:
//   16*128=2048≠1024) · intermediate 3072 · vocab 151936 · tied embeds ·
//   RoPE θ=1e6 (full rotary) · RMSNorm eps 1e-6 · SwiGLU(silu) · no bias ·
//   per-head q_norm/k_norm (RMSNorm over head_dim, applied BEFORE RoPE).
//   Linear weights stored [out,in] row-major → y = x · Wᵀ.

const SandpieQwen3 = (function () {
  'use strict';

  const E = (typeof window !== 'undefined') ? window.SandpieWebGPU : null;

  // Dense Qwen3 (Qwen3ForCausalLM) ONLY — same arch family, just bigger. NOTE:
  // there is NO dense "Qwen3-2B": Qwen/Qwen3-2B is a Qwen3.5-VL multimodal model
  // with gated-DeltaNet (linear_attention) layers + head_dim 256, which this
  // engine can't run. The closest dense, compatible model is Qwen3-1.7B.
  const CONFIGS = {
    '0.6B': { numLayers:28, hidden:1024, nHeads:16, nKvHeads:8, headDim:128, intermediate:3072, vocab:151936, ropeTheta:1000000, rmsEps:1e-6, tieEmbeddings:true },
    '1.7B': { numLayers:28, hidden:2048, nHeads:16, nKvHeads:8, headDim:128, intermediate:6144, vocab:151936, ropeTheta:1000000, rmsEps:1e-6, tieEmbeddings:true },
  };
  // Mutable in-place so all existing CONFIG.xxx references stay valid after variant switch.
  const CONFIG = Object.assign({}, CONFIGS['0.6B']);

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
  loop { if (i >= H) { break; } y[base+i] = x[base+i]*inv*f32(w[i]); i = i + ${WG_H}u; }
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
  // Vectorized + subgroup-reduced + N_ROWS reuse. Reads weights as vec4<f16> and
  // x as vec4<f32> (4 elems/load), reduces partials with subgroupAdd, and each
  // workgroup computes GEMV_NR output rows — the activation chunk x[c] is read ONCE
  // and reused across all NR rows (fewer workgroups, the activation read amortized
  // NR×). Requires K % 4 == 0 (all Qwen3 matrices). N need not divide NR (row<N
  // guards; out-of-range weight reads are bounds-checked to 0 and never written).
  const GEMV_WG = 64;
  const GEMV_NR = 4;   // output rows per workgroup
  const GEMV_WGSL = `
enable f16;
enable subgroups;
struct D { N:u32, K:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       x : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> part : array<f32, ${GEMV_NR * GEMV_WG}>;   // part[r*WG + subgroupIdx]
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
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
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GEMV_NR}u; r=r+1u) {
    let ssum = subgroupAdd(acc[r]);
    if (sgi == 0u) { part[r*${GEMV_WG}u + sgIdx] = ssum; }
  }
  workgroupBarrier();
  // NR threads each finalize one row.
  if (lid.x < ${GEMV_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.N) {
      let nsg = (${GEMV_WG}u + sgs - 1u) / sgs;
      var tot : f32 = 0.0;
      for (var i:u32=0u; i<nsg; i=i+1u) { tot = tot + part[lid.x*${GEMV_WG}u + i]; }
      y[row] = tot;
    }
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
  // forward reads the weights once, like decode. vec4 loads + subgroupAdd. The old
  // tiled GEMM read weights per 16-tile and was ~3.7× slower at T=8 (measured).
  // Handles T ≤ MATVEC_MAXT in one dispatch; K % 4 == 0.
  const MATVEC_WG = 64;
  // Tokens per matvecQ dispatch. linearQ loops ceil(T/MAXT) tiles and RE-READS the
  // full weight matrix each tile, so prefill VRAM traffic ∝ (T/MAXT)·weights. Raising
  // MAXT amortizes each weight read over more tokens → fewer re-reads → faster prefill.
  // Bounds: part[MAXT·WG] workgroup mem = MAXT·256B (≤16KB → MAXT≤64); final reduction
  // maps thread lid.x→token so MAXT ≤ MATVEC_WG (64). 16→32 ≈ halves prefill re-reads.
  const MATVEC_MAXT = 32;
  const MATVEC_WGSL = `
enable f16;
enable subgroups;
struct D { T:u32, N:u32, K:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<vec4<f32>>;   // [T, K/4]
@group(0) @binding(1) var<storage, read>       W : array<vec4<f16>>;   // [N, K/4]
@group(0) @binding(2) var<storage, read_write> y : array<f32>;          // [T, N]
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> part : array<f32, ${MATVEC_MAXT * MATVEC_WG}>;  // part[t*WG + subgroupIdx]
@compute @workgroup_size(${MATVEC_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
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
  let sgIdx = lid.x / sgs;
  for (var t:u32=0u; t<T; t=t+1u) {
    let s = subgroupAdd(acc[t]);
    if (sgi == 0u) { part[t*${MATVEC_WG}u + sgIdx] = s; }
  }
  workgroupBarrier();
  if (lid.x < T) {
    let t = lid.x; let nsg = (${MATVEC_WG}u + sgs - 1u) / sgs;
    var tot : f32 = 0.0;
    for (var i:u32=0u; i<nsg; i=i+1u) { tot = tot + part[t*${MATVEC_WG}u + i]; }
    y[t*d.N + n] = tot;
  }
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
  const GEMVQ_WGSL = `
enable f16;
enable subgroups;
struct D { N:u32, K:u32, acc:u32, _b:u32 };   // acc=1 → y[n] += result (fused residual)
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;   // [K/4]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;          // [N*K/8] packed nibbles
@group(0) @binding(2) var<storage, read>       sc : array<f16>;          // [N*K/QGROUP] scales
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;          // [N]
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${GEMVQ_NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
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
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) {
    let ss = subgroupAdd(acc[r]);
    if (sgi == 0u) { part[r*${GEMV_WG}u + sgIdx] = ss; }
  }
  workgroupBarrier();
  if (lid.x < ${GEMVQ_NR}u) {
    let row = rowBase + lid.x;
    if (row < d.N) {
      let nsg=(${GEMV_WG}u+sgs-1u)/sgs; var t:f32=0.0;
      for(var i:u32=0u;i<nsg;i=i+1u){ t = t + part[lid.x*${GEMV_WG}u + i]; }
      y[row] = select(0.0, y[row], d.acc != 0u) + t;
    }
  }
}`;
  function gemvQ(xBuf, packBuf, scBuf, yBuf, N, K, acc) {
    const pipe = E.getPipeline('q3.gemvQ', GEMVQ_WGSL);
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const nWG = Math.ceil(N / GEMVQ_NR);
    const gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [xBuf, packBuf, scBuf, yBuf, d], [gx, gy, 1]);
  }

  // ---- DP4A int8 decode GEMV (T=1) ------------------------------------------
  // gemvQ profiled at ~20-28 GB/s (peak ~50) → compute-limited by the f32 dequant+dot, not
  // bandwidth. This path quantizes the activation to per-group int8 (QUANTQ8) then replaces
  // the two f32 dot4 with two dot4I8Packed (DP4A) — ~4× less ALU on the inner dot, so the
  // kernel can rise toward the weight-bandwidth ceiling. Weights stay int4 (same bytes);
  // nibbles are unpacked + repacked to int8 in-shader. int32 accum per word × wscale × xscale.
  // One workgroup per QGROUP (32 threads): parallel abs-max reduction, per-thread int8
  // quantize, then threads 0..7 pack 4 int8 each → 8 u32. (The old 1-thread-per-group
  // version ran ~32-96 threads total = pathological occupancy, 17-34us of pure overhead.)
  const QUANTQ8_WGSL = `
struct Q { K:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<f32>;     // [K]
@group(0) @binding(1) var<storage, read_write> xq : array<u32>;     // [K/4] packed int8
@group(0) @binding(2) var<storage, read_write> xs : array<f32>;     // [K/QGROUP] group scales
@group(0) @binding(3) var<uniform>             q  : Q;
var<workgroup> msh : array<f32, ${QGROUP}>;
var<workgroup> qsh : array<i32, ${QGROUP}>;
@compute @workgroup_size(${QGROUP},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let g = wg.x + wg.y*nwg.x; let ng = q.K / ${QGROUP}u; if (g >= ng) { return; }
  let lid = lidv.x; let val = x[g*${QGROUP}u + lid];
  msh[lid] = abs(val); workgroupBarrier();
  var st = ${QGROUP / 2}u; loop { if(st==0u){break;} if(lid<st){ msh[lid]=max(msh[lid],msh[lid+st]); } workgroupBarrier(); st=st/2u; }
  let scale = msh[0] / 127.0; let inv = select(0.0, 1.0/scale, scale > 0.0);
  if (lid==0u) { xs[g] = scale; }
  qsh[lid] = clamp(i32(round(val*inv)), -127, 127); workgroupBarrier();
  if (lid < ${QGROUP / 4}u) {
    xq[g*${QGROUP / 4}u + lid] = pack4xI8(vec4<i32>(qsh[lid*4u], qsh[lid*4u+1u], qsh[lid*4u+2u], qsh[lid*4u+3u]));
  }
}`;
  const GEMVDP4_WGSL = `
enable f16;
enable subgroups;
struct D { N:u32, K:u32, acc:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;          // [K/4] packed int8 activations
@group(0) @binding(1) var<storage, read>       W  : array<u32>;          // [N*K/8] packed nibbles
@group(0) @binding(2) var<storage, read>       sc : array<f16>;          // [N*K/QGROUP] weight scales
@group(0) @binding(3) var<storage, read>       xs : array<f32>;          // [K/QGROUP] activation scales
@group(0) @binding(4) var<storage, read_write> y  : array<f32>;          // [N]
@group(0) @binding(5) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${GEMVQ_NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
  let rowBase = (wg.x + wg.y * nwg.x) * ${GEMVQ_NR}u;
  if (rowBase >= d.N) { return; }
  let words = d.K / 8u; let gpr = d.K / ${QGROUP}u;
  var acc : array<f32, ${GEMVQ_NR}>;
  for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = xq[2u*w]; let xb = xq[2u*w + 1u];   // packed int8 activations — read once, reused across rows
    let grp = (w*8u)/${QGROUP}u;
    let xsc = xs[grp];
    for (var r:u32=0u; r<${GEMVQ_NR}u; r=r+1u) {
      let row = rowBase + r;
      let p = W[row*words + w];
      let s = f32(sc[row*gpr + grp]) * xsc;
      let lo = vec4<i32>(unpack4xU8(p & 0x0F0F0F0Fu));
      let hi = vec4<i32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu));
      let wa = pack4xI8(vec4<i32>(lo.x, hi.x, lo.y, hi.y) - vec4<i32>(8));
      let wb = pack4xI8(vec4<i32>(lo.z, hi.z, lo.w, hi.w) - vec4<i32>(8));
      acc[r] = acc[r] + s * f32(dot4I8Packed(wa, xa) + dot4I8Packed(wb, xb));
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
    let row = rowBase + lid.x;
    if (row < d.N) {
      let nsg=(${GEMV_WG}u+sgs-1u)/sgs; var t:f32=0.0;
      for(var i:u32=0u;i<nsg;i=i+1u){ t = t + part[lid.x*${GEMV_WG}u + i]; }
      y[row] = select(0.0, y[row], d.acc != 0u) + t;
    }
  }
}`;
  let _dp4 = null, _dp4dead = [];   // scratch {xq, xs, cap} for the quantized activation, sized to K
  function ensureDp4(K) {
    if (_dp4 && _dp4.cap >= K) return;
    // DON'T destroy the old buffers here: with pipelined (submitOnly) forwards in flight,
    // queued command buffers still reference them — destroying mid-flight triggers
    // "[Buffer xs] used in submit while destroyed" and reads garbage. Defer to unload().
    // (Reallocation happens at most once or twice ever — K only grows to intermediate.)
    if (_dp4) { _dp4dead.push(_dp4.xq, _dp4.xs); }
    _dp4 = { cap: K, xq: E.createBuffer((K / 4) * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'xq'), xs: E.createBuffer((K / QGROUP) * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'xs') };
  }
  function gemvDP4A(xBuf, packBuf, scBuf, yBuf, N, K, acc) {
    ensureDp4(K);
    const qp = E.getPipeline('q3.quantq8', QUANTQ8_WGSL);
    const qd = uniform(new Uint32Array([K, 0, 0, 0]));
    const groups = K / QGROUP, qgx = Math.min(groups, 65535), qgy = Math.ceil(groups / qgx);
    E.dispatch(qp, [xBuf, _dp4.xq, _dp4.xs, qd], [qgx, qgy, 1]);
    const pipe = E.getPipeline('q3.gemvDP4', GEMVDP4_WGSL);
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const nWG = Math.ceil(N / GEMVQ_NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [_dp4.xq, packBuf, scBuf, _dp4.xs, yBuf, d], [gx, gy, 1]);
  }

  // ---- FUSED int4 gate+up+SwiGLU (T=1): swi[i] = silu(gate·x)*(up·x) ----
  // 3 decode passes (gate gemv, up gemv, swiglu) → 1. Reads x once per chunk and
  // dequant-dots it against BOTH gate and up weights; one fewer pass barrier ×2
  // per layer (the matmuls) plus the swiglu pass removed.
  const GUSQ_NR = 4;
  const GATEUPQ_WGSL = `
enable f16;
enable subgroups;
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
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
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
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) {
    let g = subgroupAdd(ga[r]); let u = subgroupAdd(ua[r]);
    if (sgi == 0u) { pg[r*${GEMV_WG}u + sgIdx] = g; pu[r*${GEMV_WG}u + sgIdx] = u; }
  }
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
}`;
  // DP4A int8 variant of the fused gate+up+SwiGLU (same win as gemvDP4A: int8 dot4I8Packed
  // instead of f32 dot4). The int8 activation is read ONCE and dotted against BOTH gate and up.
  const GATEUPDP4_WGSL = `
enable f16;
enable subgroups;
struct D { I:u32, H:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;          // [H/4] packed int8 activations
@group(0) @binding(1) var<storage, read>       gW : array<u32>;
@group(0) @binding(2) var<storage, read>       gS : array<f16>;
@group(0) @binding(3) var<storage, read>       uW : array<u32>;
@group(0) @binding(4) var<storage, read>       uS : array<f16>;
@group(0) @binding(5) var<storage, read>       xs : array<f32>;          // [H/QGROUP] activation scales
@group(0) @binding(6) var<storage, read_write> swi: array<f32>;          // [I]
@group(0) @binding(7) var<uniform>             d  : D;
var<workgroup> pg : array<f32, ${GUSQ_NR * GEMV_WG}>;
var<workgroup> pu : array<f32, ${GUSQ_NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
  let rowBase = (wg.x + wg.y * nwg.x) * ${GUSQ_NR}u;
  if (rowBase >= d.I) { return; }
  let words = d.H / 8u; let gpr = d.H / ${QGROUP}u;
  var ga : array<f32, ${GUSQ_NR}>; var ua : array<f32, ${GUSQ_NR}>;
  for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) { ga[r]=0.0; ua[r]=0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = xq[2u*w]; let xb = xq[2u*w + 1u]; let grp = (w*8u)/${QGROUP}u; let xsc = xs[grp];
    for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) {
      let row = rowBase + r; let wi = row*words + w; let si = row*gpr + grp;
      let gp = gW[wi]; let gsc = f32(gS[si])*xsc;
      let glo = vec4<i32>(unpack4xU8(gp & 0x0F0F0F0Fu)); let ghi = vec4<i32>(unpack4xU8((gp >> 4u) & 0x0F0F0F0Fu));
      let gwa = pack4xI8(vec4<i32>(glo.x,ghi.x,glo.y,ghi.y) - vec4<i32>(8));
      let gwb = pack4xI8(vec4<i32>(glo.z,ghi.z,glo.w,ghi.w) - vec4<i32>(8));
      ga[r] = ga[r] + gsc*f32(dot4I8Packed(gwa, xa) + dot4I8Packed(gwb, xb));
      let up = uW[wi]; let usc = f32(uS[si])*xsc;
      let ulo = vec4<i32>(unpack4xU8(up & 0x0F0F0F0Fu)); let uhi = vec4<i32>(unpack4xU8((up >> 4u) & 0x0F0F0F0Fu));
      let uwa = pack4xI8(vec4<i32>(ulo.x,uhi.x,ulo.y,uhi.y) - vec4<i32>(8));
      let uwb = pack4xI8(vec4<i32>(ulo.z,uhi.z,ulo.w,uhi.w) - vec4<i32>(8));
      ua[r] = ua[r] + usc*f32(dot4I8Packed(uwa, xa) + dot4I8Packed(uwb, xb));
    }
    w = w + ${GEMV_WG}u;
  }
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${GUSQ_NR}u; r=r+1u) {
    let g = subgroupAdd(ga[r]); let u = subgroupAdd(ua[r]);
    if (sgi == 0u) { pg[r*${GEMV_WG}u + sgIdx] = g; pu[r*${GEMV_WG}u + sgIdx] = u; }
  }
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
}`;
  function gateUpSiluQ(xBuf, gRec, uRec, swiBuf, I, H) {
    if (globalThis.__noDp4) {
      const pipe = E.getPipeline('q3.gateupQ', GATEUPQ_WGSL);
      const d = uniform(new Uint32Array([I, H, 0, 0]));
      const nWG = Math.ceil(I / GUSQ_NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
      return E.dispatch(pipe, [xBuf, gRec.pack, gRec.scales, uRec.pack, uRec.scales, swiBuf, d], [gx, gy, 1]);
    }
    ensureDp4(H);
    const qp = E.getPipeline('q3.quantq8', QUANTQ8_WGSL);
    const qd = uniform(new Uint32Array([H, 0, 0, 0]));
    const groups = H / QGROUP, qgx = Math.min(groups, 65535), qgy = Math.ceil(groups / qgx);
    E.dispatch(qp, [xBuf, _dp4.xq, _dp4.xs, qd], [qgx, qgy, 1]);
    const pipe = E.getPipeline('q3.gateupDP4', GATEUPDP4_WGSL);
    const d = uniform(new Uint32Array([I, H, 0, 0]));
    const nWG = Math.ceil(I / GUSQ_NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [_dp4.xq, gRec.pack, gRec.scales, uRec.pack, uRec.scales, _dp4.xs, swiBuf, d], [gx, gy, 1]);
  }

  // ---- INT4 batched matvec (T tokens, weight row unpacked once, reused) ----
  const MATVECQ_WGSL = `
enable f16;
enable subgroups;
struct D { T:u32, N:u32, K:u32, tBase:u32, acc:u32, _p0:u32, _p1:u32, _p2:u32 };  // acc=1 → y += result
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;   // [(tBase+t)*K/4 + ...]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;          // [Tfull*N]
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
      let base = (d.tBase + t)*K4;
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
    let idx = (d.tBase + t)*d.N + n;
    y[idx] = select(0.0, y[idx], d.acc != 0u) + tot;
  }
}`;
  function matvecQ(xBuf, packBuf, scBuf, yBuf, T, N, K, tBase, acc) {
    const pipe = E.getPipeline('q3.matvecQ', MATVECQ_WGSL);
    const d = uniform(new Uint32Array([T, N, K, tBase || 0, acc ? 1 : 0, 0, 0, 0]));
    const gx = Math.min(N, 65535), gy = Math.ceil(N / gx);
    return E.dispatch(pipe, [xBuf, packBuf, scBuf, yBuf, d], [gx, gy, 1]);
  }

  // ---- INT4 tiled GEMM (prefill): Y[T,N] = X[T,K]·dequant(W)[N,K]ᵀ ----
  // Ported from llama.cpp ggml-webgpu mul_mat_reg_tile. A workgroup owns a BM×BN output
  // block; the weight tile for its BN columns is dequantized int4→f32 into shared memory
  // ONCE per K-tile and reused across ALL BM tokens. Unlike matvecQ (acc[MAXT] registers
  // cap tokens-per-weight-read at ~32), here each thread holds only TILE_M×TILE_N outputs,
  // so BM can be large with no register blowup → each weight read from VRAM serves BM tokens.
  // Portable: no subgroups / subgroup-matrix → runs on Iris Xe gen-12lp.
  const GEMM_WG_M = 32, GEMM_WG_N = 8, GEMM_TILE_M = 4, GEMM_TILE_N = 4, GEMM_TILE_K = 16;
  const GEMM_BM = GEMM_WG_M * GEMM_TILE_M;   // 128 tokens/block (BM=256 hurt occupancy on Iris Xe; T=256 fills 2 blocks "free")
  const GEMM_BN = GEMM_WG_N * GEMM_TILE_N;   // 32 outputs/block
  const GEMMQ_WGSL = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<f32>;   // [T,K]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;   // int4 packed, words=K/8 per row
@group(0) @binding(2) var<storage, read>       sc : array<f16>;   // scales, gpr=K/${QGROUP} per row
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;   // [T,N]
@group(0) @binding(4) var<uniform>             d  : D;
const WG_M=${GEMM_WG_M}u; const WG_N=${GEMM_WG_N}u; const TILE_M=${GEMM_TILE_M}u; const TILE_N=${GEMM_TILE_N}u; const TILE_K=${GEMM_TILE_K}u;
const BM=WG_M*TILE_M; const BN=WG_N*TILE_N; const NTHREAD=WG_M*WG_N;
var<workgroup> xs : array<f32, BM*TILE_K>;
var<workgroup> ws : array<f32, BN*TILE_K>;
@compute @workgroup_size(${GEMM_WG_M * GEMM_WG_N},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>) {
  let tid = lidv.x;
  let nBlkN = (d.N + BN - 1u)/BN;
  let blk = wg.x + wg.y*nwg.x;
  let blockM = (blk / nBlkN) * BM;
  let blockN = (blk % nBlkN) * BN;
  let lm = tid / WG_N; let ln = tid % WG_N;
  let words = d.K/8u; let gpr = d.K/${QGROUP}u;
  var acc : array<array<f32,TILE_N>,TILE_M>;
  for (var i=0u;i<TILE_M;i=i+1u){ for(var j=0u;j<TILE_N;j=j+1u){ acc[i][j]=0.0; } }
  var k0=0u;
  loop {
    if (k0 >= d.K) { break; }
    var e = tid;
    loop { if (e >= BM*TILE_K) { break; }                 // load X tile [BM,TILE_K]
      let r = e / TILE_K; let c = e % TILE_K;
      let gm = blockM + r; let gk = k0 + c;
      xs[e] = select(0.0, x[gm*d.K + gk], gm < d.T && gk < d.K);
      e = e + NTHREAD;
    }
    e = tid;
    loop { if (e >= BN*TILE_K) { break; }                 // load + dequant W tile [BN,TILE_K]
      let r = e / TILE_K; let c = e % TILE_K;
      let gn = blockN + r; let gk = k0 + c;
      var v = 0.0;
      if (gn < d.N && gk < d.K) {
        let nib = (W[gn*words + (gk>>3u)] >> (4u*(gk & 7u))) & 0xFu;
        v = (f32(nib) - 8.0) * f32(sc[gn*gpr + gk/${QGROUP}u]);
      }
      ws[e] = v;
      e = e + NTHREAD;
    }
    workgroupBarrier();
    let kEnd = min(TILE_K, d.K - k0);
    var kk=0u;
    loop { if (kk >= kEnd) { break; }
      var xr : array<f32,TILE_M>;
      for (var i=0u;i<TILE_M;i=i+1u){ xr[i] = xs[(lm*TILE_M+i)*TILE_K + kk]; }
      for (var j=0u;j<TILE_N;j=j+1u){
        let wv = ws[(ln*TILE_N+j)*TILE_K + kk];
        for (var i=0u;i<TILE_M;i=i+1u){ acc[i][j] = acc[i][j] + xr[i]*wv; }
      }
      kk = kk + 1u;
    }
    workgroupBarrier();
    k0 = k0 + TILE_K;
  }
  for (var i=0u;i<TILE_M;i=i+1u){
    let gm = blockM + lm*TILE_M + i;
    if (gm < d.T) {
      for (var j=0u;j<TILE_N;j=j+1u){
        let gn = blockN + ln*TILE_N + j;
        if (gn < d.N) {
          let idx = gm*d.N + gn;
          y[idx] = select(0.0, y[idx], d.acc != 0u) + acc[i][j];
        }
      }
    }
  }
}`;
  function gemmQ(xBuf, wrec, yBuf, T, N, K, acc) {
    const pipe = E.getPipeline('q3.gemmQ', GEMMQ_WGSL);
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0, 0, 0, 0, 0]));
    const blocks = Math.ceil(T / GEMM_BM) * Math.ceil(N / GEMM_BN);
    const gx = Math.min(blocks, 65535), gy = Math.ceil(blocks / gx);
    return E.dispatch(pipe, [xBuf, wrec.pack, wrec.scales, yBuf, d], [gx, gy, 1]);
  }

  // Router for the int4 weight path. wrec = { pack, scales, N, K }. acc=true →
  // y += result (fused residual add, saves a separate addInPlace pass).
  async function linearQ(xBuf, wrec, yBuf, T, N, K, acc) {
    if (T === 1) return (globalThis.__noDp4 ? gemvQ : gemvDP4A)(xBuf, wrec.pack, wrec.scales, yBuf, N, K, acc);
    return gemmQ(xBuf, wrec, yBuf, T, N, K, acc);   // prefill → tiled GEMM (weights read once per BM tokens)
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
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let idx=gid.y*(nwg.x*64u)+gid.x; let total=p.T*p.H; if(idx>=total){return;}
  let t=idx/p.H; let h=idx%p.H;
  y[idx]=f32(embed[ids[p.idOff + t]*p.H + h]);
}`;
  function embedGather(idsBuf, embedBuf, yBuf, T, H, idOff) {
    const pipe = E.getPipeline('q3.embed', EMBED_WGSL);
    const p = uniform(new Uint32Array([T, H, idOff || 0, 0]));
    const nWG = Math.ceil((T * H) / 64), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [idsBuf, embedBuf, yBuf, p], [gx, gy, 1]);
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
  // TILED FLASH attention (online softmax). The old kernel ran one workgroup per
  // (query,head): for prefill the T queries of a head EACH re-streamed the entire KV
  // cache from VRAM (~T× redundant) — measured 9.7s for ONE 128-token chunk at S=2790,
  // i.e. attention (bandwidth-bound), not the matmul, dominated prefill. Here one
  // workgroup owns a head × a block of QT(16) queries and streams K/V in KT(8)-key
  // tiles loaded to shared memory ONCE and reused across all 16 queries (16× fewer KV
  // reads), keeping a running max/sum/acc per query (online softmax). Portable (no
  // subgroups). hd ≤ 128. Shared ≈ 25KB.
  const ATTN_QT = 16, ATTN_KT = 8, ATTN_HDMAX = 128, ATTN_WG = 128;   // QT*KT == WG
  // VEC4 vectorization: the score dot was a single serial f32 accumulator over hd=128
  // (`dot += a*b`, each add waiting on the last) → ~27% of GPU compute peak (latency/ILP
  // bound, NOT bandwidth: measured 33 GFLOP/s vs 124 peak, 4 GB/s). Here Q/K/V/O are read
  // as vec4 and the dot accumulates into a vec4 (4 independent lanes) → 4-wide ILP + 4×
  // fewer loop iterations. acc/V-accumulation vectorized the same way. Storage stride
  // HD4 = HDMAX/4; the live loop bound is hd4 = hd/4 (hd is a multiple of 4 for Qwen3).
  const ATTN_WGSL = `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       K : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       V : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             p : P;
const QT=${ATTN_QT}u; const KT=${ATTN_KT}u; const HD4=${ATTN_HDMAX / 4}u; const WG=${ATTN_WG}u;
var<workgroup> qsh : array<vec4<f32>, QT*HD4>;   // QT queries × hd4
var<workgroup> ksh : array<vec4<f32>, KT*HD4>;   // KT keys × hd4 (one tile)
var<workgroup> vsh : array<vec4<f32>, KT*HD4>;
var<workgroup> acc : array<vec4<f32>, QT*HD4>;   // running output per query
var<workgroup> scr : array<f32, QT*KT>;          // score/prob tile
var<workgroup> msh : array<f32, QT>;             // running max
var<workgroup> lsh : array<f32, QT>;             // running denom
var<workgroup> csh : array<f32, QT>;             // rescale factor this tile
@compute @workgroup_size(${ATTN_WG},1,1)
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
  let qmax = (p.S - p.T) + min(qbase + QT, p.T) - 1u;   // highest global key any query in this block attends (causal)
  // load Q tile + init acc
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
    if (k0 >= p.S || k0 > qmax) { break; }     // causal early-exit: skip fully-masked key-tiles
    e = tid;                                   // load K/V tile
    loop { if (e >= KT*hd4) { break; }
      let kj = e/hd4; let d4 = e%hd4; let gk = k0+kj; let ok = gk < p.S;
      ksh[kj*HD4+d4] = select(vec4<f32>(0.0), K[gk*kv4 + hkoff + d4], ok);
      vsh[kj*HD4+d4] = select(vec4<f32>(0.0), V[gk*kv4 + hkoff + d4], ok);
      e = e + WG;
    }
    workgroupBarrier();
    { let qi = tid / KT; let kj = tid % KT;    // scores: 1 thread per (qi,kj), QT*KT==WG
      var s4 = vec4<f32>(0.0);                 // 4 independent accumulator lanes (ILP)
      for (var i4=0u;i4<hd4;i4=i4+1u){ s4 = s4 + qsh[qi*HD4+i4]*ksh[kj*HD4+i4]; }
      let dot = s4.x + s4.y + s4.z + s4.w;
      let gq = qbase+qi; let gk = k0+kj; let gqpos = (p.S - p.T) + gq;
      let valid = (gq < p.T) && (gk < p.S) && (gk <= gqpos);
      scr[tid] = select(-3.0e38, dot*scale, valid);
    }
    workgroupBarrier();
    if (tid < QT) {                            // online-softmax update per query
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
    e = tid;                                   // acc[qi][d4] = acc*corr + Σ_kj prob*V
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
  e = tid;                                     // write O = acc / denom
  loop { if (e >= QT*hd4) { break; }
    let qi = e/hd4; let d4 = e%hd4; let gq = qbase+qi;
    if (gq < p.T) { O[gq*qhs4 + hqoff + d4] = acc[qi*HD4+d4] / lsh[qi]; }
    e = e + WG;
  }
}`;
  // DECODE attention (T==1). The prefill kernel above wastes ~15/16 of its work at T=1
  // (a QT=16 query block with 1 valid query) → measured 5 GFLOP/s (4% of peak), which is
  // why long-context decode crawls and the GPU idles. This is the dedicated single-query
  // path (llama.cpp's flash_attn_vec analogue): ONE workgroup per head, all 128 threads
  // split the S keys — score via vec4 dot + parallel max/sum reduction, then a 128-thread
  // PV accumulation (nd dims × ng key-groups, reduced). Query at the last position attends
  // all keys (T=1 ⇒ causal limit = S-1), so no per-key mask.
  const ATTN_DEC_WGSL = `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       K : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       V : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             p : P;
const DWG=128u; const HD4=${ATTN_HDMAX / 4}u; const MAXS=${4096}u;
var<workgroup> qd  : array<vec4<f32>, HD4>;   // the single query (hd4 vec4)
var<workgroup> sc  : array<f32, MAXS>;        // scores / probs per key
var<workgroup> red : array<f32, DWG>;         // reduction scratch
var<workgroup> part: array<vec4<f32>, DWG>;   // PV partials
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>) {
  let lid = lidv.x;
  let hq = wg.x + wg.y*nwg.x;
  if (hq >= p.nHq) { return; }
  let hd4 = p.hd/4u;
  let grp = p.nHq/p.nKv; let hk = hq/grp;
  let kv4 = (p.nKv*p.hd)/4u; let qhs4 = (p.nHq*p.hd)/4u; let hqoff = hq*hd4; let hkoff = hk*hd4;
  let scale = 1.0/sqrt(f32(p.hd));
  let S = p.S;
  var e = lid; loop { if (e>=hd4) {break;} qd[e] = Q[hqoff + e]; e = e + DWG; }   // query gq=0
  workgroupBarrier();
  var key = lid;                                // scores: each thread strides the keys
  loop { if (key>=S) {break;}
    var s4 = vec4<f32>(0.0);
    for (var i4=0u;i4<hd4;i4=i4+1u){ s4 = s4 + qd[i4]*K[key*kv4 + hkoff + i4]; }
    sc[key] = (s4.x+s4.y+s4.z+s4.w)*scale;
    key = key + DWG;
  }
  workgroupBarrier();
  var lmax = -3.0e38; key = lid; loop { if(key>=S){break;} lmax = max(lmax, sc[key]); key = key+DWG; }
  red[lid] = lmax; workgroupBarrier();
  var st = DWG/2u; loop { if(st==0u){break;} if(lid<st){ red[lid]=max(red[lid],red[lid+st]); } workgroupBarrier(); st=st/2u; }
  let m = red[0]; workgroupBarrier();
  var lsum = 0.0; key = lid; loop { if(key>=S){break;} let pe = exp(sc[key]-m); sc[key]=pe; lsum=lsum+pe; key=key+DWG; }
  red[lid] = lsum; workgroupBarrier();
  st = DWG/2u; loop { if(st==0u){break;} if(lid<st){ red[lid]=red[lid]+red[lid+st]; } workgroupBarrier(); st=st/2u; }
  let denom = red[0]; workgroupBarrier();
  // PV: 128 threads = nd dims × ng key-groups; each accumulates its key subset, then reduce
  let nd = hd4; let ng = DWG / nd;
  let d4 = lid % nd; let g = lid / nd;
  var acc4 = vec4<f32>(0.0);
  var s = g; loop { if (s>=S) {break;} acc4 = acc4 + sc[s]*V[s*kv4 + hkoff + d4]; s = s + ng; }
  part[lid] = acc4; workgroupBarrier();
  if (lid < nd) {
    var sum4 = vec4<f32>(0.0);
    for (var gg=0u; gg<ng; gg=gg+1u){ sum4 = sum4 + part[gg*nd + lid]; }
    O[hqoff + lid] = sum4 / denom;
  }
}`;
  function attention(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd) {
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    if (T === 1) {   // decode: dedicated single-query kernel, one workgroup per head
      const pipe = E.getPipeline('q3.attnDecode', ATTN_DEC_WGSL);
      const gx = Math.min(nHq, 65535), gy = Math.ceil(nHq / gx);
      return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
    }
    const pipe = E.getPipeline('q3.attnFlash', ATTN_WGSL);
    const blocks = nHq * Math.ceil(T / ATTN_QT);
    const gx = Math.min(blocks, 65535), gy = Math.ceil(blocks / gx);
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
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
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i=gid.y*(nwg.x*64u)+gid.x; if(i>=p.n){return;}
  let g=gate[i]; let silu=g/(1.0+exp(-g)); y[i]=silu*up[i];
}`;
  function swiglu(gateBuf, upBuf, yBuf, n) {
    const pipe = E.getPipeline('q3.swiglu', SWIGLU_WGSL);
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    const nWG = Math.ceil(n / 64), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [gateBuf, upBuf, yBuf, p], [gx, gy, 1]);
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
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){ let i=gid.y*(nwg.x*64u)+gid.x; if(i>=p.n){return;} a[i]=a[i]+b[i]; }`;
  function addInPlace(aBuf, bBuf, n) {
    const pipe = E.getPipeline('q3.add', ADD_WGSL);
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    const nWG = Math.ceil(n / 64), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [aBuf, bBuf, p], [gx, gy, 1]);
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
      for(let t=0;t<T;t++){let ss=0;for(let i=0;i<H;i++)ss+=x[t*H+i]**2;const inv=1/Math.sqrt(ss/H+eps);for(let i=0;i<H;i++)y[t*H+i]=x[t*H+i]*inv*wR[i];}
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
    // --- attention tiled (multi q-block + multi k-tile + GQA + causal + partial tiles) ---
    {
      const T=40,S=40,nHq=8,nKv=2,hd=128;
      const Q=new Float32Array(T*nHq*hd),Kk=new Float32Array(S*nKv*hd),Vv=new Float32Array(S*nKv*hd);
      for(let i=0;i<Q.length;i++)Q[i]=Math.sin(i*0.013);
      for(let i=0;i<Kk.length;i++)Kk[i]=Math.cos(i*0.011);
      for(let i=0;i<Vv.length;i++)Vv[i]=Math.sin(i*0.007+0.5);
      const qb=f32buf(Q),kb=f32buf(Kk),vb=f32buf(Vv),ob=E.createBuffer(Q.length*4,ST(),'o');
      await attention(qb,kb,vb,ob,T,S,nHq,nKv,hd);
      const got=await E.readF32(ob,Q.length);
      const y=new Float32Array(Q.length); const grp=nHq/nKv; const scale=1/Math.sqrt(hd);
      for(let t=0;t<T;t++)for(let hq=0;hq<nHq;hq++){const hk=Math.floor(hq/grp);const qo=t*(nHq*hd)+hq*hd;const last=(S-T)+t;
        let m=-1e38;for(let s=0;s<=last;s++){const ko=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qo+i]*Kk[ko+i];d*=scale;if(d>m)m=d;}
        let den=0;const a=new Float32Array(hd);for(let s=0;s<=last;s++){const ko=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qo+i]*Kk[ko+i];const w=Math.exp(d*scale-m);den+=w;for(let i=0;i<hd;i++)a[i]+=w*Vv[ko+i];}
        for(let i=0;i<hd;i++)y[qo+i]=a[i]/den;}
      check('attentionTiled', maxAbs(got,y), 2e-3);
      [qb,kb,vb,ob].forEach(b=>b.destroy());
    }
    // --- attention decode (T=1 path: routes to the dedicated single-query kernel) ---
    {
      const T=1,S=130,nHq=16,nKv=8,hd=128;
      const Q=new Float32Array(T*nHq*hd),Kk=new Float32Array(S*nKv*hd),Vv=new Float32Array(S*nKv*hd);
      for(let i=0;i<Q.length;i++)Q[i]=Math.sin(i*0.021);
      for(let i=0;i<Kk.length;i++)Kk[i]=Math.cos(i*0.013);
      for(let i=0;i<Vv.length;i++)Vv[i]=Math.sin(i*0.006+0.3);
      const qb=f32buf(Q),kb=f32buf(Kk),vb=f32buf(Vv),ob=E.createBuffer(Q.length*4,ST(),'o');
      await attention(qb,kb,vb,ob,T,S,nHq,nKv,hd);   // T=1 → decode kernel
      const got=await E.readF32(ob,Q.length);
      const y=new Float32Array(Q.length); const grp=nHq/nKv; const scale=1/Math.sqrt(hd);
      for(let hq=0;hq<nHq;hq++){const hk=Math.floor(hq/grp);const qo=hq*hd;
        let m=-1e38;for(let s=0;s<S;s++){const ko=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qo+i]*Kk[ko+i];d*=scale;if(d>m)m=d;}
        let den=0;const a=new Float32Array(hd);for(let s=0;s<S;s++){const ko=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qo+i]*Kk[ko+i];const w=Math.exp(d*scale-m);den+=w;for(let i=0;i<hd;i++)a[i]+=w*Vv[ko+i];}
        for(let i=0;i<hd;i++)y[qo+i]=a[i]/den;}
      check('attentionDecode', maxAbs(got,y), 2e-3);
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
    // --- gemvDP4A (DP4A int8 activation) vs CPU dequant — relative tol (int8 quant ~1%) ---
    {
      const N=200,K=320;
      const x=new Float32Array(K),Wf=new Float32Array(N*K);
      for(let i=0;i<K;i++)x[i]=Math.sin(i*0.2);
      for(let i=0;i<Wf.length;i++)Wf[i]=Math.cos(i*0.013);
      const {pack,scales}=quantizeInt4Bf16(f32ToBf16(Wf),N,K);
      const Wdq=dequantInt4(pack,scales,N,K);
      const xb=f32buf(x),pb=qbuf(pack),sb=sbuf(scales),yb=E.createBuffer(N*4,ST(),'y');
      await gemvDP4A(xb,pb,sb,yb,N,K);
      const got=await E.readF32(yb,N); const y=new Float32Array(N); let ref=1e-9;
      for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[k]*Wdq[n*K+k];y[n]=a;ref=Math.max(ref,Math.abs(a));}
      check('gemvDP4A', maxAbs(got,y)/ref, 2e-2);
      [xb,pb,sb,yb].forEach(b=>b.destroy());
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
    // --- gemmQ (int4 tiled GEMM, prefill) vs CPU dequant ---
    {
      const T=70,N=96,K=128;   // T>BM(64) and N>BN(32) → multi-block + tail rows
      const x=new Float32Array(T*K),Wf=new Float32Array(N*K);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.05);
      for(let i=0;i<Wf.length;i++)Wf[i]=Math.cos(i*0.017);
      const {pack,scales}=quantizeInt4Bf16(f32ToBf16(Wf),N,K);
      const Wdq=dequantInt4(pack,scales,N,K);
      const xb=f32buf(x),pb=qbuf(pack),sb=sbuf(scales),yb=E.createBuffer(T*N*4,ST(),'y');
      await gemmQ(xb,{pack:pb,scales:sb},yb,T,N,K);
      const got=await E.readF32(yb,T*N); const y=new Float32Array(T*N);
      for(let t=0;t<T;t++)for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[t*K+k]*Wdq[n*K+k];y[t*N+n]=a;}
      check('gemmQ', maxAbs(got,y), 1e-2);
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
      let ref=1e-9; for(let i=0;i<I;i++){let g=0,u=0;for(let k=0;k<H;k++){g+=x[k]*gdq[i*H+k];u+=x[k]*udq[i*H+k];}const silu=g/(1+Math.exp(-g));y[i]=silu*u;ref=Math.max(ref,Math.abs(y[i]));}
      check('gateUpSiluQ', maxAbs(got,y)/ref, 3e-2);   // DP4A default path → int8 activation quant, relative tol
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
    return out;
  }

  // ============================================================
  // Tokenizer — byte-level BPE (Qwen2Tokenizer), faithful to tokenizer.json:
  // pretokenizer regex (contractions expanded since JS lacks (?i:) groups) →
  // byte-level encode → rank-ordered BPE merges → vocab ids. Specials spliced
  // directly. Loaded from tokenizer.json (vocab + merges + added_tokens).
  // ============================================================
  const SPECIAL = { endoftext: 151643, im_start: 151644, im_end: 151645, think: 151667, think_end: 151668 };
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

    // Build the Qwen3 ChatML prompt with special ids spliced in.
    function encodeChat(messages, { addGenerationPrompt = true } = {}) {
      const ids = [];
      const seg = (role, content) => {
        ids.push(SPECIAL.im_start);
        ids.push(...encodeText(role + '\n' + content));
        ids.push(SPECIAL.im_end);
        ids.push(...encodeText('\n'));
      };
      for (const m of messages) seg(m.role, typeof m.content === 'string' ? m.content : '');
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
  const MODEL_ROOTS = {
    '0.6B': 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/',
    '1.7B': 'https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/',
  };
  let MODEL_ROOT = MODEL_ROOTS['0.6B'];
  let _variant = '0.6B';
  const CACHE_NAME = 'sandpie-webgpu-models';
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
  const isQuantWeight = (name) => name.includes('_proj.weight') || name === 'lm_head.weight';

  async function fetchModelBytes(url, onProgress) {
    let cache = null; try { cache = await caches.open(CACHE_NAME); } catch (_) {}
    if (cache) { const hit = await cache.match(url); if (hit) { onProgress && onProgress({ phase: 'cache', pct: 100 }); return await hit.arrayBuffer(); } }
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('download failed: HTTP ' + resp.status + ' for ' + url);
    const total = +(resp.headers.get('content-length') || 0);
    const reader = resp.body.getReader();
    let recv = 0, out;
    if (total) {
      // Known size → preallocate ONE buffer and write directly into it. The
      // multi-GB shards (Qwen3-1.7B = 3.28GB) would otherwise need the chunks[]
      // array AND a second copy buffer alive at once (~2× peak) → "Array buffer
      // allocation failed" on a 16GB box.
      out = new Uint8Array(total);
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        out.set(value, recv); recv += value.length;
        onProgress && onProgress({ phase: 'download', pct: Math.round(recv / total * 100), recv, total });
      }
    } else {
      const chunks = [];
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        chunks.push(value); recv += value.length;
      }
      out = new Uint8Array(recv); let off = 0; for (const c of chunks) { out.set(c, off); off += c.length; }
    }
    if (cache) { try { await cache.put(url, new Response(out, { headers: { 'content-length': String(recv) } })); } catch (_) {} }
    return out.buffer;
  }

  // Parse a single safetensors ArrayBuffer and upload all tensors into _weights.
  function _parseSafetensors(ab, onPct) {
    const headerLen = Number(new DataView(ab, 0, 8).getBigUint64(0, true));
    const header = JSON.parse(dec_(new Uint8Array(ab, 8, headerLen)));
    const dataStart = 8 + headerLen;
    const names = Object.keys(header).filter(n => n !== '__metadata__');
    for (let i = 0; i < names.length; i++) {
      const name = names[i], info = header[name];
      const [begin, end] = info.data_offsets;
      const numel = info.shape.reduce((a, b) => a * b, 1);
      const raw = new Uint8Array(ab, dataStart + begin, end - begin);
      if (info.dtype !== 'BF16' && isQuantWeight(name)) throw new Error('quant path expects BF16 for ' + name);
      if (isQuantWeight(name)) {
        const N = info.shape[0], K = info.shape[1];
        const { pack, scales } = quantizeInt4Bf16(new Uint16Array(raw.buffer, raw.byteOffset, numel), N, K);
        const packBuf = E.createBuffer(pack.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, name + '.pack');
        const scBuf = E.createBuffer(scales.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, name + '.sc');
        E.device().queue.writeBuffer(packBuf, 0, pack);
        E.device().queue.writeBuffer(scBuf, 0, scales);
        _weights[name] = { pack: packBuf, scales: scBuf, N, K, int4: true, shape: info.shape, numel };
      } else {
        let f16bits;
        if (info.dtype === 'BF16') f16bits = bf16ToF16bits(new Uint16Array(raw.buffer, raw.byteOffset, numel));
        else if (info.dtype === 'F16') f16bits = new Uint16Array(raw.buffer, raw.byteOffset, numel);
        else if (info.dtype === 'F32') f16bits = f32ToF16bits(new Float32Array(raw.buffer, raw.byteOffset, numel));
        else throw new Error('unsupported dtype ' + info.dtype + ' for ' + name);
        const buf = E.createBuffer(numel * 2, U.STORAGE | U.COPY_DST | U.COPY_SRC, name);
        E.device().queue.writeBuffer(buf, 0, f16bits);
        _weights[name] = { buf, shape: info.shape, numel };
      }
      if ((i & 15) === 0) onPct && onPct(Math.round(i / names.length * 100));
    }
    onPct && onPct(100);
  }

  async function loadModel({ onProgress, variant = '0.6B' } = {}) {
    if (_loaded && _variant === variant) return;
    if (_loaded) unload();
    if (!(variant in CONFIGS)) throw new Error('unknown Qwen3 variant: ' + variant);
    Object.assign(CONFIG, CONFIGS[variant]);
    MODEL_ROOT = MODEL_ROOTS[variant];
    _variant = variant;
    await E.init();
    await TOK.load(MODEL_ROOT);
    onProgress && onProgress({ phase: 'tokenizer', pct: 100 });

    // Discover shards via index.json; fall back to single model.safetensors.
    let shardFiles = null;
    try {
      const r = await fetch(MODEL_ROOT + 'model.safetensors.index.json');
      if (r.ok) {
        const idx = await r.json();
        const seen = new Set();
        shardFiles = [];
        for (const fn of Object.values(idx.weight_map || {})) {
          if (!seen.has(fn)) { seen.add(fn); shardFiles.push(fn); }
        }
        shardFiles.sort();
      }
    } catch (_) {}

    _weights = {};
    if (shardFiles && shardFiles.length > 0) {
      for (let s = 0; s < shardFiles.length; s++) {
        const url = MODEL_ROOT + shardFiles[s];
        const ab = await fetchModelBytes(url, p => {
          if (!p) return;
          const base = s / shardFiles.length, step = 1 / shardFiles.length;
          if (p.phase === 'download') onProgress && onProgress({ phase: 'download', pct: Math.round((base + step * p.pct / 100) * 100), recv: p.recv, total: p.total });
          else if (p.phase === 'cache') onProgress && onProgress({ phase: 'cache', pct: 100 });
        });
        _parseSafetensors(ab, pct => onProgress && onProgress({ phase: 'parse', pct: Math.round((s + pct / 100) / shardFiles.length * 100) }));
      }
    } else {
      const ab = await fetchModelBytes(MODEL_ROOT + 'model.safetensors', onProgress);
      onProgress && onProgress({ phase: 'parse', pct: 0 });
      _parseSafetensors(ab, pct => onProgress && onProgress({ phase: 'parse', pct }));
    }

    onProgress && onProgress({ phase: 'parse', pct: 100 });
    _loaded = true;
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
  const MAX_SEQ = 4096;
  let _PERF = false, _perfData = null;   // CPU phase profiler (encode vs readback)
  let _kv = null;     // [{k,v}] per layer, sized MAX_SEQ
  let _scr = null;    // scratch buffers, sized to _scrT rows
  let _scrT = 0;
  let _idsBuf = null, _idsCap = 0;
  let _tokHist = null;   // GPU token history: argmax of pos P writes [P+1]; decode embed at pos P reads [P]. Enables GPU-resident chaining (no per-token CPU readback in the loop).

  function scrBuf(n, label) { return E.createBuffer(n * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, label); }
  function ensureKv() {
    if (_kv) return;
    const { numLayers, nKvHeads, headDim } = CONFIG;
    const per = MAX_SEQ * nKvHeads * headDim;
    _kv = [];
    for (let l = 0; l < numLayers; l++) _kv.push({ k: scrBuf(per, 'k' + l), v: scrBuf(per, 'v' + l) });
    _tokHist = E.createBuffer(MAX_SEQ * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'tokHist');
  }
  function ensureScratch(T) {
    if (_scr && _scrT >= T) return;
    if (_scr) for (const b of Object.values(_scr)) b.destroy && b.destroy();
    const { hidden: H, nHeads, nKvHeads, headDim, intermediate: I } = CONFIG;
    _scr = {
      x: scrBuf(T * H, 'x'), normed: scrBuf(T * H, 'normed'),
      q: scrBuf(T * nHeads * headDim, 'q'), k: scrBuf(T * nKvHeads * headDim, 'k'), v: scrBuf(T * nKvHeads * headDim, 'v'),
      qr: scrBuf(T * nHeads * headDim, 'qr'), kr: scrBuf(T * nKvHeads * headDim, 'kr'),
      attn: scrBuf(T * nHeads * headDim, 'attn'), oproj: scrBuf(T * H, 'oproj'),
      gate: scrBuf(T * I, 'gate'), up: scrBuf(T * I, 'up'), swi: scrBuf(T * I, 'swi'), down: scrBuf(T * H, 'down'),
      last: scrBuf(H, 'last'), logits: scrBuf(CONFIG.vocab, 'logits'),
      tok: scrBuf(1, 'tok'),   // 4 bytes — GPU argmax result (u32 token id)
    };
    _scrT = T;
  }
  function copyRange(src, dst, dstFloatOffset, floatCount) {
    E.copyBuffer(src, 0, dst, dstFloatOffset * 4, floatCount * 4);   // batch-aware
  }
  function setIds(arr) {
    if (!_idsBuf || _idsCap < arr.length) { if (_idsBuf) _idsBuf.destroy(); _idsBuf = E.createBuffer(Math.max(16, arr.length * 4), U.STORAGE | U.COPY_DST, 'ids'); _idsCap = arr.length; }
    E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(arr));
    return _idsBuf;
  }

  // Run the transformer over T tokens at absolute positions [posBase, posBase+T).
  // Updates the KV cache; returns logits (Float32Array[vocab]) for the LAST token.
  // chain=true: embed reads the GPU token-history at posBase (no CPU-set ids) so
  //   the loop runs GPU-resident. submitOnly=true: submit the batch but DON'T await
  //   the drain or read back — lets the caller pipeline (encode N+1 while GPU runs N).
  async function forward(idsArray, posBase, opts) {
    const chain = !!(opts && opts.chain), submitOnly = !!(opts && opts.submitOnly);
    const _t0 = _PERF ? performance.now() : 0;
    const C = CONFIG, H = C.hidden, nHq = C.nHeads, nKv = C.nKvHeads, hd = C.headDim, I = C.intermediate;
    const T = chain ? 1 : idsArray.length, S = posBase + T;
    ensureKv(); ensureScratch(T);
    const W = (n) => _weights[n].buf;      // f16 weight buffer (embed/norms)
    const Wq = (n) => _weights[n];         // int4 record (projections/lm_head)
    const s = _scr;
    // embed source: prefill/normal → CPU-set ids buffer; chain → GPU token history at posBase.
    const embIds = chain ? _tokHist : setIds(idsArray);
    const embOff = chain ? posBase : 0;
    uniformReset();   // pooled uniforms get stable buffers per call-site → bind-group cache hits
    // DECODE (T=1): record the whole forward as ONE submit (no mid-forward flush) — ~14ms of
    // GPU work, far under the watchdog, so the ~15 submit-boundary stalls/token vanish (they
    // were ~half the per-token GPU idle). PREFILL (T>1): keep the 32-op flush (big kernels →
    // a single multi-second submit would trip the OS GPU watchdog / TDR).
    E.beginBatch(T === 1 ? Infinity : undefined);
    await embedGather(embIds, W('model.embed_tokens.weight'), s.x, T, H, embOff);
    for (let l = 0; l < C.numLayers; l++) {
      const p = 'model.layers.' + l + '.';
      await rmsnorm(s.x, W(p + 'input_layernorm.weight'), s.normed, T, H, C.rmsEps);
      await linearQ(s.normed, Wq(p + 'self_attn.q_proj.weight'), s.q, T, nHq * hd, H);
      await linearQ(s.normed, Wq(p + 'self_attn.k_proj.weight'), s.k, T, nKv * hd, H);
      await linearQ(s.normed, Wq(p + 'self_attn.v_proj.weight'), s.v, T, nKv * hd, H);
      // NOTE: ropeQK must NOT be called in-place — aliasing the same buffer to a
      // read and a read_write binding is undefined behavior in WebGPU (miscompiles
      // on Intel). Write rope output to a separate buffer.
      await ropeQK(s.q, W(p + 'self_attn.q_norm.weight'), s.qr, T, nHq, hd, posBase, C.ropeTheta, C.rmsEps);
      await ropeQK(s.k, W(p + 'self_attn.k_norm.weight'), s.kr, T, nKv, hd, posBase, C.ropeTheta, C.rmsEps);
      copyRange(s.kr, _kv[l].k, posBase * nKv * hd, T * nKv * hd);
      copyRange(s.v, _kv[l].v, posBase * nKv * hd, T * nKv * hd);
      await attention(s.qr, _kv[l].k, _kv[l].v, s.attn, T, S, nHq, nKv, hd);
      await linearQ(s.attn, Wq(p + 'self_attn.o_proj.weight'), s.x, T, H, nHq * hd, true);   // fused residual: x += o_proj
      await rmsnorm(s.x, W(p + 'post_attention_layernorm.weight'), s.normed, T, H, C.rmsEps);
      if (T === 1) {
        await gateUpSiluQ(s.normed, Wq(p + 'mlp.gate_proj.weight'), Wq(p + 'mlp.up_proj.weight'), s.swi, I, H);  // fused gate+up+silu (3 passes→1)
      } else {
        await linearQ(s.normed, Wq(p + 'mlp.gate_proj.weight'), s.gate, T, I, H);
        await linearQ(s.normed, Wq(p + 'mlp.up_proj.weight'), s.up, T, I, H);
        await swiglu(s.gate, s.up, s.swi, T * I);
      }
      await linearQ(s.swi, Wq(p + 'mlp.down_proj.weight'), s.x, T, H, I, true);              // fused residual: x += down_proj
    }
    await rmsnorm(s.x, W('model.norm.weight'), s.normed, T, H, C.rmsEps);
    // last token row → its own [H] buffer, then lm_head
    E.copyBuffer(s.normed, (T - 1) * H * 4, s.last, 0, H * 4);
    await linearQ(s.last, Wq('lm_head.weight'), s.logits, 1, C.vocab, H);
    // GPU-side greedy argmax → write the predicted token straight into the token
    // history at the NEXT position (posBase+T), so the next forward's embed reads it.
    await argmaxKernel(s.logits, _tokHist, C.vocab, posBase + T);
    const _t1 = _PERF ? performance.now() : 0;   // all commands recorded
    const drain = E.endBatch();                   // single submit (returns the drain promise)
    if (submitOnly) return undefined;             // pipelined: caller doesn't wait here
    await drain;
    const _t2 = _PERF ? performance.now() : 0;
    const tok = await readU32At(_tokHist, posBase + T);   // the token just predicted
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

  // ---- Double-buffered GPU-resident decode -----------------------------------
  // Decode tokens chain through _tokHist on the GPU (argmax@P writes _tokHist[P+1],
  // embed@P+1 reads it), so a batch of GEN_BATCH chained forwards needs no readback
  // between tokens. The defect this replaces: the old loop did `await readU32Range`
  // after every batch, which DRAINS the queue — the GPU then sat idle through the CPU
  // emit() (token decode + host markdown/stream render) and the re-encode of the next
  // batch's first forward, every GEN_BATCH tokens. Chained forwards are strictly serial
  // (token k+1 depends on k), so that batch boundary is the only schedulable slack.
  // Here we SUBMIT batch N+1 before AWAITING batch N's readback, so the GPU runs the
  // next batch while the CPU reads + emits the current one. The queue only drains at end.
  //   emitTok(t) → false to stop (stop token); count() → tokens emitted so far.
  // forward()s in a batch are awaited (each must finish encoding+submitting before the
  // next, and before the readback copy is enqueued after them); the readback promise is
  // created but NOT awaited here, so its copy submit lands right after the batch forwards.
  async function decodeLoop(pos, maxTokens, emitTok, signal, count) {
    const submitBatch = async () => {
      if (signal && signal.aborted) return null;
      const K = Math.min(GEN_BATCH, MAX_SEQ - 1 - pos);
      if (K <= 0) return null;
      const base = pos;
      for (let k = 0; k < K; k++) await forward(null, base + k, { chain: true, submitOnly: true });
      pos += K;
      return { read: readU32Range(_tokHist, base + 1, K), K };   // copy submit enqueued AFTER the forwards
    };
    let inflight = (count() < maxTokens) ? await submitBatch() : null;
    while (inflight) {
      // Queue the next batch FIRST → GPU runs it while we read+emit the current one.
      // Skip if the current batch already reaches maxTokens (avoids a wasted batch).
      const next = (count() + inflight.K < maxTokens) ? await submitBatch() : null;
      const toks = await inflight.read;
      let stop = false;
      for (let k = 0; k < inflight.K; k++) { if (!emitTok(toks[k]) || count() >= maxTokens) { stop = true; break; } }
      inflight = stop ? null : next;   // a stop discards the already-computed `next` batch (≤1 wasted batch)
    }
  }

  // Greedy generate — thin wrapper over the double-buffered decodeLoop (see above):
  // GEN_BATCH chained forwards per batch (chained through _tokHist on the GPU, no
  // mid-batch readback), and the next batch is submitted before the current batch's
  // readback is awaited so the GPU never drains between batches.
  const STOP = (t) => t === SPECIAL.im_end || t === SPECIAL.endoftext;
  const GEN_BATCH = 8;   // tokens generated per GPU-resident batch (1 readback per batch)
  async function generate(prompt, { maxTokens = 64, onToken, signal } = {}) {
    await loadModel({ variant: _variant });
    const ids = TOK.encodeChat([{ role: 'user', content: prompt }]);
    const L = ids.length;
    const tok0 = await forward(ids, 0);          // prefill → _tokHist[L]=token0
    const outIds = []; let pos = L;
    const emit = (t) => { if (STOP(t)) return false; outIds.push(t); if (onToken) { try { onToken(TOK.decode([t])); } catch (_) {} } return true; };
    if (!emit(tok0)) return TOK.decode(outIds);
    await decodeLoop(pos, maxTokens, emit, signal, () => outIds.length);
    return TOK.decode(outIds);
  }

  // Debug bench (no model load): time the tiled gemmQ vs the old matvecQ tile-loop at a
  // realistic projection size, to confirm the GEMM actually speeds up prefill.
  async function _benchMatmul({ T = 256, N = 3072, K = 1024, iters = 4 } = {}) {
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); scales.fill(0x3c00);   // bf16 ≈ 1.0
    const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.01);
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bench'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), pb = mk(pack), sb = mk(scales), yb = E.createBuffer(T * N * 4, UF, 'ybench');
    const wrec = { pack: pb, scales: sb };
    const time = async (fn) => {
      uniformReset(); await fn(); await E.device().queue.onSubmittedWorkDone();   // warm
      const t0 = performance.now();
      for (let i = 0; i < iters; i++) { uniformReset(); await fn(); }
      await E.device().queue.onSubmittedWorkDone();
      return (performance.now() - t0) / iters;
    };
    const gemm_ms = await time(() => gemmQ(xb, wrec, yb, T, N, K));
    const matvec_ms = await time(async () => { for (let t0 = 0; t0 < T; t0 += MATVEC_MAXT) await matvecQ(xb, pb, sb, yb, Math.min(MATVEC_MAXT, T - t0), N, K, t0); });
    [xb, pb, sb, yb].forEach(b => b.destroy());
    return { T, N, K, iters, gemm_ms: +gemm_ms.toFixed(2), matvec_ms: +matvec_ms.toFixed(2), speedup: +(matvec_ms / gemm_ms).toFixed(2) };
  }

  // Debug bench (no model load): time the tiled attention at a realistic prefill shape.
  async function _benchAttn({ T = 128, S = 2790, nHq = 16, nKv = 8, hd = 128, iters = 3 } = {}) {
    const Q = new Float32Array(T * nHq * hd); for (let i = 0; i < Q.length; i++) Q[i] = Math.sin(i * 0.01);
    const Kk = new Float32Array(S * nKv * hd); for (let i = 0; i < Kk.length; i++) Kk[i] = Math.cos(i * 0.007);
    const Vv = new Float32Array(S * nKv * hd); for (let i = 0; i < Vv.length; i++) Vv[i] = Math.sin(i * 0.005);
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'ba'); E.device().queue.writeBuffer(b, 0, a.buffer, 0, a.byteLength); return b; };
    const qb = mk(Q), kb = mk(Kk), vb = mk(Vv), ob = E.createBuffer(Q.byteLength, UF, 'bo');
    uniformReset(); await attention(qb, kb, vb, ob, T, S, nHq, nKv, hd); await E.device().queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) { uniformReset(); await attention(qb, kb, vb, ob, T, S, nHq, nKv, hd); }
    await E.device().queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / iters;
    [qb, kb, vb, ob].forEach(b => b.destroy());
    return { T, S, nHq, nKv, hd, attn_ms: +ms.toFixed(2) };
  }

  // Debug bench (no model load): time the int4 decode GEMV at N,K + report GB/s & GFLOP/s.
  async function _benchGemv({ N = 2048, K = 1024, iters = 50 } = {}) {
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); scales.fill(0x3c00);
    const x = new Float32Array(K); for (let i = 0; i < K; i++) x[i] = Math.sin(i * 0.01);
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), pb = mk(pack), sb = mk(scales), yb = E.createBuffer(N * 4, UF, 'yg');
    uniformReset(); await gemvQ(xb, pb, sb, yb, N, K); await E.device().queue.onSubmittedWorkDone();   // warm
    // GPU-timestamp the kernel itself (batched, no per-dispatch submit overhead)
    E.beginProfile(iters + 8);
    E.beginBatch();
    for (let i = 0; i < iters; i++) await gemvQ(xb, pb, sb, yb, N, K);
    await E.endBatch();
    const prof = await E.endProfile();
    [xb, pb, sb, yb].forEach(b => b.destroy());
    const us = prof.filter(r => r.label === 'q3.gemvQ').reduce((s, r) => s + r.us, 0) / iters;
    const ms = us / 1000;
    const bytes = N * words * 4 + N * gpr * 2;
    return { N, K, gpu_us: +us.toFixed(1), GBs: +(bytes / ms / 1e6).toFixed(1), GFLOPs: +(2 * N * K / ms / 1e6).toFixed(1) };
  }

  // Bench + correctness for the DP4A GEMV vs the f32-dequant gemvQ (same inputs).
  async function _benchDP4({ N = 4096, K = 1024, iters = 50 } = {}) {
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);   // ~0.12..
    const x = new Float32Array(K); for (let i = 0; i < K; i++) x[i] = Math.sin(i * 0.017) * 0.8;
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), pb = mk(pack), sb = mk(scales), yq = E.createBuffer(N * 4, UF, 'yq'), yd = E.createBuffer(N * 4, UF, 'yd');
    // correctness: gemvQ (f32 dequant) vs gemvDP4A (int8)
    uniformReset(); await gemvQ(xb, pb, sb, yq, N, K);
    uniformReset(); await gemvDP4A(xb, pb, sb, yd, N, K);
    await E.device().queue.onSubmittedWorkDone();
    const a = await E.readF32(yq, N), b = await E.readF32(yd, N);
    let mx = 0, ref = 0; for (let i = 0; i < N; i++) { mx = Math.max(mx, Math.abs(a[i] - b[i])); ref = Math.max(ref, Math.abs(a[i])); }
    const relErr = +(mx / ref).toExponential(2);
    const timeLbls = async (fn, lbls) => { uniformReset(); await fn(); await E.device().queue.onSubmittedWorkDone(); E.beginProfile(iters * 3 + 8); E.beginBatch(); for (let i = 0; i < iters; i++) await fn(); await E.endBatch(); const p = await E.endProfile(); const o = {}; for (const l of lbls) o[l] = +(p.filter(r => r.label === l).reduce((s, r) => s + r.us, 0) / iters).toFixed(1); return o; };
    const baseT = await timeLbls(() => gemvQ(xb, pb, sb, yq, N, K), ['q3.gemvQ']);
    const dpT = await timeLbls(() => gemvDP4A(xb, pb, sb, yd, N, K), ['q3.quantq8', 'q3.gemvDP4']);
    [xb, pb, sb, yq, yd].forEach(bf => bf.destroy());
    const dpTotal = dpT['q3.quantq8'] + dpT['q3.gemvDP4'];
    return { N, K, relErr, gemvQ_us: baseT['q3.gemvQ'], dp4_gemv_us: dpT['q3.gemvDP4'], dp4_quant_us: dpT['q3.quantq8'], dp4_total_us: +dpTotal.toFixed(1), speedup: +(baseT['q3.gemvQ'] / dpTotal).toFixed(2) };
  }

  // Interleaved-in-one-batch GPU-timestamp A/B for fused gate+up: DP4A (default) vs f32 (__noDp4).
  async function _benchGateUp({ I = 3072, H = 1024, iters = 60 } = {}) {
    const words = H / 8, gpr = H / QGROUP;
    const mkPack = () => { const p = new Uint32Array(I * words); for (let i = 0; i < p.length; i++) p[i] = (Math.imul(i, 2654435761) >>> 0); return p; };
    const scales = new Uint16Array(I * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);
    const x = new Float32Array(H); for (let i = 0; i < H; i++) x[i] = Math.sin(i * 0.017) * 0.8;
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), gW = mk(mkPack()), gS = mk(scales), uW = mk(mkPack()), uS = mk(scales), yb = E.createBuffer(I * 4, UF, 'yb');
    const gRec = { pack: gW, scales: gS }, uRec = { pack: uW, scales: uS };
    const saved = globalThis.__noDp4;
    globalThis.__noDp4 = false; await gateUpSiluQ(xb, gRec, uRec, yb, I, H);
    globalThis.__noDp4 = true;  await gateUpSiluQ(xb, gRec, uRec, yb, I, H);
    await E.device().queue.onSubmittedWorkDone();
    E.beginProfile(iters * 3 + 8); E.beginBatch();
    for (let i = 0; i < iters; i++) { globalThis.__noDp4 = (i % 2 === 1); await gateUpSiluQ(xb, gRec, uRec, yb, I, H); }
    await E.endBatch(); const p = await E.endProfile();
    globalThis.__noDp4 = saved;
    [xb, gW, gS, uW, uS, yb].forEach(bf => bf.destroy());
    const half = iters / 2;
    const sum = (l) => p.filter(r => r.label === l).reduce((s, r) => s + r.us, 0) / half;
    const f32 = sum('q3.gateupQ'), dp4 = sum('q3.gateupDP4') + sum('q3.quantq8');
    return { I, H, f32_us: +f32.toFixed(1), dp4_us: +dp4.toFixed(1), speedup: +(f32 / dp4).toFixed(3) };
  }

  // ============================================================
  // App host contract — minimal shim over the fast subgroup generate().
  // (Ported from the dense app build; runs the stale single-submit-prefill +
  //  GEN_BATCH decode path, which uses subgroupAdd → fast on Iris Xe.)
  // ============================================================
  function unload() {
    try { if (_weights) for (const k in _weights) { const w = _weights[k]; if (!w) continue; if (w.pack && w.pack.destroy) try { w.pack.destroy(); } catch (_) {} if (w.scales && w.scales.destroy) try { w.scales.destroy(); } catch (_) {} if (w.buf && w.buf.destroy) try { w.buf.destroy(); } catch (_) {} } } catch (_) {}
    try { if (_kv) for (const l of _kv) { if (l.k && l.k.destroy) l.k.destroy(); if (l.v && l.v.destroy) l.v.destroy(); } } catch (_) {}
    try { for (const b of _dp4dead) { if (b && b.destroy) try { b.destroy(); } catch (_) {} } if (_dp4) { _dp4.xq.destroy(); _dp4.xs.destroy(); } } catch (_) {}
    _dp4 = null; _dp4dead = [];
    _weights = null; _kv = null; _scr = null; _loaded = false;
  }

  // Stream from pre-encoded ids (same prefill+decode as generate(), but the
  // caller supplies the full chat token sequence and gets clean UTF-8 deltas).
  async function _streamIds(ids, { maxTokens = 512, onToken, signal } = {}) {
    await loadModel({ variant: _variant });
    const L = ids.length;
    if (L >= MAX_SEQ) throw new Error('prompt too long: ' + L + ' tokens >= MAX_SEQ ' + MAX_SEQ);
    // One-time device fingerprint — compare against the harness to spot a different
    // adapter / power state / memory limits between the two pages (same backend).
    try {
      const dev = (window.SandpieWebGPU && window.SandpieWebGPU.device && window.SandpieWebGPU.device());
      const lim = dev && dev.limits;
      const cp = (window.SandpieWebGPU && window.SandpieWebGPU.caps && window.SandpieWebGPU.caps()) || null;
      const ad = cp && cp.adapter;
      if (lim && !_streamIds._dumped) {
        _streamIds._dumped = true;
        const feats = dev && dev.features ? Array.from(dev.features) : null;
        console.log('[qwen3 dev] adapter=' + (ad ? JSON.stringify(ad) : '?') + ' | maxWGStorage=' + lim.maxComputeWorkgroupStorageSize + ' | MAX_SEQ=' + MAX_SEQ + ' | hasSubgroups=' + (cp ? !!cp.hasSubgroups : '?') + ' hasF16=' + (cp ? !!cp.hasF16 : '?') + ' enabled=' + (cp ? JSON.stringify(cp.enabled) : '?') + ' deviceFeatures=' + (feats ? JSON.stringify(feats) : '?'));
      }
    } catch (_) {}
    // CHUNKED PREFILL — process the prompt in PREFILL_CHUNK-token blocks, each its OWN
    // forward = its OWN queue.submit. A full-prompt forward is a single multi-second GPU
    // submit; on an iGPU that trips the OS GPU watchdog (Windows TDR) → driver reset →
    // whole-PC freeze. Bounding T per forward keeps every submit short (~1s). Correctness:
    // RoPE, the KV write offset, and the attention causal mask (last=(S-T)+t) all key off
    // global position posBase+t — already proven by decode (T=1,posBase>0) — so KV
    // accumulates correctly across chunks. This does NOT reduce total work (the tiled-GEMM
    // + flash-attn ports do that); it only makes a long prefill safe instead of fatal.
    const PREFILL_CHUNK = 256;   // == GEMM_BM: each chunk fills one M-block; engine submit-split keeps TDR safe
    const _tp0 = performance.now();
    const _savedPerf = _PERF; _PERF = true;
    let tok0, _gpuMs = 0, _encMs = 0;
    for (let off = 0; off < L; off += PREFILL_CHUNK) {
      if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
      const chunk = ids.slice(off, Math.min(off + PREFILL_CHUNK, L));
      tok0 = await forward(chunk, off);                     // bounded submit (no TDR)
      if (_perfData) { _gpuMs += _perfData.gpu_drain_ms; _encMs += _perfData.encode_ms; }
    }
    _PERF = _savedPerf;
    const _pf = { encode_ms: +_encMs.toFixed(1), gpu_drain_ms: +_gpuMs.toFixed(1), map_ms: 0 };
    const _tp1 = performance.now();
    const outIds = []; let pos = L, prevText = '';
    const pushTok = (t) => {
      if (STOP(t)) return false;
      outIds.push(t);
      const txt = TOK.decode(outIds);                       // decode full seq → byte-accurate UTF-8
      if (onToken && txt.length > prevText.length) { try { onToken(txt.slice(prevText.length)); } catch (_) {} }
      prevText = txt; return true;
    };
    if (!pushTok(tok0)) return prevText;
    await decodeLoop(pos, maxTokens, pushTok, signal, () => outIds.length);
    const _te = performance.now();
    const _dms = _te - _tp1, _n = outIds.length;
    const _split = _pf ? (' | prefill split: encode=' + _pf.encode_ms + 'ms gpu=' + _pf.gpu_drain_ms + 'ms map=' + _pf.map_ms + 'ms') : '';
    console.log('[qwen3 perf] prompt=' + L + ' tok | prefill=' + ((_tp1 - _tp0) / 1000).toFixed(2) + 's (' + (L / ((_tp1 - _tp0) / 1000)).toFixed(0) + ' tok/s) | decode=' + _n + ' tok in ' + (_dms / 1000).toFixed(2) + 's (' + (_n / (_dms / 1000)).toFixed(1) + ' tok/s) | maxTokens=' + maxTokens + _split);
    return prevText;
  }

  function toolPreamble(tools) {
    const fns = (tools || []).filter(t => t && t.type === 'function').map(t => t.function).filter(Boolean);
    if (!fns.length) return '';
    const specs = fns.map(f => `- ${f.name}: ${f.description || ''}\n  arguments (JSON schema): ${JSON.stringify(f.parameters || {})}`).join('\n');
    return ['You can call a tool by emitting a line: <tool_call>{"name":"...","arguments":{...}}</tool_call>', 'Available tools:', specs].join('\n');
  }
  const _cleanContent = (t) => (t || '').replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();

  async function runConversation({ provider, messages, systemPrompt, tools, convId, signal }, emit) {
    const maxTokens = (provider && (provider.maxTokens | 0)) || 512;
    // conversations.js stores the dropdown's modelId in provider.endpoint (the
    // model picker sets spEndpoint = modelId). Accept either field.
    const _wantV = (provider && (provider.endpoint || provider.modelId)) || '';
    const variant = CONFIGS[_wantV] ? _wantV : '0.6B';
    try {
      emit({ type: 'info', message: 'Loading Qwen3-' + variant + ' dense (WebGPU)… first run downloads the weights.' });
      let lastPct = -1;
      await loadModel({ variant, onProgress: (p) => {
        if (!p) return;
        if (p.phase === 'download') { const pct = p.pct | 0; if (pct === lastPct) return; lastPct = pct; emit({ type: 'info', message: 'Downloading… ' + pct + '%' + (p.recv ? ' (' + (p.recv / 1e9).toFixed(2) + 'GB)' : '') }); }
        else if (p.phase === 'parse') emit({ type: 'info', message: 'Preparing weights… ' + (p.pct || 0) + '%' });
        else if (p.phase === 'tokenizer') emit({ type: 'info', message: 'Loading tokenizer…' });
      } });
    } catch (e) {
      emit({ type: 'info', message: null });
      if (e && e.name === 'AbortError') throw e;
      emit({ type: 'error', message: 'Qwen3 dense: ' + ((e && e.message) || e) }); emit({ type: 'agent_done' }); return;
    }
    let sys = (systemPrompt && typeof systemPrompt === 'object') ? (systemPrompt.content || '') : (systemPrompt || '');
    const pre = toolPreamble(Array.isArray(tools) ? tools : []);
    if (pre) sys = sys ? (sys + '\n\n' + pre) : pre;
    const work = [];
    if (sys) work.push({ role: 'system', content: sys });
    for (const m of (messages || [])) {
      if (!m || !m.role) continue;
      let c = m.content;
      if (Array.isArray(c)) c = c.filter(p => p && p.type === 'text').map(p => p.text || '').join('\n');
      work.push({ role: (m.role === 'tool' ? 'user' : m.role), content: c == null ? '' : String(c) });
    }
    try {
      if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
      emit({ type: 'round_start' });
      let firstTok = false, full = '';
      const ids = TOK.encodeChat(work);
      await _streamIds(ids, { maxTokens, signal, onToken: (piece) => {
        if (!firstTok) { firstTok = true; emit({ type: 'info', message: null }); }
        full += piece; emit({ type: 'delta', delta: { content: piece } });
      } });
      emit({ type: 'info', message: null });
      const content = _cleanContent(full);
      emit({ type: 'round_end', content });
      emit({ type: 'message_added', message: { role: 'assistant', content } });
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      emit({ type: 'info', message: null });
      emit({ type: 'error', message: 'Qwen3 dense: ' + ((e && e.message) || e) });
    }
    emit({ type: 'agent_done' });
  }
  const DEFAULT_MODELS = [
    { id: 'qwen3-0.6b', modelId: '0.6B', label: 'Qwen3-0.6B dense (~1.1GB download)' },
    { id: 'qwen3-1.7b', modelId: '1.7B', label: 'Qwen3-1.7B dense (~3.9GB download)' },
  ];
  const DEFAULT_N_CTX = MAX_SEQ;

  return {
    CONFIG,
    rmsnorm, linearT, gemv, linear, embedGather, ropeQK, attention, swiglu, addInPlace,
    selfTestKernels,
    TOK, loadModel, forward, generate, readLogits, isLoaded: () => _loaded, variant: () => _variant,
    runConversation, DEFAULT_MODELS, DEFAULT_N_CTX, unload, _benchMatmul, _benchAttn, _benchGemv, _benchDP4, _benchGateUp,
    _setMatvec: (b) => { _USE_MATVEC = !!b; },
    _setPerf: (b) => { _PERF = !!b; }, _perf: () => _perfData,
    _dbg: {
      weight: async (name, n) => readF16(_weights[name].buf, n || _weights[name].numel),
      weightInfo: (name) => ({ shape: _weights[name].shape, numel: _weights[name].numel }),
      names: () => Object.keys(_weights || {}),
    },
  };
})();

if (typeof window !== 'undefined') window.SandpieQwen3 = SandpieQwen3;
