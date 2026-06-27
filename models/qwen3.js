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
  // Quantize the activation vector x[K] → _dp4.xq (packed int8) + _dp4.xs (group scales).
  // Split out of gemvDP4A so several projections that share the SAME activation (q/k/v all
  // read the rmsnorm output) can quantize ONCE and reuse — see the q/k/v fusion in forward().
  function quantQ8(xBuf, K) {
    ensureDp4(K);
    const qp = E.getPipeline('q3.quantq8', QUANTQ8_WGSL);
    const qd = uniform(new Uint32Array([K, 0, 0, 0]));
    const groups = K / QGROUP, qgx = Math.min(groups, 65535), qgy = Math.ceil(groups / qgx);
    E.dispatch(qp, [xBuf, _dp4.xq, _dp4.xs, qd], [qgx, qgy, 1]);
  }
  // GEMV against the CURRENTLY-quantized activation in _dp4 (caller ran quantQ8 first).
  function gemvDP4_only(packBuf, scBuf, yBuf, N, K, acc) {
    const pipe = E.getPipeline('q3.gemvDP4', GEMVDP4_WGSL);
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const nWG = Math.ceil(N / GEMVQ_NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [_dp4.xq, packBuf, scBuf, _dp4.xs, yBuf, d], [gx, gy, 1]);
  }
  function gemvDP4A(xBuf, packBuf, scBuf, yBuf, N, K, acc) {
    quantQ8(xBuf, K);
    return gemvDP4_only(packBuf, scBuf, yBuf, N, K, acc);
  }

  // FUSED rmsnorm + int8 quantize (decode T=1): emit the per-group int8 activation that the
  // DP4A gemvs consume DIRECTLY from the norm, so the separate quantize dispatch disappears
  // (the dispatch-overhead lever — RMSNorm fusion was the paper's biggest single win). Output
  // layout is BIT-IDENTICAL to rmsnorm→QUANTQ8: xq[g*8+k]=pack4xI8 of 4 int8, xs[g]=scale.
  // normed = x*inv*w is recomputed inline during quantize (cheap; avoids a shared-mem array,
  // so there's NO hidden context/hidden-size cap). One workgroup over the single row.
  const RMSNORMQ_WGSL = `
enable f16;
struct P { H:u32, eps:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<f32>;
@group(0) @binding(1) var<storage, read>       w  : array<f16>;
@group(0) @binding(2) var<storage, read_write> xq : array<u32>;
@group(0) @binding(3) var<storage, read_write> xs : array<f32>;
@group(0) @binding(4) var<uniform>             p  : P;
var<workgroup> red : array<f32, ${WG_H}>;
@compute @workgroup_size(${WG_H},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>) {
  let H = p.H;
  var s : f32 = 0.0;
  var i = lid.x;
  loop { if (i >= H) { break; } let v = x[i]; s = s + v*v; i = i + ${WG_H}u; }
  red[lid.x] = s; workgroupBarrier();
  var stride = ${WG_H}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){red[lid.x]=red[lid.x]+red[lid.x+stride];} workgroupBarrier(); stride=stride/2u; }
  let inv = inverseSqrt(red[0]/f32(H) + p.eps);
  let ng = H / ${QGROUP}u;
  var g = lid.x;
  loop {
    if (g >= ng) { break; }
    let base = g * ${QGROUP}u;
    var mx : f32 = 0.0;
    for (var j:u32=0u; j<${QGROUP}u; j=j+1u) { mx = max(mx, abs(x[base+j]*inv*f32(w[base+j]))); }
    let scale = mx/127.0; let invs = select(0.0, 1.0/scale, scale > 0.0);
    xs[g] = scale;
    for (var k:u32=0u; k<${QGROUP / 4}u; k=k+1u) {
      let b = base + k*4u;
      let q0 = clamp(i32(round(x[b]    *inv*f32(w[b])    *invs)), -127, 127);
      let q1 = clamp(i32(round(x[b+1u] *inv*f32(w[b+1u]) *invs)), -127, 127);
      let q2 = clamp(i32(round(x[b+2u] *inv*f32(w[b+2u]) *invs)), -127, 127);
      let q3 = clamp(i32(round(x[b+3u] *inv*f32(w[b+3u]) *invs)), -127, 127);
      xq[g*${QGROUP / 4}u + k] = pack4xI8(vec4<i32>(q0, q1, q2, q3));
    }
    g = g + ${WG_H}u;
  }
}`;
  // Fused norm+quantize → _dp4.xq/_dp4.xs (the caller then runs gemvDP4_only / gateUpSiluDP4_only).
  function rmsnormQ(xBuf, wBuf, H, eps) {
    ensureDp4(H);
    const pipe = E.getPipeline('q3.rmsnormQ', RMSNORMQ_WGSL);
    const u = new Uint32Array(4); const du = new DataView(u.buffer);
    du.setUint32(0, H, true); du.setFloat32(4, eps, true);
    const p = uniform(u);
    return E.dispatch(pipe, [xBuf, wBuf, _dp4.xq, _dp4.xs, p], [1, 1, 1]);
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
    quantQ8(xBuf, H);
    return gateUpSiluDP4_only(gRec, uRec, swiBuf, I, H);
  }
  // gate+up+SwiGLU against the CURRENTLY-quantized activation in _dp4 (caller ran quantQ8 or
  // rmsnormQ first) — lets the MLP norm feed it pre-quantized, dropping the quantize dispatch.
  function gateUpSiluDP4_only(gRec, uRec, swiBuf, I, H) {
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
  // FAST GEMM (ported verbatim from the deleted Qwen3.5 fork, where it measured ~548
  // GFLOP/s on THIS Iris Xe gen-12lp). The dense engine had been left on an older
  // 128×32 / BK=16 / f32-scalar kernel at ~193 GFLOP/s — that was the ~3× prefill gap
  // vs llama.cpp. 64×64 output tile/workgroup, BK=QGROUP=32 deep: each k-step stages a
  // 64-tok×32 X tile + a 64-row×32 dequantized W tile into f16 shared mem (weights
  // dequant ONCE), then each thread accumulates a 4×4 register block via vec4 dot().
  // Accumulators are codegen-UNROLLED to static scalars (acc0..15) — a WGSL array<>
  // here spills to memory and tanks throughput (a naive array-based vectorize measured
  // 84 GFLOP/s, SLOWER than the f32 original). f16math = native-f16 dot (×2 FMA rate),
  // cross-K accumulate in f32; picked by probeF16Gemm (defaults to caps.hasF16). Bank-
  // pad OFF (measured 10% regression on gen-12lp). No subgroup/coop-matrix (gated off).
  const GEMMQ_BM = 64, GEMMQ_BN = 64, GEMMQ_BK = QGROUP, GEMMQ_TM = 4, GEMMQ_TN = 4;
  let _f16Math = null;   // null = auto (caps.hasF16); set by probeF16Gemm / _setF16Math
  function _useF16Math() { return _f16Math !== null ? _f16Math : !!(E.caps && E.caps() && E.caps().hasF16); }
  const _gemmPad = false;   // bank-conflict padding — MEASURED 10% regression on gen-12lp → off
  function gemmqWgsl(f16math, pad) {
    const BM = GEMMQ_BM, BN = GEMMQ_BN, BK = GEMMQ_BK, TM = GEMMQ_TM, TN = GEMMQ_TN, BK4 = BK / 4;
    const NTH = (BM / TM) * (BN / TN), TILEA4 = BM * BK4, TILEB4 = BN * BK4, RN = BN / TN;
    const SW = BK4 + (pad ? 1 : 0);
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var<storage, read>       X  : array<vec4<f32>>;   // [T*K/4]
@group(0) @binding(1) var<storage, read>       W  : array<u32>;          // [N*K/8] packed nibbles (q+8)
@group(0) @binding(2) var<storage, read>       sc : array<f16>;          // [N*K/${QGROUP}]
@group(0) @binding(3) var<storage, read_write> Y  : array<f32>;          // [T*N]
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> As : array<vec4<f16>, ${BM * SW}>;
var<workgroup> Bs : array<vec4<f16>, ${BN * SW}>;
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
      As[lt*${SW}u + kk4] = select(vec4<f16>(0.0), vec4<f16>(X[gt*K4 + k0/4u + kk4]), gt<d.T);
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
      Bs[ln*${SW}u + kk4] = vec4<f16>(v);
    }
    workgroupBarrier();
    for (var kk4:u32=0u; kk4<${BK4}u; kk4=kk4+1u) {
`;
    for (let i = 0; i < TM; i++) s += `      let a${i} = As[(tM*${TM}u + ${i}u)*${SW}u + kk4];\n`;
    for (let j = 0; j < TN; j++) s += `      let b${j} = Bs[(tN*${TN}u + ${j}u)*${SW}u + kk4];\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += f16math
      ? `      acc${i * TN + j} = acc${i * TN + j} + f32(dot(a${i}, b${j}));\n`
      : `      acc${i * TN + j} = acc${i * TN + j} + dot(vec4<f32>(a${i}), vec4<f32>(b${j}));\n`;
    s += `    }
    workgroupBarrier();
  }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  { let gm=mBase+tM*${TM}u+${i}u; let gn=nBase+tN*${TN}u+${j}u; if (gm<d.T && gn<d.N) { let idx=gm*d.N+gn; Y[idx]=select(0.0,Y[idx],d.acc!=0u)+acc${i * TN + j}; } }\n`;
    s += `}`;
    return s;
  }
  function gemmQ(xBuf, wrec, yBuf, T, N, K, acc) {
    const f16 = _useF16Math(), pad = _gemmPad;
    const key = (f16 ? 'q3.gemmQ.f16' : 'q3.gemmQ') + (pad ? '.p' : '');
    const pipe = E.getPipeline(key, gemmqWgsl(f16, pad));
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    return E.dispatch(pipe, [xBuf, wrec.pack, wrec.scales, yBuf, d], [Math.ceil(N / GEMMQ_BN), Math.ceil(T / GEMMQ_BM), 1]);
  }
  // Verify the f16-dot GEMM vs the f32-dot reference at load; a GPU that computes f16
  // wrong falls back to f32. Defaults to caps.hasF16 if never called.
  let _f16Probed = false;
  async function probeF16Gemm() {
    if (_f16Probed) return; _f16Probed = true;
    if (_f16Math !== null) return;
    if (!(E.caps && E.caps() && E.caps().hasF16)) { _f16Math = false; return; }
    let bufs = [];
    try {
      const T = 8, N = 64, K = 256;
      const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.17);
      const Wf = new Float32Array(N * K); for (let i = 0; i < Wf.length; i++) Wf[i] = Math.cos(i * 0.013);
      const u16 = new Uint16Array(Wf.length); const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
      for (let i = 0; i < Wf.length; i++) { t[0] = Wf[i]; u16[i] = ti[0] >>> 16; }
      const { pack, scales } = quantizeInt4Bf16(u16, N, K);
      const xb = f32buf(x);
      const pb = E.createBuffer(pack.byteLength, ST(), 'f16probe.pk'); E.device().queue.writeBuffer(pb, 0, pack);
      const sb = E.createBuffer(scales.byteLength, ST(), 'f16probe.sc'); E.device().queue.writeBuffer(sb, 0, scales);
      const yb = E.createBuffer(T * N * 4, ST(), 'f16probe.y');
      const wrec = { pack: pb, scales: sb };
      bufs = [xb, pb, sb, yb];
      _f16Math = true;  await gemmQ(xb, wrec, yb, T, N, K, false); const yF16 = Array.from(await E.readF32(yb, T * N));
      _f16Math = false; await gemmQ(xb, wrec, yb, T, N, K, false); const yRef = Array.from(await E.readF32(yb, T * N));
      _f16Math = null;
      let maxErr = 0, ref = 0;
      for (let i = 0; i < T * N; i++) { maxErr = Math.max(maxErr, Math.abs(yF16[i] - yRef[i])); ref = Math.max(ref, Math.abs(yRef[i])); }
      const rel = maxErr / (ref || 1);
      if (rel > 3e-2) { _f16Math = false; console.warn('[qwen3] f16 prefill GEMM WRONG (rel ' + rel.toFixed(3) + ') — using f32 dot'); }
      else console.log('[qwen3] f16 prefill GEMM verified (rel ' + rel.toExponential(1) + ') — keeping f16 dot');
    } catch (e) {
      _f16Math = false;
      console.warn('[qwen3] f16 GEMM probe failed — using f32 dot:', (e && e.message) || e);
    } finally {
      for (const b of bufs) { try { b.destroy(); } catch (_) {} }
    }
  }

  // ---- DP4A int8 tiled GEMM (prefill) — experimental A/B vs gemmQ ----
  // Same tiling as gemmQ, but instead of dequantizing W→f32 and doing f32 FMA, it
  // keeps both operands int8 and uses dot4I8Packed (4 int8 MACs/instr). The X[T,K]
  // activation is quantized to per-(token,group) int8 first (QUANTQ8T). TILE_K is
  // pinned to QGROUP=32 so each K-tile is exactly one quant group: accumulate int32
  // within the tile, then flush to f32 once with xscale*wscale. Decode's gemvDP4A
  // won ~1.45× this way (bandwidth-bound); whether a COMPUTE-bound GEMM wins on
  // gen-12lp is uncertain (extra int32 acc registers may cut occupancy) → gated
  // behind __useDp4Gemm and A/B-benched (_benchGemmDP4) before any default flip.
  const QUANTQ8T_WGSL = `
struct Q { K:u32, gpr:u32, ng:u32, _c:u32 };   // ng = T*gpr total (token,group) pairs
@group(0) @binding(0) var<storage, read>       x  : array<f32>;     // [T,K]
@group(0) @binding(1) var<storage, read_write> xq : array<u32>;     // [T,K/4] packed int8
@group(0) @binding(2) var<storage, read_write> xs : array<f32>;     // [T,gpr] group scales
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
  const GEMMDP4_WGSL = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;   // [T,K/4] packed int8 activations
@group(0) @binding(1) var<storage, read>       W  : array<u32>;   // int4 packed, words=K/8 per row
@group(0) @binding(2) var<storage, read>       sc : array<f16>;   // weight scales, gpr=K/${QGROUP} per row
@group(0) @binding(3) var<storage, read>       xs : array<f32>;   // [T,gpr] activation scales
@group(0) @binding(4) var<storage, read_write> y  : array<f32>;   // [T,N]
@group(0) @binding(5) var<uniform>             d  : D;
const WG_M=${GEMM_WG_M}u; const WG_N=${GEMM_WG_N}u; const TILE_M=${GEMM_TILE_M}u; const TILE_N=${GEMM_TILE_N}u;
const TK=${QGROUP}u; const TK4=${QGROUP / 4}u;            // K-tile == one quant group
const BM=WG_M*TILE_M; const BN=WG_N*TILE_N; const NTHREAD=WG_M*WG_N;
var<workgroup> xsh : array<u32, BM*TK4>;   // BM tokens × TK int8 (packed)
var<workgroup> wsh : array<u32, BN*TK4>;   // BN rows  × TK int8 (packed)
@compute @workgroup_size(${GEMM_WG_M * GEMM_WG_N},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>) {
  let tid = lidv.x;
  let nBlkN = (d.N + BN - 1u)/BN;
  let blk = wg.x + wg.y*nwg.x;
  let blockM = (blk / nBlkN) * BM;
  let blockN = (blk % nBlkN) * BN;
  let lm = tid / WG_N; let ln = tid % WG_N;
  let words = d.K/8u; let gpr = d.K/${QGROUP}u; let k4 = d.K/4u;
  var facc : array<array<f32,TILE_N>,TILE_M>;
  for (var i=0u;i<TILE_M;i=i+1u){ for(var j=0u;j<TILE_N;j=j+1u){ facc[i][j]=0.0; } }
  var k0=0u;
  loop {
    if (k0 >= d.K) { break; }
    let gg = k0 / TK;                                       // this tile's quant group
    var e = tid;
    loop { if (e >= BM*TK4) { break; }                     // load X int8 tile [BM,TK]
      let r = e / TK4; let c = e % TK4; let gm = blockM + r;
      xsh[e] = select(0u, xq[gm*k4 + k0/4u + c], gm < d.T);
      e = e + NTHREAD;
    }
    e = tid;
    loop { if (e >= BN*TK4) { break; }                     // load W: int4 → int8 tile [BN,TK]
      let r = e / TK4; let c = e % TK4; let gn = blockN + r;
      var packed = 0u;
      if (gn < d.N) {
        let kk = k0 + c*4u;                                // 4 consecutive K (multiple of 4)
        let word = W[gn*words + (kk>>3u)];
        let base = (kk & 7u);                              // 0 or 4 → low/high nibble half
        let b0 = i32((word >> (4u*(base+0u))) & 0xFu) - 8;
        let b1 = i32((word >> (4u*(base+1u))) & 0xFu) - 8;
        let b2 = i32((word >> (4u*(base+2u))) & 0xFu) - 8;
        let b3 = i32((word >> (4u*(base+3u))) & 0xFu) - 8;
        packed = pack4xI8(vec4<i32>(b0,b1,b2,b3));
      }
      wsh[e] = packed;
      e = e + NTHREAD;
    }
    workgroupBarrier();
    var iacc : array<array<i32,TILE_N>,TILE_M>;            // int32 accum within this group
    for (var i=0u;i<TILE_M;i=i+1u){ for(var j=0u;j<TILE_N;j=j+1u){ iacc[i][j]=0; } }
    var cc=0u;
    loop { if (cc >= TK4) { break; }
      var xch : array<u32,TILE_M>;
      for (var i=0u;i<TILE_M;i=i+1u){ xch[i] = xsh[(lm*TILE_M+i)*TK4 + cc]; }
      for (var j=0u;j<TILE_N;j=j+1u){
        let wch = wsh[(ln*TILE_N+j)*TK4 + cc];
        for (var i=0u;i<TILE_M;i=i+1u){ iacc[i][j] = iacc[i][j] + dot4I8Packed(xch[i], wch); }
      }
      cc = cc + 1u;
    }
    for (var i=0u;i<TILE_M;i=i+1u){                        // flush int32 → f32 with per-group scales
      let gm = blockM + lm*TILE_M + i;
      let xsc = select(0.0, xs[gm*gpr + gg], gm < d.T);
      for (var j=0u;j<TILE_N;j=j+1u){
        let gn = blockN + ln*TILE_N + j;
        let wsc = select(0.0, f32(sc[gn*gpr + gg]), gn < d.N);
        facc[i][j] = facc[i][j] + f32(iacc[i][j]) * xsc * wsc;
      }
    }
    workgroupBarrier();
    k0 = k0 + TK;
  }
  for (var i=0u;i<TILE_M;i=i+1u){
    let gm = blockM + lm*TILE_M + i;
    if (gm < d.T) {
      for (var j=0u;j<TILE_N;j=j+1u){
        let gn = blockN + ln*TILE_N + j;
        if (gn < d.N) { let idx = gm*d.N + gn; y[idx] = select(0.0, y[idx], d.acc != 0u) + facc[i][j]; }
      }
    }
  }
}`;
  // CODEGEN-UNROLLED DP4A int8 GEMM (v2). The first gemmDP4A used array<> accumulators,
  // which WGSL SPILLS to memory (the exact trap that capped the naive f16 port at 84
  // GFLOP/s). This mirrors the fast f16 gemmqWgsl: 64×64 block, BK=QGROUP=32 (one quant
  // group/tile → one int32→f32 flush/tile), TM×TN=4×4 register block, static unrolled
  // scalar accumulators (i0..15 int32 per tile, f0..15 f32 across tiles). int8 inner via
  // dot4I8Packed — gen-12lp's NATIVE matrix path is DP4A, so the headroom is real if the
  // accumulators stay in registers. A/B vs f16 gemmQ via _benchGemmDP4.
  function gemmdp4Wgsl() {
    const BM = GEMMQ_BM, BN = GEMMQ_BN, BK = QGROUP, TM = GEMMQ_TM, TN = GEMMQ_TN, BK4 = BK / 4;
    const NTH = (BM / TM) * (BN / TN), RN = BN / TN, TILEA = BM * BK4, TILEB = BN * BK4;
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;   // [T,K/4] packed int8 activations
@group(0) @binding(1) var<storage, read>       W  : array<u32>;   // int4 packed (q+8), K/8 per row
@group(0) @binding(2) var<storage, read>       sc : array<f16>;   // [N, K/${QGROUP}]
@group(0) @binding(3) var<storage, read>       xs : array<f32>;   // [T, K/${QGROUP}]
@group(0) @binding(4) var<storage, read_write> Y  : array<f32>;   // [T,N]
@group(0) @binding(5) var<uniform>             d  : D;
var<workgroup> As : array<u32, ${TILEA}>;   // [BM][BK4] packed int8 activations
var<workgroup> Bs : array<u32, ${TILEB}>;   // [BN][BK4] packed int8 weights (int4→int8)
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

  let _dp4g = null, _dp4gDead = [];   // scratch {xq,xs,cap} for the [T,K] int8 activation
  function ensureDp4G(T, K) {
    const need = T * K;
    if (_dp4g && _dp4g.cap >= need) return;
    if (_dp4g) { _dp4gDead.push(_dp4g.xq, _dp4g.xs); }
    _dp4g = { cap: need, xq: E.createBuffer((need / 4) * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'gxq'), xs: E.createBuffer((need / QGROUP) * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'gxs') };
  }
  function gemmDP4A(xBuf, wrec, yBuf, T, N, K, acc) {
    ensureDp4G(T, K);
    const gpr = K / QGROUP, ng = T * gpr;
    const qp = E.getPipeline('q3.quantq8t', QUANTQ8T_WGSL);
    const qd = uniform(new Uint32Array([K, gpr, ng, 0]));
    const qgx = Math.min(ng, 65535), qgy = Math.ceil(ng / qgx);
    E.dispatch(qp, [xBuf, _dp4g.xq, _dp4g.xs, qd], [qgx, qgy, 1]);
    const pipe = E.getPipeline('q3.gemmDP4v2', gemmdp4Wgsl());
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    return E.dispatch(pipe, [_dp4g.xq, wrec.pack, wrec.scales, _dp4g.xs, yBuf, d], [Math.ceil(N / GEMMQ_BN), Math.ceil(T / GEMMQ_BM), 1]);
  }

  // ---- TEXTURE-BACKED GEMM (MLDrift-style, experimental A/B vs gemmDP4A) -------
  // Captured MLDrift's Gemma WGSL (2026-06-27): on this iGPU its prefill edge is
  // storing ACTIVATIONS in a texture (rgba16float) — half-bandwidth + the GPU's 2D
  // texture cache — with f32 compute (NOT f16-math/subgroups/DP4A). This is a faithful
  // port of that idea on our int4 weights: X[T,K] lives in a texture (texel(x=k/4,
  // y=token) = X[token, 4k..4k+3]), weights stay int4 in a storage buffer (dequant
  // inline to f32), 4×4 register tile, NO shared-memory staging — the texture cache
  // is the operand cache. Benched head-to-head against gemmDP4A via _benchGemmTex.
  function gemmTexWgsl() {
    const BM = GEMMQ_BM, BN = GEMMQ_BN, TM = GEMMQ_TM, TN = GEMMQ_TN;
    const WGM = BM / TM, WGN = BN / TN, NTH = WGM * WGN;
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var Xt : texture_2d<f32>;            // texel(x=k/4, y=token)
@group(0) @binding(1) var<storage, read>       W  : array<u32>;   // [N,K/8] int4 (q+8)
@group(0) @binding(2) var<storage, read>       sc : array<f16>;   // [N,K/${QGROUP}]
@group(0) @binding(3) var<storage, read_write> Y  : array<f32>;   // [T,N]
@group(0) @binding(4) var<uniform>             d  : D;
@compute @workgroup_size(${NTH}, 1, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let lx = lid.x; let tN = lx % ${WGN}u; let tM = lx / ${WGN}u;
  let mBase = wg.y*${BM}u; let nBase = wg.x*${BN}u;
  let WPR = d.K/8u; let gpr = d.K/${QGROUP}u; let K4 = d.K/4u;
`;
    for (let r = 0; r < TM * TN; r++) s += `  var acc${r}:f32=0.0;\n`;
    s += `  for (var k4:u32=0u; k4<K4; k4=k4+1u) {
    let k0 = k4*4u;
`;
    for (let i = 0; i < TM; i++) s += `    let x${i} = textureLoad(Xt, vec2<i32>(i32(k4), i32(mBase+tM*${TM}u+${i}u)), 0);\n`;
    for (let j = 0; j < TN; j++) s += `    var w${j}:vec4<f32>;
    { let gn=nBase+tN*${TN}u+${j}u; let word=W[gn*WPR + (k0>>3u)]; let h=(k0&4u);
      let n0=i32((word>>(4u*(h+0u)))&0xFu)-8; let n1=i32((word>>(4u*(h+1u)))&0xFu)-8;
      let n2=i32((word>>(4u*(h+2u)))&0xFu)-8; let n3=i32((word>>(4u*(h+3u)))&0xFu)-8;
      let sv=f32(sc[gn*gpr + k0/${QGROUP}u]); w${j}=vec4<f32>(f32(n0),f32(n1),f32(n2),f32(n3))*sv; }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `    acc${i * TN + j}=acc${i * TN + j}+dot(x${i}, w${j});\n`;
    s += `  }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  { let gm=mBase+tM*${TM}u+${i}u; let gn=nBase+tN*${TN}u+${j}u; if (gm<d.T && gn<d.N) { let idx=gm*d.N+gn; Y[idx]=select(0.0,Y[idx],d.acc!=0u)+acc${i * TN + j}; } }\n`;
    s += `}`;
    return s;
  }
  // xtView = a GPUTextureView of X[T,K] (texel(x=k/4,y=token)). wrec = {pack, scales}.
  function gemmTexQ(xtView, wrec, yBuf, T, N, K, acc) {
    const pipe = E.getPipeline('q3.gemmTex', gemmTexWgsl());
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    return E.dispatch(pipe, [xtView, wrec.pack, wrec.scales, yBuf, d], [Math.ceil(N / GEMMQ_BN), Math.ceil(T / GEMMQ_BM), 1]);
  }

  // ---- gemmTex2: FAITHFUL port of MLDrift kernel #142 -------------------------
  // The first port (gemmTexQ) was unfaithful and slow for two reasons unrelated to
  // textures: (1) no split-K, (2) it re-unpacked int4 weights per-thread-per-K (256×
  // more dequant ALU than gemmDP4A's amortized shared-mem unpack). This replicates
  // MLDrift exactly: one workgroup owns 8 tokens (M-block) × (TPX·4) channels; each
  // thread owns 4 channels and ALL 8 tokens (s0..s7 = vec4 of 4 channels each, 32
  // f32 accumulators); SK=8 threads in y split the K dimension (k4 += 8) and the
  // results tree-reduce through workgroup memory. Weights are plain f16 in a buffer
  // (no inline int4 unpack — isolating the texture-activation question). Activations
  // come from a texture: texel(x=token, y=k4) = X[token, 4·k4 .. +3].
  const G2_SK = 8, G2_CH = 4, G2_BT = 8;
  function gemmTex2Wgsl(tpx) {
    const TPX = tpx || 8, SK = G2_SK, CH = G2_CH, BT = G2_BT, NTH = TPX * SK;
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var Xt : texture_2d<f32>;                 // texel(x=token, y=k4)
@group(0) @binding(1) var<storage, read>       W  : array<vec4<f16>>;  // [N, K/4] row-major
@group(0) @binding(2) var<storage, read_write> Y  : array<f32>;        // [T,N]
@group(0) @binding(3) var<uniform>             d  : D;
var<workgroup> red : array<vec4<f32>, ${NTH}>;
@compute @workgroup_size(${TPX}, ${SK}, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let K4 = d.K/4u;
  let tokBase = wg.y*${BT}u;
  let chBase = (wg.x*${TPX}u + lid.x)*${CH}u;   // this thread's 4 channels: chBase..chBase+3
`;
    for (let t = 0; t < BT; t++) s += `  var s${t}:vec4<f32> = vec4<f32>(0.0);\n`;
    s += `  var k4 = lid.y;
  loop {
    if (k4 >= K4) { break; }
`;
    for (let t = 0; t < BT; t++) s += `    let v${t} = textureLoad(Xt, vec2<i32>(i32(tokBase+${t}u), i32(k4)), 0);\n`;
    for (let c = 0; c < CH; c++) s += `    let w${c} = vec4<f32>(W[(chBase+${c}u)*K4 + k4]);\n`;
    for (let t = 0; t < BT; t++) s += `    s${t} = s${t} + vec4<f32>(dot(v${t},w0), dot(v${t},w1), dot(v${t},w2), dot(v${t},w3));\n`;
    s += `    k4 = k4 + ${SK}u;
  }
  let slot = lid.x*${SK}u + lid.y;
`;
    // Tree-reduce each token-row's vec4 across the SK threads in y.
    for (let t = 0; t < BT; t++) {
      s += `  red[slot] = s${t}; workgroupBarrier();\n`;
      for (let st = SK >> 1; st > 0; st >>= 1) s += `  if (lid.y < ${st}u) { red[slot] = red[slot] + red[slot + ${st}u]; } workgroupBarrier();\n`;
      s += `  s${t} = red[lid.x*${SK}u]; workgroupBarrier();\n`;
    }
    s += `  if (lid.y != 0u) { return; }
`;
    for (let t = 0; t < BT; t++) {
      s += `  { let gm=tokBase+${t}u; if (gm<d.T) {
    for (var c=0u; c<${CH}u; c=c+1u) { let gn=chBase+c; if (gn<d.N) { let idx=gm*d.N+gn; Y[idx]=select(0.0,Y[idx],d.acc!=0u)+s${t}[c]; } }
  } }
`;
    }
    s += `}`;
    return s;
  }
  // xtView = GPUTextureView of X[T,K] as texel(x=token,y=k4). wf16 = f16 weight buffer [N,K/4].
  function gemmTex2(xtView, wf16, yBuf, T, N, K, acc, tpx) {
    tpx = tpx || 8;
    const pipe = E.getPipeline('q3.gemmTex2.' + tpx, gemmTex2Wgsl(tpx));
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    const chPerWG = tpx * G2_CH;   // channels / workgroup
    return E.dispatch(pipe, [xtView, wf16, yBuf, d], [Math.ceil(N / chPerWG), Math.ceil(T / G2_BT), 1]);
  }

  // ---- gemmTex3: FAITHFUL port of MLDrift's BULK prefill GEMM (kernel #23) -----
  // Re-captured MLDrift on a real ~1200-token prefill (hooking createComputePipeline
  // for the override constants + dispatch): the bulk matmul is @workgroup_size(64,1,1),
  // NO split-K, and it STAGES f16 weights in workgroup memory (array<vec4,32>, loaded
  // cooperatively, reused across the 64 threads). Each thread = 1 token computing 32
  // output channels (8 vec4 accumulators); activation comes from a texture (1 vec4 per
  // token per K-step, read once — no cache reliance). i.e. MLDrift's speed is standard
  // SHARED-MEM WEIGHT TILING (same principle as gemmDP4A), with the texture merely the
  // activation store — NOT split-K + texture-cache (what my gemmTex2 wrongly did).
  // Weights here are pre-packed f16 in cache order [block][kstep][32 vec4] and read
  // linearly, matching MLDrift's weights_buffer.data[offset+lid]. Speed isolation test.
  function gemmTex3Wgsl() {
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var Xt : texture_2d<f32>;                 // texel(x=token, y=kstep)
@group(0) @binding(1) var<storage, read>       W  : array<vec4<f16>>;  // packed [block][kstep][32]
@group(0) @binding(2) var<storage, read_write> Y  : array<f32>;        // [T,N]
@group(0) @binding(3) var<uniform>             d  : D;
var<workgroup> wc : array<vec4<f16>, 32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let tid = lid.x;
  let K4 = d.K/4u;
  let myTok = wg.y*64u + tid;
  let chBase = wg.x*32u;
  let blkBase = wg.x*K4*32u;     // weights for this 32-channel block
`;
    for (let r = 0; r < 8; r++) s += `  var r${r}:vec4<f32> = vec4<f32>(0.0);\n`;
    s += `  for (var ks:u32=0u; ks<K4; ks=ks+1u) {
    if (tid < 32u) { wc[tid] = W[blkBase + ks*32u + tid]; }
    workgroupBarrier();
    let src = textureLoad(Xt, vec2<i32>(i32(myTok), i32(ks)), 0);
`;
    for (let r = 0; r < 8; r++) s += `    r${r} = r${r} + vec4<f32>(wc[${r * 4}])*src.x + vec4<f32>(wc[${r * 4 + 1}])*src.y + vec4<f32>(wc[${r * 4 + 2}])*src.z + vec4<f32>(wc[${r * 4 + 3}])*src.w;\n`;
    s += `    workgroupBarrier();
  }
  if (myTok >= d.T) { return; }
`;
    for (let r = 0; r < 8; r++) s += `  { let gn=chBase+${r * 4}u; if (gn+3u<d.N) { let idx=myTok*d.N+gn; Y[idx]=r${r}.x; Y[idx+1u]=r${r}.y; Y[idx+2u]=r${r}.z; Y[idx+3u]=r${r}.w; } }\n`;
    s += `}`;
    return s;
  }
  function gemmTex3(view, wb, yBuf, T, N, K) {
    const pipe = E.getPipeline('q3.gemmTex3', gemmTex3Wgsl());
    const d = uniform(new Uint32Array([T, N, K, 0]));
    return E.dispatch(pipe, [view, wb, yBuf, d], [Math.ceil(N / 32), Math.ceil(T / 64), 1]);
  }

  // Router for the int4 weight path. wrec = { pack, scales, N, K }. acc=true →
  // y += result (fused residual add, saves a separate addInPlace pass).
  async function linearQ(xBuf, wrec, yBuf, T, N, K, acc) {
    if (T === 1) return (globalThis.__noDp4 ? gemvQ : gemvDP4A)(xBuf, wrec.pack, wrec.scales, yBuf, N, K, acc);
    // Prefill: int8 DP4A GEMM by default — GPU-timestamp min-of-14 measured 1.25–1.57× over
    // the f16 GEMM on EVERY 0.6B/1.7B shape (gen-12lp's native matmul path is DP4A); the
    // activation-quantize is <2% of the GEMM. __noDp4Gemm forces the f16 path.
    return (globalThis.__noDp4Gemm ? gemmQ : gemmDP4A)(xBuf, wrec, yBuf, T, N, K, acc);
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
    if (gq < p.T) { O[gq*qhs4 + hqoff + d4] = vec4<f32>(acc[qi*HD4+d4]) / lsh[qi]; }
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
  // Function (not const) so MAXS tracks the device-derived _attnMaxS (set at load).
  function attnDecWgsl() { return `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       K : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       V : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             p : P;
const DWG=128u; const HD4=${ATTN_HDMAX / 4}u; const MAXS=${_attnMaxS}u;
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
}`; }
  function attention(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd) {
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    if (T === 1) {   // decode: dedicated single-query kernel, one workgroup per head
      const pipe = E.getPipeline('q3.attnDecode.' + _attnMaxS, attnDecWgsl());   // label tracks MAXS so it can't reuse a stale-size pipeline
      const gx = Math.min(nHq, 65535), gy = Math.ceil(nHq / gx);
      return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
    }
    // PREFILL (T>1): f16 flash attention (KT=8) — ~2× the f32 kernel on this iGPU.
    // Prefill attention was f32-ALU + latency + occupancy bound (NOT FMA-bound, so f16
    // alone gave only 1.09×); the win came from f16 inner products + 4-accumulator ILP
    // (breaks the score/PV dependency chains) + f16 running-acc (halves SLM → more
    // resident workgroups). Verified relErr ~1e-3 vs the f32 kernel. Falls back to the
    // f32 kernel when shader-f16 is unavailable (or __noAttnF16 set, for A/B).
    const useF16 = !globalThis.__noAttnF16 && !!(E.caps && E.caps() && E.caps().hasF16);
    if (useF16) return attentionF16(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd, 8);
    const pipe = E.getPipeline('q3.attnFlash', ATTN_WGSL);
    const blocks = nHq * Math.ceil(T / ATTN_QT);
    const gx = Math.min(blocks, 65535), gy = Math.ceil(blocks / gx);
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
  }

  // ---- f16 prefill attention (experimental A/B vs ATTN_WGSL) ------------------
  // Prefill attention is COMPUTE-BOUND AT THE iGPU's f32 ceiling (~123 GFLOP/s;
  // measured attnFlash = 131). The matmuls are fast only because they're int8/f16.
  // This is the same kernel with the two hot inner loops in f16 (gen-12lp f16 ≈ 2×
  // f32 FMA rate): the QKᵀ dot accumulates in f16, and the per-tile PV sum (KT=8
  // terms, small → f16-safe) accumulates in f16 — while the RUNNING output acc and
  // ALL softmax stats (max/denom/rescale) stay f32 for stability. Q/K/V are read
  // from f32 global and cast to f16 when staged into shared memory.
  // Parameterized f16 prefill attention. KT (keys per tile) is DECOUPLED from the
  // workgroup size (WG=128) — each of the 3 per-tile phases (scores/softmax/PV) loops
  // its work over the 128 threads. Bigger KT → fewer tiles → fewer barriers (4/tile),
  // the lever if the kernel is barrier/structure-bound rather than FMA-bound.
  function attnF16Wgsl(KT) {
    const QT = ATTN_QT, WG = ATTN_WG, HD4 = ATTN_HDMAX / 4;
    return `
enable f16;
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       K : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       V : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             p : P;
const QT=${QT}u; const KT=${KT}u; const HD4=${HD4}u; const WG=${WG}u;
var<workgroup> qsh : array<vec4<f16>, QT*HD4>;
var<workgroup> ksh : array<vec4<f16>, KT*HD4>;
var<workgroup> vsh : array<vec4<f16>, KT*HD4>;
var<workgroup> acc : array<vec4<f16>, QT*HD4>;   // f16 to cut SLM → higher occupancy
var<workgroup> scr : array<f32, QT*KT>;
var<workgroup> msh : array<f32, QT>;
var<workgroup> lsh : array<f32, QT>;
var<workgroup> csh : array<f32, QT>;
@compute @workgroup_size(${WG},1,1)
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
    qsh[qi*HD4+d4] = select(vec4<f16>(0.0), vec4<f16>(Q[gq*qhs4 + hqoff + d4]), gq < p.T);
    acc[qi*HD4+d4] = vec4<f16>(0.0);
    e = e + WG;
  }
  if (tid < QT) { msh[tid] = -3.0e38; lsh[tid] = 0.0; }
  workgroupBarrier();
  var k0 = 0u;
  loop {
    if (k0 >= p.S || k0 > qmax) { break; }
    e = tid;                                    // load K/V tile (KT keys)
    loop { if (e >= KT*hd4) { break; }
      let kj = e/hd4; let d4 = e%hd4; let gk = k0+kj; let ok = gk < p.S;
      ksh[kj*HD4+d4] = select(vec4<f16>(0.0), vec4<f16>(K[gk*kv4 + hkoff + d4]), ok);
      vsh[kj*HD4+d4] = select(vec4<f16>(0.0), vec4<f16>(V[gk*kv4 + hkoff + d4]), ok);
      e = e + WG;
    }
    workgroupBarrier();
    e = tid;                                    // scores: QT*KT entries over WG threads
    loop { if (e >= QT*KT) { break; }
      let qi = e/KT; let kj = e%KT;
      let qo = qi*HD4; let ko = kj*HD4;
      var s0 = vec4<f16>(0.0); var s1 = vec4<f16>(0.0); var s2 = vec4<f16>(0.0); var s3 = vec4<f16>(0.0);
      for (var i4=0u;i4<hd4;i4=i4+4u){          // 4 independent accumulators → break the dependency chain (ILP)
        s0 = s0 + qsh[qo+i4]*ksh[ko+i4];
        s1 = s1 + qsh[qo+i4+1u]*ksh[ko+i4+1u];
        s2 = s2 + qsh[qo+i4+2u]*ksh[ko+i4+2u];
        s3 = s3 + qsh[qo+i4+3u]*ksh[ko+i4+3u];
      }
      let sv = (s0+s1)+(s2+s3);
      let dot = f32(sv.x + sv.y + sv.z + sv.w);
      let gq = qbase+qi; let gk = k0+kj; let gqpos = (p.S - p.T) + gq;
      let valid = (gq < p.T) && (gk < p.S) && (gk <= gqpos);
      scr[e] = select(-3.0e38, dot*scale, valid);
      e = e + WG;
    }
    workgroupBarrier();
    if (tid < QT) {                             // online-softmax update per query
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
    e = tid;                                    // PV: acc = acc*corr + Σ_kj prob*V
    loop { if (e >= QT*hd4) { break; }
      let qi = e/hd4; let d4 = e%hd4; let so = qi*KT;
      var a0 = vec4<f16>(0.0); var a1 = vec4<f16>(0.0); var a2 = vec4<f16>(0.0); var a3 = vec4<f16>(0.0);
      for (var kj=0u;kj<KT;kj=kj+4u){          // 4 independent PV accumulators (ILP)
        a0 = a0 + f16(scr[so+kj])    * vsh[(kj)*HD4+d4];
        a1 = a1 + f16(scr[so+kj+1u]) * vsh[(kj+1u)*HD4+d4];
        a2 = a2 + f16(scr[so+kj+2u]) * vsh[(kj+2u)*HD4+d4];
        a3 = a3 + f16(scr[so+kj+3u]) * vsh[(kj+3u)*HD4+d4];
      }
      let a16 = (a0+a1)+(a2+a3);
      acc[qi*HD4+d4] = acc[qi*HD4+d4]*f16(csh[qi]) + a16;
      e = e + WG;
    }
    workgroupBarrier();
    k0 = k0 + KT;
  }
  e = tid;
  loop { if (e >= QT*hd4) { break; }
    let qi = e/hd4; let d4 = e%hd4; let gq = qbase+qi;
    if (gq < p.T) { O[gq*qhs4 + hqoff + d4] = vec4<f32>(acc[qi*HD4+d4]) / lsh[qi]; }
    e = e + WG;
  }
}`;
  }
  function attentionF16(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd, KT) {
    KT = KT || ATTN_KT;
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    const pipe = E.getPipeline('q3.attnFlashF16.' + KT, attnF16Wgsl(KT));
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

    // --- rmsnormQ (FUSED norm+int8 quantize, decode) vs CPU rmsnorm → int8 dequant ---
    {
      const H=256, eps=1e-6;
      const x=new Float32Array(H), w=new Float32Array(H);
      for(let i=0;i<H;i++){ x[i]=Math.sin(i*0.07); w[i]=0.5+0.5*Math.cos(i*0.03); }
      const wR=roundF16(w);
      const xb=f32buf(x), wb=f16buf(w);
      await rmsnormQ(xb, wb, H, eps);
      const xs=await E.readF32(_dp4.xs, H/QGROUP);     // also flushes the rmsnormQ dispatch
      const xqU=await readU32Range(_dp4.xq, 0, H/4);
      const deq=new Float32Array(H);
      for(let wi=0; wi<H/4; wi++){ const word=xqU[wi]; for(let k=0;k<4;k++){ let b=(word>>>(8*k))&0xFF; if(b>127)b-=256; const i=wi*4+k; deq[i]=b*xs[Math.floor(i/QGROUP)]; } }
      let ss=0; for(let i=0;i<H;i++)ss+=x[i]**2; const inv=1/Math.sqrt(ss/H+eps);
      const y=new Float32Array(H); let ref=1e-9; for(let i=0;i<H;i++){ y[i]=x[i]*inv*wR[i]; ref=Math.max(ref,Math.abs(y[i])); }
      check('rmsnormQ', maxAbs(deq,y)/ref, 2e-2);   // int8 per-group quant → relative tol
      [xb,wb].forEach(b=>b.destroy());
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
      let _ref=0; for(let i=0;i<y.length;i++)_ref=Math.max(_ref,Math.abs(y[i]));
      check('gemmQ', maxAbs(got,y)/(_ref||1), 3e-2);   // RELATIVE — f16 shared tiles round ~0.3%
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
  // QUANTIZED-WEIGHTS CACHE: after the first download+quantize we serialize the
  // GPU-ready bytes (int4 packs/scales + f16 tensors) into fixed 64MB chunks in
  // Cache Storage + a manifest. Subsequent loads skip download AND re-quantize:
  // read the chunks in parallel and writeBuffer their slices STRAIGHT into the GPU
  // buffers (peak host memory = one 64MB chunk per in-flight read, no whole-model
  // or whole-tensor materialization). Bump QCACHE_VER to invalidate the format.
  const QCACHE_NAME = 'sandpie-webgpu-quant';
  const QCACHE_VER = 1;
  const QCHUNK = 64 * 1024 * 1024;       // 64 MiB cache chunk
  const QREAD_CONC = 4;                  // chunks read+uploaded in parallel
  const _qUrl = (variant, part) => 'https://sandpie.quant/v' + QCACHE_VER + '/' + variant + '/' + part;
  const _ceil16 = (n) => (n + 15) & ~15;
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

  // Single-file download (no raw-safetensors caching — the quantized-weights cache
  // supersedes it). Preallocates one buffer when the size is known.
  async function fetchModelBytes(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('download failed: HTTP ' + resp.status + ' for ' + url);
    const total = +(resp.headers.get('content-length') || 0);
    const reader = resp.body.getReader();
    let recv = 0, out;
    if (total) {
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
    return out.buffer;
  }

  // Quantize-or-f16 a single tensor's raw bytes and upload it into _weights.
  // `raw` is a Uint8Array covering EXACTLY this tensor (any byteOffset).
  // Quantize/convert one tensor → GPU buffer(s) and record it. If `sink` is given,
  // also stream the GPU-ready bytes into the quantized-weights cache (so future
  // loads skip download + quantize). sink.add is async (flushes 64MB chunks).
  async function _uploadTensor(name, info, raw, sink) {
    const numel = info.shape.reduce((a, b) => a * b, 1);
    if (info.dtype !== 'BF16' && isQuantWeight(name)) throw new Error('quant path expects BF16 for ' + name);
    if (isQuantWeight(name)) {
      const N = info.shape[0], K = info.shape[1];
      const { pack, scales } = quantizeInt4Bf16(new Uint16Array(raw.buffer, raw.byteOffset, numel), N, K);
      const packBuf = E.createBuffer(pack.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, name + '.pack');
      const scBuf = E.createBuffer(scales.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, name + '.sc');
      E.device().queue.writeBuffer(packBuf, 0, pack);
      E.device().queue.writeBuffer(scBuf, 0, scales);
      _weights[name] = { pack: packBuf, scales: scBuf, N, K, int4: true, shape: info.shape, numel };
      if (sink) { await sink.add(name, 'pack', new Uint8Array(pack.buffer, pack.byteOffset, pack.byteLength), { kind: 'int4', shape: info.shape, numel, N, K }); await sink.add(name, 'scales', new Uint8Array(scales.buffer, scales.byteOffset, scales.byteLength), { kind: 'int4', shape: info.shape, numel, N, K }); }
    } else {
      let f16bits;
      if (info.dtype === 'BF16') f16bits = bf16ToF16bits(new Uint16Array(raw.buffer, raw.byteOffset, numel));
      else if (info.dtype === 'F16') f16bits = new Uint16Array(raw.buffer, raw.byteOffset, numel);
      else if (info.dtype === 'F32') f16bits = f32ToF16bits(new Float32Array(raw.buffer, raw.byteOffset, numel));
      else throw new Error('unsupported dtype ' + info.dtype + ' for ' + name);
      const buf = E.createBuffer(numel * 2, U.STORAGE | U.COPY_DST | U.COPY_SRC, name);
      E.device().queue.writeBuffer(buf, 0, f16bits);
      _weights[name] = { buf, shape: info.shape, numel };
      if (sink) await sink.add(name, 'buf', new Uint8Array(f16bits.buffer, f16bits.byteOffset, f16bits.byteLength), { kind: 'f16', shape: info.shape, numel });
    }
  }

  // Parse a single safetensors ArrayBuffer (whole shard in RAM) — used for the
  // small single-file path (0.6B, ~1.1GB). Big sharded models go through the
  // OPFS slice path (_parseSafetensorsFile) to avoid a multi-GB contiguous alloc.
  async function _parseSafetensors(ab, onPct, sink) {
    const headerLen = Number(new DataView(ab, 0, 8).getBigUint64(0, true));
    const header = JSON.parse(dec_(new Uint8Array(ab, 8, headerLen)));
    const dataStart = 8 + headerLen;
    const names = Object.keys(header).filter(n => n !== '__metadata__');
    for (let i = 0; i < names.length; i++) {
      const name = names[i], info = header[name];
      const [begin, end] = info.data_offsets;
      await _uploadTensor(name, info, new Uint8Array(ab, dataStart + begin, end - begin), sink);
      if ((i & 15) === 0) onPct && onPct(Math.round(i / names.length * 100));
    }
    onPct && onPct(100);
  }

  // ---- Chunked shard streaming (big sharded models) ------------------------
  // A single contiguous ArrayBuffer for a multi-GB shard fails to allocate on a
  // RAM-constrained box ("Array buffer allocation failed") — and OPFS/CacheStorage
  // write-through needs that many GB of *storage* quota, which is often tighter
  // than RAM. Instead we hold the shard as a LIST of small chunks (each alloc is
  // tiny; total is fine in RAM) and read each tensor's byte-range ACROSS chunks.
  // Caching is best-effort (Cache Storage via a Blob — no contiguous alloc); if
  // quota blocks it we just re-download next time. No single >2GB buffer ever.
  async function _streamToChunks(body, total, onProgress) {
    const reader = body.getReader(); const chunks = []; let recv = 0;
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      chunks.push(value); recv += value.length;
      if (total && onProgress) onProgress({ phase: 'download', pct: Math.round(recv / total * 100), recv, total });
    }
    return chunks;
  }
  // Download a shard as a chunk list (no raw-safetensors caching — the quantized-weights
  // cache supersedes it). Pure downloader: stream the body into a list of small chunks.
  async function _fetchShardChunks(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('download failed: HTTP ' + resp.status + ' for ' + url);
    const total = +(resp.headers.get('content-length') || 0);
    return await _streamToChunks(resp.body, total, onProgress);
  }
  // Random-access reader over a chunk list — assembles a contiguous Uint8Array
  // for any [begin,end) byte range, copying across chunk boundaries.
  function _chunkReader(chunks) {
    const starts = new Array(chunks.length); let off = 0;
    for (let i = 0; i < chunks.length; i++) { starts[i] = off; off += chunks[i].length; }
    return function readRange(begin, end) {
      const len = end - begin, out = new Uint8Array(len);
      // binary-search the chunk containing `begin`
      let lo = 0, hi = chunks.length - 1, ci = 0;
      while (lo <= hi) { const mid = (lo + hi) >> 1; if (starts[mid] <= begin) { ci = mid; lo = mid + 1; } else hi = mid - 1; }
      let w = 0;
      for (let i = ci; i < chunks.length && w < len; i++) {
        const cStart = starts[i], cEnd = cStart + chunks[i].length;
        const from = Math.max(begin, cStart) - cStart;
        const to = Math.min(end, cEnd) - cStart;
        if (to > from) { out.set(chunks[i].subarray(from, to), w); w += (to - from); }
      }
      return out;
    };
  }
  // Parse a safetensors shard held as a chunk list (no whole-shard alloc).
  async function _parseSafetensorsChunks(chunks, onPct, sink) {
    const read = _chunkReader(chunks);
    const headLen = Number(new DataView(read(0, 8).buffer).getBigUint64(0, true));
    const header = JSON.parse(dec_(read(8, 8 + headLen)));
    const dataStart = 8 + headLen;
    const names = Object.keys(header).filter(n => n !== '__metadata__');
    for (let i = 0; i < names.length; i++) {
      const name = names[i], info = header[name];
      const [begin, end] = info.data_offsets;
      await _uploadTensor(name, info, read(dataStart + begin, dataStart + end), sink);
      if ((i & 7) === 0) onPct && onPct(Math.round(i / names.length * 100));
    }
    onPct && onPct(100);
  }

  // Dims that define the quantized layout — a cache built for one must not load for another.
  const _cfgKey = (c) => ({ numLayers: c.numLayers, hidden: c.hidden, nHeads: c.nHeads, nKvHeads: c.nKvHeads, headDim: c.headDim, intermediate: c.intermediate, vocab: c.vocab });

  // ---- Quantized-weights cache WRITER ----
  // Streams GPU-ready segment bytes into 64MB Cache Storage chunks + a manifest. Each
  // segment is 16-aligned in the global byte stream so every reader writeBuffer slice is
  // 4-aligned (a WebGPU requirement). The manifest is the commit point: deleted first, so
  // a partial write is never read; written LAST on finalize. Best-effort (caller ignores throws).
  async function _makeQuantSink(variant) {
    const cache = await caches.open(QCACHE_NAME);
    await cache.delete(_qUrl(variant, 'manifest'));      // invalidate any prior cache up-front
    let buf = new Uint8Array(QCHUNK), used = 0, chunkIdx = 0, globalOff = 0;
    const segs = [];
    const flush = async () => {
      if (used === 0) return;
      await cache.put(_qUrl(variant, 'c' + chunkIdx), new Response(new Blob([buf.subarray(0, used)])));
      chunkIdx++; buf = new Uint8Array(QCHUNK); used = 0;
    };
    const writeBytes = async (u8) => {
      let src = 0;
      while (src < u8.byteLength) {
        if (used === QCHUNK) await flush();
        const n = Math.min(QCHUNK - used, u8.byteLength - src);
        buf.set(u8.subarray(src, src + n), used); used += n; src += n;
      }
    };
    return {
      async add(name, role, u8, meta) {
        const pad = _ceil16(globalOff) - globalOff;     // 16-align this segment's start
        if (pad) { await writeBytes(new Uint8Array(pad)); globalOff += pad; }
        segs.push({ name, role, off: globalOff, len: u8.byteLength, kind: meta.kind, shape: meta.shape, numel: meta.numel, N: meta.N, K: meta.K });
        await writeBytes(u8); globalOff += u8.byteLength;
      },
      async finalize() {
        await flush();
        const manifest = { ver: QCACHE_VER, variant, qgroup: QGROUP, chunkSize: QCHUNK, nChunks: chunkIdx, totalBytes: globalOff, config: _cfgKey(CONFIG), segs };
        await cache.put(_qUrl(variant, 'manifest'), new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } }));
      },
    };
  }

  // ---- Quantized-weights cache READER (fast load) ----
  // Reads the manifest, allocates every GPU buffer, then reads the 64MB chunks in
  // PARALLEL and writeBuffer()s their slices STRAIGHT into the GPU buffers — peak host
  // memory = QREAD_CONC chunks (64MB each), no whole-model / whole-tensor staging, no
  // re-download, no re-quantize. Returns false (cold) if no valid cache for this variant.
  async function loadQuantCache(variant, onProgress) {
    let cache; try { cache = await caches.open(QCACHE_NAME); } catch (_) { return false; }
    const mResp = await cache.match(_qUrl(variant, 'manifest'));
    if (!mResp) return false;
    let m; try { m = await mResp.json(); } catch (_) { return false; }
    if (!m || m.ver !== QCACHE_VER || m.variant !== variant || m.qgroup !== QGROUP || m.chunkSize !== QCHUNK) return false;
    const want = _cfgKey(CONFIG);
    for (const k in want) if (m.config[k] !== want[k]) return false;
    // Allocate buffers + rebuild _weights records from the manifest.
    _weights = {};
    const segBufs = new Array(m.segs.length);
    for (let i = 0; i < m.segs.length; i++) {
      const sg = m.segs[i];
      const b = E.createBuffer(_ceil16(sg.len), U.STORAGE | U.COPY_DST | U.COPY_SRC, sg.name + '.' + sg.role);
      segBufs[i] = b;
      if (sg.kind === 'int4') {
        let rec = _weights[sg.name] || (_weights[sg.name] = { int4: true, N: sg.N, K: sg.K, shape: sg.shape, numel: sg.numel });
        if (sg.role === 'pack') rec.pack = b; else rec.scales = b;
      } else {
        _weights[sg.name] = { buf: b, shape: sg.shape, numel: sg.numel };
      }
    }
    const q = E.device().queue;
    let done = 0, ci = 0;
    const readChunk = async (idx) => {
      const r = await cache.match(_qUrl(variant, 'c' + idx));
      if (!r) throw new Error('quant cache missing chunk ' + idx);
      const u8 = new Uint8Array(await r.arrayBuffer());
      const cStart = idx * QCHUNK, cEnd = cStart + u8.byteLength;
      // binary-search the first segment that reaches into this chunk (segs sorted by off)
      let lo = 0, hi = m.segs.length - 1, first = m.segs.length;
      while (lo <= hi) { const mid = (lo + hi) >> 1; if (m.segs[mid].off + _ceil16(m.segs[mid].len) > cStart) { first = mid; hi = mid - 1; } else lo = mid + 1; }
      for (let i = first; i < m.segs.length; i++) {
        const sg = m.segs[i]; if (sg.off >= cEnd) break;
        const ov0 = Math.max(sg.off, cStart), ov1 = Math.min(sg.off + sg.len, cEnd);
        if (ov1 > ov0) q.writeBuffer(segBufs[i], ov0 - sg.off, u8, ov0 - cStart, ov1 - ov0);
      }
      done++; onProgress && onProgress({ phase: 'parse', pct: Math.round(done / m.nChunks * 100) });
    };
    const pool = [];
    for (let w = 0; w < Math.min(QREAD_CONC, m.nChunks); w++) pool.push((async () => { for (;;) { const my = ci++; if (my >= m.nChunks) break; await readChunk(my); } })());
    await Promise.all(pool);
    return true;
  }

  async function loadModel({ onProgress, variant = '0.6B', nCtx } = {}) {
    // nCtx (provider.contextWindow) only sets the growth CEILING (_ctxCap), clamped to _attnMaxS
    // (the decode-attention's hard limit) — it does NOT eagerly size MAX_SEQ. MAX_SEQ stays small
    // and the auto-grow in _streamIds expands it to fit each prompt, so we never pre-allocate a
    // huge KV. Already-loaded: just re-apply the ceiling (_attnMaxS already known).
    if (_loaded && _variant === variant) {
      if ((nCtx | 0) > 0) _ctxCap = Math.max(1024, Math.min(_attnMaxS, nCtx | 0));
      return;
    }
    if (_loaded) unload();
    if (!(variant in CONFIGS)) throw new Error('unknown Qwen3 variant: ' + variant);
    Object.assign(CONFIG, CONFIGS[variant]);
    MODEL_ROOT = MODEL_ROOTS[variant];
    _variant = variant;
    await E.init();
    // Decode-attention key capacity = hard context ceiling. The `sc` shared array is f32, so
    // MAXS*4 + ~3KB (qd/red/part) must fit the device's workgroup-storage limit. Leave 4KB slack.
    try { const lim = (E.device().limits.maxComputeWorkgroupStorageSize | 0); if (lim >= 8192) _attnMaxS = Math.max(2048, Math.min(8192, Math.floor((lim - 4096) / 4))); } catch (_) {}
    _ctxCap = (nCtx | 0) > 0 ? Math.max(1024, Math.min(_attnMaxS, nCtx | 0)) : _attnMaxS;
    if (MAX_SEQ > _attnMaxS) MAX_SEQ = _attnMaxS;   // never allocate beyond the decode-attn capacity
    try { await probeF16Gemm(); } catch (_) {}   // pick f16 vs f32 prefill-GEMM dot for this GPU
    await TOK.load(MODEL_ROOT);
    onProgress && onProgress({ phase: 'tokenizer', pct: 100 });

    // FAST PATH: quantized-weights cache (skips download AND re-quantize; streams the
    // 64MB chunks in parallel straight into GPU buffers).
    try {
      onProgress && onProgress({ phase: 'cache', pct: 0 });
      if (await loadQuantCache(variant, onProgress)) {
        onProgress && onProgress({ phase: 'parse', pct: 100 });
        _loaded = true; return;
      }
    } catch (e) { console.warn('[qwen3] quant cache load failed — re-downloading:', (e && e.message) || e); }

    // SLOW PATH (first load / cache miss): download + quantize, capturing GPU-ready bytes
    // into the quant cache so the NEXT load takes the fast path. We no longer cache the raw
    // safetensors (the quant cache supersedes it); drop any stale raw cache to reclaim space.
    let sink = null; try { sink = await _makeQuantSink(variant); } catch (_) { sink = null; }
    try { await caches.delete(CACHE_NAME); } catch (_) {}

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
        const base = s / shardFiles.length, step = 1 / shardFiles.length;
        const dlProg = p => {
          if (!p) return;
          if (p.phase === 'download') onProgress && onProgress({ phase: 'download', pct: Math.round((base + step * p.pct / 100) * 100), recv: p.recv, total: p.total });
          else if (p.phase === 'cache') onProgress && onProgress({ phase: 'cache', pct: 100 });
        };
        const parseProg = pct => onProgress && onProgress({ phase: 'parse', pct: Math.round((s + pct / 100) / shardFiles.length * 100) });
        // Hold the shard as a chunk LIST + range-read per tensor — never a
        // contiguous multi-GB buffer (the "Array buffer allocation failed" cause).
        let chunks = await _fetchShardChunks(url, dlProg);
        await _parseSafetensorsChunks(chunks, parseProg, sink);
        chunks = null;   // free this shard before downloading the next
      }
    } else {
      const ab = await fetchModelBytes(MODEL_ROOT + 'model.safetensors', onProgress);
      onProgress && onProgress({ phase: 'parse', pct: 0 });
      await _parseSafetensors(ab, pct => onProgress && onProgress({ phase: 'parse', pct }), sink);
    }

    if (sink) { try { await sink.finalize(); } catch (e) { console.warn('[qwen3] quant cache write failed (will re-quantize next load):', (e && e.message) || e); } }
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
  // _attnMaxS = the decode-attention's per-key shared-memory capacity (the `sc` array in
  // ATTN_DEC_WGSL) and therefore the HARD context ceiling: if the sequence S exceeds it, the
  // decode kernel writes sc[] OUT OF BOUNDS → corruption → "!!!!" garbage (the bug that hit at
  // S>4096). Sized at load from the device's workgroup-storage limit (32KB on gen-12lp → 7168);
  // sc is f32 so MAXS·4 + ~3KB of other shared must fit. Context can NEVER exceed this.
  let _attnMaxS = 4096;     // decode-attn key capacity = hard context ceiling; set in loadModel
  // MAX_SEQ = KV buffer size = the context actually ALLOCATED. Starts small and grows ON DEMAND
  // to fit each prompt (auto-grow in _streamIds) so we never pre-allocate a huge KV. _ctxCap is
  // the growth ceiling = min(provider.contextWindow, _attnMaxS), set at load.
  let MAX_SEQ = 4096;       // currently-allocated KV window (grows to fit, never shrinks in a session)
  let _ctxCap = 4096;       // growth ceiling = min(provider.contextWindow, _attnMaxS)
  function _clampCtx(n) { n = n | 0; if (!n) return 4096; return Math.max(1024, Math.min(_ctxCap, n)); }
  // Grow the context window (KV buffer size) to `want`, freeing the old KV so ensureKv reallocs at
  // the new size. Used to auto-fit a prompt longer than the current MAX_SEQ instead of erroring.
  function _growCtx(want) {
    if (want <= MAX_SEQ) return;
    MAX_SEQ = want;
    try { if (_kv) for (const l of _kv) { if (l.k && l.k.destroy) l.k.destroy(); if (l.v && l.v.destroy) l.v.destroy(); } } catch (_) {}
    try { if (_tokHist && _tokHist.destroy) _tokHist.destroy(); } catch (_) {}
    _kv = null; _tokHist = null; _cachedIds = null; _sysAnchor = null;   // realloc + invalidate prefix caches
  }
  let _PERF = false, _perfData = null;   // CPU phase profiler (encode vs readback)
  let _kv = null;     // [{k,v}] per layer, sized MAX_SEQ
  let _scr = null;    // scratch buffers, sized to _scrT rows
  let _scrT = 0;
  // PREFIX CACHE: the exact token sequence currently resident in KV[0..length).
  // A new prompt that shares a leading run with this (same tokens at the same
  // absolute positions → identical RoPE phase) reuses that KV and only prefills
  // the differing tail. Invalidated on unload and on any generate() (which
  // clobbers KV from position 0). See [[reference_qwen3-dense-webgpu-variants]].
  let _cachedIds = null;
  // SYSTEM-PROMPT CACHE (in-memory, zero-copy): once the stable system block (system prompt +
  // tool descriptions + skill defs + skill guidance — all concatenated into `sys`) is prefilled,
  // its KV lives in _kv[0..P_sys) and is never overwritten (decode writes [L..); later prefills
  // start at [P_sys..)). This "anchor" records that _kv[0..ids.length) holds exactly this system
  // prefix for this variant, so a new conversation's FIRST message can skip re-prefilling it even
  // when _cachedIds is null/stale. Invalidated wherever _kv[0..P_sys) could be clobbered (model
  // switch, generate() from pos 0, unload).
  let _sysAnchor = null;   // { variant, ids } | null
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
      // Attention input norm → q/k/v. DECODE (T=1) is the fused int8 path: rmsnormQ emits the
      // per-group int8 activation DIRECTLY from the norm (no separate quantize dispatch) and
      // q/k/v all read that one quantized vector. PREFILL (T>1) keeps rmsnorm + linearQ.
      if (T === 1) {
        const qW = Wq(p + 'self_attn.q_proj.weight'), kW = Wq(p + 'self_attn.k_proj.weight'), vW = Wq(p + 'self_attn.v_proj.weight');
        rmsnormQ(s.x, W(p + 'input_layernorm.weight'), H, C.rmsEps);
        gemvDP4_only(qW.pack, qW.scales, s.q, nHq * hd, H);
        gemvDP4_only(kW.pack, kW.scales, s.k, nKv * hd, H);
        gemvDP4_only(vW.pack, vW.scales, s.v, nKv * hd, H);
      } else {
        await rmsnorm(s.x, W(p + 'input_layernorm.weight'), s.normed, T, H, C.rmsEps);
        await linearQ(s.normed, Wq(p + 'self_attn.q_proj.weight'), s.q, T, nHq * hd, H);
        await linearQ(s.normed, Wq(p + 'self_attn.k_proj.weight'), s.k, T, nKv * hd, H);
        await linearQ(s.normed, Wq(p + 'self_attn.v_proj.weight'), s.v, T, nKv * hd, H);
      }
      // NOTE: ropeQK must NOT be called in-place — aliasing the same buffer to a
      // read and a read_write binding is undefined behavior in WebGPU (miscompiles
      // on Intel). Write rope output to a separate buffer.
      await ropeQK(s.q, W(p + 'self_attn.q_norm.weight'), s.qr, T, nHq, hd, posBase, C.ropeTheta, C.rmsEps);
      await ropeQK(s.k, W(p + 'self_attn.k_norm.weight'), s.kr, T, nKv, hd, posBase, C.ropeTheta, C.rmsEps);
      copyRange(s.kr, _kv[l].k, posBase * nKv * hd, T * nKv * hd);
      copyRange(s.v, _kv[l].v, posBase * nKv * hd, T * nKv * hd);
      await attention(s.qr, _kv[l].k, _kv[l].v, s.attn, T, S, nHq, nKv, hd);
      await linearQ(s.attn, Wq(p + 'self_attn.o_proj.weight'), s.x, T, H, nHq * hd, true);   // fused residual: x += o_proj
      // MLP norm → gate/up. Decode fuses the same way: rmsnormQ emits the int8 activation,
      // gateUpSiluDP4_only consumes it (no separate quantize). Prefill (T>1) keeps the
      // rmsnorm + gate/up + swiglu path.
      if (T === 1) {
        const gW = Wq(p + 'mlp.gate_proj.weight'), uW = Wq(p + 'mlp.up_proj.weight');
        rmsnormQ(s.x, W(p + 'post_attention_layernorm.weight'), H, C.rmsEps);
        gateUpSiluDP4_only(gW, uW, s.swi, I, H);
      } else {
        await rmsnorm(s.x, W(p + 'post_attention_layernorm.weight'), s.normed, T, H, C.rmsEps);
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
  // DEEP-PIPELINED decode. The single yield point — `await read` (mapAsync) — is
  // VSYNC-PACED and contended by foreground compositing when the tab is VISIBLE, so
  // its resolution lags far behind the GPU's compute (measured: decode ~3.4× slower
  // focused vs backgrounded — 2.9 vs 9.8 tok/s @3200 ctx — even with the engine in a
  // worker, because this is GPU-process/device-tick pacing, not JS-thread contention).
  // Two defenses: (1) GEN_BATCH=32 → 4× fewer readback syncs, so each throttled
  // resolution delivers 4× more tokens; (2) keep PIPE_DEPTH batches in flight so the
  // GPU always has queued compute and never idles waiting for a readback. Each token
  // is still its own bounded submit (no single giant submit → no TDR). On stop/abort
  // we stop emitting immediately and discard the ≤PIPE_DEPTH-1 already-queued batches
  // (bounded wasted compute + bounded KV overrun, both harmless).
  async function decodeLoop(pos, maxTokens, emitTok, signal, count) {
    let submitted = 0;
    const submitBatch = async () => {
      if (signal && signal.aborted) return null;
      const K = Math.min(GEN_BATCH, MAX_SEQ - 1 - pos, maxTokens + GEN_BATCH - submitted);
      if (K <= 0) return null;
      const base = pos;
      for (let k = 0; k < K; k++) await forward(null, base + k, { chain: true, submitOnly: true });
      pos += K; submitted += K;
      return { read: readU32Range(_tokHist, base + 1, K), K };   // copy submit enqueued AFTER the forwards
    };
    const inflight = [];
    while (inflight.length < PIPE_DEPTH) { const b = await submitBatch(); if (!b) break; inflight.push(b); }
    while (inflight.length) {
      const cur = inflight.shift();
      const toks = await cur.read;
      let stop = !!(signal && signal.aborted);
      if (!stop) for (let k = 0; k < cur.K; k++) { if (!emitTok(toks[k]) || count() >= maxTokens) { stop = true; break; } }
      if (stop) break;                                          // discard remaining in-flight batches
      const b = await submitBatch(); if (b) inflight.push(b);   // top the pipeline back up
    }
  }

  // Greedy generate — thin wrapper over the double-buffered decodeLoop (see above):
  // GEN_BATCH chained forwards per batch (chained through _tokHist on the GPU, no
  // mid-batch readback), and the next batch is submitted before the current batch's
  // readback is awaited so the GPU never drains between batches.
  const STOP = (t) => t === SPECIAL.im_end || t === SPECIAL.endoftext;
  const GEN_BATCH = 32;   // tokens/GPU-resident batch (1 readback each) — big to amortize the
                          // vsync-throttled readback when the tab is focused (see decodeLoop).
  const PIPE_DEPTH = 3;   // batches kept in flight so the GPU never idles awaiting a readback.
  async function generate(prompt, { maxTokens = 64, onToken, signal } = {}) {
    await loadModel({ variant: _variant });
    _cachedIds = null; _sysAnchor = null;   // one-shot path prefills KV from pos 0 → invalidate any prefix cache
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

  // A/B + correctness: the experimental DP4A int8 GEMM (gemmDP4A) vs the f16-dequant
  // gemmQ at a realistic prefill shape. relErr is gemmDP4A-vs-gemmQ (int8-activation
  // quant ~1%); speedup = gemmQ_ms / dp4_ms (incl. the QUANTQ8T activation pre-pass).
  async function _benchGemmDP4({ T = 256, N = 3072, K = 1024, iters = 6 } = {}) {
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);
    const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.013) * 0.7;
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), pb = mk(pack), sb = mk(scales), ya = E.createBuffer(T * N * 4, UF, 'ya'), yd = E.createBuffer(T * N * 4, UF, 'yd');
    const wrec = { pack: pb, scales: sb };
    uniformReset(); await gemmQ(xb, wrec, ya, T, N, K);
    uniformReset(); await gemmDP4A(xb, wrec, yd, T, N, K);
    await E.device().queue.onSubmittedWorkDone();
    const a = await E.readF32(ya, T * N), b = await E.readF32(yd, T * N);
    let mx = 0, ref = 0; for (let i = 0; i < a.length; i++) { mx = Math.max(mx, Math.abs(a[i] - b[i])); ref = Math.max(ref, Math.abs(a[i])); }
    const relErr = +(mx / (ref || 1)).toExponential(2);
    const time = async (fn) => { uniformReset(); await fn(); await E.device().queue.onSubmittedWorkDone(); const t0 = performance.now(); for (let i = 0; i < iters; i++) { uniformReset(); await fn(); } await E.device().queue.onSubmittedWorkDone(); return (performance.now() - t0) / iters; };
    const q_ms = await time(() => gemmQ(xb, wrec, ya, T, N, K));
    const dp_ms = await time(() => gemmDP4A(xb, wrec, yd, T, N, K));
    [xb, pb, sb, ya, yd].forEach(bf => bf.destroy());
    return { T, N, K, relErr, gemmQ_ms: +q_ms.toFixed(2), gemmDP4_ms: +dp_ms.toFixed(2), speedup: +(q_ms / dp_ms).toFixed(2) };
  }

  // RELIABLE A/B: GPU-TIMESTAMP, MIN-of-reps (the only stable method on this throttling
  // iGPU — wall-clock swings ±50%). Times pure GPU compute per call and separates the DP4A
  // activation-quantize (label *quant*) from the GEMM, so dp4gemm = the cost if the quantize
  // is fused/amortized (the real forward shares one rmsnorm output across q/k/v), and
  // dp4full = the current per-call cost. min over reps ≈ the cool/unthrottled floor.
  async function _benchGemmTS({ T = 256, N = 6144, K = 2048, iters = 10, reps = 14 } = {}) {
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);
    const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.013) * 0.7;
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), pb = mk(pack), sb = mk(scales), yb = E.createBuffer(T * N * 4, UF, 'yb');
    const wrec = { pack: pb, scales: sb };
    uniformReset(); await gemmQ(xb, wrec, yb, T, N, K); uniformReset(); await gemmDP4A(xb, wrec, yb, T, N, K);
    await E.device().queue.onSubmittedWorkDone();
    const prof = async (fn, cap) => { uniformReset(); E.beginProfile(cap); E.beginBatch(); for (let i = 0; i < iters; i++) { uniformReset(); await fn(); } await E.endBatch(); return await E.endProfile(); };
    const sum = (p, pred) => p.filter(pred).reduce((s, r) => s + r.us, 0) / iters;
    let qUs = Infinity, gUs = Infinity, quUs = Infinity;
    for (let r = 0; r < reps; r++) {
      const pq = await prof(() => gemmQ(xb, wrec, yb, T, N, K), iters + 8);
      qUs = Math.min(qUs, sum(pq, () => true));
      const pd = await prof(() => gemmDP4A(xb, wrec, yb, T, N, K), iters * 2 + 8);
      quUs = Math.min(quUs, sum(pd, x => /quant/i.test(x.label)));
      gUs = Math.min(gUs, sum(pd, x => !/quant/i.test(x.label)));
    }
    [xb, pb, sb, yb].forEach(b => b.destroy());
    const gflops = (us) => +(2 * T * N * K / us / 1e3).toFixed(0);
    return { T, N, K, reps,
      f16_us: +qUs.toFixed(1), f16_gflops: gflops(qUs),
      dp4gemm_us: +gUs.toFixed(1), dp4gemm_gflops: gflops(gUs),
      dp4quant_us: +quUs.toFixed(1),
      speedup_gemmOnly: +(qUs / gUs).toFixed(2),       // if activation-quantize is fused/amortized
      speedup_perCall: +(qUs / (gUs + quUs)).toFixed(2) };   // current per-call (quant each time)
  }

  // TEXTURE-GEMM A/B (the MLDrift port). GPU-timestamp min-of-reps — same method as
  // _benchGemmTS — comparing: gemmDP4A (current default), gemmQ (f16), gemmTex on an
  // rgba32float X texture, and gemmTex on an rgba16float X texture (the bandwidth lever).
  // Correctness of both texture variants checked vs gemmQ. Returns GFLOP/s per variant
  // so we can see whether textures beat our shared-mem-staged kernels on THIS iGPU
  // before committing to a full texture rewrite. Run in docs/webgpu-engine-test.html.
  async function _benchGemmTex({ T = 256, N = 6144, K = 2048, iters = 10, reps = 14 } = {}) {
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);
    const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.013) * 0.7;
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const xb = mk(x), pb = mk(pack), sb = mk(scales), yb = E.createBuffer(T * N * 4, UF, 'yb');
    const wrec = { pack: pb, scales: sb };
    const tex32 = E.texFromF32(x, T, K, 'Xt32');
    const tex16 = E.texFromF16(x, T, K, 'Xt16');
    // Correctness: texture variants vs gemmQ (f16 reference).
    uniformReset(); await gemmQ(xb, wrec, yb, T, N, K); await E.device().queue.onSubmittedWorkDone();
    const ref = await E.readF32(yb, T * N);
    const chk = async (view) => {
      uniformReset(); await gemmTexQ(view, wrec, yb, T, N, K); await E.device().queue.onSubmittedWorkDone();
      const g = await E.readF32(yb, T * N);
      let mx = 0, rf = 0; for (let i = 0; i < g.length; i++) { mx = Math.max(mx, Math.abs(g[i] - ref[i])); rf = Math.max(rf, Math.abs(ref[i])); }
      return +(mx / (rf || 1)).toExponential(2);
    };
    const relErr32 = await chk(tex32.view), relErr16 = await chk(tex16.view);
    const prof = async (fn, cap) => { uniformReset(); E.beginProfile(cap); E.beginBatch(); for (let i = 0; i < iters; i++) { uniformReset(); await fn(); } await E.endBatch(); return await E.endProfile(); };
    const sumAll = (p) => p.reduce((s, r) => s + r.us, 0) / iters;
    let qUs = Infinity, dUs = Infinity, t32Us = Infinity, t16Us = Infinity;
    for (let r = 0; r < reps; r++) {
      qUs = Math.min(qUs, sumAll(await prof(() => gemmQ(xb, wrec, yb, T, N, K), iters + 8)));
      dUs = Math.min(dUs, sumAll((await prof(() => gemmDP4A(xb, wrec, yb, T, N, K), iters * 2 + 8)).filter(x => !/quant/i.test(x.label))));
      t32Us = Math.min(t32Us, sumAll(await prof(() => gemmTexQ(tex32.view, wrec, yb, T, N, K), iters + 8)));
      t16Us = Math.min(t16Us, sumAll(await prof(() => gemmTexQ(tex16.view, wrec, yb, T, N, K), iters + 8)));
    }
    [xb, pb, sb, yb].forEach(b => b.destroy());
    try { tex32.tex.destroy(); tex16.tex.destroy(); } catch (_) {}
    const gflops = (us) => +(2 * T * N * K / us / 1e3).toFixed(0);
    return { T, N, K, reps, relErr_tex32: relErr32, relErr_tex16: relErr16,
      f16_us: +qUs.toFixed(1), f16_gflops: gflops(qUs),
      dp4_us: +dUs.toFixed(1), dp4_gflops: gflops(dUs),
      tex32_us: +t32Us.toFixed(1), tex32_gflops: gflops(t32Us),
      tex16_us: +t16Us.toFixed(1), tex16_gflops: gflops(t16Us),
      tex16_vs_dp4: +(dUs / t16Us).toFixed(2), tex16_vs_f16: +(qUs / t16Us).toFixed(2) };
  }

  // FAITHFUL MLDrift texture-GEMM A/B. Builds a token-major X texture (texel(x=token,
  // y=k4)) + f16 weight buffer, validates gemmTex2's math vs a CPU reference at a small
  // shape, then GPU-timestamp min-of-reps benches gemmTex2 (f32 tex & f16 tex) vs our
  // gemmDP4A / gemmQ at the real shape. Weights for tex2 are plain f16 (no int4 unpack)
  // — this ISOLATES the texture-activation + split-K question. If tex2 still loses to
  // gemmDP4A despite cheaper weights, textures genuinely don't transfer to this iGPU.
  async function _benchGemmTex2({ T = 256, N = 6144, K = 2048, iters = 6, reps = 8 } = {}) {
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    // Build a token-major texture from X[rows,cols] row-major: tex[k4][token][0..3] = X[token,4k4..].
    const buildTex = (X, rows, cols, f16) => {
      const w = rows, h = cols / 4, comp = new Float32Array(w * h * 4);   // [h=k4][w=token][4]
      for (let t = 0; t < rows; t++) for (let k = 0; k < cols; k++) comp[(Math.floor(k / 4)) * w * 4 + t * 4 + (k % 4)] = X[t * cols + k];
      const fmt = f16 ? 'rgba16float' : 'rgba32float';
      const data = f16 ? E.f32ToF16(comp) : comp;
      const tex = E.createTexture2D(w, h, fmt, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, 'Xtm');
      E.device().queue.writeTexture({ texture: tex }, data, { bytesPerRow: w * (f16 ? 8 : 16), rowsPerImage: h }, { width: w, height: h });
      return { tex, view: E.texView(tex, 'Xtm') };
    };
    // ---- correctness: small shape, f16 weights, vs CPU ----
    {
      const t = 8, n = 32, k = 64;
      const Xs = new Float32Array(t * k); for (let i = 0; i < Xs.length; i++) Xs[i] = Math.sin(i * 0.11);
      const Ws = new Float32Array(n * k); for (let i = 0; i < Ws.length; i++) Ws[i] = Math.cos(i * 0.07) * 0.5;
      const Wh = E.f32ToF16(Ws);
      const wb = E.createBuffer(Wh.byteLength, UF, 'wf16'); E.device().queue.writeBuffer(wb, 0, Wh);
      const yb = E.createBuffer(t * n * 4, UF, 'ys'); const tx = buildTex(Xs, t, k, false);
      uniformReset(); await gemmTex2(tx.view, wb, yb, t, n, k, false, 8); await E.device().queue.onSubmittedWorkDone();
      const got = await E.readF32(yb, t * n);
      // CPU ref using f16-rounded weights (decode the halfs back to f32)
      const h2f = (h) => { const s = (h & 0x8000) ? -1 : 1; const e = (h >> 10) & 0x1f; const m = h & 0x3ff; if (e === 0) return s * m * 2 ** -24; if (e === 31) return m ? NaN : s * Infinity; return s * (1 + m / 1024) * 2 ** (e - 15); };
      let mx = 0, rf = 0;
      for (let r = 0; r < t; r++) for (let c = 0; c < n; c++) { let a = 0; for (let kk = 0; kk < k; kk++) a += Xs[r * k + kk] * h2f(Wh[c * k + kk]); mx = Math.max(mx, Math.abs(a - got[r * n + c])); rf = Math.max(rf, Math.abs(a)); }
      var tex2_relErr = +(mx / (rf || 1)).toExponential(2);
      [wb, yb, tx.tex].forEach(b => b.destroy && b.destroy());
    }
    // ---- speed: real shape ----
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);
    const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.013) * 0.7;
    const wf16 = new Uint16Array(N * K); for (let i = 0; i < wf16.length; i++) wf16[i] = 0x3000 + (i % 11);
    const xb = mk(x), pb = mk(pack), sb = mk(scales), wb = mk(wf16), yb = E.createBuffer(T * N * 4, UF, 'yb');
    const wrec = { pack: pb, scales: sb };
    const tx32 = buildTex(x, T, K, false), tx16 = buildTex(x, T, K, true);
    const TPXS = [8, 16, 32];   // sweep workgroup width (TPX*8 threads): 64/128/256
    uniformReset(); await gemmDP4A(xb, wrec, yb, T, N, K); for (const tp of TPXS) { uniformReset(); await gemmTex2(tx16.view, wb, yb, T, N, K, false, tp); }
    await E.device().queue.onSubmittedWorkDone();
    const prof = async (fn, cap) => { uniformReset(); E.beginProfile(cap); E.beginBatch(); for (let i = 0; i < iters; i++) { uniformReset(); await fn(); } await E.endBatch(); return await E.endProfile(); };
    const sumAll = (p) => p.reduce((s, r) => s + r.us, 0) / iters;
    const gflops = (us) => +(2 * T * N * K / us / 1e3).toFixed(0);
    let dUs = Infinity, qUs = Infinity;
    const texByTpx = {}; for (const tp of TPXS) texByTpx[tp] = Infinity;
    for (let r = 0; r < reps; r++) {
      dUs = Math.min(dUs, sumAll((await prof(() => gemmDP4A(xb, wrec, yb, T, N, K), iters * 2 + 8)).filter(z => !/quant/i.test(z.label))));
      qUs = Math.min(qUs, sumAll(await prof(() => gemmQ(xb, wrec, yb, T, N, K), iters + 8)));
      for (const tp of TPXS) texByTpx[tp] = Math.min(texByTpx[tp], sumAll(await prof(() => gemmTex2(tx16.view, wb, yb, T, N, K, false, tp), iters + 8)));
    }
    [xb, pb, sb, wb, yb].forEach(b => b.destroy());
    try { tx32.tex.destroy(); tx16.tex.destroy(); } catch (_) {}
    const texGflops = {}; let best = 0; for (const tp of TPXS) { texGflops['tex2_tpx' + tp + '_' + (tp * 8) + 'thr'] = gflops(texByTpx[tp]); best = Math.max(best, gflops(texByTpx[tp])); }
    return { T, N, K, reps, tex2_relErr,
      dp4_gflops: gflops(dUs), f16_gflops: gflops(qUs),
      ...texGflops,
      best_tex2_gflops: best, best_tex2_vs_dp4: +(best / gflops(dUs)).toFixed(2) };
  }

  // Bench the FAITHFUL MLDrift bulk-GEMM port (gemmTex3): SLM-staged f16 weights +
  // texture activations, 64-thread, no split-K. Head-to-head GFLOP/s vs gemmDP4A.
  // If tex3 lands near gemmDP4A, textures are neutral and the lever is SLM tiling
  // (which we already have); if tex3 wins, textures genuinely help on this iGPU.
  async function _benchGemmTex3({ T = 256, N = 6144, K = 2048, iters = 6, reps = 8 } = {}) {
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC, K4 = K / 4;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'bg'); E.device().queue.writeBuffer(b, 0, a.buffer, a.byteOffset || 0, a.byteLength); return b; };
    const wf = new Uint16Array(N * K); for (let i = 0; i < wf.length; i++) wf[i] = 0x3000 + (i % 11);    // packed f16 weights
    const x = new Float32Array(T * K); for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.013) * 0.7;
    const words = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(N * words); for (let i = 0; i < pack.length; i++) pack[i] = (Math.imul(i, 2654435761) >>> 0);
    const scales = new Uint16Array(N * gpr); for (let i = 0; i < scales.length; i++) scales[i] = 0x3000 + (i % 7);
    const wb = mk(wf), yb = E.createBuffer(T * N * 4, UF, 'yb'), xb = mk(x), pb = mk(pack), sb = mk(scales);
    const wrec = { pack: pb, scales: sb };
    // token-major f16 activation texture: texel(x=token, y=kstep)
    const w = T, h = K4, comp = new Float32Array(w * h * 4);
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) comp[(Math.floor(k / 4)) * w * 4 + t * 4 + (k % 4)] = x[t * K + k];
    const tex = E.createTexture2D(w, h, 'rgba16float', GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, 'Xtm');
    E.device().queue.writeTexture({ texture: tex }, E.f32ToF16(comp), { bytesPerRow: w * 8, rowsPerImage: h }, { width: w, height: h });
    const view = E.texView(tex, 'Xtm');
    uniformReset(); await gemmTex3(view, wb, yb, T, N, K); await E.device().queue.onSubmittedWorkDone();
    const ychk = await E.readF32(yb, 64); let maxAbs = 0; for (const v of ychk) maxAbs = Math.max(maxAbs, Math.abs(v));
    const prof = async (fn, cap) => { uniformReset(); E.beginProfile(cap); E.beginBatch(); for (let i = 0; i < iters; i++) { uniformReset(); await fn(); } await E.endBatch(); return await E.endProfile(); };
    const sumAll = (p) => p.reduce((s, r) => s + r.us, 0) / iters;
    let dUs = Infinity, t3 = Infinity;
    for (let r = 0; r < reps; r++) {
      dUs = Math.min(dUs, sumAll((await prof(() => gemmDP4A(xb, wrec, yb, T, N, K), iters * 2 + 8)).filter(z => !/quant/i.test(z.label))));
      t3 = Math.min(t3, sumAll(await prof(() => gemmTex3(view, wb, yb, T, N, K), iters + 8)));
    }
    [wb, yb, xb, pb, sb].forEach(b => b.destroy()); try { tex.destroy(); } catch (_) {}
    const gflops = (us) => +(2 * T * N * K / us / 1e3).toFixed(0);
    return { T, N, K, reps, outMaxAbs: +maxAbs.toFixed(2), dp4_gflops: gflops(dUs), tex3_gflops: gflops(t3), tex3_vs_dp4: +(gflops(t3) / gflops(dUs)).toFixed(2) };
  }

  // A/B: f32 attnFlash vs f16 attnFlash. Correctness (f16 vs f32 output) + GPU-timestamp
  // min-of-reps timing + GFLOP/s. Proves whether f16 lifts prefill attention off the f32
  // ceiling (~123 GFLOP/s). Default shape = real prefill self-attention (T=S).
  async function _benchAttnF16({ T = 2560, S = 2560, nHq = 16, nKv = 8, hd = 128, iters = 4, reps = 8 } = {}) {
    const Q = new Float32Array(T * nHq * hd); for (let i = 0; i < Q.length; i++) Q[i] = Math.sin(i * 0.011) * 0.5;
    const Kk = new Float32Array(S * nKv * hd); for (let i = 0; i < Kk.length; i++) Kk[i] = Math.cos(i * 0.007) * 0.5;
    const Vv = new Float32Array(S * nKv * hd); for (let i = 0; i < Vv.length; i++) Vv[i] = Math.sin(i * 0.005 + 1) * 0.5;
    const UF = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const mk = (a) => { const b = E.createBuffer(a.byteLength, UF, 'ab'); E.device().queue.writeBuffer(b, 0, a.buffer, 0, a.byteLength); return b; };
    const qb = mk(Q), kb = mk(Kk), vb = mk(Vv), of = E.createBuffer(Q.byteLength, UF, 'of'), o16 = E.createBuffer(Q.byteLength, UF, 'o16');
    const KTS = [4, 8, 16];
    // correctness vs the f32 kernel (per KT)
    uniformReset(); await attention(qb, kb, vb, of, T, S, nHq, nKv, hd); await E.device().queue.onSubmittedWorkDone();
    const n = Math.min(Q.length, 200000);
    const a = await E.readF32(of, n);
    const relErr = {};
    for (const KT of KTS) {
      uniformReset(); await attentionF16(qb, kb, vb, o16, T, S, nHq, nKv, hd, KT); await E.device().queue.onSubmittedWorkDone();
      const b = await E.readF32(o16, n);
      let mx = 0, rf = 0; for (let i = 0; i < n; i++) { mx = Math.max(mx, Math.abs(a[i] - b[i])); rf = Math.max(rf, Math.abs(a[i])); }
      relErr['KT' + KT] = +(mx / (rf || 1)).toExponential(2);
    }
    const prof = async (fn) => { uniformReset(); E.beginProfile(iters + 8); E.beginBatch(); for (let i = 0; i < iters; i++) { uniformReset(); await fn(); } await E.endBatch(); const p = await E.endProfile(); return p.reduce((s, r) => s + r.us, 0) / iters; };
    let f32u = Infinity; const f16u = {}; for (const KT of KTS) f16u[KT] = Infinity;
    for (let r = 0; r < reps; r++) {
      f32u = Math.min(f32u, await prof(() => attention(qb, kb, vb, of, T, S, nHq, nKv, hd)));
      for (const KT of KTS) f16u[KT] = Math.min(f16u[KT], await prof(() => attentionF16(qb, kb, vb, o16, T, S, nHq, nKv, hd, KT)));
    }
    [qb, kb, vb, of, o16].forEach(x => x.destroy());
    const pairs = (T === S) ? (T * (T + 1) / 2) : (T * (S - T) + T * (T + 1) / 2);
    const flop = pairs * nHq * hd * 4;
    const gflops = (us) => +(flop / us / 1e3).toFixed(0);
    const out = { T, S, relErr, f32_us: +f32u.toFixed(1), f32_gflops: gflops(f32u) };
    for (const KT of KTS) { out['f16_KT' + KT + '_us'] = +f16u[KT].toFixed(1); out['f16_KT' + KT + '_gflops'] = gflops(f16u[KT]); out['speedup_KT' + KT] = +(f32u / f16u[KT]).toFixed(2); }
    return out;
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
  // Free ALL GPU buffers (weights, KV, scratch, token history, ids, DP4A scratch). With
  // deep=true (backend switch — Qwen3 ↔ Gemma), also drop the uniform pool + DESTROY the
  // WebGPU device so ZERO GPU memory lingers and there's no two-backend overload; E.init()
  // + loadModel rebuild everything on next use. deep=false (variant swap) keeps the device.
  async function unload(deep) {
    try { if (_weights) for (const k in _weights) { const w = _weights[k]; if (!w) continue; if (w.pack && w.pack.destroy) try { w.pack.destroy(); } catch (_) {} if (w.scales && w.scales.destroy) try { w.scales.destroy(); } catch (_) {} if (w.buf && w.buf.destroy) try { w.buf.destroy(); } catch (_) {} } } catch (_) {}
    try { if (_kv) for (const l of _kv) { if (l.k && l.k.destroy) l.k.destroy(); if (l.v && l.v.destroy) l.v.destroy(); } } catch (_) {}
    try { if (_scr) for (const b of Object.values(_scr)) { if (b && b.destroy) try { b.destroy(); } catch (_) {} } } catch (_) {}
    try { if (_tokHist && _tokHist.destroy) _tokHist.destroy(); } catch (_) {}
    try { if (_idsBuf && _idsBuf.destroy) _idsBuf.destroy(); } catch (_) {}
    try { for (const b of _dp4dead) { if (b && b.destroy) try { b.destroy(); } catch (_) {} } if (_dp4) { _dp4.xq.destroy(); _dp4.xs.destroy(); } } catch (_) {}
    try { for (const b of _dp4gDead) { if (b && b.destroy) try { b.destroy(); } catch (_) {} } if (_dp4g) { _dp4g.xq.destroy(); _dp4g.xs.destroy(); } } catch (_) {}
    _dp4 = null; _dp4dead = []; _dp4g = null; _dp4gDead = [];
    _weights = null; _kv = null; _scr = null; _scrT = 0; _tokHist = null; _idsBuf = null; _idsCap = 0; _loaded = false; _cachedIds = null; _sysAnchor = null; MAX_SEQ = 4096;
    if (deep) {
      _uPool = []; _uIdx = 0;                 // uniform-pool buffers belong to the old device
      _f16Probed = false; _f16Math = null;    // re-probe against the rebuilt device
      try { await E.unload && E.unload(); } catch (_) {}
    }
  }

  // Stream from pre-encoded ids (same prefill+decode as generate(), but the
  // caller supplies the full chat token sequence and gets clean UTF-8 deltas).
  async function _streamIds(ids, { maxTokens = 512, onToken, signal } = {}) {
    await loadModel({ variant: _variant });
    const L = ids.length;
    // Grow the allocated context JUST ENOUGH to fit this prompt + decode headroom (never the full
    // configured ceiling), capped at _ctxCap. This keeps the KV cache as small as possible — a
    // 4200-tok prompt allocates ~5200 (~1.2GB), not 8192 (~1.9GB, which OOMs the iGPU → garbage).
    // Only errors if the prompt won't fit even at the ceiling.
    if (L + 64 > MAX_SEQ) {
      const want = Math.max(1024, Math.min(_ctxCap, L + 1024));
      if (L + 64 > want) throw new Error('prompt too long: ' + L + ' tokens — exceeds the context limit (' + _ctxCap + '). Raise "Context size" or shorten the prompt.');
      _growCtx(want);
    }
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
    // PREFIX CACHE — find the longest leading token run shared with the KV that's
    // already resident, and prefill only from there. The shared tokens sit at the
    // SAME absolute positions in both turns (common prefix starts at pos 0), so
    // their cached KV (RoPE-encoded by position) is bit-exact reusable. Always run
    // at least the final token so we get fresh logits to start decoding from.
    let P = 0;
    if (_cachedIds) {
      const m = Math.min(_cachedIds.length, L);
      while (P < m && _cachedIds[P] === ids[P]) P++;
    }
    if (P > L - 1) P = L - 1;
    const _reused = P;
    const PREFILL_CHUNK = 256;   // == GEMM_BM: each chunk fills one M-block; engine submit-split keeps TDR safe
    const _tp0 = performance.now();
    const _savedPerf = _PERF; _PERF = true;
    let tok0, _gpuMs = 0, _encMs = 0;
    try {
      for (let off = P; off < L; off += PREFILL_CHUNK) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const chunk = ids.slice(off, Math.min(off + PREFILL_CHUNK, L));
        tok0 = await forward(chunk, off);                     // bounded submit (no TDR)
        if (_perfData) { _gpuMs += _perfData.gpu_drain_ms; _encMs += _perfData.encode_ms; }
      }
    } catch (e) {
      _cachedIds = null; _sysAnchor = null;   // KV prefix now partial/uncertain → force a full prefill next turn
      _PERF = _savedPerf;
      throw e;
    }
    // KV[0..L) now holds exactly `ids`. The decode loop below writes KV[L..) which
    // does NOT touch this prefix, so the next turn (which appends after the current
    // assistant turn) cleanly extends it.
    _cachedIds = ids.slice(0, L);
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
    const _newTok = L - _reused;
    console.log('[qwen3 perf] prompt=' + L + ' tok (cached ' + _reused + ', prefilled ' + _newTok + ') | prefill=' + ((_tp1 - _tp0) / 1000).toFixed(2) + 's (' + (_newTok / ((_tp1 - _tp0) / 1000)).toFixed(0) + ' tok/s) | decode=' + _n + ' tok in ' + (_dms / 1000).toFixed(2) + 's (' + (_n / (_dms / 1000)).toFixed(1) + ' tok/s) | maxTokens=' + maxTokens + _split);
    return prevText;
  }

  function toolPreamble(tools) {
    const fns = (tools || []).filter(t => t && t.type === 'function' && t.function);
    if (!fns.length) return '';
    // Qwen3's NATIVE (Hermes) tool block — the EXACT text its chat template renders and the
    // model was RLHF'd against. Small dense Qwen3 (0.6B/1.7B) follow the training distribution
    // closely and won't reliably emit <tool_call> for a hand-rolled format (they narrate the
    // intent in prose instead). Signatures are the full {"type":"function","function":{…}}
    // objects inside <tools></tools>. Our assistant <tool_call> / tool <tool_response>
    // renderings (norm/tcText) already match this template.
    const sigs = fns.map(t => JSON.stringify({ type: 'function', function: t.function })).join('\n');
    return [
      '# Tools',
      '',
      'You may call one or more functions to assist with the user query.',
      '',
      'You are provided with function signatures within <tools></tools> XML tags:',
      '<tools>',
      sigs,
      '</tools>',
      '',
      'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:',
      '<tool_call>',
      '{"name": <function-name>, "arguments": <args-json-object>}',
      '</tool_call>',
    ].join('\n');
  }
  const _cleanContent = (t) => (t || '').replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();

  // Streaming 3-state splitter for Qwen3 output: <think>…</think> → onReason (the live
  // Thinking box), <tool_call>…</tool_call> → buffered into toolCalls[] (machine payload,
  // NOT streamed to the user), everything else → onContent. Holds a 12-char guard tail so
  // a tag split across token pieces is still detected. (Ported from the Qwen3.5 fork.)
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

  // Tool executor — injected by the host/worker (which can reach ./sandpie-tool on the
  // main thread). null = no tools (e.g. the bench harness). (name, args, convId, signal) →
  // Promise<{ result, artifacts? }>.
  let _toolRunner = null;
  function setToolRunner(fn) { _toolRunner = (typeof fn === 'function') ? fn : null; }

  // ---- System-prompt cache (in-memory, zero-copy; see _sysAnchor) ----
  // The cacheable prefix is the system block ONLY (no generation prompt). `sys` already holds the
  // system prompt + tool descriptions + skill defs + skill guidance, so this single encode covers
  // all four — no need to know how they were assembled.
  function _sysPrefixIds(sys) {
    try { return sys ? TOK.encodeChat([{ role: 'system', content: sys }], { addGenerationPrompt: false }) : null; }
    catch (_) { return null; }
  }
  function _commonLen(a, b) { const m = Math.min(a.length, b.length); let i = 0; while (i < m && a[i] === b[i]) i++; return i; }
  function _startsWith(ids, pre, n) { if (ids.length < n) return false; for (let i = 0; i < n; i++) if (ids[i] !== pre[i]) return false; return true; }

  // BEFORE round-0 prefill: if our anchor proves _kv[0..P_sys) already holds exactly this system
  // block's KV, claim it (set _cachedIds) so _streamIds prefills only the tail — skipping the
  // system block even on a NEW conversation's first message (when _cachedIds is null/stale).
  // No-op if the live in-memory prefix already covers it (multi-turn handles that). Hit → true.
  function _sysCacheClaim(ids, sys, variant, emit) {
    const sysIds = _sysPrefixIds(sys);
    if (!sysIds || sysIds.length < 16) return false;
    const P = sysIds.length;
    if (!_startsWith(ids, sysIds, P)) return false;                          // ids must begin with the system block
    if (_cachedIds && _commonLen(_cachedIds, ids) >= P) return false;        // live KV already covers it
    if (!_kv || !_sysAnchor || _sysAnchor.variant !== variant) return false; // no usable anchor
    if (_sysAnchor.ids.length !== P || !_startsWith(_sysAnchor.ids, sysIds, P)) return false;  // anchor != this system block
    _cachedIds = sysIds;   // _streamIds: LCP(_cachedIds, ids) === P → reuse the resident system KV
    try { emit && emit({ type: 'info', message: 'Using cached system prompt…' }); } catch (_) {}
    return true;
  }
  // AFTER round-0 prefill: _kv[0..P_sys) now definitely holds this system block's KV (reused or
  // freshly prefilled), so record/refresh the anchor for the next conversation.
  function _sysCacheRecord(sys, variant) {
    const sysIds = _sysPrefixIds(sys);
    _sysAnchor = (sysIds && sysIds.length >= 16 && _kv) ? { variant, ids: sysIds } : null;
  }

  async function runConversation({ provider, messages, systemPrompt, tools, convId, signal }, emit) {
    // Generation budget per round. NO artificial thinking cap by default — a reasoning <think>
    // block can be long, and a fixed cap (512, then 2048) truncated it mid-thought. EOS (im_end)
    // stops finished turns; the only hard bound is the context window (the decode loop caps at
    // MAX_SEQ-1-pos). Default = MAX_SEQ so context is the sole limit. Override via
    // provider.maxTokens ("Max output tokens") if you want a tighter cap.
    const maxTokens = (provider && (provider.maxTokens | 0)) || MAX_SEQ;
    // conversations.js stores the dropdown's modelId in provider.endpoint (the
    // model picker sets spEndpoint = modelId). Accept either field.
    const _wantV = (provider && (provider.endpoint || provider.modelId)) || '';
    const variant = CONFIGS[_wantV] ? _wantV : '0.6B';
    try {
      emit({ type: 'info', message: 'Loading Qwen3-' + variant + ' dense (WebGPU)… first run downloads the weights.' });
      let lastPct = -1;
      await loadModel({ variant, nCtx: (provider && (provider.contextWindow | 0)) || 0, onProgress: (p) => {
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
    const toolList = (Array.isArray(tools) ? tools : []).filter(t => t && t.type === 'function');
    const pre = toolPreamble(toolList);
    if (pre) sys = sys ? (sys + '\n\n' + pre) : pre;
    // Render every message to PLAIN TEXT for the (text-only) encoder: assistant tool_calls
    // and tool results become Hermes <tool_call>/<tool_response> text so the model sees them
    // on re-prefill.
    const tcText = (tcs) => (tcs || []).map(tc => '<tool_call>\n{"name": "' + ((tc.function && tc.function.name) || '') + '", "arguments": ' + ((tc.function && tc.function.arguments) || '{}') + '}\n</tool_call>').join('\n');
    const norm = (m) => {
      let c = m.content;
      if (Array.isArray(c)) c = c.filter(p => p && p.type === 'text').map(p => p.text || '').join('\n');
      c = (c == null) ? '' : String(c);
      if (m.role === 'tool') return { role: 'user', content: '<tool_response>\n' + c + '\n</tool_response>' };
      if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) c = c ? (c + '\n' + tcText(m.tool_calls)) : tcText(m.tool_calls);
      return { role: m.role, content: c };
    };
    const work = [];
    if (sys) work.push({ role: 'system', content: sys });
    for (const m of (messages || [])) { if (m && m.role) work.push(norm(m)); }

    const MAX_ROUNDS = toolList.length ? 8 : 1;
    try {
      for (let round = 0; round < MAX_ROUNDS; round++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        emit({ type: 'round_start' });
        let firstTok = false;
        const clearInfo = () => { if (!firstTok) { firstTok = true; emit({ type: 'info', message: null }); } };
        const parser = makeRoundParser(
          (rz) => { clearInfo(); emit({ type: 'delta', delta: { reasoning: rz } }); },   // <think> → Thinking box
          (ct) => { clearInfo(); emit({ type: 'delta', delta: { content: ct } }); },     // answer → streamed content
        );
        const ids = TOK.encodeChat(work);
        if (round === 0) _sysCacheClaim(ids, sys, variant, emit);   // skip re-prefilling the system block if resident
        await _streamIds(ids, { maxTokens, signal, onToken: (p) => parser.push(p) });
        if (round === 0) _sysCacheRecord(sys, variant);             // _kv[0..P_sys) now holds this system block
        parser.flush();
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        emit({ type: 'info', message: null });

        const content = parser.content.replace(/^\s+/, '');
        const toolCalls = [];
        parser.toolCalls.forEach((raw, k) => {
          try { const o = JSON.parse(raw); if (o && o.name) toolCalls.push({ id: 'call_' + round + '_' + k, type: 'function', function: { name: o.name, arguments: JSON.stringify(o.arguments || {}) } }); } catch (_) {}
        });
        // Synthesize the tool_calls delta so conversations.js builds the call bubbles.
        if (toolCalls.length) emit({ type: 'delta', delta: { tool_calls: toolCalls.map((tc, i) => ({ index: i, id: tc.id, type: 'function', function: { name: tc.function.name, arguments: tc.function.arguments } })) } });
        emit({ type: 'round_end', content });
        const asst = { role: 'assistant', content };
        if (toolCalls.length) asst.tool_calls = toolCalls;
        emit({ type: 'message_added', message: asst });
        work.push(norm(asst));   // text-render the assistant turn (incl. tool_calls) for re-prefill
        if (!toolCalls.length) break;   // no tools → turn complete

        if (!_toolRunner) { emit({ type: 'info', message: null }); break; }   // can't run tools in this context
        for (const tc of toolCalls) {
          if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
          emit({ type: 'tool_started', tc });
          let args = {}; try { args = JSON.parse(tc.function.arguments || '{}'); } catch (_) {}
          let out;
          try { out = await _toolRunner(tc.function.name, args, convId, signal); }
          catch (e) { if (e && e.name === 'AbortError') throw e; out = { result: 'Error: ' + ((e && e.message) || e) }; }
          const toolResult = (out && out.result != null) ? out.result : '';
          emit({ type: 'tool_result', id: tc.id, result: toolResult, artifacts: out && out.artifacts });
          work.push(norm({ role: 'tool', tool_call_id: tc.id, content: toolResult }));
          emit({ type: 'message_added', message: { role: 'tool', tool_call_id: tc.id, content: toolResult } });
        }
      }
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
    runConversation, setToolRunner, DEFAULT_MODELS, DEFAULT_N_CTX, unload, _benchMatmul, _benchAttn, _benchGemv, _benchDP4, _benchGateUp, _benchGemmDP4, _benchGemmTS, _benchGemmTex, _benchGemmTex2, _benchGemmTex3, _benchAttnF16, attentionF16, _attnF16Wgsl: (KT) => attnF16Wgsl(KT || 8),
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
