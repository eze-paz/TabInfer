// sandpie/modules/webgpu-engine.js — a hand-written WebGPU LLM inference engine.
//
// WHY THIS EXISTS
// ---------------
// The webml-community/gemma-4-webgpu-kernels HF Space proved that a from-scratch
// WebGPU engine — raw WGSL compute kernels, NO TVM/MLC, NO onnxruntime, NO library
// — runs a modern LLM in ~550KB and was dramatically faster than the generic
// backends on commodity hardware (incl. a plain Intel iGPU). This module is our
// own such engine, built generic-first rather than tuned to one device.
//
// PLAN (see session design notes):
//   • Phase 1  — bring up Qwen3-0.6B (plain GQA + RoPE + QK-norm + SwiGLU +
//                RMSNorm, tied embeds, ~151K vocab). A standard decoder-only
//                transformer: the kernel set the reference engine's GEMM/RMSNorm/
//                RoPE/attention primitives already cover. This is the de-risking
//                target AND matches the tool-calling goal.
//   • Phase 2  — Qwen3.5-0.8B. NOT a plain transformer: 18/24 layers are gated
//                DeltaNet linear-attention (Mamba2-style recurrence + causal
//                conv1d), plus gated attention, partial (25%) mRoPE and a vision
//                tower. Needs novel from-scratch recurrence kernels — tackled only
//                once Phase 1 is proven against a known-good baseline.
//
// THIS FILE, RIGHT NOW (v0 — engine core + bench)
// -----------------------------------------------
// The generic, model-agnostic core only. No model graph yet. It provides:
//   1. GPU init + a full capability report (the decisive fact: does this device
//      expose subgroups / shader-f16 / the experimental subgroup-matrix path that
//      the reference engine's fast GEMM relies on?).
//   2. A tiny buffer/tensor helper, a Jinja-lite WGSL template renderer, and a
//      compute-pipeline cache — the plumbing every kernel will reuse.
//   3. A correct, portable tiled GEMM kernel + selfTest() (correctness vs a CPU
//      reference) and bench() (GFLOP/s) so we can MEASURE this device before
//      committing to the kernel set.
//
// The host-contract methods (runConversation/streamRound) are present but throw
// "not implemented yet" — this module deliberately is NOT wired into providers.js
// or the model picker until a model actually loads. It's a runnable engine core
// you can open via docs/webgpu-engine-test.html.

const SandpieWebGPU = (function () {
  'use strict';

  // ============================================================
  // Debug logging (mirrors the other backends' convention)
  // ============================================================
  const DEBUG_KEY = 'sandpie-webgpu-debug';
  function isDebug() { try { return localStorage.getItem(DEBUG_KEY) === '1'; } catch (_) { return false; } }
  function dbg(...a) { if (isDebug()) console.log('[webgpu-engine]', ...a); }

  // ============================================================
  // 1. GPU init + capability detection
  // ============================================================
  // The WebGPU optional-feature names we care about. `subgroups` + `shader-f16`
  // are shipping; `chromium-experimental-subgroup-matrix` is the flagged
  // cooperative-matrix path the reference engine's fastest GEMM (subgroupMatrix*
  // ops) needs — its presence is the single biggest perf signal for this device.
  const WANTED_FEATURES = [
    'shader-f16',
    'subgroups',
    'subgroups-f16',
    'chromium-experimental-subgroup-matrix',
    'timestamp-query',
  ];

  let _adapter = null, _device = null, _caps = null;
  let _deviceGen = 0;   // bumped on every successful (re)init → lets callers detect a device that was lost+recreated and rebuild stale GPU buffers
  let _lastLost = null;            // { reason, message, at } of the most recent device loss (diagnostics)
  const _lostListeners = [];       // callbacks fired on device loss (proactive recovery)

  // Initialise the GPU device, requesting every wanted feature the adapter
  // actually advertises. Returns a capability report (also cached on _caps).
  async function init() {
    if (_device && _caps) return _caps;
    if (!('gpu' in navigator)) throw new Error('WebGPU not available (navigator.gpu missing)');

    _adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!_adapter) throw new Error('No WebGPU adapter (requestAdapter returned null)');

    const available = new Set(_adapter.features ? [..._adapter.features] : []);
    const requiredFeatures = WANTED_FEATURES.filter(f => available.has(f));

    // Ask for the biggest limits the adapter allows — LLM weights are large and
    // GEMM tiles want generous workgroup/storage budgets. We clamp each request
    // to the adapter's max so requestDevice never rejects on an over-ask.
    // GPUSupportedLimits exposes its values as non-enumerable accessor props, so
    // Object.entries() returns nothing — read the keys we care about explicitly.
    const LIMIT_KEYS = [
      'maxStorageBufferBindingSize',
      'maxBufferSize',
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupSizeX',
      'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupsPerDimension',
      'maxStorageBuffersPerShaderStage',
    ];
    const al = _adapter.limits || {};
    const adapterLimits = {};
    for (const k of LIMIT_KEYS) { const v = al[k]; if (v != null) adapterLimits[k] = v; }
    const requiredLimits = { ...adapterLimits };

    _device = await _adapter.requestDevice({ requiredFeatures, requiredLimits });
    _deviceGen++;   // a new device → any GPU buffer built against a prior gen is now stale
    _device.lost.then((info) => {
      // Device loss (TDR / driver reset / OOM) is THE failure mode for WebGPU LLMs on weak
      // iGPUs. Record reason+message so we can diagnose the actual cause (reason 'destroyed'
      // = app called destroy(); 'unknown' = driver/TDR/OOM — the message usually says which),
      // drop our handles so the next call re-inits, and notify any registered listeners so the
      // model can rebuild proactively instead of erroring on the user's next message.
      _lastLost = { reason: (info && info.reason) || 'unknown', message: (info && info.message) || '', at: (typeof performance !== 'undefined' ? Math.round(performance.now()) : 0) };
      console.error('[webgpu-engine] DEVICE LOST:', _lastLost.reason, '—', _lastLost.message);
      _device = null; _caps = null; _pipelineCache.clear(); _bgCache.clear();
      for (const fn of _lostListeners) { try { fn(_lastLost); } catch (_) {} }
    });

    const info = (_adapter.info) || (await (_adapter.requestAdapterInfo ? _adapter.requestAdapterInfo() : Promise.resolve({})));
    _caps = {
      features: [...available].sort(),
      enabled: requiredFeatures.slice(),
      hasSubgroups: available.has('subgroups'),
      hasF16: available.has('shader-f16'),
      hasSubgroupMatrix: available.has('chromium-experimental-subgroup-matrix'),
      hasTimestamp: available.has('timestamp-query'),
      limits: adapterLimits,
      adapter: {
        vendor: info.vendor || '', architecture: info.architecture || '',
        device: info.device || '', description: info.description || '',
      },
    };
    dbg('init caps', _caps);
    return _caps;
  }

  function device() { if (!_device) throw new Error('engine not initialised — call init() first'); return _device; }
  function caps() { return _caps; }

  // ============================================================
  // 2a. Buffer / tensor helpers
  // ============================================================
  // Minimal GPU buffer helpers. Real tensor lifetime management (pooling, the
  // KV cache) comes with the model graph; for the core + bench these suffice.
  const _bufIds = new WeakMap(); let _bufCtr = 0;   // stable id per buffer (for the bind-group cache)
  function createBuffer(byteLength, usage, label) {
    const b = device().createBuffer({ size: Math.max(4, Math.ceil(byteLength / 4) * 4), usage, label });
    _bufIds.set(b, ++_bufCtr);
    return b;
  }
  function uploadF32(arr, usage, label) {
    const u = usage || (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const buf = createBuffer(arr.byteLength, u, label);
    device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return buf;
  }
  // Read a storage buffer back to the CPU (debug / correctness checks only — this
  // stalls the pipeline; never on the hot path).
  async function readF32(buf, floatCount) {
    const bytes = floatCount * 4;
    const staging = createBuffer(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'readback');
    const enc = device().createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    device().queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return out;
  }

  // ============================================================
  // 2b. WGSL template renderer (Jinja-lite)
  // ============================================================
  // The reference engine renders *.wgsl.jinja with {{ constant }} substitution and
  // {% if %} variant gating. We only need the two forms its kernels actually use:
  //   {{ NAME }}                      → params[NAME]
  //   {% if COND %}...{% endif %}     → included when params[COND] is truthy
  // Kept intentionally tiny; expand only when a real kernel needs more.
  function renderWGSL(template, params) {
    let s = String(template);
    // Blocks first (so substitutions inside surviving blocks still run).
    s = s.replace(/\{%\s*if\s+([A-Za-z0-9_]+)\s*%\}([\s\S]*?)\{%\s*endif\s*%\}/g,
      (_, cond, body) => (params && params[cond]) ? body : '');
    s = s.replace(/\{\{\s*([A-Za-z0-9_]+)\s*\}\}/g, (m, name) => {
      if (params && Object.prototype.hasOwnProperty.call(params, name)) return String(params[name]);
      throw new Error('renderWGSL: missing template param "' + name + '"');
    });
    return s;
  }

  // ============================================================
  // 2c. Compute pipeline cache
  // ============================================================
  const _pipelineCache = new Map();   // cacheKey -> GPUComputePipeline
  const _pipeLabels = new WeakMap();  // pipeline -> cacheKey (for the profiler)
  const _bgCache = new Map();         // (pipeline,buffers) key -> GPUBindGroup
  function getPipeline(cacheKey, wgsl, entryPoint) {
    let p = _pipelineCache.get(cacheKey);
    if (p) return p;
    const module = device().createShaderModule({ code: wgsl, label: cacheKey });
    p = device().createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: entryPoint || 'main' },
      label: cacheKey,
    });
    _pipelineCache.set(cacheKey, p);
    _pipeLabels.set(p, cacheKey);
    return p;
  }

  // ---- GPU timestamp profiler --------------------------------------------------
  // beginProfile() → each subsequent dispatch() writes begin/end GPU timestamps
  // around its compute pass into a query set, tagged by the pipeline's label.
  // endProfile() resolves them and returns [{label, us}] per dispatch — the exact
  // per-stage GPU time. Requires the 'timestamp-query' feature.
  let _prof = null;
  function profiling() { return !!_prof; }
  function beginProfile(capacity) {
    if (!_caps || !_caps.hasTimestamp) throw new Error('timestamp-query not available');
    const cap = capacity || 512;
    _prof = { qs: device().createQuerySet({ type: 'timestamp', count: cap * 2 }), cap, n: 0, labels: [] };
  }
  async function endProfile() {
    const p = _prof; _prof = null;
    if (!p || p.n === 0) { if (p) p.qs.destroy(); return []; }
    const bytes = p.n * 2 * 8;   // 2 timestamps/dispatch, u64 each
    const resolve = createBuffer(bytes, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, 'qresolve');
    const read = createBuffer(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'qread');
    const enc = device().createCommandEncoder();
    enc.resolveQuerySet(p.qs, 0, p.n * 2, resolve, 0);
    enc.copyBufferToBuffer(resolve, 0, read, 0, bytes);
    device().queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const ts = new BigInt64Array(read.getMappedRange().slice(0));
    const out = [];
    for (let i = 0; i < p.n; i++) out.push({ label: p.labels[i], us: Number(ts[i * 2 + 1] - ts[i * 2]) / 1000 });
    read.unmap(); read.destroy(); resolve.destroy(); p.qs.destroy();
    return out;
  }

  // Dispatch a single compute pass over an ordered list of storage buffers bound
  // at @group(0) @binding(0..n) in declaration order. Returns when submitted (not
  // when finished) unless `await: true`.
  function dispatch(pipeline, buffers, workgroups, opts) {
    // Cache bind groups by (pipeline, buffer identities). The forward issues the
    // SAME ~250 (pipeline, buffer-set) combos every token (persistent scratch /
    // weights / pooled uniforms), so this turns ~250 createBindGroup/token into
    // ~0 after the first token — the bulk of the measured CPU-encode cost.
    let key = _pipeLabels.get(pipeline) || '';
    for (let i = 0; i < buffers.length; i++) key += '|' + (_bufIds.get(buffers[i]) || 0);
    let bindGroup = _bgCache.get(key);
    if (!bindGroup) {
      const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
      bindGroup = device().createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      if (_bgCache.size > 4096) _bgCache.clear();   // bound it (transient test buffers etc.)
      _bgCache.set(key, bindGroup);
    }
    // Batch mode: record into the shared encoder (one submit per batch) — kills
    // the per-dispatch submit bubbles. Each dispatch still gets its own compute
    // pass, so pass-boundary barriers preserve read-after-write ordering.
    const enc = _batchEncoder || device().createCommandEncoder();
    // When profiling, time this pass with begin/end GPU timestamps.
    let passDesc;
    if (_prof && _prof.n < _prof.cap) {
      const qi = _prof.n++;
      _prof.labels.push(_pipeLabels.get(pipeline) || 'op');
      passDesc = { timestampWrites: { querySet: _prof.qs, beginningOfPassWriteIndex: qi * 2, endOfPassWriteIndex: qi * 2 + 1 } };
    }
    const pass = enc.beginComputePass(passDesc);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0] || 1, workgroups[1] || 1, workgroups[2] || 1);
    pass.end();
    if (_batchEncoder) return Promise.resolve();        // submitted later by endBatch()
    device().queue.submit([enc.finish()]);
    return (opts && opts.await) ? device().queue.onSubmittedWorkDone() : Promise.resolve();
  }

  // ---- Batch: record a whole forward into ONE command buffer, submit once -----
  // beginBatch() → subsequent dispatch()/copyBuffer() append to one encoder;
  // endBatch() submits it and returns the completion promise. ~364 submits/token
  // → 1, eliminating inter-submit GPU idle bubbles (the measured decode wall).
  let _batchEncoder = null;
  function beginBatch() { _batchEncoder = device().createCommandEncoder({ label: 'forward' }); }
  function endBatch() {
    if (!_batchEncoder) return Promise.resolve();
    const enc = _batchEncoder; _batchEncoder = null;
    device().queue.submit([enc.finish()]);
    return device().queue.onSubmittedWorkDone();
  }
  // Buffer copy that respects batch mode.
  function copyBuffer(src, srcByteOff, dst, dstByteOff, bytes) {
    if (_batchEncoder) { _batchEncoder.copyBufferToBuffer(src, srcByteOff, dst, dstByteOff, bytes); return; }
    const enc = device().createCommandEncoder();
    enc.copyBufferToBuffer(src, srcByteOff, dst, dstByteOff, bytes);
    device().queue.submit([enc.finish()]);
  }

  // ============================================================
  // 3. Portable tiled GEMM kernel  C[M,N] = A[M,K] · B[K,N]   (f32)
  // ============================================================
  // The single most important kernel — every projection and the MLP is a GEMM.
  // This is the PORTABLE baseline: a classic 16×16 shared-memory tiled matmul
  // that runs on any WebGPU device with zero optional features. The subgroup /
  // subgroup-matrix fast paths (the source of the reference engine's speed) come
  // next as drop-in variants behind the capability flags from init(); having a
  // correct portable kernel first gives us a correctness oracle to validate them.
  const TILE = 16;
  const GEMM_WGSL = `
struct Dims { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>        A    : array<f32>;
@group(0) @binding(1) var<storage, read>        B    : array<f32>;
@group(0) @binding(2) var<storage, read_write>  C    : array<f32>;
@group(0) @binding(3) var<uniform>              dims : Dims;

var<workgroup> tileA : array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> tileB : array<array<f32, ${TILE}>, ${TILE}>;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  let M = dims.M; let N = dims.N; let K = dims.K;
  var acc : f32 = 0.0;
  let nTiles = (K + ${TILE}u - 1u) / ${TILE}u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let aCol = t * ${TILE}u + lid.x;
    let bRow = t * ${TILE}u + lid.y;
    tileA[lid.y][lid.x] = select(0.0, A[row * K + aCol], (row < M) && (aCol < K));
    tileB[lid.y][lid.x] = select(0.0, B[bRow * N + col], (bRow < K) && (col < N));
    workgroupBarrier();
    for (var k : u32 = 0u; k < ${TILE}u; k = k + 1u) {
      acc = acc + tileA[lid.y][k] * tileB[k][lid.x];
    }
    workgroupBarrier();
  }
  if ((row < M) && (col < N)) { C[row * N + col] = acc; }
}`;

  // Run one GEMM on GPU from CPU Float32Arrays; returns the result Float32Array.
  // (Allocates+frees buffers per call — fine for the bench, not the hot path.)
  async function gemm(a, b, M, N, K) {
    const pipeline = getPipeline('gemm-portable-f32', GEMM_WGSL, 'main');
    const bufA = uploadF32(a, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'A');
    const bufB = uploadF32(b, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'B');
    const bufC = createBuffer(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'C');
    const dims = new Uint32Array([M, N, K, 0]);
    const bufD = createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'dims');
    device().queue.writeBuffer(bufD, 0, dims);
    const wg = [Math.ceil(N / TILE), Math.ceil(M / TILE), 1];
    await dispatch(pipeline, [bufA, bufB, bufC, bufD], wg, { await: true });
    const out = await readF32(bufC, M * N);
    bufA.destroy(); bufB.destroy(); bufC.destroy(); bufD.destroy();
    return out;
  }

  // ============================================================
  // selfTest + bench — the point of v0: measure THIS device.
  // ============================================================
  // Correctness: a small GEMM vs a CPU reference, asserting max abs error.
  async function selfTest() {
    await init();
    const M = 64, N = 48, K = 80;
    const a = new Float32Array(M * K), b = new Float32Array(K * N);
    for (let i = 0; i < a.length; i++) a[i] = Math.sin(i * 0.1);
    for (let i = 0; i < b.length; i++) b[i] = Math.cos(i * 0.07);
    const got = await gemm(a, b, M, N, K);
    let maxErr = 0;
    for (let r = 0; r < M; r++) for (let c = 0; c < N; c++) {
      let acc = 0; for (let k = 0; k < K; k++) acc += a[r * K + k] * b[k * N + c];
      maxErr = Math.max(maxErr, Math.abs(acc - got[r * N + c]));
    }
    const ok = maxErr < 1e-3;
    dbg('selfTest maxErr', maxErr, ok ? 'PASS' : 'FAIL');
    return { ok, maxErr };
  }

  // Throughput: square GEMMs at increasing sizes, GPU-timed via onSubmittedWorkDone.
  // Returns rows of { size, ms, gflops } — the number that tells us whether this
  // device is worth building a kernel set for.
  async function bench(sizes) {
    await init();
    sizes = sizes || [256, 512, 1024];
    const pipeline = getPipeline('gemm-portable-f32', GEMM_WGSL, 'main');
    const rows = [];
    for (const S of sizes) {
      const a = new Float32Array(S * S), b = new Float32Array(S * S);
      for (let i = 0; i < a.length; i++) { a[i] = (i % 13) * 0.01; b[i] = (i % 7) * 0.02; }
      const bufA = uploadF32(a, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'A');
      const bufB = uploadF32(b, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'B');
      const bufC = createBuffer(S * S * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'C');
      const bufD = createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'dims');
      device().queue.writeBuffer(bufD, 0, new Uint32Array([S, S, S, 0]));
      const wg = [Math.ceil(S / TILE), Math.ceil(S / TILE), 1];
      await dispatch(pipeline, [bufA, bufB, bufC, bufD], wg, { await: true }); // warm (shader compile)
      const ITERS = 10;
      const t0 = performance.now();
      for (let i = 0; i < ITERS; i++) await dispatch(pipeline, [bufA, bufB, bufC, bufD], wg);
      await device().queue.onSubmittedWorkDone();
      const ms = (performance.now() - t0) / ITERS;
      const gflops = (2 * S * S * S) / (ms / 1000) / 1e9;
      rows.push({ size: S, ms: +ms.toFixed(2), gflops: +gflops.toFixed(1) });
      bufA.destroy(); bufB.destroy(); bufC.destroy(); bufD.destroy();
    }
    dbg('bench', rows);
    return rows;
  }

  // ============================================================
  // Host contract — present so this conforms to the local-backend shape, but the
  // model graph isn't built yet. NOT wired into providers.js until a model loads.
  // ============================================================
  const DEFAULT_MODELS = [
    // Phase 1 target. modelId resolves to an HF repo of safetensors weights once
    // the loader + Qwen3 graph land. Listed here for reference; the picker does
    // not surface this backend yet.
    { id: 'qwen3-0.6b', label: 'Qwen3 0.6B (WebGPU engine — in development)', modelId: 'Qwen/Qwen3-0.6B' },
  ];
  const DEFAULT_N_CTX = 8192;

  async function unload() {
    _pipelineCache.clear();
    if (_device) { try { _device.destroy(); } catch (_) {} }
    _device = null; _adapter = null; _caps = null;
  }
  async function streamRound() { throw new Error('webgpu-engine: model graph not implemented yet (engine core only)'); }
  async function runConversation(_ctx, emit) {
    emit && emit({ type: 'error', message: 'webgpu-engine: model graph not implemented yet — this is the engine core + bench only.' });
    emit && emit({ type: 'agent_done' });
  }

  return {
    // engine core
    init, device, caps, deviceGen: () => _deviceGen,
    lastLoss: () => _lastLost, onLost: (fn) => { if (typeof fn === 'function') _lostListeners.push(fn); },
    renderWGSL, getPipeline, dispatch, beginProfile, endProfile, profiling, beginBatch, endBatch, copyBuffer,
    createBuffer, uploadF32, readF32, gemm,
    // measurement
    selfTest, bench,
    // host contract (stubs for now)
    DEFAULT_MODELS, DEFAULT_N_CTX, unload, streamRound, runConversation,
  };
})();

if (typeof window !== 'undefined') window.SandpieWebGPU = SandpieWebGPU;
