// Main-thread shim for window.SandpieQwen3, backed by a Web Worker that runs the
// actual hand-written WebGPU engine (webgpu-worker.js → webgpu-engine.js +
// webgpu-qwen3.js). The GPU prefill/decode loop runs off the main thread so page
// rendering can't starve the submit windows (fixes the focused-tab decode slowdown
// — see webgpu-worker.js). The interface mirrors the real engine for exactly the
// members the main thread uses: DEFAULT_MODELS, DEFAULT_N_CTX, runConversation,
// unload. conversations.js / providers.js call these unchanged.
(function () {
  'use strict';

  // KEEP IN SYNC with DEFAULT_MODELS in webgpu-qwen3.js (the worker's real engine).
  const DEFAULT_MODELS = [
    { id: 'qwen3-0.6b', modelId: '0.6B', label: 'Qwen3-0.6B dense (~1.1GB download)' },
    { id: 'qwen3-1.7b', modelId: '1.7B', label: 'Qwen3-1.7B dense (~3.9GB download)' },
  ];
  const DEFAULT_N_CTX = 4096;

  let _worker = null, _seq = 0;
  const _runs = new Map();   // id -> { emit, resolve, reject }
  const _unloadWaiters = new Map();   // ackId -> resolve (deep-unload acknowledgements)

  // While local inference is running, add body.sp-decoding so CSS can kill the
  // GPU-compositor work that competes with the WebGPU decode loop for the shared
  // GPU — chiefly backdrop-filter:blur() (recomputes EVERY frame the streamed text
  // changes) and infinite CSS animations. This was the focused-tab decode collapse
  // (GPU ~20% focused / 100% backgrounded): the decode loop itself is fine (matches
  // the reference engine's depth-N per-token pipeline), but foreground compositing
  // of expensive blurs over the changing message area starves the GPU when visible.
  let _busy = 0;
  function setBusy(on) {
    try {
      const b = document.body; if (!b) return;
      if (on) { if (_busy++ === 0) b.classList.add('sp-decoding'); }
      else { _busy = Math.max(0, _busy - 1); if (_busy === 0) b.classList.remove('sp-decoding'); }
    } catch (_) {}
  }
  const _busyOff = () => setBusy(false);

  // Execute a tool the worker requested, on the MAIN thread (./sandpie-tool is page-relative
  // and the tool worker lives here), then reply to the worker. Mirrors the other backends.
  async function runTool(m) {
    let result = '', artifacts = null;
    try {
      const res = await fetch('./sandpie-tool', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: m.name, args: m.args, conversation_file_name: m.convId }),
      });
      const out = res.ok ? await res.json() : { result: 'Error: tool endpoint ' + res.status };
      result = (out && out.result != null) ? out.result : '';
      artifacts = (out && out.artifacts) || null;
    } catch (e) { result = 'Error: ' + ((e && e.message) || e); }
    try { _worker && _worker.postMessage({ t: 'toolResult', reqId: m.reqId, result, artifacts }); } catch (_) {}
  }

  function worker() {
    if (_worker) return _worker;
    _worker = new Worker('modules/webgpu-worker.js?v=7');
    _worker.onmessage = (e) => {
      const m = e.data || {};
      if (m.t === 'fatal') { console.error('[webgpu-host]', m.message); return; }
      if (m.t === 'tool') { runTool(m); return; }   // worker asked us to execute a tool on the main thread
      if (m.t === 'unloadDone') { const f = _unloadWaiters.get(m.ackId); if (f) f(); return; }
      const r = _runs.get(m.id);
      if (!r) return;
      if (m.t === 'emit') { try { r.emit(m.ev); } catch (_) {} }
      else if (m.t === 'done') { _runs.delete(m.id); r.resolve(); }
      else if (m.t === 'err') {
        _runs.delete(m.id);
        const err = (m.name === 'AbortError')
          ? new DOMException('aborted', 'AbortError')
          : Object.assign(new Error(m.message || 'WebGPU worker error'), { name: m.name || 'Error' });
        r.reject(err);
      }
    };
    _worker.onerror = (ev) => { console.error('[webgpu-host] worker error', (ev && ev.message) || ev); };
    return _worker;
  }

  // Same signature/semantics as the real engine's runConversation: resolves after
  // it finishes (agent_done already emitted), rejects with AbortError on abort.
  async function runConversation(config, emit) {
    // Single active local backend: deep-free the OTHER local LLMs' GPU (incl. their
    // WebGPU devices) and WAIT for it BEFORE loading/running Qwen3, so two backends never
    // hold GPU at once (the Gemma↔Qwen overload). Each is a cheap no-op if not loaded.
    try { await window.SandpieLiteRTLM?.unload?.(); } catch (_) {}
    try { await window.SandpieWllama?.unload?.(); } catch (_) {}
    try { await window.SandpieTransformersJS?.unload?.(); } catch (_) {}
    const w = worker();
    const id = ++_seq;
    const signal = config && config.signal;
    const cfg = {                       // strip the non-serializable AbortSignal
      provider: config.provider,
      messages: config.messages,
      systemPrompt: config.systemPrompt,
      tools: config.tools,
      convId: config.convId,
    };
    const p = new Promise((resolve, reject) => {
      _runs.set(id, { emit: emit || (function () {}), resolve, reject });
      if (signal) {
        signal.addEventListener('abort', () => { try { w.postMessage({ t: 'abort', id }); } catch (_) {} }, { once: true });
      }
      w.postMessage({ t: 'run', id, config: cfg });
      if (signal && signal.aborted) { try { w.postMessage({ t: 'abort', id }); } catch (_) {} }
    });
    setBusy(true);              // suppress GPU-compositor competition (backdrop-filter/anim) during inference
    p.then(_busyOff, _busyOff);
    return p;
  }

  // Deep-free the worker's GPU (buffers + device) and RESOLVE only once the worker acks,
  // so callers (LiteRT / applyActiveProvider) can await the GPU actually being freed before
  // they load. No-op (resolved) if the worker was never created. 4s timeout so it can't hang.
  function unload() {
    const w = _worker;
    if (!w) return Promise.resolve();
    return new Promise((resolve) => {
      const ackId = ++_seq;
      let done = false;
      const finish = () => { if (done) return; done = true; _unloadWaiters.delete(ackId); resolve(); };
      _unloadWaiters.set(ackId, finish);
      setTimeout(finish, 4000);
      try { w.postMessage({ t: 'unload', ackId }); } catch (_) { finish(); }
    });
  }

  window.SandpieQwen3 = { DEFAULT_MODELS, DEFAULT_N_CTX, runConversation, unload, _viaWorker: true };
})();
