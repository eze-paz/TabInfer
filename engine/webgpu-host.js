// Main-thread shim for window.SandpieQwen3, backed by a Web Worker that runs the
// actual hand-written WebGPU engine (webgpu-worker.js → webgpu-engine.js +
// webgpu-qwen3.js). The GPU prefill/decode loop runs off the main thread so page
// rendering can't starve the submit windows (fixes the focused-tab decode slowdown
// — see webgpu-worker.js). The interface mirrors the real engine for exactly the
// members the main thread uses: DEFAULT_MODELS, DEFAULT_N_CTX, runConversation,
// unload. conversations.js / providers.js call these unchanged.
(function () {
  'use strict';

  // Request PERSISTENT storage once on load. Without this, Cache Storage + OPFS are
  // "best-effort" and the browser evicts entries under pressure — which silently corrupts
  // the multi-hundred-MB quantized-weights and KV-prefix snapshots (a manifest can survive
  // while its data chunks are evicted → restore fails → permanent re-prefill). Granted
  // automatically by Chrome on engaged/installed origins; harmless if denied.
  try { navigator.storage && navigator.storage.persist && navigator.storage.persist().then(function (granted) { try { console.log('[sandpie] persistent storage: ' + (granted ? 'granted' : 'denied (caches may be evicted)')); } catch (_) {} }, function () {}); } catch (_) {}

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

  // Execute a tool the engine worker requested by routing it through the SHARED in-page
  // sandpie-worker.js (the SAME Web Worker the cloud agent loop and LiteRT use — it reads
  // OPFS directly and handles every tool incl. load_skill), then reply to the engine
  // worker. NOT via fetch('./sandpie-tool'): sandpie-worker is a dedicated Worker, not a
  // service worker, so nothing intercepts that path and it 404s against the server — which
  // is why ONLY the WebGPU backend's tools (incl. load_skill) were failing while cloud +
  // Gemma worked. Mirrors litertlm.js toolViaWorker.
  let _toolReqSeq = 0;
  const TOOL_TIMEOUT_MS = 5 * 60 * 1000;
  function runTool(m) {
    const reply = (result, artifacts) => {
      try { _worker && _worker.postMessage({ t: 'toolResult', reqId: m.reqId, result, artifacts: artifacts || null }); } catch (_) {}
    };
    const sw = window._sandpieWorker;
    if (!sw) { reply('Error: sandpie-worker not ready — reload the page.', null); return; }
    const id = 'wg-' + (++_toolReqSeq);
    let settled = false;
    const finish = (result, artifacts) => { if (settled) return; settled = true; cleanup(); reply(result, artifacts); };
    const onMsg = (e) => { const r = e.data || {}; if (r.id !== id || r.type !== 'tool_result') return; finish(r.result || '', r.artifacts || null); };
    const onErr = () => finish('Error: sandpie-worker crashed — retry (it restarts automatically).', null);
    const timer = setTimeout(() => finish('Error: tool timed out after ' + (TOOL_TIMEOUT_MS / 1000) + 's — the worker may be stuck.', null), TOOL_TIMEOUT_MS);
    function cleanup() { clearTimeout(timer); sw.removeEventListener('message', onMsg); sw.removeEventListener('error', onErr); }
    sw.addEventListener('message', onMsg);
    sw.addEventListener('error', onErr);
    sw.postMessage({ type: 'tool', id, name: m.name, args: m.args, conversation_file_name: m.convId, localterm: { token: localStorage.getItem('sandpie:localterm:token') || '', port: +(localStorage.getItem('sandpie:localterm:port')) || 8771 } });
  }

  function worker() {
    if (_worker) return _worker;
    _worker = new Worker('modules/webgpu-worker.js?v=22');
    _worker.onmessage = (e) => {
      const m = e.data || {};
      if (m.t === 'fatal') { console.error('[webgpu-host]', m.message); return; }
      if (m.t === 'tool') { runTool(m); return; }   // worker asked us to execute a tool on the main thread
      if (m.t === 'unloadDone') { const f = _unloadWaiters.get(m.ackId); if (f) f(); return; }
      const r = _runs.get(m.id);
      if (!r) return;
      if (m.t === 'emit') {
        // Capture the model's generated turns for window.SandpieLastResponse (debug).
        if (m.ev && m.ev.type === 'message_added' && r.resp) {
          r.resp.push(m.ev.message);
          try { if (localStorage.getItem('sandpie-llm-debug') === '1') console.log('[sandpie LLM message]', m.ev.message); } catch (_) {}
        }
        try { r.emit(m.ev); } catch (_) {}
      }
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
    // Debug: window.SandpieLastRequest = the exact request the local model sees (system
    // prompt + INPUT messages + tool defs). window.SandpieLastResponse = the assistant/tool
    // messages it GENERATES this turn, growing live — the engine runs the whole agent loop
    // inside the worker, so generated turns arrive as 'message_added' events (captured in the
    // onmessage handler below), NOT in cfg. Both logged when localStorage 'sandpie-llm-debug'==='1'.
    const respMsgs = [];
    try {
      window.SandpieLastRequest = cfg;
      window.SandpieLastResponse = respMsgs;
      if (localStorage.getItem('sandpie-llm-debug') === '1') console.log('[sandpie LLM request — local webgpu]', cfg);
    } catch (_) {}
    const p = new Promise((resolve, reject) => {
      _runs.set(id, { emit: emit || (function () {}), resolve, reject, resp: respMsgs });
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
