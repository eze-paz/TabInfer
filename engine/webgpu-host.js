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

  function worker() {
    if (_worker) return _worker;
    _worker = new Worker('modules/webgpu-worker.js?v=3');
    _worker.onmessage = (e) => {
      const m = e.data || {};
      if (m.t === 'fatal') { console.error('[webgpu-host]', m.message); return; }
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
  function runConversation(config, emit) {
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

  function unload() { try { if (_worker) _worker.postMessage({ t: 'unload' }); } catch (_) {} }

  window.SandpieQwen3 = { DEFAULT_MODELS, DEFAULT_N_CTX, runConversation, unload, _viaWorker: true };
})();
