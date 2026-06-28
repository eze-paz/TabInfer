// Web Worker host for the hand-written WebGPU Qwen3 engine.
//
// WHY: the engine's prefill/decode loop runs a tight queue.submit() / mapAsync()
// cycle whose only yield point is the per-batch readback. On the MAIN thread, the
// browser inserts a full compositing/rAF frame at that yield when the tab is
// focused, delaying the next submit and letting the iGPU clocks sag — measured
// ~3.4× slower decode focused vs backgrounded (2.9 vs 9.8 tok/s @ 3200 ctx).
// Running the loop in a dedicated worker puts the submits on a thread the page's
// rendering cannot interrupt → focused/backgrounded stop mattering and the GPU
// stays continuously fed. Needs NO COOP/COEP (only SharedArrayBuffer would).
//
// The engine modules assume a `window` global (they do `window.SandpieWebGPU` /
// `window.SandpieQwen3`); alias it to the worker global before importing them.
self.window = self;

let _ok = false;
try {
  // KEEP the ?v= in sync with sandpie.html when these modules are bumped.
  importScripts('webgpu-engine.js?v=47', 'webgpu-qwen3.js?v=116');
  _ok = !!self.SandpieQwen3;
} catch (e) {
  self.postMessage({ t: 'fatal', message: 'worker import failed: ' + ((e && e.message) || e) });
}

const Q = self.SandpieQwen3;
const _ctrls = new Map();   // run id -> AbortController

// Tool calls must run on the MAIN thread (./sandpie-tool is page-relative + the tool
// worker lives there). Proxy each call to the host and await its reply.
const _toolReqs = new Map();   // reqId -> { resolve, reject }
let _toolSeq = 0;
if (_ok && Q && Q.setToolRunner) {
  Q.setToolRunner((name, args, convId, signal) => new Promise((resolve, reject) => {
    const reqId = 'tool_' + (++_toolSeq);
    _toolReqs.set(reqId, { resolve, reject });
    const onAbort = () => { if (_toolReqs.delete(reqId)) reject(new DOMException('aborted', 'AbortError')); };
    if (signal) { if (signal.aborted) { onAbort(); return; } signal.addEventListener('abort', onAbort, { once: true }); }
    self.postMessage({ t: 'tool', reqId, name, args, convId });
  }));
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  const id = msg.id;
  if (msg.t === 'toolResult') {
    const r = _toolReqs.get(msg.reqId); if (r) { _toolReqs.delete(msg.reqId); r.resolve({ result: msg.result, artifacts: msg.artifacts }); }
    return;
  }
  if (msg.t === 'run') {
    if (!_ok || !Q || !Q.runConversation) {
      self.postMessage({ t: 'err', id, name: 'Error', message: 'WebGPU worker engine failed to load' });
      return;
    }
    const ctrl = new AbortController();
    _ctrls.set(id, ctrl);
    const cfg = Object.assign({}, msg.config, { signal: ctrl.signal });
    try {
      // runConversation emits its own round_start/delta/round_end/agent_done and
      // handles internal errors (emits {type:'error'}+agent_done); it only THROWS
      // on abort. So we forward every emit and signal 'done' when it resolves.
      await Q.runConversation(cfg, (ev) => { self.postMessage({ t: 'emit', id, ev }); });
      self.postMessage({ t: 'done', id });
    } catch (err) {
      self.postMessage({ t: 'err', id, name: (err && err.name) || 'Error', message: (err && err.message) || String(err) });
    } finally {
      _ctrls.delete(id);
    }
  } else if (msg.t === 'abort') {
    const c = _ctrls.get(id); if (c) { try { c.abort(); } catch (_) {} }
  } else if (msg.t === 'unload') {
    // Deep clean (backend switch): free all buffers AND destroy the device, then ACK so
    // the host can wait for the GPU to be freed before the other backend loads.
    try { if (Q && Q.unload) await Q.unload(true); } catch (_) {}
    self.postMessage({ t: 'unloadDone', ackId: msg.ackId });
  }
};
