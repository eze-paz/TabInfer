// Pure-CPU Bonsai-1.7B engine adapter — runs INSIDE webgpu-worker.js alongside the
// WebGPU engines. Wraps the chunked shared-weights CPU coordinator (_cpukern/
// cpuengine-mt.js, __chunkMode v2: work-stolen gemv + shared-KV attention, ~35 tok/s
// clean-box) behind the same runConversation(cfg, emit) contract the worker routes to.
//
// EXPERIMENTAL / LOCAL-ONLY: needs bonsai17.cpu.bin + cpukern*.wasm + _bonsai17/
// (tokenizer) served from the site root, and crossOriginIsolated (SharedArrayBuffer).
// Tools are not supported (emitted text only). Context capped at MAXCTX (2048).
(function () {
  'use strict';

  const MODELS = { 'bonsai-1.7b-cpu': true };
  const EOS = new Set([151645, 151643]);
  const MAX_NEW = 1024, CTX_CAP = 2000;

  // engine + tokenizer live at the site root; this module runs from modules/
  const BASE = '../';
  let _loaded = false, _loading = null;

  function ensureLoaded() {
    if (_loaded) return Promise.resolve();
    if (_loading) return _loading;
    _loading = (async () => {
      if (!self.crossOriginIsolated) throw new Error('CPU engine needs crossOriginIsolated (COOP/COEP) for SharedArrayBuffer');
      // PREFLIGHT: this model needs gitignored LOCAL files served at the site root. Probe them
      // up front and name exactly what's missing — otherwise a 404's HTML error page surfaces
      // later as a baffling `Unexpected token '<' … not valid JSON` from the tokenizer parse.
      const need = ['_cpukern/cpuengine-mt.js', 'cpukern.wasm', 'cpukern-shared.wasm', 'bonsai17.cpu.bin', '_bonsai17/tokenizer.json'];
      const missing = [];
      for (const f of need) {
        try {
          const r = await fetch(BASE + f, { method: 'HEAD' });
          const ct = (r.headers.get('content-type') || '').toLowerCase();
          if (!r.ok || ct.includes('text/html')) missing.push(f);
        } catch (_) { missing.push(f); }
      }
      if (missing.length) throw new Error('required local files not served: ' + missing.join(', ') + ' — this EXPERIMENTAL model only runs on a dev machine where bonsai17.cpu.bin, cpukern*.wasm and _bonsai17/ sit at the site root (they are gitignored and not deployed)');
      globalThis.__cpukernBase = BASE;
      importScripts(BASE + '_cpukern/cpuengine-mt.js?v=41');
      // the REAL tokenizer module is already imported by webgpu-worker (webgpu-qwen3.js)
      await self.SandpieQwen3Engine_TOK_load();
      globalThis.__chunkMode = true; globalThis.__lutMode = false;
      await globalThis.CPUEngineMT.load(BASE + 'bonsai17.cpu.bin', 8);
      _loaded = true;
    })();
    _loading.catch(() => { _loading = null; });
    return _loading;
  }

  // tokenizer access: webgpu-qwen3.js exposes window.SandpieQwen3.TOK in this worker
  // (self.window aliased). Load its files from the local _bonsai17/ dir once.
  let _tokLoaded = false;
  self.SandpieQwen3Engine_TOK_load = async function () {
    if (_tokLoaded) return;
    await self.SandpieQwen3.TOK.load(BASE + '_bonsai17/');
    _tokLoaded = true;
  };

  const yield_ = () => new Promise((r) => setTimeout(r, 0));

  async function runConversation(config, emit) {
    const signal = config && config.signal;
    try {
      emit({ type: 'round_start' });
      await ensureLoaded();
      const TOK = self.SandpieQwen3.TOK, E = globalThis.CPUEngineMT;
      // flatten messages (strings only; tools unsupported)
      const msgs = [];
      if (config.systemPrompt) msgs.push({ role: 'system', content: String(config.systemPrompt) });
      for (const m of (config.messages || [])) {
        if (!m || !m.role) continue;
        const c = typeof m.content === 'string' ? m.content : Array.isArray(m.content) ? m.content.map(p => (p && p.text) || '').join('') : '';
        if (c) msgs.push({ role: m.role, content: c });
      }
      const ids = TOK.encodeChat(msgs);
      if (ids.length > CTX_CAP) {
        emit({ type: 'error', message: 'Bonsai CPU: prompt too long (' + ids.length + ' > ' + CTX_CAP + ' tokens — context is capped in this experimental build)' });
        emit({ type: 'agent_done' });
        return;
      }
      // prefill (sequential single-token forward; ~30ms/tok)
      let pos = 0, last = 0;
      for (let i = 0; i < ids.length; i++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const t = E.forwardTok(ids[i], pos++);
        if (i === ids.length - 1) last = t;
        if ((i & 7) === 7) await yield_();
      }
      // decode + stream, splitting <think>…</think> into delta.reasoning
      const out = [];
      let sent = 0, inThink = false, thinkDone = false, started = false;
      const pump = () => {
        // decode-full-and-diff (BPE-safe); stream from the `sent` boundary, routing the
        // leading <think>…</think> span to delta.reasoning like the WebGPU engine does.
        const s = TOK.decode(out);
        if (!started) {
          if (s.startsWith('<think>')) { inThink = true; started = true; }
          else if (s.length >= 7) started = true;
          else return; // not enough text yet to classify
        }
        if (inThink && !thinkDone) {
          const end = s.indexOf('</think>');
          if (end < 0) {
            const chunk = s.slice(Math.max(sent, 7));
            if (chunk) { emit({ type: 'delta', delta: { reasoning: chunk } }); sent = s.length; }
            return;
          }
          const rz = s.slice(Math.max(sent, 7), end);
          if (rz) emit({ type: 'delta', delta: { reasoning: rz } });
          thinkDone = true; sent = end + 8;
        }
        const chunk = s.slice(sent);
        if (chunk) { emit({ type: 'delta', delta: { content: chunk } }); sent = s.length; }
      };
      for (let t = 0; t < MAX_NEW; t++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        if (EOS.has(last) || pos >= CTX_CAP + 40) break;
        out.push(last);
        last = E.forwardTok(last, pos++);
        if ((t & 3) === 3) { pump(); await yield_(); }
      }
      pump();
      const full = TOK.decode(out);
      const ti = full.indexOf('</think>');
      const content = ti >= 0 ? full.slice(ti + 8) : (full.startsWith('<think>') ? '' : full);
      emit({ type: 'round_end', content });
      emit({ type: 'agent_done' });
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      emit({ type: 'error', message: 'Bonsai CPU: ' + ((err && err.message) || err) });
      emit({ type: 'agent_done' });
    }
  }

  function unload() {
    try { if (_loaded && globalThis.CPUEngineMT) globalThis.CPUEngineMT.stop(); } catch (_) {}
    _loaded = false; _loading = null;
    return Promise.resolve();
  }

  self.SandpieCpuBonsai = { MODELS, runConversation, unload, setToolRunner: function () {} };
})();
