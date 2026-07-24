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

  // Code artifacts (coordinator JS, fwd-worker JS, both wasm kernels) are COMMITTED and must
  // be same-origin (Worker spawn forbids cross-origin), so they always load from the site
  // root. Only the two BIG data files — bonsai17.cpu.bin + tokenizer.json — may live on a
  // separate host: set localStorage['sandpie-cpu-bonsai-base'] to a dir URL serving both
  // (CORS-enabled if cross-origin); default is the site root (dev machine layout).
  const BASE = '../';   // site root relative to modules/
  let _loaded = false, _loading = null;

  function resolveDataBase(cfg) {
    let b = (cfg && cfg.cpuBonsaiBase) || '';
    if (!b) return { bin: BASE + 'bonsai17.cpu.bin', tok: BASE + '_bonsai17/', label: 'site root' };
    if (!b.endsWith('/')) b += '/';
    return { bin: b + 'bonsai17.cpu.bin', tok: b, label: b };
  }

  function ensureLoaded(data) {
    if (_loaded) return Promise.resolve();
    if (_loading) return _loading;
    _loading = (async () => {
      if (!self.crossOriginIsolated) throw new Error('CPU engine needs crossOriginIsolated (COOP/COEP) for SharedArrayBuffer');
      // PREFLIGHT: probe every required file and name exactly what's missing — otherwise a
      // 404's HTML page surfaces later as `Unexpected token '<' … not valid JSON`.
      const need = [
        [BASE + '_cpukern/cpuengine-mt.js', 'cpukern JS'],
        [BASE + 'cpukern.wasm', 'cpukern.wasm'],
        [BASE + 'cpukern-shared.wasm', 'cpukern-shared.wasm'],
        [data.bin, 'bonsai17.cpu.bin'],
        [data.tok + 'tokenizer.json', 'tokenizer.json'],
      ];
      const missing = [];
      for (const [url, name] of need) {
        // 1-byte ranged GET, aborted after headers: works across static servers AND
        // CDNs (HuggingFace, R2, S3) where HEAD is often unsupported or redirects
        // oddly. Downloads nothing (we abort before reading the body).
        const ctrl = new AbortController();
        try {
          const r = await fetch(url, { headers: { Range: 'bytes=0-0' }, signal: ctrl.signal });
          const ct = (r.headers.get('content-type') || '').toLowerCase();
          if (!r.ok || ct.includes('text/html')) missing.push(name + ' (' + url + ')');
        } catch (_) { missing.push(name + ' (' + url + ')'); }
        finally { try { ctrl.abort(); } catch (_) {} }
      }
      if (missing.length) throw new Error('required files not served: ' + missing.join(', ')
        + ' — EXPERIMENTAL model. The wasm kernels deploy with the site (redeploy if missing). Host the two big files '
        + '(bonsai17.cpu.bin + tokenizer.json) on any CORS+range host — e.g. a HuggingFace repo (resolve/main/) or R2/S3 — '
        + 'and set localStorage["sandpie-cpu-bonsai-base"] to that directory URL. On a local dev checkout they load from the site root automatically.');
      globalThis.__cpukernBase = BASE;
      importScripts(BASE + '_cpukern/cpuengine-mt.js?v=41');
      await self.SandpieQwen3.TOK.load(data.tok);
      globalThis.__chunkMode = true; globalThis.__lutMode = false;
      await globalThis.CPUEngineMT.load(data.bin, 8);
      _loaded = true;
    })();
    _loading.catch(() => { _loading = null; });
    return _loading;
  }

  const yield_ = () => new Promise((r) => setTimeout(r, 0));

  async function runConversation(config, emit) {
    const signal = config && config.signal;
    try {
      emit({ type: 'round_start' });
      await ensureLoaded(resolveDataBase(config));
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
