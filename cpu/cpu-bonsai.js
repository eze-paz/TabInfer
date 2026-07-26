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

  // Default host for the big data files (bonsai17.cpu.bin + tokenizer.json): a public
  // HuggingFace repo. Overridable via localStorage['sandpie-cpu-bonsai-base']. On a
  // LOCAL dev checkout the files sit at the site root, so localhost prefers those
  // (no needless 1GB HF fetch); everywhere else defaults to HF. Fetched once, then
  // served from the OPFS cache.
  const HF_BASE = 'https://huggingface.co/eze-paz/bonsai17-cpu/resolve/main/';
  function resolveDataBase(cfg) {
    let b = (cfg && cfg.cpuBonsaiBase) || '';
    if (!b) {
      const host = (self.location && self.location.hostname) || '';
      const local = /^(localhost|127\.|0\.0\.0\.0|::1|\[::1\])/.test(host);
      if (local) return { bin: BASE + 'bonsai17.cpu.bin', tok: BASE + '_bonsai17/', label: 'site root (local dev)' };
      b = HF_BASE;
    }
    if (!b.endsWith('/')) b += '/';
    return { bin: b + 'bonsai17.cpu.bin', tok: b, label: b };
  }

  // ---- one-time model caching ----------------------------------------------
  // The big bin (~1GB) is cached in OPFS so it downloads ONCE. Cache lives OUTSIDE
  // sandpie/ so Dropbox never syncs it. Download STREAMS to disk (low peak RAM); a
  // sibling `.ok` marker guards against a partial/interrupted download being reused.
  // The coordinator's load() just does fetch(url).arrayBuffer(), so we hand it a
  // blob: URL backed by the on-disk (disk-backed, not RAM-resident) cache File.
  const _hashStr = (s) => { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0).toString(36); };
  async function _cacheDir() { try { const root = await navigator.storage.getDirectory(); return await root.getDirectoryHandle('cpu-model-cache', { create: true }); } catch (_) { return null; } }
  async function _binCached(url) {
    const dir = await _cacheDir(); if (!dir) return false;
    const key = 'bonsai_' + _hashStr(url) + '.bin';
    try { await dir.getFileHandle(key + '.ok'); const f = await (await dir.getFileHandle(key)).getFile(); return f.size > 0; } catch (_) { return false; }
  }
  async function _cachedBinUrl(url, onProgress) {
    const dir = await _cacheDir();
    const key = 'bonsai_' + _hashStr(url) + '.bin';
    if (dir) { try { await dir.getFileHandle(key + '.ok'); const f = await (await dir.getFileHandle(key)).getFile(); if (f.size > 0) return URL.createObjectURL(f); } catch (_) {} }
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('model download failed: HTTP ' + resp.status + ' from ' + url);
    if (!dir || !resp.body) { return URL.createObjectURL(new Blob([await resp.arrayBuffer()])); }   // no OPFS → in-memory
    try { await dir.removeEntry(key + '.ok'); } catch (_) {}                       // invalidate stale marker before rewriting
    const total = +(resp.headers.get('content-length') || 0);
    const w = await (await dir.getFileHandle(key, { create: true })).createWritable();
    const reader = resp.body.getReader(); let loaded = 0, lastPct = -1;
    for (;;) { const r = await reader.read(); if (r.done) break; await w.write(r.value); loaded += r.value.length; if (onProgress) { const p = total ? Math.floor(loaded / total * 100) : -1; if (p !== lastPct) { lastPct = p; onProgress(total ? loaded / total : 0, loaded, total); } } }
    await w.close();
    try { const mw = await (await dir.getFileHandle(key + '.ok', { create: true })).createWritable(); await mw.close(); } catch (_) {}
    return URL.createObjectURL(await (await dir.getFileHandle(key)).getFile());
  }

  function ensureLoaded(data, onProgress) {
    if (_loaded) return Promise.resolve();
    if (_loading) return _loading;
    _loading = (async () => {
      if (!self.crossOriginIsolated) throw new Error('CPU engine needs crossOriginIsolated (COOP/COEP) for SharedArrayBuffer');
      // PREFLIGHT: probe required files and name exactly what's missing — otherwise a
      // 404's HTML page surfaces later as `Unexpected token '<' … not valid JSON`. Skip
      // the big bin if it's already cached (works offline / survives HF being down).
      const binHit = await _binCached(data.bin);
      const need = [
        [BASE + '_cpukern/cpuengine-mt.js', 'cpukern JS'],
        [BASE + 'cpukern.wasm', 'cpukern.wasm'],
        [BASE + 'cpukern-shared.wasm', 'cpukern-shared.wasm'],
        ...(binHit ? [] : [[data.bin, 'bonsai17.cpu.bin']]),
        [data.tok + 'tokenizer.json', 'tokenizer.json'],
      ];
      const missing = [];
      for (const [url, name] of need) {
        // 1-byte ranged GET, aborted after headers: works across static servers AND
        // CDNs (HuggingFace, R2, S3) where HEAD is often unsupported. Downloads nothing.
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
      const binUrl = await _cachedBinUrl(data.bin, onProgress);   // one-time download → OPFS cache → blob URL
      try { await globalThis.CPUEngineMT.load(binUrl, 8); } finally { try { URL.revokeObjectURL(binUrl); } catch (_) {} }
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
      let _dlAnnounced = false;
      await ensureLoaded(resolveDataBase(config), (frac, loaded, total) => {
        if (!_dlAnnounced) { _dlAnnounced = true; emit({ type: 'delta', delta: { reasoning: 'First run on this device: downloading + caching the model (~1 GB) — one-time, then it loads from cache instantly.\n' } }); }
        try { console.log('[bonsai] caching ' + Math.round(frac * 100) + '% (' + (loaded / 1e9).toFixed(2) + '/' + (total / 1e9).toFixed(2) + ' GB)'); } catch (_) {}
      });
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
