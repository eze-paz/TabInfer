// sandpie/modules/webgpu-grammar.js — grammar-constrained decoding for the WebGPU engine (P4).
//
// WHAT: a token-level constraint engine for Hermes tool calls. Once the model emits
// `<tool_call>`, every subsequent token is restricted so the span is GUARANTEED to be
//   {"name": "<one of the registered tools>", "arguments": { ...valid JSON... }}\n</tool_call>
// Malformed JSON and hallucinated tool names become impossible BY CONSTRUCTION instead of
// detected+retried after the fact — the biggest reliability lever for 0.6-2B local models,
// whose tool-calling failures are overwhelmingly FORMAT failures, not reasoning failures.
//
// HOW: a character-level automaton (frame literals → tool-name matching → a full JSON
// value PDA) plus a one-time vocab scan precomputing token classes:
//   • pureStr — tokens safe ANYWHERE inside a JSON string (no " \ control chars, no
//     partial-UTF-8 replacement chars) → a fixed bitmask, the bulk base of in-string steps
//   • quote/slash/struct/ident candidate lists — small sets SIMULATED char-by-char per
//     step (a token is allowed iff feeding its chars keeps the automaton alive)
// Each step yields {forced: text} (frame text with exactly one continuation — the engine
// injects its tokens without computing logits at all), or {maskWords} (a vocab bitmask the
// engine applies to the logits right before argmax — one tiny extra dispatch), or {done}.
//
// PDA stack discipline (the invariant that keeps nesting simple): whenever a value
// expectation ('V') is created, its FOLLOW-UP marker is already beneath it on the stack.
// Completing any value therefore just pops its own markers — whatever is then on top is
// the correct continuation, and an EMPTY stack means the arguments object closed.
//
// Pure JS, no GPU, no engine imports — loaded by webgpu-worker.js and the test page.

const SandpieGrammar = (function () {
  'use strict';

  const STR_MAX = 4096;      // max chars in one JSON string (near cap, only closers survive)
  const DEPTH_MAX = 10;      // max PDA stack depth (≈ object/array nesting inside arguments)
  const SPAN_MAX = 16384;    // max chars for the whole constrained span (hard kill switch)
  const END_TEXT = '\n</tool_call>';
  const WS = ' \t\r\n';
  // frame literals, matched char-exact with optional whitespace BETWEEN items
  const FRAME1 = ['{', '"name"', ':', '"'];
  const FRAME2 = ['", "arguments"', ':'];          // starts with the name-closing quote
  // FRAME3 ends at the wrapper '}' — the closing '</tool_call>' is a SPECIAL token (never
  // allowed inside the span, excluded from the vocab scan) so the ENGINE injects it after
  // {done} rather than the grammar trying to match it.
  const FRAME3 = ['}'];
  const CANON = { frame1: '{"name": "', frame2: '", "arguments": ', frame3: '}' };

  // ---- one-time vocab scan (independent of tool names → cached per vocab) ----------
  let _scanCache = null;
  function scanVocab(vocabSize, tokenOf, specialIds) {
    if (_scanCache && _scanCache.vocabSize === vocabSize) return _scanCache;
    const strs = new Array(vocabSize).fill(null);
    const words = Math.ceil(vocabSize / 32);
    const pureStrWords = new Uint32Array(words);
    const quoteToks = [], slashToks = [], structToks = [];
    const identIndex = new Map();   // first char -> ids (identifier-ish tokens, for tool names)
    const RE_PURE = /^[^"\\\u0000-\u001f\ufffd]+$/;
    const RE_STRUCT = /^[ \t\r\n{}\[\]:,"0-9+\-.eEtruefalsn]+$/;
    const RE_IDENT = /^[A-Za-z0-9_.\-]+$/;
    for (let id = 0; id < vocabSize; id++) {
      if (specialIds && specialIds.has(id)) continue;   // NEVER allow special/added tokens in the span
      let s;
      try { s = tokenOf(id); } catch (_) { continue; }
      if (!s) continue;
      strs[id] = s;
      if (RE_PURE.test(s)) pureStrWords[id >> 5] |= (1 << (id & 31));
      if (s.length <= 24 && s.indexOf('�') < 0) {
        if (s.indexOf('"') >= 0) quoteToks.push(id);
        else if (s.indexOf('\\') >= 0) slashToks.push(id);
      }
      if (s.length <= 16 && RE_STRUCT.test(s)) structToks.push(id);
      if (s.length <= 32 && RE_IDENT.test(s)) {
        const c = s[0];
        let a = identIndex.get(c); if (!a) identIndex.set(c, a = []);
        a.push(id);
      }
    }
    _scanCache = { vocabSize, strs, pureStrWords, quoteToks, slashToks, structToks, identIndex };
    return _scanCache;
  }

  // ---- automaton state ---------------------------------------------------------------
  // phase: ws0 → frame1 → name → frame2 → value → frame3 → done (or dead)
  function newState() {
    return {
      phase: 'ws0', item: 0, ii: 0,   // frame progress
      name: '',                        // matched tool-name prefix
      stack: [],                       // value PDA stack — codes below
      str: 0, esc: false, uni: 0,      // in-string bookkeeping
      lit: '', liti: 0,                // pending true/false/null
      num: '',                         // number chars so far
      spanChars: 0,
    };
  }
  function clone(s) { return { phase: s.phase, item: s.item, ii: s.ii, name: s.name, stack: s.stack.slice(), str: s.str, esc: s.esc, uni: s.uni, lit: s.lit, liti: s.liti, num: s.num, spanChars: s.spanChars }; }

  // PDA stack codes:
  //  'O' args object must open '{'      'K' expect key-or-'}'    'k' inside key string
  //  'C' expect ':'                     'V' expect a value       'S' inside value string
  //  'E' after object member: ','/'}'   'A' fresh array: value-or-']'   'F' after array elem: ','/']'
  function makeStepper(names) {
    const isPrefix = (p) => names.some(n => n.startsWith(p));
    const isFull = (p) => names.indexOf(p) >= 0;

    function step(st, ch) {
      if (st.phase === 'dead') return false;
      if (st.phase === 'done') { st.phase = 'dead'; return false; }
      if (++st.spanChars > SPAN_MAX) { st.phase = 'dead'; return false; }
      const die = () => { st.phase = 'dead'; return false; };

      // --- frame phases: literal items with optional whitespace between --------------
      if (st.phase === 'ws0' || st.phase === 'frame1' || st.phase === 'frame2' || st.phase === 'frame3') {
        if (st.phase === 'ws0') {
          if (WS.indexOf(ch) >= 0) return true;
          st.phase = 'frame1'; st.item = 0; st.ii = 0;
        }
        const items = st.phase === 'frame2' ? FRAME2 : (st.phase === 'frame3' ? FRAME3 : FRAME1);
        if (st.ii === 0 && st.item > 0 && WS.indexOf(ch) >= 0) return true;   // ws between items
        const it = items[st.item];
        if (ch !== it[st.ii]) return die();
        st.ii++;
        if (st.ii === it.length) {
          st.item++; st.ii = 0;
          if (st.item === items.length) {
            if (st.phase === 'frame1') { st.phase = 'name'; st.name = ''; }
            else if (st.phase === 'frame2') { st.phase = 'value'; st.stack = ['O']; }
            else st.phase = 'done';
          }
        }
        return true;
      }

      // --- tool name --------------------------------------------------------------
      if (st.phase === 'name') {
        if (ch === '"') {   // the '"' starting FRAME2's '", "arguments"' item
          if (!isFull(st.name)) return die();
          st.phase = 'frame2'; st.item = 0; st.ii = 1;   // ii=1: the quote itself is consumed
          return true;
        }
        const p = st.name + ch;
        if (!isPrefix(p)) return die();
        st.name = p; return true;
      }

      // --- value PDA ----------------------------------------------------------------
      const S = st.stack;
      const finishIfEmpty = () => { if (S.length === 0) { st.phase = 'frame3'; st.item = 0; st.ii = 0; } };
      if (st.num) {   // number in flight; delimiter re-dispatches
        // Incremental number DFA — every in-flight state must be a PREFIX of a valid JSON
        // number, or later steps hit a dead-end where NO token can complete it (the
        // "-e…" empty-mask bug the random-walk test caught). st.num = '-'|int|frac|exp form.
        if ('0123456789+-.eE'.indexOf(ch) >= 0) {
          const cand = st.num + ch;
          const PREFIX = /^-?(?:\d+)?(?:\.(?:\d+)?)?(?:[eE][+-]?(?:\d+)?)?$/;
          // must be a prefix AND each stage only opens after the prior stage has a digit
          const stageOk = /^-?$|^-?\d+$|^-?\d+\.$|^-?\d+\.\d+$|^-?\d+(\.\d+)?[eE]$|^-?\d+(\.\d+)?[eE][+-]$|^-?\d+(\.\d+)?[eE][+-]?\d+$/.test(cand);
          if (cand.length <= 32 && PREFIX.test(cand) && stageOk) { st.num = cand; return true; }
          return die();
        }
        if (!/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(st.num)) return die();
        st.num = ''; finishIfEmpty();
        if (st.phase !== 'value') return step(st, ch);   // args object can't close via a bare number, but stay safe
        return stepStructural(st, ch);
      }
      if (st.lit) {   // true/false/null in flight
        if (ch !== st.lit[st.liti]) return die();
        st.liti++;
        if (st.liti === st.lit.length) { st.lit = ''; st.liti = 0; finishIfEmpty(); }
        return true;
      }
      const t = S[S.length - 1];
      if (t === 'S' || t === 'k') {   // inside a string
        if (st.uni > 0) { if (!/[0-9a-fA-F]/.test(ch)) return die(); st.uni--; return true; }
        if (st.esc) { if ('"\\/bfnrtu'.indexOf(ch) < 0) return die(); st.esc = false; if (ch === 'u') st.uni = 4; return true; }
        if (ch === '\\') { st.esc = true; return true; }
        if (ch === '"') {
          S.pop(); st.str = 0;
          if (t === 'k') S.push('C'); else finishIfEmpty();
          return true;
        }
        if (ch.charCodeAt(0) < 0x20 || ch === '�') return die();
        return (++st.str <= STR_MAX) ? true : die();
      }
      return stepStructural(st, ch);

      function stepStructural(st2, c) {
        const S2 = st2.stack;
        const t2 = S2[S2.length - 1];
        const die2 = () => { st2.phase = 'dead'; return false; };
        const fin = () => { if (S2.length === 0) { st2.phase = 'frame3'; st2.item = 0; st2.ii = 0; } };
        if (WS.indexOf(c) >= 0) return true;
        if (t2 === 'O') { if (c !== '{') return die2(); S2.pop(); S2.push('K'); return true; }
        if (t2 === 'V' || t2 === 'A') {
          if (t2 === 'A' && c === ']') { S2.pop(); fin(); return true; }
          if (S2.length >= DEPTH_MAX) return die2();
          S2.pop();
          if (t2 === 'A') S2.push('F');   // array follow-up sits BENEATH the element (invariant)
          if (c === '"') { S2.push('S'); return true; }
          if (c === '{') { S2.push('K'); return true; }
          if (c === '[') { S2.push('A'); return true; }
          if (c === 't') { st2.lit = 'true'; st2.liti = 1; return true; }
          if (c === 'f') { st2.lit = 'false'; st2.liti = 1; return true; }
          if (c === 'n') { st2.lit = 'null'; st2.liti = 1; return true; }
          if (c === '-' || (c >= '0' && c <= '9')) { st2.num = c; return true; }
          return die2();
        }
        if (t2 === 'K') {
          if (c === '}') { S2.pop(); fin(); return true; }
          if (c === '"') { S2.pop(); S2.push('E'); S2.push('k'); return true; }   // 'E' beneath = member follow-up
          return die2();
        }
        if (t2 === 'C') { if (c !== ':') return die2(); S2.pop(); S2.push('V'); return true; }   // 'E' stays beneath 'V'
        if (t2 === 'E') {
          if (c === ',') { S2.pop(); S2.push('E'); S2.push('k2'); return true; }
          if (c === '}') { S2.pop(); fin(); return true; }
          return die2();
        }
        if (t2 === 'k2') { if (c !== '"') return die2(); S2.pop(); S2.push('k'); return true; }
        if (t2 === 'F') {
          if (c === ',') { S2.pop(); S2.push('F'); S2.push('V'); return true; }
          if (c === ']') { S2.pop(); fin(); return true; }
          return die2();
        }
        return die2();
      }
    }
    return step;
  }

  // canonical continuation text for the forced (frame) phases
  function canonicalRemainder(st) {
    if (st.phase === 'ws0') return CANON.frame1;
    const items = st.phase === 'frame2' ? FRAME2 : (st.phase === 'frame3' ? FRAME3 : FRAME1);
    const canon = CANON[st.phase];
    let done = '';
    for (let i = 0; i < st.item; i++) done += items[i];
    done += (items[st.item] || '').slice(0, st.ii);
    let ci = 0, di = 0;                       // align consumed chars against the canonical text,
    while (di < done.length && ci < canon.length) {   // skipping canonical ws the model didn't emit
      if (canon[ci] === done[di]) { ci++; di++; }
      else if (WS.indexOf(canon[ci]) >= 0) ci++;
      else break;
    }
    return canon.slice(ci);
  }

  // ---- public factory ------------------------------------------------------------------
  function createToolCallGrammar({ vocabSize, tokenOf, toolNames, specialIds }) {
    if (!toolNames || !toolNames.length) return null;
    const scan = scanVocab(vocabSize, tokenOf, specialIds || new Set());
    const { strs, pureStrWords, quoteToks, slashToks, structToks, identIndex } = scan;
    const words = Math.ceil(vocabSize / 32);
    const step = makeStepper(toolNames);
    let st = newState();

    function feedText(state, text) {
      for (let i = 0; i < text.length; i++) if (!step(state, text[i])) return false;
      return true;
    }

    const g = {
      toolNames: toolNames.slice(),
      reset(residual) { st = newState(); return residual ? feedText(st, residual) : true; },
      phase() { return st.phase; },
      // → {done:true} | {dead:true} | {forced: text} | {maskWords: Uint32Array}
      next() {
        if (st.phase === 'done') return { done: true };
        if (st.phase === 'dead') return { dead: true };
        if (st.phase !== 'name' && st.phase !== 'value') return { forced: canonicalRemainder(st) };
        const mask = new Uint32Array(words);
        const allow = (id) => { mask[id >> 5] |= (1 << (id & 31)); };
        const trySim = (id) => {
          const s = strs[id]; if (!s) return;
          const sim = clone(st);
          if (feedText(sim, s) && sim.phase !== 'dead') allow(id);
        };
        if (st.phase === 'name') {
          const nexts = new Set();
          for (const n of toolNames) if (n.startsWith(st.name) && n.length > st.name.length) nexts.add(n[st.name.length]);
          for (const c of nexts) { const a = identIndex.get(c); if (a) for (const id of a) trySim(id); }
          for (const id of quoteToks) trySim(id);   // the '", "arguments"…' continuation
        } else {
          const t = st.stack[st.stack.length - 1];
          const inStr = (t === 'S' || t === 'k') && !st.esc && st.uni === 0;
          if (inStr) {
            if (st.str < STR_MAX - 64) mask.set(pureStrWords);   // bulk-allow safe content
            for (const id of quoteToks) trySim(id);              // close + spill into structure
            for (const id of slashToks) trySim(id);              // escapes
          } else {
            for (const id of structToks) trySim(id);
            for (const id of quoteToks) trySim(id);
            if (st.esc || st.uni > 0) for (const id of slashToks) trySim(id);
          }
        }
        return { maskWords: mask };
      },
      advance(tokenId) {
        const s = strs[tokenId];
        if (s == null) { st.phase = 'dead'; return false; }
        return feedText(st, s) && st.phase !== 'dead';
      },
      advanceText(text) { return feedText(st, text) && st.phase !== 'dead'; },
      // Random-walk self-test (no GPU): every walk must terminate in valid, parseable JSON
      // with a registered name. pick(nAllowed, walk, step) → index (deterministic tests).
      selfTest(N, pick) {
        N = N || 20;
        const fails = [];
        for (let w = 0; w < N; w++) {
          this.reset('');
          let text = '', steps = 0, failed = false;
          while (steps++ < 800) {
            const nx = this.next();
            if (nx.done) break;
            if (nx.dead) { fails.push({ w, err: 'dead', text: text.slice(-200) }); failed = true; break; }
            if (nx.forced != null) {
              if (!this.advanceText(nx.forced)) { fails.push({ w, err: 'forced-died', text: text.slice(-200) }); failed = true; break; }
              text += nx.forced; continue;
            }
            const ids = []; const closers = []; const mw = nx.maskWords;
            for (let i = 0; i < vocabSize && ids.length < 30000; i++) if (mw[i >> 5] & (1 << (i & 31))) {
              ids.push(i);
              const s0 = strs[i];
              if (s0 && (s0.indexOf('"') >= 0 || s0.indexOf('}') >= 0 || s0.indexOf(']') >= 0)) closers.push(i);
            }
            if (!ids.length) { fails.push({ w, err: 'empty-mask', text: text.slice(-200) }); failed = true; break; }
            // 40% closing bias — an unbiased random walk over ~30K string-content tokens
            // essentially never closes a string within the step cap (test artifact, not grammar)
            const pool = (closers.length && (pick ? (steps % 5 < 2) : Math.random() < 0.4)) ? closers : ids;
            const id = pool[(pick ? pick(pool.length, w, steps) : Math.floor(Math.random() * pool.length)) % pool.length];
            if (!this.advance(id)) { fails.push({ w, err: 'advance-died tok=' + JSON.stringify(strs[id]), text: text.slice(-200) }); failed = true; break; }
            text += strs[id];
          }
          if (failed) continue;
          if (st.phase !== 'done') { fails.push({ w, err: 'not-done: ' + st.phase, text: text.slice(-300) }); continue; }
          const body = text.trim();
          try {
            const j = JSON.parse(body);
            if (toolNames.indexOf(j.name) < 0) fails.push({ w, err: 'bad-name ' + j.name });
            else if (typeof j.arguments !== 'object' || j.arguments === null) fails.push({ w, err: 'bad-args' });
          } catch (e) { fails.push({ w, err: 'parse: ' + e.message, text: body.slice(0, 300) }); }
        }
        return { ok: fails.length === 0, walks: N, fails: fails.slice(0, 5) };
      },
    };
    return g;
  }

  return { createToolCallGrammar, scanVocab };
})();

if (typeof window !== 'undefined') window.SandpieGrammar = SandpieGrammar;
if (typeof self !== 'undefined') self.SandpieGrammar = SandpieGrammar;
