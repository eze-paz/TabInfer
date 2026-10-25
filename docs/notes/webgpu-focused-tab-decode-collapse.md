# webgpu-local-inference-display-fix

> How local WebGPU inference was throttled to 2 tok/s (then 0.3 tok/s) by main-thread rendering work, and the multi-part fix

_Design note written during development (sandpie session memory); names like `TabInfer*` refer to the pre-extraction identifiers, now `TabInfer*`._

## Symptom
Focused browser: 2 tok/s (later 0.3 tok/s). Backgrounded browser: 20+ tok/s (later "300 tok/s / finishes instantly"). GPU at 20% when visible, 100% when hidden. The inference engine itself (webgpu-qwen3.js / webgpu-engine.js) has zero rAF, zero timers — its only yield point is `await mapAsync()` per batch.

## Root Cause 1: Parallax rAF stealing GPU submit windows (commits e80a8d8)
`_drainTick` in `conversations.js` writes `scrollHost.scrollTop = scrollHost.scrollHeight` every 16ms. This fires a `scroll` event → `onScroll()` → `requestAnimationFrame(tick)` → 60fps CSS variable loop animating `--py` on every `.msg.user` element. Those 60fps rAF callbacks run between the `await mapAsync()` resolving and the next `queue.submit()` call, starving the GPU submission window.

**Fix:** `_localInferring` flag (set true before local inference, false at finalize). `onScroll` early-returns when `_localInferring`. `_drainTick` skips the `scrollTop` write when `isLocal`.

## Root Cause 2: Bad drain interval + trickle logic (commit c1efc1f)
After fixing Root Cause 1, drain interval was changed to 100ms for local (wrong — the original reason for 16ms was not the drain but the scrollTop). `_drainTick` shows `ceil(pending/30)` chars per tick. At 100ms × 30 ticks = 3 seconds per token = 0.3 tok/s visual rate. Inference was completing at full ~25 tok/s but tokens buffered silently; appearing in one burst when backgrounded looked like "300 tok/s."

**Fix:** Revert drain to 16ms for all paths. For local mode in `_drainTick`, skip the `/30` trickle — show ALL pending chars at once per tick (local tokens arrive at a steady ~25 tok/s, not in cloud-style bursts).

## Final state (conversations.js v29)
- `_localInferring = true` at start of local sendSingle, `false` in finally before `renderer.finalize()`
- `onScroll`: `if (_localInferring) return` — no parallax rAF during decode
- `_drainTick`: skip `scrollTop` write when `this.isLocal`
- `_drainTick`: when `this.isLocal`, `displayed += pending; pending = ''` (show all, no trickle)
- `_paintContent`: when `this.isLocal`, uses `bubble.textContent = displayed` (no marked/DOMPurify per token)
- `finalize()`: does full `renderMd` once + one rAF-deferred scroll to bottom
- Drain interval: 16ms for all cases

## Diagnosis method
Symptom clean signal: GPU% goes to 100% the INSTANT the browser window loses focus. This rules out inference bugs (engine has no visibility API) and points exclusively to main-thread rendering competing with GPU submit timing.

## Root Cause 3 (REGRESSION, fixed 2026-06-25 commit 3582fa6): context-bar per-type breakdown re-tokenizing the whole transcript ~5×/s
Same focused-tab symptom RETURNED (GPU ~30% focused / 100% backgrounded) months later, affecting BOTH local backends (WebGPU Qwen3 + LiteRT/Gemma) but NOT docs/webgpu-engine-test.html (no context sidebar). The earlier fixes (worker, sp-decoding CSS, _localInferring/parallax) were all still intact — this was a NEW, independent main-thread starvation source introduced by the context sidebar's per-type token breakdown (commits e56c6c8 + 6fe90ed). Chain: the streaming HUD `paint` runs `setInterval(paint, TIMER_TICK_MS=66)` (~15fps, conversations.js ~L2224) → each tick calls `TabInferTokens.setLiveTokens` → that coalesces a `notify()` every 200ms (context.js ~L72) → `render()` → **`computeBreakdown()` ~5×/s**, which does an OPFS read + `JSON.parse` of the WHOLE conversation + `JSON.stringify(toolDefs())` + a full message walk (context.js ~L347) — EVEN WHEN THE SIDEBAR IS COLLAPSED (the `open` guard at ~L430 only skips the bar DOM, AFTER the heavy work). On the shared iGPU + few-core CPU that main-thread hammering starved the decode worker's submit windows; backgrounding throttles `setInterval` to ≥1s → hammering stops → worker feeds the GPU → full speed (the focused/background switch). Note `conversationTokens()` is already cheap during a live turn (returns `_liveTotal`, no OPFS read, context.js L47) — only `computeBreakdown()` was the offender. **Fix:** the breakdown only changes at turn boundaries, so compute it ONCE per turn. Added `TabInferTokens.isLive()` (= `_liveTotal != null && _liveConvId === active`); `render()` now reuses a module-level `_bdCache` while `isLive()` (skips computeBreakdown entirely mid-stream), and `clearLiveTokens()→notify()` at turn end recomputes it fresh against the saved file. Headline total stays live (from `_liveTotal`). VERIFIED in-browser (monkeypatched opfs.read counter, 2.4s @15fps live sim): computeBreakdown ran 0× during the live stream, 2 reads at turn end (conversationTokens + computeBreakdown). LESSON: any per-token/HUD-frequency main-thread work that touches OPFS / JSON.parse-the-transcript / stringify-toolDefs will re-create this collapse; keep streaming-time renders O(1), defer transcript-scale work to turn end. NOTE on infra: the preview server (localhost:8137) sometimes navigates into `.claude/worktrees/<name>/sandpie.html` (a separate git worktree checkout WITHOUT main-dir edits) — verify `location.href` is the bare `/sandpie.html` root + feature-probe (e.g. `TabInferTokens.isLive`) before trusting a browser test.
