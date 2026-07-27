# Sandpie — Project Handoff

**Date:** 2026-07-26 · **Repo:** eze-paz/sandpie (client SPA) · **Branch:** main · **HEAD at handoff:** `09e0bfd`

This covers two workstreams from this session: **(A) the memory system redesign** and **(B) the pure-CPU Bonsai model** (hosting, caching, and a perf investigation). Deployment is the owner's job — everything here is committed+pushed to `main`; the deployed site must pull `main` and hard-reload to pick it up. The repo also received concurrent work in this window (sharing / home / file-viewer) not authored here.

---

## A. Memory system redesign

### The problem
All durable facts (`sandpie/memory/*.md`, one fact per file, injected into the system prompt) were dumped in full every turn — **~13K tokens on the 82-fact store, 2.6× over the 5000 budget**. The sidebar "constellation" graph laid nodes out by the 4 types with meaningless edges. `paths:` frontmatter was garbage (a greedy regex over tool *output text* had scraped code fragments, URLs, `:line:col`, truncations).

### What shipped (commits)
- `55c81bf` **real path provenance + path graph** — `remember()` now derives `paths:` from **structured tool-call arguments** (`args.path/src/dest` + shell write-targets), not regex over result text; capped `tool_calls` at 8. Graph groups/links by shared **canonical path** (suffix-union, bare-basename guarded) instead of type.
- `cafa07e` **tiered injection + `project` field + `recall()` tool** — see below.
- `ea775b2` **automatic deterministic consolidation** (supersede near-dupes, no LLM).
- `3ffab0a` **event-driven consolidation trigger** (on memory write, debounced; dropped the hourly clock).
- `bdc5d91` **graph nodes open in the file viewer + clickable `[[wiki-links]]`** (frontmatter hidden in the md preview).
- `179b5fe` **repo-aware project labeling** — first-path-segment couldn't unify the sandpie client (no common prefix); new labeler maps to real repos (riscv-vm / sandpie / sandpie-server / skills/* / files/projects/*) with a keyword fallback. **Backfilled `project:` into 71 of 82 files.**
- `44fcd2e` **conversation-scoped + rank-capped activation** — fixed a new-chat bug where the *global* recent-paths accumulator activated ~4 projects at once; now uses this conversation's touched files + scores projects and activates only the top 1–2.
- `0e6e328` graph nodes coloured by activation, not type.

### How injection works now (`modules/memory.js` `systemBlock(ctx)`)
- **Tier 0** — `user`/`feedback` facts: always injected in full (identity + how-to-work).
- **Tier 2** — the *active project(s)*' `project`/`reference` facts: promoted to full, newest-first, up to a ~5K-tok budget. Overflow → index + a consolidation nudge.
- **Tier 1** — everything else: a one-line index grouped by project.
- **Tier 3** — `recall("keywords")` worker tool pulls any indexed fact in full.
- **"Active project"** = scored from (a) this conversation's touched files (`getConvPaths(activeConvId)`, **not** the global list) and (b) keyword hits on the latest user message; only the top 1–2 activate.
- Measured on the real store: cold ~3.1K tok, deep-in-a-project ~5–7K, vs the old flat ~13K.

### Consolidation (deterministic, recoverable, no LLM)
- Fires **on memory write** (debounced ~4s), gated by `sandpie-memory-autoconsolidate` (default on).
- **Explicit:** `remember(supersedes:[names])` → archive those to `sandpie/memory/.pruned/`.
- **Automatic:** near-identical same-project facts (Jaccard ≥ **0.65**) → archive the older. The 0.65 threshold was chosen from a dry-run: lower thresholds archived *distinct* facts (similarity ≠ supersession). Mid-band `[0.4,0.65)` pairs are only *reported* as candidates.
- Commands: `>>> memory consolidate [dry]`, `>>> memory restore <name>`.

### Config / localStorage keys
- `sandpie-memory-enabled` (default on), `sandpie-memory-threshold` (5000), `sandpie-memory-autoconsolidate` ('0' disables), `sandpie-memory-tiered` ('0' = legacy full-dump).

### Open items (memory)
1. **OKF conformance** (deferred): generate `index.md` from frontmatter, render `[[links]]` as real markdown links, add a `timestamp` field, optional bundle import. Interop-only; low urgency.
2. **augmentations `## Recent paths` block is still the GLOBAL list** — separate feature from memory activation; it's stale-in-a-new-chat noise. Same conversation-scoped fix (`getConvPaths`) should be pointed at it.
3. A large unified project (e.g. `sandpie` ~25 facts, `riscv-vm` ~31) can still overflow Tier-2 — that's the signal to actually *consolidate* the redundant/contradictory facts (there are ~4 overlapping "413" facts and contradictory sandpie source-location facts now clustered together).
4. **Semantic (embedding) activation** was floated as the upgrade over keyword activation; not built.

---

## B. Pure-CPU Bonsai-1.7B (ternary WASM engine)

The `bonsai-1.7b-cpu` model: `modules/cpu-bonsai.js` adapter running the coordinator `_cpukern/cpuengine-mt.js` + `cpukern*.wasm` inside `webgpu-worker.js`. It is the **universal CPU floor** — the fallback for machines with no usable GPU. Not the speed path.

### Productization (commits)
- `23014a4` robust preflight (1-byte ranged GET, not HEAD — CDN-safe) + point hosting at a CORS+range host.
- `80387ee` **OPFS caching** — the ~1GB bin downloads once into a top-level `cpu-model-cache/` OPFS dir (outside `sandpie/`, so Dropbox never syncs it), `.ok` marker guards partial downloads; `load()` gets a blob URL over the disk-backed File.
- `1b84129` **default to the hosted HF model** — no localStorage needed. Resolution: `localStorage['sandpie-cpu-bonsai-base']` override → localhost uses site-root local files → else the HF default.
- `47b93ff` **parallel ranged download** (8 concurrent 48MB chunks) + HT-aware worker default + load-phase logging.
- `0b591c9` device-sized worker pool + prefill/decode tok/s readout.
- `09e0bfd` runtime toggle for the mega-layer kernel (`__v3`) to A/B.

### Where the weights live
- **HF repo (public):** `https://huggingface.co/eze-paz/bonsai17-cpu/resolve/main/` — `bonsai17.cpu.bin` (1,160,512,406 B, **byte-identical** to the verified local build) + `tokenizer.json`. Verified CORS `*` + range OK.
- The `.cpu.bin` is a custom repack of the Bonsai safetensors (`_cpukern/pack_bin.py`: ternary→2-bit interleaved codes G=64 + f32 group scales; embed f16; norms f32). A **pure-JS port** was written and verified **byte-identical across all 311 tensors** — in the session scratchpad as `packcore.js` + `verify.js`. Kept in case an in-browser converter is revived; it was **shelved** because the only *public* 1.7B safetensors (`prism-ml/Ternary-Bonsai-1.7B-unpacked`) is a *different* snapshot (vocab 151669 vs 151936, F16 vs BF16, tied vs untied lm_head) that would need a new full-precision `lm_head` path in the engine.

### Config / localStorage keys
- `sandpie-cpu-bonsai-base` (data-file dir URL; default = the HF repo above)
- `sandpie-cpu-bonsai-workers` (sweep the worker count)
- `sandpie-cpu-bonsai-v3` ('1' = mega-layer kernel; per-run, no reload)

### Perf investigation — findings (measured on the owner's box)
- **Box:** 4 physical cores / 12 logical (HT). **Very noisy** — identical config gave 17.6 then 10.6 tok/s (±40% run-to-run). *Only trust relative A/Bs run back-to-back; never absolutes.*
- **Decode ≈ 10–17 tok/s** for the dense 1.7B. Prefill is a non-issue (batch-1, but prompts were short: ~21 tok / ~1s).
- **Not memory-bound:** ~5–8 GB/s of weight traffic = only ~1/4 of DDR bandwidth → the wall is **compute/orchestration on 4 cores**.
- **Worker count:** hyperthreading gives nothing for memory-bound work and *contends* when oversubscribed — measured **6w (17.6) > 11w (13.9)**. Default changed to `round(hc/2)` ≈ physical cores.
- **Mega-kernel (`__v3`, fewer barriers):** A/B'd — v2 10.3/10.9 vs v3 11.3, i.e. **parity within noise**. Orchestration is **not** the lever (matches the clean-box result).

### The honest conclusion
Every structural lever is now ruled out by measurement: workers (fixed at ~physical), memory (not the wall), orchestration (parity). **On 4 real cores, a dense 1.7B ternary at ~10–17 tok/s is near the hardware ceiling. No kernel trick yields ~10× here** (10× would need ~10× less compute/bytes/token — i.e. a smaller model or better hardware).
- The one remaining engine lever — **contextual FFN sparsity** — is only ~1.3–1.6× on Bonsai (SiLU, not ReLU, so not naturally sparse) and, critically, **that gain is smaller than the box's ±40% noise**, so it can't even be cleanly validated on this machine. Poor ROI.
- **Speculative decode / DFlash: ruled out** — they trade memory for compute (you're compute-bound), need a trained draft, and need batched verify (chunk mode is B=1).

### Open decision (blocks the next step)
**Does the owner's box have a usable WebGPU GPU?**
- **If yes:** the real 2–3× is *switching engines* — sandpie's WebGPU ternary Bonsai (~34 tok/s, also cached) — not squeezing the CPU engine. A picker choice, not code.
- **If no:** we're at the hardware floor; recommend **stop optimizing the CPU decode** and accept it as the runs-anywhere fallback. (Owner has not answered yet.)

### Kernel speed work (2026-07-27) — 2.1x banked in the kernel, wiring is the open piece
Established first, by reading the kernel: only **4 of ~19 instructions per 64 MACs are dots**
(~22% issue). The other 78% is *necessary* work (load + shift/mask each 2-bit weight), NOT a bug
— evidence: the same kernel scales exactly with core count (34.7 tok/s on 10 cores → 13.9
predicted on 4, measured 10–17), and it beats llama.cpp Q4 by ~2.5x on the same box. The
theoretical 111 tok/s "ceiling" is a speed-of-light bound (100% dot issue, zero loads), not an
achievable target.

Also settled: **we already use the add-trick.** Codes are stored {0,1,2} and the kernel computes
`Σc·x − Σx` (the `xsum` term) — that IS BitNet's no-multiply identity. The dot instruction is a
fused int8 MAC and is optimal; add-based ternary needs MORE instructions on SIMD. BitNet's
"no multiplication" claim is about ASIC energy, not x86 throughput.

**SHIPPED (both bit-identical, verified in Node against the same wasm the browser runs):**
- `b11ab91` **NR=4 row blocking → 1.15x.** The activation block doesn't depend on the row, but
  the kernel reloaded all 4 activation vectors per row. `gemv_tern` now delegates to
  `gemv_tern_r4`; original kept as `gemv_tern_r1`. NR=2 measured a wash. Also fixed: all four
  wasm fetches were **unversioned** (`?v=2` added) — shipping a kernel would otherwise leave
  browsers stale, or main-thread-new / workers-old.
- `50a60d8` + `8249b98` **`gemm_tern_b4` batched multi-column GEMM → 1.86x** (18 → 35 GMAC/s)
  vs 4 sequential row-blocked calls. Takes an `out_stride` param (engine needs columns strided
  by the global out width). Beat the ~1.2x instruction-count prediction because gemv's four dots
  are CHAINED (each is the next's accumulator) — 4 columns give 4 INDEPENDENT chains, so the win
  is instruction-level parallelism, not just amortized unpack. `gemm_tern_b4r2` (B=4 × NR=2)
  measured 1.72x — WORSE; 8 accumulators exceed the register budget. Annotated, unused.

**MEASUREMENT DISCIPLINE (both bit us this session):** a first benchmark timed A-then-B every
rep and produced a bogus **0.78x reversal** from monotonic drift — fixed with alternating order
+ warmup + min-of-31, and **a base-vs-base control run (must read 1.000x)** as the guard. And
Python `open(...,'w')` on Windows silently converted **LF→CRLF**, turning a 158-line diff into
2,200 lines; caught pre-push, restored with `sed -i 's/\r$//'`, verified the committed wasm still
rebuilds byte-identical. Always run the control; always check line endings after a scripted edit.

### NEXT: wire batched columns into chunk mode (NOT done — plan below)
`forwardN` already implements the whole batched graph (per-column rope, `attnThreaded(l,p0,B)`,
batched dispatch) for the NON-chunk path. Only **chunk mode** (the v2 shared-weights work-stealing
path we actually run) rejects B>1 — `dispatchMM` throws `'chunked mode: batched forwardN
unsupported'`.

**Key de-risking finding: the concurrency does NOT need to change.** Cursor, barrier, chunk
claiming, atomics all stay; each claimed chunk just computes 4 columns instead of 1. The race
surface is unchanged.

**Scoping decision that cuts most of the risk:** batch ONLY the gemv+glue (88% of decode) and
keep attention SEQUENTIAL per column — pass `attnChunk` a column offset and reuse the existing
verified single-position logic, rather than rewriting attention for B positions. Costs B attention
dispatches per layer instead of 1, so ~1.69x instead of 1.86x — most of the win, a fraction of the
risk. Batching attention can follow later.

Concretely:
1. **Layout** (shared-mem alloc, ~line 94): `swActOff` (I), `swXsumOff` (I/64), `swOutOff` (vocab),
   plus glue `sxOff`/`sxnOff`/`sswiOff` and `attnOff2` are sized for ONE column → size them `×B`.
   `sabAsc`/`sabCos`/`sabSin` already have `BMAX` slots. **KV needs no change** (each column writes
   its own position slot).
2. **Quantize**: `quantForGemv` → per-column, writing column c at `swActOff + c*K` and
   `swXsumOff + c*(K/64)*4` (kernel assumes act stride = k, xsum stride = k/64), `sabAsc[c]`.
3. **Dispatch**: `dispatchChunk(names, offs, resid, amax, B)`; in BOTH mirrored loops
   (`mainClaim` in cpuengine-mt.js AND `claimChunksW` in cpukern-fwd-worker.js — they must stay
   identical) call `gemm_tern_b4(out, vocab, …)` when B===4, and run the resid/amax/scale
   post-loops per column with per-column `sabAsc[c]`.
4. **Attention**: `attnChunk(l, pos, colBase, attnBase)` — read q/k/v from column c's output
   region, write to `attnOff2 + c*nH*hd*4`, rope with `sabCos[c*half+i]`. Call B times per layer.
5. **`forwardChunkN(tokenIds, p0)`**: mirror `forwardChunk` with per-column glue; lm_head/argmax
   only on the LAST column (prefill only needs the next token).
6. **Ship behind `__batchPrefill` (default OFF) + a self-test** that runs the same prompt through
   batched and sequential prefill and compares logits. **This cannot be verified outside a
   browser** (needs Workers + SAB + the 1GB model), and the known failure mode here is silent:
   a stale ctrl spec once made workers grind the wrong matrices and emit plausible garbage
   (`dispatch fn MUST store its full ctrl spec`). Do not enable the flag until the self-test is green.

**Payoff, stated honestly:** prefill is only **2–7% of a turn** at the owner's prompt lengths
(21 tok/1.6s vs 524 tok/37.7s), so batched prefill alone is ~2–3% end-to-end. The reason to build
it is that chunk-mode B>1 is the **same prerequisite speculative decode needs**, and that targets
the other ~95%. Note the earlier prompt-lookup speculation failed *because* verify was B=1 (zero
amortization → no benefit even at 100% acceptance); with a 1.86x batched verify the economics
change (4 tokens for ~2.15 forward-equivalents, so ≥3/4 acceptance wins) — but there is still no
trained draft, and an n-gram draft only accepts on repetitive text.

### Open items (CPU Bonsai)
1. Confirm the OPFS cache actually **hits on 2nd load** (watch the `[bonsai load] cache HIT/MISS` console line). If it MISSes every time, OPFS is being evicted → force persistent storage. Parallel download only mitigates the symptom.
2. Batched prefill is blocked by chunk-mode `B=1` (`forwardN` throws) — would need engine work; low priority (prefill is small here).
3. Optional: cap `MAX_NEW` / concise prompt to stop the model rambling (524-token outputs) — perceived-latency win, but unreliable on a 1.7B.
4. Contextual FFN sparsity prototype — only if the owner accepts the modest, hard-to-validate ceiling.

---

## Environment / working rules (carry forward)
- **Commit + push completed work to `main` automatically; stage only files changed. Never deploy** (deployment is the owner's). Check `git status` before each commit.
- **Bump the `?v=` cache-buster** in the loader (sandpie.html / the importing module) in the same commit as any JS module change. `node --check` every JS file before commit (won't catch runtime/scope bugs — also reason about scope).
- The CPU model's big files are gitignored; `cpukern*.wasm` (8KB) ARE committed and must be same-origin.
- The owner's box is noise-heavy — **relative A/Bs only** for perf.
- A live HF **write token** was used from `~/Downloads/token-hf.txt` during setup — if that file lands anywhere shared, rotate the token.
