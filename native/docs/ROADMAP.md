# Roadmap

Ordered by expected value, with the reasoning. Items that are *already ruled out
by measurement* are at the bottom — that section is the most valuable part of this
document, because it is the part that stops work being redone.

## Where we are

The ternary GEMV works natively and is verified exactly against an independent
oracle.

**Cache residency dominates the kernel number, so quote the right one.** A GEMV has
zero weight reuse *within* a call, but the benchmark reuses one matrix across reps,
so small matrices stay in cache and flatter the result. Measured on the target box,
all with clean control runs:

| working set | AVX2 | scalar | regime |
|---|---|---|---|
| 1 MiB | 23.97 GMAC/s | 3.29 | cache-resident |
| 20 MiB | 24.01 GMAC/s | 3.25 | cache-resident |
| 40 MiB | 23.54 GMAC/s | 3.13 | cache-resident |
| **80 MiB** | **15.32 GMAC/s** | 2.69 | **DRAM-streaming** |

Real decode sweeps all 513 MiB of Bonsai's ternary weights per token with nothing
resident, so **15.32 GMAC/s (= 4.8 GB/s single-threaded) is the figure that
predicts decode.** The ~24 GMAC/s numbers describe prefill-ish regimes where a
weight block is reused across columns.

### MEASURED decode: 6.3 tok/s single-threaded

Bonsai-1.7B decode costs 1.72 B MACs and 537 MB of weight traffic per token.
Three back-to-back runs of **95 decode tokens** each:

| run | decode tok/s |
|---|---|
| 1 | 6.29 |
| 2 | 6.27 |
| 3 | 6.58 |

±2.4%, so this one is trustworthy. **6.3 tok/s = 158 ms/token = 10.9 GMAC/s**,
which agrees with the profiler's independent 9.5-10.4 GMAC/s per matmul.

**Measure over ~100 tokens, not ~13.** Short runs on this model are dominated by
faulting in 1.1 GB of weights on the first pass; over 13 tokens that inflated the
spread to 3.95-6.88 tok/s (1.74x) and none of those numbers were steady-state. This
is a distinct trap from the box's general ±40% noise and it fooled us for two
commits.

### The comparison, corrected

- Native, 1 thread: **6.3 tok/s** (measured)
- WASM, multi-threaded on this box, ~6 workers: **10-17 tok/s** (measured
  previously)

Per core that is roughly **2.3x** — WASM's midpoint was ~2.7 tok/s per effective
core against native's 6.3 on one. But native single-threaded is still *below* WASM's
entire thread pool, so threading is not optional.

At 4 effective cores with realistic scaling, **~18-25 tok/s**. Bandwidth will not
bind there: 6.3 tok/s is only 3.4 GB/s of the ~25-30 GB/s available, so the ~50
tok/s bandwidth wall stays clear and compute plus thread scaling are the limits.
That is consistent with the profile finding 97.4% matmul.

Two earlier estimates in this file are **retracted**: a projected 8.9 tok/s
single-threaded (the measured figure is 29% lower, because GEMV arithmetic ignores
per-matmul overheads), and a projected 25-45 tok/s multi-threaded (now 18-25). Also
retracted: the cross-session comparison against the WASM kernel's ~18 GMAC/s
microbenchmark, which compared cache-resident numbers from different sessions.

## P0 — the forward pass — DONE, with two caveats

`n2 run` generates from Bonsai-1.7B: 28 layers, RMSNorm, GQA 16Q/8KV with per-head
qk-norm, split-half RoPE at `theta = 1e6`, SwiGLU, KV cache, greedy sampling,
allocation-free per token. The probe emits `<think>\n</think>\n\nThe capital of
France is **Paris**.` and stops on `<|im_end|>`.

Measured, single-threaded: **prefill 5.48 tok/s, decode 3.95 tok/s.**

**The bf16/f16 question is resolved and was not a bug.** The checkpoint is genuinely
mixed: embed F16, all 113 norms F16, the 197 projections BF16. `pack_bin.py` is
correct. `docs/FORMAT.md` updated.

### Caveat 1 — still not numerically verified

The model answering correctly is evidence, not proof; a subtly wrong RoPE or GQA
mapping can produce fluent, correct-looking output. The numpy oracle caches every
dequantised f32 weight and needs ~11 GiB, and this box had 2.5 GiB free and no
numpy, so it could not be run.

`tools/ref_forward.py` now takes `N2_BONSAI_DIR` so it can be pointed at the
safetensors. `n2 run --verify` prints layer-0 RMS (`xn=0.095 o=0.392
after_attn=0.391 after_mlp=0.652`) in the same shape the reference prints, so the
comparison is a diff when someone has the memory to run it.

Worth doing regardless: a no-cache mode for the oracle (dequantise per use, slower
but memory-bounded), and a golden-logits regression fixture so future graph changes
fail loudly even without the oracle.

### Caveat 2 — PROFILED: decode is 97.4% ternary matmul

`n2 run --profile` gives the per-token breakdown. Measured over 13 decode tokens:

| phase | ms/tok | share | GMAC/s |
|---|---|---|---|
| gate+up proj | 69.00 | **39.3%** | 10.21 |
| down proj | 36.03 | **20.5%** | 9.78 |
| lm_head | 29.81 | **17.0%** | 10.44 |
| qkv proj | 24.60 | 14.0% | 9.55 |
| o proj | 11.77 | 6.7% | 9.98 |
| attention | 2.54 | 1.4% | - |
| norms + rope + swiglu + residual + embed | 2.00 | 1.2% | - |

**matmul 97.4%, f32 glue 2.6%.** Phases account for 99.8% of wall time.

Consequences, including two corrections to earlier guesses in this file:

- **Threading is the whole game.** Parallelising the matmuls parallelises
  essentially all of decode. Nothing else is worth touching first.
- **The redundant-quantisation "inefficiency" is irrelevant** — all glue combined
  is 2.6%. Retracted as a target.
- **The "short streams hurt the prefetcher" hypothesis is wrong.** `lm_head` is the
  largest single tensor (78 MiB of codes) and is the *fastest* at 10.44 GMAC/s,
  while the 3 MiB per-layer tensors sit at 9.5-10.2. Stream length is not the
  factor. Retracted.
- **Part of the apparent gap was a measurement error of mine.** `n2 bench` reports
  *min of 31*; the profile reports *mean over 13 tokens*. Comparing a min to a mean
  on a ±40% box flatters the min. The same build measured 3.95 and then 5.68 tok/s
  decode on consecutive runs — that spread is noise, not change. The honest ratio is
  ~10 / 15.3 = 65% of the kernel's benchmark rate, and some of that residual is
  min-vs-mean rather than a real loss.
- **Remaining real suspect: TLB pressure.** 513 MiB per token over 4 KiB pages is
  ~131k page walks. Large pages are the lever, and worth testing after threading.

## P1 — threading

The WASM engine's finding was that **hyperthreading gives nothing here and
oversubscription actively hurts**: 6 workers measured 17.6 tok/s against 11
workers at 13.9. Its default became `round(logical/2)`.

Natively we can do better than a heuristic, because the constraint is visible: the
target is an Alder Lake mobile part, which is **heterogeneous** — a couple of
P-cores with SMT plus a cluster of E-cores. `available_parallelism()` reports 12
and means almost nothing for this workload.

- Work-stealing over row ranges (the WASM design carries over; the race surface is
  understood and documented in the handoff).
- **Thread affinity** — pin to P-cores, which WASM workers could never do. On a
  heterogeneous part an even row split is a bad split: the slowest worker sets the
  barrier. Either pin to homogeneous cores or size the chunks to the core.
- Sweep the thread count on the box and record it. Do not port the `logical/2`
  default without re-measuring; it was derived on a different topology reading.

## P2 — AVX-VNNI, and f16 scales

**Confirmed present on the target box** (`n2 cpu` reports `avxvnni yes`).

**But VNNI will do little for decode.** The streaming measurement above shows decode
is bandwidth-bound at 537 MB/token, and VNNI is a *compute* lever — once threads are
pulling 25+ GB/s a faster dot product buys nothing. Its value is in prefill (where
the ~24 GMAC/s cache-resident regime applies) and in **Maple**, where only the 8
routed experts are read per token (~150 MB) so the ratio flips back to compute-bound.
Build it for those, not for Bonsai decode.

**The cheap decode win is narrower scales.** `n2 bench 32768 8192` prints "64.0 MiB
codes + 16.00 MiB scales" — f32 scales are **20% of the streamed bytes**. Storing
them as f16 takes the format from 2.50 to 2.25 bits/param: a straight 10% traffic
cut and therefore ~10% faster decode while bandwidth-bound, for a format change and
a load-time convert. Cheaper than any kernel work, and only visible once the
regime was measured. Note it is a **format change**, so it invalidates the
byte-identity guarantee on the existing pack — version the container.

The current AVX2 inner loop spends two instructions per dot — `vpmaddubsw` then
`vpmaddwd` — where WASM's `relaxed_dot` spent one. So at 256 bits the present win
comes from halved load and unpack traffic, not from the multiply-accumulate issue
rate, which merely breaks even. `vpdpbusd` is the single instruction that closes
that gap.

Expected: a further ~1.5-2x on the kernel. This is the largest remaining lever and
it was structurally unreachable from WASM.

Implementation note: verify how `_mm256_dpbusd_avx_epi32` is gated in Rust 1.96
before committing to it, and keep the AVX2 path as the runtime fallback. Operand
convention is the same as `vpmaddubsw` — unsigned codes, signed activations — so
the surrounding code does not change.

AVX-512 is absent (fused off on this part), so 512-bit variants are not a path
here. They may be on other hardware; the backend split already allows for it.

## P3 — batched multi-column GEMM

The WASM engine measured **1.86x** for `gemm_tern_b4` over four sequential GEMVs,
and understood why: the four dots in a single-column GEMV are *chained* (each is
the next one's accumulator), so four columns give four independent chains. The win
is instruction-level parallelism, not just amortised unpack. That reasoning is
architecture-independent and should reproduce natively.

Direct payoff is small — prefill is only 2-7% of a turn at realistic prompt
lengths. The reason to build it is that batched verify is the **prerequisite for
speculative decoding**, which targets the other ~95%. Note the earlier prompt-
lookup speculation failed precisely *because* verify was B=1: zero amortisation
meant no benefit even at 100% acceptance.

## P4 — drop the f16 embedding table

`n2 info` on the real pack: the f16 embedding is **593.5 MiB of a 1106.7 MiB file,
53% of the bytes for ~18% of the parameters.** The ternary weights are 512.7 MiB.

Quantising or ternarising the embedding roughly halves the model on disk and in
RAM. Untied `lm_head` is already ternary, so this is only the input table, where
precision matters least.

## P5 — Maple 20B-A1B

`deepgrove/maple-preview` is a 20B-total / 1B-active ternary MoE from the same lab
as Bonsai — a 5.31 GB checkpoint that reportedly runs at 218 tok/s on an M4. It is
the natural second model, and the reason this engine is worth building.

New work beyond Bonsai:

- **256 experts, top-8, `moe_intermediate_size` 512.** Fine-grained MoE. Naively
  that is 8 experts x 3 matmuls x 24 layers = 576 tiny GEMVs per token. Needs a
  grouped kernel that does all 8 routed experts in one pass per layer.
- **Sliding-window attention, 3:1 SWA-512 to full.** Only 4 of 24 layers need
  full-context KV, so long context is nearly free — at int8, 8k context is ~43 MiB
  total. Per-layer KV sizing driven off the config's `layer_types`.
- **NoPE on the global layers** plus `partial_rotary_factor: 0.5`. Ten lines, and
  exactly the kind of config detail that self-tests cannot catch. Cross-check
  against torch.
- Weight conversion from the MLX 2-bit pack or the bf16 master.

## P6 — wasm32 backend, if ever wanted

The old kernel is in `reference/cpukern-wasm-lib.rs` and the SIMD abstraction
already has room for it. Not built, because the whole point of this repo is to
stop paying WASM's costs. Cheap to add back as a `cfg` branch if the browser path
returns.

---

## Ruled out by measurement — do not redo

From `docs/HANDOFF-cpubonsai.md`. These were established on the WASM engine; the
ones that are properties of the *model and box* rather than the runtime still
hold, and are flagged as such.

- **Orchestration is not the lever.** The "mega-layer" kernel with fewer barriers
  A/B'd at parity within noise (v2 10.3/10.9 vs v3 11.3). Runtime-independent.
- **Not memory-bound.** ~5-8 GB/s of weight traffic is about a quarter of
  available DDR bandwidth. The wall was compute and orchestration. Runtime-
  independent, and it is *why* wider vectors and VNNI are the right targets.
- **LUT-based ternary decode: negative.** Measured slower. Also ruled out on GPU
  separately.
- **Add-based ternary instead of the fused int8 dot: more instructions, not
  fewer.** The `xsum` identity already *is* the add trick.
- **Contextual FFN sparsity: poor ROI.** Only ~1.3-1.6x on Bonsai (SiLU is not
  naturally sparse), which is *smaller than the box's ±40% noise*, so it cannot
  even be cleanly validated here.
- **Speculative decode as previously attempted: ruled out** — needs a trained
  draft model and a batched verify. P3 is the prerequisite; revisit only after it.
- **Hyperthreading buys nothing** for this workload and oversubscription costs.
  Re-measure the exact thread count natively (see P1), but do not expect SMT to
  help.

## Ground rules that are not negotiable

1. **Every kernel is validated against the dequantise oracle**, not only against
   another kernel. Two backends sharing a layout misreading agree perfectly and
   are both wrong.
2. **Run the control.** A base-vs-base A/B must read ~1.000x or the session's
   numbers are discarded. A naive A-then-B loop once produced a 0.78x *reversal*
   from monotonic drift — a wrong sign, not a wrong magnitude.
3. **Absolute timings from this box are not evidence.** Back-to-back relative A/Bs
   only.
4. **Zero dependencies in the engine** (`n2-kernels`, `n2-core`) — both are still
   dependency-free. One documented exception exists, in the CLI only:
   `tokenizers` (with `default-features = false, features = ["fancy-regex"]`, so no
   C toolchain and no onig). `tokenizer.json` needs an NFC normaliser, the Qwen2
   split regex with `\p{L}`/`\p{N}` Unicode classes, `ignore_merges`, added-token
   handling and a ByteLevel decoder; hand-rolling that requires Unicode category
   tables, and a tokenizer that is 99% correct produces text that looks fine and
   scores badly. Correctness beat purity. Nothing in the hot path is third-party.
5. Windows-specific: check line endings after any scripted edit. Python
   `open(...,'w')` silently converted LF to CRLF once and turned a 158-line diff
   into 2,200 lines.
