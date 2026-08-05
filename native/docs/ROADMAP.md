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

### Projected decode, and why it is credible

Bonsai-1.7B decode costs 1.72 B MACs and **537 MB of weight traffic per token**.

- Native, 1 thread: 1.72e9 / 15.32e9 = 112 ms → **~8.9 tok/s**
- WASM, measured, multi-threaded on this box: **10-17 tok/s**

Cross-check: 10-17 tok/s x 537 MB = 5.4-9.1 GB/s, and the WASM handoff
independently measured its weight traffic at "5-8 GB/s". Those agree, which is why
the model below is worth trusting at all.

So **native single-threaded already matches WASM's entire multi-threaded memory
throughput.** Assuming 20-30 GB/s achievable aggregate once threading lands
(DDR4-3200 dual channel peaks at 51.2 GB/s, LPDDR4x-4267 at 68), decode projects to
**~25-45 tok/s, roughly 2-3x over WASM**, with ~52 tok/s as the saturation wall.

Caveats: this is GEMV arithmetic only — real decode adds attention, RoPE, norms,
softmax and thread barriers, so expect to land under the projection. And the
aggregate-bandwidth figure is an *assumption* until P1 exists; it is the largest
variable, and on a heterogeneous P/E part it is not a given.

The earlier cross-session comparison against the WASM kernel's own ~18 GMAC/s
microbenchmark is superseded by the table above and should not be quoted. It
compared a cache-resident WASM number to a cache-resident native number from a
different session on a box that swings ±40%.

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

### Caveat 2 — decode is at 44% of the kernel's rate

Per token the model does 1.72 GMAC, so 3.95 tok/s is **6.8 GMAC/s against the
kernel's 15.32 GMAC/s streaming rate.** Unexplained, and it should be profiled
before any optimisation work, because the answer decides what P1 is worth.

One identified, cheap inefficiency: `q`/`k`/`v` all consume the same normed vector
and `gate`/`up` both consume the same one, but `linear` re-quantises its input every
call — so 5 of 7 per-layer activation quantisations are redundant. Small in theory;
measure before assuming.

The leading hypothesis for the bulk of the gap is memory access pattern: the
benchmark streams one tensor in a tight loop, where hardware prefetchers do well,
while the forward pass touches 196 tensors totalling 513 MiB with f32 glue
interleaved. That is a hypothesis, not a finding.

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
4. **Zero dependencies** unless there is a note here saying why.
5. Windows-specific: check line endings after any scripted edit. Python
   `open(...,'w')` silently converted LF to CRLF once and turned a 158-line diff
   into 2,200 lines.
