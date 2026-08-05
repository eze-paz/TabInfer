# Roadmap

Ordered by expected value, with the reasoning. Items that are *already ruled out
by measurement* are at the bottom — that section is the most valuable part of this
document, because it is the part that stops work being redone.

## Where we are

The ternary GEMV works natively and is verified exactly against an independent
oracle. AVX2 gives **7.3x over scalar, 26.15 GMAC/s single-threaded** on the
target box, with a clean 0.975x control run.

For context, the WASM kernel's own microbenchmarks recorded ~18 GMAC/s for
`gemv_tern` and ~35 GMAC/s for the batched `gemm_tern_b4`. So the native
single-column path already appears to beat the WASM single-column path by roughly
1.45x. **Treat that as indicative, not measured** — it compares numbers from two
different sessions on a box that swings ±40%, which is exactly the comparison this
project's rules forbid. It will only be a real result when both run back-to-back.

## P0 — the forward pass

Nothing downstream can be evaluated until the model runs. Needed:

- RMSNorm, SwiGLU, RoPE, GQA attention, KV cache, argmax/sampling.
- Wire `n2-core`'s container to a Bonsai-1.7B config and decode one token.
- **Validate against `tools/ref_forward.py`** (torch CPU) layer by layer, not just
  on final output. The failure mode this project has actually hit is a shared
  misread that produces plausible text; per-layer comparison against torch is the
  only thing that catches it.
- Resolve the bf16/f16 norm-decode question in `docs/FORMAT.md` as part of this.

Target: a `n2 run <bin> "prompt"` that emits correct tokens. Speed is P1's problem.

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

## P2 — AVX-VNNI

**Confirmed present on the target box** (`n2 cpu` reports `avxvnni yes`).

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
