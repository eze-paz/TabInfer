# project_n2-inference

> Native Rust ternary inference engine at Desktop/AI_Projects/N2-inference; replaces the WASM/WebGPU local path. AVX2 GEMV shipped at 26.15 GMAC/s; AVX-VNNI confirmed present = the next lever

_Design note written during development (sandpie session memory); names like `TabInfer*` refer to the pre-extraction identifiers, now `TabInfer*`._

As of 2026-08-05 the user has **decoupled local inference from sandpie** and wants
it worked on **natively on the PC — no WebGPU, no WASM, no browser**. New repo:
`C:\Users\aezequiel\Desktop\AI_Projects\N2-inference` (local git, `main`, first
commit 649c2ff). Explicit instruction: *"just copying the branch into the repo is
not enough"* — this is a native rewrite, not a port.

**Language: Rust**, chosen and accepted. Reasons, in order: the prior ternary
kernel was already Rust (`_cpukern/src/lib.rs` on `origin/local-llm`, cargo target
with a wasm32 profile); cargo/rustc 1.96.1 are installed on this box while **gcc
and clang are not**; wasm32 stays one `cfg` branch away. Zero external
dependencies is a deliberate, documented constraint.

**Shipped and verified:** AVX2 ternary GEMV at **26.15 GMAC/s single-threaded,
7.3x over scalar**, control run 0.975x. Validated to *exact* agreement against an
independent dequantise-and-multiply oracle (not just backend-vs-backend). Plus a
`.cpu.bin` container reader verified on the real 1.16 GB Bonsai pack (311 tensors,
2.50 bits/param) and a bench harness that enforces the measurement discipline.

**The key hardware finding: this box HAS AVX-VNNI** (`avxvnni yes`; AVX-512 is
fused off). `vpdpbusd` collapses the current two-instruction `vpmaddubsw` +
`vpmaddwd` dot into one and was structurally unreachable from WASM — it is the
largest remaining kernel lever. This is why the WASM handoff's "10-17 tok/s is near
the hardware ceiling" conclusion was a statement about the *runtime*, not the
silicon.

Roadmap lives in the repo (`docs/ROADMAP.md`), ruled-out items included; the weight
format spec is `docs/FORMAT.md`. Next up is P0, the forward pass, validated
per-layer against `tools/ref_forward.py` on torch CPU.

**Unresolved, flagged in FORMAT.md:** `tools/pack_bin.py` decodes ternary weights
with a bf16 helper but norm weights with an f16 helper from the same checkpoint.
Resolve by cross-checking torch CPU — do not "fix" by inspection, since the packed
bin is verified byte-identical to a known-good build.

Build gotcha: rustc's MSVC detection finds the VC tools but not the Windows SDK, so
linking any binary fails with `LNK1181: dbghelp.lib`. Fixed by `.cargo/config.toml`
setting `LIB` (machine-specific paths, documented there).

Related: [[project_sandpie-cpu-wasm-engine]] (the predecessor and its measured
dead ends), [[reference_bonsai-8b-eval]], [[project_sandpie-webgpu-engine]],
[[project_maple-preview-deepgrove]].
