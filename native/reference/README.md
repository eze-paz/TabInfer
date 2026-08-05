# reference/ — not compiled

Nothing in this directory is built, linked, or shipped. It is here to be read.

| File | What it was |
|---|---|
| `cpukern-wasm-lib.rs` | the WASM ternary kernel: `core::arch::wasm32` intrinsics, `simd128`/`relaxed-simd`, `no_std` with a custom panic handler, `cdylib` |
| `cpuengine-mt.js` | the JS coordinator: shared-weights work-stealing, shared-KV attention, chunk claiming over `SharedArrayBuffer` |

## Why it is kept, and why it is not the engine

The native kernels in `crates/n2-kernels` were written against the **format
contract** in `docs/FORMAT.md`, not by translating this code. These files were used
to confirm two things the packer alone did not state:

1. the interleaved plane-to-activation pairing, and
2. the `xsum` accumulation grouping (separate accumulators, subtracted once per
   row) that backends must agree on for their rounding to match.

They cannot be ported directly:

- The SIMD is **WASM-specific**. `v128` is 128 bits and there is no wider vector
  in the instruction set, which was the hard ceiling this repo exists to escape.
  `i32x4_relaxed_dot_i8x16_i7x16_add` has no x86 equivalent with the same shape —
  the AVX2 analogue is `vpmaddubsw` + `vpmaddwd`, and the AVX-VNNI one is
  `vpdpbusd`.
- The coordinator's concurrency is built on `SharedArrayBuffer`, `Atomics.wait`
  and Web Workers. Native threading gets real thread affinity, which is the point
  — on a heterogeneous P/E-core part an even row split is a bad split, and the JS
  design could not express that.

The JS coordinator is worth reading for its *design* rather than its mechanism:
the work-stealing chunk protocol, and the hard-won warning that a dispatch
function **must store its full control spec** — a stale one once had workers
grinding the wrong matrices while emitting plausible output with every self-test
green.
