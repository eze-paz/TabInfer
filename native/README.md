# N2-inference

A native ternary LLM inference engine in Rust. No browser, no WebGPU, no WASM in
the hot path.

This exists because sandpie's local-inference work hit ceilings that were
properties of the *runtime*, not the hardware. The WASM CPU engine was capped at
128-bit SIMD, blind to the machine's ISA, and unable to control thread placement.
Its handoff concluded that "on 4 real cores, a dense 1.7B ternary at ~10-17 tok/s
is near the hardware ceiling" — a conclusion that was correct *for WASM* and is
what this repo re-tests with the constraints removed.

The first measurement says the ceiling moved.

## Why Rust

Not for elegance — for three specific reasons:

1. **The prior art is already Rust.** The ternary kernel that hit 34.7 tok/s was
   Rust compiled to `wasm32-unknown-unknown`. The algorithms port directly; only
   the SIMD intrinsics change.
2. **The toolchain is here.** `cargo 1.96.1` and `rustc 1.96.1` are installed on
   this box. There is no `gcc` or `clang`. Choosing C meant installing a compiler
   before writing a line.
3. **WASM stays reachable.** `wasm32` remains one `cfg` branch away, so if the
   browser path is ever wanted again it is a backend, not a rewrite.

C would not have been faster. The hot loops are intrinsics either way, and this
repo compiles with `panic = "abort"`, `lto = "fat"`, and zero dependencies.

## Status

Working and tested:

- **Ternary kernel** (`Q2_0_g64`) — scalar reference plus a new x86_64 AVX2 path.
  **7.3x over scalar, 26.15 GMAC/s single-threaded** on this box.
- **Correctness oracle** — the kernel is checked against an independent
  dequantise-and-multiply reference, not just against another backend. Exact
  agreement (0.0 relative error) on every case.
- **Container reader** — the `.cpu.bin` format, validated against the real
  1.16 GB Bonsai-1.7B pack: 311 tensors, read in 1.9 s.
- **Benchmark harness** — alternating-order, min-of-N, with a mandatory
  base-vs-base control run.

Not built yet: the forward pass. See [docs/ROADMAP.md](docs/ROADMAP.md).

## What this machine can do

`n2 cpu` on the target box (Alder Lake mobile, 12 logical CPUs):

```
avx2           yes       256-bit integer SIMD: 2x the vector width WASM could reach
fma            yes       fused multiply-add for the f32 glue
avxvnni        yes       single-instruction u8xi8 dot (vpdpbusd): the 2x on the ternary kernel
avx512f        -         fused off on consumer Alder Lake, as expected
bmi2           yes       faster bit extraction for the 2-bit unpack
```

**AVX-VNNI is present.** That is the headline: `vpdpbusd` collapses the current
two-instruction `vpmaddubsw` + `vpmaddwd` dot into one, and it was structurally
unreachable from WASM. It is the largest single kernel lever left.

## Usage

```bash
cargo test --release
```

```bash
cargo run --release -p n2-cli -- cpu
```

```bash
cargo run --release -p n2-cli -- selftest
```

```bash
cargo run --release -p n2-cli -- bench 2048 2048
```

```bash
cargo run --release -p n2-cli -- info path/to/bonsai17.cpu.bin
```

## Layout

| Path | What |
|---|---|
| `crates/n2-kernels` | SIMD kernels, CPU feature detection, benchmark harness |
| `crates/n2-core` | `.cpu.bin` container, header JSON, f16 decode |
| `crates/n2-cli` | the `n2` binary |
| `tools/` | `pack_bin.py` (safetensors → `.cpu.bin`), `ref_forward.py` (torch oracle) |
| `docs/FORMAT.md` | byte-level weight format spec |
| `docs/ROADMAP.md` | what's next, and what is already ruled out by measurement |
| `docs/HANDOFF-cpubonsai.md` | the WASM engine's findings, carried forward |
| `reference/` | **not compiled.** The old WASM kernel and JS coordinator, kept for reading |

`reference/` is documentation. Nothing in it is built or linked — the native
kernels in `crates/n2-kernels` were written against the *format contract*, with
the WASM source used only to confirm the packing and the `xsum` convention.

## Ground rules

- **Zero dependencies.** Offline builds, a trivial wasm32 target, and nothing in
  the hot path we didn't write. Adding one requires a note in the roadmap.
- **Absolute timings from this box are not evidence.** It measured ±40%
  run-to-run under WASM. Only back-to-back relative A/Bs with a clean control
  run count. The harness enforces this; don't route around it.
- **Every kernel is validated against the dequantise oracle**, never only
  against another kernel. Two backends that share a misreading of the packed
  layout agree perfectly and are both wrong — that has already happened once in
  this project's history.

## Building on Windows

`.cargo/config.toml` sets `LIB` to the Windows SDK directories. Without it,
linking a binary fails with `LNK1181: cannot open input file 'dbghelp.lib'`
because rustc's MSVC detection finds the VC tools but not the SDK. Update the
paths there if the SDK or MSVC version changes.
