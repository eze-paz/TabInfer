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

**It runs a model.** `n2 run` generates from Bonsai-1.7B end to end:

```
$ n2 run bonsai17.cpu.bin --max-new 24 --verify
prompt  13 tokens
generated ids: [151667, 198, 151668, 271, 785, 6722, 315, 9625, 374, 3070, 59604, 334, 13, 151645]

phase        tokens      seconds          tok/s
prefill          13        2.374           5.48
decode           13        3.295           3.95
```

which decodes to `<think>\n</think>\n\nThe capital of France is **Paris**.`,
stopping on `<|im_end|>`.

Working and tested:

- **Forward pass** — 28 layers, RMSNorm, GQA 16Q/8KV with per-head qk-norm,
  split-half RoPE at `theta = 1e6`, SwiGLU, KV cache, untied `lm_head`, greedy
  sampling. Allocation-free per token.
- **Ternary kernel** (`Q2_0_g64`) — scalar reference plus an x86_64 AVX2 path.
  Exact agreement with an independent dequantise-and-multiply oracle.
- **Container reader** — `.cpu.bin`, validated on the real 1.16 GB pack: 311
  tensors, 2.50 bits/param. Blob is read into a `u32`-backed allocation so the
  in-place f32 views are aligned by construction.
- **Benchmark harness** — alternating order, min-of-N, mandatory control run.

Two honest gaps:

- **Text in, text out needs a tokenizer.** Prompts are token IDs today
  (`--ids 1,2,3`). BPE is next.
- **The forward pass is not numerically verified.** It answers correctly, which is
  evidence but not proof — a subtly wrong RoPE or GQA mapping can still produce
  fluent, correct-looking output. The numpy oracle needs ~11 GiB and could not be
  run on this box. See "Ground rules" below; this is exactly the case they exist
  for, and it stays open.
- **Single-threaded, and slower than the kernel predicts.** 3.95 tok/s is 6.8
  GMAC/s against the kernel's 15.32 GMAC/s streaming rate. Unexplained; see the
  roadmap.

See [docs/ROADMAP.md](docs/ROADMAP.md).

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
