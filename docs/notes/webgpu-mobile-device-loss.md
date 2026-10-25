# webgpu-mobile-device-loss-prefill

> Real cause of sandpie WebGPU device loss on Android — oversized prefill chunk trips the GPU watchdog, not sibling backends

_Design note written during development (sandpie session memory); names like `TabInfer*` refer to the pre-extraction identifiers, now `TabInfer*`._

**SUBGROUP-REDUCTION MOBILE BUG (2026-07-04, fix 738ee46).** Adreno 7xx (probe.html: subgroup width **128**, HAS shader-f16+subgroups, but maxStorageBufferBindingSize only **128 MiB**, bandwidth ~24GB/s ≈ Iris) ran Qwen3-0.6B but returned all-"!" (constant token = degenerate logits). Per-kernel conformance (docs/webgpu-engine-test.html "Run Qwen3 kernel tests") nailed it: EVERY subgroupAdd-reduction GEMV FAILED (gemv 2.2, gemvQ 4.7, gemvDP4A 0.4, gateUpSiluQ 0.7, lfm25 idxGemv 0.87); EVERY shared-mem-reduction kernel (matvecQ/matvecT/gemmQ) PASSED on the same GPU. Cause: `subgroupAdd` at `workgroup_size=32` on a 128-wide subgroup = partial subgroup, mis-reduced by the driver (Iris Xe subgroup=16 ≤ 32 → full subgroups → bug never fires). **FIX = wgReduceWGSL / wgReduce2WGSL (fused gate+up): plain shared-memory tree reduction, subgroup-free, drop `enable subgroups`.** Correct on any subgroup width / no-subgroups GPU. Converted qwen3 gemv/gemvQ/gemvDP4A/gateUpSiluQ(+DP4) + lfm25 gemvQ8/gemvDP4/idxGemv3; left matvecQ/matvecT (passed on-device, T>1). Verified Iris: 36/37 selfTests PASS (identical to subgroupAdd), 8B decode ~12 tok/s (no regression). LESSON: never rely on subgroupAdd over a partial subgroup; mobile subgroups are 64-128 wide. Other Adreno portability gates: 128MiB binding → Qwen3-1.7B lm_head (148MiB) needs banking; 0.6B (74MiB) & LFM 8B embed (125MiB) fit. app modules bumped: qwen3 v=126, lfm25 v=52, worker v=32, host v=38.

Mobile (Android Adreno) `VK_ERROR_DEVICE_LOST` in the [[sandpie-webgpu-engine]] was **NOT** the sibling local backends (wllama/transformersjs/litertlm were removed and the bug persisted). Real cause: the **prefill chunk size**.

Each `forwardChunk()` in webgpu-qwen35.js is ONE `vkQueueSubmit` (beginBatch→endBatch, and endBatch awaits the GPU → chunks run serially). A single too-long submit trips the GPU watchdog (Windows TDR ~2s; mobile Vulkan stricter AND far slower per token) → driver resets the device. Old code used a fixed 128-token chunk tuned for Iris Xe (~0.68s/chunk); on a slower Adreno that same chunk overruns the watchdog.

This is why the test harness "worked" on the phone but sandpie didn't — SAME engine, different prefill size: harness sent a ~10-token prompt (one tiny chunk); a real sandpie turn sends system + ~2400-tok tool preamble = ~20 chunks.

**Fix (commit 329e3ad, module v73):** adaptive prefill chunking in `forwardPrefill` — 24-token probe first (safe even at ~60ms/tok), measure actual per-token cost, size each subsequent submit to a 600ms budget (≥3× under a 2s watchdog), still capped by the existing long-context shrink. Free to measure (already awaiting each chunk); fast GPUs ramp back to 128 so no desktop throughput loss.

**Verify on phone:** test harness (docs/webgpu-engine-test.html) now has a "Stress prefill (~2k tok)" button to reproduce the large-prefill load in isolation. Decode loop (`_genBatch=8`, 1 tok/submit) was never the issue.

WebGPU Qwen3.5 is now the ONLY in-browser backend (commit b900d22). See [[sandpie-local-runtime-strategy]].
