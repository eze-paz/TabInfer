# reference_llamacpp-bench-i5-1235u

> Native llama.cpp benchmark baseline on the user's i5-1235U/Iris Xe laptop — CPU+FA+8threads beats the iGPU; ~13.5 t/s on Qwen3.5-2B Q4

_Design note written during development (sandpie session memory); names like `TabInfer*` refer to the pre-extraction identifiers, now `TabInfer*`._

Native llama.cpp (b9646, prebuilt Vulkan win-x64 build, installed at `C:\Users\aezequiel\llamacpp`, models in `\models`) benchmarked on the user's daily machine: 12th-gen **i5-1235U** (2 P-cores + 8 E-cores = 12 threads), **Iris Xe** iGPU (`uma=1, fp16=1, matrix cores: none`), ~16 GB shared RAM. Model: **Qwen3.5-2B Q4_K_M** (bartowski). Measured 2026-06-15 via `llama-bench`.

Findings (t/s):
- **CPU beats the Iris Xe iGPU.** Vulkan full-offload (`-ngl 99`) gen = 9.7 vs CPU 11.0; iGPU only ~10% ahead on prompt (137 vs 124). Token gen is bandwidth-bound and the iGPU shares the same RAM → not worth it. Use `-ngl 0`.
- **Flash attention = +60% gen** (8.4 → 13.5 t/s). Always `-fa on`.
- **8 threads optimal, NOT 12** — Alder Lake P/E + HT contention makes all-12 *slower* (9.4 vs 13.5 gen).
- Best stable config `-ngl 0 -t 8 -fa on`: ~130 t/s pp512, **~13.5 t/s tg128** for a 2B (±0.08). Rough extrapolation: 4B ≈ ~6, 9B ≈ ~3 t/s.
- 15 W chip → noticeable thermal-throttle variance across back-to-back runs; the FA-on/8-thread point was by far the most stable.

**Ternary (TQ2_0) validated 2026-06-15:** mainline b9646 loads & runs upstream `TQ2_0` ternary fine — started a 2nd `llama-server` on :8081 via `Start-Process` (`-ngl 0 -t 8 -fa on -c 4096`; launcher at `C:\Users\aezequiel\llamacpp\serve-trilm.ps1`). **TriLM 3.9B TQ2_0** (1.09 GB, Green-Sky conversion) gen **~24.7 t/s** (live /completion, n=80, single run) — ~2× the Qwen3.5-2B's ~13.5 t/s despite ~2× the params: ~2 bits/weight = less memory traffic + efficient TQ2_0 AVX2 kernel. (Live /completion ≠ `llama-bench` tg128 → ballpark, not clean apples-to-apples.) Upstream `TQ1_0`/`TQ2_0` ≠ PrismML Bonsai's confusingly-named `Q2_0 (g128)`, which needs their fork and won't load here (see [[reference_bonsai-8b-eval]]). TriLM = base model (fluent but loose/repetitive).

Confirms [[project_sandpie-local-runtime-strategy]] (Intel iGPU weak for LLMs; CPU path is the viable floor) and bounds [[project_sandpie-tool-calling-goal]] (wllama CPU ceiling on this hardware). The Qwen3.5 **35B-A3B** MoE that won't load in-browser ([[reference_wllama-webgpu-crash]]) is fundamentally a 16 GB size wall (~20 GB at Q4 — whole MoE must be resident even though only ~3B is active), not just an arch-support gap.
