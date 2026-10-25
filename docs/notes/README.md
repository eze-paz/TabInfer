# Design notes

Measured findings and decisions from building the engines, in the order you would want them.

- [webgpu-engine](webgpu-engine.md) — From-scratch hand-written WebGPU LLM engine (modules/webgpu-engine.js) — own WebLLM-equiv, raw WGSL kernels; Phase1 Qwen3-0.6B, Phase2 Qwen3.5-0.8B
- [qwen35-gated-deltanet-port](qwen35-gated-deltanet-port.md) — Gated DeltaNet WGSL port (Qwen3.5-0.8B) — Phases A-C SHIPPED and verified vs HF; Phase D (app wiring) pending
- [cpu-wasm-ternary-engine](cpu-wasm-ternary-engine.md) — Hand-rolled WASM CPU ternary inference kernel — the universal floor beneath the WebGPU engine; validated single-thread, threading next
- [ternary-wasm-phase0-lut-vs-q4k](ternary-wasm-phase0-lut-vs-q4k.md) — Native bitnet.cpp + llama.cpp builds already exist for the ternary-WASM-fork Phase-0 bench (i5-1235U); the decisive TL2/LUT number is still pending
- [grammar-constrained-decoding](grammar-constrained-decoding.md) — P4 shipped (a51cfbe) — grammar-constrained Hermes tool calls in the WebGPU engine; modules/webgpu-grammar.js char-PDA + logitMask kernel + forced-token injectio
- [tool-calling-goal](tool-calling-goal.md) — /goal = browser usable tool-calling LLM; chose reach-first → backend forced to wllama; first sprint = model+backend selection (small-model bake-off)
- [local-runtime-strategy](local-runtime-strategy.md) — Sandpie in-browser backends — wllama=compat floor, WebLLM/LiteRT=GPU speed; don't fork wllama; no big MoE in-browser
- [lfm25-moe-int3-resident](lfm25-moe-int3-resident.md) — 2026-07 strategy analysis — target model LFM2(.5)-8B-A1B MoE, shader roadmap (streaming decode attn + int8 KV fused), harness roadmap (wire zeroshot as Mindfram
- [n2-native-engine](n2-native-engine.md) — Native Rust ternary inference engine at Desktop/AI_Projects/N2-inference; replaces the WASM/WebGPU local path. AVX2 GEMV shipped at 26.15 GMAC/s; AVX-VNNI confi
- [webgpu-context-ceiling-32k](webgpu-context-ceiling-32k.md) — WebGPU Qwen ctx ceiling: 7168 cap REMOVED (streaming split-KV decode attn, cap now 16384), blank-turn bug + fix, path to 32K
- [webgpu-prefill-185s-to-26s](webgpu-prefill-185s-to-26s.md) — WebGPU engine prefill is catastrophically slow + crashes the iGPU; root causes + fix plan ported from llama.cpp ggml-webgpu
- [webgpu-mobile-device-loss](webgpu-mobile-device-loss.md) — Real cause of sandpie WebGPU device loss on Android — oversized prefill chunk trips the GPU watchdog, not sibling backends
- [webgpu-focused-tab-decode-collapse](webgpu-focused-tab-decode-collapse.md) — How local WebGPU inference was throttled to 2 tok/s (then 0.3 tok/s) by main-thread rendering work, and the multi-part fix
- [adreno-bind-cap-embedding-shard](adreno-bind-cap-embedding-shard.md) — Adreno all-\"!\" TRUE root cause = embed matrix (297MB) exceeds 128MB storage-buffer bind limit; shard it. Batching/DP4A/f16 were red herrings.
- [qwen3-dense-variants](qwen3-dense-variants.md) — Dense-Qwen3 WebGPU catalog + the trap that Qwen/Qwen3-2B is NOT a dense model
- [qwen35-zero-centered-rmsnorm](qwen35-zero-centered-rmsnorm.md) — Qwen3.5 RMSNorm is zero-centered ((1+w)·x̂) — and two debugging lessons from the GDN port
- [wgsl-fast-matmul-techniques](wgsl-fast-matmul-techniques.md) — How SOTA in-browser LLM engines do fast int4 matmul in WGSL, and which techniques are portable to our targets (Iris Xe / Adreno)
- [litert-mldrift-gpu-techniques](litert-mldrift-gpu-techniques.md) — What makes Google LiteRT-LM's in-browser GPU inference fast (MLDrift runtime) and which techniques are portable to our WebGPU Qwen engine
- [llamacpp-webgpu-decode-paths](llamacpp-webgpu-decode-paths.md) — What llama.cpp's ggml-webgpu actually does on this Iris Xe gen-12lp (decode matmul + attention + KV dtype) vs our engine — the 3 levers, and why our split-KV fa
- [llamacpp-bench-i5-1235u](llamacpp-bench-i5-1235u.md) — Native llama.cpp benchmark baseline on the user's i5-1235U/Iris Xe laptop — CPU+FA+8threads beats the iGPU; ~13.5 t/s on Qwen3.5-2B Q4
- [bonsai-ternary-eval-and-wgsl-port](bonsai-ternary-eval-and-wgsl-port.md) — Bonsai (PrismML ternary LLM): Q1_0 merged upstream, Q2_0_g64 runs on mainline llama.cpp CPU/Metal/Vulkan as of 2026-07
- [bitnet-cpp-windows-build](bitnet-cpp-windows-build.md) — How Microsoft bitnet.cpp builds + runs on the i5-1235U Windows box; BitNet 2B4T i2_s works + chats but is slow (~3.5 t/s) — TL2 is the fast x86 path
