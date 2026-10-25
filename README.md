# TabInfer

**LLM inference in your browser tab, from scratch.** Hand-written WebGPU (WGSL) compute
kernels and a WebAssembly CPU engine that run small language models entirely client-side:
no server does the inference, no TVM/MLC, no onnxruntime, no library. Weights download
once from Hugging Face and are cached in the origin.

This is the local-inference stack extracted (with its full history) from a private
agent client, where it served as the offline backend for tool-calling conversations.
It was built generic-first and measured on commodity hardware: an Intel Iris Xe iGPU,
Qualcomm Adreno phones, and 4-core laptop CPUs.

## What is in here

| Engine | Model(s) | How it works |
|---|---|---|
| `engine/` + `models/qwen3.js` | Qwen3-0.6B, Qwen3-1.7B (dense) | bf16 safetensors streamed in, quantized on-device to **int4 weights + int8 KV cache**, GQA attention with split-KV streaming decode, RoPE + QK-norm, SwiGLU. 32k context on an 8 GB laptop. |
| `models/lfm25.js` | LFM2.5-8B-A1B (MoE) | int3 resident experts streamed from a quant cache; gated short-conv + attention hybrid. ~15 tok/s decode on Iris Xe. |
| `models/qwen35-gdn.js` | Qwen3.5-0.8B | 18 of 24 layers are **gated DeltaNet** linear attention: from-scratch recurrence + causal conv1d kernels, partial mRoPE, zero-centered RMSNorm baked at load. Phases A–C shipped; not wired into the demo. |
| `cpu/` | Bonsai-1.7B (ternary) | Rust → WASM SIMD ternary GEMV, work-stolen across a worker pool over one shared `WebAssembly.Memory`; shared-KV attention. 13 tok/s on the i5-1235U below; the ~35 tok/s figure in the notes was measured on a different machine and browser, so treat it as a ceiling, not a promise. |
| `grammar/` | any | **Grammar-constrained decoding**: a character-level PDA compiled from the tool JSON schema drives a logit mask, so sub-1B models emit valid tool calls. |

Everything runs in a Web Worker so page rendering cannot starve the GPU submit loop
(measured 3.4× decode collapse when the loop ran on a focused main thread).

## Measured

Intel Core i5-1235U, Iris Xe (gen-12lp), Chrome, `shader-f16` + `subgroups`, no
subgroup-matrix. Re-measured 2026-10-25 from this repo's demo page.

| | prefill | decode |
|---|---|---|
| Qwen3-0.6B int4 / int8-KV, 4k ctx, 27-token prompt, 647 generated | 27 tok/s (1.0 s) | **29.5 tok/s** |
| Portable tiled f32 GEMM, 1024³ | | 121 GFLOP/s |
| Bonsai-1.7B ternary, CPU engine, 6 workers, headless Chromium | 12.3 tok/s | **13.2 tok/s** |

Earlier measurements and the reasoning behind each design decision are in
[`docs/notes/`](docs/notes/README.md): the 185 s → 26 s prefill fix, the 32k context
ceiling via split-KV decode + int8 KV, why the LUT ternary path wins prefill but loses
decode, the Adreno 128 MB bind cap, and the WGSL fast-matmul recipe.

## Run the demo

```bash
python demo/serve.py        # http://127.0.0.1:8765/demo/
npm i -D playwright && node tests/e2e-cpu-bonsai.mjs   # headless end-to-end check of the CPU engine
```

`demo/index.html` is a minimal chat page over the same `runConversation(config, emit)`
contract the agent client used. `demo/webgpu-engine-test.html` is the capability report,
GEMM self-test/bench, the 38 WGSL kernel self-tests and the GPU-timestamp profilers.

The dev server sends COOP/COEP: WebGPU does not need them, the CPU engine's
SharedArrayBuffer does. Any static host works for the GPU engines.

First run downloads the model (Qwen3-0.6B ≈ 1.1 GB bf16, quantized on arrival and cached;
subsequent loads come from Cache Storage).

## Layout

```
engine/   webgpu-engine.js   GPU init, capability report, buffer/pipeline helpers, WGSL templating, portable GEMM
          webgpu-worker.js   Web Worker host: routes a run to the right engine by model id
          webgpu-host.js     main-thread shim with the same API (window.TabInferQwen3), pluggable tool runner
models/   qwen3.js           dense Qwen3: tokenizer, safetensors loader, quant cache, kernels, generate, runConversation
          lfm25.js           LFM2.5 MoE
          qwen35-gdn.js      Qwen3.5 gated-DeltaNet port
grammar/  webgpu-grammar.js  char-PDA + logit mask + forced tokens for schema-valid tool calls
cpu/      cpu-bonsai.js      CPU engine host; kernel/ holds the Rust crate (src/lib.rs) and the worker-pool coordinator
demo/     index.html, webgpu-engine-test.html, serve.py; legacy/ = early CPU-engine test pages
docs/     HANDOFF-memory-cpubonsai.md and notes/ (design notes and measurements)
```

## Integration

```js
window.TabInferHost = { workerUrl: 'engine/webgpu-worker.js', toolRunner: async (name, args) => ({ result: '...' }) };
await window.TabInferQwen3.runConversation({
  provider: { endpoint: '0.6B', contextWindow: 4096 },
  messages: [{ role: 'user', content: 'hi' }], systemPrompt: '', tools: [], signal,
}, (ev) => { /* round_start | delta{content|reasoning|tool_calls} | tool_started | tool_result | usage | round_end | agent_done | error */ });
```

Model ids: `0.6B`, `1.7B` (Qwen3), `8B-A1B` (LFM2.5), `bonsai-1.7b-cpu`.

## Status and honest limits

- Qwen3 dense and the CPU engine are the production-tested paths. LFM2.5 ran in
  production for a while; the Qwen3.5 GDN port is complete through its measured phases
  but is not routed by the worker.
- No CUDA, no server backend: this is a browser engine. It beats generic WebGPU backends
  on iGPUs because every kernel was profiled on one; it will not beat a native stack on a
  discrete GPU.
- Chrome and Edge are the tested browsers. Firefox WebGPU lacks `shader-f16` on most
  devices; Safari is untested.
- Identifiers were renamed from `Sandpie*` to `TabInfer*` at extraction; comments that
  mention the original app are historical.

## Credits

The engine structure was inspired by the webml-community `gemma-4-webgpu-kernels` Space,
which proved a from-scratch WGSL engine can be small and fast on an iGPU. Models are by
Qwen (Alibaba), Liquid AI and the Bonsai authors; see their licenses.

## License

Apache-2.0. See [LICENSE](LICENSE).
