# project_sandpie-tool-calling-goal

> /goal = browser usable tool-calling LLM; chose reach-first → backend forced to wllama; first sprint = model+backend selection (small-model bake-off)

_Design note written during development (sandpie session memory); names like `TabInfer*` refer to the pre-extraction identifiers, now `TabInfer*`._

The user set a standing /goal (2026-06-15): a **browser-based, usable tool-calling LLM**.

Strategic direction chosen by the user:
- **Reach-first** — must run on the widest device set incl. mobile/iOS (accepts weaker tool-calling + slower turns as the cost). This **forces the backend to wllama** (the only runtime with a pure CPU/WASM path that covers desktop + Android + iOS Safari; webllm/litertlm need WebGPU, transformersjs is an ONNX worker with weaker tools).
- **First sprint = model + backend selection** (NOT the eval harness / grammar / KV work I proposed first).

Reach-ladder model candidates (all standard GGUF + native tool template, CPU/mobile-sized, load in mainline llama.cpp):
- Qwen2.5-3B-Instruct Q4_K_M (~2 GB) — reach **ceiling**, best proven small tool-caller, already in wllama DEFAULT_MODELS.
- Qwen2.5-1.5B-Instruct Q4_K_M (~1 GB) — light **floor** for low-end/iOS, already in catalog.
- Qwen3-1.7B — contender (newer, /think + tools) but VERIFY it loads — [[reference_wllama-webgpu-crash]] notes Qwen3.5/3.6 don't load; plain Qwen3 (qwen3 arch) should.
- Llama-3.2-3B-Instruct — alternate tool template.

Consequence the user accepted: small + CPU = the weakest tool-callers AND slow (compounded by `cache_prompt:false` full re-prefill every agent round, wllama.js:398). So **grammar-constrained decoding matters MORE now, not less** — it's the lever that makes a small model a reliable tool-caller.

NATIVE TEMPLATE IS NON-NEGOTIABLE FOR SMALL MODELS (2026-06-26, commit cde00f5): local Qwen3 0.6B emitted tool/skill intent as PROSE ("Use the web_search skill to find information.") instead of `<tool_call>`, so the WebGPU agent loop saw no calls and stopped dead. Cause was NOT model size (0.6B DOES tool-call — user correctly pushed back on my "too weak" claim) — it was FORMAT DRIFT: `toolPreamble` (modules/webgpu-qwen3.js) hand-rolled a non-standard block ("You can call a tool by emitting a line…" + markdown bullet signatures). Small dense Qwen3 follow their RLHF distribution tightly and only fire when fed the EXACT native Hermes template. Fixed `toolPreamble` to emit Qwen3's verbatim block: `# Tools` / "You may call one or more functions…" / "You are provided with function signatures within <tools></tools> XML tags:" + full `{"type":"function","function":{…}}` JSON sigs inside `<tools>` / "For each function call, return a json object … within <tool_call></tool_call> XML tags:". Verified against the official Qwen3 chat template (Ollama template blob + qwen.readthedocs function_call docs). NOTE: our assistant `<tool_call>` (tcText) and tool-result `<tool_response>` (norm, role tool→user) renderings ALREADY matched Hermes — only the tool-DEFINITION preamble was wrong. LESSON: for any small in-browser model, match the model's trained chat template EXACTLY before blaming capability or reaching for grammar-constrained decoding. (Grammar-constrained decoding is still the next lever if the native template alone isn't reliable enough.) `encodeChat` (webgpu-qwen3.js ~L1668) does bare `<|im_start|>role\ncontent<|im_end|>` wrapping + the preamble appended to the system message — it does NOT apply the model's full jinja chat_template, so the tool block must be assembled by hand to match.

Deferred build order (my rec, not yet chosen): eval harness → grammar/schema-enforced tool calling → prefix KV-cache reuse. Don't: fork wllama, build a new engine, or chase exotic quants — LFM2 (VertaLily) + custom-quant (Bonsai/Ternary-Bonsai) dead-ends proved the in-browser runtime pins us to mainstream arch + standard quants. See [[project_sandpie-local-runtime-strategy]].
