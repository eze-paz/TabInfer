# reference_qwen35-zero-centered-rmsnorm

> Qwen3.5 RMSNorm is zero-centered ((1+w)·x̂) — and two debugging lessons from the GDN port

_Design note written during development (sandpie session memory); names like `TabInfer*` refer to the pre-extraction identifiers, now `TabInfer*`._

**Qwen3_5RMSNorm is ZERO-CENTERED**: checkpoint stores `weight − 1` (zeros-init) and forward multiplies by `(1.0 + weight)` — unlike Qwen3 and unlike Qwen3_5RMSNormGated (the GDN's per-head norm, ones-init, plain multiply, "norm BEFORE gate" — not Mamba2 order). Applies to input/post_attention layernorms, attention q/k_norm, final norm. The engine bakes +1 at load ([[project_sandpie-gdn-port]]). Symptom of missing it: all activations ~2× small per layer → residual stream ≈ raw embedding → tied lm_head argmax COPIES the last input token.

**Why:** two hard-won lessons. (1) Kernel self-tests vs your own CPU refs CANNOT catch a shared misread of the reference math — only a numeric cross-check against the true implementation (torch CPU, per-layer hidden-state stats → substage bisect → stopLayer sweep) localizes it. (2) JS silently coerces a GPUBuffer passed where a number is expected (`Uint32Array([buf])` → 0): after changing a wrapper's signature, grep EVERY call site — a stale 7-arg gdnDelta call set uniform T=0, no-opping the kernel and leaving stale data in its output buffer, which produced run-order-dependent symptoms that perfectly mimicked a GPU batching race (a day-long red herring: "flush>4 corrupts on gen-12lp" was THIS, not the driver).

**How to apply:** porting any new architecture → cross-check per-layer against transformers CPU before trusting self-tests; when a buffer mysteriously holds stale/frozen values, suspect a no-opped dispatch (uniform=0) before suspecting GPU ordering.
