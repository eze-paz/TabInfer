// sandpie/modules/webgpu-lfm25.js — LFM2.5-8B-A1B (Lfm2MoeForCausalLM) port for the
// hand-written WebGPU engine. PHASE A: the two architecture-specific kernels + self-tests.
//
// WHY THIS MODEL: MoE with 8.3B total / 1.5B ACTIVE params — decode reads only the active
// experts, so it runs at ~1.5B-dense cost with far more capability; unified iGPU memory
// holds all experts with no swap penalty. Native tool calling. The engine work is friendly:
// 18/24 layers are double-gated SHORT CONV blocks whose entire decode state is H×(L-1)=
// 2048×2 floats (NO KV cache!), only 6 layers are GQA attention (our existing streaming
// int8-KV kernels apply, with 4× fewer layers to pay for), and experts ship as FUSED
// gate_up tensors matching our fused gateup kernel shape.
//
// ARCHITECTURE (from config.json + transformers modeling_lfm2_moe.py, verified 2026-07-02):
//   24 layers; layer_types: attention at [2,6,10,14,18,21], conv elsewhere
//   hidden 2048 · 32 q-heads × 64 · 8 kv-heads · rope_theta 5e6 · rms_norm_eps 1e-5
//   conv: in_proj [3H,H] → (B,C,x) chunks; y = C ⊙ depthwise_causal_conv3(B ⊙ x); out_proj [H,H]
//         no activation, no conv bias; decode state = last L-1 columns of B⊙x per channel
//   attn: q/k/v/out_proj + q_layernorm/k_layernorm (RMSNorm) + rotate_half RoPE
//   FFN:  layers 0-1 dense SwiGLU (I=7168); layers 2-23 MoE — router `gate` [32,H]:
//         sigmoid(logits) + expert_bias → top-4 → weights normalized to sum 1 (norm_topk_prob),
//         routed_scaling 1.0; experts fused gate_up_proj [32, 2·1792, H], down_proj [32, H, 1792]
//   embedding_norm final; TIED embeddings; vocab 128000 (lm_head GEMV → big-N gemvQ NR4 route)
//
// PHASE PLAN:
//   A (this file, now): shortConv kernel (batched T with carry state, double-buffered) +
//     router top-k kernel — the only NEW math. Self-tests vs CPU refs incl. state continuity.
//   B: loader (16.9GB bf16 safetensors → streamed int4 quant cache; ~4.7GB — mind the Cache
//     API quota), tokenizer (new BPE + chat template), projections/attention reuse.
//   C: forward graph + MoE dispatch (predicated expert GEMVs: all-expert grid early-exits on
//     the GPU-resident router output — no CPU readback), generate/runConversation contract.

const SandpieLfm25 = (function () {
  'use strict';

  const E = (typeof window !== 'undefined' ? window : self).SandpieWebGPU;
  const U = (typeof GPUBufferUsage !== 'undefined') ? GPUBufferUsage : { STORAGE: 0x80, COPY_DST: 0x8, COPY_SRC: 0x4, UNIFORM: 0x40, MAP_READ: 0x1 };

  const CONFIG = {
    numLayers: 24, hidden: 2048, nHeads: 32, nKvHeads: 8, headDim: 64,
    convL: 3, denseLayers: 2, denseIntermediate: 7168,
    nExperts: 32, topK: 4, expertIntermediate: 1792,
    vocab: 128000, ropeTheta: 5000000, rmsEps: 1e-5, tieEmbeddings: true,
    attnLayers: [2, 6, 10, 14, 18, 21],
    bos: 124894, eos: 124900,
  };

  let _uPool = [], _uIdx = 0;
  function uniformReset() { _uIdx = 0; }
  function uniform(arr) {
    let buf = _uPool[_uIdx];
    if (!buf) { buf = E.createBuffer(32, U.UNIFORM | U.COPY_DST, 'lu' + _uIdx); _uPool[_uIdx] = buf; }
    E.device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset || 0, arr.byteLength);
    _uIdx++;
    return buf;
  }

  // ============================================================
  // Kernel 1 — double-gated short causal conv (the LFM2 mixer).
  // BCx[T, 3H] (in_proj output; chunks B|C|x) + stateIn[H, L-1] →
  //   y[T,H] = C ⊙ conv3(B ⊙ x)   (depthwise, causal, weights w[H,L], no bias/activation)
  //   stateOut[H, L-1] = last L-1 columns of B⊙x  (the whole decode "cache"!)
  // State is DOUBLE-BUFFERED (read stateIn, write stateOut, caller swaps): with T>1 the
  // threads that read the carry (t < L-1) and the thread that writes it (t = T-1) are in
  // different workgroups — in-place update would race. useState=0 zero-pads (fresh prefill).
  // ============================================================
  const CONV_WGSL = `
enable f16;
struct P { T:u32, H:u32, L:u32, useState:u32 };
@group(0) @binding(0) var<storage, read>       bcx      : array<f32>;   // [T, 3H]
@group(0) @binding(1) var<storage, read>       w        : array<f16>;   // [H, L] depthwise taps
@group(0) @binding(2) var<storage, read>       stateIn  : array<f32>;   // [H, L-1]
@group(0) @binding(3) var<storage, read_write> stateOut : array<f32>;   // [H, L-1]
@group(0) @binding(4) var<storage, read_write> y        : array<f32>;   // [T, H]
@group(0) @binding(5) var<uniform>             p        : P;
fn bx_at(t:i32, h:u32, H:u32, Lm1:i32, useState:u32) -> f32 {
  if (t >= 0) { let r = u32(t)*3u*H; return bcx[r + h] * bcx[r + 2u*H + h]; }   // B[t,h]*x[t,h]
  if (useState == 0u) { return 0.0; }
  return stateIn[h*u32(Lm1) + u32(t + Lm1)];   // t in [-Lm1, -1] → carry column
}
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let H = p.H; let T = p.T; let L = p.L; let Lm1 = i32(L - 1u);
  let i = gid.x; if (i >= T*H) { return; }
  let t = i32(i / H); let h = i % H;
  var acc = 0.0;
  for (var k:u32 = 0u; k < L; k = k + 1u) {     // causal: y(t) = Σ_k w[h,k]·bx(t-(L-1)+k)
    acc = acc + f32(w[h*L + k]) * bx_at(t - Lm1 + i32(k), h, H, Lm1, p.useState);
  }
  let r = u32(t)*3u*H;
  y[u32(t)*H + h] = bcx[r + H + h] * acc;       // C[t,h] ⊙ conv
  if (u32(t) == T - 1u) {                        // last-token thread persists the carry
    for (var c:i32 = 0; c < Lm1; c = c + 1) {
      stateOut[h*u32(Lm1) + u32(c)] = bx_at(i32(T) - Lm1 + c, h, H, Lm1, p.useState);
    }
  }
}`;
  function shortConv(bcxBuf, wBuf, stateInBuf, stateOutBuf, yBuf, T, H, L, useState) {
    const p = uniform(new Uint32Array([T, H, L, useState ? 1 : 0]));
    const pipe = E.getPipeline('lfm25.conv', CONV_WGSL);
    const n = Math.ceil((T * H) / 64), gx = Math.min(n, 65535), gy = Math.ceil(n / gx);
    return E.dispatch(pipe, [bcxBuf, wBuf, stateInBuf, stateOutBuf, yBuf, p], [gx, gy, 1]);
  }

  // ============================================================
  // Kernel 2 — MoE router (Lfm2MoeTopKRouter). logits[T, nE] →
  //   scores = sigmoid(logits); sel = top-k of (scores + expert_bias)  [bias affects SELECTION only]
  //   weights = scores[sel] / (Σ scores[sel] + 1e-6)   (norm_topk_prob) × routed_scaling(1.0)
  // One workgroup per token; nE=32 is tiny — thread 0 does the serial top-k (selection sort,
  // k=4×32 comparisons). Output idx[T,K] (u32) + wt[T,K] (f32) stays GPU-resident: Phase C's
  // expert GEMVs read it directly (predicated early-exit), no CPU readback in the loop.
  // ============================================================
  const ROUTER_WGSL = `
struct P { T:u32, nE:u32, K:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       logits : array<f32>;   // [T, nE]
@group(0) @binding(1) var<storage, read>       bias   : array<f32>;   // [nE] expert_bias
@group(0) @binding(2) var<storage, read_write> idx    : array<u32>;   // [T, K]
@group(0) @binding(3) var<storage, read_write> wt     : array<f32>;   // [T, K]
@group(0) @binding(4) var<uniform>             p      : P;
var<workgroup> sc  : array<f32, 64>;   // sigmoid scores (nE ≤ 64)
var<workgroup> sel : array<f32, 64>;   // selection scores (score + bias)
@compute @workgroup_size(64,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let t = wg.x; if (t >= p.T) { return; }
  let e = lid.x;
  if (e < p.nE) {
    let s = 1.0 / (1.0 + exp(-logits[t*p.nE + e]));
    sc[e] = s; sel[e] = s + bias[e];
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var sum = 0.0;
    for (var k:u32 = 0u; k < p.K; k = k + 1u) {
      var best = 0u; var bv = -1.0e30;
      for (var j:u32 = 0u; j < p.nE; j = j + 1u) { if (sel[j] > bv) { bv = sel[j]; best = j; } }
      sel[best] = -2.0e30;
      idx[t*p.K + k] = best;
      wt[t*p.K + k] = sc[best];
      sum = sum + sc[best];
    }
    for (var k:u32 = 0u; k < p.K; k = k + 1u) { wt[t*p.K + k] = wt[t*p.K + k] / (sum + 1e-6); }
  }
}`;
  function router(logitsBuf, biasBuf, idxBuf, wtBuf, T, nE, K) {
    const p = uniform(new Uint32Array([T, nE, K, 0]));
    const pipe = E.getPipeline('lfm25.router', ROUTER_WGSL);
    return E.dispatch(pipe, [logitsBuf, biasBuf, idxBuf, wtBuf, p], [T, 1, 1]);
  }

  // ============================================================
  // Self-tests vs CPU references (no model download needed).
  // ============================================================
  const ST = () => U.STORAGE | U.COPY_DST | U.COPY_SRC;
  function f32buf(a) { return E.uploadF32(a instanceof Float32Array ? a : new Float32Array(a), ST()); }
  function f16buf(a) { const f = a instanceof Float32Array ? a : new Float32Array(a); const bits = E.f32ToF16(f); const b = E.createBuffer(f.length * 2, ST(), 'lw16'); E.device().queue.writeBuffer(b, 0, bits); return b; }
  const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

  async function selfTestKernels() {
    await E.init();
    const out = [];
    const check = (name, err, tol) => out.push({ name, ok: err < (tol || 1e-3), err });

    // CPU reference for the conv mixer
    const convRef = (bcx, w, state, T, H, L, useState) => {
      const y = new Float32Array(T * H), Lm1 = L - 1;
      const bx = (t, h) => t >= 0 ? bcx[t * 3 * H + h] * bcx[t * 3 * H + 2 * H + h] : (useState ? state[h * Lm1 + (t + Lm1)] : 0);
      for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) {
        let acc = 0; for (let k = 0; k < L; k++) acc += w[h * L + k] * bx(t - Lm1 + k, h);
        y[t * H + h] = bcx[t * 3 * H + H + h] * acc;
      }
      const st = new Float32Array(H * Lm1);
      for (let h = 0; h < H; h++) for (let c = 0; c < Lm1; c++) st[h * Lm1 + c] = bx(T - Lm1 + c, h);
      return { y, st };
    };

    // --- shortConv: fresh prefill (no state) ---
    {
      const T = 7, H = 96, L = 3, Lm1 = 2;
      const bcx = new Float32Array(T * 3 * H); for (let i = 0; i < bcx.length; i++) bcx[i] = Math.sin(i * 0.13);
      const w = new Float32Array(H * L); for (let i = 0; i < w.length; i++) w[i] = 0.3 + 0.5 * Math.cos(i * 0.07);
      const wR = (() => { const b = E.f32ToF16(w); const o = new Float32Array(w.length); for (let i = 0; i < w.length; i++) { o[i] = w[i]; } return o; })();
      const ref = convRef(bcx, w, null, T, H, L, 0);
      const bb = f32buf(bcx), wb = f16buf(w), s0 = f32buf(new Float32Array(H * Lm1)), s1 = E.createBuffer(H * Lm1 * 4, ST(), 's1'), yb = E.createBuffer(T * H * 4, ST(), 'y');
      await shortConv(bb, wb, s0, s1, yb, T, H, L, false);
      const gotY = await E.readF32(yb, T * H), gotS = await E.readF32(s1, H * Lm1);
      check('lfm25 conv (prefill)', Math.max(maxAbs(gotY, ref.y), maxAbs(gotS, ref.st)), 5e-3);   // f16 taps
      [bb, wb, s0, s1, yb].forEach(b => b.destroy());
    }
    // --- shortConv: state continuity — one T=6 pass ≡ 6 chained T=1 passes ---
    {
      const T = 6, H = 64, L = 3, Lm1 = 2;
      const bcx = new Float32Array(T * 3 * H); for (let i = 0; i < bcx.length; i++) bcx[i] = Math.cos(i * 0.11) * (1 + (i % 5) * 0.1);
      const w = new Float32Array(H * L); for (let i = 0; i < w.length; i++) w[i] = 0.2 + 0.4 * Math.sin(i * 0.05);
      const full = convRef(bcx, w, null, T, H, L, 0);
      const wb = f16buf(w);
      let sIn = f32buf(new Float32Array(H * Lm1)), sOut = E.createBuffer(H * Lm1 * 4, ST(), 'so');
      const gotY = new Float32Array(T * H);
      for (let t = 0; t < T; t++) {
        const one = bcx.slice(t * 3 * H, (t + 1) * 3 * H);
        const bb = f32buf(one), yb = E.createBuffer(H * 4, ST(), 'y1');
        await shortConv(bb, wb, sIn, sOut, yb, 1, H, L, t > 0);
        gotY.set(await E.readF32(yb, H), t * H);
        [bb, yb].forEach(b => b.destroy());
        const tmp = sIn; sIn = sOut; sOut = tmp;   // double-buffer swap (as the forward will)
      }
      check('lfm25 conv (decode state continuity)', maxAbs(gotY, full.y), 5e-3);
      [wb, sIn, sOut].forEach(b => b.destroy());
    }
    // --- router: sigmoid + bias-selected top-4 + normalized weights ---
    {
      const T = 3, nE = 32, K = 4;
      const logits = new Float32Array(T * nE); for (let i = 0; i < logits.length; i++) logits[i] = Math.sin(i * 0.37) * 3;
      const bias = new Float32Array(nE); for (let i = 0; i < nE; i++) bias[i] = Math.cos(i * 0.9) * 0.05;
      // CPU ref
      const refIdx = new Uint32Array(T * K), refW = new Float32Array(T * K);
      for (let t = 0; t < T; t++) {
        const sc = [], sel = [];
        for (let e = 0; e < nE; e++) { const s = 1 / (1 + Math.exp(-logits[t * nE + e])); sc.push(s); sel.push(s + bias[e]); }
        let sum = 0;
        for (let k = 0; k < K; k++) {
          let best = 0, bv = -Infinity;
          for (let j = 0; j < nE; j++) if (sel[j] > bv) { bv = sel[j]; best = j; }
          sel[best] = -Infinity; refIdx[t * K + k] = best; refW[t * K + k] = sc[best]; sum += sc[best];
        }
        for (let k = 0; k < K; k++) refW[t * K + k] /= (sum + 1e-6);
      }
      const lb = f32buf(logits), bb = f32buf(bias), ib = E.createBuffer(T * K * 4, ST(), 'ri'), wb2 = E.createBuffer(T * K * 4, ST(), 'rw');
      await router(lb, bb, ib, wb2, T, nE, K);
      const gotIraw = await E.readF32(ib, T * K); const gotI = new Uint32Array(gotIraw.buffer);
      const gotW = await E.readF32(wb2, T * K);
      let idxErr = 0; for (let i = 0; i < T * K; i++) if (gotI[i] !== refIdx[i]) idxErr++;
      check('lfm25 router (top-4 indices)', idxErr);
      check('lfm25 router (normalized weights)', maxAbs(gotW, refW), 1e-5);
      [lb, bb, ib, wb2].forEach(b => b.destroy());
    }
    // --- MoE block: router → 2 experts → weighted combine (full dispatch composition) ---
    {
      const H = 64, nE = 8, K = 2, eI = 32;
      const rnd = (n, s) => { const a = new Float32Array(n); let x = s; for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; a[i] = (x / 0x3fffffff - 1) * 0.5; } return a; };
      // f32 → int4 rec (+ dequantized copy so the CPU ref matches GPU dequant exactly)
      const mk4 = (f32, N, Kd) => {
        const u16 = new Uint16Array(N * Kd), t = new Float32Array(1), ti = new Uint32Array(t.buffer);
        for (let i = 0; i < N * Kd; i++) { t[0] = f32[i]; u16[i] = ti[0] >>> 16; }
        const { pack, scales } = _quantInt4(u16, N, Kd);
        const pb = E.createBuffer(pack.byteLength, ST(), 'tp'), sb = E.createBuffer(scales.byteLength, ST(), 'ts');
        E.device().queue.writeBuffer(pb, 0, pack); E.device().queue.writeBuffer(sb, 0, scales);
        const deq = new Float32Array(N * Kd);
        for (let n = 0; n < N; n++) for (let g = 0; g < Kd / QGROUP; g++) { const sc = _f16ToF32s(scales[n * (Kd / QGROUP) + g]); for (let j = 0; j < QGROUP; j++) { const kk = g * QGROUP + j; const q = (pack[n * (Kd / 8) + (kk >> 3)] >>> (4 * (kk & 7))) & 0xF; deq[n * Kd + kk] = (q - 8) * sc; } }
        return { rec: { pack: pb, scales: sb, N, K: Kd, int4: true }, deq };
      };
      const normed = rnd(H, 7);
      const gate = mk4(rnd(nE * H, 11), nE, H);
      const bias = rnd(nE, 13);
      const experts = []; for (let e = 0; e < nE; e++) experts.push({ w1: mk4(rnd(eI * H, e * 3 + 1), eI, H), w3: mk4(rnd(eI * H, e * 3 + 2), eI, H), w2: mk4(rnd(H * eI, e * 3 + 3), H, eI) });
      const nb = f32buf(normed), bb = f32buf(bias);
      const rl = E.createBuffer(nE * 4, ST(), 'rl'), ri = E.createBuffer(K * 4, ST(), 'ri'), rw = E.createBuffer(K * 4, ST(), 'rw');
      const g_ = E.createBuffer(eI * 4, ST(), 'g'), u_ = E.createBuffer(eI * 4, ST(), 'u'), sw = E.createBuffer(eI * 4, ST(), 's'), eo = E.createBuffer(H * 4, ST(), 'eo'), xo = f32buf(new Float32Array(H));
      await matvecQ(nb, gate.rec, rl, 1, nE, H);
      await router(rl, bb, ri, rw, 1, nE, K);
      await E.device().queue.onSubmittedWorkDone();
      const idx = new Uint32Array((await E.readF32(ri, K)).buffer), wt = await E.readF32(rw, K);
      for (let k = 0; k < K; k++) { const e = idx[k]; await matvecQ(nb, experts[e].w1.rec, g_, 1, eI, H); await matvecQ(nb, experts[e].w3.rec, u_, 1, eI, H); await swiglu(g_, u_, sw, eI); await matvecQ(sw, experts[e].w2.rec, eo, 1, H, eI); await axpy(xo, eo, H, wt[k]); }
      const got = await E.readF32(xo, H);
      // CPU ref from the GPU's routing (router selection itself is covered by the router test)
      const ref = new Float32Array(H);
      for (let k = 0; k < K; k++) {
        const e = idx[k], a = new Float32Array(eI);
        for (let n = 0; n < eI; n++) { let d1 = 0, d3 = 0; for (let i = 0; i < H; i++) { d1 += normed[i] * experts[e].w1.deq[n * H + i]; d3 += normed[i] * experts[e].w3.deq[n * H + i]; } a[n] = (d1 / (1 + Math.exp(-d1))) * d3; }
        for (let h = 0; h < H; h++) { let o = 0; for (let n = 0; n < eI; n++) o += a[n] * experts[e].w2.deq[h * eI + n]; ref[h] += wt[k] * o; }
      }
      check('lfm25 MoE block (route+experts+combine)', maxAbs(got, ref), 3e-3);
      [nb, bb, rl, ri, rw, g_, u_, sw, eo, xo].forEach(b => b.destroy());
      [gate, ...experts.flatMap(e => [e.w1, e.w3, e.w2])].forEach(o => { o.rec.pack.destroy(); o.rec.scales.destroy(); });
    }
    // --- gemvDP4 (int8 activation × int4 weight) vs gemvQ8 (f32 activation) ---
    {
      const N = 512, K = 1024;
      const rnd = (n, s) => { const a = new Float32Array(n); let x = s; for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; a[i] = (x / 0x3fffffff - 1) * 0.5; } return a; };
      const xf = rnd(K, 3);
      const u16 = new Uint16Array(N * K), t = new Float32Array(1), ti = new Uint32Array(t.buffer);
      const wf = rnd(N * K, 9); for (let i = 0; i < N * K; i++) { t[0] = wf[i]; u16[i] = ti[0] >>> 16; }
      const { pack, scales } = _quantInt4(u16, N, K);
      const pb = E.createBuffer(pack.byteLength, ST(), 'p'), sb = E.createBuffer(scales.byteLength, ST(), 's');
      E.device().queue.writeBuffer(pb, 0, pack); E.device().queue.writeBuffer(sb, 0, scales);
      const rec = { pack: pb, scales: sb };
      const xb = f32buf(xf), y1 = E.createBuffer(N * 4, ST(), 'y1'), y2 = E.createBuffer(N * 4, ST(), 'y2');
      await gemvQ8(xb, rec, y1, N, K, false);
      quantAct(xb, K); await gemvDP4(rec, y2, N, K, false);
      await E.device().queue.onSubmittedWorkDone();
      const a = await E.readF32(y1, N), b = await E.readF32(y2, N);
      let mx = 0, rf = 1e-9; for (let i = 0; i < N; i++) { mx = Math.max(mx, Math.abs(a[i] - b[i])); rf = Math.max(rf, Math.abs(a[i])); }
      check('lfm25 gemvDP4 vs gemvQ8 (int8 act)', mx / rf, 3e-2);
      [pb, sb, xb, y1, y2].forEach(x => x.destroy());
    }
    // --- GPU-resident indexed MoE dispatch (packed experts, no readback) vs CPU ref ---
    {
      const H = 64, nE = 8, K = 2, eI = 32, wpr = (Kd) => Kd / 8, gpr = (Kd) => Kd / QGROUP;
      const rnd = (n, s) => { const a = new Float32Array(n); let x = s; for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; a[i] = (x / 0x3fffffff - 1) * 0.5; } return a; };
      // f32 -> {pack, scales, deq} CPU arrays (for packing + dequant ref)
      const q4 = (f32, N, Kd) => {
        const u16 = new Uint16Array(N * Kd), t = new Float32Array(1), ti = new Uint32Array(t.buffer);
        for (let i = 0; i < N * Kd; i++) { t[0] = f32[i]; u16[i] = ti[0] >>> 16; }
        const { pack, scales } = _quantInt4(u16, N, Kd);
        const deq = new Float32Array(N * Kd);
        for (let n = 0; n < N; n++) for (let g = 0; g < gpr(Kd); g++) { const sc = _f16ToF32s(scales[n * gpr(Kd) + g]); for (let j = 0; j < QGROUP; j++) { const kk = g * QGROUP + j; const qv = (pack[n * wpr(Kd) + (kk >> 3)] >>> (4 * (kk & 7))) & 0xF; deq[n * Kd + kk] = (qv - 8) * sc; } }
        return { pack, scales, deq };
      };
      const mkBuf = (arr) => { const b = E.createBuffer(arr.byteLength, ST(), 'pk'); E.device().queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength); return b; };
      const normed = rnd(H, 7), bias = rnd(nE, 13);
      const gate = q4(rnd(nE * H, 11), nE, H);
      // per-expert weights + CPU-concatenated packed buffers
      const ew = { w1: [], w3: [], w2: [] };
      const packed = {};
      for (const [wt, rows, Kd] of [['w1', eI, H], ['w3', eI, H], ['w2', H, eI]]) {
        const pk = new Uint32Array(nE * rows * wpr(Kd)), scc = new Uint16Array(nE * rows * gpr(Kd));
        for (let e = 0; e < nE; e++) { const q = q4(rnd(rows * Kd, e * 7 + (wt === 'w1' ? 1 : wt === 'w3' ? 2 : 3)), rows, Kd); ew[wt].push(q); pk.set(q.pack, e * rows * wpr(Kd)); scc.set(q.scales, e * rows * gpr(Kd)); }
        packed[wt] = { pack: mkBuf(pk), scales: mkBuf(scc), N: nE * rows, K: Kd };
      }
      const nb = f32buf(normed), bb = f32buf(bias), gb = mkBuf(gate.pack), gs = mkBuf(gate.scales);
      const rl = E.createBuffer(nE * 4, ST(), 'rl'), ri = E.createBuffer(K * 4, ST(), 'ri'), rw = E.createBuffer(K * 4, ST(), 'rw');
      const mg = E.createBuffer(K * eI * 4, ST(), 'mg'), mu = E.createBuffer(K * eI * 4, ST(), 'mu'), ma = E.createBuffer(K * eI * 4, ST(), 'ma'), mo = E.createBuffer(K * H * 4, ST(), 'mo'), xo = f32buf(new Float32Array(H));
      await matvecQ(nb, { pack: gb, scales: gs }, rl, 1, nE, H);
      await router(rl, bb, ri, rw, 1, nE, K);
      await idxGemv(nb, packed.w1, ri, mg, K, eI, H, 0);
      await idxGemv(nb, packed.w3, ri, mu, K, eI, H, 0);
      await swiglu(mg, mu, ma, K * eI);
      await idxGemv(ma, packed.w2, ri, mo, K, H, eI, eI);
      await moeCombine(xo, mo, rw, H, K);
      await E.device().queue.onSubmittedWorkDone();
      const got = await E.readF32(xo, H);
      const idx = new Uint32Array((await E.readF32(ri, K)).buffer), wt = await E.readF32(rw, K);
      const ref = new Float32Array(H);
      for (let k = 0; k < K; k++) {
        const e = idx[k], a = new Float32Array(eI);
        for (let n = 0; n < eI; n++) { let d1 = 0, d3 = 0; for (let i = 0; i < H; i++) { d1 += normed[i] * ew.w1[e].deq[n * H + i]; d3 += normed[i] * ew.w3[e].deq[n * H + i]; } a[n] = (d1 / (1 + Math.exp(-d1))) * d3; }
        for (let h = 0; h < H; h++) { let o = 0; for (let n = 0; n < eI; n++) o += a[n] * ew.w2[e].deq[h * eI + n]; ref[h] += wt[k] * o; }
      }
      check('lfm25 MoE indexed dispatch (packed, no readback)', maxAbs(got, ref), 3e-3);
      [nb, bb, gb, gs, rl, ri, rw, mg, mu, ma, mo, xo, packed.w1.pack, packed.w1.scales, packed.w3.pack, packed.w3.scales, packed.w2.pack, packed.w2.scales].forEach(b => b.destroy());
    }
    return out;
  }

  // ============================================================
  // PHASE B — tokenizer + streaming weight loader.
  // ============================================================
  const MODELS = {
    // Bring-up model: same generation/architecture family, dense FFN, small download.
    '350M': {
      root: 'https://huggingface.co/LiquidAI/LFM2.5-350M/resolve/main/',
      cfg: { numLayers: 16, hidden: 1024, nHeads: 16, nKvHeads: 8, headDim: 64, convL: 3, vocab: 65536, ropeTheta: 1000000, rmsEps: 1e-5, attnLayers: [2, 5, 8, 10, 12, 14], moe: false, bos: 1, eos: 7 },
    },
    // The target. moe fields per config.json (verified 2026-07-02).
    '8B-A1B': {
      root: 'https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/resolve/main/',
      cfg: { numLayers: 24, hidden: 2048, nHeads: 32, nKvHeads: 8, headDim: 64, convL: 3, vocab: 128000, ropeTheta: 5000000, rmsEps: 1e-5, attnLayers: [2, 6, 10, 14, 18, 21], moe: true, nExperts: 32, topK: 4, expertI: 1792, denseLayers: 2, denseI: 7168, bos: 124894, eos: 124900 },
    },
  };
  let _variant = null, _cfg = null, _weights = null, _loaded = false;

  // ---- tokenizer: byte-level BPE with the pre_tokenizer regex extracted from
  // tokenizer.json at load (LFM2's split pattern differs from Qwen's; the (?i:…)
  // contraction group — the one construct JS regexes lack — is expanded manually).
  function buildByteMaps() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const byteEnc = new Array(256), byteDec = {};
    for (let i = 0; i < bs.length; i++) { const ch = String.fromCharCode(cs[i]); byteEnc[bs[i]] = ch; byteDec[ch] = bs[i]; }
    return { byteEnc, byteDec };
  }
  function _jsRegexFrom(hfPattern) {
    // The only HF-regex constructs JS lacks in practice: (?i:…) groups. Expand the
    // standard contraction group; anything else unsupported → fall back to GPT-4-style.
    let p = hfPattern.replace(/\(\?i:('s\|'t\|'re\|'ve\|'m\|'ll\|'d)\)/i,
      "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])");
    try { return new RegExp(p, 'gu'); } catch (_) {
      return /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
    }
  }
  const TOK = (function () {
    let vocab = null, idToTok = null, bpeRanks = null, byteEnc = null, byteDec = null, ready = false, pretokRe = null;
    let imStart = -1, imEnd = -1, bosId = -1;
    const _addedIds = new Set();
    const enc = new TextEncoder(), dec = new TextDecoder();
    async function load(root) {
      if (ready) return;
      const j = await (await fetch(root + 'tokenizer.json')).json();
      vocab = j.model.vocab;
      idToTok = {}; for (const k in vocab) idToTok[vocab[k]] = k;
      for (const a of (j.added_tokens || [])) { vocab[a.content] = a.id; idToTok[a.id] = a.content; _addedIds.add(a.id); }
      bpeRanks = new Map();
      const merges = j.model.merges || [];
      for (let i = 0; i < merges.length; i++) { const m = merges[i]; bpeRanks.set(Array.isArray(m) ? (m[0] + ' ' + m[1]) : m, i); }
      // pre_tokenizer: find the Split regex wherever it sits (top-level or Sequence)
      let pat = null;
      const scan = (pt) => { if (!pt) return; if (pt.pattern && pt.pattern.Regex) pat = pt.pattern.Regex; (pt.pretokenizers || []).forEach(scan); };
      scan(j.pre_tokenizer);
      pretokRe = pat ? _jsRegexFrom(pat) : _jsRegexFrom('');
      ({ byteEnc, byteDec } = buildByteMaps());
      imStart = vocab['<|im_start|>'] ?? -1; imEnd = vocab['<|im_end|>'] ?? -1;
      bosId = vocab['<|startoftext|>'] ?? _cfg.bos;
      ready = true;
    }
    function bpe(piece) {
      let word = Array.from(piece);
      if (word.length < 2) return word;
      for (;;) {
        let bestRank = Infinity, bestI = -1;
        for (let i = 0; i < word.length - 1; i++) { const r = bpeRanks.get(word[i] + ' ' + word[i + 1]); if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; } }
        if (bestI < 0) break;
        word = word.slice(0, bestI).concat(word[bestI] + word[bestI + 1], word.slice(bestI + 2));
      }
      return word;
    }
    function encodeText(text) {
      const ids = [];
      for (const piece of (text.match(pretokRe) || [])) {
        const bytes = enc.encode(piece);
        let s = ''; for (const b of bytes) s += byteEnc[b];
        for (const sub of bpe(s)) { const id = vocab[sub]; if (id !== undefined) ids.push(id); }
      }
      return ids;
    }
    function encodeChat(messages, { addGenerationPrompt = true } = {}) {
      const ids = [bosId];   // <|startoftext|>
      for (const m of messages) {
        ids.push(imStart); ids.push(...encodeText(m.role + '\n' + (typeof m.content === 'string' ? m.content : ''))); ids.push(imEnd); ids.push(...encodeText('\n'));
      }
      if (addGenerationPrompt) { ids.push(imStart); ids.push(...encodeText('assistant\n')); }
      return ids;
    }
    function decode(ids) {
      let s = ''; for (const id of ids) { const t = idToTok[id]; if (t !== undefined) s += t; }
      const bytes = []; for (const ch of s) { const b = byteDec[ch]; if (b !== undefined) bytes.push(b); }
      return dec.decode(new Uint8Array(bytes));
    }
    return { load, encodeText, encodeChat, decode, isReady: () => ready, specialIds: () => _addedIds, imEnd: () => imEnd };
  })();

  // ---- streaming single-pass loader: safetensors header via Range request, then one
  // sequential body stream; each tensor is quantized/converted the moment its bytes are
  // complete and the raw bytes are DROPPED — peak host RAM ≈ the largest single tensor
  // (embed: ~525MB on the 8B) regardless of file size (16.9GB single-file works).
  const QGROUP = 32;
  const _f32a = new Float32Array(1), _u32a = new Uint32Array(_f32a.buffer);
  function _f32ToF16(v) { _f32a[0] = v; const x = _u32a[0]; const sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff, mant = x & 0x7fffff; if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00); let e = exp - 127 + 15; if (e >= 31) return sign | 0x7c00; if (e <= 0) { if (e < -10) return sign; const m = mant | 0x800000, sh = 14 - e; let h = m >>> sh; if ((m >>> (sh - 1)) & 1) h += 1; return sign | h; } let h = (e << 10) | (mant >>> 13); if ((mant >>> 12) & 1) h += 1; return sign | h; }
  function _f16ToF32s(h) { const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff; if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024); if (e === 31) return f ? NaN : (s ? -Infinity : Infinity); return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024); }
  function _bf16ToF16bits(u16) { const out = new Uint16Array(u16.length); const t = new Float32Array(1), ti = new Uint32Array(t.buffer); for (let i = 0; i < u16.length; i++) { ti[0] = u16[i] << 16; out[i] = _f32ToF16(t[0]); } return out; }
  // group-wise symmetric int4 along K; rows = product of leading dims (3D expert tensors
  // [E,N,K] quantize as (E·N, K) — per-expert rows are contiguous, so Phase C's expert
  // GEMV just offsets rows by e·N).
  function _quantInt4(u16, rows, K) {
    const wpr = K / 8, gpr = K / QGROUP;
    const pack = new Uint32Array(rows * wpr), scales = new Uint16Array(rows * gpr);
    const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
    for (let n = 0; n < rows; n++) {
      const rU = n * K, rP = n * wpr, rS = n * gpr;
      for (let g = 0; g < gpr; g++) {
        let maxabs = 0;
        for (let j = 0; j < QGROUP; j++) { ti[0] = u16[rU + g * QGROUP + j] << 16; const v = Math.abs(t[0]); if (v > maxabs) maxabs = v; }
        const sBits = _f32ToF16(maxabs > 0 ? maxabs / 7 : 1e-8); scales[rS + g] = sBits;
        const inv = 1 / _f16ToF32s(sBits);
        for (let j = 0; j < QGROUP; j++) {
          const k = g * QGROUP + j; ti[0] = u16[rU + k] << 16;
          let q = Math.round(t[0] * inv); if (q < -8) q = -8; else if (q > 7) q = 7;
          pack[rP + (k >> 3)] |= ((q + 8) & 0xF) << (4 * (k & 7));
        }
      }
    }
    return { pack, scales };
  }
  // Quant plan by tensor name (lfm2 + lfm2_moe families).
  const _isInt4 = (name) => /(_proj|\.w[123]|feed_forward\.gate)\.weight$/.test(name);   // router gate int4 too (reuses matvecQ)
  const _isF32 = (name) => /expert_bias/.test(name);
  const _isExpert = (name) => /\.experts\.\d+\./.test(name);
  // EXPERT STREAMING: the 8B has ~3.9GB of experts (only 12.5% touched/token) but they can't
  // all fit resident on a 15GB shared-memory iGPU. So experts live ON DISK (Cache Storage, one
  // entry per expert tensor) and only the ~4 the router picks per layer are fetched into a small
  // per-token LRU of GPU slots. Non-expert weights (~0.9GB) stay resident. Enabled for MoE.
  let _streamExperts = false;
  let _expertCatalog = {};   // name -> { N, K } for experts on disk
  let _qcache = null;        // open Cache Storage handle for on-demand expert reads
  // LOCAL FOLDER STORE: an optional user-picked directory on the real filesystem holding the
  // same quantized artifact (chunks + manifest + per-expert files). Eviction-proof, portable,
  // and never re-downloaded. When connected + granted, it takes priority over Cache Storage.
  let _fsRoot = null;        // persisted FileSystemDirectoryHandle (the folder the user picked)
  let _fsMode = false;       // current model was loaded from the folder → on-demand reads hit disk
  let _fsExpertDir = null;   // cached <variant>/e/ handle for on-demand expert reads

  async function _uploadTensor(name, info, raw, sink) {
    const numel = info.shape.reduce((a, b) => a * b, 1);
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, numel);
    const put = (buf, arr) => E.device().queue.writeBuffer(buf, 0, arr.buffer === undefined ? arr : new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    // Streaming: an expert tensor never touches the GPU at load — quantize and write it to its
    // own Cache Storage entry; ensureExpert() fetches it on demand during decode.
    if (_streamExperts && _isExpert(name)) {
      const K = info.shape[info.shape.length - 1], rows = numel / K;
      const { pack, scales } = _quantInt4(u16, rows, K);
      if (sink) await sink.addExpert(name, pack, scales, { N: rows, K });
      _expertCatalog[name] = { N: rows, K };
      return;
    }
    if (_isInt4(name) && info.shape.length >= 2 && (info.shape[info.shape.length - 1] % QGROUP) === 0) {
      const K = info.shape[info.shape.length - 1], rows = numel / K;
      const { pack, scales } = _quantInt4(u16, rows, K);
      const packBuf = E.createBuffer(pack.byteLength, ST(), name + '.pack');
      const scBuf = E.createBuffer(scales.byteLength, ST(), name + '.sc');
      put(packBuf, pack); put(scBuf, scales);
      _weights[name] = { pack: packBuf, scales: scBuf, N: rows, K, int4: true, shape: info.shape };
      if (sink) { await sink.add(name, 'pack', pack, { kind: 'int4', shape: info.shape, N: rows, K }); await sink.add(name, 'scales', scales, { kind: 'int4', shape: info.shape, N: rows, K }); }
    } else if (_isF32(name)) {
      const f32 = new Float32Array(numel); const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
      for (let i = 0; i < numel; i++) { ti[0] = u16[i] << 16; f32[i] = t[0]; }
      const buf = E.createBuffer(f32.byteLength, ST(), name);
      put(buf, f32);
      _weights[name] = { buf, f32: true, shape: info.shape };
      if (sink) await sink.add(name, 'buf', f32, { kind: 'f32', shape: info.shape });
    } else {
      const bits = info.dtype === 'BF16' ? _bf16ToF16bits(u16) : u16;
      const buf = E.createBuffer(numel * 2, ST(), name);
      put(buf, bits);
      _weights[name] = { buf, shape: info.shape };
      if (sink) await sink.add(name, 'buf', bits, { kind: 'f16', shape: info.shape });
    }
    // TIED lm_head: the embedding also serves the vocab GEMV — make its int4 twin now,
    // while the raw bf16 is still in hand (it is never resident again after this call).
    if (name === 'model.embed_tokens.weight' && !_weights['lm_head.weight']) {
      const K = info.shape[1], rows = info.shape[0];
      const { pack, scales } = _quantInt4(u16, rows, K);
      const packBuf = E.createBuffer(pack.byteLength, ST(), 'lm_head.pack');
      const scBuf = E.createBuffer(scales.byteLength, ST(), 'lm_head.sc');
      put(packBuf, pack); put(scBuf, scales);
      _weights['lm_head.weight'] = { pack: packBuf, scales: scBuf, N: rows, K, int4: true, shape: info.shape };
      if (sink) { await sink.add('lm_head.weight', 'pack', pack, { kind: 'int4', shape: info.shape, N: rows, K }); await sink.add('lm_head.weight', 'scales', scales, { kind: 'int4', shape: info.shape, N: rows, K }); }
    }
  }

  // ---- persistent quantized-weights cache -------------------------------------------
  // Same design as the qwen3 engine's (the pattern the user was right to insist on):
  // GPU-READY bytes stream into fixed 64MiB Cache Storage chunks; a manifest of 4-byte-
  // aligned segments is written LAST as the commit point (a torn write = cache miss, never
  // corruption). Reload: allocate buffers from the manifest and writeBuffer chunk slices
  // straight into them — no download, no re-quantize, peak host RAM = one chunk.
  // Namespaced under /lfm25/ so it coexists with the qwen3 cache in the same Cache bucket.
  const QC_NAME = 'sandpie-webgpu-quant';
  const QC_VER = 2;   // v2: experts stored as individual cache entries (streaming), not in chunks
  const QC_CHUNK = 64 * 1024 * 1024;
  const _qcUrl = (variant, part) => 'https://sandpie.quant/lfm25/v' + QC_VER + '/' + variant + '/' + part;
  function _makeSink(variant) {
    let cache = null, buf = new Uint8Array(QC_CHUNK), used = 0, chunkIdx = 0, globalOff = 0, dead = false;
    const segs = [], expertTensors = [];
    const flush = async () => {
      if (!used) return;
      await cache.put(_qcUrl(variant, 'c' + chunkIdx), new Response(buf.subarray(0, used)));
      chunkIdx++; buf = new Uint8Array(QC_CHUNK); used = 0;
    };
    return {
      async open() {
        try { cache = await caches.open(QC_NAME); await cache.delete(_qcUrl(variant, 'manifest')); return true; } catch (_) { dead = true; return false; }
      },
      // one Cache Storage entry per expert tensor = concat(pack, scales) — read on demand
      async addExpert(name, pack, scales, meta) {
        if (dead) return;
        try {
          const p = new Uint8Array(pack.buffer, pack.byteOffset, pack.byteLength);
          const s = new Uint8Array(scales.buffer, scales.byteOffset, scales.byteLength);
          const body = new Uint8Array(p.byteLength + s.byteLength);
          body.set(p, 0); body.set(s, p.byteLength);
          await cache.put(_qcUrl(variant, 'e/' + name), new Response(body));
          expertTensors.push({ name, packLen: p.byteLength, scLen: s.byteLength, N: meta.N, K: meta.K });
        } catch (e) { dead = true; try { console.warn('[lfm25] expert cache write failed', e); } catch (_) {} }
      },
      async add(name, part, arr, meta) {
        if (dead) return;
        try {
          const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
          const pad = (4 - (globalOff % 4)) % 4;                       // 4-align every segment start
          for (let p = 0; p < pad; p++) { if (used === QC_CHUNK) await flush(); buf[used++] = 0; globalOff++; }
          const lenPad = (4 - (bytes.byteLength % 4)) % 4;             // …and its length (writeBuffer size must be %4)
          segs.push({ name, part, off: globalOff, len: bytes.byteLength + lenPad, ...meta });
          let src = 0;
          while (src < bytes.byteLength) {
            if (used === QC_CHUNK) await flush();
            const n = Math.min(QC_CHUNK - used, bytes.byteLength - src);
            buf.set(bytes.subarray(src, src + n), used); used += n; src += n; globalOff += n;
          }
          for (let p = 0; p < lenPad; p++) { if (used === QC_CHUNK) await flush(); buf[used++] = 0; globalOff++; }
        } catch (e) { dead = true; try { console.warn('[lfm25] quant-cache write failed (quota?) — continuing uncached', e); } catch (_) {} }
      },
      async finish() {
        if (dead) return false;
        try {
          await flush();
          const manifest = { ver: QC_VER, qgroup: QGROUP, chunkSize: QC_CHUNK, nChunks: chunkIdx, totalBytes: globalOff, segs, expertTensors };
          await cache.put(_qcUrl(variant, 'manifest'), new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } }));
          return true;
        } catch (_) { return false; }
      },
    };
  }
  // Delete lfm25 cache entries from a PRIOR QC_VER (format change) so the old + new copies
  // don't coexist and exceed quota (v1 8B + v2 8B ≈ 10.4GB).
  async function _cleanOldCache() {
    try {
      if (!_qcache) return;
      const cur = '/lfm25/v' + QC_VER + '/';
      const keys = await _qcache.keys();
      let n = 0;
      for (const req of keys) { if (req.url.indexOf('/lfm25/') >= 0 && req.url.indexOf(cur) < 0) { await _qcache.delete(req); n++; } }
      if (n) try { console.log('[lfm25] cleaned ' + n + ' stale-version cache entries'); } catch (_) {}
    } catch (_) {}
  }
  async function _readQuantCache(variant, onProgress) {
    let cache; try { cache = await caches.open(QC_NAME); } catch (_) { return false; }
    const mResp = await cache.match(_qcUrl(variant, 'manifest'));
    if (!mResp) return false;
    let m; try { m = await mResp.json(); } catch (_) { return false; }
    if (!m || m.ver !== QC_VER || m.qgroup !== QGROUP || m.chunkSize !== QC_CHUNK) return false;
    // allocate every GPU buffer up front from the manifest
    const bySeg = [];
    for (const s of m.segs) {
      const rec = _weights[s.name] || (_weights[s.name] = s.kind === 'int4' ? { N: s.N, K: s.K, int4: true, shape: s.shape } : { shape: s.shape, ...(s.kind === 'f32' ? { f32: true } : {}) });
      const buf = E.createBuffer(s.len, ST(), s.name + '.' + s.part);
      if (s.part === 'pack') rec.pack = buf; else if (s.part === 'scales') rec.scales = buf; else rec.buf = buf;
      bySeg.push({ ...s, buf });
    }
    for (let ci = 0; ci < m.nChunks; ci++) {   // stream chunks → writeBuffer slices
      const resp = await cache.match(_qcUrl(variant, 'c' + ci));
      if (!resp) return false;                  // evicted mid-set → treat as full miss
      const bytes = new Uint8Array(await resp.arrayBuffer());
      const cStart = ci * QC_CHUNK, cEnd = cStart + bytes.byteLength;
      for (const s of bySeg) {
        if (s.off + s.len <= cStart || s.off >= cEnd) continue;
        const b = Math.max(s.off, cStart), e = Math.min(s.off + s.len, cEnd);
        E.device().queue.writeBuffer(s.buf, b - s.off, bytes.buffer, b - cStart, e - b);
      }
      onProgress && onProgress({ phase: 'cache', pct: Math.round((ci + 1) / m.nChunks * 100) });
    }
    // experts stay on disk — record the catalog for on-demand ensureExpert() fetches
    _expertCatalog = {};
    for (const et of (m.expertTensors || [])) _expertCatalog[et.name] = { N: et.N, K: et.K };
    return true;
  }

  // ---- local folder store (File System Access API) ---------------------------------
  // The directory handle is persisted in IndexedDB so it survives reloads; the browser still
  // requires a user gesture to RE-grant read/write each session unless it persisted the grant.
  // Layout mirrors the cache: <root>/lfm25-vN/<variant>/{manifest.json, c0..cN, e/<tensor>}.
  const IDB_DB = 'sandpie-lfm25', IDB_STORE = 'handles', IDB_KEY = 'modelDir';
  function _idb() {
    return new Promise((res, rej) => {
      const r = indexedDB.open(IDB_DB, 1);
      r.onupgradeneeded = () => { try { r.result.createObjectStore(IDB_STORE); } catch (_) {} };
      r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
    });
  }
  async function _idbSetDir(h) { const db = await _idb(); return new Promise((res, rej) => { const t = db.transaction(IDB_STORE, 'readwrite'); t.objectStore(IDB_STORE).put(h, IDB_KEY); t.oncomplete = () => res(); t.onerror = () => rej(t.error); }); }
  async function _idbGetDir() { const db = await _idb(); return new Promise((res) => { const t = db.transaction(IDB_STORE, 'readonly'); const rq = t.objectStore(IDB_STORE).get(IDB_KEY); rq.onsuccess = () => res(rq.result || null); rq.onerror = () => res(null); }); }
  async function _idbDelDir() { const db = await _idb(); return new Promise((res) => { const t = db.transaction(IDB_STORE, 'readwrite'); t.objectStore(IDB_STORE).delete(IDB_KEY); t.oncomplete = () => res(); t.onerror = () => res(); }); }
  const _perm = async (h, req) => { let p = await h.queryPermission({ mode: 'readwrite' }).catch(() => 'prompt'); if (p !== 'granted' && req) p = await h.requestPermission({ mode: 'readwrite' }).catch(() => 'denied'); return p; };
  async function _fsVariantDir(variant, create) {
    if (!_fsRoot) return null;
    try {
      const root = await _fsRoot.getDirectoryHandle('lfm25-v' + QC_VER, { create });
      return await root.getDirectoryHandle(variant, { create });
    } catch (_) { return null; }
  }
  const _fsRead = async (dir, name) => (await (await dir.getFileHandle(name)).getFile()).arrayBuffer();
  const _fsWrite = async (dir, name, blob) => { const w = await (await dir.getFileHandle(name, { create: true })).createWritable(); await w.write(blob); await w.close(); };

  // Connect (or reconnect) a model folder — MUST be called from a user gesture. Reuses the
  // remembered handle if one exists (only needs a permission re-grant), else opens the picker.
  async function connectModelFolder() {
    if (!globalThis.showDirectoryPicker) throw new Error('File System Access API needs Chrome/Edge desktop');
    let h = _fsRoot || await _idbGetDir().catch(() => null);
    if (h) { if (await _perm(h, true) === 'granted') { _fsRoot = h; await _idbSetDir(h).catch(() => {}); return { ok: true, name: h.name, reconnected: true }; } }
    h = await globalThis.showDirectoryPicker({ id: 'sandpie-lfm25', mode: 'readwrite' });
    if (await _perm(h, true) !== 'granted') throw new Error('folder permission denied');
    _fsRoot = h; await _idbSetDir(h);
    return { ok: true, name: h.name, reconnected: false };
  }
  async function folderStatus() {
    const h = _fsRoot || await _idbGetDir().catch(() => null);
    if (!h) return { connected: false };
    return { connected: (await _perm(h, false)) === 'granted', name: h.name, remembered: true };
  }
  async function forgetModelFolder() { _fsRoot = null; _fsMode = false; _fsExpertDir = null; await _idbDelDir().catch(() => {}); return { ok: true }; }

  // Copy the cached quantized artifact for `variant` from Cache Storage into the folder. The
  // model must have been loaded once (cache populated). Manifest is written LAST = commit point.
  async function exportToFolder(variant, onProgress) {
    if (!_fsRoot || await _perm(_fsRoot, false) !== 'granted') throw new Error('connect a model folder first');
    const cache = await caches.open(QC_NAME);
    const mResp = await cache.match(_qcUrl(variant, 'manifest'));
    if (!mResp) throw new Error('no cached quantized ' + variant + ' — load it once (from HF) first');
    const m = await mResp.json();
    const dir = await _fsVariantDir(variant, true);
    if (!dir) throw new Error('could not create folder for ' + variant);
    const experts = m.expertTensors || [];
    const total = m.nChunks + experts.length; let done = 0;
    for (let ci = 0; ci < m.nChunks; ci++) {
      const r = await cache.match(_qcUrl(variant, 'c' + ci)); if (!r) throw new Error('cache chunk ' + ci + ' missing');
      await _fsWrite(dir, 'c' + ci, await r.blob());
      onProgress && onProgress({ phase: 'export', pct: Math.round((++done) / total * 100) });
    }
    if (experts.length) {
      const edir = await dir.getDirectoryHandle('e', { create: true });
      for (const et of experts) {
        const r = await cache.match(_qcUrl(variant, 'e/' + et.name)); if (!r) throw new Error('cache expert ' + et.name + ' missing');
        await _fsWrite(edir, et.name, await r.blob());
        onProgress && onProgress({ phase: 'export', pct: Math.round((++done) / total * 100) });
      }
    }
    await _fsWrite(dir, 'manifest.json', new Blob([JSON.stringify(m)], { type: 'application/json' }));
    return { ok: true, variant, chunks: m.nChunks, experts: experts.length };
  }

  // Read the quantized artifact from the folder (mirror of _readQuantCache). Returns false on
  // any miss/mismatch so the caller falls back to Cache Storage, then download.
  async function _readFsCache(variant, onProgress) {
    const dir = await _fsVariantDir(variant, false);
    if (!dir) return false;
    let m; try { m = JSON.parse(new TextDecoder().decode(await _fsRead(dir, 'manifest.json'))); } catch (_) { return false; }
    if (!m || m.ver !== QC_VER || m.qgroup !== QGROUP || m.chunkSize !== QC_CHUNK) return false;
    const bySeg = [];
    for (const s of m.segs) {
      const rec = _weights[s.name] || (_weights[s.name] = s.kind === 'int4' ? { N: s.N, K: s.K, int4: true, shape: s.shape } : { shape: s.shape, ...(s.kind === 'f32' ? { f32: true } : {}) });
      const buf = E.createBuffer(s.len, ST(), s.name + '.' + s.part);
      if (s.part === 'pack') rec.pack = buf; else if (s.part === 'scales') rec.scales = buf; else rec.buf = buf;
      bySeg.push({ ...s, buf });
    }
    for (let ci = 0; ci < m.nChunks; ci++) {
      let bytes; try { bytes = new Uint8Array(await _fsRead(dir, 'c' + ci)); } catch (_) { return false; }
      const cStart = ci * QC_CHUNK, cEnd = cStart + bytes.byteLength;
      for (const s of bySeg) {
        if (s.off + s.len <= cStart || s.off >= cEnd) continue;
        const b = Math.max(s.off, cStart), e = Math.min(s.off + s.len, cEnd);
        E.device().queue.writeBuffer(s.buf, b - s.off, bytes.buffer, b - cStart, e - b);
      }
      onProgress && onProgress({ phase: 'folder', pct: Math.round((ci + 1) / m.nChunks * 100) });
    }
    _expertCatalog = {};
    for (const et of (m.expertTensors || [])) _expertCatalog[et.name] = { N: et.N, K: et.K };
    _fsMode = true;
    try { _fsExpertDir = await dir.getDirectoryHandle('e', { create: false }); } catch (_) { _fsExpertDir = null; }
    return true;
  }

  async function _streamWeights(url, onProgress, sink) {
    // header: first 8 bytes = u64 header length, then the JSON header (Range requests)
    const h8 = await (await fetch(url, { headers: { Range: 'bytes=0-7' } })).arrayBuffer();
    const headerLen = Number(new DataView(h8).getBigUint64(0, true));
    const hResp = await fetch(url, { headers: { Range: 'bytes=8-' + (7 + headerLen) } });
    const header = JSON.parse(new TextDecoder().decode(await hResp.arrayBuffer()));
    const dataStart = 8 + headerLen;
    const tensors = Object.keys(header).filter(n => n !== '__metadata__')
      .map(n => ({ name: n, info: header[n], begin: header[n].data_offsets[0], end: header[n].data_offsets[1] }))
      .sort((a, b) => a.begin - b.begin);
    // one sequential body stream; rolling chunk window, tensors consumed in offset order
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('download failed: HTTP ' + resp.status);
    const total = +(resp.headers.get('content-length') || 0);
    const reader = resp.body.getReader();
    let chunks = [], winStart = 0, recv = 0, ti = 0;   // winStart = absolute offset of chunks[0] RELATIVE TO dataStart
    const have = () => recv - winStart;
    const takeRange = (begin, end) => {   // begin/end relative to dataStart, within the window
      const out = new Uint8Array(end - begin); let w = 0, off = winStart;
      for (const c of chunks) { const cEnd = off + c.length;
        if (cEnd > begin && off < end) { const s = Math.max(begin - off, 0), e = Math.min(end - off, c.length); out.set(c.subarray(s, e), w); w += e - s; }
        off = cEnd; }
      return out;
    };
    const dropTo = (abs) => { let off = winStart;
      while (chunks.length && off + chunks[0].length <= abs) { off += chunks[0].length; chunks.shift(); }
      winStart = off; };
    let skippedHeader = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (value) {
        if (!skippedHeader) {   // the body stream re-delivers header bytes — skip them
          const priorRecv = recv; recv += value.length;
          if (recv <= dataStart) continue;
          const cut = Math.max(dataStart - priorRecv, 0);
          chunks.push(value.subarray(cut)); winStart = 0; recv = recv - dataStart; skippedHeader = true;
        } else { chunks.push(value); recv += value.length; }
        while (ti < tensors.length && tensors[ti].end <= recv) {
          const t = tensors[ti];
          await _uploadTensor(t.name, t.info, takeRange(t.begin, t.end), sink);
          ti++; dropTo(ti < tensors.length ? tensors[ti].begin : recv);
          onProgress && onProgress({ phase: 'parse', pct: Math.round(ti / tensors.length * 100) });
        }
        if (total && onProgress) onProgress({ phase: 'download', pct: Math.round((recv + dataStart) / total * 100) });
      }
      if (done) break;
    }
    if (ti < tensors.length) throw new Error('stream ended early: ' + tensors[ti].name);
  }

  // Pack each MoE layer's 32 per-expert int4 tensors into ONE buffer per weight type, GPU-side
  // (copyBuffer from the already-loaded per-expert buffers — NO re-quantize, NO re-download, so
  // the per-tensor quant cache format is untouched). Enables the indexed GPU-resident expert
  // GEMV (idxGemv) that removes the per-layer router readback. One layer at a time so the
  // transient footprint is ~1 layer's experts (~350MB), not the whole model doubled.
  async function packMoE(onProgress) {
    const C = _cfg, eI = C.expertI, H = C.hidden, nE = C.nExperts;
    const specs = [['w1', eI * H / 2, eI * (H / QGROUP) * 2, eI, H], ['w3', eI * H / 2, eI * (H / QGROUP) * 2, eI, H], ['w2', H * eI / 2, H * (eI / QGROUP) * 2, H, eI]];
    for (let l = C.denseLayers; l < C.numLayers; l++) {
      const p = 'model.layers.' + l + '.feed_forward.';
      for (const [wt, pb, sb, rows, K] of specs) {
        const packBuf = E.createBuffer(pb * nE, ST(), 'moe' + l + wt + 'p');
        const scBuf = E.createBuffer(sb * nE, ST(), 'moe' + l + wt + 's');
        E.beginBatch();
        for (let e = 0; e < nE; e++) { const r = _weights[p + 'experts.' + e + '.' + wt + '.weight']; E.copyBuffer(r.pack, 0, packBuf, e * pb, pb); E.copyBuffer(r.scales, 0, scBuf, e * sb, sb); }
        await E.endBatch();
        for (let e = 0; e < nE; e++) { const key = p + 'experts.' + e + '.' + wt + '.weight'; const r = _weights[key]; try { r.pack.destroy(); r.scales.destroy(); } catch (_) {} delete _weights[key]; }
        _weights['moe.' + l + '.' + wt] = { pack: packBuf, scales: scBuf, N: rows * nE, K, int4: true, rowsPerExpert: rows };
      }
      onProgress && onProgress({ phase: 'pack', pct: Math.round((l - C.denseLayers + 1) / (C.numLayers - C.denseLayers) * 100) });
    }
  }

  // ---- INT3 EXPERTS (all-resident) ---------------------------------------------------------
  // int4 experts (~4.35GB) don't fit this GPU resident; int3 (~3.4GB incl. f16 scales) does.
  // Experts are requantized int4→int3 from the cache at load (cascade requant: dequant the int4
  // group, rescale so max|q|→4, clamp to [-4,3]) and stored BITPLANE-packed: each group of 32
  // weights = 3 u32s where u32[b] holds bit b of all 32 (clean WGSL extraction, no straddling).
  // With everything resident the whole forward is one batch — no readback, no disk, and the
  // deep-pipelined GEN_BATCH decode path engages. Enable: globalThis.__int3Experts (experiment).
  let _int3Mode = false;
  // int4 entry (pack nibbles=q+8, f16 scales, group 32) → int3 bitplanes (stored=q3+4) + f16 scales
  function _requant43(packU32, scalesU16, rows, K) {
    const gpr = K / QGROUP, nG = rows * gpr;
    const p3 = new Uint32Array(nG * 3), s3f = new Float32Array(nG);   // scales as f32, f16-converted in one batch
    const q4 = new Int32Array(QGROUP);
    for (let gi = 0; gi < nG; gi++) {
      const wBase = gi * (QGROUP / 8);   // 4 u32s of int4 per group
      let m = 0;
      for (let wI = 0; wI < QGROUP / 8; wI++) {
        const word = packU32[wBase + wI];
        for (let i = 0; i < 8; i++) { const q = ((word >>> (4 * i)) & 15) - 8; q4[wI * 8 + i] = q; const a = q < 0 ? -q : q; if (a > m) m = a; }
      }
      if (m === 0) { s3f[gi] = 0; p3[gi * 3] = 0; p3[gi * 3 + 1] = 0; p3[gi * 3 + 2] = 0xFFFFFFFF; continue; }   // scale 0 → value irrelevant; store q3=0 (stored 4 = bit2 set)
      const ratio = 4 / m;
      s3f[gi] = _f16ToF32s(scalesU16[gi]) * m / 4;
      let b0 = 0, b1 = 0, b2 = 0;
      for (let i = 0; i < QGROUP; i++) {
        let q = Math.round(q4[i] * ratio); if (q > 3) q = 3; else if (q < -4) q = -4;
        const st = q + 4;   // 0..7
        b0 |= (st & 1) << i; b1 |= ((st >> 1) & 1) << i; b2 |= ((st >> 2) & 1) << i;
      }
      p3[gi * 3] = b0 >>> 0; p3[gi * 3 + 1] = b1 >>> 0; p3[gi * 3 + 2] = b2 >>> 0;
    }
    return { p3, s3: E.f32ToF16(s3f) };
  }
  // Build the all-resident int3 expert pools from the cached int4 entries: one pack+scales buffer
  // per (layer, w1|w3|w2), rows laid out expert-major so idxGemv3 indexes row = e*N + n.
  async function packMoE3(onProgress) {
    const C = _cfg, eI = C.expertI, H = C.hidden, nE = C.nExperts;
    const specs = [['w1', eI, H], ['w3', eI, H], ['w2', H, eI]];
    const nL = C.numLayers - C.denseLayers;
    let done = 0, requanted = 0;
    for (let l = C.denseLayers; l < C.numLayers; l++) {
      const p = 'model.layers.' + l + '.feed_forward.';
      for (const [wt, rows, K] of specs) {
        const gpr = K / QGROUP;
        const pb3 = rows * gpr * 12, sb3 = rows * gpr * 2;   // per-expert int3 bytes
        const packBuf = E.createBuffer(pb3 * nE, ST(), 'moe3' + l + wt + 'p');
        const scBuf = E.createBuffer(sb3 * nE, ST(), 'moe3' + l + wt + 's');
        // WARM PATH: the int3 artifact is cached per (layer, tensor) — one ~51MB entry of all 32
        // experts' pack bytes then all scales — so a reload skips the ~3min JS requant entirely.
        // Entries live under the same /lfm25/v2/ prefix as the int4 set (survive _cleanOldCache).
        const url = _qcUrl(_variant, 'e3/' + l + '.' + wt);
        let blob = null;
        try { const r = _qcache && await _qcache.match(url); if (r) blob = await r.arrayBuffer(); } catch (_) {}
        if (blob && blob.byteLength === nE * (pb3 + sb3)) {
          E.device().queue.writeBuffer(packBuf, 0, blob, 0, nE * pb3);
          E.device().queue.writeBuffer(scBuf, 0, blob, nE * pb3, nE * sb3);
        } else {
          // COLD PATH: requant each expert's cached int4 → int3, upload, and persist the pooled entry.
          const all = new Uint8Array(nE * (pb3 + sb3));
          for (let e = 0; e < nE; e++) {
            const ab = await _readExpertBlob(p + 'experts.' + e + '.' + wt + '.weight');
            const packLen = rows * K / 2;
            const { p3, s3 } = _requant43(new Uint32Array(ab, 0, packLen / 4), new Uint16Array(ab, packLen, rows * gpr), rows, K);
            all.set(new Uint8Array(p3.buffer, 0, p3.byteLength), e * pb3);
            all.set(new Uint8Array(s3.buffer, 0, s3.byteLength), nE * pb3 + e * sb3);
          }
          E.device().queue.writeBuffer(packBuf, 0, all.buffer, 0, nE * pb3);
          E.device().queue.writeBuffer(scBuf, 0, all.buffer, nE * pb3, nE * sb3);
          try { if (_qcache) await _qcache.put(url, new Response(all)); } catch (_) {}   // best-effort (quota)
          requanted++;
        }
        _weights['moe3.' + l + '.' + wt] = { pack: packBuf, scales: scBuf, N: rows * nE, K, int3: true, rowsPerExpert: rows };
      }
      done++;
      onProgress && onProgress({ phase: 'int3', pct: Math.round(done / nL * 100) });
    }
    try { if (requanted) console.log('[lfm25] int3: requantized+cached ' + requanted + ' tensors (next load reads the int3 cache)'); } catch (_) {}
  }
  // Full-chain int3 self-test: random int4 tensors → _requant43 → idxGemv3 vs a CPU reference
  // that dequants the SAME int3 data. Catches packing/extraction layout disagreements exactly.
  async function selfTestInt3() {
    await E.init(); uniformReset();
    const nE = 4, rows = 8, K = 64, topK = 2, gpr = K / QGROUP;
    const rnd = (n, seed) => { const a = new Float32Array(n); let s = seed; for (let i = 0; i < n; i++) { s = (s * 16807) % 2147483647; a[i] = (s / 2147483647) * 2 - 1; } return a; };
    // build per-expert int4 (via the real quantizer path: f32→bf16 u16→_quantInt4), then requant
    const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
    const toBf16 = (f) => { const u = new Uint16Array(f.length); for (let i = 0; i < f.length; i++) { t[0] = f[i]; u[i] = ti[0] >>> 16; } return u; };
    const p3All = new Uint32Array(nE * rows * gpr * 3), s3All = new Uint16Array(nE * rows * gpr);
    const deq = [];   // per-expert CPU dequant of the int3 data
    for (let e = 0; e < nE; e++) {
      const { pack, scales } = _quantInt4(toBf16(rnd(rows * K, 31 + e)), rows, K);
      const { p3, s3 } = _requant43(pack, scales, rows, K);
      p3All.set(p3, e * rows * gpr * 3); s3All.set(s3, e * rows * gpr);
      const d = new Float32Array(rows * K);
      for (let gi = 0; gi < rows * gpr; gi++) {
        const s = _f16ToF32s(s3[gi]), b0 = p3[gi * 3], b1 = p3[gi * 3 + 1], b2 = p3[gi * 3 + 2];
        for (let i = 0; i < QGROUP; i++) { const st = ((b0 >>> i) & 1) | (((b1 >>> i) & 1) << 1) | (((b2 >>> i) & 1) << 2); d[gi * QGROUP + i] = s * (st - 4); }
      }
      deq.push(d);
    }
    const mkBuf = (arr) => { const b = E.createBuffer(arr.byteLength, ST(), 'i3t'); E.device().queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength); return b; };
    const x = rnd(topK * K, 97), idx = new Uint32Array([1, 3]);
    const xb = mkBuf(new Float32Array(x)), ib = mkBuf(idx), pb = mkBuf(p3All), sb = mkBuf(s3All);
    const yb = E.createBuffer(topK * rows * 4, ST(), 'i3y');
    await idxGemv3(xb, { pack: pb, scales: sb }, ib, yb, topK, rows, K, K);   // per-k input stride K
    await E.device().queue.onSubmittedWorkDone();
    const got = await E.readF32(yb, topK * rows);
    let maxErr = 0;
    for (let k = 0; k < topK; k++) for (let n = 0; n < rows; n++) {
      let ref = 0; for (let i = 0; i < K; i++) ref += deq[idx[k]][n * K + i] * x[k * K + i];
      const err = Math.abs(ref - got[k * rows + n]); if (err > maxErr) maxErr = err;
    }
    [xb, ib, pb, sb, yb].forEach(b => b.destroy());
    return { maxErr, pass: maxErr < 1e-3 };
  }
  // Full-chain test for the token-sorted per-expert GEMM: synthetic int3 experts + routing →
  // moeGemm3 vs a CPU ref that replicates the SAME int8-activation quant + int3 dequant, so a
  // tight tolerance catches any sort/gather/scatter/tiling index bug (which produce gross errors).
  async function selfTestMoeGemm() {
    await E.init(); uniformReset();
    const nE = 32, N = 128, Kc = 256, T = 64, K = 4, P = T * K, gpr = Kc / QGROUP;
    const rnd = (n, seed) => { const a = new Float32Array(n); let s = seed; for (let i = 0; i < n; i++) { s = (s * 16807) % 2147483647; a[i] = (s / 2147483647) * 2 - 1; } return a; };
    const t = new Float32Array(1), ti = new Uint32Array(t.buffer);
    const toBf16 = (f) => { const u = new Uint16Array(f.length); for (let i = 0; i < f.length; i++) { t[0] = f[i]; u[i] = ti[0] >>> 16; } return u; };
    // per-expert int3 weights [nE, N, Kc] → pool + CPU dequant
    const pool = new Uint32Array(nE * N * gpr * 3), psc = new Uint16Array(nE * N * gpr);
    const wdeq = new Float32Array(nE * N * Kc);
    for (let e = 0; e < nE; e++) {
      const { pack, scales } = _quantInt4(toBf16(rnd(N * Kc, 5 + e)), N, Kc);
      const { p3, s3 } = _requant43(pack, scales, N, Kc);
      pool.set(p3, e * N * gpr * 3); psc.set(s3, e * N * gpr);
      for (let n = 0; n < N; n++) for (let g = 0; g < gpr; g++) {
        const sc = _f16ToF32s(s3[(n * gpr + g)]), b0 = p3[(n * gpr + g) * 3], b1 = p3[(n * gpr + g) * 3 + 1], b2 = p3[(n * gpr + g) * 3 + 2];
        for (let i = 0; i < QGROUP; i++) { const st = ((b0 >>> i) & 1) | (((b1 >>> i) & 1) << 1) | (((b2 >>> i) & 1) << 2); wdeq[(e * N + n) * Kc + g * QGROUP + i] = sc * (st - 4); }
      }
    }
    const srcDiv = 1, srcRows = srcDiv === 1 ? P : T;   // srcDiv=1 exercises the w2 path
    const src = rnd(srcRows * Kc, 999);
    const ridx = new Uint32Array(P); { let s = 7; for (let p = 0; p < P; p++) { s = (s * 16807) % 2147483647; ridx[p] = s % nE; } }
    const mk = (arr) => { const b = E.createBuffer(arr.byteLength, ST(), 'mt'); E.device().queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength); return b; };
    const srcB = mk(src), poolB = mk(pool), pscB = mk(psc), ridxB = mk(ridx);
    const yB = E.createBuffer(P * N * 4, ST(), 'my');
    await moeGemm3(srcB, srcRows, { pack: poolB, scales: pscB }, ridxB, yB, P, N, Kc, srcDiv, nE);
    await E.device().queue.onSubmittedWorkDone();
    const got = await E.readF32(yB, P * N);
    // CPU ref: int8-quant src per group (matches QUANTQ8T), int3-dequant weights
    let maxErr = 0, _dbg = null;
    for (let p = 0; p < P; p++) {
      const token = Math.floor(p / srcDiv), e = ridx[p];
      for (let n = 0; n < N; n++) {
        let acc = 0;
        for (let g = 0; g < gpr; g++) {
          let m = 0; for (let i = 0; i < QGROUP; i++) { const v = Math.abs(src[token * Kc + g * QGROUP + i]); if (v > m) m = v; }
          const asc = m / 127, inv = asc > 0 ? 1 / asc : 0;
          for (let i = 0; i < QGROUP; i++) { const kk = g * QGROUP + i; const aq = Math.max(-127, Math.min(127, Math.round(src[token * Kc + kk] * inv))); acc += aq * asc * wdeq[(e * N + n) * Kc + kk]; }
        }
        const err = Math.abs(acc - got[p * N + n]);
        if (err > maxErr) { maxErr = err; _dbg = { p, n, e, got: got[p * N + n], ref: acc }; }
      }
    }
    [srcB, poolB, pscB, ridxB, yB].forEach(b => b.destroy());
    return { maxErr, pass: maxErr < 3e-2, dbg: _dbg };   // tol covers int8 activation-quant noise
  }

  // ---- expert LRU: fixed pool of GPU slots; disk-resident experts fetched on miss ----------
  // A slot holds one expert (w1/w3/w2 pack+scales). Per-token the router picks 4/layer → 88
  // distinct experts/forward; the LRU (Map insertion-order = recency) keeps the hot working set
  // resident so skewed routing mostly hits. Misses read one Cache Storage entry (~5.5MB) → slot.
  let _elru = null, _elruFree = [], _slotPool = [], _maxSlots = 256, _estat = { hit: 0, miss: 0 };
  let _pstat = { syncMs: 0, diskMs: 0, layers: 0, covLayers: 0 };   // streaming decode timing + coverage
  let _routeStage = null;   // persistent COPY_DST|MAP_READ staging for the per-layer routing readback
  let _logitStage = null;   // persistent staging for the 32 raw router logits (cache-aware selection)
  let _biasCache = {};      // layer -> Float32Array(nE) expert_bias, primed once (sync access in forward)
  // CACHE-AWARE ROUTING: replicate the router's top-K on the CPU (sc=sigmoid(logit), sel=sc+bias,
  // top-K by sel, weights = normalized sc of the chosen) but add a residency bonus λ to experts
  // already in the LRU. λ=0 → the model's exact routing; λ>0 nudges borderline picks toward
  // resident experts → higher cache coverage → fewer disk misses. Tunable: globalThis.__cacheRouteLambda.
  function _cacheAwareSelect(l, logits, nE, K) {
    const bias = _biasCache[l] || _ZERO32, lam = +globalThis.__cacheRouteLambda || 0;
    const sc = new Float32Array(nE), sel = new Float32Array(nE);
    for (let e = 0; e < nE; e++) { const s = 1 / (1 + Math.exp(-logits[e])); sc[e] = s; sel[e] = s + bias[e] + (lam && _elru.has(l + '.' + e) ? lam : 0); }
    const idx = new Array(K), used = new Uint8Array(nE);
    for (let k = 0; k < K; k++) { let best = 0, bv = -1e30; for (let j = 0; j < nE; j++) if (!used[j] && sel[j] > bv) { bv = sel[j]; best = j; } used[best] = 1; idx[k] = best; }
    let sum = 0; for (let k = 0; k < K; k++) sum += sc[idx[k]];
    const wt = new Float32Array(K); for (let k = 0; k < K; k++) wt[k] = sc[idx[k]] / (sum + 1e-6);
    let cov = 0; for (let k = 0; k < K; k++) if (_elru.has(l + '.' + idx[k])) cov++;   // resident BEFORE fetch
    if (cov === K) _pstat.covLayers++;
    return { idx, wt };
  }
  const _ZERO32 = new Float32Array(64);
  async function _primeBias() {   // read every MoE layer's expert_bias to the CPU once (sync in forward)
    _biasCache = {};
    const C = _cfg, nE = C.nExperts;
    for (let l = C.denseLayers; l < C.numLayers; l++) {
      const rec = _weights['model.layers.' + l + '.feed_forward.expert_bias'];
      if (rec && rec.buf) _biasCache[l] = await E.readF32(rec.buf, nE);
    }
  }
  function _ensureSlotPool() {
    if (_elru) return;
    const C = _cfg, eI = C.expertI, H = C.hidden;
    const packBytes = eI * H / 2, scBytes = eI * (H / QGROUP) * 2;   // w1/w3/w2 uniform (eI*H == H*eI)
    // 512 slots ≈ 3.2GB (+ ~0.9GB non-expert = ~4.4GB resident); measured 92.6% LRU hit / 1.62
    // tok/s on the 15GB test box (2.2GB headroom). Lower __expertSlots if free RAM is tighter.
    _maxSlots = Math.max(96, (globalThis.__expertSlots | 0) || 512);
    _slotPool = [];
    for (let i = 0; i < _maxSlots; i++) _slotPool.push({
      w1p: E.createBuffer(packBytes, ST(), 'sw1p'), w1s: E.createBuffer(scBytes, ST(), 'sw1s'),
      w3p: E.createBuffer(packBytes, ST(), 'sw3p'), w3s: E.createBuffer(scBytes, ST(), 'sw3s'),
      w2p: E.createBuffer(packBytes, ST(), 'sw2p'), w2s: E.createBuffer(scBytes, ST(), 'sw2s'),
    });
    _elru = new Map(); _elruFree = _slotPool.slice(); _estat = { hit: 0, miss: 0 };
  }
  async function _readExpertBlob(name) {   // one expert entry = concat(int4 pack, f16 scales)
    if (_fsMode) {
      if (!_fsExpertDir) throw new Error('expert folder missing');
      return await _fsRead(_fsExpertDir, name);
    }
    const resp = await _qcache.match(_qcUrl(_variant, 'e/' + name));
    if (!resp) throw new Error('expert entry missing on disk: ' + name);
    return await resp.arrayBuffer();
  }
  async function _readInto(name, packBuf, scBuf) {
    const rec = _expertCatalog[name]; if (!rec) throw new Error('expert not in catalog: ' + name);
    const ab = await _readExpertBlob(name);
    const packLen = rec.N * rec.K / 2;
    E.device().queue.writeBuffer(packBuf, 0, ab, 0, packLen);
    E.device().queue.writeBuffer(scBuf, 0, ab, packLen, ab.byteLength - packLen);
  }
  // Ensure the K experts of one layer are resident; return their slots. Slot assignment is
  // SYNCHRONOUS (no async race on the LRU), then all miss reads run CONCURRENTLY (Cache Storage
  // serves parallel reads far faster than the old serial 12-reads-per-layer).
  async function ensureExpertBatch(l, es) {
    const slots = new Array(es.length), reads = [];
    for (let i = 0; i < es.length; i++) {
      const e = es[i], key = l + '.' + e;
      const hit = _elru.get(key);
      if (hit) { _elru.delete(key); _elru.set(key, hit); _estat.hit++; slots[i] = hit; continue; }
      _estat.miss++;
      let slot;
      if (_elruFree.length) slot = _elruFree.pop();
      else { const ok = _elru.keys().next().value; slot = _elru.get(ok); _elru.delete(ok); }   // evict LRU (oldest, never this batch's fresh ones)
      _elru.set(key, slot); slots[i] = slot;   // reserve now so a later miss in this batch can't evict it
      const base = 'model.layers.' + l + '.feed_forward.experts.' + e + '.';
      reads.push(_readInto(base + 'w1.weight', slot.w1p, slot.w1s), _readInto(base + 'w3.weight', slot.w3p, slot.w3s), _readInto(base + 'w2.weight', slot.w2p, slot.w2s));
    }
    if (reads.length) await Promise.all(reads);
    return slots;
  }

  async function loadModel({ variant = '350M', onProgress } = {}) {
    if (_loaded && _variant === variant) return;
    const m = MODELS[variant]; if (!m) throw new Error('unknown LFM2.5 variant ' + variant);
    // CRITICAL: free the PREVIOUS model's GPU buffers before loading a new variant. Without
    // this, switching 350M→8B leaked ~all of the 350M's weights + scratch/KV/conv buffers
    // (reassigning _weights={} orphaned them without .destroy()), so the 8B's ~5.2GB piled on
    // top → GPU OOM → whole GPU process crash (browser blanks) → device lost. Drain first so
    // no in-flight work references the buffers we're about to destroy.
    if (_loaded || _weights || _scr) {
      try { await E.device().queue.onSubmittedWorkDone(); } catch (_) {}
      _freeState(); unload();
    }
    _variant = variant; _cfg = m.cfg; _weights = {}; _expertCatalog = {}; _fsMode = false; _fsExpertDir = null;
    // EXPERT STREAMING for MoE: experts live on disk, only the ~1.3GB non-expert weights stay
    // GPU-resident (fits the 15GB iGPU). __noStreamExperts forces the old all-resident path.
    _streamExperts = !!m.cfg.moe && !globalThis.__noStreamExperts;
    // int3 all-resident is the DEFAULT MoE path (15 tok/s vs streaming's ~4.4 on the test box);
    // __noInt3Experts falls back to int4 streaming for machines without the ~4.3GB residency.
    _int3Mode = !!m.cfg.moe && !globalThis.__noInt3Experts;
    MAX_SEQ = Math.max(2048, (globalThis.__lfmMaxSeq | 0) || 8192);
    PREFILL_T = Math.max(MATVEC_MAXT, (globalThis.__lfmPrefillT | 0) || 256);
    await E.init();
    _qcache = await caches.open(QC_NAME).catch(() => null);   // on-demand expert reads
    await _cleanOldCache();   // drop prior-version lfm25 entries so v(old)+v(new) can't blow quota
    await TOK.load(m.root);
    onProgress && onProgress({ phase: 'tokenizer', pct: 100 });
    // Silently adopt a remembered folder if the browser persisted the grant (no gesture needed);
    // otherwise the user reconnects it via connectModelFolder() and reloads.
    if (!_fsRoot) { try { const h = await _idbGetDir(); if (h && (await _perm(h, false)) === 'granted') _fsRoot = h; } catch (_) {} }
    // FASTEST PATH: local folder on real disk (eviction-proof) → then Cache Storage → then download.
    let hit = false;
    if (_fsRoot) { try { hit = await _readFsCache(variant, onProgress); if (hit) console.log('[lfm25] loaded from local folder'); } catch (e) { try { console.warn('[lfm25] folder read failed — trying cache', e); } catch (_) {} _weights = {}; _expertCatalog = {}; _fsMode = false; hit = false; } }
    if (!hit) try { hit = await _readQuantCache(variant, onProgress); } catch (e) { try { console.warn('[lfm25] cache read failed — falling back to download', e); } catch (_) {} _weights = {}; _expertCatalog = {}; hit = false; }
    if (!hit) {
      _weights = {}; _expertCatalog = {};
      const sink = _makeSink(variant);
      const sinkOk = await sink.open();
      await _streamWeights(m.root + 'model.safetensors', onProgress, sinkOk ? sink : null);
      if (sinkOk) { const committed = await sink.finish(); try { console.log('[lfm25] quant cache ' + (committed ? 'written' : 'NOT written (quota?)')); } catch (_) {} }
    }
    if (_int3Mode) {
      // int3 all-resident: requant the cached int4 experts → resident int3 pools, then leave
      // streaming OFF so generate takes the chained/pipelined no-readback decode path.
      await packMoE3(onProgress);
      _streamExperts = false;
      try { console.log('[lfm25] int3 experts resident: ' + Object.keys(_expertCatalog).length + ' requantized'); } catch (_) {}
    } else if (_streamExperts) {
      _ensureSlotPool();
      await _primeBias();   // expert_bias to CPU for cache-aware routing's top-K selection
      try { console.log('[lfm25] expert streaming: ' + Object.keys(_expertCatalog).length + ' experts on disk, ' + _maxSlots + ' GPU slots'); } catch (_) {}
    } else if (_cfg.moe && !globalThis.__noMoePack) {
      // all-resident GPU-packed path (only used with __noStreamExperts)
      await packMoE(onProgress); try { console.log('[lfm25] experts packed for GPU-resident dispatch'); } catch (_) {}
    }
    _loaded = true;
  }
  function unload() {
    try { if (_weights) for (const k in _weights) { const w = _weights[k]; for (const p of ['buf', 'pack', 'scales']) if (w[p] && w[p].destroy) try { w[p].destroy(); } catch (_) {} } } catch (_) {}
    _weights = null; _loaded = false; _variant = null;
  }
  function inventory() {
    if (!_weights) return null;
    const inv = { int4: 0, f16: 0, f32: 0, tensors: 0, int4Bytes: 0, f16Bytes: 0, names: [] };
    for (const k in _weights) {
      const w = _weights[k]; inv.tensors++;
      if (w.int4) { inv.int4++; inv.int4Bytes += w.N * w.K / 2 + w.N * w.K / QGROUP * 2; }
      else if (w.f32) inv.f32++;
      else { inv.f16++; inv.f16Bytes += w.shape.reduce((a, b) => a * b, 1) * 2; }
      if (inv.names.length < 400) inv.names.push(k + (w.int4 ? ' [int4 ' + w.N + 'x' + w.K + ']' : ' ' + JSON.stringify(w.shape)));
    }
    return inv;
  }

  // ============================================================
  // PHASE C — forward graph + generate (bring-up: correctness first).
  // Kernels adapted from the proven qwen3 set: rmsnorm, embed gather, batched int4
  // matvec (T ≤ 32; prefill runs in 32-token chunks), per-head norm + rotate_half RoPE
  // (LFM2's q/k_layernorm is [head_dim] — same contract), f32 flash attention (used for
  // BOTH prefill and decode for now), SwiGLU. Greedy decode with CPU argmax readback per
  // token. Deliberately unoptimized: the GPU-argmax chain / int8 KV / DP4A come after
  // the 350M speaks coherent text.
  // ============================================================
  const MATVEC_WG = 64, MATVEC_MAXT = 32;
  let PREFILL_T = 256;     // int3 prefill chunk — big enough for ~PREFILL_T/nE tokens/expert so the
                           // per-expert GEMM's shared-mem weight tile amortizes across the batch.
                           // Set from globalThis.__lfmPrefillT at loadModel.
  // Subgroup-FREE workgroup reduction: sum acc[0..CNT) across all WG threads, leaving row r's
  // full sum in part[r*WG + 0]. Replaces the subgroupAdd epilogue, which returns WRONG sums on
  // some GPUs (measured: Adreno 7xx, subgroup width 128) when the workgroup is a PARTIAL subgroup
  // (WG < hardware subgroup width — e.g. our WG=32). A plain shared-memory tree reduce is correct
  // on every GPU regardless of subgroup width or the `subgroups` feature. Needs WG a power of two
  // and `var<workgroup> part: array<f32, CNT*WG>` declared. lid must be the local id.
  function wgReduceWGSL(CNT, WG) {
    return `
  for (var _r:u32=0u; _r<${CNT}u; _r=_r+1u) { part[_r*${WG}u + lid.x] = acc[_r]; }
  workgroupBarrier();
  for (var _st:u32=${WG >> 1}u; _st>0u; _st=_st>>1u) {
    if (lid.x < _st) { for (var _r:u32=0u; _r<${CNT}u; _r=_r+1u) { part[_r*${WG}u + lid.x] = part[_r*${WG}u + lid.x] + part[_r*${WG}u + lid.x + _st]; } }
    workgroupBarrier();
  }`;
  }
  const MATVECQ_WGSL = `
enable f16;
enable subgroups;
struct D { T:u32, N:u32, K:u32, tBase:u32, acc:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${MATVEC_MAXT * MATVEC_WG}>;
@compute @workgroup_size(${MATVEC_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
  let n = wg.x + wg.y * nwg.x;
  if (n >= d.N) { return; }
  let words = d.K / 8u; let wbase = n*words;
  let gpr = d.K / ${QGROUP}u; let sbase = n*gpr;
  let K4 = d.K / 4u; let T = d.T;
  var acc : array<f32, ${MATVEC_MAXT}>;
  for (var t:u32=0u; t<${MATVEC_MAXT}u; t=t+1u) { acc[t] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let p = W[wbase + w];
    let s = f32(sc[sbase + (w*8u)/${QGROUP}u]);
    let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
    let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
    let wa = vec4<f32>(lo.x,hi.x,lo.y,hi.y); let wb_ = vec4<f32>(lo.z,hi.z,lo.w,hi.w);
    for (var t:u32=0u; t<T; t=t+1u) {
      let base = t*K4;
      acc[t] = acc[t] + s*( dot(wa, x[base + 2u*w]) + dot(wb_, x[base + 2u*w + 1u]) );
    }
    w = w + ${MATVEC_WG}u;
  }
  let sgIdx = lid.x / sgs;
  for (var t:u32=0u; t<T; t=t+1u) {
    let ssum = subgroupAdd(acc[t]);
    if (sgi == 0u) { part[t*${MATVEC_WG}u + sgIdx] = ssum; }
  }
  workgroupBarrier();
  if (lid.x < T) {
    let t = lid.x; let nsg=(${MATVEC_WG}u+sgs-1u)/sgs; var tot:f32=0.0;
    for (var i:u32=0u; i<nsg; i=i+1u) { tot = tot + part[t*${MATVEC_WG}u + i]; }
    let idx = t*d.N + n;
    y[idx] = select(0.0, y[idx], d.acc != 0u) + tot;
  }
}`;
  function matvecQ(xBuf, rec, yBuf, T, N, K, acc) {
    const d = uniform(new Uint32Array([T, N, K, 0, acc ? 1 : 0, 0, 0, 0]));
    const pipe = E.getPipeline('lfm25.matvecQ', MATVECQ_WGSL);
    const gx = Math.min(N, 65535), gy = Math.ceil(N / gx);
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, yBuf, d], [gx, gy, 1]);
  }

  // Efficient T=1 int4 GEMV: GEMVQ_NR output rows per workgroup (input reused across the
  // block in registers), subgroupAdd reduction — the design idxGemv/qwen use. Measured ~3×
  // the old 1-row matvecQ. acc=1 → y += result (fused residual). y=[N].
  let _gemvNR = 4;    // rows/workgroup for gemvQ8 + idxGemv (8B sweep: NR4 54.1 < NR8 55.3 < NR16 57.1ms)
  let _gemvWG = 32;   // threads/workgroup = 1 Intel subgroup (8B sweep: WG32 58.3 < WG64 67.9 < WG128 82.5ms — single-subgroup reduce + 2× occupancy)
  const _NR = () => Math.max(1, (globalThis.__gemvNR | 0) || _gemvNR);
  const _WG = () => Math.max(32, (globalThis.__gemvWG | 0) || _gemvWG);
  function gemvQ8Wgsl(NR, GEMV_WG) { return `
enable f16;
struct D { N:u32, K:u32, acc:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;
@group(0) @binding(4) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>) {
  let nBase = (wg.x + wg.y*nwg.x) * ${NR}u;
  let words = d.K / 8u; let gpr = d.K / ${QGROUP}u;
  var acc : array<f32, ${NR}>;
  for (var r:u32=0u; r<${NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = x[2u*w]; let xc = x[2u*w + 1u];
    let grp = (w*8u)/${QGROUP}u;
    for (var r:u32=0u; r<${NR}u; r=r+1u) {
      let n = nBase + r; if (n >= d.N) { continue; }
      let p = W[n*words + w];
      let s = f32(sc[n*gpr + grp]);
      let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      acc[r] = acc[r] + s*( dot(vec4<f32>(lo.x,hi.x,lo.y,hi.y), xa) + dot(vec4<f32>(lo.z,hi.z,lo.w,hi.w), xc) );
    }
    w = w + ${GEMV_WG}u;
  }
${wgReduceWGSL(NR, GEMV_WG)}
  if (lid.x < ${NR}u) {
    let n = nBase + lid.x;
    if (n < d.N) { y[n] = select(0.0, y[n], d.acc != 0u) + part[lid.x*${GEMV_WG}u + 0u]; }
  }
}`; }
  function gemvQ8(xBuf, rec, yBuf, N, K, acc) {
    const NR = _NR(), WG = _WG();
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const pipe = E.getPipeline('lfm25.gemvQ8.' + NR + '.' + WG, gemvQ8Wgsl(NR, WG));
    const nWG = Math.ceil(N / NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, yBuf, d], [gx, gy, 1]);
  }
  // ---- DP4A path: int8 activation × int4 weight via dot4I8Packed ---------------------
  // The int4 GEMVs run ~15-18 GB/s (ALU/dequant-bound, not memory). dot4I8Packed does 4 int8
  // MACs in one instr → far less ALU. Activation is quantized per-32-group to int8 once per
  // distinct input; weight scale × activation scale applied per group.
  // Activation quantize: x[K] f32 → _dp4.xq[K/4] packed int8 + _dp4.xs[K/32] f32 scales.
  const QUANTQ8_WGSL = `
struct Q { K:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       x  : array<f32>;
@group(0) @binding(1) var<storage, read_write> xq : array<u32>;
@group(0) @binding(2) var<storage, read_write> xs : array<f32>;
@group(0) @binding(3) var<uniform>             q  : Q;
var<workgroup> msh : array<f32, ${QGROUP}>;
var<workgroup> qsh : array<i32, ${QGROUP}>;
@compute @workgroup_size(${QGROUP},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let g = wg.x; let i = g*${QGROUP}u + lid.x;
  let v = select(0.0, x[i], i < q.K);
  msh[lid.x] = abs(v); workgroupBarrier();
  var s = ${QGROUP}u/2u;
  loop { if (s==0u){break;} if (lid.x<s){ msh[lid.x]=max(msh[lid.x],msh[lid.x+s]); } workgroupBarrier(); s=s/2u; }
  let mx = msh[0];
  let scale = select(mx/127.0, 1e-8, mx < 1e-12);
  if (lid.x == 0u) { xs[g] = scale; }
  var qv = i32(round(v/scale)); qv = clamp(qv, -127, 127);
  qsh[lid.x] = qv; workgroupBarrier();
  if (lid.x < ${QGROUP}u/4u) {
    let b = lid.x*4u;
    let packed = (u32(qsh[b]) & 0xFFu) | ((u32(qsh[b+1u]) & 0xFFu)<<8u) | ((u32(qsh[b+2u]) & 0xFFu)<<16u) | ((u32(qsh[b+3u]) & 0xFFu)<<24u);
    xq[g*(${QGROUP}u/4u) + lid.x] = packed;
  }
}`;
  let _dp4 = null;
  function ensureDp4(K) { if (_dp4 && _dp4.cap >= K) return; if (_dp4) { try { _dp4.xq.destroy(); _dp4.xs.destroy(); } catch (_) {} } _dp4 = { cap: K, xq: E.createBuffer((K / 4) * 4, ST(), 'xq'), xs: E.createBuffer((K / QGROUP) * 4, ST(), 'xs') }; }
  function quantAct(xBuf, K) {
    ensureDp4(K);
    const q = uniform(new Uint32Array([K, 0, 0, 0]));
    const groups = K / QGROUP, gx = Math.min(groups, 65535), gy = Math.ceil(groups / gx);
    return E.dispatch(E.getPipeline('lfm25.quantq8', QUANTQ8_WGSL), [xBuf, _dp4.xq, _dp4.xs, q], [gx, gy, 1]);
  }
  function gemvDP4Wgsl(NR, GEMV_WG) { return `
enable f16;
struct D { N:u32, K:u32, acc:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;
@group(0) @binding(1) var<storage, read>       W  : array<u32>;
@group(0) @binding(2) var<storage, read>       sc : array<f16>;
@group(0) @binding(3) var<storage, read>       xs : array<f32>;
@group(0) @binding(4) var<storage, read_write> y  : array<f32>;
@group(0) @binding(5) var<uniform>             d  : D;
var<workgroup> part : array<f32, ${NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(num_workgroups) nwg:vec3<u32>) {
  let nBase = (wg.x + wg.y*nwg.x) * ${NR}u;
  let words = d.K / 8u; let gpr = d.K / ${QGROUP}u;
  var acc : array<f32, ${NR}>;
  for (var r:u32=0u; r<${NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = xq[2u*w]; let xb = xq[2u*w + 1u];
    let grp = (w*8u)/${QGROUP}u; let xsc = xs[grp];
    for (var r:u32=0u; r<${NR}u; r=r+1u) {
      let n = nBase + r; if (n >= d.N) { continue; }
      let p = W[n*words + w];
      let lo = vec4<i32>(unpack4xU8(p & 0x0F0F0F0Fu)); let hi = vec4<i32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu));
      let wa = pack4xI8(vec4<i32>(lo.x,hi.x,lo.y,hi.y) - vec4<i32>(8));
      let wb = pack4xI8(vec4<i32>(lo.z,hi.z,lo.w,hi.w) - vec4<i32>(8));
      acc[r] = acc[r] + (f32(sc[n*gpr + grp])*xsc) * f32(dot4I8Packed(wa, xa) + dot4I8Packed(wb, xb));
    }
    w = w + ${GEMV_WG}u;
  }
${wgReduceWGSL(NR, GEMV_WG)}
  if (lid.x < ${NR}u) {
    let n = nBase + lid.x;
    if (n < d.N) { y[n] = select(0.0, y[n], d.acc != 0u) + part[lid.x*${GEMV_WG}u + 0u]; }
  }
}`; }
  function gemvDP4(rec, yBuf, N, K, acc) {   // caller ran quantAct(input, K) → _dp4
    const NR = _NR(), WG = _WG();
    const d = uniform(new Uint32Array([N, K, acc ? 1 : 0, 0]));
    const pipe = E.getPipeline('lfm25.gemvDP4.' + NR + '.' + WG, gemvDP4Wgsl(NR, WG));
    const nWG = Math.ceil(N / NR), gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
    return E.dispatch(pipe, [_dp4.xq, rec.pack, rec.scales, _dp4.xs, yBuf, d], [gx, gy, 1]);
  }

  // route T=1 to the f32-dequant 8-row GEMV (MEASURED faster than DP4A on gen-12lp: 61.5 vs
  // 68.9ms — the int4→int8 repack offsets dot4I8Packed here; DP4A kept opt-in via __useDp4 for
  // devices where it wins). prefill T>1 → batched matvecQ.
  function mv(xBuf, rec, yBuf, T, N, K, acc) {
    if (T !== 1) return (globalThis.__noDp4Gemm ? matvecQ : gemmDP4A)(xBuf, rec, yBuf, T, N, K, acc);
    if (globalThis.__useDp4) { quantAct(xBuf, K); return gemvDP4(rec, yBuf, N, K, acc); }
    return gemvQ8(xBuf, rec, yBuf, N, K, acc);
  }

  // ---- DP4A int8 GEMM for T>1 prefill (ported from the qwen3 engine, commit d8db4ed) --------
  // The T≤32 matvecQ was measured at 490ms per 32-token chunk (15ms/tok) on the 8B's dense
  // projections. This is the qwen prefill recipe: activations int8-quantized per group on the
  // GPU, then a 64×64-block tiled GEMM whose inner product is dot4I8Packed — gen-12lp's native
  // matrix path. Accumulators are CODEGEN-UNROLLED SCALARS: WGSL lowers indexed array<>
  // accumulators to scratch memory (10× collapse — hit twice now, qwen v1 and lfm25's
  // abandoned expert-major gemm3e).
  const GEMMQ_BM = 64, GEMMQ_BN = 64, GEMMQ_TM = 4, GEMMQ_TN = 4;
  const QUANTQ8T_WGSL = `
struct Q { K:u32, gpr:u32, ng:u32, _c:u32 };   // ng = T*gpr total (token,group) pairs
@group(0) @binding(0) var<storage, read>       x  : array<f32>;     // [T,K]
@group(0) @binding(1) var<storage, read_write> xq : array<u32>;     // [T,K/4] packed int8
@group(0) @binding(2) var<storage, read_write> xs : array<f32>;     // [T,gpr] group scales
@group(0) @binding(3) var<uniform>             q  : Q;
var<workgroup> msh : array<f32, ${QGROUP}>;
var<workgroup> qsh : array<i32, ${QGROUP}>;
@compute @workgroup_size(${QGROUP},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let gg = wg.x + wg.y*nwg.x; if (gg >= q.ng) { return; }
  let t = gg / q.gpr; let g = gg % q.gpr;
  let lid = lidv.x; let val = x[t*q.K + g*${QGROUP}u + lid];
  msh[lid] = abs(val); workgroupBarrier();
  var st = ${QGROUP / 2}u; loop { if(st==0u){break;} if(lid<st){ msh[lid]=max(msh[lid],msh[lid+st]); } workgroupBarrier(); st=st/2u; }
  let scale = msh[0] / 127.0; let inv = select(0.0, 1.0/scale, scale > 0.0);
  if (lid==0u) { xs[gg] = scale; }
  qsh[lid] = clamp(i32(round(val*inv)), -127, 127); workgroupBarrier();
  if (lid < ${QGROUP / 4}u) {
    xq[gg*${QGROUP / 4}u + lid] = pack4xI8(vec4<i32>(qsh[lid*4u], qsh[lid*4u+1u], qsh[lid*4u+2u], qsh[lid*4u+3u]));
  }
}`;
  function gemmdp4Wgsl() {
    const BM = GEMMQ_BM, BN = GEMMQ_BN, BK = QGROUP, TM = GEMMQ_TM, TN = GEMMQ_TN, BK4 = BK / 4;
    const NTH = (BM / TM) * (BN / TN), RN = BN / TN, TILEA = BM * BK4, TILEB = BN * BK4;
    let s = `
enable f16;
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var<storage, read>       xq : array<u32>;   // [T,K/4] packed int8 activations
@group(0) @binding(1) var<storage, read>       W  : array<u32>;   // int4 packed (q+8), K/8 per row
@group(0) @binding(2) var<storage, read>       sc : array<f16>;   // [N, K/${QGROUP}]
@group(0) @binding(3) var<storage, read>       xs : array<f32>;   // [T, K/${QGROUP}]
@group(0) @binding(4) var<storage, read_write> Y  : array<f32>;   // [T,N]
@group(0) @binding(5) var<uniform>             d  : D;
var<workgroup> As : array<u32, ${TILEA}>;   // [BM][BK4] packed int8 activations
var<workgroup> Bs : array<u32, ${TILEB}>;   // [BN][BK4] packed int8 weights (int4→int8)
@compute @workgroup_size(${NTH}, 1, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let lx = lid.x; let tN = lx % ${RN}u; let tM = lx / ${RN}u;
  let mBase = wg.y*${BM}u; let nBase = wg.x*${BN}u;
  let WPR = d.K/8u; let gpr = d.K/${QGROUP}u; let K4 = d.K/4u; let nTiles = d.K/${BK}u;
`;
    for (let r = 0; r < TM * TN; r++) s += `  var f${r}:f32=0.0;\n`;
    s += `  for (var kt:u32=0u; kt<nTiles; kt=kt+1u) {
    let k0 = kt*${BK}u;
    for (var r:u32=0u; r<${TILEA / NTH}u; r=r+1u) {
      let idx = lx + r*${NTH}u; let lt = idx/${BK4}u; let kk4 = idx%${BK4}u; let gt = mBase+lt;
      As[idx] = select(0u, xq[gt*K4 + k0/4u + kk4], gt<d.T);
    }
    for (var r:u32=0u; r<${TILEB / NTH}u; r=r+1u) {
      let idx = lx + r*${NTH}u; let ln = idx/${BK4}u; let kk4 = idx%${BK4}u; let gn = nBase+ln;
      var packed = 0u;
      if (gn < d.N) {
        let kk = k0 + kk4*4u; let word = W[gn*WPR + (kk>>3u)]; let b = (kk & 7u);
        let n0 = i32((word >> (4u*(b+0u))) & 0xFu) - 8;
        let n1 = i32((word >> (4u*(b+1u))) & 0xFu) - 8;
        let n2 = i32((word >> (4u*(b+2u))) & 0xFu) - 8;
        let n3 = i32((word >> (4u*(b+3u))) & 0xFu) - 8;
        packed = pack4xI8(vec4<i32>(n0,n1,n2,n3));
      }
      Bs[idx] = packed;
    }
    workgroupBarrier();
`;
    for (let r = 0; r < TM * TN; r++) s += `    var i${r}:i32=0;\n`;
    s += `    for (var cc:u32=0u; cc<${BK4}u; cc=cc+1u) {
`;
    for (let i = 0; i < TM; i++) s += `      let a${i} = As[(tM*${TM}u+${i}u)*${BK4}u + cc];\n`;
    for (let j = 0; j < TN; j++) s += `      let b${j} = Bs[(tN*${TN}u+${j}u)*${BK4}u + cc];\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `      i${i * TN + j} = i${i * TN + j} + dot4I8Packed(a${i}, b${j});\n`;
    s += `    }
`;
    for (let i = 0; i < TM; i++) s += `    let xs${i} = select(0.0, xs[(mBase+tM*${TM}u+${i}u)*gpr + kt], (mBase+tM*${TM}u+${i}u)<d.T);\n`;
    for (let j = 0; j < TN; j++) s += `    let ws${j} = select(0.0, f32(sc[(nBase+tN*${TN}u+${j}u)*gpr + kt]), (nBase+tN*${TN}u+${j}u)<d.N);\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `    f${i * TN + j} = f${i * TN + j} + f32(i${i * TN + j}) * xs${i} * ws${j};\n`;
    s += `    workgroupBarrier();
  }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  { let gm=mBase+tM*${TM}u+${i}u; let gn=nBase+tN*${TN}u+${j}u; if (gm<d.T && gn<d.N) { let idx=gm*d.N+gn; Y[idx]=select(0.0,Y[idx],d.acc!=0u)+f${i * TN + j}; } }\n`;
    s += `}`;
    return s;
  }
  let _dp4g = null;   // scratch {xq,xs,cap} for the [T,K] int8 activation quant
  function ensureDp4G(T, K) {
    const need = T * K;
    if (_dp4g && _dp4g.cap >= need) return;
    if (_dp4g) { try { _dp4g.xq.destroy(); _dp4g.xs.destroy(); } catch (_) {} }
    _dp4g = { cap: need, xq: E.createBuffer(need, ST(), 'gxq'), xs: E.createBuffer((need / QGROUP) * 4, ST(), 'gxs') };
  }
  function gemmDP4A(xBuf, wrec, yBuf, T, N, K, acc) {
    // Floor K to 8192 so the scratch is a CONSTANT size across every projection in a forward
    // (K jumps 2048↔7168 between layers) — otherwise a mid-batch grow would destroy a buffer
    // still referenced by already-recorded dispatches → garbage. T-aware for big prefill chunks.
    ensureDp4G(Math.max(T, MATVEC_MAXT), Math.max(K, 8192));
    const gpr = K / QGROUP, ng = T * gpr;
    const qp = E.getPipeline('lfm25.quantq8t', QUANTQ8T_WGSL);
    const qd = uniform(new Uint32Array([K, gpr, ng, 0]));
    const qgx = Math.min(ng, 65535), qgy = Math.ceil(ng / qgx);
    E.dispatch(qp, [xBuf, _dp4g.xq, _dp4g.xs, qd], [qgx, qgy, 1]);
    const pipe = E.getPipeline('lfm25.gemmDP4v2', gemmdp4Wgsl());
    const d = uniform(new Uint32Array([T, N, K, acc ? 1 : 0]));
    return E.dispatch(pipe, [_dp4g.xq, wrec.pack, wrec.scales, _dp4g.xs, yBuf, d], [Math.ceil(N / GEMMQ_BN), Math.ceil(T / GEMMQ_BM), 1]);
  }
  // Standalone int8 activation quant of [rows,K] f32 → qBuf [rows,K/4] packed + sBuf [rows,gpr].
  function quantAct8(srcBuf, qBuf, sBuf, rows, K) {
    const gpr = K / QGROUP, ng = rows * gpr;
    const qp = E.getPipeline('lfm25.quantq8t', QUANTQ8T_WGSL);
    const qd = uniform(new Uint32Array([K, gpr, ng, 0]));
    const qgx = Math.min(ng, 65535), qgy = Math.ceil(ng / qgx);
    E.dispatch(qp, [srcBuf, qBuf, sBuf, qd], [qgx, qgy, 1]);
  }

  // ---- TOKEN-SORTED PER-EXPERT MoE GEMM (mul_mat_id style) — the real prefill fix ------------
  // The pair-indexed GEMV re-read each expert's weights once PER PAIR (no token reuse). llama.cpp
  // sorts the T·K (token,expert) pairs by expert, then runs a DENSE tiled GEMM per expert so the
  // weight tile loaded to shared memory is reused across ALL that expert's token columns. At a big
  // prefill chunk (~64 tokens/expert) that is the whole batched-prefill win. Steps: counting-sort
  // the pairs → eoff[nE+1] offsets + sidx[P] (sorted pos → original pair); int8-quant the input;
  // per-expert DP4A GEMM that GATHERS A-rows via sidx (src = pair/srcDiv) and SCATTERS Y-rows to
  // the original pair position — so no gather/scatter buffers and downstream stays in pair order.
  const MOESORT_WG = 256;
  const MOECOUNT_WGSL = `
struct D { P:u32, nE:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       ridx : array<u32>;
@group(0) @binding(1) var<storage, read_write> cnt  : array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             d    : D;
@compute @workgroup_size(${MOESORT_WG},1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){ let p=gid.x; if(p<d.P){ atomicAdd(&cnt[ridx[p]],1u); } }`;
  const MOEPREFIX_WGSL = `
struct D { P:u32, nE:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read_write> cnt  : array<u32>;
@group(0) @binding(1) var<storage, read_write> eoff : array<u32>;
@group(0) @binding(2) var<storage, read_write> fill : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             d    : D;
@compute @workgroup_size(1,1,1)
fn main(){ var acc=0u; for(var e=0u;e<d.nE;e=e+1u){ eoff[e]=acc; atomicStore(&fill[e],acc); acc=acc+cnt[e]; }
  eoff[d.nE]=acc;
  for(var e=0u;e<d.nE;e=e+1u){ cnt[e]=0u; }   // reset for the next sort IN THIS batch (queue.writeBuffer
}`;                                            // can't: it flushes before all encoder dispatches, not between
  const MOESCATTER_WGSL = `
struct D { P:u32, nE:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       ridx : array<u32>;
@group(0) @binding(1) var<storage, read_write> fill : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> sidx : array<u32>;
@group(0) @binding(3) var<uniform>             d    : D;
@compute @workgroup_size(${MOESORT_WG},1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){ let p=gid.x; if(p<d.P){ let pos=atomicAdd(&fill[ridx[p]],1u); sidx[pos]=p; } }`;
  function moeSort(ridxBuf, P, nE, cntBuf, eoffBuf, fillBuf, sidxBuf) {
    const d = uniform(new Uint32Array([P, nE, 0, 0]));
    // cnt starts zero (WebGPU zero-inits new buffers) and moePrefix re-zeros it after each sort,
    // so counts never accumulate across the many sorts batched into one command encoder.
    const gx = Math.ceil(P / MOESORT_WG);
    E.dispatch(E.getPipeline('lfm25.moeCount', MOECOUNT_WGSL), [ridxBuf, cntBuf, d], [gx, 1, 1]);
    E.dispatch(E.getPipeline('lfm25.moePrefix', MOEPREFIX_WGSL), [cntBuf, eoffBuf, fillBuf, d], [1, 1, 1]);
    E.dispatch(E.getPipeline('lfm25.moeScatter', MOESCATTER_WGSL), [ridxBuf, fillBuf, sidxBuf, d], [gx, 1, 1]);
  }
  // Per-expert int3 GEMM. wg=(N-tile, expert, M-tile); empty (expert,mtile) exit after one read.
  const MGEMM_BM = 32, MGEMM_BN = 64, MGEMM_TM = 4, MGEMM_TN = 4;
  function moeGemm3Wgsl() {
    const BM = MGEMM_BM, BN = MGEMM_BN, BK = QGROUP, TM = MGEMM_TM, TN = MGEMM_TN, BK4 = BK / 4;
    const NTH = (BM / TM) * (BN / TN), RN = BN / TN, TILEA = BM * BK4, TILEB = BN * BK4;
    let s = `
enable f16;
struct D { N:u32, Kc:u32, K4:u32, gpr:u32, srcDiv:u32, P:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       xq   : array<u32>;   // [srcRows, Kc/4] int8 act
@group(0) @binding(1) var<storage, read>       xs   : array<f32>;   // [srcRows, gpr] act scales
@group(0) @binding(2) var<storage, read>       W    : array<u32>;   // int3 bitplane [nE*N, Kc]
@group(0) @binding(3) var<storage, read>       sc   : array<f16>;   // [nE*N, gpr] weight scales
@group(0) @binding(4) var<storage, read>       eoff : array<u32>;   // [nE+1]
@group(0) @binding(5) var<storage, read>       sidx : array<u32>;   // [P] sorted pos → pair
@group(0) @binding(6) var<storage, read_write> Y    : array<f32>;   // [P, N] (pair-major)
@group(0) @binding(7) var<uniform>             d    : D;
var<workgroup> As : array<u32, ${TILEA}>;   // [BM][BK4] int8 act (gathered)
var<workgroup> Bs : array<u32, ${TILEB}>;   // [BN][BK4] int8 weights (int3→int8)
var<workgroup> Ss : array<u32, ${BM}>;      // sidx[mBase + lt] for this tile (0 if OOB)
var<workgroup> Vv : array<u32, ${BM}>;      // valid flag per A-row
@compute @workgroup_size(${NTH}, 1, 1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let e = wg.y;
  let eStart = eoff[e]; let eEnd = eoff[e+1u];
  let mBase = eStart + wg.z*${BM}u;
  if (mBase >= eEnd) { return; }
  let lx = lid.x; let tN = lx % ${RN}u; let tM = lx / ${RN}u;
  let nBase = wg.x*${BN}u;
  let nTiles = d.Kc/${BK}u;
  // resolve this M-tile's source rows once
  for (var r:u32=0u; r<${Math.ceil(BM / NTH)}u; r=r+1u) {
    let lt = lx + r*${NTH}u;
    if (lt < ${BM}u) {
      let sPos = mBase + lt; let ok = sPos < eEnd;
      Vv[lt] = select(0u, 1u, ok);
      Ss[lt] = select(0u, sidx[sPos] / d.srcDiv, ok);   // source activation row
    }
  }
  workgroupBarrier();
`;
    for (let r = 0; r < TM * TN; r++) s += `  var f${r}:f32=0.0;\n`;
    s += `  for (var kt:u32=0u; kt<nTiles; kt=kt+1u) {
    let k0 = kt*${BK}u;
    for (var rr:u32=0u; rr<${TILEA / NTH}u; rr=rr+1u) {
      let idx = lx + rr*${NTH}u; let lt = idx/${BK4}u; let kk4 = idx%${BK4}u;
      As[idx] = select(0u, xq[Ss[lt]*d.K4 + k0/4u + kk4], Vv[lt]==1u);
    }
    for (var rr:u32=0u; rr<${TILEB / NTH}u; rr=rr+1u) {
      let idx = lx + rr*${NTH}u; let ln = idx/${BK4}u; let kk4 = idx%${BK4}u; let gn = nBase+ln;
      var packed = 0u;
      if (gn < d.N) {
        let gwrow = e*d.N + gn; let kk = k0 + kk4*4u; let g = kk/${QGROUP}u; let off = kk%${QGROUP}u;
        let base = (gwrow*d.gpr + g)*3u; let b0=W[base]; let b1=W[base+1u]; let b2=W[base+2u];
        let n0 = i32(((b0>>(off))&1u)|(((b1>>(off))&1u)<<1u)|(((b2>>(off))&1u)<<2u)) - 4;
        let n1 = i32(((b0>>(off+1u))&1u)|(((b1>>(off+1u))&1u)<<1u)|(((b2>>(off+1u))&1u)<<2u)) - 4;
        let n2 = i32(((b0>>(off+2u))&1u)|(((b1>>(off+2u))&1u)<<1u)|(((b2>>(off+2u))&1u)<<2u)) - 4;
        let n3 = i32(((b0>>(off+3u))&1u)|(((b1>>(off+3u))&1u)<<1u)|(((b2>>(off+3u))&1u)<<2u)) - 4;
        packed = pack4xI8(vec4<i32>(n0,n1,n2,n3));
      }
      Bs[idx] = packed;
    }
    workgroupBarrier();
`;
    for (let r = 0; r < TM * TN; r++) s += `    var i${r}:i32=0;\n`;
    s += `    for (var cc:u32=0u; cc<${BK4}u; cc=cc+1u) {
`;
    for (let i = 0; i < TM; i++) s += `      let a${i} = As[(tM*${TM}u+${i}u)*${BK4}u + cc];\n`;
    for (let j = 0; j < TN; j++) s += `      let b${j} = Bs[(tN*${TN}u+${j}u)*${BK4}u + cc];\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `      i${i * TN + j} = i${i * TN + j} + dot4I8Packed(a${i}, b${j});\n`;
    s += `    }
`;
    for (let i = 0; i < TM; i++) s += `    let xsc${i} = select(0.0, xs[Ss[tM*${TM}u+${i}u]*d.gpr + kt], Vv[tM*${TM}u+${i}u]==1u);\n`;
    for (let j = 0; j < TN; j++) s += `    let wsc${j} = select(0.0, f32(sc[(e*d.N + nBase+tN*${TN}u+${j}u)*d.gpr + kt]), (nBase+tN*${TN}u+${j}u)<d.N);\n`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `    f${i * TN + j} = f${i * TN + j} + f32(i${i * TN + j}) * xsc${i} * wsc${j};\n`;
    s += `    workgroupBarrier();
  }
`;
    for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  { let lt=tM*${TM}u+${i}u; let gn=nBase+tN*${TN}u+${j}u; if (Vv[lt]==1u && gn<d.N) { Y[Ss2(lt)*d.N + gn] = f${i * TN + j}; } }\n`;
    s += `}`;
    // Ss holds src rows (pair/srcDiv); for scatter we need the ORIGINAL pair = sidx[mBase+lt].
    // Recompute inline via a helper macro replaced below (WGSL has no fn closures over wg vars).
    s = s.replace(/Ss2\(lt\)/g, 'sidx[mBase + lt]');
    return s;
  }
  let _mgemm = null;   // scratch for MoE GEMM: sort buffers + gathered int8 act
  function ensureMgemm(P, K, nE) {
    if (_mgemm && _mgemm.P >= P && _mgemm.K >= K) return;
    if (_mgemm) { for (const k in _mgemm) if (_mgemm[k] && _mgemm[k].destroy) try { _mgemm[k].destroy(); } catch (_) {} }
    _mgemm = {
      P, K,
      cnt: E.createBuffer(nE * 4, ST(), 'mgc'), eoff: E.createBuffer((nE + 1) * 4, ST(), 'mge'),
      fill: E.createBuffer(nE * 4, ST(), 'mgf'), sidx: E.createBuffer(P * 4, ST(), 'mgs'),
      xq: E.createBuffer(P * K, ST(), 'mgxq'), xs: E.createBuffer(P * (K / QGROUP) * 4, ST(), 'mgxs'),
    };
  }
  // Full MoE expert application over T tokens: y[pair,N] = expert(ridx[pair]) applied to src rows.
  // srcBuf is the input matrix; srcRows = P/srcDiv rows of width Kc (srcDiv=K → per-token normed,
  // srcDiv=1 → per-pair moeAct). Returns via yBuf [P, N].
  function moeGemm3(srcBuf, srcRows, wrec, ridxBuf, yBuf, P, N, Kc, srcDiv, nE) {
    nE = nE || (_cfg && _cfg.nExperts);
    ensureMgemm(P, Math.max(Kc, 2048), nE);
    moeSort(ridxBuf, P, nE, _mgemm.cnt, _mgemm.eoff, _mgemm.fill, _mgemm.sidx);
    quantAct8(srcBuf, _mgemm.xq, _mgemm.xs, srcRows, Kc);
    const gpr = Kc / QGROUP;
    const d = uniform(new Uint32Array([N, Kc, Kc / 4, gpr, srcDiv, P, 0, 0]));
    const pipe = E.getPipeline('lfm25.moeGemm3', moeGemm3Wgsl());
    return E.dispatch(pipe, [_mgemm.xq, _mgemm.xs, wrec.pack, wrec.scales, _mgemm.eoff, _mgemm.sidx, yBuf, d],
      [Math.ceil(N / MGEMM_BN), nE, Math.ceil(P / MGEMM_BM)]);
  }

  const RMSNORM_WGSL = `
enable f16;
struct P { T:u32, H:u32, eps:f32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       w : array<f16>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;
var<workgroup> red : array<f32, 256>;
@compute @workgroup_size(256,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let t = wg.x; let H = p.H; let base = t*H;
  var ss = 0.0;
  var i = lid.x; loop { if (i >= H) { break; } let v = x[base+i]; ss = ss + v*v; i = i + 256u; }
  red[lid.x] = ss; workgroupBarrier();
  var st = 128u; loop { if (st==0u) { break; } if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x+st]; } workgroupBarrier(); st = st/2u; }
  let inv = inverseSqrt(red[0]/f32(H) + p.eps);
  i = lid.x; loop { if (i >= H) { break; } y[base+i] = x[base+i]*inv*f32(w[i]); i = i + 256u; }
}`;
  function rmsnorm(xBuf, wBuf, yBuf, T, H, eps) {
    const u = new Uint32Array(4); new DataView(u.buffer).setUint32(0, T, true); new DataView(u.buffer).setUint32(4, H, true); new DataView(u.buffer).setFloat32(8, eps, true);
    return E.dispatch(E.getPipeline('lfm25.rmsnorm', RMSNORM_WGSL), [xBuf, wBuf, yBuf, uniform(u)], [T, 1, 1]);
  }

  const EMBED_WGSL = `
enable f16;
struct P { T:u32, H:u32, idOff:u32, _b:u32 };   // reads ids[idOff + t] — chain mode passes tokHist + posBase
@group(0) @binding(0) var<storage, read>       ids   : array<u32>;
@group(0) @binding(1) var<storage, read>       embed : array<f16>;
@group(0) @binding(2) var<storage, read_write> y     : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let idx = gid.y*(nwg.x*64u)+gid.x; let total = p.T*p.H; if (idx >= total) { return; }
  let t = idx/p.H; let h = idx%p.H;
  y[idx] = f32(embed[ids[p.idOff + t]*p.H + h]);
}`;
  function embedGather(idsBuf, embBuf, yBuf, T, H, idOff) {
    const p = uniform(new Uint32Array([T, H, idOff || 0, 0]));
    const n = Math.ceil((T * H) / 64), gx = Math.min(n, 65535), gy = Math.ceil(n / gx);
    return E.dispatch(E.getPipeline('lfm25.embed', EMBED_WGSL), [idsBuf, embBuf, yBuf, p], [gx, gy, 1]);
  }

  const ROPEQK_WGSL = `
enable f16;
struct P { T:u32, nH:u32, hd:u32, posBase:u32, theta:f32, eps:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read>       normW: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
var<workgroup> red : array<f32, 128>;
var<workgroup> nrm : array<f32, 128>;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let hd = p.hd; let j = lid.x;
  let unit = wg.x; let t = unit/p.nH; let head = unit%p.nH;
  let base = t*(p.nH*hd) + head*hd;
  var v:f32 = 0.0; if (j < hd) { v = inp[base+j]; }
  red[j] = select(0.0, v*v, j < hd); workgroupBarrier();
  var stride = 64u;
  loop { if (stride==0u) { break; } if (j < stride) { red[j] = red[j] + red[j+stride]; } workgroupBarrier(); stride = stride/2u; }
  let inv = inverseSqrt(red[0]/f32(hd) + p.eps);
  if (j < hd) { nrm[j] = v*inv*f32(normW[j]); }
  workgroupBarrier();
  if (j >= hd) { return; }
  let half = hd/2u;
  let pos = f32(p.posBase + t);
  let freqIdx = select(j-half, j, j<half);
  let ang = pos * pow(p.theta, -2.0*f32(freqIdx)/f32(hd));
  let c = cos(ang); let s = sin(ang);
  let partner = select(nrm[j-half], nrm[j+half], j<half);
  let rot = select(partner, -partner, j<half);
  outp[base+j] = nrm[j]*c + rot*s;
}`;
  function ropeQK(inBuf, normWBuf, outBuf, T, nH, hd, posBase, theta, eps) {
    const u = new Uint32Array(8); const dv = new DataView(u.buffer);
    dv.setUint32(0, T, true); dv.setUint32(4, nH, true); dv.setUint32(8, hd, true);
    dv.setUint32(12, posBase, true); dv.setFloat32(16, theta, true); dv.setFloat32(20, eps, true);
    return E.dispatch(E.getPipeline('lfm25.ropeqk', ROPEQK_WGSL), [inBuf, normWBuf, outBuf, uniform(u)], [T * nH, 1, 1]);
  }

  // f32 tiled flash attention (QT=16 query block × KT=8 key tiles, online softmax) —
  // used for prefill AND decode in the bring-up (decode wastes 15/16 of the block; the
  // dedicated T=1 kernel is a later port).
  const AQT = 16, AKT = 8, AHD4 = 32, AWG = 128;
  const ATTN_WGSL = `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       K : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       V : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             p : P;
const QT=${AQT}u; const KT=${AKT}u; const HD4=${AHD4}u; const WG=${AWG}u;
var<workgroup> qsh : array<vec4<f32>, QT*HD4>;
var<workgroup> ksh : array<vec4<f32>, KT*HD4>;
var<workgroup> vsh : array<vec4<f32>, KT*HD4>;
var<workgroup> acc : array<vec4<f32>, QT*HD4>;
var<workgroup> scr : array<f32, QT*KT>;
var<workgroup> msh : array<f32, QT>;
var<workgroup> lsh : array<f32, QT>;
var<workgroup> csh : array<f32, QT>;
@compute @workgroup_size(${AWG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lidv:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>) {
  let tid = lidv.x;
  let hd4 = p.hd / 4u;
  let nqb = (p.T + QT - 1u)/QT;
  let blk = wg.x + wg.y*nwg.x;
  let hq = blk / nqb;
  if (hq >= p.nHq) { return; }
  let qbase = (blk % nqb) * QT;
  let grp = p.nHq / p.nKv; let hk = hq / grp;
  let kv4 = (p.nKv*p.hd)/4u; let qhs4 = (p.nHq*p.hd)/4u; let hkoff = hk*hd4; let hqoff = hq*hd4;
  let scale = 1.0/sqrt(f32(p.hd));
  let qmax = (p.S - p.T) + min(qbase + QT, p.T) - 1u;
  var e = tid;
  loop { if (e >= QT*hd4) { break; }
    let qi = e/hd4; let d4 = e%hd4; let gq = qbase+qi;
    qsh[qi*HD4+d4] = select(vec4<f32>(0.0), Q[gq*qhs4 + hqoff + d4], gq < p.T);
    acc[qi*HD4+d4] = vec4<f32>(0.0);
    e = e + WG;
  }
  if (tid < QT) { msh[tid] = -3.0e38; lsh[tid] = 0.0; }
  workgroupBarrier();
  var k0 = 0u;
  loop {
    if (k0 >= p.S || k0 > qmax) { break; }
    e = tid;
    loop { if (e >= KT*hd4) { break; }
      let kj = e/hd4; let d4 = e%hd4; let gk = k0+kj; let ok = gk < p.S;
      ksh[kj*HD4+d4] = select(vec4<f32>(0.0), K[gk*kv4 + hkoff + d4], ok);
      vsh[kj*HD4+d4] = select(vec4<f32>(0.0), V[gk*kv4 + hkoff + d4], ok);
      e = e + WG;
    }
    workgroupBarrier();
    { let qi = tid / KT; let kj = tid % KT;
      var s4 = vec4<f32>(0.0);
      for (var i4=0u;i4<hd4;i4=i4+1u){ s4 = s4 + qsh[qi*HD4+i4]*ksh[kj*HD4+i4]; }
      let dot = s4.x + s4.y + s4.z + s4.w;
      let gq = qbase+qi; let gk = k0+kj; let gqpos = (p.S - p.T) + gq;
      let valid = (gq < p.T) && (gk < p.S) && (gk <= gqpos);
      scr[tid] = select(-3.0e38, dot*scale, valid);
    }
    workgroupBarrier();
    if (tid < QT) {
      let qi = tid;
      var tm = -3.0e38;
      for (var kj=0u;kj<KT;kj=kj+1u){ tm = max(tm, scr[qi*KT+kj]); }
      let mnew = max(msh[qi], tm);
      let corr = exp(msh[qi] - mnew);
      var sum = 0.0;
      for (var kj=0u;kj<KT;kj=kj+1u){
        let pw = select(0.0, exp(scr[qi*KT+kj]-mnew), scr[qi*KT+kj] > -3.0e37);
        scr[qi*KT+kj] = pw; sum = sum + pw;
      }
      lsh[qi] = lsh[qi]*corr + sum; msh[qi] = mnew; csh[qi] = corr;
    }
    workgroupBarrier();
    e = tid;
    loop { if (e >= QT*hd4) { break; }
      let qi = e/hd4; let d4 = e%hd4;
      var a = acc[qi*HD4+d4]*csh[qi];
      for (var kj=0u;kj<KT;kj=kj+1u){ a = a + scr[qi*KT+kj]*vsh[kj*HD4+d4]; }
      acc[qi*HD4+d4] = a;
      e = e + WG;
    }
    workgroupBarrier();
    k0 = k0 + KT;
  }
  e = tid;
  loop { if (e >= QT*hd4) { break; }
    let qi = e/hd4; let d4 = e%hd4; let gq = qbase+qi;
    if (gq < p.T) { O[gq*qhs4 + hqoff + d4] = acc[qi*HD4+d4] / lsh[qi]; }
    e = e + WG;
  }
}`;
  function attention(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd) {
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    const pipe = E.getPipeline('lfm25.attn', ATTN_WGSL);
    const blocks = nHq * Math.ceil(T / AQT);
    const gx = Math.min(blocks, 65535), gy = Math.ceil(blocks / gx);
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [gx, gy, 1]);
  }

  const SWIGLU_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       g : array<f32>;
@group(0) @binding(1) var<storage, read>       u : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i = gid.y*(nwg.x*64u)+gid.x; if (i >= p.n) { return; }
  let gv = g[i];
  y[i] = (gv / (1.0 + exp(-gv))) * u[i];
}`;
  function swiglu(gBuf, uBuf, yBuf, n) {
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    const c = Math.ceil(n / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('lfm25.swiglu', SWIGLU_WGSL), [gBuf, uBuf, yBuf, p], [gx, gy, 1]);
  }

  // scaled residual add: x[i] += scale * v[i]  (MoE expert-output accumulation)
  const AXPY_WGSL = `
struct P { n:u32, scale:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read_write> x : array<f32>;
@group(0) @binding(1) var<storage, read>       v : array<f32>;
@group(0) @binding(2) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i = gid.y*(nwg.x*64u)+gid.x; if (i >= p.n) { return; }
  x[i] = x[i] + p.scale * v[i];
}`;
  function axpy(xBuf, vBuf, n, scale) {
    const u = new Uint32Array(4); const dv = new DataView(u.buffer); dv.setUint32(0, n, true); dv.setFloat32(4, scale, true);
    const c = Math.ceil(n / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('lfm25.axpy', AXPY_WGSL), [xBuf, vBuf, uniform(u)], [gx, gy, 1]);
  }

  // ---- GPU-resident indexed expert GEMV (kills the per-layer router readback) --------
  // int4 GEMV over PACKED experts [nE*N, Kc]: workgroup (rowBlock, k) reads idx[k] on the
  // GPU → runs expert e=idx[k]'s block (rows e*N..), input at k*inStride (0 for w1/w3 which
  // share `normed`; eI for w2 whose input is per-expert act[k]), output at k*N. No readback:
  // the whole MoE block stays batched. GEMVQ_NR rows/workgroup, subgroupAdd reduction.
  function idxGemvWgsl(NR, GEMV_WG) { return `
enable f16;
enable subgroups;
struct D { N:u32, Kc:u32, inStride:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W   : array<u32>;
@group(0) @binding(2) var<storage, read>       sc  : array<f16>;
@group(0) @binding(3) var<storage, read>       idx : array<u32>;
@group(0) @binding(4) var<storage, read_write> y   : array<f32>;
@group(0) @binding(5) var<uniform>             d   : D;
var<workgroup> part : array<f32, ${NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(subgroup_size) sgs:u32, @builtin(subgroup_invocation_id) sgi:u32) {
  let k = wg.y;                              // selected-expert slot
  let e = idx[k];
  let nBase = wg.x * ${NR}u;                 // first output row (within the expert's N rows)
  let words = d.Kc / 8u; let gpr = d.Kc / ${QGROUP}u;
  let xb = (k * d.inStride) / 4u;            // per-k input base (vec4 units)
  var acc : array<f32, ${NR}>;
  for (var r:u32=0u; r<${NR}u; r=r+1u) { acc[r] = 0.0; }
  var w = lid.x;
  loop {
    if (w >= words) { break; }
    let xa = x[xb + 2u*w]; let xc = x[xb + 2u*w + 1u];
    let grp = (w*8u)/${QGROUP}u;
    for (var r:u32=0u; r<${NR}u; r=r+1u) {
      let n = nBase + r; if (n >= d.N) { continue; }
      let row = e*d.N + n;                    // expert e's row block in the packed tensor
      let p = W[row*words + w];
      let s = f32(sc[row*gpr + grp]);
      let lo = vec4<f32>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      let hi = vec4<f32>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);
      acc[r] = acc[r] + s*( dot(vec4<f32>(lo.x,hi.x,lo.y,hi.y), xa) + dot(vec4<f32>(lo.z,hi.z,lo.w,hi.w), xc) );
    }
    w = w + ${GEMV_WG}u;
  }
  let sgIdx = lid.x / sgs;
  for (var r:u32=0u; r<${NR}u; r=r+1u) {
    let ss = subgroupAdd(acc[r]);
    if (sgi == 0u) { part[r*${GEMV_WG}u + sgIdx] = ss; }
  }
  workgroupBarrier();
  if (lid.x < ${NR}u) {
    let n = nBase + lid.x;
    if (n < d.N) {
      let nsg=(${GEMV_WG}u+sgs-1u)/sgs; var t:f32=0.0;
      for(var i:u32=0u;i<nsg;i=i+1u){ t = t + part[lid.x*${GEMV_WG}u + i]; }
      y[k*d.N + n] = t;
    }
  }
}`; }
  function idxGemv(xBuf, rec, idxBuf, yBuf, topK, N, Kc, inStride) {
    const NR = _NR(), WG = _WG();
    const d = uniform(new Uint32Array([N, Kc, inStride, 0]));
    const pipe = E.getPipeline('lfm25.idxGemv.' + NR + '.' + WG, idxGemvWgsl(NR, WG));
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, idxBuf, yBuf, d], [Math.ceil(N / NR), topK, 1]);
  }
  // int3 variant: weights BITPLANE-packed (group of 32 = 3 u32s; u32[b] holds bit b of all 32
  // stored values, stored = q+4, q in [-4,3]). One thread processes a whole 32-weight group per
  // iteration: load the group's 8 x-vec4s once, reuse across the NR rows.
  // PAIR-INDEXED for batched prefill: wg.y ranges over T*K (token,k) pairs; the input row is
  // (pair / xdiv) * inStride — xdiv=K maps pairs to their token's activation row (w1/w3), xdiv=1
  // maps each pair to its own row (w2 reading [T*K, eI]). Decode is just T=1 of the same math.
  function idxGemv3Wgsl(NR, GEMV_WG) { return `
enable f16;
struct D { N:u32, Kc:u32, inStride:u32, xdiv:u32 };
@group(0) @binding(0) var<storage, read>       x   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W   : array<u32>;
@group(0) @binding(2) var<storage, read>       sc  : array<f16>;
@group(0) @binding(3) var<storage, read>       idx : array<u32>;
@group(0) @binding(4) var<storage, read_write> y   : array<f32>;
@group(0) @binding(5) var<uniform>             d   : D;
var<workgroup> part : array<f32, ${NR * GEMV_WG}>;
@compute @workgroup_size(${GEMV_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let k = wg.y;                              // (token,k) pair index — T*K of them (decode: T=1)
  let e = idx[k];
  let nBase = wg.x * ${NR}u;
  let gpr = d.Kc / ${QGROUP}u;
  let xb = ((k / d.xdiv) * d.inStride) / 4u;
  var acc : array<f32, ${NR}>;
  for (var r:u32=0u; r<${NR}u; r=r+1u) { acc[r] = 0.0; }
  var g = lid.x;
  loop {
    if (g >= gpr) { break; }
    var xv : array<vec4<f32>, 8>;
    for (var i:u32=0u; i<8u; i=i+1u) { xv[i] = x[xb + g*8u + i]; }
    for (var r:u32=0u; r<${NR}u; r=r+1u) {
      let n = nBase + r; if (n >= d.N) { continue; }
      let row = e*d.N + n;
      let base = (row*gpr + g)*3u;
      let b0 = W[base]; let b1 = W[base+1u]; let b2 = W[base+2u];
      let s = f32(sc[row*gpr + g]);
      var sum = 0.0;
      for (var i:u32=0u; i<8u; i=i+1u) {
        let j = i*4u;
        let q = vec4<f32>(
          f32(((b0>>(j   ))&1u) | (((b1>>(j   ))&1u)<<1u) | (((b2>>(j   ))&1u)<<2u)),
          f32(((b0>>(j+1u))&1u) | (((b1>>(j+1u))&1u)<<1u) | (((b2>>(j+1u))&1u)<<2u)),
          f32(((b0>>(j+2u))&1u) | (((b1>>(j+2u))&1u)<<1u) | (((b2>>(j+2u))&1u)<<2u)),
          f32(((b0>>(j+3u))&1u) | (((b1>>(j+3u))&1u)<<1u) | (((b2>>(j+3u))&1u)<<2u))
        ) - vec4<f32>(4.0);
        sum = sum + dot(q, xv[i]);
      }
      acc[r] = acc[r] + s*sum;
    }
    g = g + ${GEMV_WG}u;
  }
${wgReduceWGSL(NR, GEMV_WG)}
  if (lid.x < ${NR}u) {
    let n = nBase + lid.x;
    if (n < d.N) { y[k*d.N + n] = part[lid.x*${GEMV_WG}u + 0u]; }
  }
}`; }
  function idxGemv3(xBuf, rec, idxBuf, yBuf, pairs, N, Kc, inStride, xdiv) {
    const NR = _NR(), WG = _WG();
    const d = uniform(new Uint32Array([N, Kc, inStride, xdiv || 1]));
    const pipe = E.getPipeline('lfm25.idxGemv3.' + NR + '.' + WG, idxGemv3Wgsl(NR, WG));
    return E.dispatch(pipe, [xBuf, rec.pack, rec.scales, idxBuf, yBuf, d], [Math.ceil(N / NR), pairs, 1]);
  }
  // combine: x[t*H+h] += Σ_k wt[t*K+k] * o[(t*K+k)*H + h]   (per-token weighted expert sum)
  const MOECOMBINE_WGSL = `
struct D { H:u32, K:u32, T:u32, _b:u32 };
@group(0) @binding(0) var<storage, read_write> x  : array<f32>;
@group(0) @binding(1) var<storage, read>       o  : array<f32>;
@group(0) @binding(2) var<storage, read>       wt : array<f32>;
@group(0) @binding(3) var<uniform>             d  : D;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let i = gid.y*(nwg.x*64u)+gid.x; if (i >= d.T*d.H) { return; }
  let t = i / d.H; let h = i % d.H;
  var s = 0.0;
  for (var k:u32=0u; k<d.K; k=k+1u) { s = s + wt[t*d.K+k]*o[(t*d.K+k)*d.H + h]; }
  x[i] = x[i] + s;
}`;
  function moeCombine(xBuf, oBuf, wtBuf, H, K, T) {
    const d = uniform(new Uint32Array([H, K, T || 1, 0]));
    const c = Math.ceil((T || 1) * H / 64), gx = Math.min(c, 65535), gy = Math.ceil(c / gx);
    return E.dispatch(E.getPipeline('lfm25.moeCombine', MOECOMBINE_WGSL), [xBuf, oBuf, wtBuf, d], [gx, gy, 1]);
  }

  // ---- two-stage GPU argmax → token history (enables GPU-resident decode chaining) ---
  // Stage 1: AMAX_WGS workgroups reduce interleaved logit slices → per-WG (max,idx). Stage 2:
  // one workgroup folds them and writes the token id into tokHist[outPos] on the GPU — so the
  // NEXT forward's embed can read it without any CPU readback. This is what lets GEN_BATCH
  // decode tokens chain in one submit (the throughput win: 1 readback per batch, not per token).
  const ARGMAX_WG = 256, AMAX_WGS = 64;
  const AMAX_P1_WGSL = `
struct P { n:u32, outPos:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> pV : array<f32>;
@group(0) @binding(2) var<storage, read_write> pI : array<u32>;
@group(0) @binding(3) var<uniform>             p  : P;
var<workgroup> sv : array<f32, ${ARGMAX_WG}>;
var<workgroup> si : array<u32, ${ARGMAX_WG}>;
@compute @workgroup_size(${ARGMAX_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  var bv=-3.0e38; var bi=0u;
  var i = wg.x*${ARGMAX_WG}u + lid.x;
  loop { if (i>=p.n) {break;} let v=logits[i]; if (v>bv){bv=v;bi=i;} i=i+${AMAX_WGS * ARGMAX_WG}u; }
  sv[lid.x]=bv; si[lid.x]=bi; workgroupBarrier();
  var s=${ARGMAX_WG}u/2u;
  loop { if(s==0u){break;} if(lid.x<s){ if(sv[lid.x+s]>sv[lid.x]){sv[lid.x]=sv[lid.x+s];si[lid.x]=si[lid.x+s];} } workgroupBarrier(); s=s/2u; }
  if (lid.x==0u){ pV[wg.x]=sv[0]; pI[wg.x]=si[0]; }
}`;
  const AMAX_P2_WGSL = `
struct P { n:u32, outPos:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       pV : array<f32>;
@group(0) @binding(1) var<storage, read>       pI : array<u32>;
@group(0) @binding(2) var<storage, read_write> tok: array<u32>;
@group(0) @binding(3) var<uniform>             p  : P;
var<workgroup> sv : array<f32, ${AMAX_WGS}>;
var<workgroup> si : array<u32, ${AMAX_WGS}>;
@compute @workgroup_size(${AMAX_WGS},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>){
  sv[lid.x]=pV[lid.x]; si[lid.x]=pI[lid.x]; workgroupBarrier();
  var s=${AMAX_WGS}u/2u;
  loop { if(s==0u){break;} if(lid.x<s){ if(sv[lid.x+s]>sv[lid.x]){sv[lid.x]=sv[lid.x+s];si[lid.x]=si[lid.x+s];} } workgroupBarrier(); s=s/2u; }
  if (lid.x==0u){ tok[p.outPos]=si[0]; }
}`;
  let _amaxV = null, _amaxI = null;
  function argmaxKernel(logitsBuf, tokHistBuf, N, outPos) {
    if (!_amaxV) { _amaxV = E.createBuffer(AMAX_WGS * 4, ST(), 'amV'); _amaxI = E.createBuffer(AMAX_WGS * 4, ST(), 'amI'); }
    const p = uniform(new Uint32Array([N, outPos, 0, 0]));
    E.dispatch(E.getPipeline('lfm25.amaxP1', AMAX_P1_WGSL), [logitsBuf, _amaxV, _amaxI, p], [AMAX_WGS, 1, 1]);
    return E.dispatch(E.getPipeline('lfm25.amaxP2', AMAX_P2_WGSL), [_amaxV, _amaxI, tokHistBuf, p], [1, 1, 1]);
  }

  // ---- forward state --------------------------------------------------------------
  // Context window. KV is only on 6 attn layers (conv state is seq-independent) so long ctx is
  // cheap: 8192 ≈ 200MB f32 KV. Override with globalThis.__lfmMaxSeq (applied at loadModel).
  let MAX_SEQ = 8192;
  let _scr = null, _kv = null, _conv = null, _idsBuf = null;
  function _ensureState() {
    if (_scr) return;
    const C = _cfg, H = C.hidden, T = Math.max(MATVEC_MAXT, PREFILL_T);   // scratch sized for prefill chunks
    const mk = (n, l) => E.createBuffer(n * 4, ST(), l);
    // w1 shapes are auto-adjusted — read the real intermediate from the weights
    const I = _weights['model.layers.0.feed_forward.w1.weight'].N;
    _cfg.ffI = I;
    _scr = {
      x: mk(T * H, 'x'), normed: mk(T * H, 'nrm'),
      bcx: mk(T * 3 * H, 'bcx'), convy: mk(T * H, 'cy'),
      q: mk(T * C.nHeads * C.headDim, 'q'), k: mk(T * C.nKvHeads * C.headDim, 'k'), v: mk(T * C.nKvHeads * C.headDim, 'v'),
      qr: mk(T * C.nHeads * C.headDim, 'qr'), kr: mk(T * C.nKvHeads * C.headDim, 'kr'),
      attn: mk(T * C.nHeads * C.headDim, 'at'),
      gate: mk(T * I, 'g'), up: mk(T * I, 'u'), swi: mk(T * I, 's'),
      last: mk(H, 'last'), logits: mk(C.vocab, 'lg'), tokHist: E.createBuffer(MAX_SEQ * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'th'),
      // MoE scratch (harmless on dense models): expert output [H], router logits [nE], top-k idx/wt,
      // and the GPU-resident indexed-dispatch buffers [K,eI]/[K,H]
      // MoE scratch sized for T-batched prefill chunks (T*K pairs); ~1MB each at T=32, K=4
      eout: mk(H, 'eo'), rlogits: mk(Math.max(T * (C.nExperts || 1), 1), 'rl'), ridx: mk(Math.max(T * (C.topK || 1), 1), 'ri'), rwt: mk(Math.max(T * (C.topK || 1), 1), 'rw'),
      moeGate: mk(Math.max(T * (C.topK || 1) * (C.expertI || 1), 1), 'mg'), moeUp: mk(Math.max(T * (C.topK || 1) * (C.expertI || 1), 1), 'mu'),
      moeAct: mk(Math.max(T * (C.topK || 1) * (C.expertI || 1), 1), 'ma'), moeOut: mk(Math.max(T * (C.topK || 1) * H, 1), 'mo'),

    };
    _kv = {}; _conv = {};
    for (let l = 0; l < C.numLayers; l++) {
      if (C.attnLayers.includes(l)) _kv[l] = { k: mk(MAX_SEQ * C.nKvHeads * C.headDim, 'k' + l), v: mk(MAX_SEQ * C.nKvHeads * C.headDim, 'v' + l) };
      else _conv[l] = { a: mk(C.hidden * (C.convL - 1), 'ca' + l), b: mk(C.hidden * (C.convL - 1), 'cb' + l), cur: 0 };
    }
  }
  function _freeState() {
    const kill = (o) => { for (const k in o) { const v = o[k]; if (v && v.destroy) try { v.destroy(); } catch (_) {} else if (v && typeof v === 'object') kill(v); } };
    if (_scr) kill(_scr); if (_kv) kill(_kv); if (_conv) kill(_conv);
    if (_idsBuf) try { _idsBuf.destroy(); } catch (_) {}
    try { if (_amaxV) _amaxV.destroy(); if (_amaxI) _amaxI.destroy(); } catch (_) {}
    try { if (_dp4) { _dp4.xq.destroy(); _dp4.xs.destroy(); } } catch (_) {}
    try { for (const sl of _slotPool) for (const b of Object.values(sl)) { if (b && b.destroy) b.destroy(); } } catch (_) {}
    _scr = null; _kv = null; _conv = null; _idsBuf = null; _amaxV = null; _amaxI = null; _dp4 = null;
    _slotPool = []; _elru = null; _elruFree = []; _expertCatalog = {};
  }

  // One forward over T tokens at absolute positions [posBase, posBase+T).
  // opts.chain: embed reads the GPU token history at posBase (no CPU ids) — for chained decode.
  // opts.argmax: GPU-argmax the final logits into tokHist[posBase+T] (so the next chained
  //   forward's embed reads it) instead of leaving logits for a CPU readback.
  // opts.batched: caller owns beginBatch/endBatch (lets many forwards chain in one submit).
  async function forward(idsArr, posBase, opts) {
    opts = opts || {};
    const C = _cfg, H = C.hidden, T = opts.chain ? 1 : idsArr.length, S = posBase + T;
    const W = (n) => _weights[n];
    _ensureState();
    const s = _scr;
    if (!opts.chain) {
      // Sized for the prefill chunk (was MATVEC_MAXT=32 → the writeBuffer of a >64-token chunk
      // silently overflowed the 256-byte buffer and was rejected → wrong embeddings → garbage).
      if (!_idsBuf) _idsBuf = E.createBuffer(Math.max(MATVEC_MAXT, PREFILL_T) * 4, U.STORAGE | U.COPY_DST, 'ids');
      E.device().queue.writeBuffer(_idsBuf, 0, new Uint32Array(idsArr));
    }
    uniformReset();
    if (!opts.batched) E.beginBatch();
    if (opts.chain) await embedGather(s.tokHist, W('model.embed_tokens.weight').buf, s.x, T, H, posBase);
    else await embedGather(_idsBuf, W('model.embed_tokens.weight').buf, s.x, T, H, 0);
    for (let l = 0; l < C.numLayers; l++) {
      const p = 'model.layers.' + l + '.';
      await rmsnorm(s.x, W(p + 'operator_norm.weight').buf, s.normed, T, H, C.rmsEps);
      if (_conv[l]) {   // conv mixer
        await mv(s.normed, W(p + 'conv.in_proj.weight'), s.bcx, T, 3 * H, H);
        const st = _conv[l];
        const sIn = st.cur === 0 ? st.a : st.b, sOut = st.cur === 0 ? st.b : st.a;
        await shortConv(s.bcx, W(p + 'conv.conv.weight').buf, sIn, sOut, s.convy, T, H, C.convL, posBase > 0);
        st.cur ^= 1;
        await mv(s.convy, W(p + 'conv.out_proj.weight'), s.x, T, H, H, true);   // + residual
      } else {          // GQA attention
        const nHq = C.nHeads, nKv = C.nKvHeads, hd = C.headDim;
        await mv(s.normed, W(p + 'self_attn.q_proj.weight'), s.q, T, nHq * hd, H);
        await mv(s.normed, W(p + 'self_attn.k_proj.weight'), s.k, T, nKv * hd, H);
        await mv(s.normed, W(p + 'self_attn.v_proj.weight'), s.v, T, nKv * hd, H);
        await ropeQK(s.q, W(p + 'self_attn.q_layernorm.weight').buf, s.qr, T, nHq, hd, posBase, C.ropeTheta, C.rmsEps);
        await ropeQK(s.k, W(p + 'self_attn.k_layernorm.weight').buf, s.kr, T, nKv, hd, posBase, C.ropeTheta, C.rmsEps);
        E.copyBuffer(s.kr, 0, _kv[l].k, posBase * nKv * hd * 4, T * nKv * hd * 4);
        E.copyBuffer(s.v, 0, _kv[l].v, posBase * nKv * hd * 4, T * nKv * hd * 4);
        await attention(s.qr, _kv[l].k, _kv[l].v, s.attn, T, S, nHq, nKv, hd);
        await mv(s.attn, W(p + 'self_attn.out_proj.weight'), s.x, T, H, nHq * hd, true);   // + residual
      }
      await rmsnorm(s.x, W(p + 'ffn_norm.weight').buf, s.normed, T, H, C.rmsEps);
      if (C.moe && l >= C.denseLayers) {
        // Mixture-of-experts FFN (T=1). BRING-UP dispatch: compute the router on-GPU, read
        // back the 4 chosen expert ids + weights, then run those 4 standard SwiGLU experts
        // and weight-sum them into the residual. Each expert reuses the dense matvecQ/swiglu
        // path (the checkpoint stores per-expert w1/w2/w3, not a fused tensor). The per-layer
        // readback is the correctness-first shortcut; the GPU-resident indexed dispatch over
        // packed expert tensors (no readback) is the queued optimization.
        const nE = C.nExperts, K = C.topK, eI = C.expertI;
        await mv(s.normed, W(p + 'feed_forward.gate.weight'), s.rlogits, T, nE, H);   // router logits [T,nE]
        if (!_streamExperts) await router(s.rlogits, W(p + 'feed_forward.expert_bias').buf, s.ridx, s.rwt, T, nE, K);
        if (_int3Mode && _weights['moe3.' + l + '.w1']) {
          const m1 = W('moe3.' + l + '.w1'), m3 = W('moe3.' + l + '.w3'), m2 = W('moe3.' + l + '.w2');
          const P = T * K;
          if (T > 1 && !globalThis.__forceGemv) {
            // PREFILL: token-sorted per-expert tiled GEMM (mul_mat_id style). At a big chunk
            // (~PREFILL_T/nE tokens/expert) the shared-mem weight tile is reused across all of an
            // expert's token columns — the batched-prefill amortization the pair GEMV can't get.
            await moeGemm3(s.normed, T, m1, s.ridx, s.moeGate, P, eI, H, K);   // srcDiv=K (per-token in)
            await moeGemm3(s.normed, T, m3, s.ridx, s.moeUp, P, eI, H, K);
            await swiglu(s.moeGate, s.moeUp, s.moeAct, P * eI);
            await moeGemm3(s.moeAct, P, m2, s.ridx, s.moeOut, P, H, eI, 1);    // srcDiv=1 (per-pair in)
          } else {
            // DECODE (T=1): pair-indexed GEMV — only 4 pairs, sorting/tiling isn't worth it.
            await idxGemv3(s.normed, m1, s.ridx, s.moeGate, P, eI, H, H, K);
            await idxGemv3(s.normed, m3, s.ridx, s.moeUp, P, eI, H, H, K);
            await swiglu(s.moeGate, s.moeUp, s.moeAct, P * eI);
            await idxGemv3(s.moeAct, m2, s.ridx, s.moeOut, P, H, eI, eI, 1);
          }
          await moeCombine(s.x, s.moeOut, s.rwt, H, K, T);                     // x[t] += Σ_k wt[t,k]·out[t,k]
        } else if (_streamExperts) {
          // STREAMING + CACHE-AWARE ROUTING: read back the 32 RAW router logits (one round-trip/layer
          // via a persistent staging buffer) and do the top-K selection on the CPU with a residency
          // bonus λ, so borderline picks prefer experts already in the LRU → fewer disk misses. The
          // per-slot path consumes CPU idx/wt directly, so no writeBuffer back to the GPU is needed.
          const _ts = performance.now();
          if (!_logitStage) _logitStage = E.createBuffer(nE * 4, U.COPY_DST | U.MAP_READ, 'logitStage');
          E.copyBuffer(s.rlogits, 0, _logitStage, 0, nE * 4);
          E.endBatch();   // submit only (do NOT await the drain — the mapAsync below waits for the copy)
          await _logitStage.mapAsync(GPUMapMode.READ);
          const lg = new Float32Array(_logitStage.getMappedRange().slice(0, nE * 4));
          _logitStage.unmap();
          _pstat.syncMs += performance.now() - _ts;
          const { idx, wt } = _cacheAwareSelect(l, lg, nE, K);
          const _td = performance.now();
          const slots = await ensureExpertBatch(l, idx);   // sync slot assign + concurrent miss reads
          _pstat.diskMs += performance.now() - _td; _pstat.layers++;
          E.beginBatch();
          for (let k = 0; k < K; k++) {
            const sl = slots[k];
            await mv(s.normed, { pack: sl.w1p, scales: sl.w1s }, s.gate, 1, eI, H);
            await mv(s.normed, { pack: sl.w3p, scales: sl.w3s }, s.up, 1, eI, H);
            await swiglu(s.gate, s.up, s.swi, eI);
            await mv(s.swi, { pack: sl.w2p, scales: sl.w2s }, s.eout, 1, H, eI);
            await axpy(s.x, s.eout, H, wt[k]);
          }
        } else if (_weights['moe.' + l + '.w1']) {
          // GPU-RESIDENT indexed dispatch — no readback, whole block stays batched.
          const w1 = W('moe.' + l + '.w1'), w3 = W('moe.' + l + '.w3'), w2 = W('moe.' + l + '.w2');
          await idxGemv(s.normed, w1, s.ridx, s.moeGate, K, eI, H, 0);        // [K, eI]
          await idxGemv(s.normed, w3, s.ridx, s.moeUp, K, eI, H, 0);          // [K, eI]
          await swiglu(s.moeGate, s.moeUp, s.moeAct, K * eI);
          await idxGemv(s.moeAct, w2, s.ridx, s.moeOut, K, H, eI, eI);        // [K, H], per-k input stride eI
          await moeCombine(s.x, s.moeOut, s.rwt, H, K);                       // x += Σ_k wt[k]·out[k]
        } else {
          // FALLBACK (__noMoePack): readback the routing, run the per-expert buffers.
          await E.endBatch();
          const idx = new Uint32Array((await E.readF32(s.ridx, K)).buffer);
          const wt = await E.readF32(s.rwt, K);
          E.beginBatch();
          for (let k = 0; k < K; k++) {
            const ep = p + 'feed_forward.experts.' + idx[k] + '.';
            await matvecQ(s.normed, W(ep + 'w1.weight'), s.gate, 1, eI, H);
            await matvecQ(s.normed, W(ep + 'w3.weight'), s.up, 1, eI, H);
            await swiglu(s.gate, s.up, s.swi, eI);
            await matvecQ(s.swi, W(ep + 'w2.weight'), s.eout, 1, H, eI);
            await axpy(s.x, s.eout, H, wt[k]);
          }
        }
      } else {
        const I = C.ffI;
        await mv(s.normed, W(p + 'feed_forward.w1.weight'), s.gate, T, I, H);
        await mv(s.normed, W(p + 'feed_forward.w3.weight'), s.up, T, I, H);
        await swiglu(s.gate, s.up, s.swi, T * I);
        await mv(s.swi, W(p + 'feed_forward.w2.weight'), s.x, T, H, I, true);              // + residual
      }
    }
    await rmsnorm(s.x, W('model.embedding_norm.weight').buf, s.normed, T, H, C.rmsEps);
    E.copyBuffer(s.normed, (T - 1) * H * 4, s.last, 0, H * 4);
    await mv(s.last, W('lm_head.weight'), s.logits, 1, C.vocab, H);
    if (opts.argmax) await argmaxKernel(s.logits, s.tokHist, C.vocab, posBase + T);   // GPU argmax → tokHist (no CPU roundtrip)
    if (!opts.batched) await E.endBatch();
  }

  async function readU32Range(buf, idx, count) {
    const bytes = count * 4;
    const staging = E.createBuffer(bytes, U.COPY_DST | U.MAP_READ, 'rd');
    const enc = E.device().createCommandEncoder();
    enc.copyBufferToBuffer(buf, idx * 4, staging, 0, bytes);
    E.device().queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return out;
  }

  const GEN_BATCH = 8;    // decode tokens chained GPU-resident per submit (1 readback per batch)
  const PIPE_DEPTH = 3;   // batches kept in flight so CPU-encode + GPU-compute + readback overlap
  // Deep-pipelined decode: submit up to PIPE_DEPTH batches WITHOUT draining, then await the
  // oldest batch's readback while the GPU chews the queued ones and the CPU encodes the next.
  // This is the throttle fix — the old loop awaited each batch's drain, so CPU-encode (~58ms)
  // and GPU-compute (~58ms) ran serially with the vsync-paced readback idle in between.
  // Pooled uniforms stay correct under pipelining: writeBuffer is queue-ordered and the engine
  // flushes every 32 ops, so each batch's dispatches consume their uniform values before the
  // next batch overwrites the pool. emitTok(t) → false to stop. Returns when done/stopped.
  async function _pipeDecode(startPos, maxTokens, emitTok, signal) {
    let pos = startPos, submitted = 0;
    const submitBatch = async () => {
      if (signal && signal.aborted) return null;
      const K = Math.min(GEN_BATCH, maxTokens - submitted, MAX_SEQ - 1 - pos);
      if (K <= 0) return null;
      const base = pos;
      E.beginBatch();
      for (let k = 0; k < K; k++) await forward(null, base + k, { chain: true, argmax: true, batched: true });
      E.endBatch();   // submit; DON'T await the drain (the readback below implies completion)
      pos += K; submitted += K;
      return { read: readU32Range(_scr.tokHist, base + 1, K), K };
    };
    const inflight = [];
    while (inflight.length < PIPE_DEPTH) { const b = await submitBatch(); if (!b) break; inflight.push(b); }
    while (inflight.length) {
      const cur = inflight.shift();
      const toks = await cur.read;
      let stop = false;
      for (let k = 0; k < cur.K; k++) { if (!emitTok(toks[k])) { stop = true; break; } }
      if (stop) break;
      const b = await submitBatch(); if (b) inflight.push(b);
    }
  }
  async function generate(prompt, { maxTokens = 64, onToken, signal } = {}) {
    if (!_loaded) throw new Error('loadModel first');
    const C = _cfg;
    // prompt may be a full message array (system/user/assistant/…) or a bare user string.
    const ids = Array.isArray(prompt) ? TOK.encodeChat(prompt) : TOK.encodeChat([{ role: 'user', content: prompt }]);
    if (ids.length + maxTokens + 2 > MAX_SEQ) throw new Error('prompt too long: ' + ids.length + ' tokens + ' + maxTokens + ' max new > context ' + MAX_SEQ + ' (raise globalThis.__lfmMaxSeq and reload)');
    _estat = { hit: 0, miss: 0 };   // expert LRU stats for this generation
    _pstat = { syncMs: 0, diskMs: 0, layers: 0, covLayers: 0 };   // streaming timing + coverage for this generation
    // chained decode needs a single-submit forward (no mid-forward readback). Streaming and the
    // __noMoePack fallback both drain per MoE layer, so they use the per-token CPU-argmax path.
    const chainable = !(C.moe && (globalThis.__noMoePack || _streamExperts));
    let pos = 0;
    if (chainable && C.moe && !_int3Mode) {
      // PIPELINED T=1 MoE PREFILL (int4 all-resident debug path): chained forwards in GEN_BATCH
      // groups, one drain per group. int3 uses the much faster T=32 chunked prefill below.
      _ensureState();
      E.device().queue.writeBuffer(_scr.tokHist, 0, new Uint32Array(ids));
      while (pos < ids.length) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const n = Math.min(GEN_BATCH, ids.length - pos);
        E.beginBatch();
        for (let i = 0; i < n; i++) await forward(null, pos + i, { chain: true, batched: true, argmax: pos + i === ids.length - 1 });
        await E.endBatch();
        pos += n;
      }
    } else {
      // CHUNKED PREFILL: int3 MoE runs big PREFILL_T chunks (token-sorted per-expert GEMM);
      // dense chunks at MATVEC_MAXT; streaming MoE stays T=1 (its readback path is per-token).
      const CH = _int3Mode ? PREFILL_T : ((C.moe && !_int3Mode) ? 1 : MATVEC_MAXT);
      for (let off = 0; off < ids.length; off += CH) {   // prefill; last forward GPU-argmaxes → tokHist[L]
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const chunk = ids.slice(off, Math.min(off + CH, ids.length));
        await forward(chunk, pos, { argmax: chainable && (off + CH >= ids.length) });
        pos += chunk.length;
      }
    }
    _pstat = { syncMs: 0, diskMs: 0, layers: 0, covLayers: 0 };   // reset after prefill → decode-only split
    const outIds = [], imEnd = TOK.imEnd();
    if (chainable) {
      // The FIRST token was argmax'd by the last prefill forward into tokHist[L]; emit it, then
      // the deep-pipelined loop consumes it as input and produces tokHist[L+1..].
      const t0 = performance.now();
      let stopped = false;
      { const f = (await readU32Range(_scr.tokHist, ids.length, 1))[0];
        if (f === C.eos || f === imEnd) stopped = true;
        else { outIds.push(f); if (onToken) { try { onToken(TOK.decode([f]), f); } catch (_) {} } } }
      if (!stopped) {
        await _pipeDecode(pos, maxTokens, (t) => {
          if (t === C.eos || t === imEnd) return false;
          outIds.push(t); if (onToken) { try { onToken(TOK.decode(outIds.slice(-4)).slice(-24), t); } catch (_) {} }
          return outIds.length < maxTokens;
        }, signal);
      }
      _lastProf = { mode: 'pipelined', batch: GEN_BATCH, depth: PIPE_DEPTH, tokps: +(outIds.length / Math.max(1e-3, (performance.now() - t0) / 1000)).toFixed(2) };
    } else {
      let _amMs = 0, _fwdMs = 0;   // fallback: per-token CPU argmax (readback path)
      for (let i = 0; i < maxTokens; i++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        const ta = performance.now();
        const logits = await E.readF32(_scr.logits, C.vocab);
        let best = 0, bv = -Infinity;
        for (let j = 0; j < C.vocab; j++) if (logits[j] > bv) { bv = logits[j]; best = j; }
        _amMs += performance.now() - ta;
        if (best === C.eos || best === imEnd) break;
        outIds.push(best);
        if (onToken) { try { onToken(TOK.decode(outIds.slice(-4)).slice(-24), best); } catch (_) {} }
        const tf = performance.now();
        await forward([best], pos); pos++;
        _fwdMs += performance.now() - tf;
      }
      const _n = Math.max(1, outIds.length);
      _lastProf = { mode: 'cpu-argmax', argmax_ms_per_tok: +(_amMs / _n).toFixed(1), forward_ms_per_tok: +(_fwdMs / _n).toFixed(1), sync_ms_per_tok: +(_pstat.syncMs / _n).toFixed(1), disk_ms_per_tok: +(_pstat.diskMs / _n).toFixed(1), moe_layers_per_tok: +(_pstat.layers / _n).toFixed(1), lambda: +globalThis.__cacheRouteLambda || 0, layer_coverage: +(_pstat.covLayers / Math.max(1, _pstat.layers)).toFixed(3) };
    }
    try { console.log('[lfm25 prof] ' + JSON.stringify(_lastProf)); } catch (_) {}
    return TOK.decode(outIds);
  }
  let _lastProf = null;

  // GPU-timestamp per-kernel breakdown of ONE decode forward (the real bottleneck map).
  async function _benchDecode(prompt) {
    if (!_loaded) throw new Error('loadModel first');
    const C = _cfg;
    const ids = TOK.encodeChat([{ role: 'user', content: (prompt || 'Hello, tell me about yourself.') }]);
    let pos = 0; const CH = C.moe ? 1 : MATVEC_MAXT;
    for (let off = 0; off < ids.length; off += CH) { const chunk = ids.slice(off, Math.min(off + CH, ids.length)); await forward(chunk, pos, { argmax: off + CH >= ids.length }); pos += chunk.length; }
    E.beginBatch(); await forward(null, pos, { chain: true, argmax: true, batched: true }); await E.endBatch(); pos++;   // warm
    E.beginProfile(1024); E.beginBatch();
    await forward(null, pos, { chain: true, argmax: true, batched: true });
    await E.endBatch();
    const prof = await E.endProfile();
    const agg = {}; let total = 0;
    for (const r of prof) { agg[r.label] = (agg[r.label] || 0) + r.us; total += r.us; }
    const rows = Object.entries(agg).map(([k, v]) => ({ k, us: +v.toFixed(0), n: prof.filter(x => x.label === k).length })).sort((a, b) => b.us - a.us);
    return { total_ms: +(total / 1000).toFixed(1), n_dispatch: prof.length, rows };
  }

  // ============================================================
  // App integration: the engine-facing surface conversations.js/providers.js use, mirroring
  // webgpu-qwen3's (DEFAULT_MODELS, DEFAULT_N_CTX, setToolRunner, runConversation, unload).
  // The main-thread shim (webgpu-host.js) forwards to a Worker (webgpu-worker.js) that runs
  // THIS engine off the main thread — so the GPU decode loop can't be starved by page compositing.
  // ============================================================
  const DEFAULT_MODELS = [
    { id: 'lfm25-8b', modelId: '8B-A1B', label: 'LFM2.5-8B-A1B MoE (int3, ~4.3GB — ~15 tok/s)' },
  ];
  const DEFAULT_N_CTX = 8192;
  const isVariant = (v) => v && Object.prototype.hasOwnProperty.call(MODELS, v);

  let _toolRunner = null;
  function setToolRunner(fn) { _toolRunner = (typeof fn === 'function') ? fn : null; }

  // Hermes tool block — the same format webgpu-qwen3 uses; LFM2.5 is a capable instruct model
  // and follows it via the system prompt. (Its native Pythonic call format is a later refinement.)
  function toolPreamble(tools) {
    const fns = (tools || []).filter(t => t && t.type === 'function' && t.function);
    if (!fns.length) return '';
    const sigs = fns.map(t => JSON.stringify({ type: 'function', function: t.function })).join('\n');
    return ['# Tools', '', 'You may call one or more functions to assist with the user query.', '',
      'You are provided with function signatures within <tools></tools> XML tags:', '<tools>', sigs, '</tools>', '',
      'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:',
      '<tool_call>', '{"name": <function-name>, "arguments": <args-json-object>}', '</tool_call>'].join('\n');
  }
  // Streaming splitter: <think>…</think> → reasoning, <tool_call>…</tool_call> → buffered payload,
  // else content. 12-char guard tail so a tag split across token pieces is still caught.
  function makeRoundParser(onReason, onContent) {
    let buf = '', state = 'normal', cur = '';
    const acc = { content: '', reasoning: '', toolCalls: [] };
    const NEEDLES = { normal: ['<think>', '<tool_call>'], think: ['</think>'], tool: ['</tool_call>'] };
    const out = (text) => { if (!text) return; if (state === 'think') { acc.reasoning += text; onReason(text); } else if (state === 'tool') { cur += text; } else { acc.content += text; onContent(text); } };
    const step = () => {
      for (;;) {
        let bi = -1, bn = null;
        for (const n of NEEDLES[state]) { const idx = buf.indexOf(n); if (idx !== -1 && (bi === -1 || idx < bi)) { bi = idx; bn = n; } }
        if (bi === -1) break;
        out(buf.slice(0, bi)); buf = buf.slice(bi + bn.length);
        if (bn === '<think>') state = 'think'; else if (bn === '</think>') state = 'normal';
        else if (bn === '<tool_call>') { state = 'tool'; cur = ''; } else if (bn === '</tool_call>') { state = 'normal'; if (cur.trim()) acc.toolCalls.push(cur.trim()); cur = ''; }
      }
      if (buf.length > 12) { out(buf.slice(0, buf.length - 12)); buf = buf.slice(buf.length - 12); }
    };
    return { push(t) { buf += t; step(); }, flush() { out(buf); buf = ''; if (state === 'tool' && cur.trim()) acc.toolCalls.push(cur.trim()); },
      get content() { return acc.content; }, get reasoning() { return acc.reasoning; }, get toolCalls() { return acc.toolCalls; }, get state() { return state; } };
  }
  const tcText = (tcs) => (tcs || []).map(tc => '<tool_call>\n{"name": "' + ((tc.function && tc.function.name) || '') + '", "arguments": ' + ((tc.function && tc.function.arguments) || '{}') + '}\n</tool_call>').join('\n');
  const normMsg = (m) => {
    let c = m.content;
    if (Array.isArray(c)) c = c.filter(p => p && p.type === 'text').map(p => p.text || '').join('\n');
    c = (c == null) ? '' : String(c);
    if (m.role === 'tool') return { role: 'user', content: '<tool_response>\n' + c + '\n</tool_response>' };
    if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) c = c ? (c + '\n' + tcText(m.tool_calls)) : tcText(m.tool_calls);
    return { role: m.role, content: c };
  };

  async function runConversation({ provider, messages, systemPrompt, tools, convId, signal }, emit) {
    const want = (provider && (provider.endpoint || provider.modelId)) || '';
    const variant = isVariant(want) ? want : '8B-A1B';
    try {
      emit({ type: 'info', message: 'Loading LFM2.5-' + variant + ' (WebGPU)… first run downloads + quantizes; cached after.' });
      let lastPct = -1;
      await loadModel({ variant, onProgress: (p) => {
        if (!p) return;
        if (p.phase === 'download') { const pct = p.pct | 0; if (pct === lastPct) return; lastPct = pct; emit({ type: 'info', message: 'Downloading weights… ' + pct + '%' }); }
        else if (p.phase === 'int3') emit({ type: 'info', message: 'Quantizing experts to int3… ' + (p.pct || 0) + '% (one-time; cached)' });
        else if (p.phase === 'cache' || p.phase === 'folder') emit({ type: 'info', message: 'Loading from cache… ' + (p.pct || 0) + '%' });
        else if (p.phase === 'tokenizer') emit({ type: 'info', message: 'Loading tokenizer…' });
      } });
    } catch (e) {
      emit({ type: 'info', message: null });
      if (e && e.name === 'AbortError') throw e;
      emit({ type: 'error', message: 'LFM2.5: ' + ((e && e.message) || e) }); emit({ type: 'agent_done' }); return;
    }
    const ctxCap = MAX_SEQ;
    let sys = (systemPrompt && typeof systemPrompt === 'object') ? (systemPrompt.content || '') : (systemPrompt || '');
    const toolList = (Array.isArray(tools) ? tools : []).filter(t => t && t.type === 'function');
    const pre = toolPreamble(toolList);
    if (pre) sys = sys ? (sys + '\n\n' + pre) : pre;
    const work = [];
    if (sys) work.push({ role: 'system', content: sys });
    for (const m of (messages || [])) { if (m && m.role) work.push(normMsg(m)); }

    const MAX_ROUNDS = toolList.length ? 8 : 1;
    try {
      for (let round = 0; round < MAX_ROUNDS; round++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        emit({ type: 'round_start' });
        let firstTok = false;
        const clearInfo = () => { if (!firstTok) { firstTok = true; emit({ type: 'info', message: null }); } };
        const parser = makeRoundParser(
          (rz) => { clearInfo(); emit({ type: 'delta', delta: { reasoning: rz } }); },
          (ct) => { clearInfo(); emit({ type: 'delta', delta: { content: ct } }); },
        );
        // Budget: leave room under the context window for the encoded prompt. Clean incremental
        // decode in the callback (re-decode the running id list → emit only the new suffix) so
        // multibyte UTF-8 that spans tokens renders correctly and the parser sees clean deltas.
        const encLen = TOK.encodeChat(work).length;
        let budget = (provider && (provider.maxTokens | 0)) || 0;
        const room = Math.max(16, ctxCap - encLen - 8);
        budget = budget ? Math.min(budget, room) : room;
        const outIds = []; let prevLen = 0;
        await generate(work, { maxTokens: budget, signal, onToken: (_txt, id) => {
          outIds.push(id); const full = TOK.decode(outIds); const d = full.slice(prevLen); prevLen = full.length; parser.push(d);
        } });
        parser.flush();
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        emit({ type: 'info', message: null });

        let content = parser.content.replace(/^\s+/, '');
        const toolCalls = [];
        parser.toolCalls.forEach((raw, k) => { try { const o = JSON.parse(raw); if (o && o.name) toolCalls.push({ id: 'call_' + round + '_' + k, type: 'function', function: { name: o.name, arguments: JSON.stringify(o.arguments || {}) } }); } catch (_) {} });
        if (!content && !toolCalls.length) {
          content = (parser.state === 'think' || parser.reasoning.length > 0)
            ? '⚠️ I ran out of room to answer near the model\'s context limit (' + ctxCap + ' tokens). Start a new chat or remove a large earlier message.'
            : '⚠️ The model produced no output. Try resending, or start a new chat if the history is very long.';
          emit({ type: 'delta', delta: { content } });
        }
        if (toolCalls.length) emit({ type: 'delta', delta: { tool_calls: toolCalls.map((tc, i) => ({ index: i, id: tc.id, type: 'function', function: { name: tc.function.name, arguments: tc.function.arguments } })) } });
        emit({ type: 'round_end', content });
        const asst = { role: 'assistant', content };
        if (toolCalls.length) asst.tool_calls = toolCalls;
        emit({ type: 'message_added', message: asst });
        work.push(normMsg(asst));
        if (!toolCalls.length) break;
        if (!_toolRunner) { emit({ type: 'info', message: null }); break; }
        for (const tc of toolCalls) {
          if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
          emit({ type: 'tool_started', tc });
          let args = {}; try { args = JSON.parse(tc.function.arguments || '{}'); } catch (_) {}
          let out;
          try { out = await _toolRunner(tc.function.name, args, convId, signal); }
          catch (e) { if (e && e.name === 'AbortError') throw e; out = { result: 'Error: ' + ((e && e.message) || e) }; }
          const toolResult = (out && out.result != null) ? out.result : '';
          emit({ type: 'tool_result', id: tc.id, result: toolResult, artifacts: out && out.artifacts });
          work.push(normMsg({ role: 'tool', tool_call_id: tc.id, content: toolResult }));
          emit({ type: 'message_added', message: { role: 'tool', tool_call_id: tc.id, content: toolResult } });
        }
      }
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      emit({ type: 'info', message: null });
      emit({ type: 'error', message: 'LFM2.5: ' + ((e && e.message) || e) });
    }
    emit({ type: 'agent_done' });
  }

  return { CONFIG, MODELS, DEFAULT_MODELS, DEFAULT_N_CTX, shortConv, router, selfTestKernels, loadModel, unload: () => { _freeState(); unload(); }, inventory, TOK, isLoaded: () => _loaded, variant: () => _variant, generate, forward, runConversation, setToolRunner, lastProf: () => _lastProf, expertStats: () => _estat, connectModelFolder, forgetModelFolder, folderStatus, exportToFolder, selfTestInt3, selfTestMoeGemm, _benchDecode };
})();

if (typeof window !== 'undefined') window.SandpieLfm25 = SandpieLfm25;
if (typeof self !== 'undefined') self.SandpieLfm25 = SandpieLfm25;
