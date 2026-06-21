// sandpie/modules/gguf.js — GGUF container parser + llama.cpp k-quant (Q4_K / Q6_K)
// decode, for the native-GGUF WebGPU engine fork (window.SandpieQwen35GGUF).
//
// WHY: the int4 engine (webgpu-qwen35.js) uses OUR own symmetric group-32 int4, quantized
// in-browser from bf16 safetensors (the ~40s wait) or read from compressed-tensors. This
// module instead reads the STANDARD llama.cpp Q4_K_M GGUF directly — no quantize step, and
// access to the whole community GGUF ecosystem. Q4_K_M is a MIX: most tensors Q4_K, a few
// (attn_v, ffn_down in some layers; output/embedding) Q6_K, plus F32/F16 norms.
//
// Block formats (QK_K=256), verbatim from ggml-common.h / ggml-quants.c:
//   Q4_K (144 B/256): f16 d, f16 dmin, u8 scales[12] (8×6-bit scale + 8×6-bit min, packed),
//                     u8 qs[128] (256 4-bit quants). x = d*sc*(q&0xF or q>>4) - dmin*m.
//   Q6_K (210 B/256): u8 ql[128], u8 qh[64], int8 scales[16], f16 d. x = d*scale*(q-32),
//                     q = (ql nibble) | (qh 2 bits)<<4.
//
// GPU ALIGNMENT: Q4_K's 144 B is u32-aligned, but Q6_K's 210 B is not (block sb starts at
// byte sb*210, only 2-aligned). So Q6_K blocks are repacked into 212-B (53×u32) slots on
// load; every field then starts on a 4-byte boundary and the WGSL reads array<u32> cleanly.

const SandpieGGUF = (function () {
  'use strict';

  // ggml_type enum (the subset we touch). Full list in ggml.h.
  const GGML = { F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
    Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15, I8: 16, I16: 17, I32: 18 };
  const TYPE_NAME = { 0: 'f32', 1: 'f16', 8: 'q8_0', 12: 'q4_K', 13: 'q5_K', 14: 'q6_K' };
  // elements-per-block and bytes-per-block for the types we load.
  const BLK = {
    0:  { els: 1,   bytes: 4 },     // F32
    1:  { els: 1,   bytes: 2 },     // F16
    8:  { els: 32,  bytes: 34 },    // Q8_0: f16 d + 32×int8
    12: { els: 256, bytes: 144 },   // Q4_K
    13: { els: 256, bytes: 176 },   // Q5_K
    14: { els: 256, bytes: 210 },   // Q6_K
  };

  // ---- f16 → f32 (scalar, for the JS dequant reference) -----------------------
  function f16(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  const i8 = (b) => (b < 128 ? b : b - 256);

  // ============================================================
  // GGUF parser (v2/v3, little-endian). Returns header + tensor table; does NOT
  // copy tensor data (caller slices lazily from the same buffer/source).
  // ============================================================
  const GV = { U8: 0, I8: 1, U16: 2, I16: 3, U32: 4, I32: 5, F32: 6, BOOL: 7, STRING: 8, ARRAY: 9, U64: 10, I64: 11, F64: 12 };

  function parse(buf) {
    const dv = new DataView(buf);
    let p = 0;
    const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
    const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
    const i64 = () => { const v = Number(dv.getBigInt64(p, true)); p += 8; return v; };
    const str = () => { const n = u64(); const s = new TextDecoder().decode(new Uint8Array(buf, p, n)); p += n; return s; };
    const readVal = (t) => {
      switch (t) {
        case GV.U8: { const v = dv.getUint8(p); p += 1; return v; }
        case GV.I8: { const v = dv.getInt8(p); p += 1; return v; }
        case GV.U16: { const v = dv.getUint16(p, true); p += 2; return v; }
        case GV.I16: { const v = dv.getInt16(p, true); p += 2; return v; }
        case GV.U32: return u32();
        case GV.I32: { const v = dv.getInt32(p, true); p += 4; return v; }
        case GV.F32: { const v = dv.getFloat32(p, true); p += 4; return v; }
        case GV.BOOL: { const v = dv.getUint8(p) !== 0; p += 1; return v; }
        case GV.STRING: return str();
        case GV.U64: return u64();
        case GV.I64: return i64();
        case GV.F64: { const v = dv.getFloat64(p, true); p += 8; return v; }
        case GV.ARRAY: { const et = u32(); const n = u64(); const a = new Array(n); for (let i = 0; i < n; i++) a[i] = readVal(et); return a; }
        default: throw new Error('gguf: bad value type ' + t);
      }
    };

    const magic = u32();
    if (magic !== 0x46554747) throw new Error('gguf: bad magic 0x' + magic.toString(16) + ' (not a GGUF file)');
    const version = u32();
    if (version < 2 || version > 3) throw new Error('gguf: unsupported version ' + version);
    const nTensors = u64();
    const nKV = u64();

    const meta = {};
    for (let i = 0; i < nKV; i++) { const key = str(); const t = u32(); meta[key] = readVal(t); }

    const tensors = [];
    for (let i = 0; i < nTensors; i++) {
      const name = str();
      const nd = u32();
      const dims = []; for (let d = 0; d < nd; d++) dims.push(u64());
      const type = u32();
      const offset = u64();
      tensors.push({ name, dims, type, offset });
    }

    const alignment = (meta['general.alignment'] | 0) || 32;
    const dataStart = (p + alignment - 1) & ~(alignment - 1);   // tensor data begins here, aligned
    return { version, alignment, meta, tensors, dataStart };
  }

  // bytes of a tensor's data = blocks * bytesPerBlock (blocks = numel / elsPerBlock).
  function tensorByteLen(t) {
    const b = BLK[t.type]; if (!b) throw new Error('gguf: unsupported tensor type ' + t.type + ' for ' + t.name);
    const numel = t.dims.reduce((a, x) => a * x, 1);
    return (numel / b.els) * b.bytes;
  }

  // ============================================================
  // k-quant scale/min extraction (get_scale_min_k4) + JS dequant references.
  // These mirror ggml-quants.c byte-for-byte and are the ground truth the WGSL
  // kernels are self-tested against.
  // ============================================================
  // q = u8 view, qb = byte offset of scales[0]. Returns [scale6, min6] for sub-block j (0..7).
  function getScaleMinK4(j, q, qb) {
    if (j < 4) return [q[qb + j] & 63, q[qb + j + 4] & 63];
    return [(q[qb + j + 4] & 0xF) | ((q[qb + j - 4] >> 6) << 4),
            (q[qb + j + 4] >> 4)  | ((q[qb + j]     >> 6) << 4)];
  }

  // Dequantize nb Q4_K blocks (u8 view starting at the tensor data) → Float32Array(nb*256).
  function dequantQ4K(u8, nb) {
    const out = new Float32Array(nb * 256);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let o = 0;
    for (let i = 0; i < nb; i++) {
      const base = i * 144;
      const d = f16(dv.getUint16(base, true)), dmin = f16(dv.getUint16(base + 2, true));
      const scb = base + 4, qsb = base + 16;
      let is = 0;
      for (let j = 0; j < 256; j += 64) {
        let [sc, m] = getScaleMinK4(is, u8, scb); const d1 = d * sc, m1 = dmin * m;
        [sc, m] = getScaleMinK4(is + 1, u8, scb); const d2 = d * sc, m2 = dmin * m;
        const q = qsb + (j / 64) * 32;
        for (let l = 0; l < 32; l++) out[o++] = d1 * (u8[q + l] & 0xF) - m1;
        for (let l = 0; l < 32; l++) out[o++] = d2 * (u8[q + l] >> 4) - m2;
        is += 2;
      }
    }
    return out;
  }

  // Dequantize nb Q5_K blocks (176 B: f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]).
  // Same scale/min packing as Q4_K; qh supplies a 5th (high) bit per weight.
  function dequantQ5K(u8, nb) {
    const out = new Float32Array(nb * 256);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let o = 0;
    for (let i = 0; i < nb; i++) {
      const base = i * 176;
      const d = f16(dv.getUint16(base, true)), dmin = f16(dv.getUint16(base + 2, true));
      const scb = base + 4, qhb = base + 16, qsb = base + 48;
      let is = 0, u1 = 1, u2 = 2;
      for (let j = 0; j < 256; j += 64) {
        let [sc, m] = getScaleMinK4(is, u8, scb); const d1 = d * sc, m1 = dmin * m;
        [sc, m] = getScaleMinK4(is + 1, u8, scb); const d2 = d * sc, m2 = dmin * m;
        const q = qsb + (j / 64) * 32;
        for (let l = 0; l < 32; l++) out[o++] = d1 * ((u8[q + l] & 0xF) + ((u8[qhb + l] & u1) ? 16 : 0)) - m1;
        for (let l = 0; l < 32; l++) out[o++] = d2 * ((u8[q + l] >> 4)  + ((u8[qhb + l] & u2) ? 16 : 0)) - m2;
        is += 2; u1 <<= 2; u2 <<= 2;
      }
    }
    return out;
  }

  // Dequantize nb Q8_0 blocks (34 B: f16 d + int8 qs[32]). y = d*q.
  function dequantQ8_0(u8, nb) {
    const out = new Float32Array(nb * 32);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let o = 0;
    for (let i = 0; i < nb; i++) {
      const base = i * 34;
      const d = f16(dv.getUint16(base, true));
      for (let l = 0; l < 32; l++) out[o++] = d * i8(u8[base + 2 + l]);
    }
    return out;
  }

  // Dequantize nb Q6_K blocks → Float32Array(nb*256).
  function dequantQ6K(u8, nb) {
    const out = new Float32Array(nb * 256);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let o0 = 0;
    for (let i = 0; i < nb; i++) {
      const base = i * 210;
      const qlb = base, qhb = base + 128, scb = base + 192, dpos = base + 208;
      const d = f16(dv.getUint16(dpos, true));
      for (let n = 0; n < 256; n += 128) {
        const ql = qlb + (n / 128) * 64, qh = qhb + (n / 128) * 32, sc = scb + (n / 128) * 8;
        for (let l = 0; l < 32; l++) {
          const is = (l / 16) | 0;
          const q1 = ((u8[ql + l] & 0xF) | (((u8[qh + l] >> 0) & 3) << 4)) - 32;
          const q2 = ((u8[ql + l + 32] & 0xF) | (((u8[qh + l] >> 2) & 3) << 4)) - 32;
          const q3 = ((u8[ql + l] >> 4) | (((u8[qh + l] >> 4) & 3) << 4)) - 32;
          const q4 = ((u8[ql + l + 32] >> 4) | (((u8[qh + l] >> 6) & 3) << 4)) - 32;
          const o = o0 + n;
          out[o + l]      = d * i8(u8[sc + is + 0]) * q1;
          out[o + l + 32] = d * i8(u8[sc + is + 2]) * q2;
          out[o + l + 64] = d * i8(u8[sc + is + 4]) * q3;
          out[o + l + 96] = d * i8(u8[sc + is + 6]) * q4;
        }
      }
      o0 += 256;
    }
    return out;
  }

  // ============================================================
  // GPU repack — copy raw block bytes into u32-aligned, kernel-friendly buffers.
  //   Q4_K: 144 B is already 4-aligned → just a 4-aligned copy (36 u32/block).
  //   Q6_K: 210 B → 212-B (53 u32) slots so every field is 4-aligned.
  // Returns Uint32Array ready for queue.writeBuffer.
  // ============================================================
  const Q4K_U32 = 36;    // 144/4
  const Q5K_U32 = 44;    // 176/4 (already aligned)
  const Q6K_U32 = 53;    // 212/4 (210 padded to 212)
  const Q8_U32  = 9;     // 36/4 (34 padded to 36)
  // contiguous, already-aligned copy
  function repackAligned(u8, totalBytes) { const out = new Uint8Array(totalBytes); out.set(u8.subarray(0, totalBytes)); return new Uint32Array(out.buffer); }
  function repackQ4K(u8, nb) { return repackAligned(u8, nb * 144); }
  function repackQ5K(u8, nb) { return repackAligned(u8, nb * 176); }
  function repackQ6K(u8, nb) { const out = new Uint8Array(nb * 212); for (let i = 0; i < nb; i++) out.set(u8.subarray(i * 210, i * 210 + 210), i * 212); return new Uint32Array(out.buffer); }
  function repackQ8_0(u8, nb) { const out = new Uint8Array(nb * 36); for (let i = 0; i < nb; i++) out.set(u8.subarray(i * 34, i * 34 + 34), i * 36); return new Uint32Array(out.buffer); }
  // per-type: u32s-per-block + elements-per-block + repack fn (the GPU-side metadata)
  const QGPU = {
    q4_K: { u32: Q4K_U32, els: 256, repack: repackQ4K, dq: 'dq4k' },
    q5_K: { u32: Q5K_U32, els: 256, repack: repackQ5K, dq: 'dq5k' },
    q6_K: { u32: Q6K_U32, els: 256, repack: repackQ6K, dq: 'dq6k' },
    q8_0: { u32: Q8_U32,  els: 32,  repack: repackQ8_0, dq: 'dq8'  },
  };

  // ============================================================
  // WGSL decode snippets. dq4k(base,kk)/dq6k(base,kk) return the dequantized weight
  // at within-block index kk (0..255) of the block whose u32 base is `base`.
  // `W` is the repacked array<u32>. These are #included into the matmul kernels.
  // ============================================================
  // byte i (0..) of the u32 array W relative to word `wbase`:  (W[wbase + i/4] >> (8*(i%4))) & 0xFF
  // Shared by Q4_K and Q5_K (both: d,dmin @u32[0]; scales[12] @u32[1..3], same packing).
  const WGSL_KQ_COMMON = `
fn sc_byte4(base:u32, i:u32) -> u32 { let w = W[base + 1u + (i >> 2u)]; return (w >> ((i & 3u)*8u)) & 0xFFu; }
// get_scale_min_k4 for sub-block j (0..7): returns vec2(scale6, min6)
fn scale_min_k4(base:u32, j:u32) -> vec2<f32> {
  if (j < 4u) { return vec2<f32>(f32(sc_byte4(base, j) & 63u), f32(sc_byte4(base, j + 4u) & 63u)); }
  let a = sc_byte4(base, j + 4u); let b = sc_byte4(base, j - 4u); let c = sc_byte4(base, j);
  return vec2<f32>(f32((a & 0xFu) | ((b >> 6u) << 4u)), f32((a >> 4u) | ((c >> 6u) << 4u)));
}`;
  const WGSL_DQ4K = `
fn dq4k(base:u32, kk:u32) -> f32 {
  let dd = unpack2x16float(W[base]);                 // (d, dmin)
  let cc = kk >> 6u;                                  // 0..3 (which 64-chunk)
  let r  = kk & 63u;                                  // 0..63 within chunk
  let sub = cc*2u + select(0u, 1u, r >= 32u);
  let sm = scale_min_k4(base, sub);
  let qsIdx = cc*32u + (r & 31u);                     // byte index into qs[128]
  let word  = W[base + 4u + (qsIdx >> 2u)];           // qs starts at u32 offset 4 (byte 16)
  let qbyte = (word >> ((qsIdx & 3u)*8u)) & 0xFFu;
  let nib   = select(qbyte >> 4u, qbyte & 0xFu, r < 32u);
  return dd.x * sm.x * f32(nib) - dd.y * sm.y;
}`;
  const WGSL_DQ6K = `
fn dq6k(base:u32, kk:u32) -> f32 {
  // block: ql[128] @u32 0..31, qh[64] @u32 32..47, scales[16] int8 @u32 48..51, d @u32 52 low
  let d = unpack2x16float(W[base + 52u]).x;
  let half = kk >> 7u;                                // 0 or 1 (which 128-half)
  let n = kk & 127u;                                  // 0..127 within half
  let l = n & 31u;                                    // 0..31
  let grp = n >> 5u;                                  // 0..3 → selects q1..q4 and scale offset (0,2,4,6)
  let qlBase = half*16u;                              // ql u32 base for this half (64 B = 16 u32)
  let qhBase = 32u + half*8u;                         // qh u32 base (32 B = 8 u32)
  let scBase = 48u;                                   // scales u32 base
  let scHalf = half*8u;                               // scale index offset for this half
  // ql byte: grp 0/2 use ql[l], grp 1/3 use ql[l+32]; low nibble for grp<2, high for grp>=2
  let qlOff = l + select(0u, 32u, (grp & 1u) == 1u);
  let qlWord = W[base + qlBase + (qlOff >> 2u)];
  let qlByte = (qlWord >> ((qlOff & 3u)*8u)) & 0xFFu;
  let qlNib = select(qlByte >> 4u, qlByte & 0xFu, grp < 2u);
  // qh byte at index l; 2-bit field shift = grp*2
  let qhWord = W[base + qhBase + (l >> 2u)];
  let qhByte = (qhWord >> ((l & 3u)*8u)) & 0xFFu;
  let qhBits = (qhByte >> (grp*2u)) & 3u;
  let q = f32(i32(qlNib | (qhBits << 4u)) - 32);
  // scale: int8 at index scHalf + grp*2 + (l/16)
  let sIdx = scHalf + grp*2u + (l >> 4u);
  let sWord = W[base + scBase + (sIdx >> 2u)];
  let sRaw = (sWord >> ((sIdx & 3u)*8u)) & 0xFFu;
  let s = f32(select(i32(sRaw), i32(sRaw) - 256, sRaw >= 128u));
  return d * s * q;
}`;
  // Q5_K (176 B): like Q4_K + a per-weight high bit from qh[32]. Uses WGSL_KQ_COMMON.
  const WGSL_DQ5K = `
fn dq5k(base:u32, kk:u32) -> f32 {
  let dd = unpack2x16float(W[base]);                 // (d, dmin)
  let cc = kk >> 6u;                                  // 0..3
  let r  = kk & 63u;
  let sub = cc*2u + select(0u, 1u, r >= 32u);
  let sm = scale_min_k4(base, sub);
  let l = r & 31u;
  let qsIdx = cc*32u + l;                              // qs[128] @u32 offset 12 (byte 48)
  let word  = W[base + 12u + (qsIdx >> 2u)];
  let qbyte = (word >> ((qsIdx & 3u)*8u)) & 0xFFu;
  let nib   = select(qbyte >> 4u, qbyte & 0xFu, r < 32u);
  let qhWord = W[base + 4u + (l >> 2u)];              // qh[32] @u32 offset 4 (byte 16)
  let qhByte = (qhWord >> ((l & 3u)*8u)) & 0xFFu;
  let hi = (qhByte >> sub) & 1u;                       // bit index == sub (u1/u2 shift)
  return dd.x * sm.x * f32(nib + hi*16u) - dd.y * sm.y;
}`;
  // Q8_0 (36-B padded slot): f16 d @u32[0].lo, int8 qs[32] @byte 2..33. y = d*q.
  const WGSL_DQ8 = `
fn dq8(base:u32, kk:u32) -> f32 {
  let d = unpack2x16float(W[base]).x;
  let bi = 2u + kk;                                    // byte offset of qs[kk]
  let word = W[base + (bi >> 2u)];
  let raw = (word >> ((bi & 3u)*8u)) & 0xFFu;
  let q = f32(select(i32(raw), i32(raw) - 256, raw >= 128u));
  return d * q;
}`;
  // Map a type name → its decoder WGSL (plus the common helper where needed).
  const DECODER_WGSL = { q4_K: WGSL_KQ_COMMON + WGSL_DQ4K, q5_K: WGSL_KQ_COMMON + WGSL_DQ5K, q6_K: WGSL_DQ6K, q8_0: WGSL_DQ8 };

  // A plain "dequant whole tensor → f32" kernel, used ONLY by the self-test to verify the
  // WGSL decode matches the JS reference. y[i] = dq*(W, (i/els)*BLKU32, i%els).
  function dequantTestWGSL(kind) {
    const g = QGPU[kind];
    return `
struct D { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       W : array<u32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             d : D;
${DECODER_WGSL[kind]}
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let i = gid.x; if (i >= d.n) { return; }
  Y[i] = ${g.dq}(${g.u32}u * (i / ${g.els}u), i % ${g.els}u);
}`;
  }

  // ============================================================
  // Self-test: synthetic random blocks → JS dequant + GPU dequant → max abs error.
  // Requires window.SandpieWebGPU (the engine core) initialised.
  // ============================================================
  // ============================================================
  // Matmul kernels over k-quant weights (decode-in-loop). Templated per qtype so the
  // decoder is inlined (the per-call type is fixed → no dynamic branch in the hot loop).
  // Weight tensor [N,K] is row-major blocks: row n, block b at u32 base (n*(K/els)+b)*u32.
  // These are CORRECTNESS-FIRST (one row/workgroup for gemv; one output/thread for gemm) —
  // the fork engine will graft in the tiled/register-blocked versions for speed.
  // ============================================================
  const GEMVK_WG = 128;
  function gemvKWGSL(qtype) {
    const g = QGPU[qtype];
    return `
struct D { N:u32, K:u32, acc:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       W : array<u32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
${DECODER_WGSL[qtype]}
var<workgroup> part : array<f32, ${GEMVK_WG}>;
@compute @workgroup_size(${GEMVK_WG},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>) {
  let row = wg.x + wg.y*nwg.x;
  if (row >= d.N) { return; }
  let nbpr = d.K / ${g.els}u;
  var acc = 0.0; var k = lid.x;
  loop {
    if (k >= d.K) { break; }
    let base = (row*nbpr + k/${g.els}u) * ${g.u32}u;
    acc = acc + x[k] * ${g.dq}(base, k % ${g.els}u);
    k = k + ${GEMVK_WG}u;
  }
  part[lid.x] = acc; workgroupBarrier();
  var stride = ${GEMVK_WG}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){ part[lid.x]=part[lid.x]+part[lid.x+stride]; } workgroupBarrier(); stride=stride/2u; }
  if (lid.x==0u) { y[row] = select(0.0, y[row], d.acc!=0u) + part[0]; }
}`;
  }
  function gemmKWGSL(qtype) {
    const g = QGPU[qtype];
    return `
struct D { T:u32, N:u32, K:u32, acc:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       W : array<u32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
${DECODER_WGSL[qtype]}
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let idx = gid.x; if (idx >= d.T*d.N) { return; }
  let t = idx / d.N; let n = idx % d.N;
  let nbpr = d.K / ${g.els}u; let xb = t*d.K;
  var acc = 0.0;
  for (var k:u32=0u; k<d.K; k=k+1u) {
    let base = (n*nbpr + k/${g.els}u) * ${g.u32}u;
    acc = acc + x[xb + k] * ${g.dq}(base, k % ${g.els}u);
  }
  y[idx] = select(0.0, y[idx], d.acc!=0u) + acc;
}`;
  }

  // Matmul self-test: random k-quant [N,K] + random x → GPU gemvK / gemmK vs CPU (dequant
  // reference then plain matmul). Confirms the row-major block indexing is correct.
  const BYTES_PER_BLK = { q4_K: 144, q5_K: 176, q6_K: 210, q8_0: 34 };
  const DEQUANT_REF = { q4_K: dequantQ4K, q5_K: dequantQ5K, q6_K: dequantQ6K, q8_0: dequantQ8_0 };
  async function selfTest(E, nb) {
    nb = nb || 8;
    const U = GPUBufferUsage;
    const out = {};
    for (const kind of ['q4_K', 'q5_K', 'q6_K', 'q8_0']) {
      const g = QGPU[kind], els = g.els, nEl = nb * els;
      const raw = new Uint8Array(nb * BYTES_PER_BLK[kind]);
      // deterministic pseudo-random fill (no Math.random — reproducible)
      let s = 0x9e3779b9 >>> 0;
      for (let i = 0; i < raw.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; raw[i] = (s >>> 16) & 0xFF; }
      const ref = DEQUANT_REF[kind](raw, nb);
      const packed = g.repack(raw, nb);
      const wBuf = E.createBuffer(packed.byteLength, U.STORAGE | U.COPY_DST, 'dqtest.W');
      E.device().queue.writeBuffer(wBuf, 0, packed);
      const yBuf = E.createBuffer(nEl * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'dqtest.Y');
      const dBuf = E.createBuffer(16, U.UNIFORM | U.COPY_DST, 'dqtest.d');
      E.device().queue.writeBuffer(dBuf, 0, new Uint32Array([nEl, 0, 0, 0]));
      const pipe = E.getPipeline('gguf.dqtest.' + kind, dequantTestWGSL(kind));
      E.dispatch(pipe, [wBuf, yBuf, dBuf], [Math.ceil(nEl / 64), 1, 1]);
      await E.device().queue.onSubmittedWorkDone();
      const stg = E.createBuffer(nEl * 4, U.COPY_DST | U.MAP_READ, 'dqtest.rd');
      const enc = E.device().createCommandEncoder();
      enc.copyBufferToBuffer(yBuf, 0, stg, 0, nEl * 4);
      E.device().queue.submit([enc.finish()]);
      await stg.mapAsync(GPUMapMode.READ);
      const gpu = new Float32Array(stg.getMappedRange().slice(0)); stg.unmap();
      let maxErr = 0, argmax = -1;
      for (let i = 0; i < ref.length; i++) { const e = Math.abs(ref[i] - gpu[i]); if (e > maxErr) { maxErr = e; argmax = i; } }
      out[kind] = { maxErr, argmax, refSample: Array.from(ref.slice(0, 4)), gpuSample: Array.from(gpu.slice(0, 4)) };
      [wBuf, yBuf, dBuf, stg].forEach(b => b.destroy());
    }
    return out;
  }

  // GPU gemv/gemm vs CPU reference. N rows, K cols (multiple of els), T query rows.
  async function selfTestMatmul(E, opts) {
    const U = GPUBufferUsage;
    const N = (opts && opts.N) || 40, K = (opts && opts.K) || 512, T = (opts && opts.T) || 3;
    const out = {};
    for (const qtype of (opts && opts.types) || ['q4_K', 'q5_K', 'q6_K', 'q8_0']) {
      const g = QGPU[qtype], els = g.els, nbpr = K / els, nb = N * nbpr;
      const raw = new Uint8Array(nb * BYTES_PER_BLK[qtype]);
      let s = 0x12345678 >>> 0;
      for (let i = 0; i < raw.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; raw[i] = (s >>> 16) & 0xFF; }
      const Wf = DEQUANT_REF[qtype](raw, nb);           // [N,K] row-major
      const x = new Float32Array(T * K);
      for (let i = 0; i < x.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; x[i] = ((s >>> 8 & 0xFFFF) / 65535 - 0.5); }
      // CPU reference
      const cpu = new Float32Array(T * N);
      for (let t = 0; t < T; t++) for (let n = 0; n < N; n++) { let a = 0; for (let k = 0; k < K; k++) a += Wf[n * K + k] * x[t * K + k]; cpu[t * N + n] = a; }
      const packed = g.repack(raw, nb);
      const wBuf = E.createBuffer(packed.byteLength, U.STORAGE | U.COPY_DST, 'mm.W'); E.device().queue.writeBuffer(wBuf, 0, packed);
      const xBuf = E.createBuffer(x.byteLength, U.STORAGE | U.COPY_DST, 'mm.x'); E.device().queue.writeBuffer(xBuf, 0, x);
      const run = async (useGemm) => {
        const yBuf = E.createBuffer(T * N * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'mm.y');
        const dBuf = E.createBuffer(16, U.UNIFORM | U.COPY_DST, 'mm.d');
        if (useGemm) {
          E.device().queue.writeBuffer(dBuf, 0, new Uint32Array([T, N, K, 0]));
          E.dispatch(E.getPipeline('gguf.gemmK.' + qtype, gemmKWGSL(qtype)), [xBuf, wBuf, yBuf, dBuf], [Math.ceil(T * N / 64), 1, 1]);
        } else {
          E.device().queue.writeBuffer(dBuf, 0, new Uint32Array([N, K, 0, 0]));
          const nWG = N, gx = Math.min(nWG, 65535), gy = Math.ceil(nWG / gx);
          E.dispatch(E.getPipeline('gguf.gemvK.' + qtype, gemvKWGSL(qtype)), [xBuf, wBuf, yBuf, dBuf], [gx, gy, 1]);
        }
        await E.device().queue.onSubmittedWorkDone();
        const stg = E.createBuffer(T * N * 4, U.COPY_DST | U.MAP_READ, 'mm.rd');
        const enc = E.device().createCommandEncoder(); enc.copyBufferToBuffer(yBuf, 0, stg, 0, T * N * 4); E.device().queue.submit([enc.finish()]);
        await stg.mapAsync(GPUMapMode.READ); const gpu = new Float32Array(stg.getMappedRange().slice(0)); stg.unmap();
        [yBuf, dBuf, stg].forEach(b => b.destroy());
        return gpu;
      };
      const gemvGpu = await run(false);   // compares against cpu row 0 (T=1 path uses x[0..K])
      const gemmGpu = await run(true);
      const relerr = (ref, got, rows) => { let m = 0; for (let i = 0; i < rows * N; i++) { const e = Math.abs(ref[i] - got[i]) / (Math.abs(ref[i]) + 1e-6); if (e > m) m = e; } return m; };
      out[qtype] = { gemv_rel: relerr(cpu, gemvGpu, 1), gemm_rel: relerr(cpu, gemmGpu, T) };
      [wBuf, xBuf].forEach(b => b.destroy());
    }
    return out;
  }

  return {
    GGML, TYPE_NAME, BLK, QGPU, parse, tensorByteLen,
    gemvKWGSL, gemmKWGSL, selfTestMatmul,
    f16, getScaleMinK4, dequantQ4K, dequantQ5K, dequantQ6K, dequantQ8_0,
    repackQ4K, repackQ5K, repackQ6K, repackQ8_0, Q4K_U32, Q5K_U32, Q6K_U32, Q8_U32,
    WGSL_KQ_COMMON, WGSL_DQ4K, WGSL_DQ5K, WGSL_DQ6K, WGSL_DQ8, DECODER_WGSL,
    dequantTestWGSL, selfTest,
  };
})();
if (typeof window !== 'undefined') window.SandpieGGUF = SandpieGGUF;
if (typeof module !== 'undefined' && module.exports) module.exports = SandpieGGUF;   // node (tooling/tests)
