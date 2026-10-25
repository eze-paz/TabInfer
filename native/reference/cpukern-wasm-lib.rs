//! Hand-rolled WASM ternary decode GEMV — the universal CPU floor beneath the
//! WebGPU engine. Validates the core hypothesis: relaxed-SIMD int8 dot on
//! 2-bit-packed ternary weights can become bandwidth-bound (unlike mainline
//! llama.cpp's Q2_0 CPU kernel, measured at ~3 GB/s / 6.5 tok/s).
//!
//! Format (matches the GPU engine's intent):
//!   codes:  row-major, K 2-bit codes/row, 4 codes/byte little-endian.
//!           code = weight+1 in {0,1,2} (weight in {-1,0,+1}).
//!   scales: f32, one per group of G=32 along K, row-major [N, K/32].
//!   act:    int8 activations [K] (per-tensor or per-group quantized upstream).
//!   xsum:   i32, Σ act over each group of 32 [K/32] — the (c-1)·x = c·x − Σx term.
//!
//! Decode identity: Σ (code-1)·act  over a group  =  Σ code·act − Σ act
//! The relaxed dot's i7 operand slot fits codes {0,1,2} trivially; activations
//! (int8, ±127) take the i8 slot.
#![no_std]

use core::arch::wasm32::*;

// ───────────── swizzle-LUT binary gemv PROTOTYPE v2 (1-swizzle + int16 accumulation) ─────────────
// The FAST variant: int8 table (16 entries per 4-activation group), ONE i8x16_swizzle per group
// × 16 output rows, accumulate partials in int16 with periodic flush to int32. Table entries fit
// int8 by rounding-requantizing activations to int6 (4×31 = 124 <= 127). ~7 instr / 64 MACs vs
// relaxed-dot's ~18/64 → ~2.6× fewer instructions on paper. Binary weights {-1,+1}.

// ── scale-aware LUT gemv (matches gemv_tern's per-G=64 f32 scale contract) ──
// widx layout: [rowBlock][group][16 rows], byte = code index. scalesB layout: [scaleGroup][rowBlock][16].
// Accumulate int16 within a G=64 scale-group (safe: 32×126 or 16×124 << 32767), then convert→f32,
// ×per-row scale, into f32 accumulators. Output f32[nrows] (nrows padded to %16). Caller ×act-scale.

/// TERNARY g=2 scale-aware. tbl from build_lut_tern (int7 acts). 32 lut-groups per scale-group.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn gemv_lut_tern_s(out: *mut f32, widx: *const u8, scalesB: *const f32, tbl: *const i8, nrows: u32, k: u32) {
    let ng2 = (k / 2) as usize; let nb = (nrows / 16) as usize; let nsg = (k / 64) as usize;
    for b in 0..nb {
        let mut f0 = f32x4_splat(0.0); let mut f1 = f32x4_splat(0.0); let mut f2 = f32x4_splat(0.0); let mut f3 = f32x4_splat(0.0);
        for sg in 0..nsg {
            let mut lo = i16x8_splat(0); let mut hi = i16x8_splat(0);
            let g0 = sg * 32;
            let mut j = 0usize;
            while j < 32 {
                let g = g0 + j;
                let p = i8x16_swizzle(v128_load(tbl.add(g * 16) as *const v128), v128_load(widx.add((b * ng2 + g) * 16) as *const v128));
                lo = i16x8_add(lo, i16x8_extend_low_i8x16(p));
                hi = i16x8_add(hi, i16x8_extend_high_i8x16(p));
                j += 1;
            }
            let sc = scalesB.add((sg * nb + b) * 16);
            f0 = f32x4_add(f0, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo)), v128_load(sc as *const v128)));
            f1 = f32x4_add(f1, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo)), v128_load(sc.add(4) as *const v128)));
            f2 = f32x4_add(f2, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi)), v128_load(sc.add(8) as *const v128)));
            f3 = f32x4_add(f3, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi)), v128_load(sc.add(12) as *const v128)));
        }
        v128_store(out.add(b * 16) as *mut v128, f0); v128_store(out.add(b * 16 + 4) as *mut v128, f1);
        v128_store(out.add(b * 16 + 8) as *mut v128, f2); v128_store(out.add(b * 16 + 12) as *mut v128, f3);
    }
}

/// BINARY g=4 scale-aware. tbl from build_lut_bin (int6 acts). 16 lut-groups per scale-group.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn gemv_lut_bin_s(out: *mut f32, widx: *const u8, scalesB: *const f32, tbl: *const i8, nrows: u32, k: u32) {
    let ng4 = (k / 4) as usize; let nb = (nrows / 16) as usize; let nsg = (k / 64) as usize;
    for b in 0..nb {
        let mut f0 = f32x4_splat(0.0); let mut f1 = f32x4_splat(0.0); let mut f2 = f32x4_splat(0.0); let mut f3 = f32x4_splat(0.0);
        for sg in 0..nsg {
            let mut lo = i16x8_splat(0); let mut hi = i16x8_splat(0);
            let g0 = sg * 16;
            let mut j = 0usize;
            while j < 16 {
                let g = g0 + j;
                let p = i8x16_swizzle(v128_load(tbl.add(g * 16) as *const v128), v128_load(widx.add((b * ng4 + g) * 16) as *const v128));
                lo = i16x8_add(lo, i16x8_extend_low_i8x16(p));
                hi = i16x8_add(hi, i16x8_extend_high_i8x16(p));
                j += 1;
            }
            let sc = scalesB.add((sg * nb + b) * 16);
            f0 = f32x4_add(f0, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo)), v128_load(sc as *const v128)));
            f1 = f32x4_add(f1, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo)), v128_load(sc.add(4) as *const v128)));
            f2 = f32x4_add(f2, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi)), v128_load(sc.add(8) as *const v128)));
            f3 = f32x4_add(f3, f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi)), v128_load(sc.add(12) as *const v128)));
        }
        v128_store(out.add(b * 16) as *mut v128, f0); v128_store(out.add(b * 16 + 4) as *mut v128, f1);
        v128_store(out.add(b * 16 + 8) as *mut v128, f2); v128_store(out.add(b * 16 + 12) as *mut v128, f3);
    }
}

/// Build 16-entry int8 LUTs from int8 act, rounding-requantized to int6 (±31). tbl: (k/4)*16 bytes.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn build_lut_bin(tbl: *mut i8, act: *const i8, k: u32) {
    let ng = (k / 4) as usize;
    // round-requantize i8 (±127) → i6 (±31): round(a*31/127)
    let rq = |a: i8| -> i32 { let v = (a as i32) * 31; (if v >= 0 { v + 63 } else { v - 63 }) / 127 };
    for g in 0..ng {
        let a0 = rq(*act.add(4 * g)); let a1 = rq(*act.add(4 * g + 1));
        let a2 = rq(*act.add(4 * g + 2)); let a3 = rq(*act.add(4 * g + 3));
        let base = tbl.add(g * 16);
        let mut idx = 0i32;
        while idx < 16 {
            let s = (if idx & 1 != 0 { a0 } else { -a0 }) + (if idx & 2 != 0 { a1 } else { -a1 })
                + (if idx & 4 != 0 { a2 } else { -a2 }) + (if idx & 8 != 0 { a3 } else { -a3 });
            *base.add(idx as usize) = s as i8; // |s| <= 124
            idx += 1;
        }
    }
}

/// TERNARY g=2 LUT: index = 2 ternary codes {0,1,2} packed (c1<<2)|c0. int7-requant acts so a
/// 2-group sum fits int8. Keeps the ternary model (no binary downgrade). tbl: (k/2)*16 bytes.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn build_lut_tern(tbl: *mut i8, act: *const i8, k: u32) {
    let ng = (k / 2) as usize;
    let rq = |a: i8| -> i32 { let v = (a as i32) * 63; (if v >= 0 { v + 63 } else { v - 63 }) / 127 }; // →int7 ±63
    for g in 0..ng {
        let a0 = rq(*act.add(2 * g)); let a1 = rq(*act.add(2 * g + 1));
        let base = tbl.add(g * 16);
        let mut idx = 0i32;
        while idx < 16 {
            let c0 = idx & 3; let c1 = (idx >> 2) & 3;
            let s = if c0 == 3 || c1 == 3 { 0 } else { (c0 - 1) * a0 + (c1 - 1) * a1 };
            *base.add(idx as usize) = s as i8; // |s| <= 126
            idx += 1;
        }
    }
}

/// out[N] i32 = Σ (code-1)·a_i7 via 1-swizzle + int16 accum, ternary g=2. n%16==0.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn gemv_lut_tern(out: *mut i32, widx: *const u8, tbl: *const i8, n: u32, k: u32) {
    let ng = (k / 2) as usize;
    let nb = (n / 16) as usize;
    for b in 0..nb {
        let mut c0v = i32x4_splat(0); let mut c1v = i32x4_splat(0); let mut c2v = i32x4_splat(0); let mut c3v = i32x4_splat(0);
        let mut lo = i16x8_splat(0); let mut hi = i16x8_splat(0);
        let wb = widx.add(b * ng * 16);
        let mut g = 0usize; let mut cnt = 0u32;
        while g < ng {
            let p = i8x16_swizzle(v128_load(tbl.add(g * 16) as *const v128), v128_load(wb.add(g * 16) as *const v128));
            lo = i16x8_add(lo, i16x8_extend_low_i8x16(p));
            hi = i16x8_add(hi, i16x8_extend_high_i8x16(p));
            cnt += 1;
            if cnt == 256 {
                c0v = i32x4_add(c0v, i32x4_extend_low_i16x8(lo)); c1v = i32x4_add(c1v, i32x4_extend_high_i16x8(lo));
                c2v = i32x4_add(c2v, i32x4_extend_low_i16x8(hi)); c3v = i32x4_add(c3v, i32x4_extend_high_i16x8(hi));
                lo = i16x8_splat(0); hi = i16x8_splat(0); cnt = 0;
            }
            g += 1;
        }
        c0v = i32x4_add(c0v, i32x4_extend_low_i16x8(lo)); c1v = i32x4_add(c1v, i32x4_extend_high_i16x8(lo));
        c2v = i32x4_add(c2v, i32x4_extend_low_i16x8(hi)); c3v = i32x4_add(c3v, i32x4_extend_high_i16x8(hi));
        v128_store(out.add(b * 16) as *mut v128, c0v); v128_store(out.add(b * 16 + 4) as *mut v128, c1v);
        v128_store(out.add(b * 16 + 8) as *mut v128, c2v); v128_store(out.add(b * 16 + 12) as *mut v128, c3v);
    }
}

/// out[N] i32 = Σ w·a_i6 via 1-swizzle + int16 accumulation (flush every 256 groups). n%16==0.
/// Result is in int6-activation units; caller scales by (127/31)*act_scale.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn gemv_lut_bin_fast(out: *mut i32, widx: *const u8, tbl: *const i8, n: u32, k: u32) {
    let ng = (k / 4) as usize;
    let nb = (n / 16) as usize;
    for b in 0..nb {
        let mut c0 = i32x4_splat(0); let mut c1 = i32x4_splat(0); let mut c2 = i32x4_splat(0); let mut c3 = i32x4_splat(0);
        let mut lo = i16x8_splat(0); let mut hi = i16x8_splat(0); // int16 accumulators (16 rows)
        let wb = widx.add(b * ng * 16);
        let mut g = 0usize; let mut cnt = 0u32;
        while g < ng {
            let p = i8x16_swizzle(v128_load(tbl.add(g * 16) as *const v128), v128_load(wb.add(g * 16) as *const v128));
            lo = i16x8_add(lo, i16x8_extend_low_i8x16(p));
            hi = i16x8_add(hi, i16x8_extend_high_i8x16(p));
            cnt += 1;
            if cnt == 256 { // flush int16 → int32 before overflow
                c0 = i32x4_add(c0, i32x4_extend_low_i16x8(lo)); c1 = i32x4_add(c1, i32x4_extend_high_i16x8(lo));
                c2 = i32x4_add(c2, i32x4_extend_low_i16x8(hi)); c3 = i32x4_add(c3, i32x4_extend_high_i16x8(hi));
                lo = i16x8_splat(0); hi = i16x8_splat(0); cnt = 0;
            }
            g += 1;
        }
        c0 = i32x4_add(c0, i32x4_extend_low_i16x8(lo)); c1 = i32x4_add(c1, i32x4_extend_high_i16x8(lo));
        c2 = i32x4_add(c2, i32x4_extend_low_i16x8(hi)); c3 = i32x4_add(c3, i32x4_extend_high_i16x8(hi));
        v128_store(out.add(b * 16) as *mut v128, c0); v128_store(out.add(b * 16 + 4) as *mut v128, c1);
        v128_store(out.add(b * 16 + 8) as *mut v128, c2); v128_store(out.add(b * 16 + 12) as *mut v128, c3);
    }
}

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

const G: usize = 64; // scale group size along K (matches Bonsai native Q2_0_g64)

/// out[N] = Σ_k (weight[n,k]) · act[k] · scale[n, k/32], weight = code − 1.
///
/// INTERLEAVED PACK (per 64-code block = 16 bytes): code c (0..63) lives in
/// byte (c & 15) at bit-position 2·(c >> 4). So one 16-byte load + shift/mask
/// yields four v128s of 16 int8 codes each — plane j (shift 2j) holds codes
/// j·16 .. j·16+15 — with zero scalar work and no shuffle. One scale group per
/// 64-block (G=64, Bonsai native): all four planes accumulate into one i32x4.
// ── ROW-BLOCKED variants (NR rows share one activation load) ──────────────
// The activation block does NOT depend on the row, but the 1-row kernel reloads
// all four act v128s for EVERY row (4 of ~19 instrs per 64 MACs). Processing NR
// rows per block amortizes those loads NR-fold. Per-row accumulation ORDER is
// unchanged, so output is bit-identical to gemv_tern.
#[inline(always)]
unsafe fn hsum4(f: v128) -> f32 {
    f32x4_extract_lane::<0>(f) + f32x4_extract_lane::<1>(f)
        + f32x4_extract_lane::<2>(f) + f32x4_extract_lane::<3>(f)
}

/// One (row, 64-code block): unpack the row's 4 planes, dot against the CALLER'S
/// already-loaded activation vectors, accumulate scale·dot and the scale·Σx side term.
#[inline(always)]
unsafe fn tern_blk(
    rc: *const u8, rs: *const f32, b: usize,
    a0: v128, a1: v128, a2: v128, a3: v128, xs: f32,
    m3: v128, zero: v128, facc: &mut v128, sxs: &mut f32,
) {
    let raw = v128_load(rc.add(b * 16) as *const v128);
    let p0 = v128_and(raw, m3);
    let p1 = v128_and(u8x16_shr(raw, 2), m3);
    let p2 = v128_and(u8x16_shr(raw, 4), m3);
    let p3 = u8x16_shr(raw, 6);
    let dot = i32x4_relaxed_dot_i8x16_i7x16_add(a3, p3,
        i32x4_relaxed_dot_i8x16_i7x16_add(a2, p2,
            i32x4_relaxed_dot_i8x16_i7x16_add(a1, p1,
                i32x4_relaxed_dot_i8x16_i7x16_add(a0, p0, zero))));
    let s = *rs.add(b);
    *facc = f32x4_add(*facc, f32x4_mul(f32x4_convert_i32x4(dot), f32x4_splat(s)));
    *sxs += s * xs;
}

/// Scalar-tail: the original one-row-at-a-time path, for rows past the NR block.
#[inline(always)]
unsafe fn tern_rows_1(
    out: *mut f32, codes: *const u8, scales: *const f32, act: *const i8, xsum: *const i32,
    row0: usize, n: usize, nblocks: usize, bytes_per_row: usize, m3: v128, zero: v128,
) {
    let mut row = row0;
    while row < n {
        let rc = codes.add(row * bytes_per_row);
        let rs = scales.add(row * nblocks);
        let mut facc = f32x4_splat(0.0);
        let mut sxs: f32 = 0.0;
        let mut b = 0usize;
        while b < nblocks {
            let ab = act.add(b * 64);
            tern_blk(rc, rs, b,
                v128_load(ab as *const v128), v128_load(ab.add(16) as *const v128),
                v128_load(ab.add(32) as *const v128), v128_load(ab.add(48) as *const v128),
                *xsum.add(b) as f32, m3, zero, &mut facc, &mut sxs);
            b += 1;
        }
        *out.add(row) = hsum4(facc) - sxs;
        row += 1;
    }
}

/// NR=2 row-blocked gemv. Bit-identical to gemv_tern.
#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemv_tern_r2(
    out: *mut f32, codes: *const u8, scales: *const f32,
    act: *const i8, xsum: *const i32, n: u32, k: u32,
) {
    let n = n as usize; let k = k as usize;
    let nblocks = k / 64; let bytes_per_row = k / 4;
    let m3 = u8x16_splat(3); let zero = i32x4_splat(0);
    let mut row = 0usize;
    while row + 2 <= n {
        let rc0 = codes.add(row * bytes_per_row); let rc1 = rc0.add(bytes_per_row);
        let rs0 = scales.add(row * nblocks); let rs1 = rs0.add(nblocks);
        let (mut f0, mut f1) = (f32x4_splat(0.0), f32x4_splat(0.0));
        let (mut s0, mut s1) = (0.0f32, 0.0f32);
        let mut b = 0usize;
        while b < nblocks {
            let ab = act.add(b * 64);
            let a0 = v128_load(ab as *const v128);
            let a1 = v128_load(ab.add(16) as *const v128);
            let a2 = v128_load(ab.add(32) as *const v128);
            let a3 = v128_load(ab.add(48) as *const v128);
            let xs = *xsum.add(b) as f32;
            tern_blk(rc0, rs0, b, a0, a1, a2, a3, xs, m3, zero, &mut f0, &mut s0);
            tern_blk(rc1, rs1, b, a0, a1, a2, a3, xs, m3, zero, &mut f1, &mut s1);
            b += 1;
        }
        *out.add(row) = hsum4(f0) - s0;
        *out.add(row + 1) = hsum4(f1) - s1;
        row += 2;
    }
    tern_rows_1(out, codes, scales, act, xsum, row, n, nblocks, bytes_per_row, m3, zero);
}

// ── BATCHED (multi-column) ternary GEMM ───────────────────────────────────
// Row blocking amortized the ACTIVATION loads; the remaining per-weight cost is
// the 2-bit UNPACK (6 ops per 64 weights), which can only be amortized across
// COLUMNS — same weights, several activation vectors. That's what a speculative
// verify step provides (draft tokens verified in one pass).
//   act:  [B][K]      i8   column c at act  + c*k
//   xsum: [B][K/64]   i32  column c at xsum + c*nblocks
//   out:  [B][N]      f32  column c at out  + c*n
// Per (row,col) accumulation order matches gemv_tern exactly → bit-identical.

/// One column's contribution for an already-unpacked 64-weight block.
#[inline(always)]
unsafe fn tern_col(
    p0: v128, p1: v128, p2: v128, p3: v128, ab: *const i8,
    zero: v128, sv: v128, s: f32, xs: f32, facc: &mut v128, sxs: &mut f32,
) {
    let dot = i32x4_relaxed_dot_i8x16_i7x16_add(v128_load(ab.add(48) as *const v128), p3,
        i32x4_relaxed_dot_i8x16_i7x16_add(v128_load(ab.add(32) as *const v128), p2,
            i32x4_relaxed_dot_i8x16_i7x16_add(v128_load(ab.add(16) as *const v128), p1,
                i32x4_relaxed_dot_i8x16_i7x16_add(v128_load(ab as *const v128), p0, zero))));
    *facc = f32x4_add(*facc, f32x4_mul(f32x4_convert_i32x4(dot), sv));
    *sxs += s * xs;
}

/// B=4 columns, one row at a time. Unpack shared across the 4 columns.
/// MEASURED 1.85-1.89x vs 4 sequential gemv_tern calls (bit-identical), 18 -> 35
/// GMAC/s. Beat the ~1.2x instruction-count prediction because the 4 columns give
/// 4 INDEPENDENT dot chains: gemv's four dots are chained (each is the next one's
/// accumulator), so batching wins on instruction-level parallelism, not just on
/// amortized unpack.
#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemm_tern_b4(
    out: *mut f32, out_stride: u32, codes: *const u8, scales: *const f32,
    act: *const i8, xsum: *const i32, n: u32, k: u32,
) {
    let n = n as usize; let k = k as usize; let os = out_stride as usize;
    let nblocks = k / 64; let bpr = k / 4;
    let m3 = u8x16_splat(3); let zero = i32x4_splat(0);
    let mut row = 0usize;
    while row < n {
        let rc = codes.add(row * bpr);
        let rs = scales.add(row * nblocks);
        let (mut f0, mut f1, mut f2, mut f3) =
            (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        let (mut x0, mut x1, mut x2, mut x3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut b = 0usize;
        while b < nblocks {
            let raw = v128_load(rc.add(b * 16) as *const v128);
            let p0 = v128_and(raw, m3);
            let p1 = v128_and(u8x16_shr(raw, 2), m3);
            let p2 = v128_and(u8x16_shr(raw, 4), m3);
            let p3 = u8x16_shr(raw, 6);
            let s = *rs.add(b);
            let sv = f32x4_splat(s);
            let o = b * 64;
            tern_col(p0,p1,p2,p3, act.add(o),           zero, sv, s, *xsum.add(b) as f32,               &mut f0, &mut x0);
            tern_col(p0,p1,p2,p3, act.add(k + o),       zero, sv, s, *xsum.add(nblocks + b) as f32,     &mut f1, &mut x1);
            tern_col(p0,p1,p2,p3, act.add(2 * k + o),   zero, sv, s, *xsum.add(2 * nblocks + b) as f32, &mut f2, &mut x2);
            tern_col(p0,p1,p2,p3, act.add(3 * k + o),   zero, sv, s, *xsum.add(3 * nblocks + b) as f32, &mut f3, &mut x3);
            b += 1;
        }
        *out.add(row) = hsum4(f0) - x0;
        *out.add(os + row) = hsum4(f1) - x1;
        *out.add(2 * os + row) = hsum4(f2) - x2;
        *out.add(3 * os + row) = hsum4(f3) - x3;
        row += 1;
    }
}

/// B=4 columns × NR=2 rows: shares the unpack across columns AND the activation
/// loads across the 2 rows. 8 accumulators — near the register budget.
#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemm_tern_b4r2(
    out: *mut f32, codes: *const u8, scales: *const f32,
    act: *const i8, xsum: *const i32, n: u32, k: u32,
) {
    let n = n as usize; let k = k as usize;
    let nblocks = k / 64; let bpr = k / 4;
    let m3 = u8x16_splat(3); let zero = i32x4_splat(0);
    let mut row = 0usize;
    while row + 2 <= n {
        let rcA = codes.add(row * bpr); let rcB = rcA.add(bpr);
        let rsA = scales.add(row * nblocks); let rsB = rsA.add(nblocks);
        let (mut a0, mut a1, mut a2, mut a3) =
            (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        let (mut b0, mut b1, mut b2, mut b3) =
            (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        let (mut xa0, mut xa1, mut xa2, mut xa3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut xb0, mut xb1, mut xb2, mut xb3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut b = 0usize;
        while b < nblocks {
            let o = b * 64;
            let (c0, c1, c2, c3) = (act.add(o), act.add(k + o), act.add(2 * k + o), act.add(3 * k + o));
            let (y0, y1, y2, y3) = (*xsum.add(b) as f32, *xsum.add(nblocks + b) as f32,
                                    *xsum.add(2 * nblocks + b) as f32, *xsum.add(3 * nblocks + b) as f32);
            // row A
            let raw = v128_load(rcA.add(b * 16) as *const v128);
            let p0 = v128_and(raw, m3); let p1 = v128_and(u8x16_shr(raw, 2), m3);
            let p2 = v128_and(u8x16_shr(raw, 4), m3); let p3 = u8x16_shr(raw, 6);
            let s = *rsA.add(b); let sv = f32x4_splat(s);
            tern_col(p0,p1,p2,p3, c0, zero, sv, s, y0, &mut a0, &mut xa0);
            tern_col(p0,p1,p2,p3, c1, zero, sv, s, y1, &mut a1, &mut xa1);
            tern_col(p0,p1,p2,p3, c2, zero, sv, s, y2, &mut a2, &mut xa2);
            tern_col(p0,p1,p2,p3, c3, zero, sv, s, y3, &mut a3, &mut xa3);
            // row B (activations still hot)
            let raw2 = v128_load(rcB.add(b * 16) as *const v128);
            let q0 = v128_and(raw2, m3); let q1 = v128_and(u8x16_shr(raw2, 2), m3);
            let q2 = v128_and(u8x16_shr(raw2, 4), m3); let q3 = u8x16_shr(raw2, 6);
            let s2 = *rsB.add(b); let sv2 = f32x4_splat(s2);
            tern_col(q0,q1,q2,q3, c0, zero, sv2, s2, y0, &mut b0, &mut xb0);
            tern_col(q0,q1,q2,q3, c1, zero, sv2, s2, y1, &mut b1, &mut xb1);
            tern_col(q0,q1,q2,q3, c2, zero, sv2, s2, y2, &mut b2, &mut xb2);
            tern_col(q0,q1,q2,q3, c3, zero, sv2, s2, y3, &mut b3, &mut xb3);
            b += 1;
        }
        *out.add(row) = hsum4(a0) - xa0;
        *out.add(n + row) = hsum4(a1) - xa1;
        *out.add(2 * n + row) = hsum4(a2) - xa2;
        *out.add(3 * n + row) = hsum4(a3) - xa3;
        *out.add(row + 1) = hsum4(b0) - xb0;
        *out.add(n + row + 1) = hsum4(b1) - xb1;
        *out.add(2 * n + row + 1) = hsum4(b2) - xb2;
        *out.add(3 * n + row + 1) = hsum4(b3) - xb3;
        row += 2;
    }
    // odd tail row
    if row < n {
        let sub = |c: usize| unsafe { gemv_tern_r1(out.add(c * n + row), codes.add(row * bpr), scales.add(row * nblocks),
            act.add(c * k), xsum.add(c * nblocks), 1, k as u32) };
        sub(0); sub(1); sub(2); sub(3);
    }
}

/// SHIPPED gemv: delegates to the NR=4 row-blocked kernel (measured ~1.15x over the
/// one-row path, bit-identical output). gemv_tern_r1 keeps the original for A/B.
#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemv_tern(
    out: *mut f32, codes: *const u8, scales: *const f32,
    act: *const i8, xsum: *const i32, n: u32, k: u32,
) { gemv_tern_r4(out, codes, scales, act, xsum, n, k) }

/// NR=4 row-blocked gemv. Bit-identical to gemv_tern_r1.
#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemv_tern_r4(
    out: *mut f32, codes: *const u8, scales: *const f32,
    act: *const i8, xsum: *const i32, n: u32, k: u32,
) {
    let n = n as usize; let k = k as usize;
    let nblocks = k / 64; let bytes_per_row = k / 4;
    let m3 = u8x16_splat(3); let zero = i32x4_splat(0);
    let mut row = 0usize;
    while row + 4 <= n {
        let rc0 = codes.add(row * bytes_per_row);
        let rc1 = rc0.add(bytes_per_row);
        let rc2 = rc1.add(bytes_per_row);
        let rc3 = rc2.add(bytes_per_row);
        let rs0 = scales.add(row * nblocks);
        let rs1 = rs0.add(nblocks);
        let rs2 = rs1.add(nblocks);
        let rs3 = rs2.add(nblocks);
        let (mut f0, mut f1, mut f2, mut f3) =
            (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut b = 0usize;
        while b < nblocks {
            let ab = act.add(b * 64);
            let a0 = v128_load(ab as *const v128);
            let a1 = v128_load(ab.add(16) as *const v128);
            let a2 = v128_load(ab.add(32) as *const v128);
            let a3 = v128_load(ab.add(48) as *const v128);
            let xs = *xsum.add(b) as f32;
            tern_blk(rc0, rs0, b, a0, a1, a2, a3, xs, m3, zero, &mut f0, &mut s0);
            tern_blk(rc1, rs1, b, a0, a1, a2, a3, xs, m3, zero, &mut f1, &mut s1);
            tern_blk(rc2, rs2, b, a0, a1, a2, a3, xs, m3, zero, &mut f2, &mut s2);
            tern_blk(rc3, rs3, b, a0, a1, a2, a3, xs, m3, zero, &mut f3, &mut s3);
            b += 1;
        }
        *out.add(row) = hsum4(f0) - s0;
        *out.add(row + 1) = hsum4(f1) - s1;
        *out.add(row + 2) = hsum4(f2) - s2;
        *out.add(row + 3) = hsum4(f3) - s3;
        row += 4;
    }
    tern_rows_1(out, codes, scales, act, xsum, row, n, nblocks, bytes_per_row, m3, zero);
}

#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemv_tern_r1(
    out: *mut f32,
    codes: *const u8,
    scales: *const f32,
    act: *const i8,
    xsum: *const i32,
    n: u32,
    k: u32,
) {
    let n = n as usize;
    let k = k as usize;
    let nblocks = k / 64; // one 64-code block == one scale group (G=64)
    let bytes_per_row = k / 4;
    let m3 = u8x16_splat(3);
    let zero = i32x4_splat(0);
    let mut row = 0usize;
    while row < n {
        let rc = codes.add(row * bytes_per_row);
        let rs = scales.add(row * nblocks);
        // DEFERRED reduction: accumulate scale_g · dot_vec_g into an f32x4 (vertical),
        // hsum ONCE per row. The −Σx term factors out as a scalar side-sum:
        //   Σ_g scale_g·(hsum(dot_g) − xsum_g) = hsum(Σ_g scale_g·dot_g) − Σ_g scale_g·xsum_g
        let mut facc = f32x4_splat(0.0);
        let mut sxs: f32 = 0.0;
        let mut b = 0usize;
        while b < nblocks {
            let raw = v128_load(rc.add(b * 16) as *const v128);
            let p0 = v128_and(raw, m3);
            let p1 = v128_and(u8x16_shr(raw, 2), m3);
            let p2 = v128_and(u8x16_shr(raw, 4), m3);
            let p3 = u8x16_shr(raw, 6); // top 2 bits already isolated
            let ab = act.add(b * 64);
            let a0 = v128_load(ab as *const v128);
            let a1 = v128_load(ab.add(16) as *const v128);
            let a2 = v128_load(ab.add(32) as *const v128);
            let a3 = v128_load(ab.add(48) as *const v128);
            // all 4 planes (64 codes) form ONE scale group (G=64)
            let dot = i32x4_relaxed_dot_i8x16_i7x16_add(a3, p3,
                i32x4_relaxed_dot_i8x16_i7x16_add(a2, p2,
                    i32x4_relaxed_dot_i8x16_i7x16_add(a1, p1,
                        i32x4_relaxed_dot_i8x16_i7x16_add(a0, p0, zero))));
            let s = *rs.add(b);
            facc = f32x4_add(facc, f32x4_mul(f32x4_convert_i32x4(dot), f32x4_splat(s)));
            sxs += s * (*xsum.add(b) as f32);
            b += 1;
        }
        let hs = f32x4_extract_lane::<0>(facc)
            + f32x4_extract_lane::<1>(facc)
            + f32x4_extract_lane::<2>(facc)
            + f32x4_extract_lane::<3>(facc);
        *out.add(row) = hs - sxs;
        row += 1;
    }
}

// ─────────────────────────── main-thread serial ops (SIMD) ───────────────────────────
// These run on the COORDINATOR's own wasm instance (private memory). They replace the
// scalar-JS quant / rmsnorm on the critical path (blocking all workers). K is always a
// multiple of 64 here (H=2048, I=6144), so no scalar tail.

#[inline(always)]
unsafe fn hsum_i32x4(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v) + i32x4_extract_lane::<1>(v)
        + i32x4_extract_lane::<2>(v) + i32x4_extract_lane::<3>(v)
}

#[inline(always)]
unsafe fn qi32(x: *const f32, i: usize, vinv: v128) -> v128 {
    // round-to-nearest(x*inv) → i32, clamped to [-127,127] (matches JS Math.round+clamp
    // closely; half-to-even vs half-up differs only on exact .5 = quant noise).
    let v = f32x4_mul(v128_load(x.add(i) as *const v128), vinv);
    let q = i32x4_trunc_sat_f32x4(f32x4_nearest(v));
    i32x4_min(i32x4_max(q, i32x4_splat(-127)), i32x4_splat(127))
}

/// Per-tensor int8 quantize of x[k] → dst (i8) + per-group(64) sums xsum (i32).
/// Returns the activation scale asc = amax/127 (JS multiplies gemv output by it).
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn quantize(dst: *mut i8, xsum: *mut i32, x: *const f32, k: u32) -> f32 {
    let k = k as usize;
    // pass 1: amax(|x|)
    let mut vmax = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= k {
        vmax = f32x4_max(vmax, f32x4_abs(v128_load(x.add(i) as *const v128)));
        i += 4;
    }
    let amax = f32x4_extract_lane::<0>(vmax)
        .max(f32x4_extract_lane::<1>(vmax))
        .max(f32x4_extract_lane::<2>(vmax))
        .max(f32x4_extract_lane::<3>(vmax));
    let asc = if amax > 0.0 { amax / 127.0 } else { 1e-9 };
    let vinv = f32x4_splat(1.0 / asc);
    // pass 2: quantize 16/iter, 64/group; narrow i32→i16→i8 (saturating, already clamped)
    let mut b = 0usize; // block (group) index
    i = 0;
    while i < k {
        let mut gsum = i32x4_splat(0);
        let mut j = 0usize;
        while j < 64 {
            let q0 = qi32(x, i + j, vinv);
            let q1 = qi32(x, i + j + 4, vinv);
            let q2 = qi32(x, i + j + 8, vinv);
            let q3 = qi32(x, i + j + 12, vinv);
            gsum = i32x4_add(gsum, i32x4_add(i32x4_add(q0, q1), i32x4_add(q2, q3)));
            let i8v = i8x16_narrow_i16x8(i16x8_narrow_i32x4(q0, q1), i16x8_narrow_i32x4(q2, q3));
            v128_store(dst.add(i + j) as *mut v128, i8v);
            j += 16;
        }
        *xsum.add(b) = hsum_i32x4(gsum);
        b += 1;
        i += 64;
    }
    asc
}

/// out = (x / sqrt(mean(x²) + eps)) * w, over n (multiple of 4).
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn rmsnorm(dst: *mut f32, x: *const f32, w: *const f32, n: u32, eps: f32) {
    let n = n as usize;
    let mut vs = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= n {
        let v = v128_load(x.add(i) as *const v128);
        vs = f32x4_add(vs, f32x4_mul(v, v));
        i += 4;
    }
    let ss = f32x4_extract_lane::<0>(vs) + f32x4_extract_lane::<1>(vs)
        + f32x4_extract_lane::<2>(vs) + f32x4_extract_lane::<3>(vs);
    // inv = 1/sqrt(ss/n + eps) via the SIMD sqrt intrinsic (no std in no_std).
    let root = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(ss / (n as f32) + eps)));
    let vinv = f32x4_splat(1.0 / root);
    i = 0;
    while i + 4 <= n {
        let xv = v128_load(x.add(i) as *const v128);
        let wv = v128_load(w.add(i) as *const v128);
        v128_store(dst.add(i) as *mut v128, f32x4_mul(f32x4_mul(xv, vinv), wv));
        i += 4;
    }
}

// ─────────────────────────── int8 attention (SIMD) ───────────────────────────
// Each worker keeps its heads' KV as int8 (per-vector absmax scale) in its OWN private
// memory. These kernels do the O(T·hd) hot loops; softmax stays in JS (small, over T).
// Full-int8 dot: widen i8→i16, i32x4_dot_i16x8 (i7 relaxed slot can't hold ±127).

/// scores[t] = scale · qs · ks[t] · Σ_i q[i]·k[t·hd+i],  t in 0..t_len.  hd % 16 == 0.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn attn_scores(
    scores: *mut f32, q: *const i8, qs: f32, k: *const i8, ks: *const f32, t_len: u32, hd: u32, scale: f32,
) {
    let hd = hd as usize;
    let t_len = t_len as usize;
    let mut t = 0usize;
    while t < t_len {
        let kr = k.add(t * hd);
        let mut acc = i32x4_splat(0);
        let mut i = 0usize;
        while i < hd {
            let a = v128_load(q.add(i) as *const v128);
            let b = v128_load(kr.add(i) as *const v128);
            acc = i32x4_add(acc, i32x4_dot_i16x8(i16x8_extend_low_i8x16(a), i16x8_extend_low_i8x16(b)));
            acc = i32x4_add(acc, i32x4_dot_i16x8(i16x8_extend_high_i8x16(a), i16x8_extend_high_i8x16(b)));
            i += 16;
        }
        *scores.add(t) = scale * qs * (*ks.add(t)) * (hsum_i32x4(acc) as f32);
        t += 1;
    }
}

/// out[i] = Σ_t (w[t]·vs[t]) · v[t·hd+i],  i in 0..hd.  hd % 16 == 0.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn attn_accv(
    out: *mut f32, w: *const f32, v: *const i8, vs: *const f32, t_len: u32, hd: u32,
) {
    let hd = hd as usize;
    let t_len = t_len as usize;
    let mut i = 0usize;
    while i < hd { v128_store(out.add(i) as *mut v128, f32x4_splat(0.0)); i += 4; }
    let mut t = 0usize;
    while t < t_len {
        let vw = f32x4_splat((*w.add(t)) * (*vs.add(t)));
        let vr = v.add(t * hd);
        let mut i = 0usize;
        while i < hd {
            let b = v128_load(vr.add(i) as *const v128);      // 16 i8
            let bl = i16x8_extend_low_i8x16(b);
            let bh = i16x8_extend_high_i8x16(b);
            let q0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(bl));
            let q1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(bl));
            let q2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(bh));
            let q3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(bh));
            v128_store(out.add(i) as *mut v128, f32x4_add(v128_load(out.add(i) as *const v128), f32x4_mul(vw, q0)));
            v128_store(out.add(i + 4) as *mut v128, f32x4_add(v128_load(out.add(i + 4) as *const v128), f32x4_mul(vw, q1)));
            v128_store(out.add(i + 8) as *mut v128, f32x4_add(v128_load(out.add(i + 8) as *const v128), f32x4_mul(vw, q2)));
            v128_store(out.add(i + 12) as *mut v128, f32x4_add(v128_load(out.add(i + 12) as *const v128), f32x4_mul(vw, q3)));
            i += 16;
        }
        t += 1;
    }
}

/// Per-vector int8 quantize x[n] → dst with absmax scale; returns scale. n % 16 == 0.
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn quant_vec(dst: *mut i8, x: *const f32, n: u32) -> f32 {
    let n = n as usize;
    let mut vmax = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= n { vmax = f32x4_max(vmax, f32x4_abs(v128_load(x.add(i) as *const v128))); i += 4; }
    let amax = f32x4_extract_lane::<0>(vmax).max(f32x4_extract_lane::<1>(vmax))
        .max(f32x4_extract_lane::<2>(vmax)).max(f32x4_extract_lane::<3>(vmax));
    let asc = if amax > 0.0 { amax / 127.0 } else { 1e-9 };
    let vinv = f32x4_splat(1.0 / asc);
    i = 0;
    while i < n {
        let q0 = qi32(x, i, vinv); let q1 = qi32(x, i + 4, vinv);
        let q2 = qi32(x, i + 8, vinv); let q3 = qi32(x, i + 12, vinv);
        v128_store(dst.add(i) as *mut v128, i8x16_narrow_i16x8(i16x8_narrow_i32x4(q0, q1), i16x8_narrow_i32x4(q2, q3)));
        i += 16;
    }
    asc
}

// ─────────────── SwiGLU (SIMD, Cephes-style expf) ───────────────
// dst[i] = gate[i]·sigmoid(gate[i])·up[i]. Poly expf is ~1-2ulp — far below the int8
// activation-quant noise floor, but NOT bit-identical to JS Math.exp (validate tokens).
#[inline(always)]
unsafe fn exp_ps(x: v128) -> v128 {
    let x = f32x4_min(x, f32x4_splat(87.0));
    let x = f32x4_max(x, f32x4_splat(-87.0));
    let kf = f32x4_nearest(f32x4_mul(x, f32x4_splat(1.442695040)));
    let r = f32x4_sub(x, f32x4_mul(kf, f32x4_splat(0.693359375)));
    let r = f32x4_sub(r, f32x4_mul(kf, f32x4_splat(-2.12194440e-4)));
    let mut p = f32x4_splat(1.9875691500e-4);
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(1.3981999507e-3));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(8.3334519073e-3));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(4.1665795894e-2));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(1.6666665459e-1));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(5.0000001201e-1));
    let r2 = f32x4_mul(r, r);
    let e = f32x4_add(f32x4_add(f32x4_splat(1.0), r), f32x4_mul(r2, p));
    let ki = i32x4_trunc_sat_f32x4(kf);
    let pow2k = i32x4_shl(i32x4_add(ki, i32x4_splat(127)), 23); // bits reinterpret as f32
    f32x4_mul(e, pow2k)
}

#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn swiglu(dst: *mut f32, gate: *const f32, up: *const f32, n: u32) {
    let n = n as usize;
    let one = f32x4_splat(1.0);
    let mut i = 0usize;
    while i < n {
        let g = v128_load(gate.add(i) as *const v128);
        let u = v128_load(up.add(i) as *const v128);
        let e = exp_ps(f32x4_neg(g));
        let s = f32x4_div(one, f32x4_add(one, e));
        v128_store(dst.add(i) as *mut v128, f32x4_mul(f32x4_mul(g, s), u));
        i += 4;
    }
}

/// Start of usable linear memory above the module's static data + shadow stack.
/// JS reads this, grows `memory` as needed, and lays out codes/scales/act/xsum/out
/// from here on — no allocator in the wasm.
extern "C" {
    static __heap_base: u8;
}
#[no_mangle]
pub unsafe extern "C" fn heap_base() -> *const u8 {
    &__heap_base as *const u8
}
