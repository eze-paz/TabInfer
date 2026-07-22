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
#[target_feature(enable = "simd128,relaxed-simd")]
#[no_mangle]
pub unsafe extern "C" fn gemv_tern(
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
