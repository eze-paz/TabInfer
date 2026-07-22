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
