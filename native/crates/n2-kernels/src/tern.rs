//! Ternary GEMV: `out[N] = W · act`, where `W[n,k] = (code[n,k] - 1) * scale[n, k/64]`.
//!
//! Because codes are stored as {0,1,2} rather than {-1,0,+1}, the kernel computes
//!
//! ```text
//!   out[n] = Σ_b scale[n,b] · dot_b   -   Σ_b scale[n,b] · xsum[b]
//! ```
//!
//! where `dot_b = Σ_{k∈b} code[n,k]·act[k]` and `xsum[b] = Σ_{k∈b} act[k]`.
//! That is BitNet's no-multiply identity; the subtraction of `xsum` is what turns
//! the unsigned codes back into signed weights. It is *not* an optimisation we
//! can drop — the two terms are accumulated separately and subtracted once per
//! row, and every backend must preserve that grouping so their rounding matches.

use crate::{BYTES_PER_BLOCK, G};

/// Which implementation [`gemv_tern`] will dispatch to on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Avx2,
}

impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Backend::Scalar => write!(f, "scalar"),
            Backend::Avx2 => write!(f, "avx2"),
        }
    }
}

/// The backend [`gemv_tern`] selects on this machine, decided at runtime.
pub fn backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return Backend::Avx2;
        }
    }
    Backend::Scalar
}

/// Shapes that every kernel in this module assumes. Checked once, at the edge.
fn check(out: &[f32], codes: &[u8], scales: &[f32], act: &[i8], xsum: &[i32], n: usize, k: usize) {
    assert_eq!(k % G, 0, "K must be a multiple of the group size {G}");
    let nblocks = k / G;
    assert_eq!(out.len(), n, "out must be N long");
    assert_eq!(codes.len(), n * k / 4, "codes must be N*K/4 bytes");
    assert_eq!(scales.len(), n * nblocks, "scales must be N*(K/64) floats");
    assert_eq!(act.len(), k, "act must be K long");
    assert_eq!(xsum.len(), nblocks, "xsum must be K/64 long");
}

/// Dispatching entry point. Picks the best backend for this CPU.
pub fn gemv_tern(
    out: &mut [f32],
    codes: &[u8],
    scales: &[f32],
    act: &[i8],
    xsum: &[i32],
    n: usize,
    k: usize,
) {
    check(out, codes, scales, act, xsum, n, k);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { avx2::gemv_tern(out, codes, scales, act, xsum, n, k) },
        _ => gemv_tern_scalar(out, codes, scales, act, xsum, n, k),
    }
}

/// Portable reference. Every other backend is validated against this.
///
/// Deliberately written to mirror the accumulation *grouping* of the SIMD paths
/// (per-block `scale*dot` into one accumulator, `scale*xsum` into another,
/// subtracted once at the end) so that cross-backend differences come from lane
/// ordering alone and stay in the 1e-6 relative range.
pub fn gemv_tern_scalar(
    out: &mut [f32],
    codes: &[u8],
    scales: &[f32],
    act: &[i8],
    xsum: &[i32],
    n: usize,
    k: usize,
) {
    check(out, codes, scales, act, xsum, n, k);
    let nblocks = k / G;
    let bytes_per_row = k / 4;

    for row in 0..n {
        let rc = &codes[row * bytes_per_row..][..bytes_per_row];
        let rs = &scales[row * nblocks..][..nblocks];

        let mut acc = 0.0f32;
        let mut sxs = 0.0f32;

        for b in 0..nblocks {
            let blk = &rc[b * BYTES_PER_BLOCK..][..BYTES_PER_BLOCK];
            let ab = &act[b * G..][..G];

            let mut dot = 0i32;
            // Plane j (bit-position 2j of each byte) holds codes j*16 .. j*16+15.
            for j in 0..4 {
                for i in 0..BYTES_PER_BLOCK {
                    let c = ((blk[i] >> (2 * j)) & 3) as i32;
                    dot += c * ab[j * BYTES_PER_BLOCK + i] as i32;
                }
            }

            let s = rs[b];
            acc += s * dot as f32;
            sxs += s * xsum[b] as f32;
        }

        out[row] = acc - sxs;
    }
}

// ── Reference helpers: the ground truth the format itself is checked against ──

/// Pack an f32 ternary row-major matrix into the on-disk layout.
///
/// This is a direct transcription of `tools/pack_bin.py::pack_ternary`. Keeping a
/// Rust copy means the tests can generate synthetic weights in the *real* layout,
/// so a misunderstanding of the packing shows up as a test failure rather than as
/// plausible garbage at inference time.
///
/// Returns `(codes, scales)` with `codes` shaped `N x K/4` and `scales` `N x K/64`.
pub fn pack_ternary(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(w.len(), n * k);
    assert_eq!(k % G, 0);
    let nblocks = k / G;
    let mut codes = vec![0u8; n * k / 4];
    let mut scales = vec![0f32; n * nblocks];

    for row in 0..n {
        for b in 0..nblocks {
            let g = &w[row * k + b * G..][..G];
            let s = g.iter().fold(0f32, |m, v| m.max(v.abs()));
            scales[row * nblocks + b] = s;
            let thr = s * 0.5;
            for (c, &v) in g.iter().enumerate() {
                // {0,1,2} == {-1,0,+1} shifted; below threshold rounds to zero.
                let code: u8 = if v.abs() < thr {
                    1
                } else if v < 0.0 {
                    0
                } else {
                    2
                };
                let plane = c / BYTES_PER_BLOCK; // 0..4
                let byte = c % BYTES_PER_BLOCK; // 0..16
                codes[row * (k / 4) + b * BYTES_PER_BLOCK + byte] |= code << (2 * plane);
            }
        }
    }
    (codes, scales)
}

/// Dequantise to f32 and multiply — the slowest, most obviously-correct path.
///
/// This is the independent oracle: it reconstructs `W` from `(codes, scales)` and
/// does a plain dot product with no `xsum` trick and no packing cleverness.
pub fn gemv_dequant_reference(
    out: &mut [f32],
    codes: &[u8],
    scales: &[f32],
    act: &[i8],
    n: usize,
    k: usize,
) {
    let nblocks = k / G;
    for row in 0..n {
        let mut acc = 0.0f64;
        for kk in 0..k {
            let b = kk / G;
            let c = kk % G;
            let plane = c / BYTES_PER_BLOCK;
            let byte = c % BYTES_PER_BLOCK;
            let raw = codes[row * (k / 4) + b * BYTES_PER_BLOCK + byte];
            let code = ((raw >> (2 * plane)) & 3) as i32;
            let w = (code - 1) as f32 * scales[row * nblocks + b];
            acc += (w * act[kk] as f32) as f64;
        }
        out[row] = acc as f32;
    }
}

/// Quantise f32 activations to i8 and produce the per-block sums the kernel needs.
///
/// Returns `(act, xsum, inv_scale)`; the caller multiplies kernel output by
/// `1.0 / inv_scale` to get back to f32 units. `amax == 0` yields an all-zero
/// activation vector and a scale of 1.0 rather than a NaN.
pub fn quantize_act(x: &[f32]) -> (Vec<i8>, Vec<i32>, f32) {
    assert_eq!(x.len() % G, 0, "activation length must be a multiple of {G}");
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = if amax > 0.0 { 127.0 / amax } else { 1.0 };

    let mut act = vec![0i8; x.len()];
    for (a, &v) in act.iter_mut().zip(x) {
        // round-half-away-from-zero, then clamp — matches the JS engine's Math.round
        *a = (v * scale).round().clamp(-127.0, 127.0) as i8;
    }

    let mut xsum = vec![0i32; x.len() / G];
    for (b, s) in xsum.iter_mut().enumerate() {
        *s = act[b * G..][..G].iter().map(|&v| v as i32).sum();
    }
    (act, xsum, scale)
}

// ── x86_64 AVX2 ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// AVX2 ternary GEMV. Processes **two** 64-code blocks per iteration so the
    /// code load and the 2-bit unpack run at 256 bits; the two blocks' differing
    /// scales are applied by keeping block `b` in the low 128-bit half of the
    /// accumulator and block `b+1` in the high half, then folding both into one
    /// f32x8 accumulator that is horizontally summed once per row.
    ///
    /// Honest note on the dot product: `vpmaddubsw` + `vpmaddwd` is two
    /// instructions where WASM `relaxed_dot` was one, so at 256 bits this path
    /// breaks even on multiply-accumulate *issue* and wins only on the halved
    /// load + unpack traffic. The 2x on the dot itself needs AVX-VNNI's single
    /// `vpdpbusd`; see docs/ROADMAP.md. `n2 cpu` reports whether this box has it.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available and the slice shapes satisfy
    /// [`super::check`].
    #[target_feature(enable = "avx2")]
    pub unsafe fn gemv_tern(
        out: &mut [f32],
        codes: &[u8],
        scales: &[f32],
        act: &[i8],
        xsum: &[i32],
        n: usize,
        k: usize,
    ) {
        let nblocks = k / G;
        let bytes_per_row = k / 4;
        let pairs = nblocks / 2;

        let m3 = _mm256_set1_epi8(3);
        let ones = _mm256_set1_epi16(1);

        let codes_p = codes.as_ptr();
        let act_p = act.as_ptr();

        for row in 0..n {
            let rc = codes_p.add(row * bytes_per_row);
            let rs = scales.as_ptr().add(row * nblocks);

            let mut facc = _mm256_setzero_ps();
            let mut sxs = 0.0f32;

            for p in 0..pairs {
                let b = p * 2;
                // 32 bytes = block b (low half) followed by block b+1 (high half).
                let raw = _mm256_loadu_si256(rc.add(b * BYTES_PER_BLOCK) as *const __m256i);

                let mut iacc = _mm256_setzero_si256();
                // The shift amount is an immediate, so the four planes are
                // unrolled rather than looped. A 16-bit shift plus a 2-bit mask
                // is equivalent to a per-byte shift here: bits that bleed across
                // the byte boundary land above the mask.
                macro_rules! plane {
                    ($j:expr, $sh:literal) => {{
                        let plane = _mm256_and_si256(_mm256_srli_epi16(raw, $sh), m3);
                        let lo = _mm_loadu_si128(
                            act_p.add(b * G + $j * BYTES_PER_BLOCK) as *const __m128i,
                        );
                        let hi = _mm_loadu_si128(
                            act_p.add((b + 1) * G + $j * BYTES_PER_BLOCK) as *const __m128i,
                        );
                        let a = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
                        // codes are unsigned {0,1,2}, activations signed: exactly
                        // the operand convention vpmaddubsw wants. Max |2*127|*2
                        // = 508, so the i16 intermediate cannot saturate.
                        let prod = _mm256_maddubs_epi16(plane, a);
                        iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(prod, ones));
                    }};
                }
                plane!(0usize, 0);
                plane!(1usize, 2);
                plane!(2usize, 4);
                plane!(3usize, 6);

                // Lanes 0..3 belong to block b, lanes 4..7 to block b+1.
                let s0 = *rs.add(b);
                let s1 = *rs.add(b + 1);
                let sv = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_set1_ps(s0)),
                    _mm_set1_ps(s1),
                    1,
                );
                facc = _mm256_add_ps(facc, _mm256_mul_ps(_mm256_cvtepi32_ps(iacc), sv));

                sxs += s0 * *xsum.get_unchecked(b) as f32;
                sxs += s1 * *xsum.get_unchecked(b + 1) as f32;
            }

            let mut acc = hsum256_ps(facc);

            // Odd trailing block, if K/64 is not even.
            for b in (pairs * 2)..nblocks {
                let blk = &codes[row * bytes_per_row + b * BYTES_PER_BLOCK..][..BYTES_PER_BLOCK];
                let ab = &act[b * G..][..G];
                let mut dot = 0i32;
                for j in 0..4 {
                    for i in 0..BYTES_PER_BLOCK {
                        dot += (((blk[i] >> (2 * j)) & 3) as i32) * ab[j * BYTES_PER_BLOCK + i] as i32;
                    }
                }
                let s = *rs.add(b);
                acc += s * dot as f32;
                sxs += s * xsum[b] as f32;
            }

            out[row] = acc - sxs;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so failures reproduce without a dev-dependency.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        /// Ternary-ish values with a spread of magnitudes, so the 0.5*amax
        /// threshold in `pack_ternary` actually exercises all three codes.
        fn ternaryish(&mut self) -> f32 {
            match self.next_u32() % 5 {
                0 => -1.0,
                1 => 1.0,
                2 => 0.0,
                3 => 0.2,
                _ => -0.9,
            }
        }
        fn act(&mut self) -> f32 {
            (self.next_u32() % 2001) as f32 / 1000.0 - 1.0
        }
    }

    fn fixture(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>, Vec<i8>, Vec<i32>) {
        let mut rng = Rng(seed);
        let w: Vec<f32> = (0..n * k).map(|_| rng.ternaryish()).collect();
        let (codes, scales) = pack_ternary(&w, n, k);
        let x: Vec<f32> = (0..k).map(|_| rng.act()).collect();
        let (act, xsum, _) = quantize_act(&x);
        (codes, scales, act, xsum)
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = (x - y).abs();
                let m = x.abs().max(y.abs()).max(1e-3);
                d / m
            })
            .fold(0.0f32, f32::max)
    }

    /// The load-bearing test: the packed-format kernel against an independent
    /// dequantise-and-multiply oracle. This is what catches a misread of the
    /// interleaved plane layout.
    #[test]
    fn scalar_matches_dequant_oracle() {
        for &(n, k) in &[(1, 64), (3, 128), (17, 256), (8, 1024)] {
            let (codes, scales, act, xsum) = fixture(n, k, 0x2026_0805 + k as u64);
            let mut got = vec![0f32; n];
            let mut want = vec![0f32; n];
            gemv_tern_scalar(&mut got, &codes, &scales, &act, &xsum, n, k);
            gemv_dequant_reference(&mut want, &codes, &scales, &act, n, k);
            let rel = max_rel(&got, &want);
            assert!(rel < 1e-5, "N={n} K={k}: max rel err {rel:e}\ngot {got:?}\nwant {want:?}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        // Include an odd block count (K=192 -> 3 blocks) to hit the scalar tail.
        for &(n, k) in &[(1, 64), (1, 192), (5, 192), (17, 256), (64, 2048)] {
            let (codes, scales, act, xsum) = fixture(n, k, 0xC0FFEE + k as u64);
            let mut a = vec![0f32; n];
            let mut b = vec![0f32; n];
            gemv_tern_scalar(&mut a, &codes, &scales, &act, &xsum, n, k);
            unsafe { avx2::gemv_tern(&mut b, &codes, &scales, &act, &xsum, n, k) };
            let rel = max_rel(&a, &b);
            assert!(rel < 1e-5, "N={n} K={k}: avx2 vs scalar max rel err {rel:e}");
        }
    }

    /// A zero activation vector must give exactly zero, not NaN — the amax==0
    /// path in `quantize_act` used to be a live source of silent NaNs upstream.
    #[test]
    fn zero_activations_are_zero_not_nan() {
        let (n, k) = (4, 128);
        let mut rng = Rng(7);
        let w: Vec<f32> = (0..n * k).map(|_| rng.ternaryish()).collect();
        let (codes, scales) = pack_ternary(&w, n, k);
        let (act, xsum, s) = quantize_act(&vec![0.0f32; k]);
        assert_eq!(s, 1.0);
        let mut out = vec![f32::NAN; n];
        gemv_tern(&mut out, &codes, &scales, &act, &xsum, n, k);
        assert!(out.iter().all(|v| *v == 0.0), "expected all zeros, got {out:?}");
    }

    /// Packing must round-trip the three code values and place them in the plane
    /// the kernel reads. Checked directly on a hand-built row.
    #[test]
    fn packing_places_codes_in_the_expected_plane() {
        let k = 64;
        let mut w = vec![0.0f32; k];
        w[0] = 1.0; // amax -> scale 1.0, code 2, plane 0, byte 0
        w[16] = -1.0; // code 0, plane 1, byte 0
        w[33] = 1.0; // code 2, plane 2, byte 1
        let (codes, scales) = pack_ternary(&w, 1, k);
        assert_eq!(scales[0], 1.0);
        // byte 0 holds elements 0,16,32,48 at bit positions 0,2,4,6.
        // element 0 -> 2, element 16 -> 0, elements 32,48 -> 1 (below threshold)
        assert_eq!(codes[0] & 3, 2);
        assert_eq!((codes[0] >> 2) & 3, 0);
        assert_eq!((codes[0] >> 4) & 3, 1);
        // byte 1 holds elements 1,17,33,49; element 33 is plane 2.
        assert_eq!((codes[1] >> 4) & 3, 2);
    }
}
