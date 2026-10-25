//! The f32 glue between ternary matmuls: norms, RoPE, activation, softmax.
//!
//! These are transcribed from `tools/ref_forward.py`, which is the oracle the
//! forward pass is validated against. Where the reference makes a choice that has
//! a plausible alternative — and RoPE has a notorious one — the choice is called
//! out in a comment, because picking the other one produces a model that still
//! emits fluent-looking text while being wrong.

/// RMSNorm: `x / sqrt(mean(x^2) + eps) * w`.
///
/// Note this is the plain form, `v * w`. Qwen3.5 uses a *zero-centered* variant
/// where the effective weight is `(1 + w)`; Bonsai does not, and the reference
/// confirms it. Getting this wrong is survivable-looking and badly wrong.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(w.len(), x.len());
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// Per-head RMSNorm over the last dimension, for qk-norm.
///
/// `t` is `[n_heads * head_dim]` and `w` is `[head_dim]`, applied to each head
/// independently — the reference reshapes to `(n_heads, head_dim)` and norms over
/// the trailing axis.
pub fn rmsnorm_heads(t: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    debug_assert_eq!(t.len(), n_heads * head_dim);
    debug_assert_eq!(w.len(), head_dim);
    for h in 0..n_heads {
        let head = &mut t[h * head_dim..][..head_dim];
        let mean_sq = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..head_dim {
            head[i] = head[i] * inv * w[i];
        }
    }
}

/// RoPE, **split-half** convention.
///
/// The two conventions in the wild are *split-half* (pair element `i` with
/// `i + head_dim/2`) and *interleaved* (pair `2i` with `2i+1`). The reference uses
/// split-half:
///
/// ```text
///   x0 = t[..half]; x1 = t[half..]
///   t[..half] = x0*cos - x1*sin
///   t[half..] = x1*cos + x0*sin
/// ```
///
/// Choosing interleaved instead still yields fluent text and a wrong model, so
/// this is pinned by a test against precomputed reference values.
///
/// `inv_freq[i] = theta^(-2i/head_dim)` for `i in 0..half`.
pub fn rope_split_half(t: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    debug_assert_eq!(t.len(), n_heads * head_dim);
    let half = head_dim / 2;
    for h in 0..n_heads {
        let head = &mut t[h * head_dim..][..head_dim];
        for i in 0..half {
            let inv = theta.powf(-((2 * i) as f32) / head_dim as f32);
            let ang = pos as f32 * inv;
            let (s, c) = ang.sin_cos();
            let x0 = head[i];
            let x1 = head[half + i];
            head[i] = x0 * c - x1 * s;
            head[half + i] = x1 * c + x0 * s;
        }
    }
}

/// Precomputed RoPE cos/sin for one position, to keep `powf`/`sin_cos` out of the
/// per-head loop. Layout is `[cos(half), sin(half)]`.
pub fn rope_table(head_dim: usize, pos: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut t = vec![0.0f32; half * 2];
    for i in 0..half {
        let inv = theta.powf(-((2 * i) as f32) / head_dim as f32);
        let ang = pos as f32 * inv;
        let (s, c) = ang.sin_cos();
        t[i] = c;
        t[half + i] = s;
    }
    t
}

/// [`rope_split_half`] using a table from [`rope_table`].
pub fn rope_with_table(t: &mut [f32], n_heads: usize, head_dim: usize, table: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(table.len(), half * 2);
    let (cos, sin) = table.split_at(half);
    for h in 0..n_heads {
        let head = &mut t[h * head_dim..][..head_dim];
        for i in 0..half {
            let x0 = head[i];
            let x1 = head[half + i];
            head[i] = x0 * cos[i] - x1 * sin[i];
            head[half + i] = x1 * cos[i] + x0 * sin[i];
        }
    }
}

/// SwiGLU: `dst = silu(gate) * up`, where `silu(x) = x * sigmoid(x)`.
pub fn swiglu(dst: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(dst.len(), gate.len());
    for i in 0..gate.len() {
        let g = gate[i];
        dst[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// In-place softmax, max-subtracted for stability.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Index of the largest element. Ties go to the lowest index.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Root-mean-square of a vector. Used to compare against the reference's per-layer
/// diagnostics, which is how a forward pass gets validated without eyeballing text.
pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_matches_hand_computation() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &w, 0.0);
        // mean(x^2) = (1+4+9+16)/4 = 7.5, rms = 2.7386
        let expect = 7.5f32.sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] / expect).abs() < 1e-6, "{:?}", out);
        }
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        // pos=0 gives ang=0, cos=1, sin=0, so the vector must come back unchanged.
        let mut t: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let before = t.clone();
        rope_split_half(&mut t, 2, 128, 0, 1e6);
        for (a, b) in t.iter().zip(&before) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    /// Pins the split-half convention against a directly computed value. If someone
    /// switches to interleaved, this fails instead of the model quietly degrading.
    #[test]
    fn rope_uses_the_split_half_pairing() {
        let head_dim = 4usize;
        let half = 2usize;
        let theta = 1e6f32;
        let pos = 3usize;
        let mut t = vec![1.0f32, 2.0, 3.0, 4.0];
        rope_split_half(&mut t, 1, head_dim, pos, theta);

        // i=0: inv = theta^0 = 1, ang = 3
        // i=1: inv = theta^(-2/4) = 1/1000, ang = 0.003
        for i in 0..half {
            let inv = theta.powf(-((2 * i) as f32) / head_dim as f32);
            let ang = pos as f32 * inv;
            let (s, c) = ang.sin_cos();
            let x0 = (i + 1) as f32; // original t[i]
            let x1 = (half + i + 1) as f32; // original t[half+i]
            assert!((t[i] - (x0 * c - x1 * s)).abs() < 1e-5, "lane {i}");
            assert!((t[half + i] - (x1 * c + x0 * s)).abs() < 1e-5, "lane {}", half + i);
        }
    }

    #[test]
    fn rope_table_agrees_with_the_direct_form() {
        let mut a: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let mut b = a.clone();
        rope_split_half(&mut a, 1, 128, 17, 1e6);
        let table = rope_table(128, 17, 1e6);
        rope_with_table(&mut b, 1, 128, &table);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = vec![1000.0f32, 1001.0, 999.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "{x:?}");
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(x[1] > x[0] && x[0] > x[2]);
    }

    #[test]
    fn rmsnorm_heads_normalises_each_head_independently() {
        // Two heads with very different magnitudes must both come out at rms 1.
        let mut t = vec![0.0f32; 8];
        t[..4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        t[4..].copy_from_slice(&[100.0, 100.0, 100.0, 100.0]);
        let w = [1.0f32; 4];
        rmsnorm_heads(&mut t, &w, 2, 4, 0.0);
        for h in 0..2 {
            assert!((rms(&t[h * 4..][..4]) - 1.0).abs() < 1e-5, "head {h}: {t:?}");
        }
    }
}
