//! Benchmark harness with the measurement discipline baked in.
//!
//! This box is noise-heavy: the WASM engine measured ±40% run-to-run on an
//! identical configuration, and a naive A-then-B loop once produced a **0.78x
//! reversal from monotonic drift** — a wrong sign, not just a wrong magnitude.
//! Three rules came out of that, and they are enforced here rather than left to
//! whoever writes the next benchmark:
//!
//! 1. **Alternate the order** every repetition, so drift cancels instead of
//!    accumulating into whichever variant ran second.
//! 2. **Warm up, then take the minimum** of many reps. The minimum is the least
//!    noise-contaminated estimate of the true cost; the mean measures the OS.
//! 3. **Run a base-vs-base control.** Compare a variant against *itself* and
//!    require the ratio to read ~1.000x. If the control is off, the harness is
//!    lying and no A/B from it means anything.
//!
//! Absolute throughput numbers from this machine are not trustworthy. Relative
//! A/Bs run back-to-back are.

use std::time::{Duration, Instant};

/// Result of an A/B comparison.
pub struct AbResult {
    pub a_label: String,
    pub b_label: String,
    pub a: Duration,
    pub b: Duration,
    pub reps: usize,
}

impl AbResult {
    /// How many times faster `b` is than `a`. >1.0 means `b` won.
    pub fn speedup(&self) -> f64 {
        self.a.as_secs_f64() / self.b.as_secs_f64()
    }
}

impl core::fmt::Display for AbResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:<22} {:>9.3} ms   {:<22} {:>9.3} ms   {:>6.3}x  (min of {})",
            self.a_label,
            self.a.as_secs_f64() * 1e3,
            self.b_label,
            self.b.as_secs_f64() * 1e3,
            self.speedup(),
            self.reps,
        )
    }
}

/// Time `f` once, defeating dead-code elimination via the returned value.
fn once<T>(f: &mut impl FnMut() -> T) -> Duration {
    let t = Instant::now();
    let v = f();
    let d = t.elapsed();
    std::hint::black_box(v);
    d
}

/// Alternating-order, warmed-up, min-of-N A/B.
///
/// `warmup` reps of each are discarded before timing starts.
pub fn ab<TA, TB>(
    a_label: &str,
    mut a: impl FnMut() -> TA,
    b_label: &str,
    mut b: impl FnMut() -> TB,
    reps: usize,
    warmup: usize,
) -> AbResult {
    for _ in 0..warmup {
        once(&mut a);
        once(&mut b);
    }

    let mut ta = Duration::MAX;
    let mut tb = Duration::MAX;

    for i in 0..reps {
        // Alternate which one runs first so monotonic drift cancels.
        if i % 2 == 0 {
            ta = ta.min(once(&mut a));
            tb = tb.min(once(&mut b));
        } else {
            tb = tb.min(once(&mut b));
            ta = ta.min(once(&mut a));
        }
    }

    AbResult {
        a_label: a_label.to_string(),
        b_label: b_label.to_string(),
        a: ta,
        b: tb,
        reps,
    }
}

/// The control run: compare a closure against itself.
///
/// A trustworthy harness returns a speedup within `tol` of 1.0. Anything else
/// means the measurement is dominated by drift or thermal state and every other
/// number from the same session should be discarded.
pub fn control<T>(label: &str, mut f: impl FnMut() -> T, reps: usize, warmup: usize) -> AbResult {
    // Can't hand the same closure to `ab` twice (one &mut each), so the
    // alternating min-of-N loop is inlined here against the single closure.
    for _ in 0..warmup * 2 {
        once(&mut f);
    }

    let mut ta = Duration::MAX;
    let mut tb = Duration::MAX;
    for i in 0..reps {
        // Two timings per rep, in swapped order, so drift cancels between the
        // two accumulators exactly as it does in a real A/B.
        if i % 2 == 0 {
            ta = ta.min(once(&mut f));
            tb = tb.min(once(&mut f));
        } else {
            tb = tb.min(once(&mut f));
            ta = ta.min(once(&mut f));
        }
    }

    AbResult {
        a_label: format!("{label} [ctrl]"),
        b_label: format!("{label} [ctrl']"),
        a: ta,
        b: tb,
        reps,
    }
}

/// Does a control run read close enough to 1.000x to trust the session?
pub fn control_is_clean(r: &AbResult, tol: f64) -> bool {
    (r.speedup() - 1.0).abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_reads_about_unity() {
        let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5).collect();
        let r = control("sum", || data.iter().sum::<f32>(), 31, 3);
        // Generous tolerance: this runs on CI-grade noise as well as a hot laptop.
        assert!(
            control_is_clean(&r, 0.35),
            "control should read ~1.0, read {:.3}x",
            r.speedup()
        );
    }
}
