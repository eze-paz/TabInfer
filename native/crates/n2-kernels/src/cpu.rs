//! What this machine can actually do.
//!
//! The WASM engine could only ever see "how many logical cores" and 128-bit SIMD.
//! Natively we can see the instruction sets that decide the kernel's ceiling —
//! in particular whether AVX-VNNI is present, which is the difference between a
//! two-instruction and a one-instruction int8 dot product.

/// A SIMD/ISA feature that changes which kernel we should run.
#[derive(Debug, Clone, Copy)]
pub struct Feature {
    pub name: &'static str,
    pub present: bool,
    pub why: &'static str,
}

pub fn features() -> Vec<Feature> {
    let mut v = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        // `is_x86_feature_detected!` needs a genuine string literal at the call
        // site — a `$n:literal` macro capture is opaque to it — so these are
        // written out one by one on purpose.
        v.push(Feature {
            name: "avx2",
            present: std::is_x86_feature_detected!("avx2"),
            why: "256-bit integer SIMD: 2x the vector width WASM could reach",
        });
        v.push(Feature {
            name: "fma",
            present: std::is_x86_feature_detected!("fma"),
            why: "fused multiply-add for the f32 glue (norms, rope, softmax)",
        });
        v.push(Feature {
            name: "avxvnni",
            present: std::is_x86_feature_detected!("avxvnni"),
            why: "single-instruction u8xi8 dot (vpdpbusd): the 2x on the ternary kernel",
        });
        v.push(Feature {
            name: "avx512f",
            present: std::is_x86_feature_detected!("avx512f"),
            why: "512-bit; fused off on most consumer Alder Lake parts",
        });
        v.push(Feature {
            name: "avx512vnni",
            present: std::is_x86_feature_detected!("avx512vnni"),
            why: "512-bit vpdpbusd",
        });
        v.push(Feature {
            name: "bmi2",
            present: std::is_x86_feature_detected!("bmi2"),
            why: "faster bit extraction for the 2-bit unpack",
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        v.push(Feature {
            name: "neon",
            present: std::arch::is_aarch64_feature_detected!("neon"),
            why: "128-bit SIMD baseline",
        });
        v.push(Feature {
            name: "dotprod",
            present: std::arch::is_aarch64_feature_detected!("dotprod"),
            why: "sdot/udot: the ARM equivalent of vpdpbusd",
        });
        v.push(Feature {
            name: "i8mm",
            present: std::arch::is_aarch64_feature_detected!("i8mm"),
            why: "int8 matrix multiply, for batched prefill",
        });
    }

    v
}

/// Logical CPU count as reported by the OS.
///
/// Note this is *logical*: on the target box (Alder Lake mobile) it counts both
/// SMT siblings and E-cores, and the WASM engine's measured result was that
/// oversubscribing past the physical P-core count made things slower — 6 workers
/// beat 11. Thread-count defaults should come from measurement on the box, not
/// from this number. See `docs/HANDOFF-cpubonsai.md`.
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1)
}
