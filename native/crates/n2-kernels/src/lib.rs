//! Ternary inference kernels.
//!
//! The weight format is Bonsai-native `Q2_0_g64`, carried over verbatim from the
//! WASM engine this repo replaces (see `docs/FORMAT.md` for the byte-level spec
//! and `seed/cpukern-wasm-lib.rs` for the original kernels).
//!
//! Every backend must agree with [`tern::gemv_tern_scalar`], which in turn is
//! checked against a plain f32 dequantise-and-multiply reference in the tests.
//! That second check is the one that catches a misread of the *interleaved* code
//! layout — a bug a backend-vs-backend test cannot see, because both backends
//! would be wrong in the same way.

pub mod cpu;
pub mod tern;
pub mod timing;

/// Scale-group size along K. One f32 scale per 64 weights.
pub const G: usize = 64;

/// Bytes of packed codes per scale group (64 codes x 2 bits).
pub const BYTES_PER_BLOCK: usize = G / 4;
