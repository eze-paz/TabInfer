//! N2 core: the model container and the pieces around the kernels.
//!
//! Scope today is the weight container and the format plumbing. The forward pass
//! is next — see `docs/ROADMAP.md`.

pub mod binfmt;
pub mod json;
pub mod model;
pub mod ops;

pub use binfmt::{Container, Tensor, TernaryView};
pub use model::{Config, KvCache, Model};
