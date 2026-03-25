// ---
// tags: jali, rust
// crystal-type: source
// crystal-domain: comp
// ---
//! WGSL shader sources for Goldilocks field and ring arithmetic on GPU.
//!
//! Each constant holds the WGSL source text for a compute shader module.
//! The host-side Rust code includes these at compile time and passes them
//! to wgpu's shader compilation pipeline.

/// Goldilocks field arithmetic (Fp struct, add, sub, mul, neg).
pub const GOLDILOCKS: &str = include_str!("goldilocks.wgsl");

/// NTT butterfly stages for negacyclic NTT (forward and inverse).
pub const NTT: &str = include_str!("ntt.wgsl");

/// Ring-level operations: coefficient-wise add/sub/neg, pointwise mul, twist/untwist.
pub const RING: &str = include_str!("ring.wgsl");
