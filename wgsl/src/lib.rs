// ---
// tags: jali, rust
// crystal-type: source
// crystal-domain: comp
// ---
//! jali-wgsl — GPU backend for polynomial ring NTT via wgpu compute shaders.
//!
//! This crate provides WGSL compute shaders that implement the core operations
//! of R_q = F_p[x]/(x^n+1) over the Goldilocks field on GPU hardware.
//!
//! The shaders are organized in three layers:
//! - `goldilocks.wgsl` — field arithmetic (Fp add, sub, mul, neg) using 2x u32 limbs
//! - `ntt.wgsl`        — negacyclic NTT butterfly stages for GPU-parallel transforms
//! - `ring.wgsl`       — ring-level operations (coefficient-wise add/sub/neg/mul, twist)

pub mod shaders;
