//! CUDA backend boundary.
//!
//! Rust owns the public capability surface. Native CUDA code owns direct
//! Runtime API calls and kernel launch mechanics.

pub mod block;
pub mod graph;
pub mod sampler;
pub mod smoke;

#[cfg(test)]
mod tests;
