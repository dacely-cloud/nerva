//! CUDA backend boundary.
//!
//! Rust owns the public capability surface. Native CUDA code owns direct
//! Runtime API calls and kernel launch mechanics.

pub mod attention;
pub mod backend;
pub mod block;
pub mod decode;
pub mod graph;
pub(crate) mod json;
pub mod sampler;
pub mod smoke;

#[cfg(test)]
mod tests;
