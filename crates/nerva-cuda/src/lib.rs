//! CUDA backend boundary.
//!
//! Rust owns the public capability surface. Native CUDA code owns direct
//! Runtime API calls and kernel launch mechanics.

mod smoke;

pub use smoke::{CudaSmokeSummary, SmokeStatus, smoke};

#[cfg(test)]
mod tests;
