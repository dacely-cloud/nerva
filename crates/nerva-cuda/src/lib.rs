//! CUDA backend boundary.
//!
//! Rust owns the public capability surface. Native CUDA code owns direct
//! Runtime API calls and kernel launch mechanics.

mod graph;
mod smoke;

pub use graph::{CudaSyntheticGraphSummary, synthetic_graph_smoke};
pub use smoke::{CudaSmokeSummary, SmokeStatus, smoke};

#[cfg(test)]
mod tests;
