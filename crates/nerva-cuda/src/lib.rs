//! CUDA backend boundary.
//!
//! Rust owns the public capability surface. Native CUDA code owns direct
//! Runtime API calls and kernel launch mechanics.

pub mod attention;
pub mod backend;
pub mod block;
pub mod decode;
pub mod deepseek_kv;
pub mod deepseek_mhc;
pub mod deepseek_mla;
pub mod deepseek_moe;
pub mod deepseek_quant;
pub mod deepseek_router;
pub mod experimental_rt;
pub mod graph;
pub(crate) mod json;
pub mod projection;
pub mod sampler;
pub mod smoke;

#[cfg(test)]
mod tests;
