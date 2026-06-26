#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod capabilities;
mod engine;
pub mod graph;
pub mod probes;
pub mod token;
pub mod transport;
pub mod weights;

pub use engine::cuda_synthetic_graph_smoke;
pub use engine::*;
pub use graph::*;
pub use token::*;
