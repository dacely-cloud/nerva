#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod backend;
pub mod capabilities;
pub mod correctness;
pub mod engine;
pub mod execution;
pub mod graph;
pub mod measurements;
pub mod memory_loop;
pub mod mgpu;
pub mod production;
pub mod request;
pub mod token;
pub mod transport;
pub mod weights;
