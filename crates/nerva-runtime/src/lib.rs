#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod capabilities;
pub mod engine;
pub mod graph;
pub mod token;
pub mod transport;
pub mod weights;
