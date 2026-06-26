#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod attention;
mod common;
pub mod hf;
pub mod reference;
pub mod tiny;
pub mod warm_compute;
pub mod weights;

pub use attention::*;
pub use common::TransformerBlockShape;
pub use hf::*;
pub use reference::*;
pub use tiny::*;
pub use warm_compute::*;
pub use weights::*;

#[cfg(test)]
mod tests;
