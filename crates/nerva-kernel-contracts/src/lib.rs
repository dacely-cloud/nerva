#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

mod contract;
mod registry;

pub use contract::*;
pub use registry::*;

#[cfg(test)]
mod tests;
