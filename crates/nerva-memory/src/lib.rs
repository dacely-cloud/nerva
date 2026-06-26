#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

mod arena;
mod kv;
mod registry;

pub use arena::*;
pub use kv::*;
pub use registry::*;

#[cfg(test)]
mod tests;
