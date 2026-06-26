#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod arena;
pub mod kv;
pub mod phase;
pub mod queue;
pub mod registry;
pub mod security;

#[cfg(test)]
mod tests;
