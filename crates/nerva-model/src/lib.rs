#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod attention;
pub mod common;
pub mod hf;
pub mod precision;
pub mod prompt;
pub mod reference;
pub mod tiny;
pub mod warm_compute;
pub mod weights;

#[cfg(test)]
mod tests;
