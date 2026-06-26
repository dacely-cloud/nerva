#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

pub mod contract;
pub mod registry;

#[cfg(test)]
mod tests;
