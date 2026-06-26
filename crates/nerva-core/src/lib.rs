#![forbid(unsafe_code)]

//! Core identities, residency types, and shared errors.

#[cfg(not(target_os = "linux"))]
compile_error!(
    "NERVA currently supports Linux only. Ubuntu x86_64 and aarch64 are the M0 host targets."
);

mod types;

pub use types::*;

#[cfg(test)]
mod tests;
