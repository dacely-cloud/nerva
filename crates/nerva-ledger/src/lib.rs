#![forbid(unsafe_code)]

//! Token-level observability and scheduling ledgers.
//!
//! `types` owns the hot-path data model and validation, while `json` owns the
//! artifact serialization surface.

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

mod json;
mod types;

pub use types::*;

#[cfg(test)]
mod tests;
