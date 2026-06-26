pub mod compute_near_data;
pub mod correctness;
pub mod critical_path;
pub mod cuda;
pub mod hot_path;
pub mod kv_attention;
pub mod kv_probe;
pub mod memory_loop;
pub mod multi_gpu;
pub mod phase_handoff;
pub mod production;
pub mod residency;
pub mod resident_weights;
pub mod runtime;
pub mod security;
pub mod shared_queue;
pub mod static_arena;
pub mod synthetic;
pub mod transaction;
pub mod transport;

#[cfg(test)]
mod tests;
