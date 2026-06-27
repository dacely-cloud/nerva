pub mod compute_near_data;
pub mod cuda_block;
pub mod hf_cuda;
pub mod hot_path;
pub mod kv_attention;
pub mod kv_probe;
pub mod resident_weights;
pub mod runtime;
pub mod static_arena;
pub mod synthetic;

#[cfg(test)]
mod tests;
