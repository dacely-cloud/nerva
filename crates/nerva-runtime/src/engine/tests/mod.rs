use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_lock() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

mod arenas;
mod basic;
mod capabilities;
mod compute_near_data;
mod correctness;
mod critical_path;
mod cuda_block;
mod graph;
mod hf_cuda;
mod hf_cuda_contract;
mod hf_cuda_generate;
mod hf_cuda_qwen3;
mod hf_cuda_session;
mod hf_cuda_session_loop;
mod hf_cuda_session_stream;
mod hf_fixture;
mod hot_path;
mod kv;
mod kv_attention;
mod measurements;
mod memory_loop;
mod multi_gpu;
mod production;
mod security;
mod support;
mod synthetic;
mod transaction;
mod transport;
mod weights;
