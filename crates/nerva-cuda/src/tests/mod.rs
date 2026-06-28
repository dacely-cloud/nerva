use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_lock() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

mod attention;
mod backend;
mod block;
mod decode;
mod decode_chain;
mod decode_sequence;
mod decode_sequence_descriptors;
mod decode_sequence_file_descriptors;
mod decode_sequence_qk_norm;
mod decode_sequence_session;
mod graph;
mod json;
mod projection;
mod sampler;
mod smoke;
