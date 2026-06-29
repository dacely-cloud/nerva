use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn cuda_test_lock() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
