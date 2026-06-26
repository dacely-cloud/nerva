use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::time::Instant;

use crate::measurements::clock::elapsed_ns;
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

pub(crate) fn measure_sync_loop() -> MeasurementEntry {
    const ITERATIONS: u64 = 4096;

    let flag = AtomicU64::new(0);
    let start = Instant::now();
    for value in 0..ITERATIONS {
        flag.store(value, Ordering::Release);
        fence(Ordering::SeqCst);
        black_box(flag.load(Ordering::Acquire));
    }
    MeasurementEntry::runtime_timestamp(
        MeasurementKind::Sync,
        "atomic_release_acquire_round_trip",
        core::mem::size_of::<u64>(),
        ITERATIONS,
        elapsed_ns(start),
    )
}
