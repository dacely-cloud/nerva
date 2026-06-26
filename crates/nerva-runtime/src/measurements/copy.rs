use std::hint::black_box;
use std::time::Instant;

use crate::measurements::clock::elapsed_ns;
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

pub(crate) fn measure_cpu_copy() -> MeasurementEntry {
    const BYTES: usize = 64 * 1024;
    const ITERATIONS: u64 = 128;

    let source = vec![0x5au8; BYTES];
    let mut destination = vec![0u8; BYTES];
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        destination.copy_from_slice(&source);
        black_box(destination[0]);
    }
    MeasurementEntry::runtime_timestamp(
        MeasurementKind::CpuCopy,
        "dram_to_dram_copy_64k",
        BYTES,
        ITERATIONS,
        elapsed_ns(start),
    )
}
