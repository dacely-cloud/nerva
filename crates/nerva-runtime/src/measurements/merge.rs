use std::hint::black_box;
use std::time::Instant;

use crate::measurements::clock::elapsed_ns;
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

pub(crate) fn measure_merge() -> MeasurementEntry {
    const VALUES: usize = 4096;
    const ITERATIONS: u64 = 128;

    let left = vec![1.0f32; VALUES];
    let right = vec![2.0f32; VALUES];
    let mut output = vec![0.0f32; VALUES];
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        for ((out, a), b) in output.iter_mut().zip(left.iter()).zip(right.iter()) {
            *out = *a + *b;
        }
        black_box(output[VALUES - 1]);
    }
    MeasurementEntry::runtime_timestamp(
        MeasurementKind::Merge,
        "cpu_partial_output_merge_4096",
        VALUES * core::mem::size_of::<f32>() * 3,
        ITERATIONS,
        elapsed_ns(start),
    )
}
