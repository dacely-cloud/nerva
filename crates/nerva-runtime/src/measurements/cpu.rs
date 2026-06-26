use std::hint::black_box;
use std::time::Instant;

use crate::measurements::clock::elapsed_ns;
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

pub(crate) fn measure_cpu_dot() -> MeasurementEntry {
    const VALUES: usize = 1024;
    const ITERATIONS: u64 = 512;

    let lhs = vec![1.25f32; VALUES];
    let rhs = vec![0.5f32; VALUES];
    let mut acc = 0.0f32;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        acc += lhs
            .iter()
            .zip(rhs.iter())
            .map(|(left, right)| left * right)
            .sum::<f32>();
        black_box(acc);
    }
    MeasurementEntry::runtime_timestamp(
        MeasurementKind::CpuKernel,
        "cpu_f32_dot_1024",
        VALUES * core::mem::size_of::<f32>() * 2,
        ITERATIONS,
        elapsed_ns(start),
    )
}
