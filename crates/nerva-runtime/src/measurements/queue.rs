use std::hint::black_box;
use std::time::Instant;

use crate::measurements::clock::elapsed_ns;
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

pub(crate) fn measure_queue_round_trip() -> MeasurementEntry {
    const CAPACITY: usize = 16;
    const ITERATIONS: u64 = 4096;

    let mut ring = BoundedIndexRing::new(CAPACITY);
    let start = Instant::now();
    for value in 0..ITERATIONS {
        assert!(ring.push(value).is_ok());
        let observed = ring.pop().expect("round trip value exists");
        black_box(observed);
    }
    MeasurementEntry::runtime_timestamp(
        MeasurementKind::Queue,
        "bounded_index_queue_round_trip",
        core::mem::size_of::<u64>(),
        ITERATIONS,
        elapsed_ns(start),
    )
}

struct BoundedIndexRing {
    values: Vec<u64>,
    head: usize,
    tail: usize,
    len: usize,
}

impl BoundedIndexRing {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![0; capacity],
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    fn push(&mut self, value: u64) -> Result<(), ()> {
        if self.len == self.values.len() {
            return Err(());
        }
        self.values[self.tail] = value;
        self.tail = (self.tail + 1) % self.values.len();
        self.len += 1;
        Ok(())
    }

    fn pop(&mut self) -> Option<u64> {
        if self.len == 0 {
            return None;
        }
        let value = self.values[self.head];
        self.head = (self.head + 1) % self.values.len();
        self.len -= 1;
        Some(value)
    }
}
