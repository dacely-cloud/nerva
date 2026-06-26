use std::time::Instant;

pub(crate) fn elapsed_ns(start: Instant) -> u64 {
    let elapsed = start.elapsed().as_nanos();
    elapsed.max(1).min(u64::MAX as u128) as u64
}
