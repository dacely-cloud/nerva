use std::time::Duration;

pub(crate) fn bytes(value: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut scaled = value as f64;
    let mut unit = 0usize;
    while scaled >= 1024.0 && unit + 1 < UNITS.len() {
        scaled /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", value, UNITS[unit])
    } else {
        format!("{scaled:.2} {}", UNITS[unit])
    }
}

pub(crate) fn duration(value: Duration) -> String {
    let seconds = value.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.2}s")
    } else {
        format!("{:.1}ms", seconds * 1_000.0)
    }
}

pub(crate) fn ms_from_ns(value: u64) -> String {
    format!("{:.3} ms", value as f64 / 1_000_000.0)
}

pub(crate) fn gb_per_s(bytes: u64, elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        return "n/a".to_string();
    }
    format!("{:.2} GB/s", bytes as f64 / 1_000_000_000.0 / seconds)
}

/// Auto-scaling throughput readout (B/s .. TB/s) using decimal (1000) units,
/// matching how memory bandwidth is conventionally reported.
pub(crate) fn bandwidth(bytes_per_s: f64) -> String {
    const UNITS: [&str; 5] = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"];
    if !bytes_per_s.is_finite() || bytes_per_s <= 0.0 {
        return "n/a".to_string();
    }
    let mut scaled = bytes_per_s;
    let mut unit = 0usize;
    while scaled >= 1000.0 && unit + 1 < UNITS.len() {
        scaled /= 1000.0;
        unit += 1;
    }
    format!("{scaled:.2} {}", UNITS[unit])
}

/// Fraction of the device memory roofline at or above which a batch-one decode
/// is treated as memory-bandwidth bound ("hw limit reached").
pub(crate) const HW_LIMIT_FRACTION: f64 = 0.85;

/// Effective weight-streaming bandwidth for a batch-one decode: every generated
/// token reads the full resident weight set once, so this is the memory rate the
/// decode actually sustains and the number to compare against the device roofline.
///
/// When the device's theoretical peak bandwidth is known (`peak_bps > 0`) and the
/// sustained rate reaches [`HW_LIMIT_FRACTION`] of it, the readout is annotated
/// with `(hw limit reached)` — the decode is streaming weights as fast as the
/// memory bus allows and no kernel change can go faster without moving fewer bytes.
pub(crate) fn weight_bandwidth(
    resident_weight_bytes: u64,
    tokens: u64,
    elapsed: Duration,
    peak_bps: u64,
) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON || tokens == 0 || resident_weight_bytes == 0 {
        return "n/a".to_string();
    }
    let achieved_bps = resident_weight_bytes as f64 * tokens as f64 / seconds;
    let readout = bandwidth(achieved_bps);
    if peak_bps > 0 && achieved_bps >= HW_LIMIT_FRACTION * peak_bps as f64 {
        format!("{readout} (hw limit reached)")
    } else {
        readout
    }
}

pub(crate) fn tokens_per_s(tokens: usize, elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        return "n/a".to_string();
    }
    format!("{:.2} tok/s", tokens as f64 / seconds)
}
