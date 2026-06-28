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

pub(crate) fn tokens_per_s(tokens: usize, elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        return "n/a".to_string();
    }
    format!("{:.2} tok/s", tokens as f64 / seconds)
}
