pub(super) fn deterministic_payload(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| ((index.saturating_mul(31).saturating_add(7)) % 251) as u8)
        .collect()
}
