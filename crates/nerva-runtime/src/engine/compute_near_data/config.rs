#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ComputeNearDataProbeConfig {
    pub rows: usize,
    pub cols: usize,
    pub split_row: usize,
}

impl Default for ComputeNearDataProbeConfig {
    fn default() -> Self {
        Self {
            rows: 4,
            cols: 3,
            split_row: 2,
        }
    }
}
