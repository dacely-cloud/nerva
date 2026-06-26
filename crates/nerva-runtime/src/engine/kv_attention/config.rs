#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TieredKvAttentionProbeConfig {
    pub tokens_per_page: u32,
    pub page_bytes: usize,
    pub current_step: u64,
}

impl Default for TieredKvAttentionProbeConfig {
    fn default() -> Self {
        Self {
            tokens_per_page: 2,
            page_bytes: 64,
            current_step: 12,
        }
    }
}
