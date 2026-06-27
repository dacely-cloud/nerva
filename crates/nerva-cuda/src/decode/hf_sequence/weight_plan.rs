#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceWeightPlan {
    pub blocks: u32,
    pub gpu_resident_blocks: u32,
    pub gpu_staged_blocks: u32,
    pub weight_bytes: u64,
    pub gpu_resident_weight_bytes: u64,
    pub gpu_staged_weight_bytes: u64,
}

impl CudaHfDecodeSequenceWeightPlan {
    pub const fn empty() -> Self {
        Self {
            blocks: 0,
            gpu_resident_blocks: 0,
            gpu_staged_blocks: 0,
            weight_bytes: 0,
            gpu_resident_weight_bytes: 0,
            gpu_staged_weight_bytes: 0,
        }
    }

    pub const fn is_declared(self) -> bool {
        self.blocks != 0 || self.weight_bytes != 0
    }

    pub fn validate(self) -> Option<String> {
        if !self.is_declared() {
            return None;
        }
        if self.blocks == 0 || self.weight_bytes == 0 {
            return Some("CUDA HF decode sequence weight plan is incomplete".to_string());
        }
        if self.gpu_resident_blocks + self.gpu_staged_blocks > self.blocks {
            return Some("CUDA HF decode sequence weight block counts are invalid".to_string());
        }
        if self.gpu_resident_weight_bytes + self.gpu_staged_weight_bytes > self.weight_bytes {
            return Some("CUDA HF decode sequence weight byte counts are invalid".to_string());
        }
        None
    }
}
