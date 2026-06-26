use nerva_core::types::dtype::DType;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrecisionSafetensorsBlockSmokeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionSafetensorsBlockSmokeSummary {
    pub status: PrecisionSafetensorsBlockSmokeStatus,
    pub dtype: DType,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub output_hash: u64,
    pub expected_hash: u64,
    pub bit_parity: bool,
    pub hot_path_allocations: u64,
}

impl PrecisionSafetensorsBlockSmokeSummary {
    pub fn passed(&self) -> bool {
        self.bit_parity && self.hot_path_allocations == 0 && self.tensors_loaded == 9
    }
}
