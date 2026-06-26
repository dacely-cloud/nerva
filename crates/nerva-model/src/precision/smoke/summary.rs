use nerva_core::types::dtype::DType;

use crate::precision::bits::dtype_label;
use crate::precision::smoke::status::PrecisionBlockSmokeStatus;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PrecisionDTypeBlockSmokeSummary {
    pub dtype: DType,
    pub bit_parity: bool,
    pub output_bits: [u16; 2],
    pub expected_bits: [u16; 2],
    pub output_hash: u64,
    pub expected_hash: u64,
    pub max_abs_error: f32,
    pub hot_path_allocations: u64,
}

impl PrecisionDTypeBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let dtype = dtype_label(self.dtype).unwrap_or("unsupported");
        format!(
            "{{\"dtype\":\"{}\",\"bit_parity\":{},\"output_bits\":[{},{}],\"expected_bits\":[{},{}],\"output_hash\":{},\"expected_hash\":{},\"max_abs_error\":{},\"hot_path_allocations\":{}}}",
            dtype,
            self.bit_parity,
            self.output_bits[0],
            self.output_bits[1],
            self.expected_bits[0],
            self.expected_bits[1],
            self.output_hash,
            self.expected_hash,
            self.max_abs_error,
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PrecisionBlockSmokeSummary {
    pub status: PrecisionBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub f16: PrecisionDTypeBlockSmokeSummary,
    pub bf16: PrecisionDTypeBlockSmokeSummary,
}

impl PrecisionBlockSmokeSummary {
    pub fn passed(self) -> bool {
        self.f16.bit_parity
            && self.bf16.bit_parity
            && self.f16.hot_path_allocations == 0
            && self.bf16.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            PrecisionBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"f16\":{},\"bf16\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.f16.to_json(),
            self.bf16.to_json(),
        )
    }
}
