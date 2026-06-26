use crate::precision::file_smoke::summary::{
    PrecisionSafetensorsBlockSmokeStatus, PrecisionSafetensorsBlockSmokeSummary,
};

impl PrecisionSafetensorsBlockSmokeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            PrecisionSafetensorsBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"dtype\":\"float16\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"output_hash\":{},\"expected_hash\":{},\"bit_parity\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.tensors_loaded,
            self.bytes_loaded,
            self.data_hash,
            self.output_hash,
            self.expected_hash,
            self.bit_parity,
            self.hot_path_allocations,
        )
    }
}
