#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CorrectnessValidationStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrectnessValidationSummary {
    pub status: CorrectnessValidationStatus,
    pub accepted_cases: u64,
    pub bit_exact_cases: u64,
    pub fp_tolerance_cases: u64,
    pub distribution_preserving_cases: u64,
    pub approximate_rejections: u64,
    pub bit_exact_mismatch_rejections: u64,
    pub tolerance_rejections: u64,
    pub exactness_classes_declared: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl CorrectnessValidationSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, CorrectnessValidationStatus::Ok)
            && self.accepted_cases == 3
            && self.bit_exact_cases == 1
            && self.fp_tolerance_cases == 1
            && self.distribution_preserving_cases == 1
            && self.approximate_rejections == 1
            && self.bit_exact_mismatch_rejections == 1
            && self.tolerance_rejections == 1
            && self.exactness_classes_declared == 3
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            CorrectnessValidationStatus::Ok => "ok",
            CorrectnessValidationStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"accepted_cases\":{},\"bit_exact_cases\":{},\"fp_tolerance_cases\":{},\"distribution_preserving_cases\":{},\"approximate_rejections\":{},\"bit_exact_mismatch_rejections\":{},\"tolerance_rejections\":{},\"exactness_classes_declared\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.accepted_cases,
            self.bit_exact_cases,
            self.fp_tolerance_cases,
            self.distribution_preserving_cases,
            self.approximate_rejections,
            self.bit_exact_mismatch_rejections,
            self.tolerance_rejections,
            self.exactness_classes_declared,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
