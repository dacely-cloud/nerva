use crate::reference::smoke::status::ReferenceBlockSmokeStatus;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReferenceBlockSmokeSummary {
    pub status: ReferenceBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub hot_path_allocations: u64,
}

impl ReferenceBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            ReferenceBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"output\":[{},{}],\"output_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.hot_path_allocations,
        )
    }
}
