use crate::weights::layout::plan::HfWeightLayoutPlan;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfWeightLayoutProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfWeightLayoutProbeSummary {
    pub status: HfWeightLayoutProbeStatus,
    pub plan: HfWeightLayoutPlan,
    pub layout_hash: u64,
}

impl HfWeightLayoutProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfWeightLayoutProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"plan\":{},\"layout_hash\":{}}}",
            status,
            self.plan.to_json(),
            self.layout_hash,
        )
    }
}
