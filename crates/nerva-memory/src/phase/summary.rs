#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PhaseHandoffProbeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffProbeSummary {
    pub status: PhaseHandoffProbeStatus,
    pub planned_handoffs: u64,
    pub applied_handoffs: u64,
    pub rejected_handoffs: u64,
    pub owner_mismatch_rejections: u64,
    pub stale_version_rejections: u64,
    pub unready_rejections: u64,
    pub illegal_transition_rejections: u64,
    pub phase_handoff_syncs: u64,
    pub version_publications: u64,
    pub final_max_version: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl PhaseHandoffProbeSummary {
    pub fn passed(self) -> bool {
        matches!(self.status, PhaseHandoffProbeStatus::Ok)
            && self.planned_handoffs > 0
            && self.applied_handoffs == self.planned_handoffs
            && self.rejected_handoffs >= 4
            && self.owner_mismatch_rejections > 0
            && self.stale_version_rejections > 0
            && self.unready_rejections > 0
            && self.illegal_transition_rejections > 0
            && self.phase_handoff_syncs == self.applied_handoffs
            && self.version_publications == self.applied_handoffs
            && self.final_max_version > 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            PhaseHandoffProbeStatus::Ok => "ok",
            PhaseHandoffProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"planned_handoffs\":{},\"applied_handoffs\":{},\"rejected_handoffs\":{},\"owner_mismatch_rejections\":{},\"stale_version_rejections\":{},\"unready_rejections\":{},\"illegal_transition_rejections\":{},\"phase_handoff_syncs\":{},\"version_publications\":{},\"final_max_version\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.planned_handoffs,
            self.applied_handoffs,
            self.rejected_handoffs,
            self.owner_mismatch_rejections,
            self.stale_version_rejections,
            self.unready_rejections,
            self.illegal_transition_rejections,
            self.phase_handoff_syncs,
            self.version_publications,
            self.final_max_version,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
