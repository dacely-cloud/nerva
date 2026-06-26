use nerva_core::{CostSource, ExecutionOwner, MemoryTier, ResidentBlockId};

use crate::{
    BlockVersionDependency, CandidateCost, DeviceTimelineSpan, ExecutionDecision, FallbackDecision,
    LedgerEvent, ResidencyDecision, SyncClass, TokenLedger,
};

impl TokenLedger {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"token_index\":{},\"events\":{},\"device_timeline\":{},\"fallback_decisions\":{},\"block_version_dependencies\":{},\"residency_decisions\":{},\"execution_decisions\":{},\"hot_path_allocations\":{}}}",
            self.token_index,
            json_array(&self.events, ledger_event_json),
            json_array(&self.device_timeline, device_timeline_span_json),
            json_array(&self.fallback_decisions, fallback_decision_json),
            json_array(
                &self.block_version_dependencies,
                block_version_dependency_json
            ),
            json_array(&self.residency_decisions, residency_decision_json),
            json_array(&self.execution_decisions, execution_decision_json),
            self.hot_path_allocations,
        )
    }
}

fn ledger_event_json(event: &LedgerEvent) -> String {
    format!(
        "{{\"kind\":\"{}\",\"sync_class\":{},\"metric_source\":\"{}\",\"block_id\":{},\"from_tier\":{},\"to_tier\":{},\"bytes\":{},\"latency_ns\":{},\"label\":\"{}\"}}",
        event.kind.as_str(),
        json_opt_sync_class(event.sync_class),
        event.metric_source.as_str(),
        json_opt_block_id(event.block_id),
        json_opt_tier(event.from_tier),
        json_opt_tier(event.to_tier),
        event.bytes,
        event.latency_ns,
        json_escape(event.label),
    )
}

fn device_timeline_span_json(span: &DeviceTimelineSpan) -> String {
    format!(
        "{{\"device\":{},\"start_ns\":{},\"end_ns\":{},\"metric_source\":\"{}\",\"label\":\"{}\"}}",
        span.device.0,
        span.start_ns,
        span.end_ns,
        span.metric_source.as_str(),
        json_escape(span.label),
    )
}

fn fallback_decision_json(decision: &FallbackDecision) -> String {
    format!(
        "{{\"label\":\"{}\",\"class\":\"{}\",\"requested\":\"{}\",\"selected\":\"{}\",\"reason\":\"{}\",\"visible_ns\":{},\"metric_source\":\"{}\"}}",
        json_escape(decision.label),
        decision.class.as_str(),
        json_escape(decision.requested),
        json_escape(decision.selected),
        json_escape(decision.reason),
        json_opt_u64(decision.visible_ns),
        decision.metric_source.as_str(),
    )
}

fn block_version_dependency_json(dependency: &BlockVersionDependency) -> String {
    format!(
        "{{\"block_id\":{},\"required_version\":{},\"observed_version\":{},\"label\":\"{}\"}}",
        dependency.block_id.0,
        dependency.required_version,
        dependency.observed_version,
        json_escape(dependency.label),
    )
}

fn residency_decision_json(decision: &ResidencyDecision) -> String {
    format!(
        "{{\"block_id\":{},\"old_tier\":\"{}\",\"new_tier\":\"{}\",\"executor_selected\":\"{}\",\"candidate_costs\":{},\"reason\":\"{}\",\"predicted_overlap_ns\":{},\"actual_visible_ns\":{},\"metric_source\":\"{}\"}}",
        decision.block_id.0,
        memory_tier_str(decision.old_tier),
        memory_tier_str(decision.new_tier),
        execution_owner_json_value(decision.executor_selected),
        json_array(&decision.candidate_costs, candidate_cost_json),
        json_escape(decision.reason),
        decision.predicted_overlap_ns,
        json_opt_u64(decision.actual_visible_ns),
        decision.metric_source.as_str(),
    )
}

fn execution_decision_json(decision: &ExecutionDecision) -> String {
    format!(
        "{{\"operation\":\"{}\",\"executor_selected\":\"{}\",\"candidate_costs\":{},\"reason\":\"{}\",\"predicted_visible_ns\":{},\"actual_visible_ns\":{},\"metric_source\":\"{}\"}}",
        json_escape(decision.operation),
        execution_owner_json_value(decision.executor_selected),
        json_array(&decision.candidate_costs, candidate_cost_json),
        json_escape(decision.reason),
        decision.predicted_visible_ns,
        json_opt_u64(decision.actual_visible_ns),
        decision.metric_source.as_str(),
    )
}

fn candidate_cost_json(cost: &CandidateCost) -> String {
    format!(
        "{{\"label\":\"{}\",\"visible_ns\":{},\"source\":\"{}\"}}",
        json_escape(cost.label),
        json_opt_u64(cost.visible_ns),
        cost_source_str(cost.source),
    )
}

fn json_array<T>(values: &[T], encode: fn(&T) -> String) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&encode(value));
    }
    out.push(']');
    out
}

fn json_opt_sync_class(value: Option<SyncClass>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", value.as_str()),
    )
}

fn json_opt_block_id(value: Option<ResidentBlockId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
}

fn json_opt_tier(value: Option<MemoryTier>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", memory_tier_str(value)),
    )
}

fn json_opt_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn memory_tier_str(tier: MemoryTier) -> &'static str {
    match tier {
        MemoryTier::Vram => "vram",
        MemoryTier::SharedHbmOrLpddr => "shared_hbm_or_lpddr",
        MemoryTier::PinnedDram => "pinned_dram",
        MemoryTier::Dram => "dram",
        MemoryTier::Cxl => "cxl",
        MemoryTier::Disk => "disk",
    }
}

fn execution_owner_json_value(owner: ExecutionOwner) -> String {
    match owner {
        ExecutionOwner::Cpu => "cpu".to_string(),
        ExecutionOwner::Gpu(device) => format!("gpu:{}", device.0),
        ExecutionOwner::Nic(device) => format!("nic:{}", device.0),
        ExecutionOwner::SharedReadOnly => "shared_read_only".to_string(),
        ExecutionOwner::PhaseTransition => "phase_transition".to_string(),
        ExecutionOwner::None => "none".to_string(),
    }
}

fn cost_source_str(source: CostSource) -> &'static str {
    match source {
        CostSource::Unknown => "unknown",
        CostSource::Estimated => "estimated",
        CostSource::Measured => "measured",
    }
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}
