use nerva_core::types::error::Result;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::mgpu::config::MultiGpuNodeConfig;
use crate::mgpu::plan::{GpuIslandRole, plan_multi_gpu_node};
use crate::mgpu::summary::{MultiGpuNodeStatus, MultiGpuNodeSummary};

pub fn run_multi_gpu_node_probe(config: MultiGpuNodeConfig) -> Result<MultiGpuNodeSummary> {
    let plan = plan_multi_gpu_node(config)?;
    let mut ledger = TokenLedger::new(0);

    for island in &plan.islands {
        ledger.record_execution_decision(ExecutionDecision {
            operation: "same_node_stage_layer_range",
            executor_selected: island.owner,
            candidate_costs: vec![
                CandidateCost::estimated("gpu-local-hot-cache", island.layer_count as u64),
                CandidateCost::estimated(
                    "aggregate-vram-pool-illegal",
                    u64::try_from(config.aggregate_vram_bytes()?).unwrap_or(u64::MAX),
                ),
                CandidateCost::estimated(
                    "dram-warm-backing",
                    u64::try_from(island.dram_weight_backing_bytes).unwrap_or(u64::MAX),
                ),
            ],
            reason: "keep GPU memory islands separate and execute layer range against local hot cache",
            predicted_visible_ns: island.layer_count as u64,
            actual_visible_ns: Some(island.layer_count as u64),
            metric_source: MetricSource::EstimatedModel,
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::DeviceActivity,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::Vram),
            bytes: island
                .hot_weight_bytes
                .saturating_add(island.kv_bytes)
                .saturating_add(config.activation_bytes()?),
            latency_ns: island.layer_count as u64,
            label: island.role.label(),
        });
    }

    for boundary in &plan.boundaries {
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::Vram),
            bytes: boundary.activation_bytes,
            latency_ns: boundary.activation_bytes as u64,
            label: "same_node_activation_boundary",
        });
        if boundary.phase_handoff_required {
            ledger.record_sync(
                SyncClass::PhaseHandoff,
                None,
                Some(MemoryTier::Vram),
                Some(MemoryTier::Vram),
                boundary.activation_bytes,
                1,
                MetricSource::RuntimeTimestamp,
                "same_node_activation_owner_handoff",
            );
        }
    }

    ledger.require_zero_hot_path_allocations()?;
    ledger.require_classified_syncs()?;

    let egress_gpu = plan
        .islands
        .iter()
        .find(|island| island.role == GpuIslandRole::EgressCompute)
        .map_or(-1, |island| island.gpu.0);
    let hot_weight_cache_bytes = plan
        .islands
        .iter()
        .map(|island| island.hot_weight_bytes)
        .sum();
    let dram_weight_backing_bytes = plan
        .islands
        .iter()
        .map(|island| island.dram_weight_backing_bytes)
        .sum();

    Ok(MultiGpuNodeSummary {
        status: MultiGpuNodeStatus::Ok,
        gpu_count: plan.config.gpu_count,
        gpu_islands: plan.islands.len() as u32,
        compute_gpu_count: plan
            .islands
            .iter()
            .filter(|island| {
                matches!(
                    island.role,
                    GpuIslandRole::Compute | GpuIslandRole::EgressCompute
                )
            })
            .count() as u32,
        egress_gpu,
        nic_near_egress: egress_gpu == plan.config.nic_near_gpu as i32,
        local_vram_bytes_per_gpu: plan.config.local_vram_bytes_per_gpu,
        aggregate_vram_bytes: plan.config.aggregate_vram_bytes()?,
        aggregate_vram_pool_claimed: plan.aggregate_vram_pool_claimed,
        coherent_vram_allocation_claims: plan.coherent_vram_allocation_claims,
        max_single_allocation_bytes: plan
            .islands
            .iter()
            .map(|island| island.max_single_allocation_bytes)
            .max()
            .unwrap_or(0),
        stage_layers: plan.config.layers,
        stage_weight_bytes: plan.config.stage_weight_bytes,
        hot_weight_cache_bytes,
        dram_weight_backing_bytes,
        stage_kv_bytes: plan.config.stage_kv_bytes,
        kv_owner_count: plan
            .islands
            .iter()
            .filter(|island| island.kv_bytes > 0)
            .count() as u32,
        activation_bytes_per_boundary: plan.config.activation_bytes()?,
        local_boundaries: plan.boundaries.len() as u32,
        activation_only_boundaries: plan
            .boundaries
            .iter()
            .filter(|boundary| boundary.moved_weight_bytes == 0 && boundary.all_reduce_bytes == 0)
            .count() as u32,
        activation_bytes_moved: plan
            .boundaries
            .iter()
            .map(|boundary| boundary.activation_bytes)
            .sum(),
        inter_gpu_weight_bytes: plan
            .boundaries
            .iter()
            .map(|boundary| boundary.moved_weight_bytes)
            .sum(),
        all_reduce_bytes: plan
            .boundaries
            .iter()
            .map(|boundary| boundary.all_reduce_bytes)
            .sum(),
        execution_decisions: ledger.execution_decisions.len() as u64,
        device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        copy_events: ledger.event_count(LedgerEventKind::Copy),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        pageable_copies: 0,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
