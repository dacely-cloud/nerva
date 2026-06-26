use crate::capabilities::snapshot::CapabilitySnapshot;
use crate::transport::path::types::TransportPathClass;
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::plan::plan_stage_pipeline;
use crate::transport::stage::route::probe_stage_route_validation;
use crate::transport::stage::summary::{StagePipelineStatus, StagePipelineSummary};
use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

pub fn run_stage_pipeline_probe(
    config: StagePipelineConfig,
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<StagePipelineSummary> {
    let plan = plan_stage_pipeline(config, device, capabilities)?;
    let route_report = probe_stage_route_validation(&plan)?;
    let mut ledger = TokenLedger::new(0);
    for boundary in &plan.boundaries {
        boundary.decision.record_to_ledger(&mut ledger);
    }
    ledger.require_zero_hot_path_allocations()?;

    let mut gpu_direct_boundaries = 0u32;
    let mut host_staged_boundaries = 0u32;
    let mut cpu_produced_boundaries = 0u32;
    let mut mapped_pinned_boundaries = 0u32;
    let mut explicit_copy_bytes = 0usize;
    let mut nic_tx_bytes = 0usize;
    let mut nic_rx_bytes = 0usize;
    let mut pageable_copies = 0u64;
    let mut per_token_registrations = 0u64;

    for boundary in &plan.boundaries {
        match boundary.decision.class {
            TransportPathClass::GpuDirect => gpu_direct_boundaries += 1,
            TransportPathClass::HostStaged => host_staged_boundaries += 1,
            TransportPathClass::CpuProduced => cpu_produced_boundaries += 1,
            TransportPathClass::MappedPinned => mapped_pinned_boundaries += 1,
        }
        explicit_copy_bytes =
            explicit_copy_bytes.saturating_add(boundary.decision.explicit_copy_bytes);
        nic_tx_bytes = nic_tx_bytes.saturating_add(boundary.decision.nic_tx_bytes);
        nic_rx_bytes = nic_rx_bytes.saturating_add(boundary.decision.nic_rx_bytes);
        if boundary.decision.pageable_copy {
            pageable_copies = pageable_copies.saturating_add(1);
        }
        if boundary.decision.per_token_registration {
            per_token_registrations = per_token_registrations.saturating_add(1);
        }
    }

    Ok(StagePipelineSummary {
        status: StagePipelineStatus::Ok,
        stages: plan.config.stages,
        layers: plan
            .stages
            .iter()
            .map(|stage| stage.layer_count)
            .sum::<u32>(),
        boundaries: plan.boundaries.len() as u32,
        activation_bytes_per_boundary: plan.config.activation_bytes()?,
        total_activation_tx_bytes: plan
            .boundaries
            .iter()
            .map(|boundary| boundary.activation_bytes)
            .sum(),
        stage_local_weight_bytes: plan.stages.iter().map(|stage| stage.weight_bytes).sum(),
        stage_local_kv_bytes: plan.stages.iter().map(|stage| stage.kv_bytes).sum(),
        inter_stage_weight_bytes: plan
            .boundaries
            .iter()
            .map(|boundary| boundary.moved_weight_bytes)
            .sum(),
        all_reduce_bytes: plan
            .boundaries
            .iter()
            .map(|boundary| boundary.all_reduce_bytes)
            .sum(),
        activation_only_boundaries: plan
            .boundaries
            .iter()
            .filter(|boundary| boundary.moved_weight_bytes == 0 && boundary.all_reduce_bytes == 0)
            .count() as u32,
        gpu_direct_boundaries,
        host_staged_boundaries,
        cpu_produced_boundaries,
        mapped_pinned_boundaries,
        stage_route_validations: route_report.route_validations,
        stage_route_identity_checks: route_report.route_identity_checks,
        wrong_source_stage_rejections: route_report.wrong_source_stage_rejections,
        wrong_destination_stage_rejections: route_report.wrong_destination_stage_rejections,
        non_adjacent_route_rejections: route_report.non_adjacent_route_rejections,
        endpoint_identity_rejections: route_report.endpoint_identity_rejections,
        activation_size_rejections: route_report.activation_size_rejections,
        fallback_decisions: ledger.fallback_count(),
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        copy_events: ledger.event_count(LedgerEventKind::Copy),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        explicit_copy_bytes,
        nic_tx_bytes,
        nic_rx_bytes,
        pageable_copies,
        per_token_registrations,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
