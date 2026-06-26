use crate::capabilities::snapshot::CapabilitySnapshot;
use crate::transport::path::planner::plan_transport_path;
use crate::transport::path::request::TransportPathRequest;
use crate::transport::path::types::TransferMode;
use crate::transport::probe::accumulator::TransportProbeAccumulator;
use crate::transport::probe::summary::TransportPathProbeSummary;
use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

pub fn run_transport_path_probe(
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<TransportPathProbeSummary> {
    let sizes = [
        (32 * 1024, TransferMode::Decode),
        (256 * 1024, TransferMode::Decode),
        (1024 * 1024, TransferMode::Decode),
        (16 * 1024 * 1024, TransferMode::Prefill),
        (64 * 1024 * 1024, TransferMode::Prefill),
        (256 * 1024 * 1024, TransferMode::Prefill),
    ];
    let mut probe = TransportProbeAccumulator::new();

    for (bytes, mode) in sizes {
        let request = TransportPathRequest::from_capabilities(
            MemoryTier::Vram,
            MemoryTier::Vram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            capabilities,
        );
        let decision = plan_transport_path(request)?;
        probe.record(decision);
    }

    let cpu_request = TransportPathRequest::from_capabilities(
        MemoryTier::Dram,
        MemoryTier::PinnedDram,
        32 * 1024,
        TransferMode::Decode,
        ExecutionOwner::Cpu,
        capabilities,
    );
    let cpu_decision = plan_transport_path(cpu_request)?;
    probe.record(cpu_decision);
    probe.ledger.require_zero_hot_path_allocations()?;
    probe.ledger.require_classified_syncs()?;

    Ok(probe.finish())
}
