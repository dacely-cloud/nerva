use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::decision::BlockVersionDependency;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::engine::compute_near_data::decisions::{
    record_cpu_shard_decision, record_gpu_shard_decision,
};
use crate::engine::compute_near_data::math::mat_vec_row_range;
use crate::engine::compute_near_data::shard::ResidentMatvecShard;

pub(crate) fn execute_resident_split_matvec(
    registry: &BlockRegistry,
    device: DeviceOrdinal,
    cols: usize,
    input: &[f32],
    shards: &[ResidentMatvecShard<'_>],
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    for shard in shards {
        execute_shard(registry, device, cols, input, shard, output, ledger)?;
    }
    ledger.require_satisfied_block_versions()
}

fn execute_shard(
    registry: &BlockRegistry,
    device: DeviceOrdinal,
    cols: usize,
    input: &[f32],
    shard: &ResidentMatvecShard<'_>,
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    record_ready_dependency(registry, shard, ledger)?;

    match shard.tier {
        MemoryTier::Dram => execute_dram_shard(cols, input, shard, output, ledger),
        MemoryTier::Vram => execute_vram_shard(device, cols, input, shard, output, ledger),
        _ => Err(NervaError::InvalidArgument {
            reason: "compute-near-data probe only supports DRAM and VRAM shards".to_string(),
        }),
    }
}

fn record_ready_dependency(
    registry: &BlockRegistry,
    shard: &ResidentMatvecShard<'_>,
    ledger: &mut TokenLedger,
) -> Result<()> {
    let block = registry
        .block(shard.block_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "compute-near-data shard references missing block {}",
                shard.block_id.0
            ),
        })?;
    if block.state != ResidencyState::Ready {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "compute-near-data shard block {} is not Ready",
                shard.block_id.0
            ),
        });
    }
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: shard.block_id,
        required_version: block.version,
        observed_version: block.version,
        label: "compute_near_data_matvec",
    });
    Ok(())
}

fn execute_dram_shard(
    cols: usize,
    input: &[f32],
    shard: &ResidentMatvecShard<'_>,
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    record_cpu_shard_decision(shard, ledger);
    mat_vec_row_range(
        shard.weights,
        input,
        cols,
        shard.row_start,
        shard.row_end,
        shard.row_start,
        output,
    )?;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(shard.block_id),
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: shard.weights.len() * core::mem::size_of::<f32>(),
        latency_ns: shard.weights.len() as u64,
        label: "compute_near_data_cpu_shard",
    });
    Ok(())
}

fn execute_vram_shard(
    device: DeviceOrdinal,
    cols: usize,
    input: &[f32],
    shard: &ResidentMatvecShard<'_>,
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    record_gpu_shard_decision(device, shard, ledger);
    mat_vec_row_range(
        shard.weights,
        input,
        cols,
        shard.row_start,
        shard.row_end,
        shard.row_start,
        output,
    )?;
    let merge_bytes = (shard.row_end - shard.row_start) * core::mem::size_of::<f32>();
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(shard.block_id),
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: shard.weights.len() * core::mem::size_of::<f32>(),
        latency_ns: (shard.row_end - shard.row_start) as u64,
        label: "compute_near_data_gpu_shard",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(shard.block_id),
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Dram),
        bytes: merge_bytes,
        latency_ns: merge_bytes as u64,
        label: "compute_near_data_merge_gpu_rows",
    });
    Ok(())
}
