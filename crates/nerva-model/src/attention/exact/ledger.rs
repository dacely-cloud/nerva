use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::attention::block::KvAttentionBlock;
use crate::common::shape::TransformerBlockShape;

pub(crate) fn record_attention_block_event(
    shape: TransformerBlockShape,
    block: &KvAttentionBlock<'_>,
    ledger: &mut TokenLedger,
) {
    let (kind, executor_selected, reason) = match block.tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => (
            LedgerEventKind::DeviceActivity,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            "hot KV block is already device resident",
        ),
        MemoryTier::PinnedDram | MemoryTier::Dram | MemoryTier::Cxl | MemoryTier::Disk => (
            LedgerEventKind::CpuActivity,
            ExecutionOwner::Cpu,
            "warm KV block is cheaper to compute near than stage",
        ),
    };
    let label = match kind {
        LedgerEventKind::DeviceActivity => "attention_hot_kv_block",
        LedgerEventKind::CpuActivity => "attention_warm_kv_block",
        _ => "attention_kv_block",
    };
    let latency_ns = block.token_count as u64;
    ledger.record_execution_decision(ExecutionDecision {
        operation: "blockwise_attention",
        executor_selected,
        candidate_costs: vec![
            CandidateCost::estimated("compute-near-current-tier", latency_ns),
            CandidateCost::estimated("stage-to-gpu", latency_ns + 2),
        ],
        reason,
        predicted_visible_ns: latency_ns,
        actual_visible_ns: Some(latency_ns),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record(LedgerEvent {
        kind,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(block.tier),
        to_tier: Some(block.tier),
        bytes: block.token_count * shape.hidden * core::mem::size_of::<f32>() * 2,
        latency_ns,
        label,
    });
}
