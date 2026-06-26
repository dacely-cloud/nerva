use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::error::NervaError;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::layout::LayoutId;

use nerva_core::types::memory::tier::MemoryTier;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::support::{SHARD_ONE, tiny_shard_plan};
use crate::residency::budget::ResidencyBudget;

#[test]
fn materializes_hf_weight_manifest_as_dram_resident_blocks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();

    assert_eq!(table.entries.len(), manifest.entries.len());
    assert_eq!(table.total_weight_bytes, manifest.total_weight_bytes);
    assert_eq!(
        table.registry.used_bytes(MemoryTier::Dram),
        manifest.total_weight_bytes
    );
    assert_eq!(table.registry.used_bytes(MemoryTier::Vram), 0);
    assert_eq!(table.ledger.hot_path_allocations, 0);
    assert_eq!(
        table.ledger.residency_decisions.len(),
        manifest.entries.len()
    );

    let first = table.entries.first().unwrap();
    let block = table.registry.block(first.block_id).unwrap();
    assert_eq!(first.name, "model.embed_tokens.weight");
    assert_eq!(block.kind, BlockKind::Weight);
    assert_eq!(
        block.semantics,
        nerva_core::types::ownership::mutation::MutationSemantics::Immutable
    );
    assert_eq!(block.tier, MemoryTier::Dram);
    assert_eq!(block.dtype, first.dtype);
    assert_eq!(block.layout, LayoutId(1));
}

#[test]
fn materialized_weight_manifest_preserves_last_block_and_decision() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();

    let last = table.entries.last().unwrap();
    let decision = table.ledger.residency_decisions.last().unwrap();
    assert_eq!(last.name, "lm_head.weight");
    assert_eq!(last.block_id, ResidentBlockId(290));
    assert_eq!(decision.block_id, last.block_id);
    assert_eq!(decision.old_tier, MemoryTier::Disk);
    assert_eq!(decision.new_tier, MemoryTier::Dram);
}

#[test]
fn materialized_weight_manifest_respects_dram_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let err = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(0, 0, manifest.total_weight_bytes - 1),
        )
        .unwrap_err();

    assert!(matches!(err, NervaError::AllocationFailed { .. }));
}

#[test]
fn materializes_safetensors_shard_plan_with_source_offsets() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header_len) = tiny_shard_plan();
    let table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();

    assert_eq!(table.entries.len(), plan.entries.len());
    assert_eq!(table.total_weight_bytes, plan.total_weight_bytes);
    assert_eq!(table.registry.used_bytes(MemoryTier::Dram), 464);
    assert_eq!(table.registry.used_bytes(MemoryTier::Vram), 0);
    assert_eq!(table.ledger.hot_path_allocations, 0);
    assert_eq!(table.ledger.residency_decisions.len(), plan.entries.len());

    let first = table.entries.first().unwrap();
    assert_eq!(first.name, "model.embed_tokens.weight");
    assert_eq!(first.source_shard.as_deref(), Some(SHARD_ONE));
    assert_eq!(first.file_offset_begin, Some(8 + header_len));
    assert_eq!(first.file_offset_end, Some(8 + header_len + first.bytes));
    assert_eq!(first.tier, MemoryTier::Dram);

    let block = table.registry.block(first.block_id).unwrap();
    assert_eq!(block.kind, BlockKind::Weight);
    assert_eq!(block.layout, LayoutId(1));
    assert_eq!(block.dtype, first.dtype);
}

#[test]
fn materialized_safetensors_shard_plan_respects_dram_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, _) = tiny_shard_plan();
    let err = runtime
        .materialize_safetensors_shard_plan_with_budget(
            &plan,
            ResidencyBudget::new(0, 0, plan.total_weight_bytes - 1),
        )
        .unwrap_err();

    assert!(matches!(err, NervaError::AllocationFailed { .. }));
}
