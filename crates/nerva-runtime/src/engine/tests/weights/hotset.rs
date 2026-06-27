use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::memory::tier::MemoryTier;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::support::tiny_llama_manifest;
use crate::residency::budget::ResidencyBudget;

#[test]
fn resident_weight_hotset_promotion_moves_bounded_prefix_to_vram() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(256, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    let summary = runtime
        .promote_resident_weight_hotset(&mut table, 200)
        .unwrap();

    assert_eq!(summary.promoted_blocks, 7);
    assert_eq!(summary.promoted_bytes, 192);
    assert_eq!(summary.considered_blocks, 12);
    assert_eq!(summary.kept_dram_blocks, 5);
    assert_eq!(summary.budget_limited_blocks, 1);
    assert_eq!(summary.capacity_limited_blocks, 0);
    assert_eq!(summary.already_hot_blocks, 0);
    assert_eq!(summary.vram_used_bytes, 192);
    assert_eq!(summary.dram_used_bytes, 280);
    assert_eq!(summary.residency_decisions, 12);
    assert_eq!(
        summary.first_promoted_tensor.as_deref(),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(
        summary.last_promoted_tensor.as_deref(),
        Some("model.layers.0.post_attention_layernorm.weight")
    );
    assert_eq!(
        summary.last_keep_reason,
        Some("keep weight in DRAM outside bounded hotset prefix")
    );
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(table.entries[0].tier, MemoryTier::Vram);
    assert_eq!(table.entries[6].tier, MemoryTier::Vram);
    assert_eq!(table.entries[7].tier, MemoryTier::Dram);
    assert!(table.entries[..7].iter().all(|entry| {
        table
            .registry
            .block(entry.block_id)
            .is_some_and(|block| block.state == ResidencyState::Ready)
    }));
    assert!(summary.to_json().contains("\"promoted_blocks\":7"));
    assert!(summary.to_json().contains("\"budget_limited_blocks\":1"));
    assert!(summary.to_json().contains("\"kept_dram_blocks\":5"));
}

#[test]
fn resident_weight_hotset_promotion_respects_vram_capacity_and_zero_limit() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(100, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    let zero = runtime
        .promote_resident_weight_hotset(&mut table, 0)
        .unwrap();
    assert_eq!(zero.considered_blocks, 12);
    assert_eq!(zero.promoted_blocks, 0);
    assert_eq!(zero.kept_dram_blocks, 12);
    assert_eq!(zero.budget_limited_blocks, 1);
    assert_eq!(zero.capacity_limited_blocks, 0);
    assert_eq!(zero.residency_decisions, 12);
    assert_eq!(zero.vram_used_bytes, 0);

    let summary = runtime
        .promote_resident_weight_hotset(&mut table, usize::MAX)
        .unwrap();
    assert_eq!(summary.considered_blocks, 12);
    assert_eq!(summary.promoted_blocks, 2);
    assert_eq!(summary.promoted_bytes, 88);
    assert_eq!(summary.kept_dram_blocks, 10);
    assert_eq!(summary.budget_limited_blocks, 0);
    assert_eq!(summary.capacity_limited_blocks, 1);
    assert_eq!(summary.residency_decisions, 12);
    assert_eq!(summary.vram_used_bytes, 88);
    assert_eq!(summary.dram_used_bytes, 384);
}
