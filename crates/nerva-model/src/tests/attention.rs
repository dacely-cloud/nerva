use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::run::exact_blockwise_attention_into;
use crate::attention::scratch::BlockwiseAttentionScratch;
use crate::attention::smoke::{BlockwiseAttentionSmokeStatus, blockwise_attention_smoke};
use crate::common::shape::TransformerBlockShape;
use crate::tests::support::dense_attention_reference;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

#[test]
fn blockwise_attention_matches_dense_reference_across_tiers() {
    let shape = TransformerBlockShape::new(4, 2, 4);
    let query = [0.5, -1.0, 0.25, 0.75];
    let keys = [0.1, 0.2, 0.3, 0.4, 0.0, -0.5, 0.6, 0.2, 0.7, 0.1, -0.2, 0.3];
    let values = [
        1.0, 0.0, 0.5, -0.5, -1.0, 2.0, 0.25, 0.75, 0.3, -0.8, 1.5, 0.2,
    ];
    let blocks = [
        KvAttentionBlock::new(&keys[..4], &values[..4], 1, MemoryTier::Dram),
        KvAttentionBlock::new(&keys[4..], &values[4..], 2, MemoryTier::Vram),
    ];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 4];
    let mut ledger = TokenLedger::new(11);

    exact_blockwise_attention_into(
        shape,
        &query,
        &blocks,
        &mut scratch,
        &mut output,
        &mut ledger,
    )
    .unwrap();

    let expected = dense_attention_reference(shape, &query, &keys, &values, 3);
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(ledger.event_count(LedgerEventKind::CpuActivity), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
    assert_eq!(ledger.total_latency_ns(), 3);
    assert_eq!(ledger.hot_path_allocations, 0);
}

#[test]
fn blockwise_attention_rejects_empty_and_malformed_blocks() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let query = [1.0, 0.0];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);

    assert!(
        exact_blockwise_attention_into(shape, &query, &[], &mut scratch, &mut output, &mut ledger)
            .is_err()
    );

    let bad_block = [KvAttentionBlock::new(
        &[1.0],
        &[1.0, 0.0],
        1,
        MemoryTier::Dram,
    )];
    assert!(
        exact_blockwise_attention_into(
            shape,
            &query,
            &bad_block,
            &mut scratch,
            &mut output,
            &mut ledger,
        )
        .is_err()
    );
}

#[test]
fn blockwise_attention_smoke_reports_tier_events() {
    let summary = blockwise_attention_smoke().unwrap();
    assert_eq!(summary.status, BlockwiseAttentionSmokeStatus::Ok);
    assert_eq!(summary.hidden, 2);
    assert_eq!(summary.heads, 1);
    assert_eq!(summary.blocks, 2);
    assert_eq!(summary.tokens, 4);
    assert_eq!(summary.cpu_block_events, 1);
    assert_eq!(summary.device_block_events, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"device_block_events\":1"));
}
