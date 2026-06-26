use crate::common::shape::TransformerBlockShape;
use crate::tiny::output::TinyGreedyDecodeStatus;
use crate::tiny::precision::output::TinyPrecisionGreedyDecodeStatus;
use crate::tiny::precision::scratch::TinyPrecisionGreedyDecodeScratch;
use crate::tiny::precision::smoke::{
    tiny_precision_cycle_model, tiny_precision_greedy_decode_smoke,
};
use crate::tiny::scratch::TinyGreedyDecodeScratch;
use crate::tiny::smoke::{tiny_cycle_model, tiny_greedy_decode_smoke};
use nerva_core::types::dtype::DType;
use nerva_core::types::id::TokenId;
use nerva_ledger::types::event::LedgerEventKind;

#[test]
fn tiny_greedy_model_matches_expected_token_cycle() {
    let model = tiny_cycle_model().unwrap();
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();
    let output = model.decode_greedy(TokenId(0), 8, &mut scratch).unwrap();

    assert_eq!(
        output.tokens,
        vec![
            TokenId(1),
            TokenId(2),
            TokenId(3),
            TokenId(0),
            TokenId(1),
            TokenId(2),
            TokenId(3),
            TokenId(0),
        ]
    );
    assert_eq!(output.ledgers.len(), 8);
    assert_eq!(
        output
            .ledgers
            .iter()
            .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
            .sum::<u64>(),
        8
    );
    assert_eq!(
        output
            .ledgers
            .iter()
            .map(|ledger| ledger.hot_path_allocations)
            .sum::<u64>(),
        0
    );
}

#[test]
fn tiny_greedy_model_rejects_bad_decode_inputs() {
    let model = tiny_cycle_model().unwrap();
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();

    assert!(model.decode_greedy(TokenId(0), 0, &mut scratch).is_err());
    assert!(model.decode_greedy(TokenId(99), 1, &mut scratch).is_err());

    let mut wrong_scratch =
        TinyGreedyDecodeScratch::new(TransformerBlockShape::new(4, 2, 4), model.vocab_size())
            .unwrap();
    assert!(
        model
            .decode_greedy(TokenId(0), 1, &mut wrong_scratch)
            .is_err()
    );
}

#[test]
fn tiny_greedy_decode_smoke_reports_parity_and_ledger() {
    let summary = tiny_greedy_decode_smoke(8).unwrap();

    assert_eq!(summary.status, TinyGreedyDecodeStatus::Ok);
    assert_eq!(summary.seed_token, TokenId(0));
    assert_eq!(summary.steps, 8);
    assert_eq!(summary.vocab_size, 4);
    assert!(summary.parity);
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_eq!(summary.ledger_count, 8);
    assert_eq!(summary.device_events, 8);
    assert_eq!(summary.total_latency_ns, 8);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"parity\":true"));
}

#[test]
fn tiny_precision_greedy_model_matches_expected_token_cycle() {
    for dtype in [DType::F16, DType::BF16] {
        let model = tiny_precision_cycle_model(dtype).unwrap();
        let mut scratch =
            TinyPrecisionGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();
        let output = model.decode_greedy(TokenId(0), 8, &mut scratch).unwrap();

        assert_eq!(
            output.tokens,
            vec![
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(0),
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(0),
            ]
        );
        assert_eq!(output.ledgers.len(), 8);
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.event_count(LedgerEventKind::CpuActivity))
                .sum::<u64>(),
            8
        );
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.execution_decisions.len() as u64)
                .sum::<u64>(),
            8
        );
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.hot_path_allocations)
                .sum::<u64>(),
            0
        );
    }
}

#[test]
fn tiny_precision_greedy_model_rejects_bad_decode_inputs() {
    let model = tiny_precision_cycle_model(DType::F16).unwrap();
    let mut scratch =
        TinyPrecisionGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();

    assert!(model.decode_greedy(TokenId(0), 0, &mut scratch).is_err());
    assert!(model.decode_greedy(TokenId(99), 1, &mut scratch).is_err());

    let mut wrong_scratch = TinyPrecisionGreedyDecodeScratch::new(
        TransformerBlockShape::new(4, 2, 4),
        model.vocab_size(),
    )
    .unwrap();
    assert!(
        model
            .decode_greedy(TokenId(0), 1, &mut wrong_scratch)
            .is_err()
    );
}

#[test]
fn tiny_precision_greedy_decode_smoke_reports_parity_and_ledger() {
    for dtype in [DType::F16, DType::BF16] {
        let summary = tiny_precision_greedy_decode_smoke(dtype, 8).unwrap();

        assert_eq!(summary.status, TinyPrecisionGreedyDecodeStatus::Ok);
        assert_eq!(summary.seed_token, TokenId(0));
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.vocab_size, 4);
        assert!(summary.parity);
        assert_eq!(summary.tokens, summary.expected_tokens);
        assert_eq!(summary.ledger_count, 8);
        assert_eq!(summary.cpu_events, 8);
        assert_eq!(summary.execution_decisions, 8);
        assert_eq!(summary.total_latency_ns, 8);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.to_json().contains("\"parity\":true"));
    }
}
