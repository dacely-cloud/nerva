use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::controller::RequestController;
use crate::request::probe::run_request_state_probe;
use crate::request::types::{RequestPhase, StopReason};

#[test]
fn request_controller_keeps_host_observation_replicated() {
    let mut controller =
        RequestController::new(RequestId(1), SequenceId(1), vec![TokenId(0)], 3, None).unwrap();

    assert_eq!(controller.begin_decode().unwrap(), TokenId(0));
    controller.record_device_token(0, TokenId(1)).unwrap();
    assert_eq!(controller.next_device_input().unwrap(), TokenId(1));
    assert_eq!(controller.host_visibility_lag(), 1);

    let batch = controller.observe_host_tokens(1);
    assert_eq!(batch.start_index, 0);
    assert_eq!(batch.tokens, vec![TokenId(1)]);
    assert_eq!(controller.host_visibility_lag(), 0);
}

#[test]
fn request_controller_rejects_duplicate_missing_and_completed_rows() {
    let mut controller = RequestController::new(
        RequestId(2),
        SequenceId(2),
        vec![TokenId(0)],
        2,
        Some(TokenId(1)),
    )
    .unwrap();
    controller.begin_decode().unwrap();

    assert!(controller.record_device_token(1, TokenId(1)).is_err());
    controller.record_device_token(0, TokenId(1)).unwrap();
    assert_eq!(controller.phase, RequestPhase::Completed);
    assert_eq!(controller.stop_reason, Some(StopReason::EosToken));
    assert!(controller.record_device_token(1, TokenId(2)).is_err());
}

#[test]
fn request_state_probe_reports_device_causality_without_host_dependency() {
    let summary = run_request_state_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.prompt_tokens, vec![TokenId(0), TokenId(1)]);
    assert_eq!(
        summary.generated_tokens,
        vec![TokenId(2), TokenId(3), TokenId(0)]
    );
    assert_eq!(summary.generated_tokens, summary.host_observed_tokens);
    assert_eq!(summary.stop_reason, StopReason::EosToken);
    assert!(summary.max_host_visibility_lag >= 2);
    assert!(summary.to_json().contains("\"stop_reason\":\"eos_token\""));
}
