use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::scheduler::admission::RequestAdmission;
use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::scheduler::probe::run_request_scheduler_probe;
use crate::request::scheduler::selection::SchedulerSelectionOutcome;

#[test]
fn bounded_scheduler_rejects_full_and_duplicate_admission() {
    let mut scheduler = BoundedRequestScheduler::new(1).unwrap();
    let admission = RequestAdmission {
        request_id: RequestId(7),
        sequence_id: SequenceId(9),
        prompt_tokens: vec![TokenId(0)],
        max_new_tokens: 2,
        eos_token: None,
    };

    assert_eq!(scheduler.admit(admission.clone()).unwrap(), 0);
    assert!(scheduler.admit(admission).is_err());
    assert!(
        scheduler
            .admit(RequestAdmission {
                request_id: RequestId(8),
                sequence_id: SequenceId(10),
                prompt_tokens: vec![TokenId(1)],
                max_new_tokens: 1,
                eos_token: None,
            })
            .is_err()
    );
}

#[test]
fn bounded_scheduler_reuses_slot_only_after_completion_and_host_drain() {
    let mut scheduler = BoundedRequestScheduler::new(1).unwrap();
    let admission = RequestAdmission {
        request_id: RequestId(10),
        sequence_id: SequenceId(10),
        prompt_tokens: vec![TokenId(0)],
        max_new_tokens: 1,
        eos_token: None,
    };

    assert_eq!(scheduler.admit(admission).unwrap(), 0);
    scheduler.begin_decode(RequestId(10)).unwrap();
    assert!(scheduler.release_completed(RequestId(10)).is_err());
    scheduler
        .record_device_token(RequestId(10), 0, TokenId(1))
        .unwrap();
    assert!(scheduler.release_completed(RequestId(10)).is_err());
    scheduler
        .observe_host_tokens(RequestId(10), usize::MAX)
        .unwrap();
    assert_eq!(scheduler.release_completed(RequestId(10)).unwrap(), 0);
    assert!(scheduler.controller(RequestId(10)).is_err());
    assert_eq!(
        scheduler
            .admit(RequestAdmission {
                request_id: RequestId(11),
                sequence_id: SequenceId(11),
                prompt_tokens: vec![TokenId(1)],
                max_new_tokens: 1,
                eos_token: None,
            })
            .unwrap(),
        0
    );
}

#[test]
fn bounded_scheduler_rotates_decoding_selection() {
    let mut scheduler = BoundedRequestScheduler::new(2).unwrap();
    scheduler
        .admit(RequestAdmission {
            request_id: RequestId(20),
            sequence_id: SequenceId(20),
            prompt_tokens: vec![TokenId(0)],
            max_new_tokens: 2,
            eos_token: None,
        })
        .unwrap();
    scheduler
        .admit(RequestAdmission {
            request_id: RequestId(21),
            sequence_id: SequenceId(21),
            prompt_tokens: vec![TokenId(1)],
            max_new_tokens: 2,
            eos_token: None,
        })
        .unwrap();

    scheduler.begin_decode(RequestId(20)).unwrap();
    scheduler.begin_decode(RequestId(21)).unwrap();
    let first = ready_selection(scheduler.select_next_decoding());
    let second = ready_selection(scheduler.select_next_decoding());

    assert_eq!(first.request_id, RequestId(20));
    assert_eq!(first.slot, 0);
    assert_eq!(second.request_id, RequestId(21));
    assert_eq!(second.slot, 1);
    assert_eq!(first.scanned_slots, 1);
    assert_eq!(second.scanned_slots, 1);
}

#[test]
fn bounded_scheduler_reports_no_ready_scan() {
    let mut scheduler = BoundedRequestScheduler::new(2).unwrap();
    let outcome = scheduler.select_next_decoding();

    match outcome {
        SchedulerSelectionOutcome::NoReady(miss) => {
            assert_eq!(miss.scanned_slots, 2);
            assert_eq!(miss.skipped_slots, 2);
            assert!(!miss.wrapped);
        }
        SchedulerSelectionOutcome::Ready(_) => panic!("empty scheduler cannot be ready"),
    }
}

#[test]
fn request_scheduler_probe_reports_bounded_admission_and_completion() {
    let summary = run_request_scheduler_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.capacity, 2);
    assert_eq!(summary.admitted_requests, 3);
    assert_eq!(summary.completed_requests, 3);
    assert_eq!(summary.full_rejections, 1);
    assert_eq!(summary.duplicate_rejections, 1);
    assert_eq!(summary.missing_request_rejections, 1);
    assert_eq!(summary.premature_release_rejections, 1);
    assert_eq!(summary.released_slots, 3);
    assert_eq!(summary.reused_slots, 1);
    assert_eq!(summary.selection_decisions, summary.generated_tokens);
    assert_eq!(summary.no_ready_selection_rejections, 1);
    assert_eq!(summary.no_ready_selection_scanned_slots, 2);
    assert_eq!(summary.no_ready_selection_skipped_slots, 2);
    assert!(summary.selection_scanned_slots >= summary.selection_decisions);
    assert!(summary.selection_skipped_slots > 0);
    assert_eq!(summary.generated_tokens, summary.host_observed_tokens);
    assert_eq!(summary.token_ledgers, summary.generated_tokens);
    assert_eq!(summary.critical_path_reports, summary.generated_tokens);
    assert_eq!(summary.graph_replay_events, summary.generated_tokens);
    assert_eq!(summary.device_activity_events, summary.generated_tokens);
    assert_eq!(summary.copy_events, summary.generated_tokens);
    assert_eq!(summary.soft_visibility_syncs, summary.generated_tokens);
    assert_eq!(summary.gpu_idle_ns, 0);
    assert_eq!(summary.unclassified_syncs, 0);
    assert!(summary.host_event_wait_ns > 0);
    assert!(summary.estimated_events > 0);
    assert!(summary.runtime_timestamp_events > 0);
    assert!(summary.host_wait_gpu_idle_separated);
    assert!(summary.to_json().contains("\"bounded_slots\":true"));
    assert!(summary.to_json().contains("\"token_ledgers\":5"));
    assert!(summary.to_json().contains("\"selection_decisions\":5"));
}

fn ready_selection(
    outcome: SchedulerSelectionOutcome,
) -> crate::request::scheduler::selection::SchedulerSelection {
    match outcome {
        SchedulerSelectionOutcome::Ready(selection) => selection,
        SchedulerSelectionOutcome::NoReady(_) => panic!("expected ready scheduler selection"),
    }
}
