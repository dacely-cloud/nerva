use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::scheduler::admission::RequestAdmission;
use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::scheduler::probe::run_request_scheduler_probe;

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
fn request_scheduler_probe_reports_bounded_admission_and_completion() {
    let summary = run_request_scheduler_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.capacity, 2);
    assert_eq!(summary.admitted_requests, 2);
    assert_eq!(summary.completed_requests, 2);
    assert_eq!(summary.full_rejections, 1);
    assert_eq!(summary.duplicate_rejections, 1);
    assert_eq!(summary.missing_request_rejections, 1);
    assert_eq!(summary.generated_tokens, summary.host_observed_tokens);
    assert!(summary.to_json().contains("\"bounded_slots\":true"));
}
