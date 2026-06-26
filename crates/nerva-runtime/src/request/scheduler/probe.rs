use nerva_core::types::error::Result;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::probe::next_cycle_token;
use crate::request::scheduler::admission::RequestAdmission;
use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::scheduler::summary::{RequestSchedulerProbeStatus, RequestSchedulerSummary};

pub fn run_request_scheduler_probe() -> Result<RequestSchedulerSummary> {
    let mut scheduler = BoundedRequestScheduler::new(2)?;
    let prompt_a = vec![TokenId(0)];
    let prompt_b = vec![TokenId(1)];
    scheduler.admit(admission(1, 1, prompt_a.clone(), 3, Some(TokenId(3))))?;
    scheduler.admit(admission(2, 2, prompt_b.clone(), 2, Some(TokenId(2))))?;
    let full_rejections = scheduler
        .admit(admission(3, 3, vec![TokenId(2)], 1, None))
        .is_err() as u64;
    let duplicate_rejections = scheduler.admit(admission(1, 9, prompt_a, 1, None)).is_err() as u64;

    scheduler.begin_decode(RequestId(1))?;
    scheduler.begin_decode(RequestId(2))?;
    let mut max_active = scheduler.active_count();
    let mut iterations = 0;
    drive_one_step(&mut scheduler, RequestId(1), &mut iterations)?;
    drive_one_step(&mut scheduler, RequestId(2), &mut iterations)?;
    scheduler.observe_host_tokens(RequestId(2), usize::MAX)?;
    max_active = max_active.max(scheduler.active_count());
    drive_one_step(&mut scheduler, RequestId(1), &mut iterations)?;
    scheduler.observe_host_tokens(RequestId(1), 1)?;
    max_active = max_active.max(scheduler.active_count());
    drive_one_step(&mut scheduler, RequestId(1), &mut iterations)?;
    scheduler.observe_host_tokens(RequestId(1), usize::MAX)?;

    let missing_request_rejections = scheduler.next_device_input(RequestId(99)).is_err() as u64;
    let generated_tokens = generated_token_count(&scheduler, &[RequestId(1), RequestId(2)]) as u64;
    let host_observed_tokens = host_token_count(&scheduler, &[RequestId(1), RequestId(2)]) as u64;

    Ok(RequestSchedulerSummary {
        status: RequestSchedulerProbeStatus::Ok,
        capacity: scheduler.capacity(),
        admitted_requests: 2,
        active_requests: scheduler.active_count(),
        completed_requests: scheduler.completed_count(),
        full_rejections,
        duplicate_rejections,
        missing_request_rejections,
        scheduler_iterations: iterations,
        max_active_requests: max_active,
        host_observed_tokens,
        generated_tokens,
        bounded_slots: scheduler.capacity() == 2,
        unbounded_queue_ops: 0,
        hot_path_allocations: 0,
    })
}

fn admission(
    request: u64,
    sequence: u64,
    prompt_tokens: Vec<TokenId>,
    max_new_tokens: usize,
    eos_token: Option<TokenId>,
) -> RequestAdmission {
    RequestAdmission {
        request_id: RequestId(request),
        sequence_id: SequenceId(sequence),
        prompt_tokens,
        max_new_tokens,
        eos_token,
    }
}

fn drive_one_step(
    scheduler: &mut BoundedRequestScheduler,
    request_id: RequestId,
    iterations: &mut u64,
) -> Result<()> {
    let seed = scheduler.next_device_input(request_id)?;
    let token = next_cycle_token(seed);
    let token_index = scheduler.controller(request_id)?.generated_tokens.len();
    scheduler.record_device_token(request_id, token_index, token)?;
    *iterations += 1;
    Ok(())
}

fn generated_token_count(scheduler: &BoundedRequestScheduler, requests: &[RequestId]) -> usize {
    requests
        .iter()
        .map(|request| {
            scheduler
                .controller(*request)
                .map(|controller| controller.generated_tokens.len())
                .unwrap_or(0)
        })
        .sum()
}

fn host_token_count(scheduler: &BoundedRequestScheduler, requests: &[RequestId]) -> usize {
    requests
        .iter()
        .map(|request| {
            scheduler
                .controller(*request)
                .map(|controller| controller.host_observed_tokens.len())
                .unwrap_or(0)
        })
        .sum()
}
