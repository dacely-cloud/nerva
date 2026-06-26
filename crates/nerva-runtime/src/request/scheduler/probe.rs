use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::probe::next_cycle_token;
use crate::request::scheduler::admission::RequestAdmission;
use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::scheduler::ledger::scheduler_token_ledger;
use crate::request::scheduler::summary::{RequestSchedulerProbeStatus, RequestSchedulerSummary};
use crate::request::scheduler::totals::SchedulerLedgerTotals;

pub fn run_request_scheduler_probe() -> Result<RequestSchedulerSummary> {
    let device = DeviceOrdinal(0);
    let mut scheduler = BoundedRequestScheduler::new(2)?;
    let mut ledger_totals = SchedulerLedgerTotals::default();
    let mut released_slots = 0;
    let mut reused_slots = 0;
    let mut host_observed_tokens = 0;
    let prompt_a = vec![TokenId(0)];
    let prompt_b = vec![TokenId(1)];
    let slot_b = scheduler.admit(admission(2, 2, prompt_b.clone(), 2, Some(TokenId(2))))?;
    scheduler.admit(admission(1, 1, prompt_a.clone(), 3, Some(TokenId(3))))?;
    let reuse_admission = admission(3, 3, vec![TokenId(2)], 1, Some(TokenId(3)));
    let full_rejections = scheduler.admit(reuse_admission.clone()).is_err() as u64;
    let duplicate_rejections = scheduler.admit(admission(1, 9, prompt_a, 1, None)).is_err() as u64;

    scheduler.begin_decode(RequestId(1))?;
    scheduler.begin_decode(RequestId(2))?;
    let mut max_active = scheduler.active_count();
    let mut iterations = 0;
    drive_one_step(
        &mut scheduler,
        RequestId(1),
        device,
        &mut ledger_totals,
        &mut iterations,
    )?;
    drive_one_step(
        &mut scheduler,
        RequestId(2),
        device,
        &mut ledger_totals,
        &mut iterations,
    )?;
    let premature_release_rejections = scheduler.release_completed(RequestId(2)).is_err() as u64;
    host_observed_tokens += observe_and_count(&mut scheduler, RequestId(2), usize::MAX)?;
    let released_slot = scheduler.release_completed(RequestId(2))?;
    released_slots += 1;
    let reused_slot = scheduler.admit(reuse_admission)?;
    reused_slots += (reused_slot == released_slot && reused_slot == slot_b) as u64;
    scheduler.begin_decode(RequestId(3))?;
    max_active = max_active.max(scheduler.active_count());
    drive_one_step(
        &mut scheduler,
        RequestId(3),
        device,
        &mut ledger_totals,
        &mut iterations,
    )?;
    host_observed_tokens += observe_and_count(&mut scheduler, RequestId(3), usize::MAX)?;
    scheduler.release_completed(RequestId(3))?;
    released_slots += 1;
    max_active = max_active.max(scheduler.active_count());
    drive_one_step(
        &mut scheduler,
        RequestId(1),
        device,
        &mut ledger_totals,
        &mut iterations,
    )?;
    host_observed_tokens += observe_and_count(&mut scheduler, RequestId(1), 1)?;
    max_active = max_active.max(scheduler.active_count());
    drive_one_step(
        &mut scheduler,
        RequestId(1),
        device,
        &mut ledger_totals,
        &mut iterations,
    )?;
    host_observed_tokens += observe_and_count(&mut scheduler, RequestId(1), usize::MAX)?;

    let missing_request_rejections = scheduler.next_device_input(RequestId(99)).is_err() as u64;
    let generated_tokens = ledger_totals.token_ledgers;

    Ok(RequestSchedulerSummary {
        status: RequestSchedulerProbeStatus::Ok,
        capacity: scheduler.capacity(),
        admitted_requests: 3,
        active_requests: scheduler.active_count(),
        completed_requests: scheduler.completed_count() as u64 + released_slots,
        full_rejections,
        duplicate_rejections,
        missing_request_rejections,
        premature_release_rejections,
        released_slots,
        reused_slots,
        scheduler_iterations: iterations,
        max_active_requests: max_active,
        host_observed_tokens: host_observed_tokens as u64,
        generated_tokens,
        token_ledgers: ledger_totals.token_ledgers,
        critical_path_reports: ledger_totals.critical_path_reports,
        graph_replay_events: ledger_totals.graph_replay_events,
        device_activity_events: ledger_totals.device_activity_events,
        copy_events: ledger_totals.copy_events,
        soft_visibility_syncs: ledger_totals.soft_visibility_syncs,
        host_event_wait_ns: ledger_totals.host_event_wait_ns,
        gpu_idle_ns: ledger_totals.gpu_idle_ns,
        estimated_events: ledger_totals.estimated_events,
        runtime_timestamp_events: ledger_totals.runtime_timestamp_events,
        unclassified_syncs: ledger_totals.unclassified_syncs,
        bounded_slots: scheduler.capacity() == 2,
        unbounded_queue_ops: 0,
        host_wait_gpu_idle_separated: ledger_totals.host_wait_gpu_idle_separated,
        hot_path_allocations: ledger_totals.hot_path_allocations,
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
    device: DeviceOrdinal,
    ledger_totals: &mut SchedulerLedgerTotals,
    iterations: &mut u64,
) -> Result<()> {
    let seed = scheduler.next_device_input(request_id)?;
    let token = next_cycle_token(seed);
    let token_index = scheduler.controller(request_id)?.generated_tokens.len();
    let (ledger, report) = scheduler_token_ledger(device, request_id, token_index as u64, token)?;
    scheduler.record_device_token(request_id, token_index, token)?;
    ledger_totals.record(&ledger, &report, device)?;
    *iterations += 1;
    Ok(())
}

fn observe_and_count(
    scheduler: &mut BoundedRequestScheduler,
    request_id: RequestId,
    max_tokens: usize,
) -> Result<usize> {
    scheduler
        .observe_host_tokens(request_id, max_tokens)
        .map(|batch| batch.tokens.len())
}
