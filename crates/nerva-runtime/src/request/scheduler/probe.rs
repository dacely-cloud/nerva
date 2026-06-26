use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;

use crate::request::scheduler::probe::fixture::RequestSchedulerProbeFixture;
use crate::request::scheduler::probe::step::{drive_selected_step, observe_and_count};
use crate::request::scheduler::selection::SchedulerSelectionOutcome;
use crate::request::scheduler::selection_totals::SchedulerSelectionTotals;
use crate::request::scheduler::summary::{RequestSchedulerProbeStatus, RequestSchedulerSummary};
use crate::request::scheduler::totals::SchedulerLedgerTotals;

mod fixture;
mod step;

pub fn run_request_scheduler_probe() -> Result<RequestSchedulerSummary> {
    let device = DeviceOrdinal(0);
    let mut fixture = RequestSchedulerProbeFixture::new()?;
    let mut ledger_totals = SchedulerLedgerTotals::default();
    let mut selection_totals = SchedulerSelectionTotals::default();
    let mut released_slots = 0;
    let mut reused_slots = 0;
    let mut host_observed_tokens = 0;

    fixture.scheduler.begin_decode(RequestId(1))?;
    fixture.scheduler.begin_decode(RequestId(2))?;
    let mut max_active = fixture.scheduler.active_count();
    let mut iterations = 0;
    drive_selected_step(
        &mut fixture.scheduler,
        device,
        &mut ledger_totals,
        &mut selection_totals,
        &mut iterations,
    )?;
    let premature_release_rejections =
        fixture.scheduler.release_completed(RequestId(2)).is_err() as u64;
    host_observed_tokens += observe_and_count(&mut fixture.scheduler, RequestId(2), usize::MAX)?;
    let released_slot = fixture.scheduler.release_completed(RequestId(2))?;
    released_slots += 1;
    let reused_slot = fixture.scheduler.admit(fixture.reuse_admission)?;
    reused_slots += (reused_slot == released_slot && reused_slot == fixture.slot_b) as u64;
    fixture.scheduler.begin_decode(RequestId(3))?;
    max_active = max_active.max(fixture.scheduler.active_count());
    drive_selected_step(
        &mut fixture.scheduler,
        device,
        &mut ledger_totals,
        &mut selection_totals,
        &mut iterations,
    )?;
    host_observed_tokens += observe_and_count(&mut fixture.scheduler, RequestId(1), 1)?;
    drive_selected_step(
        &mut fixture.scheduler,
        device,
        &mut ledger_totals,
        &mut selection_totals,
        &mut iterations,
    )?;
    host_observed_tokens += observe_and_count(&mut fixture.scheduler, RequestId(3), usize::MAX)?;
    fixture.scheduler.release_completed(RequestId(3))?;
    released_slots += 1;
    max_active = max_active.max(fixture.scheduler.active_count());
    drive_selected_step(
        &mut fixture.scheduler,
        device,
        &mut ledger_totals,
        &mut selection_totals,
        &mut iterations,
    )?;
    drive_selected_step(
        &mut fixture.scheduler,
        device,
        &mut ledger_totals,
        &mut selection_totals,
        &mut iterations,
    )?;
    host_observed_tokens += observe_and_count(&mut fixture.scheduler, RequestId(1), usize::MAX)?;
    fixture.scheduler.release_completed(RequestId(1))?;
    released_slots += 1;
    if let SchedulerSelectionOutcome::NoReady(miss) = fixture.scheduler.select_next_decoding() {
        selection_totals.record_no_ready(miss);
    }

    let missing_request_rejections =
        fixture.scheduler.next_device_input(RequestId(99)).is_err() as u64;
    let generated_tokens = ledger_totals.token_ledgers;

    Ok(RequestSchedulerSummary {
        status: RequestSchedulerProbeStatus::Ok,
        capacity: fixture.scheduler.capacity(),
        admitted_requests: 3,
        active_requests: fixture.scheduler.active_count(),
        completed_requests: fixture.scheduler.completed_count() as u64 + released_slots,
        full_rejections: fixture.full_rejections,
        duplicate_rejections: fixture.duplicate_rejections,
        missing_request_rejections,
        premature_release_rejections,
        released_slots,
        reused_slots,
        scheduler_iterations: iterations,
        selection_decisions: selection_totals.decisions,
        selection_scanned_slots: selection_totals.scanned_slots,
        selection_skipped_slots: selection_totals.skipped_slots,
        selection_wraps: selection_totals.wraps,
        no_ready_selection_rejections: selection_totals.no_ready_rejections,
        no_ready_selection_scanned_slots: selection_totals.no_ready_scanned_slots,
        no_ready_selection_skipped_slots: selection_totals.no_ready_skipped_slots,
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
        bounded_slots: fixture.scheduler.capacity() == 2,
        unbounded_queue_ops: 0,
        host_wait_gpu_idle_separated: ledger_totals.host_wait_gpu_idle_separated,
        hot_path_allocations: ledger_totals.hot_path_allocations,
    })
}
