use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;

use crate::request::probe::next_cycle_token;
use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::scheduler::ledger::scheduler_token_ledger;
use crate::request::scheduler::selection::SchedulerSelectionOutcome;
use crate::request::scheduler::selection_totals::SchedulerSelectionTotals;
use crate::request::scheduler::totals::SchedulerLedgerTotals;

pub(super) fn drive_selected_step(
    scheduler: &mut BoundedRequestScheduler,
    device: DeviceOrdinal,
    ledger_totals: &mut SchedulerLedgerTotals,
    selection_totals: &mut SchedulerSelectionTotals,
    iterations: &mut u64,
) -> Result<()> {
    let outcome = scheduler.select_next_decoding();
    let selection = match outcome {
        SchedulerSelectionOutcome::Ready(selection) => {
            selection_totals.record_ready(selection);
            selection
        }
        SchedulerSelectionOutcome::NoReady(miss) => {
            selection_totals.record_no_ready(miss);
            return Err(NervaError::InvalidArgument {
                reason: "scheduler has no decodable request".to_string(),
            });
        }
    };
    let request_id = selection.request_id;
    let seed = scheduler.next_device_input(request_id)?;
    let token = next_cycle_token(seed);
    let token_index = scheduler.controller(request_id)?.generated_tokens.len();
    let (ledger, report) = scheduler_token_ledger(device, request_id, token_index as u64, token)?;
    scheduler.record_device_token(request_id, token_index, token)?;
    ledger_totals.record(&ledger, &report, device)?;
    *iterations += 1;
    Ok(())
}

pub(super) fn observe_and_count(
    scheduler: &mut BoundedRequestScheduler,
    request_id: RequestId,
    max_tokens: usize,
) -> Result<usize> {
    scheduler
        .observe_host_tokens(request_id, max_tokens)
        .map(|batch| batch.tokens.len())
}
