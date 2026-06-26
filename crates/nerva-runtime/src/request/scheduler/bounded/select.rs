use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::scheduler::selection::{
    SchedulerSelection, SchedulerSelectionMiss, SchedulerSelectionOutcome,
};
use crate::request::types::RequestPhase;

impl BoundedRequestScheduler {
    pub fn select_next_decoding(&mut self) -> SchedulerSelectionOutcome {
        let capacity = self.slots.len();
        let start = self.next_selection_slot.min(capacity.saturating_sub(1));
        let mut skipped_slots = 0;
        for offset in 0..capacity {
            let slot = (start + offset) % capacity;
            let Some(controller) = self.slots[slot].as_ref() else {
                skipped_slots += 1;
                continue;
            };
            if controller.phase != RequestPhase::Decoding {
                skipped_slots += 1;
                continue;
            }
            self.next_selection_slot = (slot + 1) % capacity;
            return SchedulerSelectionOutcome::Ready(SchedulerSelection {
                slot,
                request_id: controller.request_id,
                scanned_slots: offset + 1,
                skipped_slots,
                wrapped: slot < start,
            });
        }
        SchedulerSelectionOutcome::NoReady(SchedulerSelectionMiss {
            scanned_slots: capacity,
            skipped_slots,
            wrapped: start != 0,
        })
    }
}
