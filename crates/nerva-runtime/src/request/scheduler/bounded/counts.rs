use crate::request::scheduler::bounded::BoundedRequestScheduler;
use crate::request::types::RequestPhase;

impl BoundedRequestScheduler {
    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|controller| controller.phase != RequestPhase::Completed)
            .count()
    }

    pub fn completed_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|controller| controller.phase == RequestPhase::Completed)
            .count()
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}
