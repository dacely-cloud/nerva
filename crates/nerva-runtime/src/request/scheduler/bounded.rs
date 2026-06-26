use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::token::TokenId;

use crate::request::controller::RequestController;
use crate::request::scheduler::admission::RequestAdmission;
use crate::request::scheduler::selection::{
    SchedulerSelection, SchedulerSelectionMiss, SchedulerSelectionOutcome,
};
use crate::request::types::{HostObservationBatch, RequestPhase};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedRequestScheduler {
    slots: Vec<Option<RequestController>>,
    next_selection_slot: usize,
}

impl BoundedRequestScheduler {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "request scheduler capacity must be non-zero".to_string(),
            });
        }
        Ok(Self {
            slots: vec![None; capacity],
            next_selection_slot: 0,
        })
    }

    pub fn admit(&mut self, admission: RequestAdmission) -> Result<usize> {
        if self.find_slot(admission.request_id).is_some() {
            return Err(NervaError::InvalidArgument {
                reason: format!("request {} already admitted", admission.request_id.0),
            });
        }
        let slot = self.slots.iter().position(Option::is_none).ok_or_else(|| {
            NervaError::AllocationFailed {
                bytes: 0,
                reason: "bounded request scheduler is full".to_string(),
            }
        })?;
        self.slots[slot] = Some(RequestController::new(
            admission.request_id,
            admission.sequence_id,
            admission.prompt_tokens,
            admission.max_new_tokens,
            admission.eos_token,
        )?);
        Ok(slot)
    }

    pub fn begin_decode(&mut self, request_id: RequestId) -> Result<TokenId> {
        self.controller_mut(request_id)?.begin_decode()
    }

    pub fn next_device_input(&self, request_id: RequestId) -> Result<TokenId> {
        self.controller(request_id)?.next_device_input()
    }

    pub fn record_device_token(
        &mut self,
        request_id: RequestId,
        token_index: usize,
        token: TokenId,
    ) -> Result<()> {
        self.controller_mut(request_id)?
            .record_device_token(token_index, token)
    }

    pub fn observe_host_tokens(
        &mut self,
        request_id: RequestId,
        max_tokens: usize,
    ) -> Result<HostObservationBatch> {
        Ok(self
            .controller_mut(request_id)?
            .observe_host_tokens(max_tokens))
    }

    pub fn release_completed(&mut self, request_id: RequestId) -> Result<usize> {
        let slot = self
            .find_slot(request_id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("request {} is not admitted", request_id.0),
            })?;
        let controller = self.slots[slot]
            .as_ref()
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("request {} slot is empty", request_id.0),
            })?;
        if controller.phase != RequestPhase::Completed {
            return Err(NervaError::InvalidArgument {
                reason: format!("request {} is not completed", request_id.0),
            });
        }
        if controller.host_visibility_lag() != 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!("request {} has unobserved host tokens", request_id.0),
            });
        }
        self.slots[slot] = None;
        Ok(slot)
    }

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

    pub fn controller(&self, request_id: RequestId) -> Result<&RequestController> {
        self.find_slot(request_id)
            .and_then(|slot| self.slots[slot].as_ref())
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("request {} is not admitted", request_id.0),
            })
    }

    fn controller_mut(&mut self, request_id: RequestId) -> Result<&mut RequestController> {
        let slot = self
            .find_slot(request_id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("request {} is not admitted", request_id.0),
            })?;
        self.slots[slot]
            .as_mut()
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("request {} slot is empty", request_id.0),
            })
    }

    fn find_slot(&self, request_id: RequestId) -> Option<usize> {
        self.slots.iter().position(|slot| {
            slot.as_ref()
                .is_some_and(|item| item.request_id == request_id)
        })
    }
}
