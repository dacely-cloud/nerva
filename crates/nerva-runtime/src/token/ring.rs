use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeviceTokenCompletion {
    Empty,
    DeviceComplete,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenRef {
    pub slot_index: usize,
    pub token_index: u64,
    pub version: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenInput {
    pub token: TokenId,
    pub token_ref: DeviceTokenRef,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenInputSource {
    Seed,
    DeviceRing(DeviceTokenRef),
    HostObservation,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenSlot {
    pub request_id: Option<RequestId>,
    pub sequence_id: Option<SequenceId>,
    pub token_index: u64,
    pub token: Option<TokenId>,
    pub version: u64,
    pub completion: DeviceTokenCompletion,
    pub host_copied: bool,
}

impl Default for DeviceTokenSlot {
    fn default() -> Self {
        Self {
            request_id: None,
            sequence_id: None,
            token_index: 0,
            token: None,
            version: 0,
            completion: DeviceTokenCompletion::Empty,
            host_copied: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenRing {
    slots: Vec<DeviceTokenSlot>,
}

impl DeviceTokenRing {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "device token ring capacity must be non-zero".to_string(),
            });
        }
        Ok(Self {
            slots: vec![DeviceTokenSlot::default(); capacity],
        })
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn slot(&self, slot_index: usize) -> Option<&DeviceTokenSlot> {
        self.slots.get(slot_index)
    }

    pub fn publish(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        token: TokenId,
    ) -> Result<DeviceTokenRef> {
        let slot_index = self.slot_index(token_index);
        let slot = &mut self.slots[slot_index];
        if slot.completion == DeviceTokenCompletion::DeviceComplete
            && !slot.host_copied
            && slot.token_index != token_index
        {
            return Err(NervaError::ResidencyViolation {
                block_id: ResidentBlockId(0),
                reason: "device token ring slot reused before host observation".to_string(),
            });
        }
        slot.request_id = Some(request_id);
        slot.sequence_id = Some(sequence_id);
        slot.token_index = token_index;
        slot.token = Some(token);
        slot.version = slot.version.saturating_add(1);
        slot.completion = DeviceTokenCompletion::DeviceComplete;
        slot.host_copied = false;
        Ok(DeviceTokenRef {
            slot_index,
            token_index,
            version: slot.version,
        })
    }

    pub fn consume_device_input(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
        previous_token_index: u64,
    ) -> Result<TokenId> {
        self.consume_device_input_ref(request_id, sequence_id, previous_token_index)
            .map(|input| input.token)
    }

    pub fn consume_device_input_ref(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
        previous_token_index: u64,
    ) -> Result<DeviceTokenInput> {
        let slot_index = self.slot_index(previous_token_index);
        let slot = &self.slots[slot_index];
        if slot.request_id != Some(request_id)
            || slot.sequence_id != Some(sequence_id)
            || slot.token_index != previous_token_index
            || slot.completion != DeviceTokenCompletion::DeviceComplete
        {
            return Err(NervaError::ResidencyViolation {
                block_id: ResidentBlockId(0),
                reason: "device token ring read was stale or incomplete".to_string(),
            });
        }
        let token = slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: ResidentBlockId(0),
            reason: "device token ring slot has no token".to_string(),
        })?;
        Ok(DeviceTokenInput {
            token,
            token_ref: DeviceTokenRef {
                slot_index,
                token_index: previous_token_index,
                version: slot.version,
            },
        })
    }

    pub fn host_observe(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
    ) -> Result<TokenId> {
        let slot_index = self.slot_index(token_index);
        let slot = &mut self.slots[slot_index];
        if slot.request_id != Some(request_id)
            || slot.sequence_id != Some(sequence_id)
            || slot.token_index != token_index
            || slot.completion != DeviceTokenCompletion::DeviceComplete
        {
            return Err(NervaError::ResidencyViolation {
                block_id: ResidentBlockId(0),
                reason: "host token observation read stale device state".to_string(),
            });
        }
        slot.host_copied = true;
        slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: ResidentBlockId(0),
            reason: "host-visible token slot has no token".to_string(),
        })
    }

    fn slot_index(&self, token_index: u64) -> usize {
        token_index as usize % self.slots.len()
    }
}
