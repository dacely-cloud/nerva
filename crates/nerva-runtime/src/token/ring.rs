use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

mod consume;
mod observe;
mod publish;

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

impl DeviceTokenSlot {
    fn blocks_reuse_for(&self, token_index: u64) -> bool {
        self.completion == DeviceTokenCompletion::DeviceComplete
            && !self.host_copied
            && self.token_index != token_index
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

    fn slot_index(&self, token_index: u64) -> usize {
        token_index as usize % self.slots.len()
    }
}
