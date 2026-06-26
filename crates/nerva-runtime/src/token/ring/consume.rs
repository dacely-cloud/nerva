use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::token::ring::{
    DeviceTokenCompletion, DeviceTokenInput, DeviceTokenRef, DeviceTokenRing,
};

impl DeviceTokenRing {
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
}
