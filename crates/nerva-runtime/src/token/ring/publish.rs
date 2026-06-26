use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::token::ring::{DeviceTokenCompletion, DeviceTokenRef, DeviceTokenRing};

impl DeviceTokenRing {
    pub fn publish(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        token: TokenId,
    ) -> Result<DeviceTokenRef> {
        let slot_index = self.slot_index(token_index);
        let slot = &mut self.slots[slot_index];
        if slot.blocks_reuse_for(token_index) {
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
}
