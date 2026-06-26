use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::token::ring::{DeviceTokenCompletion, DeviceTokenRing};

impl DeviceTokenRing {
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
}
