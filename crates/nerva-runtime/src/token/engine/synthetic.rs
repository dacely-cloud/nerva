use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;
use nerva_core::types::id::transaction::TransactionId;

use crate::graph::layout::GraphLayout;
use crate::graph::pool::GraphPool;
use crate::token::engine::pending::PendingSyntheticStep;
use crate::token::ring::{DeviceTokenRing, TokenInputSource};
use crate::token::step::SyntheticStepPlan;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntheticEngine {
    pub(crate) graph_pool: GraphPool,
    pub(crate) token_ring: DeviceTokenRing,
    next_transaction_id: u64,
    pub(crate) layout: GraphLayout,
    pub(crate) device: DeviceOrdinal,
}

impl SyntheticEngine {
    pub fn new(token_ring_capacity: usize, device: DeviceOrdinal) -> Result<Self> {
        let layout = GraphLayout::new(1, 1, token_ring_capacity as u32, 1);
        let mut graph_pool = GraphPool::new();
        graph_pool.capture_synthetic(layout);
        Ok(Self {
            graph_pool,
            token_ring: DeviceTokenRing::new(token_ring_capacity)?,
            next_transaction_id: 1,
            layout,
            device,
        })
    }

    pub fn token_ring(&self) -> &DeviceTokenRing {
        &self.token_ring
    }

    pub fn graph_pool(&self) -> &GraphPool {
        &self.graph_pool
    }

    pub fn launch(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        input_token: TokenId,
    ) -> Result<PendingSyntheticStep<'_>> {
        let input_source = if token_index == 0 {
            TokenInputSource::Seed
        } else {
            let device_input = self.token_ring.consume_device_input_ref(
                request_id,
                sequence_id,
                token_index - 1,
            )?;
            if device_input.token != input_token {
                return Err(NervaError::ResidencyViolation {
                    block_id: ResidentBlockId(0),
                    reason: "next input token does not match prior sampled device token"
                        .to_string(),
                });
            }
            TokenInputSource::DeviceRing(device_input.token_ref)
        };

        self.launch_with_source(
            request_id,
            sequence_id,
            token_index,
            input_token,
            input_source,
        )
    }

    pub fn launch_device_next(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        seed_token: TokenId,
    ) -> Result<PendingSyntheticStep<'_>> {
        let (input_token, input_source) = if token_index == 0 {
            (seed_token, TokenInputSource::Seed)
        } else {
            let input = self.token_ring.consume_device_input_ref(
                request_id,
                sequence_id,
                token_index - 1,
            )?;
            (input.token, TokenInputSource::DeviceRing(input.token_ref))
        };

        self.launch_with_source(
            request_id,
            sequence_id,
            token_index,
            input_token,
            input_source,
        )
    }

    pub fn launch_host_policy_next(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        host_visible_previous_token: TokenId,
    ) -> Result<PendingSyntheticStep<'_>> {
        if token_index == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "host policy path requires a prior sampled token".to_string(),
            });
        }
        let device_input =
            self.token_ring
                .consume_device_input_ref(request_id, sequence_id, token_index - 1)?;
        if device_input.token != host_visible_previous_token {
            return Err(NervaError::ResidencyViolation {
                block_id: ResidentBlockId(0),
                reason: "host policy token does not match authoritative device token".to_string(),
            });
        }
        self.launch_with_source(
            request_id,
            sequence_id,
            token_index,
            host_visible_previous_token,
            TokenInputSource::HostObservation,
        )
    }

    fn launch_with_source(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        input_token: TokenId,
        input_source: TokenInputSource,
    ) -> Result<PendingSyntheticStep<'_>> {
        self.graph_pool.check_before_replay(self.layout)?;
        let transaction_id = TransactionId(self.next_transaction_id);
        self.next_transaction_id = self.next_transaction_id.saturating_add(1);
        let layout = self.layout;
        Ok(PendingSyntheticStep {
            engine: self,
            plan: Some(SyntheticStepPlan {
                transaction_id,
                request_id,
                sequence_id,
                token_index,
                input_token,
                input_source,
                layout,
            }),
        })
    }
}
