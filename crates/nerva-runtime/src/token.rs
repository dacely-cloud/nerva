use crate::graph::{GraphLayout, GraphPool};
use nerva_core::types::{
    DeviceOrdinal, MemoryTier, NervaError, RequestId, Result, SequenceId, TokenId, TransactionId,
};
use nerva_ledger::types::{
    DeviceTimelineSpan, LedgerEvent, LedgerEventKind, MetricSource, SyncClass, TokenLedger,
};

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
                block_id: nerva_core::types::ResidentBlockId(0),
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
                block_id: nerva_core::types::ResidentBlockId(0),
                reason: "device token ring read was stale or incomplete".to_string(),
            });
        }
        let token = slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: nerva_core::types::ResidentBlockId(0),
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
                block_id: nerva_core::types::ResidentBlockId(0),
                reason: "host token observation read stale device state".to_string(),
            });
        }
        slot.host_copied = true;
        slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: nerva_core::types::ResidentBlockId(0),
            reason: "host-visible token slot has no token".to_string(),
        })
    }

    fn slot_index(&self, token_index: u64) -> usize {
        token_index as usize % self.slots.len()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticStepPlan {
    pub transaction_id: TransactionId,
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
    pub input_token: TokenId,
    pub input_source: TokenInputSource,
    pub layout: GraphLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
    pub input_source: TokenInputSource,
    pub device_token_ref: DeviceTokenRef,
    pub token: TokenId,
    pub finished: bool,
    pub ledger: TokenLedger,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntheticEngine {
    graph_pool: GraphPool,
    token_ring: DeviceTokenRing,
    next_transaction_id: u64,
    layout: GraphLayout,
    device: DeviceOrdinal,
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
                    block_id: nerva_core::types::ResidentBlockId(0),
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

#[must_use = "PendingSyntheticStep must be collect()-ed; dropping it loses the launched transaction"]
#[derive(Debug)]
pub struct PendingSyntheticStep<'engine> {
    engine: &'engine mut SyntheticEngine,
    plan: Option<SyntheticStepPlan>,
}

impl<'engine> PendingSyntheticStep<'engine> {
    pub fn plan(&self) -> Option<&SyntheticStepPlan> {
        self.plan.as_ref()
    }

    pub fn collect(mut self) -> Result<StepOutput> {
        let plan = self
            .plan
            .take()
            .expect("PendingSyntheticStep::collect called twice");
        self.engine.graph_pool.replay(plan.layout)?;

        let token = TokenId(plan.input_token.0.wrapping_add(1));
        let device_token_ref = self.engine.token_ring.publish(
            plan.request_id,
            plan.sequence_id,
            plan.token_index,
            token,
        )?;

        let host_visible_token = self.engine.token_ring.host_observe(
            plan.request_id,
            plan.sequence_id,
            plan.token_index,
        )?;
        let mut ledger = TokenLedger::new(plan.token_index);
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::GraphReplay,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 1,
            label: "synthetic_graph_replay",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::DeviceActivity,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 3,
            label: "synthetic_decode_kernel",
        });
        ledger.record_device_span(DeviceTimelineSpan::new(
            self.engine.device,
            0,
            3,
            MetricSource::EstimatedModel,
            "synthetic_decode_device_active",
        ))?;
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: core::mem::size_of::<TokenId>(),
            latency_ns: 1,
            label: "async_host_token_observation",
        });
        ledger.record_sync(
            SyncClass::SoftVisibilitySync,
            None,
            Some(MemoryTier::Vram),
            Some(MemoryTier::PinnedDram),
            0,
            1,
            MetricSource::EstimatedModel,
            "soft_visibility_host_wait",
        );

        Ok(StepOutput {
            request_id: plan.request_id,
            sequence_id: plan.sequence_id,
            token_index: plan.token_index,
            input_source: plan.input_source,
            device_token_ref,
            token: host_visible_token,
            finished: false,
            ledger,
        })
    }
}

impl Drop for PendingSyntheticStep<'_> {
    fn drop(&mut self) {
        debug_assert!(
            self.plan.is_none(),
            "PendingSyntheticStep dropped without collect(); transaction output leaked"
        );
    }
}
