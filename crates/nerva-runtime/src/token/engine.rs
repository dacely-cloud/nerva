use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{
    DeviceOrdinal, RequestId, ResidentBlockId, SequenceId, TokenId, TransactionId,
};
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

use crate::graph::layout::GraphLayout;
use crate::graph::pool::GraphPool;
use crate::token::ring::{DeviceTokenRing, TokenInputSource};
use crate::token::step::{StepOutput, SyntheticStepPlan};

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
        if plan.input_source == TokenInputSource::HostObservation {
            ledger.record_sync(
                SyncClass::PolicySync,
                None,
                Some(MemoryTier::PinnedDram),
                Some(MemoryTier::Vram),
                0,
                1,
                MetricSource::EstimatedModel,
                "host_policy_barrier",
            );
        }

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
