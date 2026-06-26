use nerva_core::types::error::Result;
use nerva_core::types::id::token::TokenId;

use crate::token::engine::ledger;
use crate::token::engine::synthetic::SyntheticEngine;
use crate::token::ring::TokenInputSource;
use crate::token::step::{StepOutput, SyntheticStepPlan};

#[must_use = "PendingSyntheticStep must be collect()-ed; dropping it loses the launched transaction"]
#[derive(Debug)]
pub struct PendingSyntheticStep<'engine> {
    pub(crate) engine: &'engine mut SyntheticEngine,
    pub(crate) plan: Option<SyntheticStepPlan>,
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
        let ledger = ledger::synthetic_step_ledger(
            self.engine.device,
            plan.token_index,
            plan.input_source == TokenInputSource::HostObservation,
        )?;

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
