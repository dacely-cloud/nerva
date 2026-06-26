use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{RequestId, ResidentBlockId, SequenceId, TokenId};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::Runtime;
use crate::token::policy::summary::{TokenPolicyStatus, TokenPolicySummary};
use crate::token::policy::types::{TokenPolicyPath, TokenPolicyPlan};
use crate::token::ring::TokenInputSource;

impl Runtime {
    pub fn run_token_policy_probe(&self) -> Result<TokenPolicySummary> {
        let plan = TokenPolicyPlan::probe_plan();
        if plan.steps.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "token policy probe requires at least one step".to_string(),
            });
        }

        let mut engine = self.synthetic_engine(8)?;
        let mut counters = TokenPolicyCounters::new();
        let request_id = RequestId(7);
        let sequence_id = SequenceId(1);
        let seed_token = TokenId(5);
        let mut last_host_visible = None;

        for step in plan.steps {
            let output = match step.path {
                TokenPolicyPath::DeviceFastPath | TokenPolicyPath::HybridValidationPath => engine
                    .launch_device_next(request_id, sequence_id, step.token_index, seed_token)?
                    .collect()?,
                TokenPolicyPath::HostPolicyPath => {
                    let Some(previous_token) = last_host_visible else {
                        return Err(NervaError::ResidencyViolation {
                            block_id: ResidentBlockId(0),
                            reason: "host policy path requires a previous host-visible token"
                                .to_string(),
                        });
                    };
                    engine
                        .launch_host_policy_next(
                            request_id,
                            sequence_id,
                            step.token_index,
                            previous_token,
                        )?
                        .collect()?
                }
            };
            output.ledger.require_zero_hot_path_allocations()?;
            output.ledger.require_classified_syncs()?;
            counters.record(
                step.path,
                output.input_source,
                output.token_index,
                output.token,
            );
            counters.graph_replays = counters
                .graph_replays
                .saturating_add(output.ledger.event_count(LedgerEventKind::GraphReplay));
            counters.policy_syncs = counters
                .policy_syncs
                .saturating_add(output.ledger.sync_count_for(SyncClass::PolicySync));
            counters.soft_visibility_syncs = counters
                .soft_visibility_syncs
                .saturating_add(output.ledger.sync_count_for(SyncClass::SoftVisibilitySync));
            counters.hot_path_allocations = counters
                .hot_path_allocations
                .saturating_add(output.ledger.hot_path_allocations);
            last_host_visible = Some(output.token);
        }

        Ok(counters.summary())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TokenPolicyCounters {
    steps: u64,
    device_fast_steps: u64,
    host_policy_steps: u64,
    hybrid_validation_steps: u64,
    seed_edges: u64,
    device_ring_edges: u64,
    host_causality_edges: u64,
    policy_syncs: u64,
    soft_visibility_syncs: u64,
    host_visibility_hard_dependencies: u64,
    device_fast_host_dependencies: u64,
    graph_replays: u64,
    observed_tokens: u64,
    mismatched_tokens: u64,
    hot_path_allocations: u64,
}

impl TokenPolicyCounters {
    const fn new() -> Self {
        Self {
            steps: 0,
            device_fast_steps: 0,
            host_policy_steps: 0,
            hybrid_validation_steps: 0,
            seed_edges: 0,
            device_ring_edges: 0,
            host_causality_edges: 0,
            policy_syncs: 0,
            soft_visibility_syncs: 0,
            host_visibility_hard_dependencies: 0,
            device_fast_host_dependencies: 0,
            graph_replays: 0,
            observed_tokens: 0,
            mismatched_tokens: 0,
            hot_path_allocations: 0,
        }
    }

    fn record(
        &mut self,
        path: TokenPolicyPath,
        input_source: TokenInputSource,
        token_index: u64,
        token: TokenId,
    ) {
        self.steps = self.steps.saturating_add(1);
        self.observed_tokens = self.observed_tokens.saturating_add(1);
        match path {
            TokenPolicyPath::DeviceFastPath => {
                self.device_fast_steps = self.device_fast_steps.saturating_add(1);
                if input_source == TokenInputSource::HostObservation {
                    self.device_fast_host_dependencies =
                        self.device_fast_host_dependencies.saturating_add(1);
                }
            }
            TokenPolicyPath::HostPolicyPath => {
                self.host_policy_steps = self.host_policy_steps.saturating_add(1);
                self.host_visibility_hard_dependencies =
                    self.host_visibility_hard_dependencies.saturating_add(1);
            }
            TokenPolicyPath::HybridValidationPath => {
                self.hybrid_validation_steps = self.hybrid_validation_steps.saturating_add(1);
                if input_source == TokenInputSource::HostObservation {
                    self.device_fast_host_dependencies =
                        self.device_fast_host_dependencies.saturating_add(1);
                }
            }
        }
        match input_source {
            TokenInputSource::Seed => self.seed_edges = self.seed_edges.saturating_add(1),
            TokenInputSource::DeviceRing(_) => {
                self.device_ring_edges = self.device_ring_edges.saturating_add(1);
            }
            TokenInputSource::HostObservation => {
                self.host_causality_edges = self.host_causality_edges.saturating_add(1);
            }
        }
        let expected = TokenId(5u32.wrapping_add((token_index as u32).wrapping_add(1)));
        if token != expected {
            self.mismatched_tokens = self.mismatched_tokens.saturating_add(1);
        }
    }

    fn summary(self) -> TokenPolicySummary {
        TokenPolicySummary {
            status: TokenPolicyStatus::Ok,
            steps: self.steps,
            device_fast_steps: self.device_fast_steps,
            host_policy_steps: self.host_policy_steps,
            hybrid_validation_steps: self.hybrid_validation_steps,
            seed_edges: self.seed_edges,
            device_ring_edges: self.device_ring_edges,
            host_causality_edges: self.host_causality_edges,
            policy_syncs: self.policy_syncs,
            soft_visibility_syncs: self.soft_visibility_syncs,
            host_visibility_hard_dependencies: self.host_visibility_hard_dependencies,
            device_fast_host_dependencies: self.device_fast_host_dependencies,
            graph_replays: self.graph_replays,
            observed_tokens: self.observed_tokens,
            mismatched_tokens: self.mismatched_tokens,
            hot_path_allocations: self.hot_path_allocations,
            error: None,
        }
    }
}
