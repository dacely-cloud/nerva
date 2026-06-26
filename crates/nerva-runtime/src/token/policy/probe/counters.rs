use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;

use crate::token::policy::summary::{TokenPolicyStatus, TokenPolicySummary};
use crate::token::policy::types::TokenPolicyPath;
use crate::token::ring::TokenInputSource;
use crate::token::step::StepOutput;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct TokenPolicyCounters {
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
    pub(super) const fn new() -> Self {
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

    pub(super) fn record_output(&mut self, path: TokenPolicyPath, output: &StepOutput) {
        self.record_path(path, output.input_source);
        self.record_input_edge(output.input_source);
        self.record_output_token(output.token_index, output.token);
        self.record_ledger(output);
    }

    pub(super) fn summary(self) -> TokenPolicySummary {
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

    fn record_path(&mut self, path: TokenPolicyPath, input_source: TokenInputSource) {
        self.steps = self.steps.saturating_add(1);
        match path {
            TokenPolicyPath::DeviceFastPath => {
                self.device_fast_steps = self.device_fast_steps.saturating_add(1);
                self.count_fast_path_host_dependency(input_source);
            }
            TokenPolicyPath::HostPolicyPath => {
                self.host_policy_steps = self.host_policy_steps.saturating_add(1);
                self.host_visibility_hard_dependencies =
                    self.host_visibility_hard_dependencies.saturating_add(1);
            }
            TokenPolicyPath::HybridValidationPath => {
                self.hybrid_validation_steps = self.hybrid_validation_steps.saturating_add(1);
                self.count_fast_path_host_dependency(input_source);
            }
        }
    }

    fn record_input_edge(&mut self, input_source: TokenInputSource) {
        match input_source {
            TokenInputSource::Seed => self.seed_edges = self.seed_edges.saturating_add(1),
            TokenInputSource::DeviceRing(_) => {
                self.device_ring_edges = self.device_ring_edges.saturating_add(1);
            }
            TokenInputSource::HostObservation => {
                self.host_causality_edges = self.host_causality_edges.saturating_add(1);
            }
        }
    }

    fn record_output_token(&mut self, token_index: u64, token: TokenId) {
        self.observed_tokens = self.observed_tokens.saturating_add(1);
        let expected = TokenId(5u32.wrapping_add((token_index as u32).wrapping_add(1)));
        if token != expected {
            self.mismatched_tokens = self.mismatched_tokens.saturating_add(1);
        }
    }

    fn record_ledger(&mut self, output: &StepOutput) {
        self.graph_replays = self
            .graph_replays
            .saturating_add(output.ledger.event_count(LedgerEventKind::GraphReplay));
        self.policy_syncs = self
            .policy_syncs
            .saturating_add(output.ledger.sync_count_for(SyncClass::PolicySync));
        self.soft_visibility_syncs = self
            .soft_visibility_syncs
            .saturating_add(output.ledger.sync_count_for(SyncClass::SoftVisibilitySync));
        self.hot_path_allocations = self
            .hot_path_allocations
            .saturating_add(output.ledger.hot_path_allocations);
    }

    fn count_fast_path_host_dependency(&mut self, input_source: TokenInputSource) {
        if input_source == TokenInputSource::HostObservation {
            self.device_fast_host_dependencies =
                self.device_fast_host_dependencies.saturating_add(1);
        }
    }
}
