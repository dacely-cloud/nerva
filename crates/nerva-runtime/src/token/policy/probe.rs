use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::engine::runtime::Runtime;
use crate::token::policy::probe::counters::TokenPolicyCounters;
use crate::token::policy::summary::TokenPolicySummary;
use crate::token::policy::types::{TokenPolicyPath, TokenPolicyPlan};

mod counters;

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
            counters.record_output(step.path, &output);
            last_host_visible = Some(output.token);
        }

        Ok(counters.summary())
    }
}
