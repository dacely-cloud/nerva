use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::ownership::{ExecutionOwner, MutationSemantics};

use crate::phase::types::{
    PhaseHandoffEntry, PhaseHandoffPlan, PhaseHandoffPlanner, PhaseHandoffRejection,
    PhaseHandoffRejectionKind, PhaseHandoffRequest,
};
use crate::registry::table::BlockRegistry;

impl PhaseHandoffPlanner {
    pub fn plan(
        registry: &BlockRegistry,
        requests: &[PhaseHandoffRequest],
    ) -> Result<PhaseHandoffPlan> {
        if requests.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "phase handoff planner requires at least one request".to_string(),
            });
        }

        let mut entries = Vec::new();
        let mut rejections = Vec::new();
        for request in requests {
            match registry.block(request.block_id) {
                Some(block) => match validate_request(block, request) {
                    Ok(entry) => entries.push(entry),
                    Err(rejection) => rejections.push(rejection),
                },
                None => rejections.push(PhaseHandoffRejection {
                    block_id: request.block_id,
                    requested_from: request.from,
                    requested_to: request.to,
                    kind: PhaseHandoffRejectionKind::MissingBlock,
                    observed_owner: ExecutionOwner::None,
                    observed_version: 0,
                    reason: request.reason,
                }),
            }
        }
        Ok(PhaseHandoffPlan {
            entries,
            rejections,
        })
    }
}

fn validate_request(
    block: &ResidentBlock,
    request: &PhaseHandoffRequest,
) -> core::result::Result<PhaseHandoffEntry, PhaseHandoffRejection> {
    if block.state != ResidencyState::Ready {
        return Err(rejection(
            block,
            request,
            PhaseHandoffRejectionKind::BlockNotReady,
        ));
    }
    if block.version < request.required_version {
        return Err(rejection(
            block,
            request,
            PhaseHandoffRejectionKind::StaleVersion,
        ));
    }
    if block.owner != request.from {
        return Err(rejection(
            block,
            request,
            PhaseHandoffRejectionKind::OwnerMismatch,
        ));
    }
    if !legal_transition(block, request.from, request.to) {
        return Err(rejection(
            block,
            request,
            PhaseHandoffRejectionKind::IllegalTransition,
        ));
    }
    Ok(PhaseHandoffEntry {
        block_id: block.id,
        from: request.from,
        to: request.to,
        bytes: block.bytes,
        version_before: block.version,
        predicted_visible_ns: estimate_handoff_ns(block.bytes),
        reason: request.reason,
    })
}

fn legal_transition(block: &ResidentBlock, from: ExecutionOwner, to: ExecutionOwner) -> bool {
    match (from, to) {
        (ExecutionOwner::Cpu, ExecutionOwner::Gpu(_))
        | (ExecutionOwner::Gpu(_), ExecutionOwner::Cpu)
        | (ExecutionOwner::Gpu(_), ExecutionOwner::Nic(_))
        | (ExecutionOwner::Nic(_), ExecutionOwner::Gpu(_))
        | (ExecutionOwner::Cpu, ExecutionOwner::Nic(_))
        | (ExecutionOwner::Nic(_), ExecutionOwner::Cpu) => true,
        (ExecutionOwner::Cpu, ExecutionOwner::SharedReadOnly)
        | (ExecutionOwner::Gpu(_), ExecutionOwner::SharedReadOnly)
        | (ExecutionOwner::Nic(_), ExecutionOwner::SharedReadOnly) => {
            block.semantics == MutationSemantics::Immutable
        }
        _ => false,
    }
}

fn rejection(
    block: &ResidentBlock,
    request: &PhaseHandoffRequest,
    kind: PhaseHandoffRejectionKind,
) -> PhaseHandoffRejection {
    PhaseHandoffRejection {
        block_id: request.block_id,
        requested_from: request.from,
        requested_to: request.to,
        kind,
        observed_owner: block.owner,
        observed_version: block.version,
        reason: request.reason,
    }
}

fn estimate_handoff_ns(bytes: usize) -> u64 {
    1 + (bytes as u64 / 4096)
}
