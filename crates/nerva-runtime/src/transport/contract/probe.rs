use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::transport::TransportDeviceId;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::engine::runtime::Runtime;
use crate::transport::contract::backend::PinnedHostLoopbackTransport;
use crate::transport::contract::probe::allocation::allocate_ready_contract_buffer;
use crate::transport::contract::probe::ledger::{empty_completion, record_completion};
use crate::transport::contract::summary::{TransportContractStatus, TransportContractSummary};
use crate::transport::contract::traits::TensorTransportContract;
use crate::transport::contract::types::{ReceiveDescriptor, TransferDescriptor, TransportEndpoint};
use crate::transport::contract::visibility::TransportVisibilityTracker;
use crate::transport::path::types::TransferMode;

mod allocation;
mod ledger;

impl Runtime {
    pub fn run_transport_contract_probe(&self) -> Result<TransportContractSummary> {
        run_transport_contract_probe()
    }
}

pub fn run_transport_contract_probe() -> Result<TransportContractSummary> {
    let mut registry = BlockRegistry::new([(MemoryTier::PinnedDram, 256 * 1024)]);
    let source_id = allocate_ready_contract_buffer(&mut registry, AllocationId(70), 0)?;
    let destination_id = allocate_ready_contract_buffer(&mut registry, AllocationId(71), 0)?;
    let source = registry
        .block(source_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "transport contract source block is missing".to_string(),
        })?;
    let destination =
        registry
            .block(destination_id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "transport contract destination block is missing".to_string(),
            })?;

    let mut transport = PinnedHostLoopbackTransport::new(4)?;
    let mut ledger = TokenLedger::new(0);
    let endpoint = TransportEndpoint {
        device: TransportDeviceId(0),
        stage_id: 1,
        lane_id: 0,
    };

    let source_registration = transport.register(source, source.authoritative_copy)?;
    let destination_registration =
        transport.register(destination, destination.authoritative_copy)?;
    let receive = ReceiveDescriptor {
        request_id: RequestId(1),
        sequence_id: SequenceId(1),
        destination: destination_registration,
        destination_replica: destination.authoritative_copy,
        destination_offset: 0,
        expected_source_block: source_id,
        expected_version: source.version,
        bytes: 32 * 1024,
        mode: TransferMode::Decode,
    };
    let transfer = TransferDescriptor {
        request_id: RequestId(1),
        sequence_id: SequenceId(1),
        source: source_registration,
        source_replica: source.authoritative_copy,
        source_offset: 0,
        block_version: source.version,
        bytes: 32 * 1024,
        mode: TransferMode::Decode,
    };

    let _posted = transport.post_receive(&endpoint, receive)?;
    let _sent = transport.send(&endpoint, transfer)?;
    let mut completions = [empty_completion()];
    let completion_count = transport.poll(&mut completions)?;
    record_completion(&mut ledger, completions[0]);
    let mut visibility = TransportVisibilityTracker::default();
    visibility.observe_completion(completions[0])?;
    let pre_visibility_consume_rejections = u64::from(
        visibility
            .consume_visible(completions[0].transfer_id)
            .is_err(),
    );
    visibility.publish_visibility_fence(completions[0].transfer_id, &mut ledger)?;
    let visible_consumes = u64::from(
        visibility
            .consume_visible(completions[0].transfer_id)
            .is_ok(),
    );

    let unposted_send_rejections = u64::from(transport.send(&endpoint, transfer).is_err());
    let stale_transfer = TransferDescriptor {
        block_version: source.version.saturating_sub(1),
        ..transfer
    };
    let stale_version_rejections = u64::from(transport.send(&endpoint, stale_transfer).is_err());
    let oversized_receive = ReceiveDescriptor {
        bytes: destination.bytes + 1,
        ..receive
    };
    let descriptor_rejections = u64::from(
        transport
            .post_receive(&endpoint, oversized_receive)
            .is_err(),
    );

    ledger.require_zero_hot_path_allocations()?;
    ledger.require_classified_syncs()?;

    Ok(TransportContractSummary {
        status: TransportContractStatus::Ok,
        backend: transport.registration_backend().as_str(),
        registrations: 2,
        registered_entries: transport.registered_entries() as u64,
        preposted_receives: transport.preposted_receives() as u64,
        sends: 1,
        completions: completion_count as u64,
        bytes_completed: completions[0].bytes,
        unposted_send_rejections,
        stale_version_rejections,
        descriptor_rejections,
        pre_visibility_consume_rejections,
        visibility_fences: ledger.sync_count_for(SyncClass::PhaseHandoff),
        visible_consumes,
        per_transfer_registrations: 0,
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
