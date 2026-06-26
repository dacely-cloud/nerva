use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::probe::counters::RegistrationProbeCounters;
use crate::transport::registration::probe::fixture::RegistrationProbeBlocks;
use crate::transport::registration::probe::lookup::record_lookup;
use crate::transport::registration::types::TransportRegistrationBackend;

pub(crate) fn bootstrap_registration_cache(
    registry: &BlockRegistry,
    cache: &mut TransportRegistrationCache,
    counters: &mut RegistrationProbeCounters,
    blocks: RegistrationProbeBlocks,
    direct_verified: bool,
) -> Result<()> {
    for (id, backend) in [
        (
            blocks.pinned_send,
            TransportRegistrationBackend::RdmaPinnedHost,
        ),
        (
            blocks.pinned_recv,
            TransportRegistrationBackend::RdmaPinnedHost,
        ),
        (
            blocks.pinned_send,
            TransportRegistrationBackend::DpdkPinnedHost,
        ),
    ] {
        let block = registry.block(id).expect("probe block exists");
        cache.register(block, block.authoritative_copy, backend)?;
        counters.record_bootstrap_registration(backend, direct_verified);
    }

    if direct_verified {
        let block = registry
            .block(blocks.gpu_direct)
            .expect("probe block exists");
        cache.register(
            block,
            block.authoritative_copy,
            TransportRegistrationBackend::RdmaGpuDirect,
        )?;
        counters.record_bootstrap_registration(
            TransportRegistrationBackend::RdmaGpuDirect,
            direct_verified,
        );
    } else {
        counters.record_gpu_direct_registration_skip();
    }
    Ok(())
}

pub(crate) fn record_registered_lookup_hits(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    ledger: &mut TokenLedger,
    counters: &mut RegistrationProbeCounters,
    blocks: RegistrationProbeBlocks,
    direct_verified: bool,
) {
    for (id, backend, label) in [
        (
            blocks.pinned_send,
            TransportRegistrationBackend::RdmaPinnedHost,
            "registration_cache_rdma_send_hit",
        ),
        (
            blocks.pinned_recv,
            TransportRegistrationBackend::RdmaPinnedHost,
            "registration_cache_rdma_recv_hit",
        ),
        (
            blocks.pinned_send,
            TransportRegistrationBackend::DpdkPinnedHost,
            "registration_cache_dpdk_send_hit",
        ),
    ] {
        record_lookup(registry, cache, ledger, counters, id, backend, 0, label);
    }

    if direct_verified {
        record_lookup(
            registry,
            cache,
            ledger,
            counters,
            blocks.gpu_direct,
            TransportRegistrationBackend::RdmaGpuDirect,
            0,
            "registration_cache_gpu_direct_hit",
        );
    }
}
