use nerva_core::types::id::TokenId;
use nerva_core::types::memory::MemoryFabricKind;
use nerva_runtime::capabilities::snapshot::CapabilityState;
use nerva_runtime::engine::kv_probe::{KvResidencyProbeConfig, KvResidencyProbeStatus};
use nerva_runtime::engine::residency::ResidencyBudget;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use nerva_runtime::engine::synthetic::{SyntheticDecodeConfig, SyntheticDecodeStatus};
use nerva_runtime::transport::matrix::TransportCapabilityMatrixStatus;
use nerva_runtime::transport::probe::TransportPathProbeStatus;

mod audit;
mod files;
mod manifest;
mod report;
mod resident_weights;
mod vllm;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn build_acceptance_report() -> Result<AcceptanceReport, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut report = AcceptanceReport::default();

    let capabilities = runtime.discover_capabilities();
    let capability_passed = capabilities.target_os == "linux"
        && !capabilities.target_arch.is_empty()
        && capabilities.kernel_release.is_some()
        && matches!(capabilities.fabric, MemoryFabricKind::DiscreteExplicit)
        && !matches!(
            capabilities.pinned_host_staging,
            CapabilityState::Unsupported
        );
    report.push(
        "capability_provenance",
        capability_passed,
        format!(
            "target={}-{} kernel_present={} fabric={:?} pinned_host_staging={:?} gpu_direct_rdma={:?} rdma_core_loaded={} mlx5_core_loaded={} peer_memory_module={} topology_cpu_count={}",
            capabilities.target_os,
            capabilities.target_arch,
            capabilities.kernel_release.is_some(),
            capabilities.fabric,
            capabilities.pinned_host_staging,
            capabilities.gpu_direct_rdma,
            capabilities.rdma_core_loaded,
            capabilities.mlx5_core_loaded,
            capabilities
                .nvidia_peer_memory_module
                .as_deref()
                .unwrap_or("none"),
            capabilities.topology.cpu_count,
        ),
    );

    let (audit_passed, audit_details) = audit::audit_acceptance();
    report.push("vllm_rvllm_audit", audit_passed, audit_details);

    let cuda_smoke = nerva_runtime::capabilities::discovery::cuda_smoke();
    let cuda_smoke_passed = format!("{:?}", cuda_smoke.status) == "Ok"
        && cuda_smoke.kernel_value == Some(0x4e45_5256)
        && cuda_smoke.hot_path_allocations == 0;
    report.push(
        "cuda_runtime_smoke",
        cuda_smoke_passed,
        format!(
            "status={:?} gpu={} cc={}.{} memory_bytes={} pci_bus_id={} value={} hot_path_allocations={} error={}",
            cuda_smoke.status,
            cuda_smoke.gpu_name.as_deref().unwrap_or("none"),
            cuda_smoke
                .compute_capability_major
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke
                .compute_capability_minor
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke
                .device_total_memory_bytes
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke.pci_bus_id.as_deref().unwrap_or("none"),
            cuda_smoke
                .kernel_value
                .map_or_else(|| "none".to_string(), |value| format!("0x{value:08x}")),
            cuda_smoke.hot_path_allocations,
            cuda_smoke.error.as_deref().unwrap_or("none"),
        ),
    );

    let cuda_graph = nerva_runtime::engine::cuda::cuda_synthetic_graph_smoke(1024, 64, 1);
    let cuda_graph_passed = format!("{:?}", cuda_graph.status) == "Ok"
        && cuda_graph.steps == 1024
        && cuda_graph.ring_capacity == 64
        && cuda_graph.last_token == Some(1025)
        && cuda_graph.graph_replays == 1024
        && cuda_graph.graph_launches == 1024
        && cuda_graph.graph_nodes >= 2
        && cuda_graph.observed_tokens == 1024
        && cuda_graph.observed_token_hash != 0
        && cuda_graph.token_ring_slots_touched == 64
        && cuda_graph.token_ring_reuses == 960
        && cuda_graph.token_ring_max_slot_version == 16
        && cuda_graph.sync_calls == 1024
        && cuda_graph.d2h_bytes > 0
        && cuda_graph.device_arena_bytes > 0
        && cuda_graph.pinned_host_bytes > 0
        && cuda_graph.hot_path_allocations == 0
        && cuda_graph.stale_tokens == 0
        && cuda_graph.missing_tokens == 0
        && cuda_graph.extra_tokens == 0
        && cuda_graph.mismatched_tokens == 0
        && cuda_graph.host_causality_edges == 0;
    report.push(
        "cuda_graph_transaction",
        cuda_graph_passed,
        format!(
            "status={:?} steps={} ring_capacity={} graph_replays={} graph_launches={} graph_nodes={} observed={} observed_token_hash={} ring_slots={} ring_reuses={} ring_max_version={} sync_calls={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} hot_path_allocations={} stale={} missing={} extra={} mismatched={} host_causality_edges={} error={}",
            cuda_graph.status,
            cuda_graph.steps,
            cuda_graph.ring_capacity,
            cuda_graph.graph_replays,
            cuda_graph.graph_launches,
            cuda_graph.graph_nodes,
            cuda_graph.observed_tokens,
            cuda_graph.observed_token_hash,
            cuda_graph.token_ring_slots_touched,
            cuda_graph.token_ring_reuses,
            cuda_graph.token_ring_max_slot_version,
            cuda_graph.sync_calls,
            cuda_graph.d2h_bytes,
            cuda_graph.device_arena_bytes,
            cuda_graph.pinned_host_bytes,
            cuda_graph.hot_path_allocations,
            cuda_graph.stale_tokens,
            cuda_graph.missing_tokens,
            cuda_graph.extra_tokens,
            cuda_graph.mismatched_tokens,
            cuda_graph.host_causality_edges,
            cuda_graph.error.as_deref().unwrap_or("none"),
        ),
    );

    let cuda_sampler = nerva_runtime::engine::cuda::cuda_greedy_sampler_smoke();
    let cuda_sampler_passed = format!("{:?}", cuda_sampler.status) == "Ok"
        && cuda_sampler.vocab_size == 4
        && cuda_sampler.token_index == 0
        && cuda_sampler.token == 2
        && cuda_sampler.slot_version == 1
        && cuda_sampler.completion == 1
        && cuda_sampler.h2d_bytes == 16
        && cuda_sampler.d2h_bytes > 0
        && cuda_sampler.device_arena_bytes > 0
        && cuda_sampler.pinned_host_bytes > 0
        && cuda_sampler.kernel_launches == 1
        && cuda_sampler.sync_calls == 2
        && cuda_sampler.hot_path_allocations == 0;
    report.push(
        "cuda_device_sampler",
        cuda_sampler_passed,
        format!(
            "status={:?} vocab_size={} token_index={} token={} slot_version={} completion={} H2D_bytes={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
            cuda_sampler.status,
            cuda_sampler.vocab_size,
            cuda_sampler.token_index,
            cuda_sampler.token,
            cuda_sampler.slot_version,
            cuda_sampler.completion,
            cuda_sampler.h2d_bytes,
            cuda_sampler.d2h_bytes,
            cuda_sampler.device_arena_bytes,
            cuda_sampler.pinned_host_bytes,
            cuda_sampler.kernel_launches,
            cuda_sampler.sync_calls,
            cuda_sampler.hot_path_allocations,
            cuda_sampler.error.as_deref().unwrap_or("none"),
        ),
    );

    match runtime.static_arena_probe(ResidencyBudget::new(1024, 2048, 4096)) {
        Ok(summary) => report.push(
            "static_arenas",
            summary.device_capacity_bytes == 1024
                && summary.pinned_host_capacity_bytes == 2048
                && summary.host_capacity_bytes == 4096
                && summary.bootstrap_blocks == 3
                && summary.ready_blocks == 3
                && summary.hot_path_rejections == 3
                && summary.hot_path_allocation_attempts == 3
                && summary.usage_preserved_after_rejections,
            format!(
                "device_capacity={} pinned_host_capacity={} host_capacity={} device_used={} pinned_host_used={} host_used={} bootstrap_blocks={} ready_blocks={} hot_path_rejections={} hot_path_allocation_attempts={} usage_preserved={}",
                summary.device_capacity_bytes,
                summary.pinned_host_capacity_bytes,
                summary.host_capacity_bytes,
                summary.device_used_bytes,
                summary.pinned_host_used_bytes,
                summary.host_used_bytes,
                summary.bootstrap_blocks,
                summary.ready_blocks,
                summary.hot_path_rejections,
                summary.hot_path_allocation_attempts,
                summary.usage_preserved_after_rejections,
            ),
        ),
        Err(err) => report.push("static_arenas", false, format!("{err:?}")),
    }

    let topology = runtime.discover_topology();
    report.push(
        "topology_snapshot",
        topology.cpu_count > 0
            && topology.numa_node_count > 0
            && topology.pci_device_count >= topology.pci_gpu_count
            && topology.pci_device_count >= topology.pci_network_count
            && topology.pci_device_count >= topology.pci_nvme_count
            && (topology.pci_root_complex_count == 0
                || topology.pci_bus_count >= topology.pci_root_complex_count)
            && topology.block_device_count >= topology.nvme_block_device_count
            && topology.rdma_device_count == topology.rdma_device_names.len(),
        format!(
            "cpu_count={} numa_nodes={} pci_devices={} pci_roots={} pci_buses={} pci_gpu={} pci_network={} pci_nvme={} block_devices={} nvme_block_devices={} rdma_devices={} rdma_links={} iommu_groups={} iommu_mode={}",
            topology.cpu_count,
            topology.numa_node_count,
            topology.pci_device_count,
            topology.pci_root_complex_count,
            topology.pci_bus_count,
            topology.pci_gpu_count,
            topology.pci_network_count,
            topology.pci_nvme_count,
            topology.block_device_count,
            topology.nvme_block_device_count,
            topology.rdma_device_count,
            topology.rdma_netdev_links.join("|"),
            topology.iommu_group_count,
            topology.iommu_mode,
        ),
    );

    match runtime.run_synthetic_decode(SyntheticDecodeConfig::new(1024, 64, TokenId(1))) {
        Ok(summary) => {
            let transaction_passed = matches!(summary.status, SyntheticDecodeStatus::Ok)
                && summary.steps == 1024
                && summary.graph_replays == summary.steps
                && summary.graph_replay_events == summary.steps
                && summary.kernel_events >= summary.steps
                && summary.device_events == summary.steps
                && summary.copy_events == summary.steps
                && summary.host_wait_events == summary.steps
                && summary.graph_replay_latency_ns > 0
                && summary.device_latency_ns > 0
                && summary.copy_latency_ns > 0
                && summary.host_wait_latency_ns > 0
                && summary.hot_path_allocations == 0;
            report.push(
                "synthetic_transaction",
                transaction_passed,
                format!(
                    "steps={} graph_replays={} graph_events={} kernel_events={} device_events={} copy_events={} host_wait_events={} graph_ns={} device_ns={} copy_ns={} host_wait_ns={} hot_path_allocations={}",
                    summary.steps,
                    summary.graph_replays,
                    summary.graph_replay_events,
                    summary.kernel_events,
                    summary.device_events,
                    summary.copy_events,
                    summary.host_wait_events,
                    summary.graph_replay_latency_ns,
                    summary.device_latency_ns,
                    summary.copy_latency_ns,
                    summary.host_wait_latency_ns,
                    summary.hot_path_allocations,
                ),
            );

            let passed = matches!(summary.status, SyntheticDecodeStatus::Ok)
                && summary.steps == 1024
                && summary.graph_replays == 1024
                && summary.observed_tokens == 1024
                && summary.observed_token_hash != 0
                && summary.token_ring_slots_touched == 64
                && summary.token_ring_reuses == 960
                && summary.token_ring_max_slot_version == 16
                && summary.soft_visibility_syncs == 1024
                && summary.device_timeline_active_ns > 0
                && summary.device_timeline_idle_ns == 0
                && summary.hot_path_allocations == 0
                && summary.stale_tokens == 0
                && summary.missing_tokens == 0
                && summary.extra_tokens == 0
                && summary.mismatched_tokens == 0
                && summary.host_causality_edges == 0;
            report.push(
                "synthetic_device_token",
                passed,
                format!(
                    "steps={} observed={} observed_token_hash={} ring_slots={} ring_reuses={} ring_max_version={} soft_visibility_syncs={} hot_path_allocations={} stale={} missing={} extra={} mismatched={} host_causality_edges={} gpu_idle_ns={}",
                    summary.steps,
                    summary.observed_tokens,
                    summary.observed_token_hash,
                    summary.token_ring_slots_touched,
                    summary.token_ring_reuses,
                    summary.token_ring_max_slot_version,
                    summary.soft_visibility_syncs,
                    summary.hot_path_allocations,
                    summary.stale_tokens,
                    summary.missing_tokens,
                    summary.extra_tokens,
                    summary.mismatched_tokens,
                    summary.host_causality_edges,
                    summary.device_timeline_idle_ns,
                ),
            );
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("synthetic_transaction", false, details.clone());
            report.push("synthetic_device_token", false, details);
        }
    }

    match nerva_model::reference::smoke::reference_block_smoke() {
        Ok(summary) => report.push(
            "reference_block",
            summary.hot_path_allocations == 0,
            format!(
                "hidden={} heads={} output_hash={} hot_path_allocations={}",
                summary.hidden, summary.heads, summary.output_hash, summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("reference_block", false, format!("{err:?}")),
    }

    match nerva_model::precision::smoke::precision_block_smoke() {
        Ok(summary) => report.push(
            "fp16_bf16_precision_block",
            summary.passed(),
            format!(
                "f16_hash={} bf16_hash={} f16_expected_hash={} bf16_expected_hash={} f16_max_abs_error={} bf16_max_abs_error={} f16_hot_path_allocations={} bf16_hot_path_allocations={}",
                summary.f16.output_hash,
                summary.bf16.output_hash,
                summary.f16.expected_hash,
                summary.bf16.expected_hash,
                summary.f16.max_abs_error,
                summary.bf16.max_abs_error,
                summary.f16.hot_path_allocations,
                summary.bf16.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("fp16_bf16_precision_block", false, format!("{err:?}")),
    }

    match nerva_model::precision::file_smoke::precision_block_from_safetensors_smoke() {
        Ok(summary) => {
            report.push(
                "safetensors_precision_block",
                summary.passed(),
                format!(
                    "tensors_loaded={} bytes_loaded={} data_hash={} output_hash={} expected_hash={} bit_parity={} hot_path_allocations={}",
                    summary.tensors_loaded,
                    summary.bytes_loaded,
                    summary.data_hash,
                    summary.output_hash,
                    summary.expected_hash,
                    summary.bit_parity,
                    summary.hot_path_allocations,
                ),
            );

            let cuda_block = nerva_runtime::engine::cuda::cuda_tiny_block_smoke();
            let cuda_block_passed = format!("{:?}", cuda_block.status) == "Ok"
                && cuda_block.hidden == summary.hidden as u32
                && cuda_block.intermediate == summary.intermediate as u32
                && cuda_block.output_hash == summary.expected_hash
                && cuda_block.kernel_launches == 1
                && cuda_block.sync_calls == 1
                && cuda_block.d2h_bytes == 4
                && cuda_block.device_arena_bytes == 4
                && cuda_block.pinned_host_bytes == 4
                && cuda_block.hot_path_allocations == 0;
            report.push(
                "cuda_real_block",
                cuda_block_passed,
                format!(
                    "status={:?} hidden={} intermediate={} output_hash={} expected_hash={} output_bits=[{},{}] kernel_launches={} sync_calls={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} hot_path_allocations={} error={}",
                    cuda_block.status,
                    cuda_block.hidden,
                    cuda_block.intermediate,
                    cuda_block.output_hash,
                    summary.expected_hash,
                    cuda_block.output[0],
                    cuda_block.output[1],
                    cuda_block.kernel_launches,
                    cuda_block.sync_calls,
                    cuda_block.d2h_bytes,
                    cuda_block.device_arena_bytes,
                    cuda_block.pinned_host_bytes,
                    cuda_block.hot_path_allocations,
                    cuda_block.error.as_deref().unwrap_or("none"),
                ),
            );

            let cuda_resident_block = nerva_runtime::engine::cuda::cuda_loaded_tiny_block_smoke();
            let cuda_resident_block_passed = format!("{:?}", cuda_resident_block.status) == "Ok"
                && cuda_resident_block.hidden == summary.hidden as u32
                && cuda_resident_block.intermediate == summary.intermediate as u32
                && cuda_resident_block.output_hash == summary.expected_hash
                && cuda_resident_block.output == cuda_block.output
                && cuda_resident_block.resident_weight_bytes == summary.bytes_loaded as u64
                && cuda_resident_block.device_arena_bytes
                    >= cuda_resident_block.resident_weight_bytes + 8
                && cuda_resident_block.pinned_host_bytes == cuda_resident_block.device_arena_bytes
                && cuda_resident_block.h2d_bytes == cuda_resident_block.device_arena_bytes
                && cuda_resident_block.d2h_bytes == 4
                && cuda_resident_block.kernel_launches == 1
                && cuda_resident_block.sync_calls == 2
                && cuda_resident_block.hot_path_allocations == 0;
            report.push(
                "cuda_resident_block",
                cuda_resident_block_passed,
                format!(
                    "status={:?} hidden={} intermediate={} output_hash={} expected_hash={} output_bits=[{},{}] resident_weight_bytes={} H2D_bytes={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
                    cuda_resident_block.status,
                    cuda_resident_block.hidden,
                    cuda_resident_block.intermediate,
                    cuda_resident_block.output_hash,
                    summary.expected_hash,
                    cuda_resident_block.output[0],
                    cuda_resident_block.output[1],
                    cuda_resident_block.resident_weight_bytes,
                    cuda_resident_block.h2d_bytes,
                    cuda_resident_block.d2h_bytes,
                    cuda_resident_block.device_arena_bytes,
                    cuda_resident_block.pinned_host_bytes,
                    cuda_resident_block.kernel_launches,
                    cuda_resident_block.sync_calls,
                    cuda_resident_block.hot_path_allocations,
                    cuda_resident_block.error.as_deref().unwrap_or("none"),
                ),
            );
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("safetensors_precision_block", false, details.clone());
            report.push(
                "cuda_real_block",
                false,
                format!("canonical precision block prerequisite failed: {details}"),
            );
            report.push(
                "cuda_resident_block",
                false,
                format!("canonical precision block prerequisite failed: {details}"),
            );
        }
    }

    match nerva_model::tiny::smoke::tiny_greedy_decode_smoke(8) {
        Ok(summary) => {
            report.push(
                "tiny_model_greedy_parity",
                summary.parity
                    && summary.ledger_count == summary.steps as u64
                    && summary.hot_path_allocations == 0,
                format!(
                    "steps={} parity={} ledger_count={} device_events={} hot_path_allocations={} output_hash={}",
                    summary.steps,
                    summary.parity,
                    summary.ledger_count,
                    summary.device_events,
                    summary.hot_path_allocations,
                    summary.output_hash,
                ),
            );

            let cuda_decode =
                nerva_runtime::engine::cuda::cuda_tiny_decode_smoke(summary.steps as u32, 4, 0);
            let cuda_decode_passed = format!("{:?}", cuda_decode.status) == "Ok"
                && cuda_decode.steps == summary.steps as u32
                && cuda_decode.ring_capacity == 4
                && cuda_decode.seed_token == summary.seed_token.0
                && cuda_decode.vocab_size == summary.vocab_size as u32
                && cuda_decode.hidden == 2
                && cuda_decode.last_token == summary.tokens.last().map(|token| token.0)
                && cuda_decode.graph_replays == summary.steps as u64
                && cuda_decode.graph_launches == summary.steps as u64
                && cuda_decode.kernel_launches == summary.steps as u64
                && cuda_decode.observed_tokens == summary.steps as u64
                && cuda_decode.observed_token_hash == summary.output_hash
                && cuda_decode.token_ring_slots_touched == 4
                && cuda_decode.token_ring_reuses == 4
                && cuda_decode.token_ring_max_slot_version == 2
                && cuda_decode.resident_weight_bytes == 64
                && cuda_decode.h2d_bytes >= cuda_decode.resident_weight_bytes
                && cuda_decode.d2h_bytes > 0
                && cuda_decode.sync_calls == summary.steps as u64
                && cuda_decode.hot_path_allocations == 0
                && cuda_decode.stale_tokens == 0
                && cuda_decode.missing_tokens == 0
                && cuda_decode.extra_tokens == 0
                && cuda_decode.mismatched_tokens == 0
                && cuda_decode.host_causality_edges == 0;
            report.push(
                "cuda_tiny_decode_model",
                cuda_decode_passed,
                format!(
                    "status={:?} steps={} ring_capacity={} graph_replays={} graph_nodes={} observed={} observed_token_hash={} reference_hash={} last_token={:?} ring_slots={} ring_reuses={} ring_max_version={} resident_weight_bytes={} H2D_bytes={} D2H_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} stale={} missing={} extra={} mismatched={} host_causality_edges={} error={}",
                    cuda_decode.status,
                    cuda_decode.steps,
                    cuda_decode.ring_capacity,
                    cuda_decode.graph_replays,
                    cuda_decode.graph_nodes,
                    cuda_decode.observed_tokens,
                    cuda_decode.observed_token_hash,
                    summary.output_hash,
                    cuda_decode.last_token,
                    cuda_decode.token_ring_slots_touched,
                    cuda_decode.token_ring_reuses,
                    cuda_decode.token_ring_max_slot_version,
                    cuda_decode.resident_weight_bytes,
                    cuda_decode.h2d_bytes,
                    cuda_decode.d2h_bytes,
                    cuda_decode.kernel_launches,
                    cuda_decode.sync_calls,
                    cuda_decode.hot_path_allocations,
                    cuda_decode.stale_tokens,
                    cuda_decode.missing_tokens,
                    cuda_decode.extra_tokens,
                    cuda_decode.mismatched_tokens,
                    cuda_decode.host_causality_edges,
                    cuda_decode.error.as_deref().unwrap_or("none"),
                ),
            );
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("tiny_model_greedy_parity", false, details.clone());
            report.push(
                "cuda_tiny_decode_model",
                false,
                format!("tiny reference model prerequisite failed: {details}"),
            );
        }
    }

    match vllm::vllm_token_identity_acceptance() {
        Ok((passed, details)) => report.push("vllm_token_identity_parity", passed, details),
        Err(err) => report.push("vllm_token_identity_parity", false, err),
    }

    match manifest::model_manifest_acceptance() {
        Ok((passed, details)) => report.push("hf_model_manifest", passed, details),
        Err(err) => report.push("hf_model_manifest", false, err),
    }

    match files::safetensors_file_header_acceptance() {
        Ok((passed, details)) => report.push("safetensors_file_header", passed, details),
        Err(err) => report.push("safetensors_file_header", false, err),
    }

    match files::safetensors_file_prefetch_acceptance() {
        Ok((passed, details)) => report.push("safetensors_file_prefetch", passed, details),
        Err(err) => report.push("safetensors_file_prefetch", false, err),
    }

    match nerva_model::attention::smoke::blockwise_attention_smoke() {
        Ok(summary) => report.push(
            "tiered_blockwise_attention",
            summary.cpu_block_events > 0
                && summary.device_block_events > 0
                && summary.hot_path_allocations == 0,
            format!(
                "blocks={} tokens={} cpu_block_events={} device_block_events={} hot_path_allocations={} output_hash={}",
                summary.blocks,
                summary.tokens,
                summary.cpu_block_events,
                summary.device_block_events,
                summary.hot_path_allocations,
                summary.output_hash,
            ),
        ),
        Err(err) => report.push("tiered_blockwise_attention", false, format!("{err:?}")),
    }

    match nerva_model::warm_compute::probe::warm_compute_probe() {
        Ok(summary) => report.push(
            "warm_compute_selection",
            summary.parity
                && summary.cpu_beats_staged
                && summary.execution_decisions > 0
                && summary.hot_path_allocations == 0,
            format!(
                "selected_strategy={} parity={} cpu_beats_staged={} execution_decisions={} hot_path_allocations={} output_hash={}",
                summary.selected_strategy.label(),
                summary.parity,
                summary.cpu_beats_staged,
                summary.execution_decisions,
                summary.hot_path_allocations,
                summary.output_hash,
            ),
        ),
        Err(err) => report.push("warm_compute_selection", false, format!("{err:?}")),
    }

    match nerva_kernel_contracts::registry::kernel_registry_probe() {
        Ok(summary) => report.push(
            "kernel_contract_fallbacks",
            summary.direct_plans > 0
                && summary.fallback_plans > 0
                && summary.rejected_plans > 0
                && summary.exact_fallbacks > 0,
            format!(
                "implementations={} fallbacks={} direct_plans={} fallback_plans={} rejected_plans={} exact_fallbacks={}",
                summary.implementations,
                summary.fallbacks,
                summary.direct_plans,
                summary.fallback_plans,
                summary.rejected_plans,
                summary.exact_fallbacks,
            ),
        ),
        Err(err) => report.push("kernel_contract_fallbacks", false, format!("{err:?}")),
    }

    match runtime.run_kv_residency_probe(KvResidencyProbeConfig::default()) {
        Ok(summary) => report.push(
            "kv_residency_tiering",
            matches!(summary.status, KvResidencyProbeStatus::Ok)
                && summary.decisions > 0
                && summary.prefetches > 0
                && summary.demotions > 0
                && summary.evictions > 0
                && summary.stall_events > 0
                && summary.hot_path_allocations == 0,
            format!(
                "pages={} decisions={} prefetches={} demotions={} evictions={} stall_events={} hot_path_allocations={}",
                summary.pages,
                summary.decisions,
                summary.prefetches,
                summary.demotions,
                summary.evictions,
                summary.stall_events,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("kv_residency_tiering", false, format!("{err:?}")),
    }

    match runtime.run_transport_path_probe() {
        Ok(summary) => report.push(
            "transport_pinned_fallback",
            matches!(summary.status, TransportPathProbeStatus::Ok)
                && summary.requests > 0
                && summary.pinned_host_paths > 0
                && summary.fallback_decisions > 0
                && summary.phase_handoff_syncs > 0
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "requests={} pinned_host_paths={} fallback_decisions={} phase_handoff_syncs={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.requests,
                summary.pinned_host_paths,
                summary.fallback_decisions,
                summary.phase_handoff_syncs,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("transport_pinned_fallback", false, format!("{err:?}")),
    }

    match runtime.run_transport_capability_matrix_probe() {
        Ok(summary) => report.push(
            "transport_capability_matrix",
            matches!(summary.status, TransportCapabilityMatrixStatus::Ok)
                && summary.sizes == 6
                && summary.entries.len() == 24
                && summary.degraded_to_pinned_host_entries > 0
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.registration_cache_hits == summary.entries.len() as u64
                && summary.estimated_cpu_core_ns > 0
                && summary.pcie_tx_bytes > 0
                && summary.pcie_rx_bytes > 0
                && summary.credit_stall_ns == 0
                && summary.hot_path_allocations == 0,
            format!(
                "sizes={} entries={} host_staged={} gpu_direct={} degraded_to_pinned_host={} p95_estimated_visible_ns={} cpu_core_ns={} pcie_tx_bytes={} pcie_rx_bytes={} registration_cache_hits={} pageable_copies={} per_token_registrations={} credit_stall_ns={} hot_path_allocations={}",
                summary.sizes,
                summary.entries.len(),
                summary.host_staged_entries,
                summary.gpu_direct_entries,
                summary.degraded_to_pinned_host_entries,
                summary.p95_estimated_visible_ns,
                summary.estimated_cpu_core_ns,
                summary.pcie_tx_bytes,
                summary.pcie_rx_bytes,
                summary.registration_cache_hits,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.credit_stall_ns,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("transport_capability_matrix", false, format!("{err:?}")),
    }

    match resident_weights::resident_weight_execution_acceptance(&runtime) {
        Ok((passed, details)) => report.push("resident_weight_execution", passed, details),
        Err(err) => report.push("resident_weight_execution", false, err),
    }

    Ok(report)
}

pub(crate) fn run_acceptance_probe() -> Result<String, String> {
    build_acceptance_report().map(|report| report.to_json())
}
