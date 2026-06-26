use std::{fs, path::Path};

use nerva_core::types::{DType, MemoryFabricKind, TokenId};
use nerva_runtime::capabilities::CapabilityState;
use nerva_runtime::engine::{
    KvResidencyProbeConfig, KvResidencyProbeStatus, ResidencyBudget, Runtime, RuntimeConfig,
    SyntheticDecodeConfig, SyntheticDecodeStatus,
};
use nerva_runtime::transport::{TransportCapabilityMatrixStatus, TransportPathProbeStatus};

use crate::{json::json_escape, parity::compare_vllm_token_identity};

#[derive(Clone, Debug, Eq, PartialEq)]
struct AcceptanceCheck {
    name: &'static str,
    passed: bool,
    details: String,
}

impl AcceptanceCheck {
    fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"passed\":{},\"details\":\"{}\"}}",
            self.name,
            self.passed,
            json_escape(&self.details),
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct AcceptanceReport {
    checks: Vec<AcceptanceCheck>,
}

impl AcceptanceReport {
    fn push(&mut self, name: &'static str, passed: bool, details: impl Into<String>) {
        self.checks.push(AcceptanceCheck {
            name,
            passed,
            details: details.into(),
        });
    }

    pub(crate) fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|check| check.passed)
    }

    fn passed_count(&self) -> usize {
        self.checks.iter().filter(|check| check.passed).count()
    }

    fn failed_count(&self) -> usize {
        self.checks.len() - self.passed_count()
    }

    pub(crate) fn to_json(&self) -> String {
        let mut items = String::from("[");
        for (index, check) in self.checks.iter().enumerate() {
            if index != 0 {
                items.push(',');
            }
            items.push_str(&check.to_json());
        }
        items.push(']');
        format!(
            "{{\"status\":\"{}\",\"acceptance_schema\":\"nerva-acceptance-v1\",\"checks\":{},\"passed\":{},\"failed\":{},\"items\":{}}}",
            if self.passed() { "ok" } else { "failed" },
            self.checks.len(),
            self.passed_count(),
            self.failed_count(),
            items,
        )
    }
}

const AUDIT_PATH: &str = "docs/audits/VLLM_RVLLM_ARCHITECTURE_AUDIT.md";
const REQUIRED_AUDIT_ROWS: &[&str] = &[
    "runtime language",
    "hot path owner",
    "request scheduler",
    "GPU context ownership",
    "graph capture/replay",
    "static arenas",
    "hot-path allocation",
    "token source of truth",
    "sampling",
    "host output handoff",
    "KV cache",
    "weight loading",
    "kernel contracts",
    "silent fallback behavior",
    "CUDA portability",
    "AMD/HIP portability",
    "model coverage",
    "old hardware viability",
    "exact FP16/BF16 viability",
    "DRAM warm-tier compute",
    "transport assumptions",
    "ResidentBlock compatibility",
];

fn audit_acceptance() -> (bool, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(AUDIT_PATH);
    let Ok(contents) = fs::read_to_string(&path) else {
        return (false, format!("missing audit file at {}", path.display()));
    };
    let required_sections = [
        "## vLLM Summary",
        "## rvLLM Summary",
        "| Area | vLLM | rvLLM | NERVA decision |",
        "## Required Questions",
    ];
    let section_hits = required_sections
        .iter()
        .filter(|section| contents.contains(**section))
        .count();
    let missing_rows = REQUIRED_AUDIT_ROWS
        .iter()
        .filter(|row| !audit_has_table_row(&contents, row))
        .copied()
        .collect::<Vec<_>>();
    let passed = section_hits == required_sections.len() && missing_rows.is_empty();
    let missing = if missing_rows.is_empty() {
        "none".to_string()
    } else {
        missing_rows.join("|")
    };
    (
        passed,
        format!(
            "path={} sections={}/{} required_rows={} missing_rows={}",
            AUDIT_PATH,
            section_hits,
            required_sections.len(),
            REQUIRED_AUDIT_ROWS.len(),
            missing,
        ),
    )
}

fn audit_has_table_row(contents: &str, row: &str) -> bool {
    contents
        .lines()
        .any(|line| line.trim_start().starts_with(&format!("| {row} |")))
}

fn model_manifest_acceptance() -> Result<(bool, String), String> {
    let metadata = nerva_model::hf::probe::hf_metadata_probe()
        .map_err(|err| format!("HF metadata probe failed: {err:?}"))?;
    let layout = nerva_model::weights::layout::hf_weight_layout_probe()
        .map_err(|err| format!("HF layout probe failed: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .map_err(|err| format!("HF manifest probe failed: {err:?}"))?;
    let safetensors = nerva_model::weights::safetensors::safetensors_header_probe()
        .map_err(|err| format!("safetensors header probe failed: {err:?}"))?;

    let metadata_body = &metadata.metadata;
    let expected_static_blocks = if metadata_body.tie_word_embeddings {
        1
    } else {
        2
    };
    let expected_blocks = metadata_body
        .num_hidden_layers
        .checked_mul(9)
        .and_then(|layer_blocks| layer_blocks.checked_add(expected_static_blocks))
        .ok_or_else(|| "expected HF block count overflowed".to_string())?;
    let metadata_passed = metadata_body.architecture.as_str() == "llama"
        && metadata_body.hidden_size == 4096
        && metadata_body.num_hidden_layers == 32
        && metadata_body.num_attention_heads == 32
        && metadata_body.num_key_value_heads == 8
        && metadata_body.torch_dtype == Some(DType::BF16)
        && metadata.metadata_hash != 0;
    let layout_passed = layout.plan.metadata == *metadata_body
        && layout.plan.blocks.len() == expected_blocks
        && layout.plan.total_weight_bytes > 0
        && layout.layout_hash != 0;
    let manifest_passed = manifest.manifest.entries.len() == layout.plan.blocks.len()
        && manifest.manifest.total_weight_bytes == layout.plan.total_weight_bytes
        && manifest.manifest.manifest_hash != 0;
    let validation = &safetensors.validation;
    let safetensors_passed = validation.manifest_entries == manifest.manifest.entries.len()
        && validation.validated_tensors == manifest.manifest.entries.len()
        && validation.total_data_bytes == manifest.manifest.total_weight_bytes
        && validation.manifest_hash == manifest.manifest.manifest_hash
        && validation.header_bytes > 0
        && validation.header_hash != 0;

    Ok((
        metadata_passed && layout_passed && manifest_passed && safetensors_passed,
        format!(
            "architecture={} layers={} hidden={} kv_heads={} dtype={:?} expected_blocks={} layout_blocks={} manifest_entries={} safetensors_validated={} total_weight_bytes={} metadata_hash={} layout_hash={} manifest_hash={} header_hash={}",
            metadata_body.architecture.as_str(),
            metadata_body.num_hidden_layers,
            metadata_body.hidden_size,
            metadata_body.num_key_value_heads,
            metadata_body.torch_dtype,
            expected_blocks,
            layout.plan.blocks.len(),
            manifest.manifest.entries.len(),
            validation.validated_tensors,
            manifest.manifest.total_weight_bytes,
            metadata.metadata_hash,
            layout.layout_hash,
            manifest.manifest.manifest_hash,
            validation.header_hash,
        ),
    ))
}

fn safetensors_file_header_acceptance() -> Result<(bool, String), String> {
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "torch_dtype": "float16"
        }"#;
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("HF layout probe failed: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout)
        .map_err(|err| format!("HF manifest probe failed: {err:?}"))?;
    let header =
        nerva_model::weights::safetensors::synthetic_safetensors_header_for_manifest(&manifest)
            .map_err(|err| format!("safetensors header generation failed: {err:?}"))?;
    let dir = std::env::temp_dir().join(format!(
        "nerva-acceptance-safetensors-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let path = dir.join("model.safetensors");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + manifest.total_weight_bytes, 0);
    fs::write(&path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))?;

    let file_header = nerva_model::weights::file::read_safetensors_header_file(&path)
        .map_err(|err| format!("safetensors header read failed: {err:?}"))?;
    let validation = nerva_model::weights::safetensors::validate_safetensors_header_for_manifest(
        &file_header.header_json,
        &manifest,
    )
    .map_err(|err| format!("safetensors manifest validation failed: {err:?}"))?;

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir(&dir);

    let passed = file_header.header_bytes == header.len()
        && file_header.data_start == 8 + header.len()
        && file_header.payload_bytes == manifest.total_weight_bytes as u64
        && validation.validated_tensors == manifest.entries.len()
        && validation.total_data_bytes == manifest.total_weight_bytes
        && validation.header_hash != 0;
    Ok((
        passed,
        format!(
            "manifest_entries={} validated_tensors={} header_bytes={} data_start={} payload_bytes={} total_data_bytes={} header_hash={}",
            manifest.entries.len(),
            validation.validated_tensors,
            file_header.header_bytes,
            file_header.data_start,
            file_header.payload_bytes,
            validation.total_data_bytes,
            validation.header_hash,
        ),
    ))
}

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

    let (audit_passed, audit_details) = audit_acceptance();
    report.push("vllm_rvllm_audit", audit_passed, audit_details);

    let cuda_smoke = nerva_runtime::capabilities::cuda_smoke();
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

    let cuda_graph = nerva_runtime::engine::cuda_synthetic_graph_smoke(1024, 64, 1);
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

    match nerva_model::tiny::tiny_greedy_decode_smoke(8) {
        Ok(summary) => report.push(
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
        ),
        Err(err) => report.push("tiny_model_greedy_parity", false, format!("{err:?}")),
    }

    match vllm_token_identity_acceptance() {
        Ok((passed, details)) => report.push("vllm_token_identity_parity", passed, details),
        Err(err) => report.push("vllm_token_identity_parity", false, err),
    }

    match model_manifest_acceptance() {
        Ok((passed, details)) => report.push("hf_model_manifest", passed, details),
        Err(err) => report.push("hf_model_manifest", false, err),
    }

    match safetensors_file_header_acceptance() {
        Ok((passed, details)) => report.push("safetensors_file_header", passed, details),
        Err(err) => report.push("safetensors_file_header", false, err),
    }

    match nerva_model::attention::blockwise_attention_smoke() {
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

    match nerva_model::warm_compute::warm_compute_probe() {
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

    match resident_weight_execution_acceptance(&runtime) {
        Ok((passed, details)) => report.push("resident_weight_execution", passed, details),
        Err(err) => report.push("resident_weight_execution", false, err),
    }

    Ok(report)
}

fn resident_weight_execution_acceptance(runtime: &Runtime) -> Result<(bool, String), String> {
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .map_err(|err| format!("HF tensor manifest probe failed: {err:?}"))?
        .manifest;
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(512 * 1024 * 1024, 0, manifest.total_weight_bytes),
        )
        .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
    let hotset = runtime
        .promote_resident_weight_hotset(&mut table, 512 * 1024 * 1024)
        .map_err(|err| format!("resident weight hotset promotion failed: {err:?}"))?;
    let plan = runtime
        .plan_resident_weight_execution(&table, 32, Some(89))
        .map_err(|err| format!("resident weight execution planning failed: {err:?}"))?;
    let run = runtime
        .execute_resident_weight_execution_plan(&table, &plan)
        .map_err(|err| format!("resident weight execution run failed: {err:?}"))?;

    let passed = hotset.promoted_blocks > 0
        && hotset.hot_path_allocations == 0
        && !plan.steps.is_empty()
        && plan.gpu_resident_steps > 0
        && plan.gpu_staged_steps > 0
        && plan.block_version_dependencies == plan.steps.len() as u64
        && plan.ledger.hot_path_allocations == 0
        && run.steps == plan.steps.len()
        && run.gpu_resident_steps == plan.gpu_resident_steps
        && run.gpu_staged_steps == plan.gpu_staged_steps
        && run.block_version_dependencies == run.steps as u64
        && run.hot_path_allocations == 0;

    Ok((
        passed,
        format!(
            "promoted_blocks={} plan_steps={} plan_gpu_resident={} plan_gpu_staged={} plan_fallbacks={} plan_block_versions={} run_steps={} run_gpu_resident={} run_gpu_staged={} run_fallbacks={} run_block_versions={} hot_path_allocations={}",
            hotset.promoted_blocks,
            plan.steps.len(),
            plan.gpu_resident_steps,
            plan.gpu_staged_steps,
            plan.fallback_decisions,
            plan.block_version_dependencies,
            run.steps,
            run.gpu_resident_steps,
            run.gpu_staged_steps,
            run.fallback_decisions,
            run.block_version_dependencies,
            hotset.hot_path_allocations
                + plan.ledger.hot_path_allocations
                + run.hot_path_allocations,
        ),
    ))
}

fn vllm_token_identity_acceptance() -> Result<(bool, String), String> {
    let vllm_style_json =
        r#"{"request_id":"nerva-m4-parity","outputs":[{"token_ids":[1,2,3,0,1,2,3,0]}]}"#;
    let summary = compare_vllm_token_identity(vllm_style_json, 8)?;
    Ok((
        summary.passed(),
        format!(
            "source_format={} steps={} matched={} mismatched={} missing={} extra={} first_mismatch={} vllm_hash={} nerva_hash={} hot_path_allocations={}",
            summary.source_format,
            summary.steps,
            summary.matched_tokens,
            summary.mismatched_tokens,
            summary.missing_tokens,
            summary.extra_tokens,
            summary
                .first_mismatch_index
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.vllm_token_hash,
            summary.nerva_token_hash,
            summary.hot_path_allocations,
        ),
    ))
}

pub(crate) fn run_acceptance_probe() -> Result<String, String> {
    build_acceptance_report().map(|report| report.to_json())
}
