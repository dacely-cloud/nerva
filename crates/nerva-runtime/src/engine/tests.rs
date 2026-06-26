use super::*;
use nerva_core::types::ResidentBlockId;

const SHARD_ONE: &str = "model-00001-of-00001.safetensors";

fn tiny_llama_manifest() -> nerva_model::weights::manifest::HfTensorManifest {
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap()
}

fn single_shard_index_json(manifest: &nerva_model::weights::manifest::HfTensorManifest) -> String {
    let mut out = format!(
        "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
        manifest.total_weight_bytes
    );
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&entry.name);
        out.push_str("\":\"");
        out.push_str(SHARD_ONE);
        out.push('"');
    }
    out.push_str("}}");
    out
}

fn tiny_shard_plan() -> (
    nerva_model::weights::safetensors::SafetensorsShardPlan,
    usize,
) {
    let manifest = tiny_llama_manifest();
    let index = single_shard_index_json(&manifest);
    let header =
        nerva_model::weights::safetensors::synthetic_safetensors_header_for_manifest(&manifest)
            .unwrap();
    let header_len = header.len();
    let plan = nerva_model::weights::safetensors::plan_safetensors_shards_for_manifest(
        &index,
        &[nerva_model::weights::safetensors::SafetensorsShardHeader::new(SHARD_ONE, &header)],
        &manifest,
    )
    .unwrap();
    (plan, header_len)
}

#[test]
fn runtime_uses_device_zero_by_default() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    assert_eq!(runtime.config().device, DeviceOrdinal(0));
    assert_eq!(runtime.empty_token_ledger(9).token_index, 9);
}

#[test]
fn capability_snapshot_reports_conservative_discrete_profile() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let snapshot = runtime.discover_capabilities();

    assert_eq!(snapshot.host_arch, host_arch());
    assert_eq!(snapshot.target_os, env::consts::OS);
    assert_eq!(snapshot.target_arch, env::consts::ARCH);
    assert!(
        snapshot
            .kernel_release
            .as_deref()
            .is_none_or(|value| !value.is_empty())
    );
    assert_eq!(snapshot.fabric, MemoryFabricKind::DiscreteExplicit);
    assert!(matches!(
        snapshot.cuda,
        CapabilityState::SupportedAndVerified | CapabilityState::Unsupported
    ));
    assert_eq!(snapshot.hip, CapabilityState::Unsupported);
    assert_eq!(
        snapshot.pinned_host_staging,
        CapabilityState::SupportedUnverified
    );
    assert!(matches!(
        snapshot.gpu_direct_rdma,
        CapabilityState::DegradedToPinnedHost | CapabilityState::SupportedUnverified
    ));
    assert_eq!(snapshot.amd_peerdirect, CapabilityState::Unsupported);
    assert_eq!(snapshot.dma_buf_export, CapabilityState::Unsupported);
    assert_eq!(snapshot.cxl, CapabilityState::Unsupported);
    assert!(snapshot.topology.cpu_count > 0);

    let json = snapshot.to_json();
    assert!(json.contains("\"target_os\":\"linux\""));
    assert!(json.contains("\"kernel_release\""));
    assert!(json.contains("\"fabric\":\"DiscreteExplicit\""));
    assert!(json.contains("\"cuda_compute_capability\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\""));
    assert!(json.contains("\"cuda_pci_bus_id\""));
    assert!(json.contains("\"rdma_core_loaded\""));
    assert!(json.contains("\"mlx5_core_loaded\""));
    assert!(json.contains("\"nvidia_peer_memory_module\""));
    assert!(json.contains("\"gpu_direct_rdma\""));
    assert!(json.contains("\"topology\""));
    assert!(json.contains("\"cpu_count\""));
}

#[test]
fn capability_snapshot_json_escapes_cuda_error() {
    let snapshot = CapabilitySnapshot {
        host_arch: HostArch::X86_64,
        target_os: "linux",
        target_arch: "x86_64",
        kernel_release: Some("kernel\" release".to_string()),
        fabric: MemoryFabricKind::DiscreteExplicit,
        cuda: CapabilityState::Unsupported,
        cuda_status: "failed",
        cuda_error: Some("quote\" slash\\ newline\n".to_string()),
        cuda_visible_devices: Some("0,1".to_string()),
        cuda_compute_capability: Some("8.9".to_string()),
        cuda_device_total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
        cuda_pci_bus_id: Some("0000:65:00.0".to_string()),
        hip: CapabilityState::Unsupported,
        hip_visible_devices: Some("2".to_string()),
        nvidia_driver_version: Some("driver\\version".to_string()),
        rdma_core_loaded: true,
        mlx5_core_loaded: true,
        nvidia_peer_memory_module: Some("nvidia_peermem".to_string()),
        pinned_host_staging: CapabilityState::SupportedUnverified,
        gpu_direct_rdma: CapabilityState::SupportedUnverified,
        amd_peerdirect: CapabilityState::Unsupported,
        dma_buf_export: CapabilityState::Unsupported,
        cxl: CapabilityState::Unsupported,
        topology: TopologySnapshot {
            cpu_online: Some("0-1".to_string()),
            cpu_count: 2,
            numa_node_count: 1,
            pci_device_count: 3,
            pci_root_complex_count: 1,
            pci_bus_count: 2,
            pci_gpu_count: 1,
            pci_network_count: 1,
            pci_nvme_count: 1,
            block_device_count: 2,
            nvme_block_device_count: 1,
            rdma_device_count: 1,
            rdma_device_names: vec!["mlx5_0".to_string()],
            rdma_netdev_links: vec!["mlx5_0:enp1s0f0".to_string()],
            iommu_group_count: 3,
            iommu_mode: "passthrough_groups_present".to_string(),
            iommu_kernel_args: Some("intel_iommu=on iommu=pt".to_string()),
        },
    };

    let json = snapshot.to_json();
    assert!(json.contains("quote\\\" slash\\\\ newline\\n"));
    assert!(json.contains("kernel\\\" release"));
    assert!(json.contains("driver\\\\version"));
    assert!(json.contains("\"cuda_compute_capability\":\"8.9\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\":25769803776"));
    assert!(json.contains("\"cuda_pci_bus_id\":\"0000:65:00.0\""));
    assert!(json.contains("\"rdma_core_loaded\":true"));
    assert!(json.contains("\"mlx5_core_loaded\":true"));
    assert!(json.contains("\"nvidia_peer_memory_module\":\"nvidia_peermem\""));
    assert!(json.contains("\"gpu_direct_rdma\":\"SUPPORTED_UNVERIFIED\""));
    assert!(json.contains("\"cpu_online\":\"0-1\""));
    assert!(json.contains("\"pci_root_complex_count\":1"));
    assert!(json.contains("\"pci_bus_count\":2"));
    assert!(json.contains("\"rdma_device_names\":[\"mlx5_0\"]"));
    assert!(json.contains("\"rdma_netdev_links\":[\"mlx5_0:enp1s0f0\"]"));
    assert!(json.contains("\"iommu_mode\":\"passthrough_groups_present\""));
    assert!(json.contains("\"iommu_kernel_args\":\"intel_iommu=on iommu=pt\""));
}

#[test]
fn topology_snapshot_reports_basic_sysfs_counts() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let snapshot = runtime.discover_topology();

    assert!(snapshot.cpu_count > 0);
    assert!(snapshot.numa_node_count > 0);
    assert!(snapshot.pci_device_count >= snapshot.pci_gpu_count);
    assert!(snapshot.pci_device_count >= snapshot.pci_network_count);
    assert!(snapshot.pci_device_count >= snapshot.pci_nvme_count);
    if snapshot.pci_root_complex_count > 0 {
        assert!(snapshot.pci_bus_count >= snapshot.pci_root_complex_count);
    }
    assert!(snapshot.block_device_count >= snapshot.nvme_block_device_count);
    assert_eq!(snapshot.rdma_device_count, snapshot.rdma_device_names.len());
    assert!(snapshot.rdma_netdev_links.len() >= snapshot.rdma_device_names.len());
    assert!(!snapshot.iommu_mode.is_empty());
    let json = snapshot.to_json();
    assert!(json.contains("\"cpu_count\""));
    assert!(json.contains("\"pci_device_count\""));
    assert!(json.contains("\"pci_root_complex_count\""));
    assert!(json.contains("\"pci_bus_count\""));
    assert!(json.contains("\"rdma_device_names\""));
    assert!(json.contains("\"rdma_netdev_links\""));
    assert!(json.contains("\"iommu_mode\""));
}

#[test]
fn topology_helpers_parse_linux_id_and_pci_class_values() {
    assert_eq!(count_linux_id_list("0-3"), Some(4));
    assert_eq!(count_linux_id_list("0-1,4,8-9"), Some(5));
    assert_eq!(count_linux_id_list("2-1"), None);
    assert_eq!(parse_pci_class("0x030000"), Some(0x030000));
    assert_eq!(parse_pci_class("010802"), Some(0x010802));
    assert_eq!(parse_pci_class("not-hex"), None);
    assert_eq!(
        json_string_array(&["a\"b".to_string(), "c\\d".to_string()]),
        "[\"a\\\"b\",\"c\\\\d\"]"
    );
    assert_eq!(
        extract_iommu_kernel_args("root=/dev/sda intel_iommu=on quiet iommu=pt"),
        Some("intel_iommu=on iommu=pt".to_string())
    );
    assert_eq!(extract_iommu_kernel_args("root=/dev/sda quiet"), None);
    assert_eq!(
        discover_iommu_mode(9, Some("intel_iommu=on iommu=pt")),
        "passthrough_groups_present"
    );
    assert_eq!(
        discover_iommu_mode(9, Some("intel_iommu=off")),
        "disabled_by_kernel_arg"
    );
    assert_eq!(discover_iommu_mode(9, None), "enabled_groups_present");
    assert_eq!(
        discover_iommu_mode(0, Some("amd_iommu=on")),
        "enabled_requested"
    );
    assert_eq!(discover_iommu_mode(0, None), "not_detected");

    assert_eq!(
        gpu_direct_rdma_capability(
            CapabilityState::SupportedAndVerified,
            2,
            Some("nvidia_peermem")
        ),
        CapabilityState::SupportedUnverified
    );
    assert_eq!(
        gpu_direct_rdma_capability(CapabilityState::Unsupported, 2, Some("nvidia_peermem")),
        CapabilityState::DegradedToPinnedHost
    );
    assert_eq!(
        gpu_direct_rdma_capability(
            CapabilityState::SupportedAndVerified,
            0,
            Some("nv_peer_mem")
        ),
        CapabilityState::DegradedToPinnedHost
    );
    assert_eq!(
        gpu_direct_rdma_capability(CapabilityState::SupportedAndVerified, 2, None),
        CapabilityState::DegradedToPinnedHost
    );
}

#[test]
fn transport_planner_uses_verified_gpu_direct_only() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let decision = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::Vram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::SupportedAndVerified,
            CapabilityState::Unsupported,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();

    assert_eq!(decision.path, TransportPathKind::TrueGpuDirectRdma);
    assert_eq!(decision.class, TransportPathClass::GpuDirect);
    assert_eq!(decision.explicit_copy_bytes, 0);
    assert!(!decision.pageable_copy);
    assert!(!decision.per_token_registration);
}

#[test]
fn transport_planner_degrades_unverified_direct_path_to_pinned_host() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let decision = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::Vram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::SupportedUnverified,
            CapabilityState::Unsupported,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();

    assert_eq!(decision.path, TransportPathKind::OptimizedPinnedHostBounce);
    assert_eq!(decision.class, TransportPathClass::HostStaged);
    assert_eq!(decision.explicit_copy_bytes, 64 * 1024);
    assert!(!decision.pageable_copy);
    assert!(!decision.per_token_registration);
}

#[test]
fn transport_planner_can_select_mapped_pinned_for_small_decode_only() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let small_decode = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::PinnedDram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::Unsupported,
            CapabilityState::SupportedAndVerified,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();
    let prefill = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::PinnedDram,
            16 * 1024 * 1024,
            TransferMode::Prefill,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::Unsupported,
            CapabilityState::SupportedAndVerified,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();

    assert_eq!(small_decode.path, TransportPathKind::MappedPinnedHostWrite);
    assert_eq!(small_decode.class, TransportPathClass::MappedPinned);
    assert_eq!(prefill.path, TransportPathKind::OptimizedPinnedHostBounce);
}

#[test]
fn transport_path_probe_reports_explicit_fallback_without_hot_allocations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_path_probe().unwrap();

    assert_eq!(summary.status, TransportPathProbeStatus::Ok);
    assert_eq!(summary.requests, 7);
    assert_eq!(summary.decode_requests, 4);
    assert_eq!(summary.prefill_requests, 3);
    assert_eq!(summary.pinned_host_paths, 6);
    assert_eq!(summary.cpu_produced_paths, 1);
    assert_eq!(summary.transport_events, 7);
    assert_eq!(summary.copy_events, 6);
    assert_eq!(summary.sync_events, 7);
    assert_eq!(summary.phase_handoff_syncs, 7);
    assert_eq!(summary.fallback_decisions, 6);
    assert_eq!(summary.estimated_events, 20);
    assert_eq!(summary.estimated_latency_ns, summary.total_latency_ns);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.explicit_copy_bytes > 0);
    assert!(summary.to_json().contains("\"pinned_host_paths\":6"));
}

#[test]
fn transport_capability_matrix_reports_required_sizes_and_degradation() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_capability_matrix_probe().unwrap();

    assert_eq!(summary.status, TransportCapabilityMatrixStatus::Ok);
    assert_eq!(summary.sizes, 6);
    assert_eq!(summary.entries.len(), 24);
    assert_eq!(summary.decode_entries, 12);
    assert_eq!(summary.prefill_entries, 12);
    assert_eq!(summary.host_staged_entries, 18);
    assert_eq!(summary.cpu_produced_entries, 6);
    assert_eq!(summary.gpu_direct_entries, 0);
    assert_eq!(summary.mapped_pinned_entries, 0);
    assert_eq!(summary.degraded_to_pinned_host_entries, 12);
    assert_eq!(summary.supported_unverified_entries, 6);
    assert_eq!(summary.supported_verified_entries, 6);
    assert_eq!(summary.unsupported_entries, 0);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(
        summary.registration_cache_hits,
        summary.entries.len() as u64
    );
    assert_eq!(summary.credit_stall_ns, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.explicit_copy_bytes > 0);
    assert!(summary.estimated_cpu_core_ns > 0);
    assert!(summary.dram_read_bytes > 0);
    assert!(summary.dram_write_bytes > 0);
    assert!(summary.pcie_tx_bytes > 0);
    assert!(summary.pcie_rx_bytes > 0);
    assert!(summary.total_estimated_visible_ns > 0);
    assert!(summary.p50_estimated_visible_ns > 0);
    assert!(summary.p95_estimated_visible_ns >= summary.p50_estimated_visible_ns);
    assert!(summary.p99_estimated_visible_ns >= summary.p95_estimated_visible_ns);
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.effective_payload_bandwidth_bps > 0)
    );
    assert!(summary.entries.iter().all(|entry| entry.queue_depth > 0));
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.registration_cache_hit)
    );
    let json = summary.to_json();
    assert!(json.contains("\"requested_path\":\"A_GPU_DIRECT_RDMA\""));
    assert!(json.contains("\"size_bytes\":32768"));
    assert!(json.contains("\"capability_result\":\"DEGRADED_TO_PINNED_HOST\""));
    assert!(json.contains("\"metric_source\":\"estimated_model\""));
    assert!(json.contains("\"p95_estimated_visible_ns\""));
    assert!(json.contains("\"effective_payload_bandwidth_bps\""));
    assert!(json.contains("\"estimated_cpu_core_ns\""));
    assert!(json.contains("\"dram_read_bytes\""));
    assert!(json.contains("\"dram_write_bytes\""));
    assert!(json.contains("\"pcie_tx_bytes\""));
    assert!(json.contains("\"pcie_rx_bytes\""));
    assert!(json.contains("\"registration_cache_hits\""));
    assert!(json.contains("\"credit_stall_ns\""));
}

#[test]
fn materializes_hf_weight_manifest_as_dram_resident_blocks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();

    assert_eq!(table.entries.len(), manifest.entries.len());
    assert_eq!(table.total_weight_bytes, manifest.total_weight_bytes);
    assert_eq!(
        table.registry.used_bytes(MemoryTier::Dram),
        manifest.total_weight_bytes
    );
    assert_eq!(table.registry.used_bytes(MemoryTier::Vram), 0);
    assert_eq!(table.ledger.hot_path_allocations, 0);
    assert_eq!(
        table.ledger.residency_decisions.len(),
        manifest.entries.len()
    );

    let first = table.entries.first().unwrap();
    let block = table.registry.block(first.block_id).unwrap();
    assert_eq!(first.name, "model.embed_tokens.weight");
    assert_eq!(block.kind, BlockKind::Weight);
    assert_eq!(
        block.semantics,
        nerva_core::types::MutationSemantics::Immutable
    );
    assert_eq!(block.tier, MemoryTier::Dram);
    assert_eq!(block.dtype, first.dtype);
    assert_eq!(block.layout, LayoutId(1));
}

#[test]
fn materialized_weight_manifest_preserves_last_block_and_decision() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();

    let last = table.entries.last().unwrap();
    let decision = table.ledger.residency_decisions.last().unwrap();
    assert_eq!(last.name, "lm_head.weight");
    assert_eq!(last.block_id, ResidentBlockId(290));
    assert_eq!(decision.block_id, last.block_id);
    assert_eq!(decision.old_tier, MemoryTier::Disk);
    assert_eq!(decision.new_tier, MemoryTier::Dram);
}

#[test]
fn materialized_weight_manifest_respects_dram_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let err = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(0, 0, manifest.total_weight_bytes - 1),
        )
        .unwrap_err();

    assert!(matches!(err, NervaError::AllocationFailed { .. }));
}

#[test]
fn materializes_safetensors_shard_plan_with_source_offsets() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header_len) = tiny_shard_plan();
    let table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();

    assert_eq!(table.entries.len(), plan.entries.len());
    assert_eq!(table.total_weight_bytes, plan.total_weight_bytes);
    assert_eq!(table.registry.used_bytes(MemoryTier::Dram), 464);
    assert_eq!(table.registry.used_bytes(MemoryTier::Vram), 0);
    assert_eq!(table.ledger.hot_path_allocations, 0);
    assert_eq!(table.ledger.residency_decisions.len(), plan.entries.len());

    let first = table.entries.first().unwrap();
    assert_eq!(first.name, "model.embed_tokens.weight");
    assert_eq!(first.source_shard.as_deref(), Some(SHARD_ONE));
    assert_eq!(first.file_offset_begin, Some(8 + header_len));
    assert_eq!(first.file_offset_end, Some(8 + header_len + first.bytes));
    assert_eq!(first.tier, MemoryTier::Dram);

    let block = table.registry.block(first.block_id).unwrap();
    assert_eq!(block.kind, BlockKind::Weight);
    assert_eq!(block.layout, LayoutId(1));
    assert_eq!(block.dtype, first.dtype);
}

#[test]
fn materialized_safetensors_shard_plan_respects_dram_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, _) = tiny_shard_plan();
    let err = runtime
        .materialize_safetensors_shard_plan_with_budget(
            &plan,
            ResidencyBudget::new(0, 0, plan.total_weight_bytes - 1),
        )
        .unwrap_err();

    assert!(matches!(err, NervaError::AllocationFailed { .. }));
}

#[test]
fn resident_weight_prefetch_plan_records_bounded_tasks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header_len) = tiny_shard_plan();
    let table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 128).unwrap();

    assert_eq!(prefetch.tasks.len(), table.entries.len());
    assert_eq!(prefetch.total_bytes, table.total_weight_bytes);
    assert_eq!(prefetch.shard_count, 1);
    assert_eq!(prefetch.prefetch_events, prefetch.tasks.len() as u64);
    assert_eq!(prefetch.copy_events, prefetch.tasks.len() as u64);
    assert_eq!(prefetch.ledger.hot_path_allocations, 0);
    assert_eq!(prefetch.first_source_shard.as_deref(), Some(SHARD_ONE));
    assert_eq!(prefetch.last_source_shard.as_deref(), Some(SHARD_ONE));

    let first = prefetch.tasks.first().unwrap();
    assert_eq!(first.task_index, 0);
    assert_eq!(first.name, "model.embed_tokens.weight");
    assert_eq!(first.file_offset_begin, 8 + header_len);
    assert_eq!(first.file_offset_end, 8 + header_len + first.bytes);
    assert_eq!(first.target_tier, MemoryTier::Dram);
    assert!(prefetch.to_json().contains("\"tasks\":11"));
}

#[test]
fn resident_weight_prefetch_plan_splits_large_source_spans() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header_len) = tiny_shard_plan();
    let table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();

    assert!(prefetch.tasks.len() > table.entries.len());
    assert_eq!(prefetch.total_bytes, table.total_weight_bytes);
    assert_eq!(prefetch.tasks[0].bytes, 16);
    assert_eq!(prefetch.tasks[0].file_offset_begin, 8 + header_len);
    assert_eq!(prefetch.tasks[1].file_offset_begin, 8 + header_len + 16);
    assert_eq!(prefetch.tasks[4].file_offset_end, 8 + header_len + 80);
    assert_eq!(
        prefetch.tasks[5].name,
        "model.layers.0.input_layernorm.weight"
    );
}

#[test]
fn resident_weight_prefetch_requires_source_offsets() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();

    assert!(runtime.plan_resident_weight_prefetch(&table, 128).is_err());
    assert!(runtime.plan_resident_weight_prefetch(&table, 0).is_err());
}

#[test]
fn resident_weight_prefetch_execution_marks_blocks_ready() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, _) = tiny_shard_plan();
    let mut table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();
    let summary = runtime
        .execute_resident_weight_prefetch_plan(&mut table, &prefetch)
        .unwrap();

    assert_eq!(summary.tasks, prefetch.tasks.len());
    assert_eq!(summary.completed_blocks, table.entries.len());
    assert_eq!(summary.total_bytes, table.total_weight_bytes);
    assert_eq!(summary.prefetch_events, prefetch.tasks.len() as u64);
    assert_eq!(summary.copy_events, prefetch.tasks.len() as u64);
    assert_eq!(summary.ready_blocks, table.entries.len());
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"ready_blocks\":11"));
    assert!(table.entries.iter().all(|entry| {
        table
            .registry
            .block(entry.block_id)
            .is_some_and(|block| block.state == ResidencyState::Ready)
    }));
}

#[test]
fn resident_weight_prefetch_execution_rejects_incomplete_plan() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, _) = tiny_shard_plan();
    let mut table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let mut prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();
    prefetch.tasks.pop();

    assert!(
        runtime
            .execute_resident_weight_prefetch_plan(&mut table, &prefetch)
            .is_err()
    );
}

#[test]
fn resident_weight_hotset_promotion_moves_bounded_prefix_to_vram() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(256, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    let summary = runtime
        .promote_resident_weight_hotset(&mut table, 200)
        .unwrap();

    assert_eq!(summary.promoted_blocks, 7);
    assert_eq!(summary.promoted_bytes, 192);
    assert_eq!(summary.vram_used_bytes, 192);
    assert_eq!(summary.dram_used_bytes, 272);
    assert_eq!(summary.residency_decisions, 7);
    assert_eq!(
        summary.first_promoted_tensor.as_deref(),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(
        summary.last_promoted_tensor.as_deref(),
        Some("model.layers.0.post_attention_layernorm.weight")
    );
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(table.entries[0].tier, MemoryTier::Vram);
    assert_eq!(table.entries[6].tier, MemoryTier::Vram);
    assert_eq!(table.entries[7].tier, MemoryTier::Dram);
    assert!(table.entries[..7].iter().all(|entry| {
        table
            .registry
            .block(entry.block_id)
            .is_some_and(|block| block.state == ResidencyState::Ready)
    }));
    assert!(summary.to_json().contains("\"promoted_blocks\":7"));
}

#[test]
fn resident_weight_hotset_promotion_respects_vram_capacity_and_zero_limit() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(100, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    let zero = runtime
        .promote_resident_weight_hotset(&mut table, 0)
        .unwrap();
    assert_eq!(zero.promoted_blocks, 0);
    assert_eq!(zero.vram_used_bytes, 0);

    let summary = runtime
        .promote_resident_weight_hotset(&mut table, usize::MAX)
        .unwrap();
    assert_eq!(summary.promoted_blocks, 2);
    assert_eq!(summary.promoted_bytes, 88);
    assert_eq!(summary.vram_used_bytes, 88);
    assert_eq!(summary.dram_used_bytes, 376);
}

#[test]
fn resident_weight_execution_plans_gpu_staged_for_dram_fp16_weights() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 3, Some(89))
        .unwrap();

    assert_eq!(plan.steps.len(), 3);
    assert_eq!(plan.cpu_steps, 0);
    assert_eq!(plan.gpu_resident_steps, 0);
    assert_eq!(plan.gpu_staged_steps, 3);
    assert_eq!(plan.fallback_steps, 0);
    assert_eq!(plan.fallback_decisions, 0);
    assert_eq!(plan.block_version_dependencies, 3);
    assert_eq!(plan.ledger.execution_decisions.len(), 3);
    assert!(plan.ledger.require_satisfied_block_versions().is_ok());
    assert_eq!(
        plan.steps[0].strategy,
        ResidentWeightExecutionStrategy::GpuStaged
    );
    assert_eq!(
        plan.steps[0].kernel_name,
        "cuda_decode_dense_matvec_fp16_bf16"
    );
    assert!(plan.to_json().contains("\"gpu_staged_steps\":3"));
}

#[test]
fn resident_weight_execution_uses_gpu_resident_hotset_blocks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(128, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    runtime
        .promote_resident_weight_hotset(&mut table, 100)
        .unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 3, Some(89))
        .unwrap();

    assert_eq!(plan.gpu_resident_steps, 2);
    assert_eq!(plan.gpu_staged_steps, 1);
    assert_eq!(
        plan.steps[0].strategy,
        ResidentWeightExecutionStrategy::GpuResident
    );
    assert_eq!(
        plan.steps[2].strategy,
        ResidentWeightExecutionStrategy::GpuStaged
    );
}

#[test]
fn resident_weight_execution_uses_exact_cpu_fallback_for_f32_cuda() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float32"
            }"#,
    )
    .unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();

    assert_eq!(plan.cpu_steps, 2);
    assert_eq!(plan.fallback_steps, 2);
    assert_eq!(plan.fallback_decisions, 2);
    assert_eq!(plan.ledger.fallback_count_for(FallbackClass::ExactNamed), 2);
    assert!(plan.steps.iter().all(|step| step.fallback));
    assert!(
        plan.steps
            .iter()
            .all(|step| step.kernel_name == "cpu_reference_dense_matvec_f32")
    );
}

#[test]
fn resident_weight_execution_rejects_non_ready_blocks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let first = table.entries.first().unwrap().block_id;
    table
        .registry
        .transition(first, ResidencyState::Prefetching)
        .unwrap();

    assert!(
        runtime
            .plan_resident_weight_execution(&table, 1, Some(89))
            .is_err()
    );
}

#[test]
fn resident_weight_execution_run_ledgers_gpu_resident_and_staged_work() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(128, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    runtime
        .promote_resident_weight_hotset(&mut table, 100)
        .unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 3, Some(89))
        .unwrap();
    let summary = runtime
        .execute_resident_weight_execution_plan(&table, &plan)
        .unwrap();

    assert_eq!(summary.steps, 3);
    assert_eq!(summary.gpu_resident_steps, 2);
    assert_eq!(summary.gpu_staged_steps, 1);
    assert_eq!(summary.fallback_steps, 0);
    assert_eq!(summary.fallback_decisions, 0);
    assert_eq!(summary.block_version_dependencies, 3);
    assert_eq!(summary.cpu_events, 0);
    assert_eq!(summary.device_events, 3);
    assert_eq!(summary.copy_events, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.total_latency_ns > 0);
    assert!(summary.to_json().contains("\"device_events\":3"));
}

#[test]
fn resident_weight_execution_run_ledgers_exact_cpu_fallback() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float32"
            }"#,
    )
    .unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();
    let summary = runtime
        .execute_resident_weight_execution_plan(&table, &plan)
        .unwrap();

    assert_eq!(summary.steps, 2);
    assert_eq!(summary.cpu_events, 2);
    assert_eq!(summary.device_events, 0);
    assert_eq!(summary.copy_events, 0);
    assert_eq!(summary.fallback_steps, 2);
    assert_eq!(summary.fallback_decisions, 2);
    assert_eq!(
        summary.ledger.fallback_count_for(FallbackClass::ExactNamed),
        2
    );
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn resident_weight_execution_run_rejects_unsatisfied_block_version() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let mut plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();
    plan.steps[0].block_version = plan.steps[0].block_version.saturating_add(1);

    assert!(
        runtime
            .execute_resident_weight_execution_plan(&table, &plan)
            .is_err()
    );
}

#[test]
fn resident_weight_execution_run_rejects_stale_plan_after_tier_change() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(128, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();
    runtime
        .promote_resident_weight_hotset(&mut table, 100)
        .unwrap();

    assert!(
        runtime
            .execute_resident_weight_execution_plan(&table, &plan)
            .is_err()
    );
}

#[test]
fn resident_weight_probe_reports_manifest_materialization() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_resident_weight_probe().unwrap();

    assert_eq!(summary.status, ResidentWeightProbeStatus::Ok);
    assert_eq!(summary.blocks, 290);
    assert_eq!(summary.total_weight_bytes, 11_866_210_304);
    assert_eq!(summary.dram_used_bytes, summary.total_weight_bytes);
    assert_eq!(summary.vram_used_bytes, 0);
    assert_eq!(summary.residency_decisions, 290);
    assert_eq!(summary.first_block_id, Some(ResidentBlockId(1)));
    assert_eq!(summary.last_block_id, Some(ResidentBlockId(290)));
    assert_eq!(
        summary.first_tensor.as_deref(),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(summary.last_tensor.as_deref(), Some("lm_head.weight"));
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"blocks\":290"));
}

#[test]
fn runtime_creates_residency_registry_from_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let registry = runtime.block_registry(ResidencyBudget::new(1024, 2048, 4096));
    assert_eq!(registry.remaining_bytes(MemoryTier::Vram), Some(1024));
    assert_eq!(registry.remaining_bytes(MemoryTier::PinnedDram), Some(2048));
    assert_eq!(registry.remaining_bytes(MemoryTier::Dram), Some(4096));
}

#[test]
fn runtime_creates_static_arenas_from_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let arenas = runtime.static_arenas(ResidencyBudget::new(1024, 2048, 4096));
    assert_eq!(arenas.device().capacity(), 1024);
    assert_eq!(arenas.pinned_host().capacity(), 2048);
    assert_eq!(arenas.host().capacity(), 4096);
}

#[test]
fn static_arena_probe_bootstraps_and_rejects_hot_path_allocations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .static_arena_probe(ResidencyBudget::new(1024, 2048, 4096))
        .unwrap();

    assert_eq!(summary.bootstrap_blocks, 3);
    assert_eq!(summary.ready_blocks, 3);
    assert_eq!(summary.device_used_bytes, 256);
    assert_eq!(summary.pinned_host_used_bytes, 256);
    assert_eq!(summary.host_used_bytes, 512);
    assert_eq!(summary.hot_path_rejections, 3);
    assert_eq!(summary.hot_path_allocation_attempts, 3);
    assert!(summary.usage_preserved_after_rejections);
    assert!(summary.to_json().contains("\"ready_blocks\":3"));
}

#[test]
fn graph_pool_rejects_layout_drift() {
    let captured = GraphLayout::new(1, 8, 16, 4);
    let replay = GraphLayout::new(1, 8, 32, 4);
    let mut pool = GraphPool::new();
    pool.capture_synthetic(captured);

    assert!(pool.check_before_replay(captured).is_ok());
    assert!(pool.check_before_replay(replay).is_err());
}

#[test]
fn synthetic_launch_collect_records_token_and_ledger() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let step = engine
        .launch(RequestId(1), SequenceId(1), 0, TokenId(41))
        .unwrap();
    let output = step.collect().unwrap();

    assert_eq!(output.token, TokenId(42));
    assert_eq!(output.input_source, TokenInputSource::Seed);
    assert_eq!(output.device_token_ref.token_index, 0);
    assert_eq!(output.ledger.hot_path_allocations, 0);
    assert_eq!(output.ledger.events.len(), 4);
    assert_eq!(output.ledger.event_count(LedgerEventKind::GraphReplay), 1);
    assert_eq!(
        output.ledger.event_count(LedgerEventKind::DeviceActivity),
        1
    );
    assert_eq!(output.ledger.event_count(LedgerEventKind::Copy), 1);
    assert_eq!(output.ledger.event_count(LedgerEventKind::Sync), 1);
    assert_eq!(
        output.ledger.sync_count_for(SyncClass::SoftVisibilitySync),
        1
    );
    assert_eq!(output.ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 3);
    assert_eq!(output.ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 0);
    assert!(output.ledger.require_classified_syncs().is_ok());
    assert_eq!(
        engine
            .token_ring()
            .consume_device_input(RequestId(1), SequenceId(1), 0)
            .unwrap(),
        TokenId(42)
    );
    assert_eq!(
        engine
            .graph_pool()
            .replay_count(GraphKey {
                bucket: 1,
                max_blocks: 1
            })
            .unwrap(),
        1
    );
}

#[test]
fn synthetic_next_step_must_use_device_token() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let output = engine
        .launch(RequestId(2), SequenceId(9), 0, TokenId(10))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(11));

    let err = engine
        .launch(RequestId(2), SequenceId(9), 1, TokenId(99))
        .unwrap_err();
    assert!(matches!(err, NervaError::ResidencyViolation { .. }));

    let output = engine
        .launch_device_next(RequestId(2), SequenceId(9), 1, TokenId(10))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(12));
    assert!(matches!(
        output.input_source,
        TokenInputSource::DeviceRing(DeviceTokenRef { token_index: 0, .. })
    ));
}

#[test]
fn device_token_ring_rejects_stale_reads() {
    let mut ring = DeviceTokenRing::new(2).unwrap();
    ring.publish(RequestId(1), SequenceId(1), 0, TokenId(7))
        .unwrap();
    assert!(
        ring.consume_device_input(RequestId(1), SequenceId(2), 0)
            .is_err()
    );
    assert_eq!(
        ring.consume_device_input(RequestId(1), SequenceId(1), 0)
            .unwrap(),
        TokenId(7)
    );
}

#[test]
fn synthetic_decode_summary_runs_1024_device_first_steps() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(1024, 64, TokenId(1)))
        .unwrap();

    assert_eq!(summary.status, SyntheticDecodeStatus::Ok);
    assert_eq!(summary.steps, 1024);
    assert_eq!(summary.last_token, Some(TokenId(1025)));
    assert_eq!(summary.graph_replays, 1024);
    assert_eq!(summary.graph_replay_events, 1024);
    assert_eq!(summary.kernel_events, 2048);
    assert_eq!(summary.device_events, 1024);
    assert_eq!(summary.copy_events, 1024);
    assert_eq!(summary.host_wait_events, 1024);
    assert_eq!(summary.soft_visibility_syncs, 1024);
    assert_eq!(summary.device_timeline_active_ns, 3072);
    assert_eq!(summary.device_timeline_idle_ns, 0);
    assert_eq!(summary.graph_replay_latency_ns, 1024);
    assert_eq!(summary.device_latency_ns, 3072);
    assert_eq!(summary.copy_latency_ns, 1024);
    assert_eq!(summary.host_wait_latency_ns, 1024);
    assert_eq!(summary.soft_visibility_sync_latency_ns, 1024);
    assert_eq!(summary.estimated_events, 4096);
    assert_eq!(summary.estimated_latency_ns, summary.total_latency_ns);
    assert_eq!(summary.total_latency_ns, 6144);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.observed_tokens, 1024);
    assert_ne!(summary.observed_token_hash, 0);
    assert_eq!(summary.token_ring_slots_touched, 64);
    assert_eq!(summary.token_ring_reuses, 960);
    assert_eq!(summary.token_ring_max_slot_version, 16);
    assert_eq!(summary.stale_tokens, 0);
    assert_eq!(summary.missing_tokens, 0);
    assert_eq!(summary.extra_tokens, 0);
    assert_eq!(summary.mismatched_tokens, 0);
    assert_eq!(summary.host_causality_edges, 0);
    assert!(summary.to_json().contains("\"steps\":1024"));
    assert!(summary.to_json().contains("\"observed_token_hash\""));
    assert!(summary.to_json().contains("\"token_ring_reuses\":960"));
    assert!(summary.to_json().contains("\"host_causality_edges\":0"));
}

#[test]
fn synthetic_decode_rejects_zero_steps() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let err = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(0, 64, TokenId(1)))
        .unwrap_err();
    assert!(matches!(err, NervaError::InvalidArgument { .. }));
}

#[test]
fn kv_residency_probe_exercises_prefetch_demote_and_evict_paths() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig::default())
        .unwrap();

    assert_eq!(summary.status, KvResidencyProbeStatus::Ok);
    assert_eq!(summary.decisions, 4);
    assert_eq!(summary.prefetches, 2);
    assert_eq!(summary.demotions, 1);
    assert_eq!(summary.evictions, 1);
    assert_eq!(summary.copy_events, 3);
    assert_eq!(summary.prefetch_events, 2);
    assert_eq!(summary.eviction_events, 2);
    assert_eq!(summary.stall_events, 3);
    assert_eq!(summary.copy_bytes, 384);
    assert_eq!(summary.changed_bytes, 384);
    assert_eq!(summary.visible_stall_ns, 684);
    assert_eq!(summary.total_latency_ns, 684);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.vram_used_bytes, 256);
    assert_eq!(summary.dram_used_bytes, 256);
    assert!(summary.to_json().contains("\"prefetches\":2"));
    assert!(summary.to_json().contains("\"stall_events\":3"));
}

#[test]
fn kv_residency_probe_rejects_too_few_pages() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let err = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig {
            pages: 3,
            ..KvResidencyProbeConfig::default()
        })
        .unwrap_err();
    assert!(matches!(err, NervaError::InvalidArgument { .. }));
}
