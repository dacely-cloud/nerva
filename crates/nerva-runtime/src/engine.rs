use std::{collections::BTreeMap, env, fs};

use nerva_core::{
    AllocationId, BlockKind, DeviceOrdinal, ExecutionOwner, HostArch, LayoutId, MemoryFabricKind,
    MemoryTier, NervaError, RequestId, ResidencyState, ResidentBlockId, Result, SequenceId,
    TokenId, ensure_supported_linux_host, host_arch,
};
use nerva_kernel_contracts::{
    KernelBackend, KernelOperation, KernelPlan, KernelQuery, bootstrap_registry,
};
use nerva_ledger::{
    BlockVersionDependency, CandidateCost, ExecutionDecision, FallbackClass, FallbackDecision,
    LedgerEvent, LedgerEventKind, MetricSource, ResidencyDecision, SyncClass, TokenLedger,
};
use nerva_memory::{
    ArenaKind, BlockAllocationRequest, BlockRegistry, KvPageSpec, KvPrefixKey, KvResidencyAction,
    KvResidencyPlanner, KvResidencyPolicy, StaticArenaBootstrapSpec, StaticArenaSet,
};

use crate::graph::GraphKey;
#[cfg(test)]
use crate::graph::{GraphLayout, GraphPool};
#[cfg(test)]
use crate::token::{DeviceTokenRef, DeviceTokenRing};
use crate::token::{SyntheticEngine, TokenInputSource};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub device: DeviceOrdinal,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceOrdinal(0),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ResidencyBudget {
    pub vram_bytes: usize,
    pub pinned_dram_bytes: usize,
    pub dram_bytes: usize,
}

impl ResidencyBudget {
    pub const fn new(vram_bytes: usize, pinned_dram_bytes: usize, dram_bytes: usize) -> Self {
        Self {
            vram_bytes,
            pinned_dram_bytes,
            dram_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaProbeSummary {
    pub device_capacity_bytes: usize,
    pub pinned_host_capacity_bytes: usize,
    pub host_capacity_bytes: usize,
    pub device_used_bytes: usize,
    pub pinned_host_used_bytes: usize,
    pub host_used_bytes: usize,
    pub bootstrap_blocks: usize,
    pub ready_blocks: usize,
    pub hot_path_rejections: u64,
    pub hot_path_allocation_attempts: u64,
    pub usage_preserved_after_rejections: bool,
}

impl StaticArenaProbeSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"device_capacity_bytes\":{},\"pinned_host_capacity_bytes\":{},\"host_capacity_bytes\":{},\"device_used_bytes\":{},\"pinned_host_used_bytes\":{},\"host_used_bytes\":{},\"bootstrap_blocks\":{},\"ready_blocks\":{},\"hot_path_rejections\":{},\"hot_path_allocation_attempts\":{},\"usage_preserved_after_rejections\":{}}}",
            self.device_capacity_bytes,
            self.pinned_host_capacity_bytes,
            self.host_capacity_bytes,
            self.device_used_bytes,
            self.pinned_host_used_bytes,
            self.host_used_bytes,
            self.bootstrap_blocks,
            self.ready_blocks,
            self.hot_path_rejections,
            self.hot_path_allocation_attempts,
            self.usage_preserved_after_rejections,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CapabilityState {
    SupportedAndVerified,
    SupportedUnverified,
    Unsupported,
    DegradedToPinnedHost,
}

impl CapabilityState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SupportedAndVerified => "SUPPORTED_AND_VERIFIED",
            Self::SupportedUnverified => "SUPPORTED_UNVERIFIED",
            Self::Unsupported => "UNSUPPORTED",
            Self::DegradedToPinnedHost => "DEGRADED_TO_PINNED_HOST",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TopologySnapshot {
    pub cpu_online: Option<String>,
    pub cpu_count: usize,
    pub numa_node_count: usize,
    pub pci_device_count: usize,
    pub pci_root_complex_count: usize,
    pub pci_bus_count: usize,
    pub pci_gpu_count: usize,
    pub pci_network_count: usize,
    pub pci_nvme_count: usize,
    pub block_device_count: usize,
    pub nvme_block_device_count: usize,
    pub rdma_device_count: usize,
    pub rdma_device_names: Vec<String>,
    pub rdma_netdev_links: Vec<String>,
    pub iommu_group_count: usize,
    pub iommu_mode: String,
    pub iommu_kernel_args: Option<String>,
}

impl TopologySnapshot {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"cpu_online\":{},\"cpu_count\":{},\"numa_node_count\":{},\"pci_device_count\":{},\"pci_root_complex_count\":{},\"pci_bus_count\":{},\"pci_gpu_count\":{},\"pci_network_count\":{},\"pci_nvme_count\":{},\"block_device_count\":{},\"nvme_block_device_count\":{},\"rdma_device_count\":{},\"rdma_device_names\":{},\"rdma_netdev_links\":{},\"iommu_group_count\":{},\"iommu_mode\":\"{}\",\"iommu_kernel_args\":{}}}",
            json_opt_string(self.cpu_online.as_deref()),
            self.cpu_count,
            self.numa_node_count,
            self.pci_device_count,
            self.pci_root_complex_count,
            self.pci_bus_count,
            self.pci_gpu_count,
            self.pci_network_count,
            self.pci_nvme_count,
            self.block_device_count,
            self.nvme_block_device_count,
            self.rdma_device_count,
            json_string_array(&self.rdma_device_names),
            json_string_array(&self.rdma_netdev_links),
            self.iommu_group_count,
            json_escape(&self.iommu_mode),
            json_opt_string(self.iommu_kernel_args.as_deref()),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilitySnapshot {
    pub host_arch: HostArch,
    pub target_os: &'static str,
    pub target_arch: &'static str,
    pub kernel_release: Option<String>,
    pub fabric: MemoryFabricKind,
    pub cuda: CapabilityState,
    pub cuda_status: &'static str,
    pub cuda_error: Option<String>,
    pub cuda_visible_devices: Option<String>,
    pub cuda_compute_capability: Option<String>,
    pub cuda_device_total_memory_bytes: Option<usize>,
    pub cuda_pci_bus_id: Option<String>,
    pub hip: CapabilityState,
    pub hip_visible_devices: Option<String>,
    pub nvidia_driver_version: Option<String>,
    pub rdma_core_loaded: bool,
    pub mlx5_core_loaded: bool,
    pub nvidia_peer_memory_module: Option<String>,
    pub pinned_host_staging: CapabilityState,
    pub gpu_direct_rdma: CapabilityState,
    pub amd_peerdirect: CapabilityState,
    pub dma_buf_export: CapabilityState,
    pub cxl: CapabilityState,
    pub topology: TopologySnapshot,
}

impl CapabilitySnapshot {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"host_arch\":\"{}\",\"target_os\":\"{}\",\"target_arch\":\"{}\",\"kernel_release\":{},\"fabric\":\"{}\",\"cuda\":\"{}\",\"cuda_status\":\"{}\",\"cuda_error\":{},\"cuda_visible_devices\":{},\"cuda_compute_capability\":{},\"cuda_device_total_memory_bytes\":{},\"cuda_pci_bus_id\":{},\"hip\":\"{}\",\"hip_visible_devices\":{},\"nvidia_driver_version\":{},\"rdma_core_loaded\":{},\"mlx5_core_loaded\":{},\"nvidia_peer_memory_module\":{},\"pinned_host_staging\":\"{}\",\"gpu_direct_rdma\":\"{}\",\"amd_peerdirect\":\"{}\",\"dma_buf_export\":\"{}\",\"cxl\":\"{}\",\"topology\":{}}}",
            host_arch_to_str(self.host_arch),
            self.target_os,
            self.target_arch,
            json_opt_string(self.kernel_release.as_deref()),
            memory_fabric_to_str(self.fabric),
            self.cuda.as_str(),
            self.cuda_status,
            json_opt_string(self.cuda_error.as_deref()),
            json_opt_string(self.cuda_visible_devices.as_deref()),
            json_opt_string(self.cuda_compute_capability.as_deref()),
            json_opt_usize(self.cuda_device_total_memory_bytes),
            json_opt_string(self.cuda_pci_bus_id.as_deref()),
            self.hip.as_str(),
            json_opt_string(self.hip_visible_devices.as_deref()),
            json_opt_string(self.nvidia_driver_version.as_deref()),
            self.rdma_core_loaded,
            self.mlx5_core_loaded,
            json_opt_string(self.nvidia_peer_memory_module.as_deref()),
            self.pinned_host_staging.as_str(),
            self.gpu_direct_rdma.as_str(),
            self.amd_peerdirect.as_str(),
            self.dma_buf_export.as_str(),
            self.cxl.as_str(),
            self.topology.to_json(),
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransferMode {
    Decode,
    Prefill,
}

impl TransferMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathKind {
    TrueGpuDirectRdma,
    OptimizedPinnedHostBounce,
    CpuProducedBoundary,
    MappedPinnedHostWrite,
}

impl TransportPathKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrueGpuDirectRdma => "true_gpu_direct_rdma",
            Self::OptimizedPinnedHostBounce => "optimized_pinned_host_bounce",
            Self::CpuProducedBoundary => "cpu_produced_boundary",
            Self::MappedPinnedHostWrite => "mapped_pinned_host_write",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathClass {
    GpuDirect,
    HostStaged,
    CpuProduced,
    MappedPinned,
}

impl TransportPathClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuDirect => "GPU_DIRECT",
            Self::HostStaged => "HOST_STAGED",
            Self::CpuProduced => "CPU_PRODUCED",
            Self::MappedPinned => "MAPPED_PINNED",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathRequest {
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
    pub bytes: usize,
    pub mode: TransferMode,
    pub producer: ExecutionOwner,
    pub gpu_direct_rdma: CapabilityState,
    pub mapped_pinned_output: CapabilityState,
    pub pinned_host_staging: CapabilityState,
}

impl TransportPathRequest {
    pub const fn new(
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        bytes: usize,
        mode: TransferMode,
        producer: ExecutionOwner,
        gpu_direct_rdma: CapabilityState,
        mapped_pinned_output: CapabilityState,
        pinned_host_staging: CapabilityState,
    ) -> Self {
        Self {
            source_tier,
            destination_tier,
            bytes,
            mode,
            producer,
            gpu_direct_rdma,
            mapped_pinned_output,
            pinned_host_staging,
        }
    }

    pub fn from_capabilities(
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        bytes: usize,
        mode: TransferMode,
        producer: ExecutionOwner,
        capabilities: &CapabilitySnapshot,
    ) -> Self {
        Self::new(
            source_tier,
            destination_tier,
            bytes,
            mode,
            producer,
            capabilities.gpu_direct_rdma,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathDecision {
    pub path: TransportPathKind,
    pub class: TransportPathClass,
    pub request: TransportPathRequest,
    pub reason: &'static str,
    pub estimated_visible_ns: u64,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub pageable_copy: bool,
    pub per_token_registration: bool,
}

impl TransportPathDecision {
    pub fn record_to_ledger(self, ledger: &mut TokenLedger) {
        if self.class == TransportPathClass::HostStaged {
            ledger.record_fallback_decision(FallbackDecision {
                label: "transport_host_staged_fallback",
                class: FallbackClass::CapabilityDegraded,
                requested: "gpu_direct_rdma",
                selected: self.path.as_str(),
                reason: self.reason,
                visible_ns: Some(self.estimated_visible_ns),
                metric_source: MetricSource::EstimatedModel,
            });
        }
        if self.explicit_copy_bytes > 0 {
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(self.request.source_tier),
                to_tier: Some(MemoryTier::PinnedDram),
                bytes: self.explicit_copy_bytes,
                latency_ns: self.estimated_visible_ns / 2,
                label: "transport_explicit_copy",
            });
        }
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(self.request.source_tier),
            to_tier: Some(self.request.destination_tier),
            bytes: self.request.bytes,
            latency_ns: self.estimated_visible_ns,
            label: self.path.as_str(),
        });
        ledger.record_sync(
            SyncClass::PhaseHandoff,
            None,
            Some(self.request.source_tier),
            Some(self.request.destination_tier),
            0,
            1,
            MetricSource::EstimatedModel,
            "transport_ordering_barrier",
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightBlockRef {
    pub name: String,
    pub block_id: ResidentBlockId,
    pub bytes: usize,
    pub dtype: nerva_core::DType,
    pub tier: MemoryTier,
    pub source_shard: Option<String>,
    pub file_offset_begin: Option<usize>,
    pub file_offset_end: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightTable {
    pub registry: BlockRegistry,
    pub entries: Vec<ResidentWeightBlockRef>,
    pub total_weight_bytes: usize,
    pub manifest_hash: u64,
    pub ledger: TokenLedger,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchTask {
    pub task_index: u64,
    pub block_id: ResidentBlockId,
    pub name: String,
    pub source_shard: String,
    pub file_offset_begin: usize,
    pub file_offset_end: usize,
    pub bytes: usize,
    pub target_tier: MemoryTier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchPlan {
    pub tasks: Vec<ResidentWeightPrefetchTask>,
    pub total_bytes: usize,
    pub shard_count: usize,
    pub max_task_bytes: usize,
    pub prefetch_events: u64,
    pub copy_events: u64,
    pub first_source_shard: Option<String>,
    pub last_source_shard: Option<String>,
    pub ledger: TokenLedger,
}

impl ResidentWeightPrefetchPlan {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"tasks\":{},\"total_bytes\":{},\"shard_count\":{},\"max_task_bytes\":{},\"prefetch_events\":{},\"copy_events\":{},\"first_source_shard\":{},\"last_source_shard\":{},\"hot_path_allocations\":{}}}",
            self.tasks.len(),
            self.total_bytes,
            self.shard_count,
            self.max_task_bytes,
            self.prefetch_events,
            self.copy_events,
            json_opt_string(self.first_source_shard.as_deref()),
            json_opt_string(self.last_source_shard.as_deref()),
            self.ledger.hot_path_allocations,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchExecutionSummary {
    pub tasks: usize,
    pub completed_blocks: usize,
    pub total_bytes: usize,
    pub prefetch_events: u64,
    pub copy_events: u64,
    pub ready_blocks: usize,
    pub hot_path_allocations: u64,
    pub ledger: TokenLedger,
}

impl ResidentWeightPrefetchExecutionSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"tasks\":{},\"completed_blocks\":{},\"total_bytes\":{},\"prefetch_events\":{},\"copy_events\":{},\"ready_blocks\":{},\"hot_path_allocations\":{}}}",
            self.tasks,
            self.completed_blocks,
            self.total_bytes,
            self.prefetch_events,
            self.copy_events,
            self.ready_blocks,
            self.hot_path_allocations,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightHotsetSummary {
    pub promoted_blocks: usize,
    pub promoted_bytes: usize,
    pub dram_used_bytes: usize,
    pub vram_used_bytes: usize,
    pub residency_decisions: u64,
    pub first_promoted_tensor: Option<String>,
    pub last_promoted_tensor: Option<String>,
    pub hot_path_allocations: u64,
}

impl ResidentWeightHotsetSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"promoted_blocks\":{},\"promoted_bytes\":{},\"dram_used_bytes\":{},\"vram_used_bytes\":{},\"residency_decisions\":{},\"first_promoted_tensor\":{},\"last_promoted_tensor\":{},\"hot_path_allocations\":{}}}",
            self.promoted_blocks,
            self.promoted_bytes,
            self.dram_used_bytes,
            self.vram_used_bytes,
            self.residency_decisions,
            json_opt_string(self.first_promoted_tensor.as_deref()),
            json_opt_string(self.last_promoted_tensor.as_deref()),
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidentWeightExecutionStrategy {
    CpuDram,
    GpuResident,
    GpuStaged,
    CpuExactFallback,
}

impl ResidentWeightExecutionStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuDram => "cpu-dram",
            Self::GpuResident => "gpu-resident",
            Self::GpuStaged => "gpu-staged",
            Self::CpuExactFallback => "cpu-exact-fallback",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightExecutionStep {
    pub step_index: u64,
    pub block_id: ResidentBlockId,
    pub name: String,
    pub strategy: ResidentWeightExecutionStrategy,
    pub executor: ExecutionOwner,
    pub bytes: usize,
    pub block_version: u64,
    pub predicted_visible_ns: u64,
    pub kernel_name: &'static str,
    pub fallback: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightExecutionPlan {
    pub steps: Vec<ResidentWeightExecutionStep>,
    pub total_weight_bytes: usize,
    pub total_predicted_visible_ns: u64,
    pub cpu_steps: u64,
    pub gpu_resident_steps: u64,
    pub gpu_staged_steps: u64,
    pub fallback_steps: u64,
    pub fallback_decisions: u64,
    pub block_version_dependencies: u64,
    pub first_tensor: Option<String>,
    pub last_tensor: Option<String>,
    pub ledger: TokenLedger,
}

impl ResidentWeightExecutionPlan {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"steps\":{},\"total_weight_bytes\":{},\"total_predicted_visible_ns\":{},\"cpu_steps\":{},\"gpu_resident_steps\":{},\"gpu_staged_steps\":{},\"fallback_steps\":{},\"fallback_decisions\":{},\"block_version_dependencies\":{},\"first_tensor\":{},\"last_tensor\":{},\"execution_decisions\":{},\"hot_path_allocations\":{}}}",
            self.steps.len(),
            self.total_weight_bytes,
            self.total_predicted_visible_ns,
            self.cpu_steps,
            self.gpu_resident_steps,
            self.gpu_staged_steps,
            self.fallback_steps,
            self.fallback_decisions,
            self.block_version_dependencies,
            json_opt_string(self.first_tensor.as_deref()),
            json_opt_string(self.last_tensor.as_deref()),
            self.ledger.execution_decisions.len(),
            self.ledger.hot_path_allocations,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightExecutionRunSummary {
    pub steps: usize,
    pub total_weight_bytes: usize,
    pub total_latency_ns: u64,
    pub cpu_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub gpu_resident_steps: u64,
    pub gpu_staged_steps: u64,
    pub fallback_steps: u64,
    pub fallback_decisions: u64,
    pub block_version_dependencies: u64,
    pub hot_path_allocations: u64,
    pub ledger: TokenLedger,
}

impl ResidentWeightExecutionRunSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"steps\":{},\"total_weight_bytes\":{},\"total_latency_ns\":{},\"cpu_events\":{},\"device_events\":{},\"copy_events\":{},\"gpu_resident_steps\":{},\"gpu_staged_steps\":{},\"fallback_steps\":{},\"fallback_decisions\":{},\"block_version_dependencies\":{},\"hot_path_allocations\":{}}}",
            self.steps,
            self.total_weight_bytes,
            self.total_latency_ns,
            self.cpu_events,
            self.device_events,
            self.copy_events,
            self.gpu_resident_steps,
            self.gpu_staged_steps,
            self.fallback_steps,
            self.fallback_decisions,
            self.block_version_dependencies,
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidentWeightProbeStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightProbeSummary {
    pub status: ResidentWeightProbeStatus,
    pub blocks: usize,
    pub total_weight_bytes: usize,
    pub dram_used_bytes: usize,
    pub vram_used_bytes: usize,
    pub residency_decisions: u64,
    pub first_block_id: Option<ResidentBlockId>,
    pub last_block_id: Option<ResidentBlockId>,
    pub first_tensor: Option<String>,
    pub last_tensor: Option<String>,
    pub manifest_hash: u64,
    pub hot_path_allocations: u64,
}

impl ResidentWeightProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            ResidentWeightProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"vram_used_bytes\":{},\"residency_decisions\":{},\"first_block_id\":{},\"last_block_id\":{},\"first_tensor\":{},\"last_tensor\":{},\"manifest_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.blocks,
            self.total_weight_bytes,
            self.dram_used_bytes,
            self.vram_used_bytes,
            self.residency_decisions,
            json_opt_block_id(self.first_block_id),
            json_opt_block_id(self.last_block_id),
            json_opt_string(self.first_tensor.as_deref()),
            json_opt_string(self.last_tensor.as_deref()),
            self.manifest_hash,
            self.hot_path_allocations,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Runtime {
    config: RuntimeConfig,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticDecodeConfig {
    pub steps: u64,
    pub token_ring_capacity: usize,
    pub seed_token: TokenId,
}

impl SyntheticDecodeConfig {
    pub const fn new(steps: u64, token_ring_capacity: usize, seed_token: TokenId) -> Self {
        Self {
            steps,
            token_ring_capacity,
            seed_token,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticDecodeSummary {
    pub status: SyntheticDecodeStatus,
    pub steps: u64,
    pub token_ring_capacity: usize,
    pub token_ring_slots_touched: u64,
    pub token_ring_reuses: u64,
    pub token_ring_max_slot_version: u64,
    pub seed_token: TokenId,
    pub last_token: Option<TokenId>,
    pub graph_replays: u64,
    pub graph_replay_events: u64,
    pub kernel_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub host_wait_events: u64,
    pub soft_visibility_syncs: u64,
    pub device_timeline_active_ns: u64,
    pub device_timeline_idle_ns: u64,
    pub graph_replay_latency_ns: u64,
    pub device_latency_ns: u64,
    pub copy_latency_ns: u64,
    pub host_wait_latency_ns: u64,
    pub soft_visibility_sync_latency_ns: u64,
    pub estimated_events: u64,
    pub estimated_latency_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub observed_tokens: u64,
    pub observed_token_hash: u64,
    pub stale_tokens: u64,
    pub missing_tokens: u64,
    pub extra_tokens: u64,
    pub mismatched_tokens: u64,
    pub host_causality_edges: u64,
    pub error: Option<&'static str>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyntheticDecodeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyProbeConfig {
    pub pages: u32,
    pub page_bytes: usize,
    pub current_step: u64,
    pub hot_page_limit: usize,
    pub prefetch_distance: u64,
    pub evict_after_idle: u64,
}

impl Default for KvResidencyProbeConfig {
    fn default() -> Self {
        Self {
            pages: 4,
            page_bytes: 128,
            current_step: 10,
            hot_page_limit: 2,
            prefetch_distance: 2,
            evict_after_idle: 4,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KvResidencyProbeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyProbeSummary {
    pub status: KvResidencyProbeStatus,
    pub pages: u32,
    pub page_bytes: usize,
    pub current_step: u64,
    pub hot_page_limit: usize,
    pub decisions: u64,
    pub keep_hot: u64,
    pub keep_warm: u64,
    pub prefetches: u64,
    pub demotions: u64,
    pub evictions: u64,
    pub copy_events: u64,
    pub prefetch_events: u64,
    pub eviction_events: u64,
    pub stall_events: u64,
    pub copy_bytes: usize,
    pub changed_bytes: usize,
    pub visible_stall_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub vram_used_bytes: usize,
    pub dram_used_bytes: usize,
    pub error: Option<&'static str>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathProbeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathProbeSummary {
    pub status: TransportPathProbeStatus,
    pub requests: u64,
    pub decode_requests: u64,
    pub prefill_requests: u64,
    pub gpu_direct_paths: u64,
    pub pinned_host_paths: u64,
    pub cpu_produced_paths: u64,
    pub mapped_pinned_paths: u64,
    pub transport_events: u64,
    pub copy_events: u64,
    pub sync_events: u64,
    pub phase_handoff_syncs: u64,
    pub fallback_decisions: u64,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub explicit_copy_bytes: usize,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub estimated_events: u64,
    pub estimated_latency_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportCapabilityMatrixStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportMatrixRequestedPath {
    GpuDirectRdma,
    PinnedHostBounce,
    CpuProducedBoundary,
    MappedPinnedWrite,
}

impl TransportMatrixRequestedPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuDirectRdma => "A_GPU_DIRECT_RDMA",
            Self::PinnedHostBounce => "B_PINNED_HOST_BOUNCE",
            Self::CpuProducedBoundary => "C_CPU_PRODUCED_BOUNDARY",
            Self::MappedPinnedWrite => "D_MAPPED_PINNED_WRITE",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportCapabilityMatrixEntry {
    pub requested_path: TransportMatrixRequestedPath,
    pub size_bytes: usize,
    pub mode: TransferMode,
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
    pub selected_path: TransportPathKind,
    pub class: TransportPathClass,
    pub capability_result: CapabilityState,
    pub estimated_visible_ns: u64,
    pub effective_payload_bandwidth_bps: u64,
    pub estimated_cpu_core_ns: u64,
    pub dram_read_bytes: usize,
    pub dram_write_bytes: usize,
    pub pcie_tx_bytes: usize,
    pub pcie_rx_bytes: usize,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub pageable_copy: bool,
    pub per_token_registration: bool,
    pub registration_cache_hit: bool,
    pub queue_depth: u32,
    pub credit_stall_ns: u64,
}

impl TransportCapabilityMatrixEntry {
    pub fn to_json(self) -> String {
        format!(
            "{{\"requested_path\":\"{}\",\"size_bytes\":{},\"mode\":\"{}\",\"source_tier\":\"{}\",\"destination_tier\":\"{}\",\"selected_path\":\"{}\",\"class\":\"{}\",\"capability_result\":\"{}\",\"estimated_visible_ns\":{},\"metric_source\":\"estimated_model\",\"effective_payload_bandwidth_bps\":{},\"estimated_cpu_core_ns\":{},\"dram_read_bytes\":{},\"dram_write_bytes\":{},\"pcie_tx_bytes\":{},\"pcie_rx_bytes\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"pageable_copy\":{},\"per_token_registration\":{},\"registration_cache_hit\":{},\"queue_depth\":{},\"credit_stall_ns\":{}}}",
            self.requested_path.as_str(),
            self.size_bytes,
            self.mode.as_str(),
            memory_tier_to_str(self.source_tier),
            memory_tier_to_str(self.destination_tier),
            self.selected_path.as_str(),
            self.class.as_str(),
            self.capability_result.as_str(),
            self.estimated_visible_ns,
            self.effective_payload_bandwidth_bps,
            self.estimated_cpu_core_ns,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.pcie_tx_bytes,
            self.pcie_rx_bytes,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.pageable_copy,
            self.per_token_registration,
            self.registration_cache_hit,
            self.queue_depth,
            self.credit_stall_ns,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportCapabilityMatrixSummary {
    pub status: TransportCapabilityMatrixStatus,
    pub sizes: u64,
    pub entries: Vec<TransportCapabilityMatrixEntry>,
    pub decode_entries: u64,
    pub prefill_entries: u64,
    pub gpu_direct_entries: u64,
    pub host_staged_entries: u64,
    pub cpu_produced_entries: u64,
    pub mapped_pinned_entries: u64,
    pub supported_verified_entries: u64,
    pub supported_unverified_entries: u64,
    pub degraded_to_pinned_host_entries: u64,
    pub unsupported_entries: u64,
    pub total_estimated_visible_ns: u64,
    pub p50_estimated_visible_ns: u64,
    pub p95_estimated_visible_ns: u64,
    pub p99_estimated_visible_ns: u64,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub estimated_cpu_core_ns: u64,
    pub dram_read_bytes: usize,
    pub dram_write_bytes: usize,
    pub pcie_tx_bytes: usize,
    pub pcie_rx_bytes: usize,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub registration_cache_hits: u64,
    pub credit_stall_ns: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransportProbeAccumulator {
    ledger: TokenLedger,
    requests: u64,
    decode_requests: u64,
    prefill_requests: u64,
    gpu_direct_paths: u64,
    pinned_host_paths: u64,
    cpu_produced_paths: u64,
    mapped_pinned_paths: u64,
    nic_tx_bytes: usize,
    nic_rx_bytes: usize,
    explicit_copy_bytes: usize,
    pageable_copies: u64,
    per_token_registrations: u64,
}

impl TransportProbeAccumulator {
    fn new() -> Self {
        Self {
            ledger: TokenLedger::new(0),
            requests: 0,
            decode_requests: 0,
            prefill_requests: 0,
            gpu_direct_paths: 0,
            pinned_host_paths: 0,
            cpu_produced_paths: 0,
            mapped_pinned_paths: 0,
            nic_tx_bytes: 0,
            nic_rx_bytes: 0,
            explicit_copy_bytes: 0,
            pageable_copies: 0,
            per_token_registrations: 0,
        }
    }

    fn record(&mut self, decision: TransportPathDecision) {
        self.requests = self.requests.saturating_add(1);
        match decision.request.mode {
            TransferMode::Decode => self.decode_requests = self.decode_requests.saturating_add(1),
            TransferMode::Prefill => {
                self.prefill_requests = self.prefill_requests.saturating_add(1)
            }
        }
        match decision.path {
            TransportPathKind::TrueGpuDirectRdma => {
                self.gpu_direct_paths = self.gpu_direct_paths.saturating_add(1)
            }
            TransportPathKind::OptimizedPinnedHostBounce => {
                self.pinned_host_paths = self.pinned_host_paths.saturating_add(1)
            }
            TransportPathKind::CpuProducedBoundary => {
                self.cpu_produced_paths = self.cpu_produced_paths.saturating_add(1)
            }
            TransportPathKind::MappedPinnedHostWrite => {
                self.mapped_pinned_paths = self.mapped_pinned_paths.saturating_add(1)
            }
        }
        self.nic_tx_bytes = self.nic_tx_bytes.saturating_add(decision.nic_tx_bytes);
        self.nic_rx_bytes = self.nic_rx_bytes.saturating_add(decision.nic_rx_bytes);
        self.explicit_copy_bytes = self
            .explicit_copy_bytes
            .saturating_add(decision.explicit_copy_bytes);
        if decision.pageable_copy {
            self.pageable_copies = self.pageable_copies.saturating_add(1);
        }
        if decision.per_token_registration {
            self.per_token_registrations = self.per_token_registrations.saturating_add(1);
        }
        decision.record_to_ledger(&mut self.ledger);
    }

    fn finish(self) -> TransportPathProbeSummary {
        TransportPathProbeSummary {
            status: TransportPathProbeStatus::Ok,
            requests: self.requests,
            decode_requests: self.decode_requests,
            prefill_requests: self.prefill_requests,
            gpu_direct_paths: self.gpu_direct_paths,
            pinned_host_paths: self.pinned_host_paths,
            cpu_produced_paths: self.cpu_produced_paths,
            mapped_pinned_paths: self.mapped_pinned_paths,
            transport_events: self.ledger.event_count(LedgerEventKind::Transport),
            copy_events: self.ledger.event_count(LedgerEventKind::Copy),
            sync_events: self.ledger.event_count(LedgerEventKind::Sync),
            phase_handoff_syncs: self.ledger.sync_count_for(SyncClass::PhaseHandoff),
            fallback_decisions: self.ledger.fallback_count(),
            nic_tx_bytes: self.nic_tx_bytes,
            nic_rx_bytes: self.nic_rx_bytes,
            explicit_copy_bytes: self.explicit_copy_bytes,
            pageable_copies: self.pageable_copies,
            per_token_registrations: self.per_token_registrations,
            estimated_events: self
                .ledger
                .event_count_for_source(MetricSource::EstimatedModel),
            estimated_latency_ns: self
                .ledger
                .latency_ns_for_source(MetricSource::EstimatedModel),
            total_latency_ns: self.ledger.total_latency_ns(),
            hot_path_allocations: self.ledger.hot_path_allocations,
            error: None,
        }
    }
}

impl KvResidencyProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            KvResidencyProbeStatus::Ok => "ok",
            KvResidencyProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"pages\":{},\"page_bytes\":{},\"current_step\":{},\"hot_page_limit\":{},\"decisions\":{},\"keep_hot\":{},\"keep_warm\":{},\"prefetches\":{},\"demotions\":{},\"evictions\":{},\"copy_events\":{},\"prefetch_events\":{},\"eviction_events\":{},\"stall_events\":{},\"copy_bytes\":{},\"changed_bytes\":{},\"visible_stall_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"vram_used_bytes\":{},\"dram_used_bytes\":{},\"error\":{}}}",
            status,
            self.pages,
            self.page_bytes,
            self.current_step,
            self.hot_page_limit,
            self.decisions,
            self.keep_hot,
            self.keep_warm,
            self.prefetches,
            self.demotions,
            self.evictions,
            self.copy_events,
            self.prefetch_events,
            self.eviction_events,
            self.stall_events,
            self.copy_bytes,
            self.changed_bytes,
            self.visible_stall_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.vram_used_bytes,
            self.dram_used_bytes,
            json_opt_static_str(self.error),
        )
    }
}

impl TransportPathProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            TransportPathProbeStatus::Ok => "ok",
            TransportPathProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"requests\":{},\"decode_requests\":{},\"prefill_requests\":{},\"gpu_direct_paths\":{},\"pinned_host_paths\":{},\"cpu_produced_paths\":{},\"mapped_pinned_paths\":{},\"transport_events\":{},\"copy_events\":{},\"sync_events\":{},\"phase_handoff_syncs\":{},\"fallback_decisions\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"explicit_copy_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.requests,
            self.decode_requests,
            self.prefill_requests,
            self.gpu_direct_paths,
            self.pinned_host_paths,
            self.cpu_produced_paths,
            self.mapped_pinned_paths,
            self.transport_events,
            self.copy_events,
            self.sync_events,
            self.phase_handoff_syncs,
            self.fallback_decisions,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.explicit_copy_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.estimated_events,
            self.estimated_latency_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

impl TransportCapabilityMatrixSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            TransportCapabilityMatrixStatus::Ok => "ok",
            TransportCapabilityMatrixStatus::Failed => "failed",
        };
        let mut entries = String::from("[");
        for (index, entry) in self.entries.iter().enumerate() {
            if index != 0 {
                entries.push(',');
            }
            entries.push_str(&entry.to_json());
        }
        entries.push(']');
        format!(
            "{{\"status\":\"{}\",\"sizes\":{},\"entries_count\":{},\"decode_entries\":{},\"prefill_entries\":{},\"gpu_direct_entries\":{},\"host_staged_entries\":{},\"cpu_produced_entries\":{},\"mapped_pinned_entries\":{},\"supported_verified_entries\":{},\"supported_unverified_entries\":{},\"degraded_to_pinned_host_entries\":{},\"unsupported_entries\":{},\"total_estimated_visible_ns\":{},\"p50_estimated_visible_ns\":{},\"p95_estimated_visible_ns\":{},\"p99_estimated_visible_ns\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"estimated_cpu_core_ns\":{},\"dram_read_bytes\":{},\"dram_write_bytes\":{},\"pcie_tx_bytes\":{},\"pcie_rx_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"registration_cache_hits\":{},\"credit_stall_ns\":{},\"hot_path_allocations\":{},\"error\":{},\"entries\":{}}}",
            status,
            self.sizes,
            self.entries.len(),
            self.decode_entries,
            self.prefill_entries,
            self.gpu_direct_entries,
            self.host_staged_entries,
            self.cpu_produced_entries,
            self.mapped_pinned_entries,
            self.supported_verified_entries,
            self.supported_unverified_entries,
            self.degraded_to_pinned_host_entries,
            self.unsupported_entries,
            self.total_estimated_visible_ns,
            self.p50_estimated_visible_ns,
            self.p95_estimated_visible_ns,
            self.p99_estimated_visible_ns,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.estimated_cpu_core_ns,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.pcie_tx_bytes,
            self.pcie_rx_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.registration_cache_hits,
            self.credit_stall_ns,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
            entries,
        )
    }
}

impl SyntheticDecodeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            SyntheticDecodeStatus::Ok => "ok",
            SyntheticDecodeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"token_ring_capacity\":{},\"token_ring_slots_touched\":{},\"token_ring_reuses\":{},\"token_ring_max_slot_version\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"graph_replay_events\":{},\"kernel_events\":{},\"device_events\":{},\"copy_events\":{},\"host_wait_events\":{},\"soft_visibility_syncs\":{},\"device_timeline_active_ns\":{},\"device_timeline_idle_ns\":{},\"graph_replay_latency_ns\":{},\"device_latency_ns\":{},\"copy_latency_ns\":{},\"host_wait_latency_ns\":{},\"soft_visibility_sync_latency_ns\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"error\":{}}}",
            status,
            self.steps,
            self.token_ring_capacity,
            self.token_ring_slots_touched,
            self.token_ring_reuses,
            self.token_ring_max_slot_version,
            self.seed_token.0,
            json_opt_token(self.last_token),
            self.graph_replays,
            self.graph_replay_events,
            self.kernel_events,
            self.device_events,
            self.copy_events,
            self.host_wait_events,
            self.soft_visibility_syncs,
            self.device_timeline_active_ns,
            self.device_timeline_idle_ns,
            self.graph_replay_latency_ns,
            self.device_latency_ns,
            self.copy_latency_ns,
            self.host_wait_latency_ns,
            self.soft_visibility_sync_latency_ns,
            self.estimated_events,
            self.estimated_latency_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.observed_tokens,
            self.observed_token_hash,
            self.stale_tokens,
            self.missing_tokens,
            self.extra_tokens,
            self.mismatched_tokens,
            self.host_causality_edges,
            json_opt_static_str(self.error),
        )
    }
}

impl Runtime {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        ensure_supported_linux_host()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> RuntimeConfig {
        self.config
    }

    pub fn empty_token_ledger(&self, token_index: u64) -> TokenLedger {
        let _ = self.config;
        TokenLedger::new(token_index)
    }

    pub fn block_registry(&self, budget: ResidencyBudget) -> BlockRegistry {
        let _ = self.config;
        BlockRegistry::new([
            (MemoryTier::Vram, budget.vram_bytes),
            (MemoryTier::PinnedDram, budget.pinned_dram_bytes),
            (MemoryTier::Dram, budget.dram_bytes),
        ])
    }

    pub fn static_arenas(&self, budget: ResidencyBudget) -> StaticArenaSet {
        let _ = self.config;
        StaticArenaSet::new(
            budget.vram_bytes,
            budget.pinned_dram_bytes,
            budget.dram_bytes,
        )
    }

    pub fn static_arena_probe(&self, budget: ResidencyBudget) -> Result<StaticArenaProbeSummary> {
        let mut registry = self.block_registry(budget);
        let mut arenas = self.static_arenas(budget);
        let bootstrap = arenas
            .preallocate_decode_bootstrap(&mut registry, StaticArenaBootstrapSpec::default())?;
        let bootstrap_blocks = [
            bootstrap.device_token_state,
            bootstrap.pinned_observation,
            bootstrap.host_metadata,
        ];
        let ready_blocks = bootstrap_blocks
            .iter()
            .filter(|id| {
                registry
                    .block(**id)
                    .is_some_and(|block| block.state == ResidencyState::Ready)
            })
            .count();

        let device_used_bytes = arenas.device().used();
        let pinned_host_used_bytes = arenas.pinned_host().used();
        let host_used_bytes = arenas.host().used();

        let mut ledger = self.empty_token_ledger(0);
        let mut hot_path_rejections = 0u64;
        for (kind, name) in [
            (ArenaKind::Device, "guard-device"),
            (ArenaKind::PinnedHost, "guard-pinned-host"),
            (ArenaKind::Host, "guard-host"),
        ] {
            if arenas
                .reject_hot_path_reservation_with_ledger(kind, name, 64, 64, &mut ledger)
                .is_err()
            {
                hot_path_rejections += 1;
            }
        }
        let usage_preserved_after_rejections = arenas.device().used() == device_used_bytes
            && arenas.pinned_host().used() == pinned_host_used_bytes
            && arenas.host().used() == host_used_bytes;

        Ok(StaticArenaProbeSummary {
            device_capacity_bytes: arenas.device().capacity(),
            pinned_host_capacity_bytes: arenas.pinned_host().capacity(),
            host_capacity_bytes: arenas.host().capacity(),
            device_used_bytes,
            pinned_host_used_bytes,
            host_used_bytes,
            bootstrap_blocks: bootstrap_blocks.len(),
            ready_blocks,
            hot_path_rejections,
            hot_path_allocation_attempts: ledger.hot_path_allocations,
            usage_preserved_after_rejections,
        })
    }

    pub fn synthetic_engine(&self, token_ring_capacity: usize) -> Result<SyntheticEngine> {
        SyntheticEngine::new(token_ring_capacity, self.config.device)
    }

    pub fn discover_topology(&self) -> TopologySnapshot {
        let _ = self.config;
        discover_topology_snapshot()
    }

    pub fn discover_capabilities(&self) -> CapabilitySnapshot {
        let _ = self.config;
        let cuda_smoke = cuda_smoke();
        let cuda = match cuda_smoke.status {
            nerva_cuda::SmokeStatus::Ok => CapabilityState::SupportedAndVerified,
            nerva_cuda::SmokeStatus::Unavailable | nerva_cuda::SmokeStatus::Failed => {
                CapabilityState::Unsupported
            }
        };
        let cuda_status = match cuda_smoke.status {
            nerva_cuda::SmokeStatus::Ok => "ok",
            nerva_cuda::SmokeStatus::Unavailable => "unavailable",
            nerva_cuda::SmokeStatus::Failed => "failed",
        };
        let cuda_compute_capability = cuda_compute_capability(&cuda_smoke);
        let cuda_device_total_memory_bytes = cuda_smoke.device_total_memory_bytes;
        let cuda_pci_bus_id = cuda_smoke.pci_bus_id.clone();
        let topology = discover_topology_snapshot();
        let rdma_core_loaded = module_loaded("ib_core");
        let mlx5_core_loaded = module_loaded("mlx5_core");
        let nvidia_peer_memory_module = detect_nvidia_peer_memory_module();
        let gpu_direct_rdma = gpu_direct_rdma_capability(
            cuda,
            topology.rdma_device_count,
            nvidia_peer_memory_module.as_deref(),
        );

        CapabilitySnapshot {
            host_arch: host_arch(),
            target_os: env::consts::OS,
            target_arch: env::consts::ARCH,
            kernel_release: read_trimmed_first_line("/proc/sys/kernel/osrelease"),
            fabric: MemoryFabricKind::DiscreteExplicit,
            cuda,
            cuda_status,
            cuda_error: cuda_smoke.error,
            cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
            cuda_compute_capability,
            cuda_device_total_memory_bytes,
            cuda_pci_bus_id,
            hip: CapabilityState::Unsupported,
            hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
            nvidia_driver_version: read_trimmed_first_line("/proc/driver/nvidia/version"),
            rdma_core_loaded,
            mlx5_core_loaded,
            nvidia_peer_memory_module,
            pinned_host_staging: CapabilityState::SupportedUnverified,
            gpu_direct_rdma,
            amd_peerdirect: CapabilityState::Unsupported,
            dma_buf_export: CapabilityState::Unsupported,
            cxl: CapabilityState::Unsupported,
            topology,
        }
    }

    pub fn plan_transport_path(
        &self,
        request: TransportPathRequest,
    ) -> Result<TransportPathDecision> {
        let _ = self.config;
        if request.bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transport path request bytes must be non-zero".to_string(),
            });
        }

        if matches!(request.producer, ExecutionOwner::Cpu)
            && matches!(
                request.source_tier,
                MemoryTier::Dram | MemoryTier::PinnedDram
            )
        {
            return Ok(make_transport_decision(
                request,
                TransportPathKind::CpuProducedBoundary,
                TransportPathClass::CpuProduced,
                "CPU owns boundary result and can produce it into a registered send buffer",
                0,
            ));
        }

        if request.source_tier == MemoryTier::Vram
            && request.destination_tier == MemoryTier::Vram
            && request.gpu_direct_rdma == CapabilityState::SupportedAndVerified
        {
            return Ok(make_transport_decision(
                request,
                TransportPathKind::TrueGpuDirectRdma,
                TransportPathClass::GpuDirect,
                "verified GPU-direct RDMA path avoids host staging",
                0,
            ));
        }

        if request.source_tier == MemoryTier::Vram
            && request.mode == TransferMode::Decode
            && request.bytes <= 256 * 1024
            && request.mapped_pinned_output == CapabilityState::SupportedAndVerified
        {
            return Ok(make_transport_decision(
                request,
                TransportPathKind::MappedPinnedHostWrite,
                TransportPathClass::MappedPinned,
                "small decode payload can be written directly to mapped pinned output",
                0,
            ));
        }

        if matches!(
            request.pinned_host_staging,
            CapabilityState::SupportedAndVerified | CapabilityState::SupportedUnverified
        ) {
            let copy_bytes = if request.source_tier == MemoryTier::Vram
                && request.destination_tier == MemoryTier::Vram
            {
                request.bytes.saturating_mul(2)
            } else {
                request.bytes
            };
            return Ok(make_transport_decision(
                request,
                TransportPathKind::OptimizedPinnedHostBounce,
                TransportPathClass::HostStaged,
                "GPU-direct path is not verified; using preallocated pinned-host staging",
                copy_bytes,
            ));
        }

        Err(NervaError::BackendUnavailable {
            backend: "transport",
            reason: "no verified direct path and pinned-host staging is unavailable".to_string(),
        })
    }

    pub fn run_transport_path_probe(&self) -> Result<TransportPathProbeSummary> {
        let capabilities = self.discover_capabilities();
        let sizes = [
            (32 * 1024, TransferMode::Decode),
            (256 * 1024, TransferMode::Decode),
            (1024 * 1024, TransferMode::Decode),
            (16 * 1024 * 1024, TransferMode::Prefill),
            (64 * 1024 * 1024, TransferMode::Prefill),
            (256 * 1024 * 1024, TransferMode::Prefill),
        ];
        let mut probe = TransportProbeAccumulator::new();

        for (bytes, mode) in sizes {
            let request = TransportPathRequest::from_capabilities(
                MemoryTier::Vram,
                MemoryTier::Vram,
                bytes,
                mode,
                ExecutionOwner::Gpu(self.config.device),
                &capabilities,
            );
            let decision = self.plan_transport_path(request)?;
            probe.record(decision);
        }

        let cpu_request = TransportPathRequest::from_capabilities(
            MemoryTier::Dram,
            MemoryTier::PinnedDram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Cpu,
            &capabilities,
        );
        let cpu_decision = self.plan_transport_path(cpu_request)?;
        probe.record(cpu_decision);
        probe.ledger.require_zero_hot_path_allocations()?;
        probe.ledger.require_classified_syncs()?;

        Ok(probe.finish())
    }

    pub fn run_transport_capability_matrix_probe(
        &self,
    ) -> Result<TransportCapabilityMatrixSummary> {
        let capabilities = self.discover_capabilities();
        let sizes = [
            (32 * 1024, TransferMode::Decode),
            (256 * 1024, TransferMode::Decode),
            (1024 * 1024, TransferMode::Decode),
            (16 * 1024 * 1024, TransferMode::Prefill),
            (64 * 1024 * 1024, TransferMode::Prefill),
            (256 * 1024 * 1024, TransferMode::Prefill),
        ];
        let requested_paths = [
            TransportMatrixRequestedPath::GpuDirectRdma,
            TransportMatrixRequestedPath::PinnedHostBounce,
            TransportMatrixRequestedPath::CpuProducedBoundary,
            TransportMatrixRequestedPath::MappedPinnedWrite,
        ];
        let mut entries = Vec::with_capacity(sizes.len() * requested_paths.len());
        let ledger = TokenLedger::new(0);

        for (bytes, mode) in sizes {
            for requested_path in requested_paths {
                let request = transport_matrix_request(
                    requested_path,
                    bytes,
                    mode,
                    self.config.device,
                    &capabilities,
                );
                let decision = self.plan_transport_path(request)?;
                let capability_result =
                    transport_matrix_capability_result(requested_path, decision, &capabilities);
                let resource = transport_resource_estimate(decision);
                entries.push(TransportCapabilityMatrixEntry {
                    requested_path,
                    size_bytes: bytes,
                    mode,
                    source_tier: decision.request.source_tier,
                    destination_tier: decision.request.destination_tier,
                    selected_path: decision.path,
                    class: decision.class,
                    capability_result,
                    estimated_visible_ns: decision.estimated_visible_ns,
                    effective_payload_bandwidth_bps: effective_payload_bandwidth_bps(
                        decision.request.bytes,
                        decision.estimated_visible_ns,
                    ),
                    estimated_cpu_core_ns: resource.estimated_cpu_core_ns,
                    dram_read_bytes: resource.dram_read_bytes,
                    dram_write_bytes: resource.dram_write_bytes,
                    pcie_tx_bytes: resource.pcie_tx_bytes,
                    pcie_rx_bytes: resource.pcie_rx_bytes,
                    explicit_copy_bytes: decision.explicit_copy_bytes,
                    nic_tx_bytes: decision.nic_tx_bytes,
                    nic_rx_bytes: decision.nic_rx_bytes,
                    pageable_copy: decision.pageable_copy,
                    per_token_registration: decision.per_token_registration,
                    registration_cache_hit: resource.registration_cache_hit,
                    queue_depth: resource.queue_depth,
                    credit_stall_ns: resource.credit_stall_ns,
                });
            }
        }

        ledger.require_zero_hot_path_allocations()?;
        Ok(transport_capability_matrix_summary(
            sizes.len() as u64,
            entries,
            ledger.hot_path_allocations,
        ))
    }

    pub fn materialize_hf_weight_manifest(
        &self,
        manifest: &nerva_model::HfTensorManifest,
    ) -> Result<ResidentWeightTable> {
        self.materialize_hf_weight_manifest_with_budget(
            manifest,
            ResidencyBudget::new(0, 0, manifest.total_weight_bytes),
        )
    }

    pub fn materialize_hf_weight_manifest_with_budget(
        &self,
        manifest: &nerva_model::HfTensorManifest,
        budget: ResidencyBudget,
    ) -> Result<ResidentWeightTable> {
        let _ = self.config;
        let mut registry = self.block_registry(budget);
        let mut ledger = TokenLedger::new(0);
        let mut entries = Vec::with_capacity(manifest.entries.len());
        let mut materialized_bytes = 0usize;

        for entry in &manifest.entries {
            let block_id = registry.allocate(
                BlockAllocationRequest::new(BlockKind::Weight, entry.tier, entry.bytes)
                    .with_dtype(entry.dtype)
                    .with_layout(LayoutId(weight_role_layout_id(entry.role))),
            )?;
            registry.mark_ready(block_id)?;
            materialized_bytes = materialized_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident weight byte count overflow".to_string(),
                }
            })?;
            ledger.record_residency_decision(ResidencyDecision {
                block_id,
                old_tier: MemoryTier::Disk,
                new_tier: entry.tier,
                executor_selected: ExecutionOwner::Cpu,
                candidate_costs: vec![
                    CandidateCost::estimated("cold-disk-backing", entry.bytes as u64),
                    CandidateCost::estimated("resident-dram-backing", 0),
                ],
                reason: "initialize exact weight block as DRAM-resident immutable backing",
                predicted_overlap_ns: 0,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            entries.push(ResidentWeightBlockRef {
                name: entry.name.clone(),
                block_id,
                bytes: entry.bytes,
                dtype: entry.dtype,
                tier: entry.tier,
                source_shard: None,
                file_offset_begin: None,
                file_offset_end: None,
            });
        }

        if materialized_bytes != manifest.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight byte count does not match manifest".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightTable {
            registry,
            entries,
            total_weight_bytes: materialized_bytes,
            manifest_hash: manifest.manifest_hash,
            ledger,
        })
    }

    pub fn run_resident_weight_probe(&self) -> Result<ResidentWeightProbeSummary> {
        let manifest = nerva_model::hf_tensor_manifest_probe()?.manifest;
        let table = self.materialize_hf_weight_manifest(&manifest)?;
        let first = table.entries.first();
        let last = table.entries.last();

        Ok(ResidentWeightProbeSummary {
            status: ResidentWeightProbeStatus::Ok,
            blocks: table.entries.len(),
            total_weight_bytes: table.total_weight_bytes,
            dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
            vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
            residency_decisions: table.ledger.residency_decisions.len() as u64,
            first_block_id: first.map(|entry| entry.block_id),
            last_block_id: last.map(|entry| entry.block_id),
            first_tensor: first.map(|entry| entry.name.clone()),
            last_tensor: last.map(|entry| entry.name.clone()),
            manifest_hash: table.manifest_hash,
            hot_path_allocations: table.ledger.hot_path_allocations,
        })
    }

    pub fn materialize_safetensors_shard_plan(
        &self,
        plan: &nerva_model::SafetensorsShardPlan,
    ) -> Result<ResidentWeightTable> {
        self.materialize_safetensors_shard_plan_with_budget(
            plan,
            ResidencyBudget::new(0, 0, plan.total_weight_bytes),
        )
    }

    pub fn materialize_safetensors_shard_plan_with_budget(
        &self,
        plan: &nerva_model::SafetensorsShardPlan,
        budget: ResidencyBudget,
    ) -> Result<ResidentWeightTable> {
        let _ = self.config;
        let mut registry = self.block_registry(budget);
        let mut ledger = TokenLedger::new(0);
        let mut entries = Vec::with_capacity(plan.entries.len());
        let mut materialized_bytes = 0usize;

        for entry in &plan.entries {
            let block_id = registry.allocate(
                BlockAllocationRequest::new(BlockKind::Weight, entry.tier, entry.bytes)
                    .with_dtype(entry.dtype)
                    .with_layout(LayoutId(weight_role_layout_id(entry.role))),
            )?;
            registry.mark_ready(block_id)?;
            materialized_bytes = materialized_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident shard-plan weight byte count overflow".to_string(),
                }
            })?;
            ledger.record_residency_decision(ResidencyDecision {
                block_id,
                old_tier: MemoryTier::Disk,
                new_tier: entry.tier,
                executor_selected: ExecutionOwner::Cpu,
                candidate_costs: vec![
                    CandidateCost::estimated("safetensors-shard-read", entry.bytes as u64),
                    CandidateCost::estimated("file-offset-begin", entry.file_offset_begin as u64),
                ],
                reason: "initialize exact sharded safetensors weight block as resident immutable backing",
                predicted_overlap_ns: 0,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            entries.push(ResidentWeightBlockRef {
                name: entry.tensor_name.clone(),
                block_id,
                bytes: entry.bytes,
                dtype: entry.dtype,
                tier: entry.tier,
                source_shard: Some(entry.shard_file.clone()),
                file_offset_begin: Some(entry.file_offset_begin),
                file_offset_end: Some(entry.file_offset_end),
            });
        }

        if materialized_bytes != plan.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident shard-plan weight byte count does not match plan".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightTable {
            registry,
            entries,
            total_weight_bytes: materialized_bytes,
            manifest_hash: plan.manifest_hash,
            ledger,
        })
    }

    pub fn plan_resident_weight_prefetch(
        &self,
        table: &ResidentWeightTable,
        max_task_bytes: usize,
    ) -> Result<ResidentWeightPrefetchPlan> {
        if max_task_bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight prefetch max_task_bytes must be non-zero".to_string(),
            });
        }

        let mut tasks = Vec::new();
        let mut ledger = TokenLedger::new(0);
        let mut total_bytes = 0usize;
        let mut shards = BTreeMap::new();

        for entry in &table.entries {
            let source_shard =
                entry
                    .source_shard
                    .as_ref()
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!("resident weight {} has no source shard", entry.name),
                    })?;
            let file_offset_begin =
                entry
                    .file_offset_begin
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "resident weight {} has no source file begin offset",
                            entry.name
                        ),
                    })?;
            let file_offset_end =
                entry
                    .file_offset_end
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "resident weight {} has no source file end offset",
                            entry.name
                        ),
                    })?;
            if file_offset_end < file_offset_begin
                || file_offset_end - file_offset_begin != entry.bytes
            {
                return Err(NervaError::InvalidArgument {
                    reason: format!("resident weight {} source span is invalid", entry.name),
                });
            }
            shards.insert(source_shard.clone(), ());

            let mut cursor = file_offset_begin;
            while cursor < file_offset_end {
                let remaining = file_offset_end - cursor;
                let bytes = remaining.min(max_task_bytes);
                let task_index = tasks.len() as u64;
                let file_end =
                    cursor
                        .checked_add(bytes)
                        .ok_or_else(|| NervaError::AllocationFailed {
                            bytes,
                            reason: "resident weight prefetch file offset overflow".to_string(),
                        })?;
                total_bytes =
                    total_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| NervaError::AllocationFailed {
                            bytes,
                            reason: "resident weight prefetch byte count overflow".to_string(),
                        })?;
                ledger.record(LedgerEvent {
                    kind: LedgerEventKind::Prefetch,
                    sync_class: None,
                    metric_source: MetricSource::EstimatedModel,
                    block_id: Some(entry.block_id),
                    from_tier: Some(MemoryTier::Disk),
                    to_tier: Some(MemoryTier::PinnedDram),
                    bytes,
                    latency_ns: 0,
                    label: "weight_prefetch_scheduled",
                });
                ledger.record(LedgerEvent {
                    kind: LedgerEventKind::Copy,
                    sync_class: None,
                    metric_source: MetricSource::EstimatedModel,
                    block_id: Some(entry.block_id),
                    from_tier: Some(MemoryTier::PinnedDram),
                    to_tier: Some(entry.tier),
                    bytes,
                    latency_ns: 0,
                    label: "weight_prefetch_copy",
                });
                tasks.push(ResidentWeightPrefetchTask {
                    task_index,
                    block_id: entry.block_id,
                    name: entry.name.clone(),
                    source_shard: source_shard.clone(),
                    file_offset_begin: cursor,
                    file_offset_end: file_end,
                    bytes,
                    target_tier: entry.tier,
                });
                cursor = file_end;
            }
        }

        if total_bytes != table.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight prefetch byte count does not match table".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        let first_source_shard = tasks.first().map(|task| task.source_shard.clone());
        let last_source_shard = tasks.last().map(|task| task.source_shard.clone());
        Ok(ResidentWeightPrefetchPlan {
            prefetch_events: ledger.event_count(LedgerEventKind::Prefetch),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            tasks,
            total_bytes,
            shard_count: shards.len(),
            max_task_bytes,
            first_source_shard,
            last_source_shard,
            ledger,
        })
    }

    pub fn execute_resident_weight_prefetch_plan(
        &self,
        table: &mut ResidentWeightTable,
        plan: &ResidentWeightPrefetchPlan,
    ) -> Result<ResidentWeightPrefetchExecutionSummary> {
        if plan.total_bytes != table.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight prefetch plan bytes do not match table".to_string(),
            });
        }

        let mut ledger = TokenLedger::new(0);
        let mut bytes_by_block = BTreeMap::new();
        let mut total_bytes = 0usize;

        for task in &plan.tasks {
            let block =
                table
                    .registry
                    .block(task.block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "prefetch task references unknown block {}",
                            task.block_id.0
                        ),
                    })?;
            if block.kind != BlockKind::Weight || block.tier != task.target_tier {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "prefetch task block {} does not match resident weight",
                        task.block_id.0
                    ),
                });
            }
            if task.file_offset_end < task.file_offset_begin
                || task.file_offset_end - task.file_offset_begin != task.bytes
            {
                return Err(NervaError::InvalidArgument {
                    reason: format!("prefetch task {} has invalid file span", task.task_index),
                });
            }

            let first_task_for_block = !bytes_by_block.contains_key(&task.block_id);
            if first_task_for_block {
                table
                    .registry
                    .transition(task.block_id, ResidencyState::Prefetching)?;
            }
            let block_bytes = bytes_by_block.entry(task.block_id).or_insert(0usize);
            *block_bytes = block_bytes.checked_add(task.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: task.bytes,
                    reason: "prefetch task byte accounting overflow".to_string(),
                }
            })?;
            total_bytes = total_bytes.checked_add(task.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: task.bytes,
                    reason: "prefetch execution byte accounting overflow".to_string(),
                }
            })?;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Prefetch,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: Some(task.block_id),
                from_tier: Some(MemoryTier::Disk),
                to_tier: Some(MemoryTier::PinnedDram),
                bytes: task.bytes,
                latency_ns: 0,
                label: "weight_prefetch_execute",
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: Some(task.block_id),
                from_tier: Some(MemoryTier::PinnedDram),
                to_tier: Some(task.target_tier),
                bytes: task.bytes,
                latency_ns: 0,
                label: "weight_prefetch_commit",
            });
        }

        for (block_id, bytes) in &bytes_by_block {
            let block =
                table
                    .registry
                    .block(*block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "prefetch completion references unknown block {}",
                            block_id.0
                        ),
                    })?;
            if *bytes != block.bytes {
                return Err(NervaError::InvalidArgument {
                    reason: format!("prefetch completion for block {} is incomplete", block_id.0),
                });
            }
            table.registry.mark_ready(*block_id)?;
        }

        if total_bytes != table.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "prefetch execution bytes do not match table".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        let ready_blocks = table
            .entries
            .iter()
            .filter(|entry| {
                table
                    .registry
                    .block(entry.block_id)
                    .is_some_and(|block| block.state == ResidencyState::Ready)
            })
            .count();

        Ok(ResidentWeightPrefetchExecutionSummary {
            tasks: plan.tasks.len(),
            completed_blocks: bytes_by_block.len(),
            total_bytes,
            prefetch_events: ledger.event_count(LedgerEventKind::Prefetch),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            ready_blocks,
            hot_path_allocations: ledger.hot_path_allocations,
            ledger,
        })
    }

    pub fn promote_resident_weight_hotset(
        &self,
        table: &mut ResidentWeightTable,
        max_promote_bytes: usize,
    ) -> Result<ResidentWeightHotsetSummary> {
        if max_promote_bytes == 0 {
            return Ok(ResidentWeightHotsetSummary {
                promoted_blocks: 0,
                promoted_bytes: 0,
                dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
                vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
                residency_decisions: 0,
                first_promoted_tensor: None,
                last_promoted_tensor: None,
                hot_path_allocations: table.ledger.hot_path_allocations,
            });
        }

        let mut promoted_blocks = 0usize;
        let mut promoted_bytes = 0usize;
        let mut first_promoted_tensor = None;
        let mut last_promoted_tensor = None;
        let decision_start = table.ledger.residency_decisions.len();

        for (index, entry) in table.entries.iter_mut().enumerate() {
            if entry.tier == MemoryTier::Vram {
                continue;
            }
            let next_promoted_bytes = promoted_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident weight hotset byte count overflow".to_string(),
                }
            })?;
            if next_promoted_bytes > max_promote_bytes {
                break;
            }
            if table
                .registry
                .remaining_bytes(MemoryTier::Vram)
                .unwrap_or(0)
                < entry.bytes
            {
                break;
            }

            let allocation = AllocationId(10_000 + index as u64);
            table.registry.move_block(
                entry.block_id,
                MemoryTier::Vram,
                allocation,
                promoted_bytes as u64,
            )?;
            table.registry.mark_ready(entry.block_id)?;
            table.ledger.record_residency_decision(ResidencyDecision {
                block_id: entry.block_id,
                old_tier: entry.tier,
                new_tier: MemoryTier::Vram,
                executor_selected: ExecutionOwner::Gpu(self.config.device),
                candidate_costs: vec![
                    CandidateCost::estimated("keep-dram", entry.bytes as u64),
                    CandidateCost::estimated("promote-vram-hotset", 0),
                ],
                reason: "promote bounded exact weight hotset to VRAM",
                predicted_overlap_ns: 0,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            entry.tier = MemoryTier::Vram;
            promoted_blocks += 1;
            promoted_bytes = next_promoted_bytes;
            if first_promoted_tensor.is_none() {
                first_promoted_tensor = Some(entry.name.clone());
            }
            last_promoted_tensor = Some(entry.name.clone());
        }

        table.ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightHotsetSummary {
            promoted_blocks,
            promoted_bytes,
            dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
            vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
            residency_decisions: (table.ledger.residency_decisions.len() - decision_start) as u64,
            first_promoted_tensor,
            last_promoted_tensor,
            hot_path_allocations: table.ledger.hot_path_allocations,
        })
    }

    pub fn plan_resident_weight_execution(
        &self,
        table: &ResidentWeightTable,
        max_steps: usize,
        compute_capability: Option<u32>,
    ) -> Result<ResidentWeightExecutionPlan> {
        if max_steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution max_steps must be non-zero".to_string(),
            });
        }

        let registry = bootstrap_registry();
        let mut ledger = TokenLedger::new(0);
        let mut steps = Vec::new();
        let mut total_weight_bytes = 0usize;
        let mut total_predicted_visible_ns = 0u64;
        let mut cpu_steps = 0u64;
        let mut gpu_resident_steps = 0u64;
        let mut gpu_staged_steps = 0u64;
        let mut fallback_steps = 0u64;

        for (index, entry) in table.entries.iter().take(max_steps).enumerate() {
            let block = table.registry.block(entry.block_id).ok_or_else(|| {
                NervaError::InvalidArgument {
                    reason: format!("resident weight {} references unknown block", entry.name),
                }
            })?;
            if block.kind != BlockKind::Weight
                || block.tier != entry.tier
                || block.dtype != entry.dtype
            {
                return Err(NervaError::InvalidArgument {
                    reason: format!("resident weight {} block metadata drifted", entry.name),
                });
            }
            if block.state != ResidencyState::Ready {
                return Err(NervaError::InvalidArgument {
                    reason: format!("resident weight {} is not Ready", entry.name),
                });
            }
            ledger.record_block_version_dependency(BlockVersionDependency {
                block_id: entry.block_id,
                required_version: block.version,
                observed_version: block.version,
                label: "resident_weight_execution_plan",
            });

            let cuda_plan = registry.resolve(KernelQuery::new(
                KernelOperation::DenseMatVec,
                KernelBackend::Cuda,
                entry.dtype,
                compute_capability,
            ))?;
            let cpu_direct = registry
                .resolve(KernelQuery::new(
                    KernelOperation::DenseMatVec,
                    KernelBackend::CpuReference,
                    entry.dtype,
                    None,
                ))
                .ok()
                .and_then(|plan| match plan {
                    KernelPlan::Direct { implementation } => Some(implementation),
                    KernelPlan::Fallback { .. } => None,
                });

            let (strategy, executor, predicted_visible_ns, kernel_name, fallback, reason) =
                match cuda_plan {
                    KernelPlan::Direct { implementation } => {
                        if entry.tier == MemoryTier::Vram {
                            (
                                ResidentWeightExecutionStrategy::GpuResident,
                                ExecutionOwner::Gpu(self.config.device),
                                estimate_gpu_resident_weight_ns(entry.bytes),
                                implementation.name,
                                false,
                                "weight is already resident in VRAM",
                            )
                        } else if let Some(cpu_implementation) = cpu_direct {
                            let cpu_ns = estimate_cpu_dram_weight_ns(entry.bytes);
                            let staged_ns = estimate_gpu_staged_weight_ns(entry.bytes);
                            if cpu_ns <= staged_ns {
                                (
                                    ResidentWeightExecutionStrategy::CpuDram,
                                    ExecutionOwner::Cpu,
                                    cpu_ns,
                                    cpu_implementation.name,
                                    false,
                                    "CPU compute wins for DRAM-resident weight",
                                )
                            } else {
                                (
                                    ResidentWeightExecutionStrategy::GpuStaged,
                                    ExecutionOwner::Gpu(self.config.device),
                                    staged_ns,
                                    implementation.name,
                                    false,
                                    "GPU staged compute wins despite transfer",
                                )
                            }
                        } else {
                            (
                                ResidentWeightExecutionStrategy::GpuStaged,
                                ExecutionOwner::Gpu(self.config.device),
                                estimate_gpu_staged_weight_ns(entry.bytes),
                                implementation.name,
                                false,
                                "no exact CPU contract; use declared GPU staged kernel",
                            )
                        }
                    }
                    KernelPlan::Fallback {
                        fallback: implementation,
                        ..
                    } => (
                        ResidentWeightExecutionStrategy::CpuExactFallback,
                        ExecutionOwner::Cpu,
                        estimate_cpu_fallback_weight_ns(entry.bytes, entry.tier),
                        implementation.name,
                        true,
                        "CUDA request selected exact named CPU fallback",
                    ),
                };

            total_weight_bytes = total_weight_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident weight execution byte count overflow".to_string(),
                }
            })?;
            total_predicted_visible_ns = total_predicted_visible_ns
                .checked_add(predicted_visible_ns)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: predicted_visible_ns as usize,
                    reason: "resident weight execution visible cost overflow".to_string(),
                })?;
            match strategy {
                ResidentWeightExecutionStrategy::CpuDram
                | ResidentWeightExecutionStrategy::CpuExactFallback => cpu_steps += 1,
                ResidentWeightExecutionStrategy::GpuResident => gpu_resident_steps += 1,
                ResidentWeightExecutionStrategy::GpuStaged => gpu_staged_steps += 1,
            }
            fallback_steps += u64::from(fallback);

            ledger.record_execution_decision(ExecutionDecision {
                operation: "resident_weight_dense_matvec",
                executor_selected: executor,
                candidate_costs: vec![
                    CandidateCost::estimated("cpu-dram", estimate_cpu_dram_weight_ns(entry.bytes)),
                    CandidateCost::estimated(
                        "gpu-resident",
                        estimate_gpu_resident_weight_ns(entry.bytes),
                    ),
                    CandidateCost::estimated(
                        "gpu-staged",
                        estimate_gpu_staged_weight_ns(entry.bytes),
                    ),
                ],
                reason,
                predicted_visible_ns,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            if fallback {
                ledger.record_fallback_decision(FallbackDecision {
                    label: "resident_weight_exact_cpu_fallback",
                    class: FallbackClass::ExactNamed,
                    requested: "cuda_dense_matvec",
                    selected: kernel_name,
                    reason,
                    visible_ns: Some(predicted_visible_ns),
                    metric_source: MetricSource::EstimatedModel,
                });
            }
            steps.push(ResidentWeightExecutionStep {
                step_index: index as u64,
                block_id: entry.block_id,
                name: entry.name.clone(),
                strategy,
                executor,
                bytes: entry.bytes,
                block_version: block.version,
                predicted_visible_ns,
                kernel_name,
                fallback,
            });
        }

        if steps.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution has no steps".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;
        ledger.require_satisfied_block_versions()?;

        let first_tensor = steps.first().map(|step| step.name.clone());
        let last_tensor = steps.last().map(|step| step.name.clone());
        Ok(ResidentWeightExecutionPlan {
            steps,
            total_weight_bytes,
            total_predicted_visible_ns,
            cpu_steps,
            gpu_resident_steps,
            gpu_staged_steps,
            fallback_steps,
            fallback_decisions: ledger.fallback_count(),
            block_version_dependencies: ledger.block_version_dependencies.len() as u64,
            first_tensor,
            last_tensor,
            ledger,
        })
    }

    pub fn execute_resident_weight_execution_plan(
        &self,
        table: &ResidentWeightTable,
        plan: &ResidentWeightExecutionPlan,
    ) -> Result<ResidentWeightExecutionRunSummary> {
        if plan.steps.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution run has no steps".to_string(),
            });
        }

        let mut ledger = TokenLedger::new(0);
        let mut total_weight_bytes = 0usize;
        let mut gpu_resident_steps = 0u64;
        let mut gpu_staged_steps = 0u64;
        let mut fallback_steps = 0u64;

        for step in &plan.steps {
            let block =
                table
                    .registry
                    .block(step.block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "execution step references unknown block {}",
                            step.block_id.0
                        ),
                    })?;
            if block.kind != BlockKind::Weight || block.bytes != step.bytes {
                return Err(NervaError::InvalidArgument {
                    reason: format!("execution step {} block metadata drifted", step.step_index),
                });
            }
            if block.state != ResidencyState::Ready {
                return Err(NervaError::InvalidArgument {
                    reason: format!("execution step {} block is not Ready", step.step_index),
                });
            }
            ledger.record_block_version_dependency(BlockVersionDependency {
                block_id: step.block_id,
                required_version: step.block_version,
                observed_version: block.version,
                label: "resident_weight_execution_run",
            });
            ledger.require_satisfied_block_versions()?;
            let entry = table
                .entries
                .iter()
                .find(|entry| entry.block_id == step.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("execution step {} has no table entry", step.step_index),
                })?;
            if entry.name != step.name || entry.tier != block.tier {
                return Err(NervaError::InvalidArgument {
                    reason: format!("execution step {} table entry drifted", step.step_index),
                });
            }

            total_weight_bytes = total_weight_bytes.checked_add(step.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: step.bytes,
                    reason: "resident weight execution run byte count overflow".to_string(),
                }
            })?;

            match step.strategy {
                ResidentWeightExecutionStrategy::CpuDram => {
                    if block.tier != MemoryTier::Dram {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "CPU DRAM step {} is not DRAM-resident",
                                step.step_index
                            ),
                        });
                    }
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::CpuActivity,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(step.block_id),
                        from_tier: Some(MemoryTier::Dram),
                        to_tier: Some(MemoryTier::Dram),
                        bytes: step.bytes,
                        latency_ns: step.predicted_visible_ns,
                        label: "resident_weight_cpu_dram_matvec",
                    });
                }
                ResidentWeightExecutionStrategy::GpuResident => {
                    if block.tier != MemoryTier::Vram {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "GPU resident step {} is not VRAM-resident",
                                step.step_index
                            ),
                        });
                    }
                    gpu_resident_steps += 1;
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::DeviceActivity,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(step.block_id),
                        from_tier: Some(MemoryTier::Vram),
                        to_tier: Some(MemoryTier::Vram),
                        bytes: step.bytes,
                        latency_ns: step.predicted_visible_ns,
                        label: "resident_weight_gpu_matvec",
                    });
                }
                ResidentWeightExecutionStrategy::GpuStaged => {
                    if block.tier == MemoryTier::Vram {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "GPU staged step {} is already VRAM-resident",
                                step.step_index
                            ),
                        });
                    }
                    gpu_staged_steps += 1;
                    let copy_ns = div_ceil_u64(step.bytes as u64, 24);
                    let compute_ns = estimate_gpu_resident_weight_ns(step.bytes);
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(step.block_id),
                        from_tier: Some(block.tier),
                        to_tier: Some(MemoryTier::Vram),
                        bytes: step.bytes,
                        latency_ns: copy_ns,
                        label: "resident_weight_stage_to_gpu",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::DeviceActivity,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(step.block_id),
                        from_tier: Some(MemoryTier::Vram),
                        to_tier: Some(MemoryTier::Vram),
                        bytes: step.bytes,
                        latency_ns: compute_ns,
                        label: "resident_weight_gpu_staged_matvec",
                    });
                }
                ResidentWeightExecutionStrategy::CpuExactFallback => {
                    fallback_steps += 1;
                    ledger.record_fallback_decision(FallbackDecision {
                        label: "resident_weight_exact_cpu_fallback_run",
                        class: FallbackClass::ExactNamed,
                        requested: "cuda_dense_matvec",
                        selected: step.kernel_name,
                        reason: "executing declared exact CPU fallback step",
                        visible_ns: Some(step.predicted_visible_ns),
                        metric_source: MetricSource::EstimatedModel,
                    });
                    if block.tier == MemoryTier::Vram || block.tier == MemoryTier::SharedHbmOrLpddr
                    {
                        let copy_ns = div_ceil_u64(step.bytes as u64, 24);
                        ledger.record(LedgerEvent {
                            kind: LedgerEventKind::Copy,
                            sync_class: None,
                            metric_source: MetricSource::EstimatedModel,
                            block_id: Some(step.block_id),
                            from_tier: Some(block.tier),
                            to_tier: Some(MemoryTier::Dram),
                            bytes: step.bytes,
                            latency_ns: copy_ns,
                            label: "resident_weight_fallback_to_cpu",
                        });
                    }
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::CpuActivity,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(step.block_id),
                        from_tier: Some(MemoryTier::Dram),
                        to_tier: Some(MemoryTier::Dram),
                        bytes: step.bytes,
                        latency_ns: estimate_cpu_dram_weight_ns(step.bytes),
                        label: "resident_weight_cpu_exact_fallback",
                    });
                }
            }
        }

        if total_weight_bytes != plan.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution run bytes do not match plan".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightExecutionRunSummary {
            steps: plan.steps.len(),
            total_weight_bytes,
            total_latency_ns: ledger.total_latency_ns(),
            cpu_events: ledger.event_count(LedgerEventKind::CpuActivity),
            device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            gpu_resident_steps,
            gpu_staged_steps,
            fallback_steps,
            fallback_decisions: ledger.fallback_count(),
            block_version_dependencies: ledger.block_version_dependencies.len() as u64,
            hot_path_allocations: ledger.hot_path_allocations,
            ledger,
        })
    }

    pub fn run_synthetic_decode(
        &self,
        config: SyntheticDecodeConfig,
    ) -> Result<SyntheticDecodeSummary> {
        if config.steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "synthetic decode steps must be non-zero".to_string(),
            });
        }

        let mut engine = self.synthetic_engine(config.token_ring_capacity)?;
        let mut last_token = None;
        let mut graph_replay_events: u64 = 0;
        let mut kernel_events: u64 = 0;
        let mut device_events: u64 = 0;
        let mut copy_events: u64 = 0;
        let mut host_wait_events: u64 = 0;
        let mut soft_visibility_syncs: u64 = 0;
        let mut device_timeline_active_ns: u64 = 0;
        let mut device_timeline_idle_ns: u64 = 0;
        let mut graph_replay_latency_ns: u64 = 0;
        let mut device_latency_ns: u64 = 0;
        let mut copy_latency_ns: u64 = 0;
        let mut host_wait_latency_ns: u64 = 0;
        let mut soft_visibility_sync_latency_ns: u64 = 0;
        let mut estimated_events: u64 = 0;
        let mut estimated_latency_ns: u64 = 0;
        let mut total_latency_ns: u64 = 0;
        let mut hot_path_allocations: u64 = 0;
        let mut observed_tokens: u64 = 0;
        let mut observed_token_hash: u64 = TOKEN_STREAM_HASH_SEED;
        let mut stale_tokens: u64 = 0;
        let mut extra_tokens: u64 = 0;
        let mut mismatched_tokens: u64 = 0;
        let mut host_causality_edges: u64 = 0;
        let mut token_ring_slots_seen = vec![false; config.token_ring_capacity];
        let mut token_ring_slots_touched: u64 = 0;
        let mut token_ring_reuses: u64 = 0;
        let mut token_ring_max_slot_version: u64 = 0;

        for token_index in 0..config.steps {
            let output = engine
                .launch_device_next(RequestId(1), SequenceId(1), token_index, config.seed_token)?
                .collect()?;
            output.ledger.require_zero_hot_path_allocations()?;
            output.ledger.require_classified_syncs()?;
            let token_graph_events = output.ledger.event_count(LedgerEventKind::GraphReplay);
            let token_device_events = output.ledger.event_count(LedgerEventKind::DeviceActivity);
            let token_kernel_events = output.ledger.event_count(LedgerEventKind::KernelLaunch)
                + token_graph_events
                + token_device_events;
            let token_copy_events = output.ledger.event_count(LedgerEventKind::Copy);
            let token_host_wait_events = output.ledger.event_count(LedgerEventKind::Sync);
            let token_soft_visibility_syncs =
                output.ledger.sync_count_for(SyncClass::SoftVisibilitySync);

            graph_replay_events += token_graph_events;
            kernel_events += token_kernel_events;
            device_events += token_device_events;
            copy_events += token_copy_events;
            host_wait_events += token_host_wait_events;
            soft_visibility_syncs += token_soft_visibility_syncs;
            device_timeline_active_ns = device_timeline_active_ns
                .saturating_add(output.ledger.device_active_ns(self.config.device)?);
            device_timeline_idle_ns = device_timeline_idle_ns
                .saturating_add(output.ledger.device_idle_ns(self.config.device)?);
            graph_replay_latency_ns = graph_replay_latency_ns
                .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::GraphReplay));
            device_latency_ns = device_latency_ns.saturating_add(
                output
                    .ledger
                    .latency_ns_for(LedgerEventKind::DeviceActivity),
            );
            copy_latency_ns =
                copy_latency_ns.saturating_add(output.ledger.latency_ns_for(LedgerEventKind::Copy));
            host_wait_latency_ns = host_wait_latency_ns
                .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::Sync));
            soft_visibility_sync_latency_ns = soft_visibility_sync_latency_ns.saturating_add(
                output
                    .ledger
                    .sync_latency_ns_for(SyncClass::SoftVisibilitySync),
            );
            estimated_events = estimated_events.saturating_add(
                output
                    .ledger
                    .event_count_for_source(MetricSource::EstimatedModel),
            );
            estimated_latency_ns = estimated_latency_ns.saturating_add(
                output
                    .ledger
                    .latency_ns_for_source(MetricSource::EstimatedModel),
            );
            total_latency_ns = total_latency_ns.saturating_add(output.ledger.total_latency_ns());
            hot_path_allocations =
                hot_path_allocations.saturating_add(output.ledger.hot_path_allocations);
            observed_tokens = observed_tokens.saturating_add(1);
            observed_token_hash =
                hash_observed_token(observed_token_hash, output.token_index, output.token);
            if let Some(seen) = token_ring_slots_seen.get_mut(output.device_token_ref.slot_index) {
                if !*seen {
                    *seen = true;
                    token_ring_slots_touched = token_ring_slots_touched.saturating_add(1);
                }
            }
            if output.device_token_ref.version > 1 {
                token_ring_reuses = token_ring_reuses.saturating_add(1);
            }
            token_ring_max_slot_version =
                token_ring_max_slot_version.max(output.device_token_ref.version);
            if output.token_index < token_index {
                stale_tokens = stale_tokens.saturating_add(1);
            } else if output.token_index > token_index {
                extra_tokens = extra_tokens.saturating_add(1);
            }
            let expected_token = TokenId(
                config
                    .seed_token
                    .0
                    .wrapping_add((token_index as u32).wrapping_add(1)),
            );
            if output.token != expected_token {
                mismatched_tokens = mismatched_tokens.saturating_add(1);
            }
            if output.input_source == TokenInputSource::HostObservation {
                host_causality_edges = host_causality_edges.saturating_add(1);
            }
            last_token = Some(output.token);
        }
        let missing_tokens = config.steps.saturating_sub(observed_tokens);
        if stale_tokens != 0
            || missing_tokens != 0
            || extra_tokens != 0
            || mismatched_tokens != 0
            || host_causality_edges != 0
        {
            return Err(NervaError::ResidencyViolation {
                block_id: nerva_core::ResidentBlockId(0),
                reason: "synthetic device token audit failed".to_string(),
            });
        }

        Ok(SyntheticDecodeSummary {
            status: SyntheticDecodeStatus::Ok,
            steps: config.steps,
            token_ring_capacity: config.token_ring_capacity,
            token_ring_slots_touched,
            token_ring_reuses,
            token_ring_max_slot_version,
            seed_token: config.seed_token,
            last_token,
            graph_replays: engine
                .graph_pool()
                .replay_count(GraphKey {
                    bucket: 1,
                    max_blocks: 1,
                })
                .unwrap_or(0),
            graph_replay_events,
            kernel_events,
            device_events,
            copy_events,
            host_wait_events,
            soft_visibility_syncs,
            device_timeline_active_ns,
            device_timeline_idle_ns,
            graph_replay_latency_ns,
            device_latency_ns,
            copy_latency_ns,
            host_wait_latency_ns,
            soft_visibility_sync_latency_ns,
            estimated_events,
            estimated_latency_ns,
            total_latency_ns,
            hot_path_allocations,
            observed_tokens,
            observed_token_hash,
            stale_tokens,
            missing_tokens,
            extra_tokens,
            mismatched_tokens,
            host_causality_edges,
            error: None,
        })
    }

    pub fn run_kv_residency_probe(
        &self,
        config: KvResidencyProbeConfig,
    ) -> Result<KvResidencyProbeSummary> {
        if config.pages < 4 {
            return Err(NervaError::InvalidArgument {
                reason: "KV residency probe requires at least four pages".to_string(),
            });
        }
        if config.page_bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "KV residency probe page size must be non-zero".to_string(),
            });
        }

        let total_bytes = config
            .page_bytes
            .checked_mul(config.pages as usize)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: config.page_bytes,
                reason: "KV residency probe byte count overflow".to_string(),
            })?;
        let mut arenas = StaticArenaSet::new(0, 0, total_bytes);
        let mut registry = self.block_registry(ResidencyBudget::new(total_bytes, 0, total_bytes));
        let mut pool = nerva_memory::KvPagePool::preallocate(
            &mut arenas,
            &mut registry,
            config.pages,
            KvPageSpec::new(
                0,
                0,
                16,
                config.page_bytes,
                MemoryTier::Dram,
                ArenaKind::Host,
                64,
            ),
        )?;

        let active = pool.allocate_page(0, 16, config.current_step.saturating_sub(1))?;
        pool.set_next_use(active.page_index, Some(config.current_step))?;

        let soon = pool.allocate_page(16, 16, config.current_step.saturating_sub(2))?;
        pool.cache_page(
            soon.page_index,
            KvPrefixKey {
                hash: [2; 32],
                group_id: 0,
            },
            16,
        )?;
        pool.set_next_use(soon.page_index, Some(config.current_step.saturating_add(1)))?;

        let cold = pool.allocate_page(32, 16, 0)?;
        pool.cache_page(
            cold.page_index,
            KvPrefixKey {
                hash: [3; 32],
                group_id: 0,
            },
            16,
        )?;

        let warm_vram = pool.allocate_page(48, 16, config.current_step.saturating_sub(1))?;
        pool.cache_page(
            warm_vram.page_index,
            KvPrefixKey {
                hash: [4; 32],
                group_id: 0,
            },
            16,
        )?;

        pool.release_page(soon.page_index, config.current_step.saturating_sub(2))?;
        pool.release_page(cold.page_index, 0)?;
        pool.release_page(warm_vram.page_index, config.current_step.saturating_sub(1))?;

        registry.move_block(
            warm_vram.block_id,
            MemoryTier::Vram,
            AllocationId(warm_vram.block_id.0),
            0,
        )?;
        registry.mark_ready(warm_vram.block_id)?;

        let policy = KvResidencyPolicy::new(
            config.hot_page_limit,
            config.prefetch_distance,
            config.evict_after_idle,
        );
        let plan = KvResidencyPlanner::plan(&pool, &registry, config.current_step, policy)?;
        let mut ledger = TokenLedger::new(config.current_step);
        plan.record_to_ledger(&mut ledger);
        plan.apply(&mut registry)?;
        ledger.require_zero_hot_path_allocations()?;

        let copy_bytes = ledger
            .events
            .iter()
            .filter(|event| event.kind == LedgerEventKind::Copy)
            .map(|event| event.bytes)
            .sum();

        Ok(KvResidencyProbeSummary {
            status: KvResidencyProbeStatus::Ok,
            pages: config.pages,
            page_bytes: config.page_bytes,
            current_step: config.current_step,
            hot_page_limit: config.hot_page_limit,
            decisions: ledger.residency_decisions.len() as u64,
            keep_hot: plan.action_count(KvResidencyAction::KeepHot),
            keep_warm: plan.action_count(KvResidencyAction::KeepWarm),
            prefetches: plan.action_count(KvResidencyAction::PrefetchToHot),
            demotions: plan.action_count(KvResidencyAction::DemoteToWarm),
            evictions: plan.action_count(KvResidencyAction::EvictCold),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            prefetch_events: ledger.event_count(LedgerEventKind::Prefetch),
            eviction_events: ledger.event_count(LedgerEventKind::Eviction),
            stall_events: ledger.event_count(LedgerEventKind::Stall),
            copy_bytes,
            changed_bytes: plan.changed_bytes(),
            visible_stall_ns: ledger.latency_ns_for(LedgerEventKind::Stall),
            total_latency_ns: ledger.total_latency_ns(),
            hot_path_allocations: ledger.hot_path_allocations,
            vram_used_bytes: registry.used_bytes(MemoryTier::Vram),
            dram_used_bytes: registry.used_bytes(MemoryTier::Dram),
            error: None,
        })
    }
}

pub fn cuda_smoke() -> nerva_cuda::CudaSmokeSummary {
    nerva_cuda::smoke()
}

pub fn cuda_synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> nerva_cuda::CudaSyntheticGraphSummary {
    nerva_cuda::synthetic_graph_smoke(steps, ring_capacity, seed_token)
}

fn cuda_compute_capability(summary: &nerva_cuda::CudaSmokeSummary) -> Option<String> {
    Some(format!(
        "{}.{}",
        summary.compute_capability_major?, summary.compute_capability_minor?
    ))
}

fn module_loaded(name: &str) -> bool {
    fs::metadata(format!("/sys/module/{name}"))
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
}

fn detect_nvidia_peer_memory_module() -> Option<String> {
    ["nvidia_peermem", "nv_peer_mem"]
        .into_iter()
        .find(|name| module_loaded(name))
        .map(ToOwned::to_owned)
}

fn gpu_direct_rdma_capability(
    cuda: CapabilityState,
    rdma_device_count: usize,
    nvidia_peer_memory_module: Option<&str>,
) -> CapabilityState {
    if cuda == CapabilityState::SupportedAndVerified
        && rdma_device_count > 0
        && nvidia_peer_memory_module.is_some()
    {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::DegradedToPinnedHost
    }
}

fn transport_matrix_request(
    requested_path: TransportMatrixRequestedPath,
    bytes: usize,
    mode: TransferMode,
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> TransportPathRequest {
    match requested_path {
        TransportMatrixRequestedPath::GpuDirectRdma => TransportPathRequest::from_capabilities(
            MemoryTier::Vram,
            MemoryTier::Vram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            capabilities,
        ),
        TransportMatrixRequestedPath::PinnedHostBounce => TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::Vram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        ),
        TransportMatrixRequestedPath::CpuProducedBoundary => TransportPathRequest::new(
            MemoryTier::Dram,
            MemoryTier::PinnedDram,
            bytes,
            mode,
            ExecutionOwner::Cpu,
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        ),
        TransportMatrixRequestedPath::MappedPinnedWrite => TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::PinnedDram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        ),
    }
}

fn transport_matrix_capability_result(
    requested_path: TransportMatrixRequestedPath,
    decision: TransportPathDecision,
    capabilities: &CapabilitySnapshot,
) -> CapabilityState {
    match requested_path {
        TransportMatrixRequestedPath::GpuDirectRdma => {
            if decision.class == TransportPathClass::GpuDirect {
                CapabilityState::SupportedAndVerified
            } else if decision.class == TransportPathClass::HostStaged {
                CapabilityState::DegradedToPinnedHost
            } else {
                CapabilityState::Unsupported
            }
        }
        TransportMatrixRequestedPath::PinnedHostBounce => {
            if decision.class == TransportPathClass::HostStaged {
                capabilities.pinned_host_staging
            } else {
                CapabilityState::Unsupported
            }
        }
        TransportMatrixRequestedPath::CpuProducedBoundary => {
            if decision.class == TransportPathClass::CpuProduced {
                CapabilityState::SupportedAndVerified
            } else {
                CapabilityState::Unsupported
            }
        }
        TransportMatrixRequestedPath::MappedPinnedWrite => {
            if decision.class == TransportPathClass::MappedPinned {
                CapabilityState::SupportedAndVerified
            } else if decision.class == TransportPathClass::HostStaged {
                CapabilityState::DegradedToPinnedHost
            } else {
                CapabilityState::Unsupported
            }
        }
    }
}

fn transport_capability_matrix_summary(
    sizes: u64,
    entries: Vec<TransportCapabilityMatrixEntry>,
    hot_path_allocations: u64,
) -> TransportCapabilityMatrixSummary {
    let decode_entries = entries
        .iter()
        .filter(|entry| entry.mode == TransferMode::Decode)
        .count() as u64;
    let prefill_entries = entries
        .iter()
        .filter(|entry| entry.mode == TransferMode::Prefill)
        .count() as u64;
    let gpu_direct_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::GpuDirect)
        .count() as u64;
    let host_staged_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::HostStaged)
        .count() as u64;
    let cpu_produced_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::CpuProduced)
        .count() as u64;
    let mapped_pinned_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::MappedPinned)
        .count() as u64;
    let supported_verified_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::SupportedAndVerified)
        .count() as u64;
    let supported_unverified_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::SupportedUnverified)
        .count() as u64;
    let degraded_to_pinned_host_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::DegradedToPinnedHost)
        .count() as u64;
    let unsupported_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::Unsupported)
        .count() as u64;
    let total_estimated_visible_ns = entries.iter().map(|entry| entry.estimated_visible_ns).sum();
    let p50_estimated_visible_ns = percentile_estimated_visible_ns(&entries, 50);
    let p95_estimated_visible_ns = percentile_estimated_visible_ns(&entries, 95);
    let p99_estimated_visible_ns = percentile_estimated_visible_ns(&entries, 99);
    let explicit_copy_bytes = entries.iter().map(|entry| entry.explicit_copy_bytes).sum();
    let nic_tx_bytes = entries.iter().map(|entry| entry.nic_tx_bytes).sum();
    let nic_rx_bytes = entries.iter().map(|entry| entry.nic_rx_bytes).sum();
    let estimated_cpu_core_ns = entries
        .iter()
        .map(|entry| entry.estimated_cpu_core_ns)
        .sum();
    let dram_read_bytes = entries.iter().map(|entry| entry.dram_read_bytes).sum();
    let dram_write_bytes = entries.iter().map(|entry| entry.dram_write_bytes).sum();
    let pcie_tx_bytes = entries.iter().map(|entry| entry.pcie_tx_bytes).sum();
    let pcie_rx_bytes = entries.iter().map(|entry| entry.pcie_rx_bytes).sum();
    let pageable_copies = entries.iter().filter(|entry| entry.pageable_copy).count() as u64;
    let per_token_registrations = entries
        .iter()
        .filter(|entry| entry.per_token_registration)
        .count() as u64;
    let registration_cache_hits = entries
        .iter()
        .filter(|entry| entry.registration_cache_hit)
        .count() as u64;
    let credit_stall_ns = entries.iter().map(|entry| entry.credit_stall_ns).sum();

    TransportCapabilityMatrixSummary {
        status: TransportCapabilityMatrixStatus::Ok,
        sizes,
        entries,
        decode_entries,
        prefill_entries,
        gpu_direct_entries,
        host_staged_entries,
        cpu_produced_entries,
        mapped_pinned_entries,
        supported_verified_entries,
        supported_unverified_entries,
        degraded_to_pinned_host_entries,
        unsupported_entries,
        total_estimated_visible_ns,
        p50_estimated_visible_ns,
        p95_estimated_visible_ns,
        p99_estimated_visible_ns,
        explicit_copy_bytes,
        nic_tx_bytes,
        nic_rx_bytes,
        estimated_cpu_core_ns,
        dram_read_bytes,
        dram_write_bytes,
        pcie_tx_bytes,
        pcie_rx_bytes,
        pageable_copies,
        per_token_registrations,
        registration_cache_hits,
        credit_stall_ns,
        hot_path_allocations,
        error: None,
    }
}

fn percentile_estimated_visible_ns(
    entries: &[TransportCapabilityMatrixEntry],
    percentile: u64,
) -> u64 {
    if entries.is_empty() {
        return 0;
    }
    let mut values = entries
        .iter()
        .map(|entry| entry.estimated_visible_ns)
        .collect::<Vec<_>>();
    values.sort_unstable();
    let rank = div_ceil_u64(percentile.saturating_mul(values.len() as u64), 100).saturating_sub(1)
        as usize;
    values[rank.min(values.len() - 1)]
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TransportResourceEstimate {
    estimated_cpu_core_ns: u64,
    dram_read_bytes: usize,
    dram_write_bytes: usize,
    pcie_tx_bytes: usize,
    pcie_rx_bytes: usize,
    registration_cache_hit: bool,
    queue_depth: u32,
    credit_stall_ns: u64,
}

fn transport_resource_estimate(decision: TransportPathDecision) -> TransportResourceEstimate {
    let bytes = decision.request.bytes;
    let explicit_half = decision.explicit_copy_bytes / 2;
    let queue_depth = match decision.request.mode {
        TransferMode::Decode => 1,
        TransferMode::Prefill => 4,
    };
    let estimated_cpu_core_ns = match (decision.path, decision.request.mode) {
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Decode) => 300,
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Prefill) => 800,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Decode) => 1_000,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Prefill) => 2_500,
        (TransportPathKind::CpuProducedBoundary, TransferMode::Decode) => {
            decision.estimated_visible_ns / 2
        }
        (TransportPathKind::CpuProducedBoundary, TransferMode::Prefill) => {
            decision.estimated_visible_ns / 3
        }
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Decode) => 800,
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Prefill) => 1_800,
    };

    match decision.path {
        TransportPathKind::TrueGpuDirectRdma => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: 0,
            dram_write_bytes: 0,
            pcie_tx_bytes: decision.nic_tx_bytes,
            pcie_rx_bytes: decision.nic_rx_bytes,
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
        TransportPathKind::OptimizedPinnedHostBounce => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: decision.nic_tx_bytes,
            dram_write_bytes: decision.nic_rx_bytes.saturating_add(explicit_half),
            pcie_tx_bytes: decision.nic_tx_bytes.saturating_add(explicit_half),
            pcie_rx_bytes: decision.nic_rx_bytes.saturating_add(explicit_half),
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
        TransportPathKind::CpuProducedBoundary => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: bytes,
            dram_write_bytes: bytes,
            pcie_tx_bytes: decision.nic_tx_bytes,
            pcie_rx_bytes: decision.nic_rx_bytes,
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
        TransportPathKind::MappedPinnedHostWrite => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: decision.nic_tx_bytes,
            dram_write_bytes: bytes,
            pcie_tx_bytes: bytes.saturating_add(decision.nic_tx_bytes),
            pcie_rx_bytes: decision.nic_rx_bytes,
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
    }
}

fn effective_payload_bandwidth_bps(bytes: usize, latency_ns: u64) -> u64 {
    if latency_ns == 0 {
        return 0;
    }
    let bps = (bytes as u128).saturating_mul(1_000_000_000) / latency_ns as u128;
    bps.min(u64::MAX as u128) as u64
}

fn make_transport_decision(
    request: TransportPathRequest,
    path: TransportPathKind,
    class: TransportPathClass,
    reason: &'static str,
    explicit_copy_bytes: usize,
) -> TransportPathDecision {
    TransportPathDecision {
        path,
        class,
        request,
        reason,
        estimated_visible_ns: estimate_transport_visible_ns(path, request.bytes, request.mode),
        explicit_copy_bytes,
        nic_tx_bytes: request.bytes,
        nic_rx_bytes: request.bytes,
        pageable_copy: false,
        per_token_registration: false,
    }
}

fn estimate_transport_visible_ns(path: TransportPathKind, bytes: usize, mode: TransferMode) -> u64 {
    let setup_ns = match (path, mode) {
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Decode) => 900,
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Prefill) => 1_500,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Decode) => 2_500,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Prefill) => 4_000,
        (TransportPathKind::CpuProducedBoundary, TransferMode::Decode) => 1_200,
        (TransportPathKind::CpuProducedBoundary, TransferMode::Prefill) => 2_400,
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Decode) => 1_700,
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Prefill) => 4_500,
    };
    let bytes_per_ns = match path {
        TransportPathKind::TrueGpuDirectRdma => 64,
        TransportPathKind::OptimizedPinnedHostBounce => 24,
        TransportPathKind::CpuProducedBoundary => 48,
        TransportPathKind::MappedPinnedHostWrite => 32,
    };
    setup_ns + div_ceil_u64(bytes as u64, bytes_per_ns)
}

fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    value / divisor + u64::from(value % divisor != 0)
}

const TOKEN_STREAM_HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn hash_observed_token(current: u64, token_index: u64, token: TokenId) -> u64 {
    let mut hash = current ^ token_index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash = hash.rotate_left(13) ^ u64::from(token.0);
    hash.wrapping_mul(0xff51_afd7_ed55_8ccd)
}

fn estimate_cpu_dram_weight_ns(bytes: usize) -> u64 {
    250 + div_ceil_u64(bytes as u64, 8)
}

fn estimate_gpu_resident_weight_ns(bytes: usize) -> u64 {
    80 + div_ceil_u64(bytes as u64, 128)
}

fn estimate_gpu_staged_weight_ns(bytes: usize) -> u64 {
    5_000 + div_ceil_u64(bytes as u64, 24) + estimate_gpu_resident_weight_ns(bytes)
}

fn estimate_cpu_fallback_weight_ns(bytes: usize, tier: MemoryTier) -> u64 {
    let copy_ns = match tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => div_ceil_u64(bytes as u64, 24),
        _ => 0,
    };
    copy_ns + estimate_cpu_dram_weight_ns(bytes)
}

fn discover_topology_snapshot() -> TopologySnapshot {
    let cpu_online = read_trimmed_first_line("/sys/devices/system/cpu/online");
    let cpu_count = cpu_online
        .as_deref()
        .and_then(count_linux_id_list)
        .filter(|count| *count > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        });
    let numa_node_count = read_trimmed_first_line("/sys/devices/system/node/online")
        .as_deref()
        .and_then(count_linux_id_list)
        .filter(|count| *count > 0)
        .unwrap_or_else(|| count_prefixed_entries("/sys/devices/system/node", "node").max(1));
    let pci = pci_class_counts("/sys/bus/pci/devices");
    let iommu_group_count = count_dirs("/sys/kernel/iommu_groups");
    let kernel_cmdline = read_trimmed_first_line("/proc/cmdline");
    let iommu_kernel_args = kernel_cmdline
        .as_deref()
        .and_then(extract_iommu_kernel_args);
    let iommu_mode = discover_iommu_mode(iommu_group_count, iommu_kernel_args.as_deref());
    let rdma_device_names = list_entry_names("/sys/class/infiniband");
    let rdma_netdev_links = rdma_netdev_links("/sys/class/infiniband", &rdma_device_names);

    TopologySnapshot {
        cpu_online,
        cpu_count,
        numa_node_count,
        pci_device_count: pci.total,
        pci_root_complex_count: count_prefixed_entries("/sys/devices", "pci"),
        pci_bus_count: count_entries("/sys/class/pci_bus"),
        pci_gpu_count: pci.gpu,
        pci_network_count: pci.network,
        pci_nvme_count: pci.nvme,
        block_device_count: count_entries("/sys/block"),
        nvme_block_device_count: count_prefixed_entries("/sys/block", "nvme"),
        rdma_device_count: rdma_device_names.len(),
        rdma_device_names,
        rdma_netdev_links,
        iommu_group_count,
        iommu_mode,
        iommu_kernel_args,
    }
}

fn extract_iommu_kernel_args(cmdline: &str) -> Option<String> {
    let args = cmdline
        .split_whitespace()
        .filter(|arg| arg.contains("iommu"))
        .collect::<Vec<_>>();
    (!args.is_empty()).then(|| args.join(" "))
}

fn discover_iommu_mode(iommu_group_count: usize, iommu_kernel_args: Option<&str>) -> String {
    let args = iommu_kernel_args.unwrap_or_default();
    if has_kernel_arg(args, &["iommu=off", "intel_iommu=off", "amd_iommu=off"]) {
        return "disabled_by_kernel_arg".to_string();
    }
    if iommu_group_count > 0 && has_kernel_arg(args, &["iommu=pt"]) {
        return "passthrough_groups_present".to_string();
    }
    if iommu_group_count > 0 {
        return "enabled_groups_present".to_string();
    }
    if has_kernel_arg(args, &["iommu=pt"]) {
        return "passthrough_requested".to_string();
    }
    if has_kernel_arg(args, &["iommu=on", "intel_iommu=on", "amd_iommu=on"]) {
        return "enabled_requested".to_string();
    }
    "not_detected".to_string()
}

fn has_kernel_arg(args: &str, candidates: &[&str]) -> bool {
    args.split_whitespace()
        .any(|arg| candidates.iter().any(|candidate| arg == *candidate))
}

fn rdma_netdev_links(root: &str, rdma_device_names: &[String]) -> Vec<String> {
    let mut links = Vec::new();
    for rdma in rdma_device_names {
        let netdev_path = format!("{root}/{rdma}/device/net");
        let netdevs = list_entry_names(&netdev_path);
        if netdevs.is_empty() {
            links.push(format!("{rdma}:"));
        } else {
            links.extend(netdevs.into_iter().map(|netdev| format!("{rdma}:{netdev}")));
        }
    }
    links.sort();
    links
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct PciClassCounts {
    total: usize,
    gpu: usize,
    network: usize,
    nvme: usize,
}

fn pci_class_counts(path: &str) -> PciClassCounts {
    let Ok(entries) = fs::read_dir(path) else {
        return PciClassCounts::default();
    };
    let mut counts = PciClassCounts::default();
    for entry in entries.flatten() {
        counts.total = counts.total.saturating_add(1);
        let class_path = entry.path().join("class");
        let Some(class) = read_trimmed_first_line(&class_path.to_string_lossy()) else {
            continue;
        };
        let Some(class_value) = parse_pci_class(&class) else {
            continue;
        };
        let base_class = ((class_value >> 16) & 0xff) as u8;
        let subclass = ((class_value >> 8) & 0xff) as u8;
        let programming_interface = (class_value & 0xff) as u8;
        if base_class == 0x03 {
            counts.gpu = counts.gpu.saturating_add(1);
        }
        if base_class == 0x02 {
            counts.network = counts.network.saturating_add(1);
        }
        if base_class == 0x01 && subclass == 0x08 && programming_interface == 0x02 {
            counts.nvme = counts.nvme.saturating_add(1);
        }
    }
    counts
}

fn parse_pci_class(value: &str) -> Option<u32> {
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}

fn count_linux_id_list(value: &str) -> Option<usize> {
    let mut total = 0usize;
    for part in value.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<usize>().ok()?;
            let end = end.trim().parse::<usize>().ok()?;
            if end < start {
                return None;
            }
            total = total.checked_add(end - start + 1)?;
        } else {
            part.parse::<usize>().ok()?;
            total = total.checked_add(1)?;
        }
    }
    Some(total)
}

fn count_entries(path: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries.flatten().count()
}

fn count_dirs(path: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .count()
}

fn count_prefixed_entries(path: &str, prefix: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
        .count()
}

fn list_entry_names(path: &str) -> Vec<String> {
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut names = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn read_trimmed_first_line(path: &str) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn weight_role_layout_id(role: nerva_model::WeightBlockRole) -> u32 {
    match role {
        nerva_model::WeightBlockRole::TokenEmbedding => 1,
        nerva_model::WeightBlockRole::AttentionNorm => 2,
        nerva_model::WeightBlockRole::QueryProjection => 3,
        nerva_model::WeightBlockRole::KeyProjection => 4,
        nerva_model::WeightBlockRole::ValueProjection => 5,
        nerva_model::WeightBlockRole::OutputProjection => 6,
        nerva_model::WeightBlockRole::MlpNorm => 7,
        nerva_model::WeightBlockRole::GateProjection => 8,
        nerva_model::WeightBlockRole::UpProjection => 9,
        nerva_model::WeightBlockRole::DownProjection => 10,
        nerva_model::WeightBlockRole::LmHead => 11,
    }
}

fn json_opt_token(value: Option<TokenId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
}

fn json_opt_block_id(value: Option<ResidentBlockId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn host_arch_to_str(value: HostArch) -> &'static str {
    match value {
        HostArch::X86_64 => "x86_64",
        HostArch::Aarch64 => "aarch64",
        HostArch::Other => "other",
    }
}

fn memory_fabric_to_str(value: MemoryFabricKind) -> &'static str {
    match value {
        MemoryFabricKind::DiscreteExplicit => "DiscreteExplicit",
        MemoryFabricKind::UnifiedVirtualManaged => "UnifiedVirtualManaged",
        MemoryFabricKind::CoherentSharedPhysical => "CoherentSharedPhysical",
        MemoryFabricKind::CxlCoherentFabric => "CxlCoherentFabric",
    }
}

fn memory_tier_to_str(value: MemoryTier) -> &'static str {
    match value {
        MemoryTier::Vram => "VRAM",
        MemoryTier::SharedHbmOrLpddr => "SHARED_HBM_OR_LPDDR",
        MemoryTier::PinnedDram => "PINNED_DRAM",
        MemoryTier::Dram => "DRAM",
        MemoryTier::Cxl => "CXL",
        MemoryTier::Disk => "DISK",
    }
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn json_opt_string(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{}\"", value))
}

#[cfg(test)]
mod tests;
