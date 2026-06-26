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
use crate::transport::{
    self, TransportCapabilityMatrixSummary, TransportPathDecision, TransportPathProbeSummary,
    TransportPathRequest,
};
#[cfg(test)]
use crate::transport::{
    TransferMode, TransportCapabilityMatrixStatus, TransportPathClass, TransportPathKind,
    TransportPathProbeStatus,
};

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
        transport::plan_transport_path(request)
    }

    pub fn run_transport_path_probe(&self) -> Result<TransportPathProbeSummary> {
        let capabilities = self.discover_capabilities();
        transport::run_transport_path_probe(self.config.device, &capabilities)
    }

    pub fn run_transport_capability_matrix_probe(
        &self,
    ) -> Result<TransportCapabilityMatrixSummary> {
        let capabilities = self.discover_capabilities();
        transport::run_transport_capability_matrix_probe(self.config.device, &capabilities)
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
