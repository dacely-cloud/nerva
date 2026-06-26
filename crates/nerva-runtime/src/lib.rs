#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::{collections::BTreeMap, env, fs};

use nerva_core::{
    AllocationId, BlockKind, DeviceOrdinal, ExecutionOwner, HostArch, LayoutId, MemoryFabricKind,
    MemoryTier, NervaError, RequestId, ResidencyState, ResidentBlockId, Result, SequenceId,
    TokenId, TransactionId, ensure_supported_linux_host, host_arch,
};
use nerva_kernel_contracts::{
    KernelBackend, KernelOperation, KernelPlan, KernelQuery, bootstrap_registry,
};
use nerva_ledger::{
    BlockVersionDependency, CandidateCost, DeviceTimelineSpan, ExecutionDecision, FallbackClass,
    FallbackDecision, LedgerEvent, LedgerEventKind, MetricSource, ResidencyDecision, SyncClass,
    TokenLedger,
};
use nerva_memory::{
    ArenaKind, BlockAllocationRequest, BlockRegistry, KvPageSpec, KvPrefixKey, KvResidencyAction,
    KvResidencyPlanner, KvResidencyPolicy, StaticArenaSet,
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
    pub hip: CapabilityState,
    pub hip_visible_devices: Option<String>,
    pub nvidia_driver_version: Option<String>,
    pub pinned_host_staging: CapabilityState,
    pub gpu_direct_rdma: CapabilityState,
    pub amd_peerdirect: CapabilityState,
    pub dma_buf_export: CapabilityState,
    pub cxl: CapabilityState,
}

impl CapabilitySnapshot {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"host_arch\":\"{}\",\"target_os\":\"{}\",\"target_arch\":\"{}\",\"kernel_release\":{},\"fabric\":\"{}\",\"cuda\":\"{}\",\"cuda_status\":\"{}\",\"cuda_error\":{},\"cuda_visible_devices\":{},\"hip\":\"{}\",\"hip_visible_devices\":{},\"nvidia_driver_version\":{},\"pinned_host_staging\":\"{}\",\"gpu_direct_rdma\":\"{}\",\"amd_peerdirect\":\"{}\",\"dma_buf_export\":\"{}\",\"cxl\":\"{}\"}}",
            host_arch_to_str(self.host_arch),
            self.target_os,
            self.target_arch,
            json_opt_string(self.kernel_release.as_deref()),
            memory_fabric_to_str(self.fabric),
            self.cuda.as_str(),
            self.cuda_status,
            json_opt_string(self.cuda_error.as_deref()),
            json_opt_string(self.cuda_visible_devices.as_deref()),
            self.hip.as_str(),
            json_opt_string(self.hip_visible_devices.as_deref()),
            json_opt_string(self.nvidia_driver_version.as_deref()),
            self.pinned_host_staging.as_str(),
            self.gpu_direct_rdma.as_str(),
            self.amd_peerdirect.as_str(),
            self.dma_buf_export.as_str(),
            self.cxl.as_str(),
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphLayoutHash(pub [u8; 32]);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphFingerprint(pub [u8; 32]);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GraphKey {
    pub bucket: u32,
    pub max_blocks: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphLayout {
    pub key: GraphKey,
    pub token_ring_capacity: u32,
    pub static_address_count: u32,
}

impl GraphLayout {
    pub const fn new(
        bucket: u32,
        max_blocks: u32,
        token_ring_capacity: u32,
        static_address_count: u32,
    ) -> Self {
        Self {
            key: GraphKey { bucket, max_blocks },
            token_ring_capacity,
            static_address_count,
        }
    }

    pub fn hash(self) -> GraphLayoutHash {
        let mut out = [0u8; 32];
        mix_u32(&mut out, 0, self.key.bucket);
        mix_u32(&mut out, 4, self.key.max_blocks);
        mix_u32(&mut out, 8, self.token_ring_capacity);
        mix_u32(&mut out, 12, self.static_address_count);
        GraphLayoutHash(out)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphDescriptor {
    pub key: GraphKey,
    pub layout_hash: GraphLayoutHash,
    pub fingerprint: GraphFingerprint,
    pub replay_count: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GraphPool {
    graphs: BTreeMap<GraphKey, CapturedGraphDescriptor>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, graph: CapturedGraphDescriptor) {
        self.graphs.insert(graph.key, graph);
    }

    pub fn capture_synthetic(&mut self, layout: GraphLayout) {
        let hash = layout.hash();
        self.insert(CapturedGraphDescriptor {
            key: layout.key,
            layout_hash: hash,
            fingerprint: GraphFingerprint(hash.0),
            replay_count: 0,
        });
    }

    pub fn check_before_replay(&self, layout: GraphLayout) -> Result<&CapturedGraphDescriptor> {
        let graph = self
            .graphs
            .get(&layout.key)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!(
                    "missing captured graph bucket={} max_blocks={}",
                    layout.key.bucket, layout.key.max_blocks
                ),
            })?;
        if graph.layout_hash != layout.hash() {
            return Err(NervaError::InvalidArgument {
                reason: "captured graph layout hash does not match replay layout".to_string(),
            });
        }
        Ok(graph)
    }

    pub fn replay(&mut self, layout: GraphLayout) -> Result<()> {
        self.check_before_replay(layout)?;
        let graph = self
            .graphs
            .get_mut(&layout.key)
            .expect("graph was checked before replay");
        graph.replay_count = graph.replay_count.saturating_add(1);
        Ok(())
    }

    pub fn replay_count(&self, key: GraphKey) -> Option<u64> {
        self.graphs.get(&key).map(|graph| graph.replay_count)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeviceTokenCompletion {
    Empty,
    DeviceComplete,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenRef {
    pub slot_index: usize,
    pub token_index: u64,
    pub version: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenInput {
    pub token: TokenId,
    pub token_ref: DeviceTokenRef,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenInputSource {
    Seed,
    DeviceRing(DeviceTokenRef),
    HostObservation,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenSlot {
    pub request_id: Option<RequestId>,
    pub sequence_id: Option<SequenceId>,
    pub token_index: u64,
    pub token: Option<TokenId>,
    pub version: u64,
    pub completion: DeviceTokenCompletion,
    pub host_copied: bool,
}

impl Default for DeviceTokenSlot {
    fn default() -> Self {
        Self {
            request_id: None,
            sequence_id: None,
            token_index: 0,
            token: None,
            version: 0,
            completion: DeviceTokenCompletion::Empty,
            host_copied: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceTokenRing {
    slots: Vec<DeviceTokenSlot>,
}

impl DeviceTokenRing {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "device token ring capacity must be non-zero".to_string(),
            });
        }
        Ok(Self {
            slots: vec![DeviceTokenSlot::default(); capacity],
        })
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn slot(&self, slot_index: usize) -> Option<&DeviceTokenSlot> {
        self.slots.get(slot_index)
    }

    pub fn publish(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        token: TokenId,
    ) -> Result<DeviceTokenRef> {
        let slot_index = self.slot_index(token_index);
        let slot = &mut self.slots[slot_index];
        if slot.completion == DeviceTokenCompletion::DeviceComplete
            && !slot.host_copied
            && slot.token_index != token_index
        {
            return Err(NervaError::ResidencyViolation {
                block_id: nerva_core::ResidentBlockId(0),
                reason: "device token ring slot reused before host observation".to_string(),
            });
        }
        slot.request_id = Some(request_id);
        slot.sequence_id = Some(sequence_id);
        slot.token_index = token_index;
        slot.token = Some(token);
        slot.version = slot.version.saturating_add(1);
        slot.completion = DeviceTokenCompletion::DeviceComplete;
        slot.host_copied = false;
        Ok(DeviceTokenRef {
            slot_index,
            token_index,
            version: slot.version,
        })
    }

    pub fn consume_device_input(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
        previous_token_index: u64,
    ) -> Result<TokenId> {
        self.consume_device_input_ref(request_id, sequence_id, previous_token_index)
            .map(|input| input.token)
    }

    pub fn consume_device_input_ref(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
        previous_token_index: u64,
    ) -> Result<DeviceTokenInput> {
        let slot_index = self.slot_index(previous_token_index);
        let slot = &self.slots[slot_index];
        if slot.request_id != Some(request_id)
            || slot.sequence_id != Some(sequence_id)
            || slot.token_index != previous_token_index
            || slot.completion != DeviceTokenCompletion::DeviceComplete
        {
            return Err(NervaError::ResidencyViolation {
                block_id: nerva_core::ResidentBlockId(0),
                reason: "device token ring read was stale or incomplete".to_string(),
            });
        }
        let token = slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: nerva_core::ResidentBlockId(0),
            reason: "device token ring slot has no token".to_string(),
        })?;
        Ok(DeviceTokenInput {
            token,
            token_ref: DeviceTokenRef {
                slot_index,
                token_index: previous_token_index,
                version: slot.version,
            },
        })
    }

    pub fn host_observe(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
    ) -> Result<TokenId> {
        let slot_index = self.slot_index(token_index);
        let slot = &mut self.slots[slot_index];
        if slot.request_id != Some(request_id)
            || slot.sequence_id != Some(sequence_id)
            || slot.token_index != token_index
            || slot.completion != DeviceTokenCompletion::DeviceComplete
        {
            return Err(NervaError::ResidencyViolation {
                block_id: nerva_core::ResidentBlockId(0),
                reason: "host token observation read stale device state".to_string(),
            });
        }
        slot.host_copied = true;
        slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: nerva_core::ResidentBlockId(0),
            reason: "host-visible token slot has no token".to_string(),
        })
    }

    fn slot_index(&self, token_index: u64) -> usize {
        token_index as usize % self.slots.len()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticStepPlan {
    pub transaction_id: TransactionId,
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
    pub input_token: TokenId,
    pub input_source: TokenInputSource,
    pub layout: GraphLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
    pub input_source: TokenInputSource,
    pub device_token_ref: DeviceTokenRef,
    pub token: TokenId,
    pub finished: bool,
    pub ledger: TokenLedger,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntheticEngine {
    graph_pool: GraphPool,
    token_ring: DeviceTokenRing,
    next_transaction_id: u64,
    layout: GraphLayout,
    device: DeviceOrdinal,
}

impl SyntheticEngine {
    pub fn new(token_ring_capacity: usize, device: DeviceOrdinal) -> Result<Self> {
        let layout = GraphLayout::new(1, 1, token_ring_capacity as u32, 1);
        let mut graph_pool = GraphPool::new();
        graph_pool.capture_synthetic(layout);
        Ok(Self {
            graph_pool,
            token_ring: DeviceTokenRing::new(token_ring_capacity)?,
            next_transaction_id: 1,
            layout,
            device,
        })
    }

    pub fn token_ring(&self) -> &DeviceTokenRing {
        &self.token_ring
    }

    pub fn graph_pool(&self) -> &GraphPool {
        &self.graph_pool
    }

    pub fn launch(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        input_token: TokenId,
    ) -> Result<PendingSyntheticStep<'_>> {
        let input_source = if token_index == 0 {
            TokenInputSource::Seed
        } else {
            let device_input = self.token_ring.consume_device_input_ref(
                request_id,
                sequence_id,
                token_index - 1,
            )?;
            if device_input.token != input_token {
                return Err(NervaError::ResidencyViolation {
                    block_id: nerva_core::ResidentBlockId(0),
                    reason: "next input token does not match prior sampled device token"
                        .to_string(),
                });
            }
            TokenInputSource::DeviceRing(device_input.token_ref)
        };

        self.launch_with_source(
            request_id,
            sequence_id,
            token_index,
            input_token,
            input_source,
        )
    }

    pub fn launch_device_next(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        seed_token: TokenId,
    ) -> Result<PendingSyntheticStep<'_>> {
        let (input_token, input_source) = if token_index == 0 {
            (seed_token, TokenInputSource::Seed)
        } else {
            let input = self.token_ring.consume_device_input_ref(
                request_id,
                sequence_id,
                token_index - 1,
            )?;
            (input.token, TokenInputSource::DeviceRing(input.token_ref))
        };

        self.launch_with_source(
            request_id,
            sequence_id,
            token_index,
            input_token,
            input_source,
        )
    }

    fn launch_with_source(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        token_index: u64,
        input_token: TokenId,
        input_source: TokenInputSource,
    ) -> Result<PendingSyntheticStep<'_>> {
        self.graph_pool.check_before_replay(self.layout)?;
        let transaction_id = TransactionId(self.next_transaction_id);
        self.next_transaction_id = self.next_transaction_id.saturating_add(1);
        let layout = self.layout;
        Ok(PendingSyntheticStep {
            engine: self,
            plan: Some(SyntheticStepPlan {
                transaction_id,
                request_id,
                sequence_id,
                token_index,
                input_token,
                input_source,
                layout,
            }),
        })
    }
}

#[must_use = "PendingSyntheticStep must be collect()-ed; dropping it loses the launched transaction"]
#[derive(Debug)]
pub struct PendingSyntheticStep<'engine> {
    engine: &'engine mut SyntheticEngine,
    plan: Option<SyntheticStepPlan>,
}

impl<'engine> PendingSyntheticStep<'engine> {
    pub fn plan(&self) -> Option<&SyntheticStepPlan> {
        self.plan.as_ref()
    }

    pub fn collect(mut self) -> Result<StepOutput> {
        let plan = self
            .plan
            .take()
            .expect("PendingSyntheticStep::collect called twice");
        self.engine.graph_pool.replay(plan.layout)?;

        let token = TokenId(plan.input_token.0.wrapping_add(1));
        let device_token_ref = self.engine.token_ring.publish(
            plan.request_id,
            plan.sequence_id,
            plan.token_index,
            token,
        )?;

        let host_visible_token = self.engine.token_ring.host_observe(
            plan.request_id,
            plan.sequence_id,
            plan.token_index,
        )?;
        let mut ledger = TokenLedger::new(plan.token_index);
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::GraphReplay,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 1,
            label: "synthetic_graph_replay",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::DeviceActivity,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 3,
            label: "synthetic_decode_kernel",
        });
        ledger.record_device_span(DeviceTimelineSpan::new(
            self.engine.device,
            0,
            3,
            MetricSource::EstimatedModel,
            "synthetic_decode_device_active",
        ))?;
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: core::mem::size_of::<TokenId>(),
            latency_ns: 1,
            label: "async_host_token_observation",
        });
        ledger.record_sync(
            SyncClass::SoftVisibilitySync,
            None,
            Some(MemoryTier::Vram),
            Some(MemoryTier::PinnedDram),
            0,
            1,
            MetricSource::EstimatedModel,
            "soft_visibility_host_wait",
        );

        Ok(StepOutput {
            request_id: plan.request_id,
            sequence_id: plan.sequence_id,
            token_index: plan.token_index,
            input_source: plan.input_source,
            device_token_ref,
            token: host_visible_token,
            finished: false,
            ledger,
        })
    }
}

impl Drop for PendingSyntheticStep<'_> {
    fn drop(&mut self) {
        debug_assert!(
            self.plan.is_none(),
            "PendingSyntheticStep dropped without collect(); transaction output leaked"
        );
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

impl SyntheticDecodeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            SyntheticDecodeStatus::Ok => "ok",
            SyntheticDecodeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"token_ring_capacity\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"graph_replay_events\":{},\"kernel_events\":{},\"device_events\":{},\"copy_events\":{},\"host_wait_events\":{},\"soft_visibility_syncs\":{},\"device_timeline_active_ns\":{},\"device_timeline_idle_ns\":{},\"graph_replay_latency_ns\":{},\"device_latency_ns\":{},\"copy_latency_ns\":{},\"host_wait_latency_ns\":{},\"soft_visibility_sync_latency_ns\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"observed_tokens\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"error\":{}}}",
            status,
            self.steps,
            self.token_ring_capacity,
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

    pub fn synthetic_engine(&self, token_ring_capacity: usize) -> Result<SyntheticEngine> {
        SyntheticEngine::new(token_ring_capacity, self.config.device)
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
            hip: CapabilityState::Unsupported,
            hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
            nvidia_driver_version: read_trimmed_first_line("/proc/driver/nvidia/version"),
            pinned_host_staging: CapabilityState::SupportedUnverified,
            gpu_direct_rdma: CapabilityState::DegradedToPinnedHost,
            amd_peerdirect: CapabilityState::Unsupported,
            dma_buf_export: CapabilityState::Unsupported,
            cxl: CapabilityState::Unsupported,
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
        let mut stale_tokens: u64 = 0;
        let mut extra_tokens: u64 = 0;
        let mut mismatched_tokens: u64 = 0;
        let mut host_causality_edges: u64 = 0;

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

fn mix_u32(out: &mut [u8; 32], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    for (idx, byte) in bytes.iter().enumerate() {
        out[offset + idx] ^= *byte;
        out[31 - offset - idx] = out[31 - offset - idx].wrapping_add(byte.rotate_left(1));
    }
}

fn json_opt_token(value: Option<TokenId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
}

fn json_opt_block_id(value: Option<ResidentBlockId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
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

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{}\"", value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHARD_ONE: &str = "model-00001-of-00001.safetensors";

    fn tiny_llama_manifest() -> nerva_model::HfTensorManifest {
        let metadata = nerva_model::parse_hf_config_metadata(
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
        let layout = nerva_model::plan_hf_weight_layout(&metadata).unwrap();
        nerva_model::build_hf_tensor_manifest(&layout).unwrap()
    }

    fn single_shard_index_json(manifest: &nerva_model::HfTensorManifest) -> String {
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

    fn tiny_shard_plan() -> (nerva_model::SafetensorsShardPlan, usize) {
        let manifest = tiny_llama_manifest();
        let index = single_shard_index_json(&manifest);
        let header = nerva_model::synthetic_safetensors_header_for_manifest(&manifest).unwrap();
        let header_len = header.len();
        let plan = nerva_model::plan_safetensors_shards_for_manifest(
            &index,
            &[nerva_model::SafetensorsShardHeader::new(SHARD_ONE, &header)],
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
        assert_eq!(
            snapshot.gpu_direct_rdma,
            CapabilityState::DegradedToPinnedHost
        );
        assert_eq!(snapshot.amd_peerdirect, CapabilityState::Unsupported);
        assert_eq!(snapshot.dma_buf_export, CapabilityState::Unsupported);
        assert_eq!(snapshot.cxl, CapabilityState::Unsupported);

        let json = snapshot.to_json();
        assert!(json.contains("\"target_os\":\"linux\""));
        assert!(json.contains("\"kernel_release\""));
        assert!(json.contains("\"fabric\":\"DiscreteExplicit\""));
        assert!(json.contains("\"gpu_direct_rdma\":\"DEGRADED_TO_PINNED_HOST\""));
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
            hip: CapabilityState::Unsupported,
            hip_visible_devices: Some("2".to_string()),
            nvidia_driver_version: Some("driver\\version".to_string()),
            pinned_host_staging: CapabilityState::SupportedUnverified,
            gpu_direct_rdma: CapabilityState::DegradedToPinnedHost,
            amd_peerdirect: CapabilityState::Unsupported,
            dma_buf_export: CapabilityState::Unsupported,
            cxl: CapabilityState::Unsupported,
        };

        let json = snapshot.to_json();
        assert!(json.contains("quote\\\" slash\\\\ newline\\n"));
        assert!(json.contains("kernel\\\" release"));
        assert!(json.contains("driver\\\\version"));
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
    fn materializes_hf_weight_manifest_as_dram_resident_blocks() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        let manifest = nerva_model::hf_tensor_manifest_probe().unwrap().manifest;
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
        assert_eq!(block.semantics, nerva_core::MutationSemantics::Immutable);
        assert_eq!(block.tier, MemoryTier::Dram);
        assert_eq!(block.dtype, first.dtype);
        assert_eq!(block.layout, LayoutId(1));
    }

    #[test]
    fn materialized_weight_manifest_preserves_last_block_and_decision() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        let manifest = nerva_model::hf_tensor_manifest_probe().unwrap().manifest;
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
        let manifest = nerva_model::hf_tensor_manifest_probe().unwrap().manifest;
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
        let metadata = nerva_model::parse_hf_config_metadata(
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
        let layout = nerva_model::plan_hf_weight_layout(&metadata).unwrap();
        let manifest = nerva_model::build_hf_tensor_manifest(&layout).unwrap();
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
        let metadata = nerva_model::parse_hf_config_metadata(
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
        let layout = nerva_model::plan_hf_weight_layout(&metadata).unwrap();
        let manifest = nerva_model::build_hf_tensor_manifest(&layout).unwrap();
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
        assert_eq!(summary.stale_tokens, 0);
        assert_eq!(summary.missing_tokens, 0);
        assert_eq!(summary.extra_tokens, 0);
        assert_eq!(summary.mismatched_tokens, 0);
        assert_eq!(summary.host_causality_edges, 0);
        assert!(summary.to_json().contains("\"steps\":1024"));
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
}
