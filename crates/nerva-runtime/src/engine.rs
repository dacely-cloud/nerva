use std::collections::BTreeMap;

use nerva_core::{
    AllocationId, BlockKind, DeviceOrdinal, ExecutionOwner, LayoutId, MemoryTier, NervaError,
    RequestId, ResidencyState, Result, SequenceId, TokenId, ensure_supported_linux_host,
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

use crate::capabilities::{
    CapabilitySnapshot, TopologySnapshot, discover_capabilities, discover_topology_snapshot,
};
#[cfg(test)]
use crate::capabilities::{
    CapabilityState, count_linux_id_list, discover_iommu_mode, extract_iommu_kernel_args,
    gpu_direct_rdma_capability, json_string_array, parse_pci_class,
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
use crate::weights::{
    ResidentWeightBlockRef, ResidentWeightExecutionPlan, ResidentWeightExecutionRunSummary,
    ResidentWeightExecutionStep, ResidentWeightExecutionStrategy, ResidentWeightHotsetSummary,
    ResidentWeightPrefetchExecutionSummary, ResidentWeightPrefetchPlan, ResidentWeightPrefetchTask,
    ResidentWeightProbeStatus, ResidentWeightProbeSummary, ResidentWeightTable,
};
#[cfg(test)]
use nerva_core::{HostArch, MemoryFabricKind, host_arch};
#[cfg(test)]
use std::env;

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
        discover_capabilities()
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

pub fn cuda_synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> nerva_cuda::CudaSyntheticGraphSummary {
    nerva_cuda::synthetic_graph_smoke(steps, ring_capacity, seed_token)
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

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{}\"", value))
}

#[cfg(test)]
mod tests;
