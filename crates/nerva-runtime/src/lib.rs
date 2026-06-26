#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::collections::BTreeMap;

use nerva_core::{
    AllocationId, DeviceOrdinal, MemoryTier, NervaError, RequestId, Result, SequenceId, TokenId,
    TransactionId, ensure_supported_linux_host,
};
use nerva_ledger::{LedgerEvent, LedgerEventKind, TokenLedger};
use nerva_memory::{
    ArenaKind, BlockRegistry, KvPageSpec, KvPrefixKey, KvResidencyAction, KvResidencyPlanner,
    KvResidencyPolicy, StaticArenaSet,
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
}

impl SyntheticEngine {
    pub fn new(token_ring_capacity: usize) -> Result<Self> {
        let layout = GraphLayout::new(1, 1, token_ring_capacity as u32, 1);
        let mut graph_pool = GraphPool::new();
        graph_pool.capture_synthetic(layout);
        Ok(Self {
            graph_pool,
            token_ring: DeviceTokenRing::new(token_ring_capacity)?,
            next_transaction_id: 1,
            layout,
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
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 1,
            label: "synthetic_graph_replay",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::DeviceActivity,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 3,
            label: "synthetic_decode_kernel",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: core::mem::size_of::<TokenId>(),
            latency_ns: 1,
            label: "async_host_token_observation",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Sync,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: 0,
            latency_ns: 1,
            label: "soft_visibility_host_wait",
        });

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
    pub graph_replay_latency_ns: u64,
    pub device_latency_ns: u64,
    pub copy_latency_ns: u64,
    pub host_wait_latency_ns: u64,
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
            "{{\"status\":\"{}\",\"steps\":{},\"token_ring_capacity\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"graph_replay_events\":{},\"kernel_events\":{},\"device_events\":{},\"copy_events\":{},\"host_wait_events\":{},\"graph_replay_latency_ns\":{},\"device_latency_ns\":{},\"copy_latency_ns\":{},\"host_wait_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"observed_tokens\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"error\":{}}}",
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
            self.graph_replay_latency_ns,
            self.device_latency_ns,
            self.copy_latency_ns,
            self.host_wait_latency_ns,
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
        let _ = self.config;
        SyntheticEngine::new(token_ring_capacity)
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
        let mut graph_replay_latency_ns: u64 = 0;
        let mut device_latency_ns: u64 = 0;
        let mut copy_latency_ns: u64 = 0;
        let mut host_wait_latency_ns: u64 = 0;
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
            let token_graph_events = output.ledger.event_count(LedgerEventKind::GraphReplay);
            let token_device_events = output.ledger.event_count(LedgerEventKind::DeviceActivity);
            let token_kernel_events = output.ledger.event_count(LedgerEventKind::KernelLaunch)
                + token_graph_events
                + token_device_events;
            let token_copy_events = output.ledger.event_count(LedgerEventKind::Copy);
            let token_host_wait_events = output.ledger.event_count(LedgerEventKind::Sync);

            graph_replay_events += token_graph_events;
            kernel_events += token_kernel_events;
            device_events += token_device_events;
            copy_events += token_copy_events;
            host_wait_events += token_host_wait_events;
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
            graph_replay_latency_ns,
            device_latency_ns,
            copy_latency_ns,
            host_wait_latency_ns,
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

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{}\"", value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_uses_device_zero_by_default() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        assert_eq!(runtime.config().device, DeviceOrdinal(0));
        assert_eq!(runtime.empty_token_ledger(9).token_index, 9);
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
        assert_eq!(summary.graph_replay_latency_ns, 1024);
        assert_eq!(summary.device_latency_ns, 3072);
        assert_eq!(summary.copy_latency_ns, 1024);
        assert_eq!(summary.host_wait_latency_ns, 1024);
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
