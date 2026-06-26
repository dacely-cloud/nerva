#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::collections::BTreeMap;

use nerva_core::{
    DeviceOrdinal, MemoryTier, NervaError, RequestId, Result, SequenceId, TokenId, TransactionId,
    ensure_supported_linux_host,
};
use nerva_ledger::{LedgerEvent, LedgerEventKind, TokenLedger};
use nerva_memory::{BlockRegistry, StaticArenaSet};

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
        let slot = &self.slots[self.slot_index(previous_token_index)];
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
        slot.token.ok_or_else(|| NervaError::ResidencyViolation {
            block_id: nerva_core::ResidentBlockId(0),
            reason: "device token ring slot has no token".to_string(),
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
    pub layout: GraphLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
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
        if token_index > 0 {
            let device_token =
                self.token_ring
                    .consume_device_input(request_id, sequence_id, token_index - 1)?;
            if device_token != input_token {
                return Err(NervaError::ResidencyViolation {
                    block_id: nerva_core::ResidentBlockId(0),
                    reason: "next input token does not match prior sampled device token"
                        .to_string(),
                });
            }
        }

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
        self.engine.token_ring.publish(
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
            kind: LedgerEventKind::KernelLaunch,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 1,
            label: "synthetic_graph_replay",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::KernelLaunch,
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

        Ok(StepOutput {
            request_id: plan.request_id,
            sequence_id: plan.sequence_id,
            token_index: plan.token_index,
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
    pub kernel_events: u64,
    pub copy_events: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyntheticDecodeStatus {
    Ok,
    Failed,
}

impl SyntheticDecodeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            SyntheticDecodeStatus::Ok => "ok",
            SyntheticDecodeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"token_ring_capacity\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"kernel_events\":{},\"copy_events\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.steps,
            self.token_ring_capacity,
            self.seed_token.0,
            json_opt_token(self.last_token),
            self.graph_replays,
            self.kernel_events,
            self.copy_events,
            self.total_latency_ns,
            self.hot_path_allocations,
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
        let mut input_token = config.seed_token;
        let mut last_token = None;
        let mut kernel_events: u64 = 0;
        let mut copy_events: u64 = 0;
        let mut total_latency_ns: u64 = 0;
        let mut hot_path_allocations: u64 = 0;

        for token_index in 0..config.steps {
            let output = engine
                .launch(RequestId(1), SequenceId(1), token_index, input_token)?
                .collect()?;
            output.ledger.require_zero_hot_path_allocations()?;
            kernel_events += output
                .ledger
                .events
                .iter()
                .filter(|event| event.kind == LedgerEventKind::KernelLaunch)
                .count() as u64;
            copy_events += output
                .ledger
                .events
                .iter()
                .filter(|event| event.kind == LedgerEventKind::Copy)
                .count() as u64;
            total_latency_ns = total_latency_ns.saturating_add(output.ledger.total_latency_ns());
            hot_path_allocations =
                hot_path_allocations.saturating_add(output.ledger.hot_path_allocations);
            input_token = output.token;
            last_token = Some(output.token);
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
            kernel_events,
            copy_events,
            total_latency_ns,
            hot_path_allocations,
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
        assert_eq!(output.ledger.hot_path_allocations, 0);
        assert_eq!(output.ledger.events.len(), 3);
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
            .launch(RequestId(2), SequenceId(9), 1, TokenId(11))
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(output.token, TokenId(12));
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
        assert_eq!(summary.kernel_events, 2048);
        assert_eq!(summary.copy_events, 1024);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.to_json().contains("\"steps\":1024"));
    }

    #[test]
    fn synthetic_decode_rejects_zero_steps() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        let err = runtime
            .run_synthetic_decode(SyntheticDecodeConfig::new(0, 64, TokenId(1)))
            .unwrap_err();
        assert!(matches!(err, NervaError::InvalidArgument { .. }));
    }
}
