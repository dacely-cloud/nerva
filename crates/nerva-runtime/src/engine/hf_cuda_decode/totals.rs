use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::footprint::CudaHfDecodeSequenceFootprint;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::critical::TokenCriticalPathReport;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::hf_cuda_decode::hash::hash_tokens;
use crate::engine::hf_cuda_decode::summary::{
    HfCudaResidentWeightSummary, HfCudaSeedDecodeSummary,
};

#[derive(Default)]
pub(super) struct CudaDecodeCounters {
    resident_weight_bytes: u64,
    resident_kv_bytes: u64,
    kv_tokens: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    graph_replays: u64,
    graph_nodes: u64,
    graph_launches: u64,
    kernel_launches: u64,
    sync_calls: u64,
    host_causality_edges: u64,
    hot_path_allocations: u64,
    cuda_footprint: CudaHfDecodeSequenceFootprint,
    cuda_device_total_memory_bytes: Option<usize>,
    cuda_device_free_memory_bytes: Option<usize>,
    cuda_fits_device_free_memory: Option<bool>,
}

impl CudaDecodeCounters {
    pub(super) fn record_sequence(&mut self, cuda: &CudaHfDecodeSequenceSummary) {
        self.resident_weight_bytes += cuda.resident_weight_bytes;
        self.resident_kv_bytes += cuda.resident_kv_bytes;
        self.kv_tokens = self.kv_tokens.max(cuda.kv_tokens);
        self.h2d_bytes += cuda.h2d_bytes;
        self.d2h_bytes += cuda.d2h_bytes;
        self.graph_replays += cuda.graph_replays;
        self.graph_nodes = self.graph_nodes.max(cuda.graph_nodes);
        self.graph_launches += cuda.graph_launches;
        self.kernel_launches += cuda.kernel_launches;
        self.sync_calls += cuda.sync_calls;
        self.host_causality_edges += cuda.host_causality_edges;
        self.hot_path_allocations += cuda.hot_path_allocations;
        self.cuda_footprint = cuda.planned_footprint;
        self.cuda_device_total_memory_bytes = cuda.device_total_memory_bytes;
        self.cuda_device_free_memory_bytes = cuda.device_free_memory_bytes;
        self.cuda_fits_device_free_memory = cuda.fits_device_free_memory;
    }
}

pub(super) struct DecodeParts {
    steps_requested: usize,
    tokens: Vec<TokenId>,
    expected_tokens: Vec<TokenId>,
    ledgers: Vec<TokenLedger>,
    resident_weights: HfCudaResidentWeightSummary,
}

impl DecodeParts {
    pub(super) fn new(
        steps_requested: usize,
        tokens: Vec<TokenId>,
        expected_tokens: Vec<TokenId>,
        ledgers: Vec<TokenLedger>,
    ) -> Self {
        Self {
            steps_requested,
            tokens,
            expected_tokens,
            ledgers,
            resident_weights: HfCudaResidentWeightSummary::default(),
        }
    }
}

pub(super) fn build_summary(
    status: SmokeStatus,
    parts: DecodeParts,
    cpu_ledgers: &[TokenLedger],
    counters: CudaDecodeCounters,
    error: Option<String>,
) -> HfCudaSeedDecodeSummary {
    let output_hash = hash_tokens(&parts.tokens);
    let expected_hash = hash_tokens(&parts.expected_tokens);
    let critical_paths = critical_paths(&parts.ledgers);
    HfCudaSeedDecodeSummary {
        status,
        steps_requested: parts.steps_requested,
        parity: parts.tokens == parts.expected_tokens,
        ledger_count: parts.ledgers.len() as u64,
        device_events: event_count(&parts.ledgers, LedgerEventKind::DeviceActivity),
        copy_events: event_count(&parts.ledgers, LedgerEventKind::Copy),
        hard_syncs: sync_count(&parts.ledgers, SyncClass::HardSync),
        soft_visibility_syncs: sync_count(&parts.ledgers, SyncClass::SoftVisibilitySync),
        execution_decisions: execution_decisions(&parts.ledgers),
        resident_weight_bytes: counters.resident_weight_bytes,
        cuda_footprint: counters.cuda_footprint,
        cuda_device_total_memory_bytes: counters.cuda_device_total_memory_bytes,
        cuda_device_free_memory_bytes: counters.cuda_device_free_memory_bytes,
        cuda_fits_device_free_memory: counters.cuda_fits_device_free_memory,
        resident_kv_bytes: counters.resident_kv_bytes,
        kv_tokens: counters.kv_tokens,
        h2d_bytes: counters.h2d_bytes,
        d2h_bytes: counters.d2h_bytes,
        graph_replays: counters.graph_replays,
        graph_nodes: counters.graph_nodes,
        graph_launches: counters.graph_launches,
        graph_replay_events: event_count(&parts.ledgers, LedgerEventKind::GraphReplay),
        kernel_launches: counters.kernel_launches,
        sync_calls: counters.sync_calls,
        host_causality_edges: counters.host_causality_edges,
        hot_path_allocations: counters.hot_path_allocations
            + hot_path_allocations(&parts.ledgers)
            + hot_path_allocations(cpu_ledgers),
        output_hash,
        expected_hash,
        resident_weights: parts.resident_weights,
        critical_paths,
        token_ledgers: parts.ledgers,
        tokens: parts.tokens,
        expected_tokens: parts.expected_tokens,
        error,
    }
}

fn critical_paths(ledgers: &[TokenLedger]) -> Vec<TokenCriticalPathReport> {
    ledgers
        .iter()
        .map(|ledger| {
            TokenCriticalPathReport::from_ledger(ledger, DeviceOrdinal(0))
                .expect("HF CUDA token ledgers have valid device timelines")
        })
        .collect()
}

fn event_count(ledgers: &[TokenLedger], kind: LedgerEventKind) -> u64 {
    ledgers.iter().map(|ledger| ledger.event_count(kind)).sum()
}

fn sync_count(ledgers: &[TokenLedger], class: SyncClass) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.sync_count_for(class))
        .sum()
}

fn execution_decisions(ledgers: &[TokenLedger]) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum()
}

fn hot_path_allocations(ledgers: &[TokenLedger]) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum()
}
