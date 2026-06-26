use nerva_core::types::{DType, ExecutionOwner, MemoryTier, ResidentBlockId};
use nerva_ledger::types::TokenLedger;
use nerva_memory::registry::BlockRegistry;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightBlockRef {
    pub name: String,
    pub block_id: ResidentBlockId,
    pub bytes: usize,
    pub dtype: DType,
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
pub struct ResidentWeightPrefetchIoSummary {
    pub tasks: usize,
    pub completed_blocks: usize,
    pub total_bytes: usize,
    pub shard_count: usize,
    pub disk_read_events: u64,
    pub copy_events: u64,
    pub ready_blocks: usize,
    pub data_hash: u64,
    pub hot_path_allocations: u64,
    pub ledger: TokenLedger,
}

impl ResidentWeightPrefetchIoSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"tasks\":{},\"completed_blocks\":{},\"total_bytes\":{},\"shard_count\":{},\"disk_read_events\":{},\"copy_events\":{},\"ready_blocks\":{},\"data_hash\":{},\"hot_path_allocations\":{}}}",
            self.tasks,
            self.completed_blocks,
            self.total_bytes,
            self.shard_count,
            self.disk_read_events,
            self.copy_events,
            self.ready_blocks,
            self.data_hash,
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

fn json_opt_block_id(value: Option<ResidentBlockId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
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
