use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::compute_near_data::allocation::allocate_weight_shard;
use crate::engine::compute_near_data::config::ComputeNearDataProbeConfig;
use crate::engine::compute_near_data::execute::execute_resident_split_matvec;
use crate::engine::compute_near_data::fixture::ComputeNearDataFixture;
use crate::engine::compute_near_data::math::{hash_f32s, mat_vec_row_major, max_abs_error};
use crate::engine::compute_near_data::shard::ResidentMatvecShard;
use crate::engine::compute_near_data::summary::{
    ComputeNearDataProbeStatus, ComputeNearDataProbeSummary,
};
use crate::engine::compute_near_data::validation::validate_config;
use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_compute_near_data_probe(
        &self,
        config: ComputeNearDataProbeConfig,
    ) -> Result<ComputeNearDataProbeSummary> {
        validate_config(config)?;

        let fixture = ComputeNearDataFixture::new();
        let cpu_weights = fixture.cpu_weights(config);
        let gpu_weights = fixture.gpu_weights(config);
        let cpu_bytes = cpu_weights.len() * core::mem::size_of::<f32>();
        let gpu_bytes = gpu_weights.len() * core::mem::size_of::<f32>();

        let mut registry = self.block_registry(ResidencyBudget::new(gpu_bytes, 0, cpu_bytes));
        let cpu_block = allocate_weight_shard(
            &mut registry,
            MemoryTier::Dram,
            cpu_bytes,
            config.split_row,
            config.cols,
        )?;
        let gpu_block = allocate_weight_shard(
            &mut registry,
            MemoryTier::Vram,
            gpu_bytes,
            config.rows - config.split_row,
            config.cols,
        )?;
        let shards = [
            ResidentMatvecShard {
                block_id: cpu_block,
                tier: MemoryTier::Dram,
                row_start: 0,
                row_end: config.split_row,
                weights: cpu_weights,
            },
            ResidentMatvecShard {
                block_id: gpu_block,
                tier: MemoryTier::Vram,
                row_start: config.split_row,
                row_end: config.rows,
                weights: gpu_weights,
            },
        ];

        let mut ledger = TokenLedger::new(0);
        let mut output = vec![0.0; config.rows];
        execute_resident_split_matvec(
            &registry,
            self.config.device,
            config.cols,
            &fixture.input,
            &shards,
            &mut output,
            &mut ledger,
        )?;
        ledger.require_zero_hot_path_allocations()?;

        let mut reference = vec![0.0; config.rows];
        mat_vec_row_major(&fixture.matrix, &fixture.input, &mut reference);
        let max_abs_error = max_abs_error(&output, &reference);
        let output_hash = hash_f32s(&output);
        let reference_hash = hash_f32s(&reference);

        Ok(ComputeNearDataProbeSummary {
            status: ComputeNearDataProbeStatus::Ok,
            rows: config.rows,
            cols: config.cols,
            split_row: config.split_row,
            blocks: shards.len(),
            dram_blocks: shards
                .iter()
                .filter(|shard| shard.tier == MemoryTier::Dram)
                .count() as u64,
            vram_blocks: shards
                .iter()
                .filter(|shard| shard.tier == MemoryTier::Vram)
                .count() as u64,
            output,
            reference,
            output_hash,
            reference_hash,
            max_abs_error,
            parity: max_abs_error <= 0.000001,
            execution_decisions: ledger.execution_decisions.len() as u64,
            block_version_dependencies: ledger.block_version_dependencies.len() as u64,
            cpu_events: ledger.event_count(LedgerEventKind::CpuActivity),
            device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            merge_bytes: (config.rows - config.split_row) * core::mem::size_of::<f32>(),
            hot_path_allocations: ledger.hot_path_allocations,
        })
    }
}
