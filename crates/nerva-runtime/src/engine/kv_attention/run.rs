use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::static_set::StaticArenaSet;
use nerva_memory::kv::page::KvPageSpec;
use nerva_memory::kv::pool::table::KvPagePool;
use nerva_model::attention::exact::run::exact_blockwise_attention_into;
use nerva_model::attention::scratch::BlockwiseAttentionScratch;
use nerva_model::common::shape::TransformerBlockShape;

use crate::engine::kv_attention::blocks::resident_attention_blocks;
use crate::engine::kv_attention::compare::{hash_f32s, max_abs_error};
use crate::engine::kv_attention::config::TieredKvAttentionProbeConfig;
use crate::engine::kv_attention::payload::ResidentKvAttentionPayload;
use crate::engine::kv_attention::reference::reference_attention;
use crate::engine::kv_attention::summary::{
    TieredKvAttentionProbeStatus, TieredKvAttentionProbeSummary,
};
use crate::engine::kv_attention::validate::validate_config;
use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_tiered_kv_attention_probe(
        &self,
        config: TieredKvAttentionProbeConfig,
    ) -> Result<TieredKvAttentionProbeSummary> {
        validate_config(config)?;

        let shape = TransformerBlockShape::new(2, 1, 2);
        let query = [1.0, 0.25];
        let dram_keys = [0.2, 0.0, 0.0, 0.4];
        let dram_values = [1.0, 0.0, 0.5, 0.5];
        let vram_keys = [0.5, 0.1, -0.2, 0.3];
        let vram_values = [0.0, 1.0, 2.0, -1.0];

        let total_page_bytes =
            config
                .page_bytes
                .checked_mul(2)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: config.page_bytes,
                    reason: "tiered KV attention probe page byte count overflow".to_string(),
                })?;
        let mut arenas = StaticArenaSet::new(0, 0, total_page_bytes);
        let mut registry =
            self.block_registry(ResidencyBudget::new(config.page_bytes, 0, total_page_bytes));
        let mut pool = KvPagePool::preallocate(
            &mut arenas,
            &mut registry,
            2,
            KvPageSpec::new(
                0,
                0,
                config.tokens_per_page,
                config.page_bytes,
                MemoryTier::Dram,
                ArenaKind::Host,
                64,
            ),
        )?;

        let dram_page = pool.allocate_page(0, config.tokens_per_page, config.current_step)?;
        let vram_page = pool.allocate_page(
            config.tokens_per_page,
            config.tokens_per_page,
            config.current_step,
        )?;
        registry.move_block(
            vram_page.block_id,
            MemoryTier::Vram,
            AllocationId(vram_page.block_id.0),
            0,
        )?;
        registry.mark_ready(vram_page.block_id)?;

        let payloads = [
            ResidentKvAttentionPayload {
                page_index: dram_page.page_index,
                keys: &dram_keys,
                values: &dram_values,
            },
            ResidentKvAttentionPayload {
                page_index: vram_page.page_index,
                keys: &vram_keys,
                values: &vram_values,
            },
        ];

        let mut ledger = TokenLedger::new(config.current_step);
        let blocks = resident_attention_blocks(&pool, &registry, &payloads, &mut ledger)?;
        let mut scratch = BlockwiseAttentionScratch::new(shape)?;
        let mut output = [0.0; 2];
        exact_blockwise_attention_into(
            shape,
            &query,
            &blocks,
            &mut scratch,
            &mut output,
            &mut ledger,
        )?;
        ledger.require_zero_hot_path_allocations()?;

        let (reference, reference_hash) = reference_attention(
            shape,
            &query,
            &dram_keys,
            &dram_values,
            &vram_keys,
            &vram_values,
        )?;
        let max_abs_error = max_abs_error(&output, &reference);
        let output_hash = hash_f32s(&output);
        let parity = max_abs_error <= 0.000001;

        Ok(TieredKvAttentionProbeSummary {
            status: TieredKvAttentionProbeStatus::Ok,
            pages: blocks.len(),
            tokens: blocks.iter().map(|block| block.token_count).sum(),
            dram_pages: blocks
                .iter()
                .filter(|block| block.tier == MemoryTier::Dram)
                .count() as u64,
            vram_pages: blocks
                .iter()
                .filter(|block| block.tier == MemoryTier::Vram)
                .count() as u64,
            output,
            reference,
            max_abs_error,
            parity,
            output_hash,
            reference_hash,
            execution_decisions: ledger.execution_decisions.len() as u64,
            block_version_dependencies: ledger.block_version_dependencies.len() as u64,
            cpu_block_events: ledger.event_count(LedgerEventKind::CpuActivity),
            device_block_events: ledger.event_count(LedgerEventKind::DeviceActivity),
            hot_path_allocations: ledger.hot_path_allocations,
        })
    }
}
