use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::AllocationId;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::decision::BlockVersionDependency;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::StaticArenaSet;
use nerva_memory::kv::page::KvPageSpec;
use nerva_memory::kv::pool::table::KvPagePool;
use nerva_memory::registry::table::BlockRegistry;
use nerva_model::attention::block::KvAttentionBlock;
use nerva_model::attention::exact::run::exact_blockwise_attention_into;
use nerva_model::attention::scratch::BlockwiseAttentionScratch;
use nerva_model::common::shape::TransformerBlockShape;

use crate::engine::kv_attention::config::TieredKvAttentionProbeConfig;
use crate::engine::kv_attention::summary::{
    TieredKvAttentionProbeStatus, TieredKvAttentionProbeSummary,
};
use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

struct ResidentKvAttentionPayload<'a> {
    page_index: u32,
    keys: &'a [f32],
    values: &'a [f32],
}

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

fn validate_config(config: TieredKvAttentionProbeConfig) -> Result<()> {
    if config.tokens_per_page != 2 {
        return Err(NervaError::InvalidArgument {
            reason: "tiered KV attention probe currently requires two tokens per page".to_string(),
        });
    }
    let required_page_bytes = config.tokens_per_page as usize * 2 * core::mem::size_of::<f32>() * 2;
    if config.page_bytes < required_page_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "tiered KV attention probe page bytes cannot hold keys and values".to_string(),
        });
    }
    Ok(())
}

fn resident_attention_blocks<'a>(
    pool: &KvPagePool,
    registry: &BlockRegistry,
    payloads: &'a [ResidentKvAttentionPayload<'a>],
    ledger: &mut TokenLedger,
) -> Result<Vec<KvAttentionBlock<'a>>> {
    let mut blocks = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let page = pool
            .page(payload.page_index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!(
                    "tiered KV attention references missing page {}",
                    payload.page_index
                ),
            })?;
        if page.token_count == 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!("tiered KV attention page {} is empty", page.page_index),
            });
        }
        let block = registry
            .block(page.block_id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!(
                    "tiered KV attention page {} references missing block",
                    page.page_index
                ),
            })?;
        if block.state != ResidencyState::Ready {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "tiered KV attention page {} block is not Ready",
                    page.page_index
                ),
            });
        }
        ledger.record_block_version_dependency(BlockVersionDependency {
            block_id: page.block_id,
            required_version: block.version,
            observed_version: block.version,
            label: "tiered_kv_attention",
        });
        blocks.push(KvAttentionBlock::new(
            payload.keys,
            payload.values,
            page.token_count as usize,
            block.tier,
        ));
    }
    ledger.require_satisfied_block_versions()?;
    Ok(blocks)
}

fn reference_attention(
    shape: TransformerBlockShape,
    query: &[f32],
    dram_keys: &[f32],
    dram_values: &[f32],
    vram_keys: &[f32],
    vram_values: &[f32],
) -> Result<([f32; 2], u64)> {
    let mut reference_keys = Vec::with_capacity(dram_keys.len() + vram_keys.len());
    reference_keys.extend_from_slice(dram_keys);
    reference_keys.extend_from_slice(vram_keys);
    let mut reference_values = Vec::with_capacity(dram_values.len() + vram_values.len());
    reference_values.extend_from_slice(dram_values);
    reference_values.extend_from_slice(vram_values);

    let reference_block =
        KvAttentionBlock::new(&reference_keys, &reference_values, 4, MemoryTier::Dram);
    let mut scratch = BlockwiseAttentionScratch::new(shape)?;
    let mut reference = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    exact_blockwise_attention_into(
        shape,
        query,
        &[reference_block],
        &mut scratch,
        &mut reference,
        &mut ledger,
    )?;
    let hash = hash_f32s(&reference);
    Ok((reference, hash))
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn hash_f32s(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
