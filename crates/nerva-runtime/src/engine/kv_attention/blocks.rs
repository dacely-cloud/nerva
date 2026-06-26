use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::decision::BlockVersionDependency;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::kv::pool::table::KvPagePool;
use nerva_memory::registry::table::registry::BlockRegistry;
use nerva_model::attention::block::KvAttentionBlock;

use crate::engine::kv_attention::payload::ResidentKvAttentionPayload;

pub(super) fn resident_attention_blocks<'a>(
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
