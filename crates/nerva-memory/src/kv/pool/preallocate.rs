use std::collections::{BTreeMap, VecDeque};

use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::error::{NervaError, Result};

use crate::arena::kind::AllocationPhase;
use crate::arena::set::StaticArenaSet;
use crate::kv::page::{KvPageDescriptor, KvPageSpec};
use crate::kv::pool::table::KvPagePool;
use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::BlockRegistry;

impl KvPagePool {
    pub fn preallocate(
        arenas: &mut StaticArenaSet,
        registry: &mut BlockRegistry,
        num_pages: u32,
        spec: KvPageSpec,
    ) -> Result<Self> {
        if spec.tier != spec.arena_kind.tier() {
            return Err(NervaError::InvalidArgument {
                reason: "KV page spec tier and arena kind do not match".to_string(),
            });
        }
        if spec.block_size_tokens == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "KV page block size must be non-zero".to_string(),
            });
        }

        let mut pages = Vec::with_capacity(num_pages as usize);
        let mut free_pages = VecDeque::with_capacity(num_pages as usize);
        for page_index in 0..num_pages {
            let block_id = arenas.reserve_resident_block(
                registry,
                spec.arena_kind,
                "kv-page",
                BlockAllocationRequest::new(BlockKind::KvPage, spec.tier, spec.page_bytes),
                spec.align,
                AllocationPhase::Initialization,
            )?;
            registry.mark_ready(block_id)?;
            pages.push(KvPageDescriptor {
                page_index,
                block_id,
                layer_id: spec.layer_id,
                head_group_id: spec.head_group_id,
                token_start: 0,
                token_count: 0,
                block_size_tokens: spec.block_size_tokens,
                page_bytes: spec.page_bytes,
                ref_count: 0,
                prefix_key: None,
                prefix_tokens: None,
                last_use: 0,
                next_use: None,
                is_free: true,
            });
            free_pages.push_back(page_index);
        }

        Ok(Self {
            pages,
            free_pages,
            prefix_cache: BTreeMap::new(),
        })
    }
}
