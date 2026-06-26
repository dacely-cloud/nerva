use std::collections::{BTreeMap, VecDeque};

use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::error::{NervaError, Result};

use crate::arena::kind::AllocationPhase;
use crate::arena::set::StaticArenaSet;
use crate::kv::page::{KvPageDescriptor, KvPageHandle, KvPageSpec, KvPrefixKey};
use crate::registry::{BlockAllocationRequest, BlockRegistry};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPagePool {
    pages: Vec<KvPageDescriptor>,
    free_pages: VecDeque<u32>,
    prefix_cache: BTreeMap<KvPrefixKey, u32>,
}

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

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    pub fn num_free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn usage(&self) -> f32 {
        if self.pages.is_empty() {
            0.0
        } else {
            1.0 - (self.num_free_pages() as f32 / self.pages.len() as f32)
        }
    }

    pub fn page(&self, page_index: u32) -> Option<&KvPageDescriptor> {
        self.pages.get(page_index as usize)
    }

    pub fn pages(&self) -> &[KvPageDescriptor] {
        &self.pages
    }

    pub fn lookup_cached(&self, key: KvPrefixKey) -> Option<KvPageHandle> {
        let page_index = *self.prefix_cache.get(&key)?;
        let page = self.page(page_index)?;
        Some(KvPageHandle {
            page_index,
            block_id: page.block_id,
        })
    }

    pub fn allocate_page(
        &mut self,
        token_start: u32,
        token_count: u32,
        step: u64,
    ) -> Result<KvPageHandle> {
        let page_index =
            self.free_pages
                .pop_front()
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: 0,
                    reason: "KV page pool exhausted".to_string(),
                })?;
        let page = self.page_mut(page_index)?;
        if token_count > page.block_size_tokens {
            self.free_pages.push_front(page_index);
            return Err(NervaError::InvalidArgument {
                reason: "KV page token count exceeds page block size".to_string(),
            });
        }
        page.token_start = token_start;
        page.token_count = token_count;
        page.ref_count = 1;
        page.last_use = step;
        page.next_use = None;
        page.is_free = false;
        Ok(KvPageHandle {
            page_index,
            block_id: page.block_id,
        })
    }

    pub fn retain_page(&mut self, page_index: u32, step: u64) -> Result<KvPageHandle> {
        let was_free = self
            .page(page_index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown KV page index {page_index}"),
            })?
            .is_free;
        if was_free {
            self.free_pages.retain(|free| *free != page_index);
        }
        let page = self.page_mut(page_index)?;
        page.is_free = false;
        page.ref_count =
            page.ref_count
                .checked_add(1)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: "KV page reference count overflow".to_string(),
                })?;
        page.last_use = step;
        Ok(KvPageHandle {
            page_index,
            block_id: page.block_id,
        })
    }

    pub fn retain_cached(&mut self, key: KvPrefixKey, step: u64) -> Result<Option<KvPageHandle>> {
        let Some(page_index) = self.prefix_cache.get(&key).copied() else {
            return Ok(None);
        };
        self.retain_page(page_index, step).map(Some)
    }

    pub fn release_page(&mut self, page_index: u32, step: u64) -> Result<()> {
        let page = self.page_mut(page_index)?;
        if page.ref_count == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "KV page released with zero references".to_string(),
            });
        }
        page.ref_count -= 1;
        page.last_use = step;
        if page.ref_count == 0 && !page.is_free {
            page.is_free = true;
            if page.prefix_key.is_none() {
                page.token_count = 0;
            }
            self.free_pages.push_back(page_index);
        }
        Ok(())
    }

    pub fn set_next_use(&mut self, page_index: u32, next_use: Option<u64>) -> Result<()> {
        let page = self.page_mut(page_index)?;
        page.next_use = next_use;
        Ok(())
    }

    pub fn cache_page(
        &mut self,
        page_index: u32,
        key: KvPrefixKey,
        prefix_tokens: u32,
    ) -> Result<()> {
        let old_key = {
            let page = self.page_mut(page_index)?;
            if prefix_tokens == 0 {
                return Err(NervaError::InvalidArgument {
                    reason: "cached KV prefix must cover at least one token".to_string(),
                });
            }
            let old_key = page.prefix_key;
            page.prefix_key = Some(key);
            page.prefix_tokens = Some(prefix_tokens);
            old_key
        };
        if let Some(old_key) = old_key {
            self.prefix_cache.remove(&old_key);
        }
        self.prefix_cache.insert(key, page_index);
        Ok(())
    }

    pub fn evict_cached_page(&mut self, page_index: u32) -> Result<Option<KvPrefixKey>> {
        let old_key = {
            let page = self.page_mut(page_index)?;
            let old_key = page.prefix_key.take();
            page.prefix_tokens = None;
            old_key
        };
        if let Some(old_key) = old_key {
            self.prefix_cache.remove(&old_key);
        }
        Ok(old_key)
    }

    fn page_mut(&mut self, page_index: u32) -> Result<&mut KvPageDescriptor> {
        self.pages
            .get_mut(page_index as usize)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown KV page index {page_index}"),
            })
    }
}
