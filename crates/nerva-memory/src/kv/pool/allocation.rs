use nerva_core::types::error::{NervaError, Result};

use crate::kv::page::KvPageHandle;
use crate::kv::pool::table::KvPagePool;

impl KvPagePool {
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
}
