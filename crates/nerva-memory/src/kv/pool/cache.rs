use nerva_core::types::error::{NervaError, Result};

use crate::kv::page::{KvPageHandle, KvPrefixKey};
use crate::kv::pool::table::KvPagePool;

impl KvPagePool {
    pub fn retain_cached(&mut self, key: KvPrefixKey, step: u64) -> Result<Option<KvPageHandle>> {
        let Some(page_index) = self.prefix_cache.get(&key).copied() else {
            return Ok(None);
        };
        self.retain_page(page_index, step).map(Some)
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
}
