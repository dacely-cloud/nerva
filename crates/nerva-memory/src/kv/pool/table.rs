use std::collections::{BTreeMap, VecDeque};

use nerva_core::types::error::{NervaError, Result};

use crate::kv::page::{KvPageDescriptor, KvPageHandle, KvPrefixKey};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPagePool {
    pub(crate) pages: Vec<KvPageDescriptor>,
    pub(crate) free_pages: VecDeque<u32>,
    pub(crate) prefix_cache: BTreeMap<KvPrefixKey, u32>,
}

impl KvPagePool {
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

    pub(crate) fn page_mut(&mut self, page_index: u32) -> Result<&mut KvPageDescriptor> {
        self.pages
            .get_mut(page_index as usize)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown KV page index {page_index}"),
            })
    }
}
