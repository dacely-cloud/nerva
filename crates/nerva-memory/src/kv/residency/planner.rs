use nerva_core::types::error::{NervaError, Result};

use crate::kv::pool::table::KvPagePool;
use crate::kv::residency::entries::plan_page;
use crate::kv::residency::selection::select_hot_pages;
use crate::kv::residency::types::{KvResidencyPlan, KvResidencyPlanner, KvResidencyPolicy};
use crate::registry::table::registry::BlockRegistry;

impl KvResidencyPlanner {
    pub fn plan(
        pool: &KvPagePool,
        registry: &BlockRegistry,
        current_step: u64,
        policy: KvResidencyPolicy,
    ) -> Result<KvResidencyPlan> {
        let hot_pages = select_hot_pages(pool, current_step, policy);
        let mut entries = Vec::new();
        for page in pool.pages() {
            if page.is_free && page.prefix_key.is_none() {
                continue;
            }
            let old_tier = registry
                .block(page.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("KV page {} references missing block", page.page_index),
                })?
                .tier;
            entries.push(plan_page(page, old_tier, &hot_pages, current_step, policy));
        }
        Ok(KvResidencyPlan { entries })
    }
}
