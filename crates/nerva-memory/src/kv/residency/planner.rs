use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;

use crate::kv::page::KvPageDescriptor;
use crate::kv::pool::table::KvPagePool;
use crate::kv::residency::types::{
    KvResidencyAction, KvResidencyPlan, KvResidencyPlanEntry, KvResidencyPlanner, KvResidencyPolicy,
};
use crate::registry::table::BlockRegistry;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct KvPagePriority {
    page_index: u32,
    pinned: bool,
    distance: u64,
    last_use: u64,
}

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

fn select_hot_pages(pool: &KvPagePool, current_step: u64, policy: KvResidencyPolicy) -> Vec<u32> {
    let mut hot_candidates = Vec::new();
    for page in pool.pages() {
        if page.is_free && page.prefix_key.is_none() {
            continue;
        }
        let pinned = page.ref_count > 0;
        let distance = page
            .next_use
            .map(|next_use| next_use.saturating_sub(current_step))
            .unwrap_or(u64::MAX);
        if pinned || distance <= policy.prefetch_distance {
            hot_candidates.push(KvPagePriority {
                page_index: page.page_index,
                pinned,
                distance,
                last_use: page.last_use,
            });
        }
    }
    hot_candidates.sort_by_key(|candidate| {
        (
            !candidate.pinned,
            candidate.distance,
            core::cmp::Reverse(candidate.last_use),
            candidate.page_index,
        )
    });
    take_hot_pages(hot_candidates, policy.hot_page_limit)
}

fn take_hot_pages(hot_candidates: Vec<KvPagePriority>, hot_page_limit: usize) -> Vec<u32> {
    let pinned_count = hot_candidates
        .iter()
        .filter(|candidate| candidate.pinned)
        .count();
    let speculative_budget = hot_page_limit.saturating_sub(pinned_count);
    let mut speculative_taken = 0usize;
    hot_candidates
        .into_iter()
        .filter_map(|candidate| {
            if candidate.pinned {
                Some(candidate.page_index)
            } else if speculative_taken < speculative_budget {
                speculative_taken += 1;
                Some(candidate.page_index)
            } else {
                None
            }
        })
        .collect()
}

fn plan_page(
    page: &KvPageDescriptor,
    old_tier: MemoryTier,
    hot_pages: &[u32],
    current_step: u64,
    policy: KvResidencyPolicy,
) -> KvResidencyPlanEntry {
    let wants_hot = hot_pages.contains(&page.page_index);
    let idle = current_step.saturating_sub(page.last_use);
    if wants_hot && old_tier == MemoryTier::Vram {
        keep_hot_entry(page, old_tier)
    } else if wants_hot {
        prefetch_entry(page, old_tier)
    } else if page.ref_count == 0 && idle >= policy.evict_after_idle {
        evict_entry(page, old_tier)
    } else if old_tier == MemoryTier::Vram {
        demote_entry(page, old_tier)
    } else {
        keep_warm_entry(page, old_tier)
    }
}

fn keep_hot_entry(page: &KvPageDescriptor, old_tier: MemoryTier) -> KvResidencyPlanEntry {
    KvResidencyPlanEntry {
        page_index: page.page_index,
        block_id: page.block_id,
        bytes: page_token_bytes(page),
        old_tier,
        new_tier: MemoryTier::Vram,
        action: KvResidencyAction::KeepHot,
        reason: "KV page is already hot and needed soon",
        predicted_visible_ns: 0,
    }
}

fn prefetch_entry(page: &KvPageDescriptor, old_tier: MemoryTier) -> KvResidencyPlanEntry {
    KvResidencyPlanEntry {
        page_index: page.page_index,
        block_id: page.block_id,
        bytes: page_token_bytes(page),
        old_tier,
        new_tier: MemoryTier::Vram,
        action: KvResidencyAction::PrefetchToHot,
        reason: "KV page next use is within prefetch window",
        predicted_visible_ns: transfer_cost_ns(page_token_bytes(page), old_tier, MemoryTier::Vram),
    }
}

fn evict_entry(page: &KvPageDescriptor, old_tier: MemoryTier) -> KvResidencyPlanEntry {
    KvResidencyPlanEntry {
        page_index: page.page_index,
        block_id: page.block_id,
        bytes: page_token_bytes(page),
        old_tier,
        new_tier: MemoryTier::Dram,
        action: KvResidencyAction::EvictCold,
        reason: "KV page is idle beyond eviction threshold",
        predicted_visible_ns: transfer_cost_ns(page_token_bytes(page), old_tier, MemoryTier::Dram),
    }
}

fn demote_entry(page: &KvPageDescriptor, old_tier: MemoryTier) -> KvResidencyPlanEntry {
    KvResidencyPlanEntry {
        page_index: page.page_index,
        block_id: page.block_id,
        bytes: page_token_bytes(page),
        old_tier,
        new_tier: MemoryTier::Dram,
        action: KvResidencyAction::DemoteToWarm,
        reason: "KV page is outside hot budget",
        predicted_visible_ns: transfer_cost_ns(page_token_bytes(page), old_tier, MemoryTier::Dram),
    }
}

fn keep_warm_entry(page: &KvPageDescriptor, old_tier: MemoryTier) -> KvResidencyPlanEntry {
    KvResidencyPlanEntry {
        page_index: page.page_index,
        block_id: page.block_id,
        bytes: page_token_bytes(page),
        old_tier,
        new_tier: old_tier,
        action: KvResidencyAction::KeepWarm,
        reason: "KV page remains warm",
        predicted_visible_ns: 0,
    }
}

fn page_token_bytes(page: &KvPageDescriptor) -> usize {
    page.page_bytes
}

fn transfer_cost_ns(bytes: usize, old_tier: MemoryTier, new_tier: MemoryTier) -> u64 {
    if old_tier == new_tier {
        0
    } else {
        100 + bytes as u64
    }
}
