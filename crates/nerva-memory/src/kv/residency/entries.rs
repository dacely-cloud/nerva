use nerva_core::types::memory::tier::MemoryTier;

use crate::kv::page::KvPageDescriptor;
use crate::kv::residency::cost::transfer_cost_ns;
use crate::kv::residency::types::{KvResidencyAction, KvResidencyPlanEntry, KvResidencyPolicy};

pub(super) fn plan_page(
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
