use crate::kv::pool::table::KvPagePool;
use crate::kv::residency::types::KvResidencyPolicy;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct KvPagePriority {
    page_index: u32,
    pinned: bool,
    distance: u64,
    last_use: u64,
}

pub(super) fn select_hot_pages(
    pool: &KvPagePool,
    current_step: u64,
    policy: KvResidencyPolicy,
) -> Vec<u32> {
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
