use std::collections::{BTreeMap, VecDeque};

use nerva_core::{
    AllocationId, BlockKind, ExecutionOwner, MemoryTier, NervaError, ResidencyState,
    ResidentBlockId, Result,
};
use nerva_ledger::{
    CandidateCost, LedgerEvent, LedgerEventKind, MetricSource, ResidencyDecision, TokenLedger,
};

use crate::arena::{AllocationPhase, ArenaKind, StaticArenaSet};
use crate::registry::{BlockAllocationRequest, BlockRegistry};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPageSpec {
    pub layer_id: u32,
    pub head_group_id: u32,
    pub block_size_tokens: u32,
    pub page_bytes: usize,
    pub tier: MemoryTier,
    pub arena_kind: ArenaKind,
    pub align: usize,
}

impl KvPageSpec {
    pub const fn new(
        layer_id: u32,
        head_group_id: u32,
        block_size_tokens: u32,
        page_bytes: usize,
        tier: MemoryTier,
        arena_kind: ArenaKind,
        align: usize,
    ) -> Self {
        Self {
            layer_id,
            head_group_id,
            block_size_tokens,
            page_bytes,
            tier,
            arena_kind,
            align,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct KvPrefixKey {
    pub hash: [u8; 32],
    pub group_id: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPageHandle {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPageDescriptor {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
    pub layer_id: u32,
    pub head_group_id: u32,
    pub token_start: u32,
    pub token_count: u32,
    pub block_size_tokens: u32,
    pub page_bytes: usize,
    pub ref_count: u32,
    pub prefix_key: Option<KvPrefixKey>,
    pub prefix_tokens: Option<u32>,
    pub last_use: u64,
    pub next_use: Option<u64>,
    is_free: bool,
}

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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyPolicy {
    pub hot_page_limit: usize,
    pub prefetch_distance: u64,
    pub evict_after_idle: u64,
}

impl KvResidencyPolicy {
    pub const fn new(hot_page_limit: usize, prefetch_distance: u64, evict_after_idle: u64) -> Self {
        Self {
            hot_page_limit,
            prefetch_distance,
            evict_after_idle,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KvResidencyAction {
    KeepHot,
    PrefetchToHot,
    KeepWarm,
    DemoteToWarm,
    EvictCold,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyPlanEntry {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
    pub bytes: usize,
    pub old_tier: MemoryTier,
    pub new_tier: MemoryTier,
    pub action: KvResidencyAction,
    pub reason: &'static str,
    pub predicted_visible_ns: u64,
}

impl KvResidencyPlanEntry {
    pub fn changes_tier(self) -> bool {
        self.old_tier != self.new_tier
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvResidencyPlan {
    pub entries: Vec<KvResidencyPlanEntry>,
}

impl KvResidencyPlan {
    pub fn record_decisions_to_ledger(&self, ledger: &mut TokenLedger) {
        for entry in &self.entries {
            ledger.record_residency_decision(ResidencyDecision {
                block_id: entry.block_id,
                old_tier: entry.old_tier,
                new_tier: entry.new_tier,
                executor_selected: ExecutionOwner::Gpu(nerva_core::DeviceOrdinal(0)),
                candidate_costs: vec![
                    CandidateCost::estimated("keep-current-tier", 0),
                    CandidateCost::estimated("planned-tier", entry.predicted_visible_ns),
                ],
                reason: entry.reason,
                predicted_overlap_ns: 0,
                actual_visible_ns: None,
                metric_source: MetricSource::EstimatedModel,
            });
        }
    }

    pub fn record_events_to_ledger(&self, ledger: &mut TokenLedger) {
        for entry in &self.entries {
            match entry.action {
                KvResidencyAction::PrefetchToHot => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Prefetch,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_prefetch_scheduled",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_prefetch_copy",
                    });
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::DemoteToWarm => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Eviction,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_demote_scheduled",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_demote_copy",
                    });
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::EvictCold => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Eviction,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_cold_eviction",
                    });
                    if entry.changes_tier() {
                        ledger.record(LedgerEvent {
                            kind: LedgerEventKind::Copy,
                            sync_class: None,
                            metric_source: MetricSource::EstimatedModel,
                            block_id: Some(entry.block_id),
                            from_tier: Some(entry.old_tier),
                            to_tier: Some(entry.new_tier),
                            bytes: entry.bytes,
                            latency_ns: 0,
                            label: "kv_eviction_copy",
                        });
                    }
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::KeepHot | KvResidencyAction::KeepWarm => {}
            }
        }
    }

    pub fn record_to_ledger(&self, ledger: &mut TokenLedger) {
        self.record_decisions_to_ledger(ledger);
        self.record_events_to_ledger(ledger);
    }

    pub fn apply(&self, registry: &mut BlockRegistry) -> Result<()> {
        for entry in &self.entries {
            if entry.changes_tier() {
                registry.move_block(
                    entry.block_id,
                    entry.new_tier,
                    AllocationId(entry.block_id.0),
                    0,
                )?;
            }
            match entry.action {
                KvResidencyAction::KeepHot
                | KvResidencyAction::PrefetchToHot
                | KvResidencyAction::KeepWarm => registry.mark_ready(entry.block_id)?,
                KvResidencyAction::DemoteToWarm => {
                    registry.transition(entry.block_id, ResidencyState::Draining)?
                }
                KvResidencyAction::EvictCold => {
                    registry.transition(entry.block_id, ResidencyState::Evicting)?
                }
            }
        }
        Ok(())
    }

    pub fn action_count(&self, action: KvResidencyAction) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.action == action)
            .count() as u64
    }

    pub fn changed_bytes(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.changes_tier())
            .map(|entry| entry.bytes)
            .sum()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct KvPagePriority {
    page_index: u32,
    pinned: bool,
    distance: u64,
    last_use: u64,
}

pub struct KvResidencyPlanner;

impl KvResidencyPlanner {
    pub fn plan(
        pool: &KvPagePool,
        registry: &BlockRegistry,
        current_step: u64,
        policy: KvResidencyPolicy,
    ) -> Result<KvResidencyPlan> {
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
        let pinned_count = hot_candidates
            .iter()
            .filter(|candidate| candidate.pinned)
            .count();
        let speculative_budget = policy.hot_page_limit.saturating_sub(pinned_count);
        let mut speculative_taken = 0usize;
        let hot_pages = hot_candidates
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
            .collect::<Vec<_>>();

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
            let wants_hot = hot_pages.contains(&page.page_index);
            let idle = current_step.saturating_sub(page.last_use);
            let entry = if wants_hot && old_tier == MemoryTier::Vram {
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
            } else if wants_hot {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Vram,
                    action: KvResidencyAction::PrefetchToHot,
                    reason: "KV page next use is within prefetch window",
                    predicted_visible_ns: transfer_cost_ns(
                        page_token_bytes(page),
                        old_tier,
                        MemoryTier::Vram,
                    ),
                }
            } else if page.ref_count == 0 && idle >= policy.evict_after_idle {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Dram,
                    action: KvResidencyAction::EvictCold,
                    reason: "KV page is idle beyond eviction threshold",
                    predicted_visible_ns: transfer_cost_ns(
                        page_token_bytes(page),
                        old_tier,
                        MemoryTier::Dram,
                    ),
                }
            } else if old_tier == MemoryTier::Vram {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Dram,
                    action: KvResidencyAction::DemoteToWarm,
                    reason: "KV page is outside hot budget",
                    predicted_visible_ns: transfer_cost_ns(
                        page_token_bytes(page),
                        old_tier,
                        MemoryTier::Dram,
                    ),
                }
            } else {
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
            };
            entries.push(entry);
        }
        Ok(KvResidencyPlan { entries })
    }
}

fn page_token_bytes(page: &KvPageDescriptor) -> usize {
    page.page_bytes
}

fn record_visible_transfer_stall(ledger: &mut TokenLedger, entry: &KvResidencyPlanEntry) {
    if entry.predicted_visible_ns == 0 {
        return;
    }
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Stall,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(entry.block_id),
        from_tier: Some(entry.old_tier),
        to_tier: Some(entry.new_tier),
        bytes: entry.bytes,
        latency_ns: entry.predicted_visible_ns,
        label: "kv_visible_transfer_stall",
    });
}

fn transfer_cost_ns(bytes: usize, old_tier: MemoryTier, new_tier: MemoryTier) -> u64 {
    if old_tier == new_tier {
        0
    } else {
        100 + bytes as u64
    }
}
