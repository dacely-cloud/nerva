use nerva_core::types::id::ResidentBlockId;
use nerva_core::types::memory::MemoryTier;

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

pub struct KvResidencyPlanner;
