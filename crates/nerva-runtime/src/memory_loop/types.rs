use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryLoopTaskKind {
    DiskRead,
    Prefetch,
    Stage,
    Evict,
    PrepareTransportBuffer,
}

impl MemoryLoopTaskKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::DiskRead => "disk_read",
            Self::Prefetch => "prefetch",
            Self::Stage => "stage",
            Self::Evict => "evict",
            Self::PrepareTransportBuffer => "prepare_transport_buffer",
        }
    }

    pub const fn is_eviction(self) -> bool {
        matches!(self, Self::Evict)
    }

    pub const fn is_prefetch_like(self) -> bool {
        matches!(self, Self::DiskRead | Self::Prefetch | Self::Stage)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MemoryLoopTaskSpec {
    pub block_id: ResidentBlockId,
    pub kind: MemoryLoopTaskKind,
    pub from_tier: MemoryTier,
    pub to_tier: MemoryTier,
    pub bytes: usize,
    pub predicted_visible_ns: u64,
    pub overlap_window_ns: u64,
    pub label: &'static str,
}

impl MemoryLoopTaskSpec {
    pub const fn new(
        block_id: ResidentBlockId,
        kind: MemoryLoopTaskKind,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        bytes: usize,
        predicted_visible_ns: u64,
        label: &'static str,
    ) -> Self {
        Self {
            block_id,
            kind,
            from_tier,
            to_tier,
            bytes,
            predicted_visible_ns,
            overlap_window_ns: 0,
            label,
        }
    }

    pub const fn with_overlap(mut self, overlap_window_ns: u64) -> Self {
        self.overlap_window_ns = overlap_window_ns;
        self
    }

    pub const fn visible_after_overlap(self) -> u64 {
        self.predicted_visible_ns
            .saturating_sub(self.overlap_window_ns)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryLoopConfig {
    pub queue_capacity: usize,
    pub max_inflight: usize,
    pub tasks: Vec<MemoryLoopTaskSpec>,
}

impl MemoryLoopConfig {
    pub fn new(queue_capacity: usize, max_inflight: usize) -> Self {
        Self {
            queue_capacity,
            max_inflight,
            tasks: Vec::new(),
        }
    }

    pub fn with_task(mut self, task: MemoryLoopTaskSpec) -> Self {
        self.tasks.push(task);
        self
    }
}
