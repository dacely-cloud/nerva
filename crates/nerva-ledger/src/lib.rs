#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{MemoryTier, ResidentBlockId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LedgerEventKind {
    KernelLaunch,
    Copy,
    Sync,
    Allocation,
    Eviction,
    Prefetch,
    Stall,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEvent {
    pub kind: LedgerEventKind,
    pub block_id: Option<ResidentBlockId>,
    pub from_tier: Option<MemoryTier>,
    pub to_tier: Option<MemoryTier>,
    pub bytes: usize,
    pub latency_ns: u64,
    pub label: &'static str,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenLedger {
    pub token_index: u64,
    pub events: Vec<LedgerEvent>,
    pub hot_path_allocations: u64,
}

impl TokenLedger {
    pub fn new(token_index: u64) -> Self {
        Self {
            token_index,
            events: Vec::new(),
            hot_path_allocations: 0,
        }
    }

    pub fn record(&mut self, event: LedgerEvent) {
        if event.kind == LedgerEventKind::Allocation {
            self.hot_path_allocations += 1;
        }
        self.events.push(event);
    }

    pub fn total_latency_ns(&self) -> u64 {
        self.events.iter().map(|event| event.latency_ns).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_events_increment_hot_path_count() {
        let mut ledger = TokenLedger::new(0);
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Allocation,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 64,
            latency_ns: 10,
            label: "test",
        });
        assert_eq!(ledger.hot_path_allocations, 1);
        assert_eq!(ledger.total_latency_ns(), 10);
    }
}
