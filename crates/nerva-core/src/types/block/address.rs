use crate::types::id::allocation::AllocationId;
use crate::types::id::memory::MemoryDomainId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GlobalBlockAddress {
    pub domain: MemoryDomainId,
    pub allocation: AllocationId,
    pub offset: u64,
}

impl GlobalBlockAddress {
    pub const fn unmapped() -> Self {
        Self {
            domain: MemoryDomainId(0),
            allocation: AllocationId(0),
            offset: 0,
        }
    }
}
