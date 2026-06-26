#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct BlockFlags {
    bits: u32,
}

impl BlockFlags {
    pub const PREFETCHABLE: u32 = 1 << 0;
    pub const EVICTABLE: u32 = 1 << 1;
    pub const TRANSPORT_REGISTERED: u32 = 1 << 2;
    pub const SENSITIVE: u32 = 1 << 3;

    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    pub const fn from_bits(bits: u32) -> Self {
        Self { bits }
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }

    pub const fn contains(self, flag: u32) -> bool {
        (self.bits & flag) == flag
    }
}
