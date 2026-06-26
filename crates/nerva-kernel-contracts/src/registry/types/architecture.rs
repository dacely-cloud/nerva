#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArchitectureRange {
    pub min_compute_capability: u32,
    pub max_compute_capability: u32,
}

impl ArchitectureRange {
    pub const fn new(min_compute_capability: u32, max_compute_capability: u32) -> Self {
        Self {
            min_compute_capability,
            max_compute_capability,
        }
    }

    pub const fn contains(self, compute_capability: u32) -> bool {
        compute_capability >= self.min_compute_capability
            && compute_capability <= self.max_compute_capability
    }
}
