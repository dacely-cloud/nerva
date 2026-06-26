use nerva_core::types::dtype::DType;

use crate::registry::types::architecture::ArchitectureRange;
use crate::registry::types::backend::KernelBackend;
use crate::registry::types::exactness::KernelExactness;
use crate::registry::types::operation::KernelOperation;
use crate::registry::types::query::KernelQuery;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelImplementation {
    pub name: &'static str,
    pub operation: KernelOperation,
    pub backend: KernelBackend,
    pub architecture: Option<ArchitectureRange>,
    pub dtypes: &'static [DType],
    pub graph_safe: bool,
    pub deterministic: bool,
    pub exactness: KernelExactness,
}

impl KernelImplementation {
    pub fn matches(self, query: KernelQuery) -> bool {
        if self.operation != query.operation || self.backend != query.backend {
            return false;
        }
        if !self.dtypes.contains(&query.dtype) {
            return false;
        }
        match (self.architecture, query.compute_capability) {
            (Some(range), Some(compute_capability)) => range.contains(compute_capability),
            (Some(_), None) => false,
            (None, _) => true,
        }
    }
}
