use nerva_core::types::dtype::DType;

use crate::registry::types::backend::KernelBackend;
use crate::registry::types::operation::KernelOperation;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelQuery {
    pub operation: KernelOperation,
    pub backend: KernelBackend,
    pub dtype: DType,
    pub compute_capability: Option<u32>,
}

impl KernelQuery {
    pub const fn new(
        operation: KernelOperation,
        backend: KernelBackend,
        dtype: DType,
        compute_capability: Option<u32>,
    ) -> Self {
        Self {
            operation,
            backend,
            dtype,
            compute_capability,
        }
    }
}
