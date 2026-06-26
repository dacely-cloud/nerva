use nerva_core::types::dtype::DType;

use crate::registry::types::backend::KernelBackend;
use crate::registry::types::operation::KernelOperation;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelFallbackClass {
    ExactNamed,
    ApproximateNamed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelFallback {
    pub operation: KernelOperation,
    pub requested_backend: KernelBackend,
    pub requested_dtype: DType,
    pub fallback_backend: KernelBackend,
    pub fallback_dtype: DType,
    pub name: &'static str,
    pub class: KernelFallbackClass,
}
