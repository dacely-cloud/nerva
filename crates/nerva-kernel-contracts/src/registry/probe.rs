use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;

use crate::registry::bootstrap::bootstrap_registry;
use crate::registry::types::backend::KernelBackend;
use crate::registry::types::fallback::{KernelFallback, KernelFallbackClass};
use crate::registry::types::operation::KernelOperation;
use crate::registry::types::plan::KernelPlan;
use crate::registry::types::query::KernelQuery;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelRegistryProbeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelRegistryProbeSummary {
    pub status: KernelRegistryProbeStatus,
    pub implementations: usize,
    pub fallbacks: usize,
    pub direct_plans: u64,
    pub fallback_plans: u64,
    pub rejected_plans: u64,
    pub graph_safe_direct: u64,
    pub exact_fallbacks: u64,
}

impl KernelRegistryProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            KernelRegistryProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"implementations\":{},\"fallbacks\":{},\"direct_plans\":{},\"fallback_plans\":{},\"rejected_plans\":{},\"graph_safe_direct\":{},\"exact_fallbacks\":{}}}",
            status,
            self.implementations,
            self.fallbacks,
            self.direct_plans,
            self.fallback_plans,
            self.rejected_plans,
            self.graph_safe_direct,
            self.exact_fallbacks,
        )
    }
}

pub fn kernel_registry_probe() -> Result<KernelRegistryProbeSummary> {
    let registry = bootstrap_registry();
    let direct = registry.resolve(KernelQuery::new(
        KernelOperation::DenseMatVec,
        KernelBackend::Cuda,
        DType::F16,
        Some(89),
    ))?;
    let fallback = registry.resolve(KernelQuery::new(
        KernelOperation::DenseMatVec,
        KernelBackend::Cuda,
        DType::F32,
        Some(89),
    ))?;
    let rejected = registry
        .resolve(KernelQuery::new(
            KernelOperation::DenseMatVec,
            KernelBackend::Hip,
            DType::BF16,
            Some(1100),
        ))
        .is_err();

    Ok(KernelRegistryProbeSummary {
        status: KernelRegistryProbeStatus::Ok,
        implementations: registry.implementations().len(),
        fallbacks: registry.fallbacks().len(),
        direct_plans: u64::from(!direct.is_fallback()),
        fallback_plans: u64::from(fallback.is_fallback()),
        rejected_plans: u64::from(rejected),
        graph_safe_direct: u64::from(direct.implementation().graph_safe),
        exact_fallbacks: u64::from(matches!(
            fallback,
            KernelPlan::Fallback {
                policy: KernelFallback {
                    class: KernelFallbackClass::ExactNamed,
                    ..
                },
                ..
            }
        )),
    })
}
