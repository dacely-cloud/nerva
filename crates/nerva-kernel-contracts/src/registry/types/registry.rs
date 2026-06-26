use nerva_core::types::error::{NervaError, Result};

use crate::registry::types::fallback::{KernelFallback, KernelFallbackClass};
use crate::registry::types::implementation::KernelImplementation;
use crate::registry::types::plan::KernelPlan;
use crate::registry::types::query::KernelQuery;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelContractRegistry {
    implementations: Vec<KernelImplementation>,
    fallbacks: Vec<KernelFallback>,
}

impl KernelContractRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_implementation(mut self, implementation: KernelImplementation) -> Self {
        self.implementations.push(implementation);
        self
    }

    pub fn with_fallback(mut self, fallback: KernelFallback) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    pub fn implementations(&self) -> &[KernelImplementation] {
        &self.implementations
    }

    pub fn fallbacks(&self) -> &[KernelFallback] {
        &self.fallbacks
    }

    pub fn resolve(&self, query: KernelQuery) -> Result<KernelPlan> {
        if let Some(implementation) = self
            .implementations
            .iter()
            .copied()
            .find(|implementation| implementation.matches(query))
        {
            return Ok(KernelPlan::Direct { implementation });
        }

        for policy in &self.fallbacks {
            if policy.operation != query.operation
                || policy.requested_backend != query.backend
                || policy.requested_dtype != query.dtype
            {
                continue;
            }
            if policy.class != KernelFallbackClass::ExactNamed {
                return Err(NervaError::InvalidArgument {
                    reason: format!("kernel fallback {} is not exact", policy.name),
                });
            }
            let fallback_query = KernelQuery::new(
                query.operation,
                policy.fallback_backend,
                policy.fallback_dtype,
                None,
            );
            if let Some(fallback) = self
                .implementations
                .iter()
                .copied()
                .find(|implementation| implementation.matches(fallback_query))
            {
                return Ok(KernelPlan::Fallback {
                    requested: query,
                    fallback,
                    policy: *policy,
                });
            }
            return Err(NervaError::InvalidArgument {
                reason: format!("declared fallback {} has no matching contract", policy.name),
            });
        }

        Err(NervaError::InvalidArgument {
            reason: format!("no kernel contract for {:?}", query),
        })
    }
}
