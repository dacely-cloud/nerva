use crate::types::metric::MetricSource;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FallbackClass {
    ExactNamed,
    CapabilityDegraded,
    PolicySelected,
    DebugOnly,
}

impl FallbackClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactNamed => "exact_named",
            Self::CapabilityDegraded => "capability_degraded",
            Self::PolicySelected => "policy_selected",
            Self::DebugOnly => "debug_only",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FallbackDecision {
    pub label: &'static str,
    pub class: FallbackClass,
    pub requested: &'static str,
    pub selected: &'static str,
    pub reason: &'static str,
    pub visible_ns: Option<u64>,
    pub metric_source: MetricSource,
}
