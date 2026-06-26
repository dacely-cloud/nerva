#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CostSource {
    Unknown,
    Estimated,
    Measured,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CostEstimate {
    pub nanos: Option<u64>,
    pub source: CostSource,
}

impl CostEstimate {
    pub const fn unknown() -> Self {
        Self {
            nanos: None,
            source: CostSource::Unknown,
        }
    }

    pub const fn estimated_nanos(nanos: u64) -> Self {
        Self {
            nanos: Some(nanos),
            source: CostSource::Estimated,
        }
    }

    pub const fn measured_nanos(nanos: u64) -> Self {
        Self {
            nanos: Some(nanos),
            source: CostSource::Measured,
        }
    }
}
