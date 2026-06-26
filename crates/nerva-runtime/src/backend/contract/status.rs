#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BackendContractProbeStatus {
    Ok,
    Unavailable,
    Failed,
}

impl BackendContractProbeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Unavailable => "unavailable",
            Self::Failed => "failed",
        }
    }
}
