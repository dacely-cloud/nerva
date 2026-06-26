#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyncClass {
    HardSync,
    SoftVisibilitySync,
    PolicySync,
    PhaseHandoff,
    DebugSync,
}

impl SyncClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HardSync => "hard_sync",
            Self::SoftVisibilitySync => "soft_visibility_sync",
            Self::PolicySync => "policy_sync",
            Self::PhaseHandoff => "phase_handoff",
            Self::DebugSync => "debug_sync",
        }
    }
}
