#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HotPathGuardStatus {
    Ok,
    Failed,
}

impl HotPathGuardStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Failed => "failed",
        }
    }
}
