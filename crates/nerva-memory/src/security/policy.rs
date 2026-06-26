#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SanitizationPhase {
    Maintenance,
    HotPath,
}
