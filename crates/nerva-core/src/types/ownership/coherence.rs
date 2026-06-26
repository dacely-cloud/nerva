#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CoherencePolicy {
    ExplicitVersioned,
    CoherentReadMostly,
    CoherentPhaseOwned,
    AtomicControlOnly,
}
