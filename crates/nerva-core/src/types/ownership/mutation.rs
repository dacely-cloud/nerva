#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MutationSemantics {
    Immutable,
    AppendOnly,
    SingleWriter,
    Ephemeral,
    AtomicControl,
}
