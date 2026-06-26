#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockKind {
    Weight,
    KvPage,
    Activation,
    Logits,
    TokenState,
    SamplerState,
    Workspace,
    Queue,
    TransportBuffer,
    Ledger,
    Metadata,
}
