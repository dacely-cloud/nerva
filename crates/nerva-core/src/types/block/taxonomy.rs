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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Lifetime {
    Static,
    Request,
    Token,
    Scratch,
    External,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Hotness {
    Cold,
    Warm,
    Hot,
}
