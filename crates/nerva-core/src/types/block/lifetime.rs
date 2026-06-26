#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Lifetime {
    Static,
    Request,
    Token,
    Scratch,
    External,
}
