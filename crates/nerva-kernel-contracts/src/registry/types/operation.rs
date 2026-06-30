#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelOperation {
    DenseMatVec,
    BlockDequant,
    BlockwiseAttention,
    KvAppend,
    SparseMoeExpert,
    GreedySample,
}
