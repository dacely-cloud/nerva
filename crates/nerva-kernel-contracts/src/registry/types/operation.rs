#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelOperation {
    DenseMatVec,
    BlockwiseAttention,
    KvAppend,
    GreedySample,
}
