use nerva_core::types::id::block::ResidentBlockId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReferenceTransactionBlocks {
    pub(crate) device_token: ResidentBlockId,
    pub(crate) weight_tile: ResidentBlockId,
    pub(crate) kv_page: ResidentBlockId,
    pub(crate) logits: ResidentBlockId,
    pub(crate) host_token: ResidentBlockId,
}
