use crate::graph::layout::{GraphKey, GraphLayoutHash};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphFingerprint(pub [u8; 32]);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphDescriptor {
    pub key: GraphKey,
    pub layout_hash: GraphLayoutHash,
    pub fingerprint: GraphFingerprint,
    pub replay_count: u64,
}
