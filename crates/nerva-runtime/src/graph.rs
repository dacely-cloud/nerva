use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::graph::CudaSyntheticGraphSummary;

pub fn cuda_synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> CudaSyntheticGraphSummary {
    crate::engine::cuda::cuda_synthetic_graph_smoke(steps, ring_capacity, seed_token)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphLayoutHash(pub [u8; 32]);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphFingerprint(pub [u8; 32]);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GraphKey {
    pub bucket: u32,
    pub max_blocks: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphLayout {
    pub key: GraphKey,
    pub token_ring_capacity: u32,
    pub static_address_count: u32,
}

impl GraphLayout {
    pub const fn new(
        bucket: u32,
        max_blocks: u32,
        token_ring_capacity: u32,
        static_address_count: u32,
    ) -> Self {
        Self {
            key: GraphKey { bucket, max_blocks },
            token_ring_capacity,
            static_address_count,
        }
    }

    pub fn hash(self) -> GraphLayoutHash {
        let mut out = [0u8; 32];
        mix_u32(&mut out, 0, self.key.bucket);
        mix_u32(&mut out, 4, self.key.max_blocks);
        mix_u32(&mut out, 8, self.token_ring_capacity);
        mix_u32(&mut out, 12, self.static_address_count);
        GraphLayoutHash(out)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphDescriptor {
    pub key: GraphKey,
    pub layout_hash: GraphLayoutHash,
    pub fingerprint: GraphFingerprint,
    pub replay_count: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GraphPool {
    graphs: BTreeMap<GraphKey, CapturedGraphDescriptor>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, graph: CapturedGraphDescriptor) {
        self.graphs.insert(graph.key, graph);
    }

    pub fn capture_synthetic(&mut self, layout: GraphLayout) {
        let hash = layout.hash();
        self.insert(CapturedGraphDescriptor {
            key: layout.key,
            layout_hash: hash,
            fingerprint: GraphFingerprint(hash.0),
            replay_count: 0,
        });
    }

    pub fn check_before_replay(&self, layout: GraphLayout) -> Result<&CapturedGraphDescriptor> {
        let graph = self
            .graphs
            .get(&layout.key)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!(
                    "missing captured graph bucket={} max_blocks={}",
                    layout.key.bucket, layout.key.max_blocks
                ),
            })?;
        if graph.layout_hash != layout.hash() {
            return Err(NervaError::InvalidArgument {
                reason: "captured graph layout hash does not match replay layout".to_string(),
            });
        }
        Ok(graph)
    }

    pub fn replay(&mut self, layout: GraphLayout) -> Result<()> {
        self.check_before_replay(layout)?;
        let graph = self
            .graphs
            .get_mut(&layout.key)
            .expect("graph was checked before replay");
        graph.replay_count = graph.replay_count.saturating_add(1);
        Ok(())
    }

    pub fn replay_count(&self, key: GraphKey) -> Option<u64> {
        self.graphs.get(&key).map(|graph| graph.replay_count)
    }
}

fn mix_u32(out: &mut [u8; 32], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    for (idx, byte) in bytes.iter().enumerate() {
        out[offset + idx] ^= *byte;
        out[31 - offset - idx] = out[31 - offset - idx].wrapping_add(byte.rotate_left(1));
    }
}
