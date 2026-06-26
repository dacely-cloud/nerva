use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};

use crate::graph::descriptor::{CapturedGraphDescriptor, GraphFingerprint};
use crate::graph::layout::{GraphKey, GraphLayout};

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
