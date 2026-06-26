use crate::types::id::replica::ReplicaId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidencyState {
    Unmapped,
    Allocated,
    Prefetching,
    Ready,
    InUse,
    Draining,
    Evicting,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencySet {
    replicas: Vec<ReplicaId>,
}

impl ResidencySet {
    pub fn empty() -> Self {
        Self {
            replicas: Vec::new(),
        }
    }

    pub fn single(replica: ReplicaId) -> Self {
        Self {
            replicas: vec![replica],
        }
    }

    pub fn contains(&self, replica: ReplicaId) -> bool {
        self.replicas.contains(&replica)
    }

    pub fn add(&mut self, replica: ReplicaId) {
        if !self.contains(replica) {
            self.replicas.push(replica);
        }
    }

    pub fn replicas(&self) -> &[ReplicaId] {
        &self.replicas
    }
}
