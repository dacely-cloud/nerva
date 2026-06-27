use crate::common::hash::hash_bytes;

pub(super) struct LoadAccounting {
    pub bytes_loaded: usize,
    data_hash: u64,
}

impl LoadAccounting {
    pub const fn new() -> Self {
        Self {
            bytes_loaded: 0,
            data_hash: 0xcbf2_9ce4_8422_2325,
        }
    }

    pub fn record(&mut self, bytes: usize, data_hash: u64) {
        self.bytes_loaded += bytes;
        self.data_hash = fold_hash(self.data_hash, data_hash);
    }

    pub const fn data_hash(&self, available: bool) -> u64 {
        if available { self.data_hash } else { 0 }
    }
}

fn fold_hash(hash: u64, value: u64) -> u64 {
    let mut bytes = hash.to_le_bytes().to_vec();
    bytes.extend_from_slice(&value.to_le_bytes());
    hash_bytes(&bytes)
}
