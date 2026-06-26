#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphLayoutHash(pub [u8; 32]);

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

fn mix_u32(out: &mut [u8; 32], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    for (idx, byte) in bytes.iter().enumerate() {
        out[offset + idx] ^= *byte;
        out[31 - offset - idx] = out[31 - offset - idx].wrapping_add(byte.rotate_left(1));
    }
}
