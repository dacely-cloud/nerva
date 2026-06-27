pub const CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT: u32 = 1;
pub const CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED: u32 = 2;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceWeightBlock {
    pub block_id: u64,
    pub block_version: u64,
    pub offset_bytes: u64,
    pub bytes: u64,
    pub strategy: u32,
    pub reserved: u32,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceWeightPlan {
    pub blocks: u32,
    pub gpu_resident_blocks: u32,
    pub gpu_staged_blocks: u32,
    pub weight_bytes: u64,
    pub gpu_resident_weight_bytes: u64,
    pub gpu_staged_weight_bytes: u64,
    pub descriptor_hash: u64,
}

impl CudaHfDecodeSequenceWeightPlan {
    pub const fn empty() -> Self {
        Self {
            blocks: 0,
            gpu_resident_blocks: 0,
            gpu_staged_blocks: 0,
            weight_bytes: 0,
            gpu_resident_weight_bytes: 0,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 0,
        }
    }

    pub const fn is_declared(self) -> bool {
        self.blocks != 0 || self.weight_bytes != 0
    }

    pub fn validate(self) -> Option<String> {
        if !self.is_declared() {
            return None;
        }
        if self.blocks == 0 || self.weight_bytes == 0 || self.descriptor_hash == 0 {
            return Some("CUDA HF decode sequence weight plan is incomplete".to_string());
        }
        if self.gpu_resident_blocks > self.blocks
            || self.gpu_staged_blocks > self.blocks - self.gpu_resident_blocks
        {
            return Some("CUDA HF decode sequence weight block counts are invalid".to_string());
        }
        if self.gpu_resident_weight_bytes > self.weight_bytes
            || self.gpu_staged_weight_bytes > self.weight_bytes - self.gpu_resident_weight_bytes
        {
            return Some("CUDA HF decode sequence weight byte counts are invalid".to_string());
        }
        None
    }

    pub fn validate_descriptors(
        self,
        descriptors: &[CudaHfDecodeSequenceWeightBlock],
    ) -> Option<String> {
        if !self.is_declared() {
            return None;
        }
        if descriptors.len() != self.blocks as usize {
            return Some("CUDA HF decode sequence weight descriptor count is invalid".to_string());
        }
        if hash_weight_blocks(descriptors) != self.descriptor_hash {
            return Some("CUDA HF decode sequence weight descriptor hash is invalid".to_string());
        }
        None
    }
}

pub fn hash_weight_blocks(descriptors: &[CudaHfDecodeSequenceWeightBlock]) -> u64 {
    let mut hash = FNV_OFFSET;
    for descriptor in descriptors {
        hash_u64(&mut hash, descriptor.block_id);
        hash_u64(&mut hash, descriptor.block_version);
        hash_u64(&mut hash, descriptor.offset_bytes);
        hash_u64(&mut hash, descriptor.bytes);
        hash_u32(&mut hash, descriptor.strategy);
    }
    hash
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}
