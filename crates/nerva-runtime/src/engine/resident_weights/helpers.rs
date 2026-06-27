use nerva_core::types::memory::tier::MemoryTier;
use nerva_model::weights::layout::entry::WeightBlockRole;

pub(super) fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    value / divisor + u64::from(value % divisor != 0)
}

pub(super) fn estimate_cpu_dram_weight_ns(bytes: usize) -> u64 {
    250 + div_ceil_u64(bytes as u64, 8)
}

pub(super) fn estimate_gpu_resident_weight_ns(bytes: usize) -> u64 {
    80 + div_ceil_u64(bytes as u64, 128)
}

pub(super) fn estimate_gpu_staged_weight_ns(bytes: usize) -> u64 {
    5_000 + div_ceil_u64(bytes as u64, 24) + estimate_gpu_resident_weight_ns(bytes)
}

pub(super) fn estimate_cpu_fallback_weight_ns(bytes: usize, tier: MemoryTier) -> u64 {
    let copy_ns = match tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => div_ceil_u64(bytes as u64, 24),
        _ => 0,
    };
    copy_ns + estimate_cpu_dram_weight_ns(bytes)
}

pub(super) fn weight_role_layout_id(role: WeightBlockRole) -> u32 {
    match role {
        WeightBlockRole::TokenEmbedding => 1,
        WeightBlockRole::AttentionNorm => 2,
        WeightBlockRole::QueryProjection => 3,
        WeightBlockRole::KeyProjection => 4,
        WeightBlockRole::ValueProjection => 5,
        WeightBlockRole::OutputProjection => 6,
        WeightBlockRole::MlpNorm => 7,
        WeightBlockRole::GateProjection => 8,
        WeightBlockRole::UpProjection => 9,
        WeightBlockRole::DownProjection => 10,
        WeightBlockRole::FinalNorm => 11,
        WeightBlockRole::LmHead => 12,
        WeightBlockRole::QueryBias => 13,
        WeightBlockRole::KeyBias => 14,
        WeightBlockRole::ValueBias => 15,
        WeightBlockRole::OutputBias => 16,
    }
}

pub(super) fn update_prefetch_data_hash(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
