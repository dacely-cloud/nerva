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
        WeightBlockRole::QueryNorm => 17,
        WeightBlockRole::KeyNorm => 18,
        WeightBlockRole::RouterProjection => 19,
        WeightBlockRole::ExpertGateProjection => 20,
        WeightBlockRole::ExpertUpProjection => 21,
        WeightBlockRole::ExpertGateUpProjection => 22,
        WeightBlockRole::ExpertDownProjection => 23,
        WeightBlockRole::SharedExpertGateProjection => 24,
        WeightBlockRole::SharedExpertUpProjection => 25,
        WeightBlockRole::SharedExpertDownProjection => 26,
        WeightBlockRole::SharedExpertRouterProjection => 27,
        WeightBlockRole::LinearConvProjection => 28,
        WeightBlockRole::LinearQkvProjection => 29,
        WeightBlockRole::LinearZProjection => 30,
        WeightBlockRole::LinearBProjection => 31,
        WeightBlockRole::LinearAProjection => 32,
        WeightBlockRole::LinearDtBias => 33,
        WeightBlockRole::LinearALog => 34,
        WeightBlockRole::LinearNorm => 35,
        WeightBlockRole::LinearOutputProjection => 36,
        WeightBlockRole::DeepSeekQALoraProjection => 37,
        WeightBlockRole::DeepSeekQALoraScaleInv => 38,
        WeightBlockRole::DeepSeekQALoraNorm => 39,
        WeightBlockRole::DeepSeekQBProjection => 40,
        WeightBlockRole::DeepSeekQBScaleInv => 41,
        WeightBlockRole::DeepSeekKvAProjection => 42,
        WeightBlockRole::DeepSeekKvAScaleInv => 43,
        WeightBlockRole::DeepSeekKvANorm => 44,
        WeightBlockRole::DeepSeekKvBProjection => 45,
        WeightBlockRole::DeepSeekKvBScaleInv => 46,
        WeightBlockRole::DeepSeekOutputScaleInv => 47,
        WeightBlockRole::DeepSeekIndexerQueryProjection => 48,
        WeightBlockRole::DeepSeekIndexerQueryScaleInv => 49,
        WeightBlockRole::DeepSeekIndexerKeyProjection => 50,
        WeightBlockRole::DeepSeekIndexerKeyScaleInv => 51,
        WeightBlockRole::DeepSeekIndexerKeyNorm => 52,
        WeightBlockRole::DeepSeekIndexerKeyNormBias => 53,
        WeightBlockRole::DeepSeekIndexerWeightsProjection => 54,
        WeightBlockRole::GateScaleInv => 55,
        WeightBlockRole::UpScaleInv => 56,
        WeightBlockRole::DownScaleInv => 57,
        WeightBlockRole::RouterCorrectionBias => 58,
        WeightBlockRole::ExpertGateScaleInv => 59,
        WeightBlockRole::ExpertUpScaleInv => 60,
        WeightBlockRole::ExpertDownScaleInv => 61,
        WeightBlockRole::SharedExpertGateScaleInv => 62,
        WeightBlockRole::SharedExpertUpScaleInv => 63,
        WeightBlockRole::SharedExpertDownScaleInv => 64,
        WeightBlockRole::DeepSeekV4HcHeadBase => 65,
        WeightBlockRole::DeepSeekV4HcHeadFn => 66,
        WeightBlockRole::DeepSeekV4HcHeadScale => 67,
        WeightBlockRole::DeepSeekV4HcAttnBase => 68,
        WeightBlockRole::DeepSeekV4HcAttnFn => 69,
        WeightBlockRole::DeepSeekV4HcAttnScale => 70,
        WeightBlockRole::DeepSeekV4HcFfnBase => 71,
        WeightBlockRole::DeepSeekV4HcFfnFn => 72,
        WeightBlockRole::DeepSeekV4HcFfnScale => 73,
        WeightBlockRole::DeepSeekV4AttentionSink => 74,
        WeightBlockRole::DeepSeekV4WqAProjection => 75,
        WeightBlockRole::DeepSeekV4WqAScale => 76,
        WeightBlockRole::DeepSeekV4WqBProjection => 77,
        WeightBlockRole::DeepSeekV4WqBScale => 78,
        WeightBlockRole::DeepSeekV4QNorm => 79,
        WeightBlockRole::DeepSeekV4WkvProjection => 80,
        WeightBlockRole::DeepSeekV4WkvScale => 81,
        WeightBlockRole::DeepSeekV4KvNorm => 82,
        WeightBlockRole::DeepSeekV4WoAProjection => 83,
        WeightBlockRole::DeepSeekV4WoAScale => 84,
        WeightBlockRole::DeepSeekV4WoBProjection => 85,
        WeightBlockRole::DeepSeekV4WoBScale => 86,
        WeightBlockRole::DeepSeekV4CompressorApe => 87,
        WeightBlockRole::DeepSeekV4CompressorWkvProjection => 88,
        WeightBlockRole::DeepSeekV4CompressorWgateProjection => 89,
        WeightBlockRole::DeepSeekV4CompressorNorm => 90,
        WeightBlockRole::DeepSeekV4IndexerWqBProjection => 91,
        WeightBlockRole::DeepSeekV4IndexerWqBScale => 92,
        WeightBlockRole::DeepSeekV4IndexerCompressorApe => 93,
        WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection => 94,
        WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection => 95,
        WeightBlockRole::DeepSeekV4IndexerCompressorNorm => 96,
        WeightBlockRole::DeepSeekV4IndexerWeightsProjection => 97,
        WeightBlockRole::DeepSeekV4HashRouteTable => 98,
        WeightBlockRole::DeepSeekV4ExpertGateScale => 99,
        WeightBlockRole::DeepSeekV4ExpertUpScale => 100,
        WeightBlockRole::DeepSeekV4ExpertDownScale => 101,
        WeightBlockRole::DeepSeekV4SharedExpertGateScale => 102,
        WeightBlockRole::DeepSeekV4SharedExpertUpScale => 103,
        WeightBlockRole::DeepSeekV4SharedExpertDownScale => 104,
    }
}

pub(super) fn update_prefetch_data_hash(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
