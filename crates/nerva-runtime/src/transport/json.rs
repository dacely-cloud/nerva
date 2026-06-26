use nerva_core::types::memory::MemoryTier;

pub(crate) fn memory_tier_to_str(value: MemoryTier) -> &'static str {
    match value {
        MemoryTier::Vram => "VRAM",
        MemoryTier::SharedHbmOrLpddr => "SHARED_HBM_OR_LPDDR",
        MemoryTier::PinnedDram => "PINNED_DRAM",
        MemoryTier::Dram => "DRAM",
        MemoryTier::Cxl => "CXL",
        MemoryTier::Disk => "DISK",
    }
}

pub(crate) fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
