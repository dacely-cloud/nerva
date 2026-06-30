#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfArchitectureKind {
    Llama,
    Mistral,
    MixtralMoe,
    Gemma,
    Qwen2,
    Qwen2Moe,
    Qwen3,
    Qwen3Moe,
    Qwen35,
    Qwen35Moe,
    DeepSeekV3,
    DeepSeekV32,
    DeepSeekV4,
    Unknown,
}

impl HfArchitectureKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::MixtralMoe => "mixtral_moe",
            Self::Gemma => "gemma",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Moe => "qwen2_moe",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Qwen35 => "qwen3.5",
            Self::Qwen35Moe => "qwen3.5_moe",
            Self::DeepSeekV3 => "deepseek_v3",
            Self::DeepSeekV32 => "deepseek_v3.2",
            Self::DeepSeekV4 => "deepseek_v4",
            Self::Unknown => "unknown",
        }
    }

    pub const fn is_deepseek(self) -> bool {
        matches!(
            self,
            Self::DeepSeekV3 | Self::DeepSeekV32 | Self::DeepSeekV4
        )
    }
}

pub(crate) fn architecture_kind_from_str(value: &str) -> HfArchitectureKind {
    let lower = value.to_ascii_lowercase();
    if lower.contains("deepseekv32")
        || lower.contains("deepseek_v32")
        || lower.contains("deepseek-v32")
        || lower.contains("deepseekv3.2")
        || lower.contains("deepseek_v3.2")
        || lower.contains("deepseek-v3.2")
    {
        HfArchitectureKind::DeepSeekV32
    } else if lower.contains("deepseekv4")
        || lower.contains("deepseek_v4")
        || lower.contains("deepseek-v4")
    {
        HfArchitectureKind::DeepSeekV4
    } else if lower.contains("deepseekv3")
        || lower.contains("deepseek_v3")
        || lower.contains("deepseek-v3")
    {
        HfArchitectureKind::DeepSeekV3
    } else if lower.contains("llama") {
        HfArchitectureKind::Llama
    } else if lower.contains("mixtral") {
        HfArchitectureKind::MixtralMoe
    } else if lower.contains("mistral") {
        HfArchitectureKind::Mistral
    } else if lower.contains("gemma") {
        HfArchitectureKind::Gemma
    } else if lower.contains("qwen2moe")
        || lower.contains("qwen2_moe")
        || lower.contains("qwen2-moe")
        || lower.contains("qwen1.5-moe")
        || lower.contains("qwen1_5_moe")
        || lower.contains("qwen1.5moe")
    {
        HfArchitectureKind::Qwen2Moe
    } else if lower.contains("qwen3_5_moe")
        || lower.contains("qwen3.5_moe")
        || lower.contains("qwen35_moe")
        || lower.contains("qwen35moe")
        || lower.contains("qwen3_5moe")
        || lower.contains("qwen3.5moe")
    {
        HfArchitectureKind::Qwen35Moe
    } else if lower.contains("qwen3_5") || lower.contains("qwen3.5") || lower.contains("qwen35") {
        HfArchitectureKind::Qwen35
    } else if lower.contains("qwen3moe") || lower.contains("qwen3_moe") {
        HfArchitectureKind::Qwen3Moe
    } else if lower.contains("qwen3next")
        || lower.contains("qwen3_next")
        || lower.contains("qwen3vl")
        || lower.contains("qwen3_vl")
    {
        HfArchitectureKind::Unknown
    } else if lower == "qwen3" || lower == "qwen3forcausallm" {
        HfArchitectureKind::Qwen3
    } else if lower.contains("qwen2") {
        HfArchitectureKind::Qwen2
    } else {
        HfArchitectureKind::Unknown
    }
}
