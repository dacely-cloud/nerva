#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfArchitectureKind {
    Llama,
    Mistral,
    Gemma,
    Qwen2,
    Qwen3,
    Unknown,
}

impl HfArchitectureKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Unknown => "unknown",
        }
    }
}

pub(crate) fn architecture_kind_from_str(value: &str) -> HfArchitectureKind {
    let lower = value.to_ascii_lowercase();
    if lower.contains("llama") {
        HfArchitectureKind::Llama
    } else if lower.contains("mistral") {
        HfArchitectureKind::Mistral
    } else if lower.contains("gemma") {
        HfArchitectureKind::Gemma
    } else if lower.contains("qwen3_5")
        || lower.contains("qwen3.5")
        || lower.contains("qwen3next")
        || lower.contains("qwen3_next")
        || lower.contains("qwen3moe")
        || lower.contains("qwen3_moe")
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
