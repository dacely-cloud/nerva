use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn resolve_prompt_text(prompt: &str) -> Result<String, String> {
    let Some(path) = prompt.strip_prefix('@') else {
        return Ok(prompt.to_string());
    };
    if path.is_empty() {
        return Err("prompt file path is empty".to_string());
    }
    fs::read_to_string(path).map_err(|err| format!("failed to read prompt file {path}: {err}"))
}

pub(crate) fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
    let direct = PathBuf::from(model);
    if direct.exists() {
        return Ok(direct);
    }
    let repo = model_alias(model).unwrap_or(model);
    if repo.contains('/') {
        return resolve_hf_snapshot(repo);
    }
    Err(format!(
        "unknown model '{model}'. Use a checkpoint path, Hugging Face repo id, or a known alias like qwen3-8b, qwen3-moe, or qwen-coder"
    ))
}

pub(crate) fn model_alias(model: &str) -> Option<&'static str> {
    match model.to_ascii_lowercase().as_str() {
        "qwen3" | "qwen3-8b" | "qwen/qwen3-8b" => Some("Qwen/Qwen3-8B"),
        "qwen-moe"
        | "qwen2-moe"
        | "qwen1.5-moe"
        | "qwen1.5-moe-a2.7b"
        | "qwen/qwen1.5-moe-a2.7b" => Some("Qwen/Qwen1.5-MoE-A2.7B"),
        "qwen-moe-chat"
        | "qwen2-moe-chat"
        | "qwen1.5-moe-chat"
        | "qwen1.5-moe-a2.7b-chat"
        | "qwen/qwen1.5-moe-a2.7b-chat" => Some("Qwen/Qwen1.5-MoE-A2.7B-Chat"),
        "qwen3-moe" | "qwen3-a3b" | "qwen3-30b-a3b" | "qwen/qwen3-30b-a3b" => {
            Some("Qwen/Qwen3-30B-A3B")
        }
        "qwen3-235b" | "qwen3-235b-a22b" | "qwen3-a22b" | "qwen/qwen3-235b-a22b" => {
            Some("Qwen/Qwen3-235B-A22B")
        }
        "qwen-coder"
        | "coder"
        | "qwen2.5-coder"
        | "qwen2.5-coder-7b"
        | "qwen2.5-coder-7b-instruct"
        | "qwen/qwen2.5-coder-7b-instruct" => Some("Qwen/Qwen2.5-Coder-7B-Instruct"),
        "qwen2.5" | "qwen2.5-7b" | "qwen2.5-7b-instruct" | "qwen/qwen2.5-7b-instruct" => {
            Some("Qwen/Qwen2.5-7B-Instruct")
        }
        "qwen3.5" | "qwen3.5-4b" | "qwen/qwen3.5-4b" => Some("Qwen/Qwen3.5-4B"),
        "qwen3.5-8b" | "qwen/qwen3.5-8b" => Some("Qwen/Qwen3.5-8B"),
        "qwen3.5-moe"
        | "qwen3.5-a3b"
        | "qwen3.5-35b"
        | "qwen3.5-35b-a3b"
        | "qwen/qwen3.5-35b-a3b" => Some("Qwen/Qwen3.5-35B-A3B"),
        "qwen3-coder"
        | "qwen3-coder-30b"
        | "qwen3-coder-30b-a3b"
        | "qwen3-coder-30b-a3b-instruct"
        | "qwen/qwen3-coder-30b-a3b-instruct" => Some("Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        "qwen3-coder-480b"
        | "qwen3-coder-480b-a35b"
        | "qwen3-coder-480b-a35b-instruct"
        | "qwen/qwen3-coder-480b-a35b-instruct" => Some("Qwen/Qwen3-Coder-480B-A35B-Instruct"),
        "mixtral"
        | "mixtral-8x7b"
        | "mixtral-8x7b-instruct"
        | "mistralai/mixtral-8x7b-instruct-v0.1" => Some("mistralai/Mixtral-8x7B-Instruct-v0.1"),
        "llama3.1"
        | "llama3.1-8b"
        | "llama3.1-8b-instruct"
        | "meta-llama/llama-3.1-8b-instruct" => Some("meta-llama/Llama-3.1-8B-Instruct"),
        "mistral" | "mistral-7b" | "mistral-7b-instruct" | "mistralai/mistral-7b-instruct-v0.3" => {
            Some("mistralai/Mistral-7B-Instruct-v0.3")
        }
        _ => None,
    }
}

fn resolve_hf_snapshot(repo: &str) -> Result<PathBuf, String> {
    let hub_dir = hf_hub_dir().join(format!("models--{}", repo.replace('/', "--")));
    let snapshots_dir = hub_dir.join("snapshots");
    let mut snapshots = fs::read_dir(&snapshots_dir)
        .map_err(|err| {
            format!(
                "model {repo} is not available in local Hugging Face cache at {}: {err}",
                snapshots_dir.display()
            )
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    snapshots.sort();
    snapshots.pop().ok_or_else(|| {
        format!(
            "model {repo} has no local snapshots in {}",
            snapshots_dir.display()
        )
    })
}

fn hf_hub_dir() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .map(|path| path.join("hub"))
        })
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join(".cache/huggingface/hub"))
        })
        .unwrap_or_else(|| Path::new("/root/.cache/huggingface/hub").to_path_buf())
}

pub(crate) fn detect_cuda_compute_capability() -> Option<u32> {
    let smoke = nerva_runtime::capabilities::discovery::cuda_smoke();
    let major = smoke.compute_capability_major?;
    let minor = smoke.compute_capability_minor?;
    if major < 0 || minor < 0 {
        return None;
    }
    Some((major as u32) * 10 + minor as u32)
}

#[cfg(test)]
mod tests {
    use super::model_alias;

    #[test]
    fn resolves_qwen_alias() {
        assert_eq!(model_alias("qwen3-8b"), Some("Qwen/Qwen3-8B"));
        assert_eq!(model_alias("Qwen/Qwen3-8B"), Some("Qwen/Qwen3-8B"));
        assert_eq!(model_alias("qwen2-moe"), Some("Qwen/Qwen1.5-MoE-A2.7B"));
        assert_eq!(
            model_alias("qwen1.5-moe-chat"),
            Some("Qwen/Qwen1.5-MoE-A2.7B-Chat")
        );
        assert_eq!(model_alias("qwen3-moe"), Some("Qwen/Qwen3-30B-A3B"));
        assert_eq!(model_alias("qwen3-235b-a22b"), Some("Qwen/Qwen3-235B-A22B"));
        assert_eq!(model_alias("qwen3.5-moe"), Some("Qwen/Qwen3.5-35B-A3B"));
    }

    #[test]
    fn resolves_supported_dense_model_aliases() {
        assert_eq!(
            model_alias("qwen-coder"),
            Some("Qwen/Qwen2.5-Coder-7B-Instruct")
        );
        assert_eq!(
            model_alias("qwen2.5-7b-instruct"),
            Some("Qwen/Qwen2.5-7B-Instruct")
        );
        assert_eq!(model_alias("qwen3.5"), Some("Qwen/Qwen3.5-4B"));
        assert_eq!(
            model_alias("qwen3-coder"),
            Some("Qwen/Qwen3-Coder-30B-A3B-Instruct")
        );
        assert_eq!(
            model_alias("mixtral-8x7b-instruct"),
            Some("mistralai/Mixtral-8x7B-Instruct-v0.1")
        );
        assert_eq!(
            model_alias("llama3.1-8b-instruct"),
            Some("meta-llama/Llama-3.1-8B-Instruct")
        );
        assert_eq!(
            model_alias("mistral-7b-instruct"),
            Some("mistralai/Mistral-7B-Instruct-v0.3")
        );
    }
}
