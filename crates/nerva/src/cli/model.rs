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
        "unknown model '{model}'. Use a checkpoint path or a known alias like qwen3-8b"
    ))
}

pub(crate) fn model_alias(model: &str) -> Option<&'static str> {
    match model.to_ascii_lowercase().as_str() {
        "qwen3" | "qwen3-8b" | "qwen/qwen3-8b" => Some("Qwen/Qwen3-8B"),
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
    }
}
