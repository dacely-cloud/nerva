use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn find_cuda_root() -> Option<PathBuf> {
    if let Ok(cuda_home) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        let root = PathBuf::from(cuda_home);
        if root.join("bin/nvcc").is_file() {
            return Some(root);
        }
    }

    let mut candidates = Vec::new();
    for parent in ["/usr/local", "/opt"] {
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if name.starts_with("cuda") && path.join("bin/nvcc").is_file() {
                    candidates.push(path);
                }
            }
        }
    }
    candidates.push(PathBuf::from("/usr/lib/cuda"));
    candidates.retain(|candidate| candidate.join("bin/nvcc").is_file());
    candidates.sort_by(|left, right| right.cmp(left));
    candidates.into_iter().next()
}

pub(crate) fn find_nvcc(cuda_root: Option<&PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cudacxx) = env::var("CUDACXX") {
        candidates.push(PathBuf::from(cudacxx));
    }
    if let Some(cuda_root) = cuda_root {
        candidates.push(cuda_root.join("bin/nvcc"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    candidates.push(PathBuf::from("nvcc"));

    candidates.into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    })
}

pub(crate) fn find_cuda_target_dir(cuda_root: Option<&PathBuf>, target: &str) -> Option<String> {
    let cuda_root = cuda_root?;
    let targets_dir = cuda_root.join("targets");
    let candidates = cuda_target_dir_candidates(target);
    candidates
        .into_iter()
        .find(|candidate| targets_dir.join(candidate).is_dir())
}

pub(crate) fn cuda_target_dir_candidates(target: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    if target.starts_with("x86_64-") {
        candidates.push("x86_64-linux".to_string());
    } else if target.starts_with("aarch64-") {
        candidates.push("aarch64-linux".to_string());
        candidates.push("sbsa-linux".to_string());
    }
    candidates
}

pub(crate) fn find_cuda_lib_dir(
    cuda_root: Option<&PathBuf>,
    cuda_target_dir: Option<&str>,
    target: &str,
) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let (Some(cuda_root), Some(cuda_target_dir)) = (cuda_root, cuda_target_dir) {
        let target_root = cuda_root.join("targets").join(cuda_target_dir);
        candidates.push(target_root.join("lib64"));
        candidates.push(target_root.join("lib"));
    }
    if let Some(cuda_root) = cuda_root {
        candidates.push(cuda_root.join("lib64"));
        candidates.push(cuda_root.join("lib"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/lib64"));
    candidates.extend(system_cuda_lib_dirs(target));
    candidates
        .into_iter()
        .find(|candidate| candidate.join("libcudart.so").is_file())
}

pub(crate) fn find_cuda_stub_lib_dir(
    cuda_root: Option<&PathBuf>,
    cuda_target_dir: Option<&str>,
) -> Option<PathBuf> {
    let cuda_root = cuda_root?;
    let mut candidates = Vec::new();
    if let Some(cuda_target_dir) = cuda_target_dir {
        let target_root = cuda_root.join("targets").join(cuda_target_dir);
        candidates.push(target_root.join("lib/stubs"));
        candidates.push(target_root.join("lib64/stubs"));
    }
    candidates.push(cuda_root.join("lib64/stubs"));
    candidates.push(cuda_root.join("lib/stubs"));
    candidates
        .into_iter()
        .find(|candidate| candidate.join("libcuda.so").is_file())
}

pub(crate) fn find_cuda_include_dir(
    cuda_root: Option<&PathBuf>,
    cuda_target_dir: Option<&str>,
) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let (Some(cuda_root), Some(cuda_target_dir)) = (cuda_root, cuda_target_dir) {
        candidates.push(
            cuda_root
                .join("targets")
                .join(cuda_target_dir)
                .join("include"),
        );
    }
    if let Some(cuda_root) = cuda_root {
        candidates.push(cuda_root.join("include"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/include"));
    candidates.push(PathBuf::from("/usr/include"));
    candidates
        .into_iter()
        .find(|candidate| candidate.join("nvrtc.h").is_file())
}

pub(crate) fn find_host_cxx_for_target(target: &str) -> Option<PathBuf> {
    let target_key = target_env_key(target);
    for var in [
        format!("CXX_{target_key}"),
        format!("CARGO_TARGET_{target_key}_CXX"),
        "CXX".to_string(),
    ] {
        if let Ok(value) = env::var(var) {
            let path = PathBuf::from(value);
            if command_exists(&path) {
                return Some(path);
            }
        }
    }
    let candidates = if target.starts_with("aarch64-") {
        vec!["aarch64-linux-gnu-g++"]
    } else if target.starts_with("x86_64-") {
        vec!["g++", "c++"]
    } else {
        Vec::new()
    };
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| command_exists(candidate))
}

pub(crate) fn find_ar_for_target(target: &str) -> String {
    let target_key = target_env_key(target);
    for var in [
        format!("AR_{target_key}"),
        format!("CARGO_TARGET_{target_key}_AR"),
        "AR".to_string(),
    ] {
        if let Ok(value) = env::var(var) {
            if command_exists(&PathBuf::from(&value)) {
                return value;
            }
        }
    }
    if target.starts_with("aarch64-") && command_exists(&PathBuf::from("aarch64-linux-gnu-ar")) {
        return "aarch64-linux-gnu-ar".to_string();
    }
    "ar".to_string()
}

pub(crate) fn target_env_key(target: &str) -> String {
    target.replace('-', "_").to_ascii_uppercase()
}

fn system_cuda_lib_dirs(target: &str) -> Vec<PathBuf> {
    if target.starts_with("aarch64-") {
        vec![PathBuf::from("/usr/lib/aarch64-linux-gnu")]
    } else if target.starts_with("x86_64-") {
        vec![PathBuf::from("/usr/lib/x86_64-linux-gnu")]
    } else {
        Vec::new()
    }
}

fn command_exists(command: &PathBuf) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}
