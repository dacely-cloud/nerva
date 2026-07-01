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

pub(crate) fn find_cuda_lib_dir(cuda_root: Option<&PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(cuda_root) = cuda_root {
        candidates.push(cuda_root.join("lib64"));
        candidates.push(cuda_root.join("lib"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/lib64"));
    candidates.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    candidates.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
    candidates
        .into_iter()
        .find(|candidate| candidate.join("libcudart.so").is_file() || candidate.exists())
}
