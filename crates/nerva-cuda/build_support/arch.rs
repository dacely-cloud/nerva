use std::env;
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn add_cuda_arch_flags(command: &mut Command, arches: &[String]) {
    for arch in arches {
        command
            .arg("-gencode")
            .arg(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if let Some(ptx_arch) = highest_arch(arches) {
        command
            .arg("-gencode")
            .arg(format!("arch=compute_{ptx_arch},code=compute_{ptx_arch}"));
    }
}

pub(crate) fn cuda_architectures(nvcc: &PathBuf) -> Vec<String> {
    for var in [
        "NERVA_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "CMAKE_CUDA_ARCHITECTURES",
        "NERVA_CUDA_ARCH",
        "CUDA_ARCH",
    ] {
        if let Ok(raw) = env::var(var) {
            let arches = parse_arch_list(&raw);
            if !arches.is_empty() {
                return arches;
            }
        }
    }

    if let Some(arch) = detect_gpu_arch() {
        return vec![arch];
    }

    let supported = nvcc_supported_architectures(nvcc);
    if !supported.is_empty() {
        return supported;
    }

    default_architectures_for_nvcc(nvcc)
}

fn parse_arch_list(raw: &str) -> Vec<String> {
    let mut arches = Vec::new();
    for token in raw.split(|ch: char| ch == ';' || ch == ',' || ch.is_whitespace()) {
        if let Some(arch) = normalize_arch_token(token) {
            if !arches.contains(&arch) {
                arches.push(arch);
            }
        }
    }
    arches
}

fn normalize_arch_token(token: &str) -> Option<String> {
    let mut arch = token
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_ascii_lowercase();
    for prefix in ["sm_", "compute_"] {
        if let Some(stripped) = arch.strip_prefix(prefix) {
            arch = stripped.to_string();
        }
    }
    for suffix in ["-real", "-virtual"] {
        if let Some(stripped) = arch.strip_suffix(suffix) {
            arch = stripped.to_string();
        }
    }
    if arch.contains('.') && arch.chars().all(|ch| ch.is_ascii_digit() || ch == '.') {
        arch.retain(|ch| ch != '.');
    }
    if arch.is_empty() || !arch.chars().all(|ch| ch.is_ascii_alphanumeric()) {
        return None;
    }
    Some(arch)
}

fn detect_gpu_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(normalize_arch_token)
}

fn nvcc_supported_architectures(nvcc: &PathBuf) -> Vec<String> {
    let output = Command::new(nvcc).arg("--list-gpu-code").output().ok();
    let Some(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_arch_list(&stdout)
}

fn default_architectures_for_nvcc(nvcc: &PathBuf) -> Vec<String> {
    let version = nvcc_version(nvcc);
    let raw = match version {
        Some((major, _)) if major < 11 => "50;52;60;61;70;75",
        Some((11, 0)) => "52;60;61;70;75;80",
        Some((11, minor)) if minor < 8 => "52;60;61;70;75;80;86",
        Some((11, _)) => "52;60;61;70;75;80;86;89;90",
        Some((12, minor)) if minor < 8 => "60;61;70;75;80;86;89;90",
        Some((12, _)) => "60;61;70;75;80;86;89;90;120",
        Some((13, _)) => "75;80;86;87;88;89;90;100;103;110;120;121",
        _ => "75;80;86;89;90;120",
    };
    parse_arch_list(raw)
}

fn nvcc_version(nvcc: &PathBuf) -> Option<(u32, u32)> {
    let output = Command::new(nvcc).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let release = text.split("release ").nth(1)?;
    let version = release.split([',', ' ']).next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

fn highest_arch(arches: &[String]) -> Option<String> {
    arches
        .iter()
        .max_by_key(|arch| arch_numeric_key(arch))
        .cloned()
}

fn arch_numeric_key(arch: &str) -> u32 {
    arch.chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}
