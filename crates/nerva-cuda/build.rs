use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = manifest_dir.join("../../native/cuda");
    let cuda_sources = [
        native_dir.join("nerva_cuda_device_smoke.cu"),
        native_dir.join("nerva_cuda_synthetic_graph.cu"),
        native_dir.join("nerva_cuda_tiny_block.cu"),
        native_dir.join("nerva_cuda_greedy_sampler.cu"),
    ];
    let header = native_dir.join("nerva_cuda_api.h");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archive = out_dir.join("libnerva_cuda_api.a");

    for cuda_source in &cuda_sources {
        println!("cargo:rerun-if-changed={}", cuda_source.display());
    }
    println!("cargo:rerun-if-changed={}", header.display());
    for var in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDACXX",
        "NERVA_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "CMAKE_CUDA_ARCHITECTURES",
        "NERVA_CUDA_ARCH",
        "CUDA_ARCH",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let cuda_root = find_cuda_root();
    let nvcc = find_nvcc(cuda_root.as_ref()).unwrap_or_else(|| {
        panic!(
            "nerva-cuda requires nvcc to build the native CUDA runtime smoke; set CUDA_HOME/CUDA_PATH/CUDACXX or install CUDA"
        )
    });
    let cuda_arches = cuda_architectures(&nvcc);
    println!("cargo:warning=nerva-cuda using nvcc at {}", nvcc.display());
    println!(
        "cargo:warning=nerva-cuda CUDA architectures: {}",
        cuda_arches.join(";")
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nerva_cuda_api");
    if let Some(cuda_lib_dir) = find_cuda_lib_dir(cuda_root.as_ref()) {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib_dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    let mut cuda_objects = Vec::with_capacity(cuda_sources.len());
    for cuda_source in &cuda_sources {
        let object_name = cuda_source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(|stem| format!("{stem}.o"))
            .expect("CUDA source must have a valid file stem");
        let cuda_object = out_dir.join(object_name);
        let mut command = Command::new(&nvcc);
        command.arg("-std=c++17").arg("-Xcompiler").arg("-fPIC");
        add_cuda_arch_flags(&mut command, &cuda_arches);
        command
            .arg("-I")
            .arg(&native_dir)
            .arg("-c")
            .arg(cuda_source)
            .arg("-o")
            .arg(&cuda_object);
        run(&mut command, &format!("compile {}", cuda_source.display()));
        cuda_objects.push(cuda_object);
    }

    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let mut ar_command = Command::new(&ar);
    ar_command.arg("crs").arg(&archive);
    for cuda_object in &cuda_objects {
        ar_command.arg(cuda_object);
    }
    run(&mut ar_command, &format!("archive {}", archive.display()));
}

fn add_cuda_arch_flags(command: &mut Command, arches: &[String]) {
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

fn run(command: &mut Command, label: &str) {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("{label} failed to start: {err}"));
    if !output.status.success() {
        panic!(
            "{label} failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

fn find_cuda_root() -> Option<PathBuf> {
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

fn find_nvcc(cuda_root: Option<&PathBuf>) -> Option<PathBuf> {
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

fn find_cuda_lib_dir(cuda_root: Option<&PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(cuda_root) = cuda_root {
        candidates.push(cuda_root.join("lib64"));
        candidates.push(cuda_root.join("lib"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/lib64"));
    candidates.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    candidates
        .into_iter()
        .find(|candidate| candidate.join("libcudart.so").is_file() || candidate.exists())
}

fn cuda_architectures(nvcc: &PathBuf) -> Vec<String> {
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
        Some((major, minor)) if major < 11 => "50;52;60;61;70;75",
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
