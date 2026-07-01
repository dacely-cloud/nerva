mod build_support;

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(nerva_cuda_native_stubs)");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sources = build_support::sources::NativeCudaSources::new(&manifest_dir);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archive = out_dir.join("libnerva_cuda_api.a");
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_else(|_| host.clone());

    sources.print_rerun_directives();
    build_support::sources::print_env_rerun_directives();
    build_support::sources::print_build_support_rerun_directives(&manifest_dir);
    let target_env_key = build_support::paths::target_env_key(&target);
    println!("cargo:rerun-if-env-changed=CXX_{target_env_key}");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_{target_env_key}_CXX");
    println!("cargo:rerun-if-env-changed=AR_{target_env_key}");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_{target_env_key}_AR");

    let cuda_root = build_support::paths::find_cuda_root();
    let nvcc = build_support::paths::find_nvcc(cuda_root.as_ref());
    let cuda_link_target_dir =
        build_support::paths::find_cuda_target_dir(cuda_root.as_ref(), &target);
    let cuda_nvcc_target_dir = (host != target)
        .then(|| cuda_link_target_dir.as_deref())
        .flatten();

    if should_use_native_cuda(&host, &target, nvcc.as_ref(), cuda_nvcc_target_dir) {
        build_native_cuda(
            &manifest_dir,
            &sources,
            &out_dir,
            &archive,
            cuda_root.as_ref(),
            nvcc.as_ref().expect("native CUDA build requires nvcc"),
            cuda_link_target_dir.as_deref(),
            cuda_nvcc_target_dir,
            &target,
        );
    } else {
        println!("cargo:rustc-cfg=nerva_cuda_native_stubs");
        println!(
            "cargo:warning=nerva-cuda using Rust native stubs for target {target}; install a target CUDA toolkit or set NERVA_CUDA_BUILD=native for real CUDA"
        );
    }
}

fn build_native_cuda(
    manifest_dir: &PathBuf,
    sources: &build_support::sources::NativeCudaSources,
    out_dir: &PathBuf,
    archive: &PathBuf,
    cuda_root: Option<&PathBuf>,
    nvcc: &PathBuf,
    cuda_link_target_dir: Option<&str>,
    cuda_nvcc_target_dir: Option<&str>,
    target: &str,
) {
    let cuda_arches = build_support::arch::cuda_architectures(&nvcc);
    println!("cargo:warning=nerva-cuda using nvcc at {}", nvcc.display());
    if let Some(cuda_target_dir) = cuda_link_target_dir {
        println!("cargo:warning=nerva-cuda CUDA target directory: {cuda_target_dir}");
    }
    println!(
        "cargo:warning=nerva-cuda CUDA architectures: {}",
        cuda_arches.join(";")
    );

    let cudnn = cudnn_frontend_paths(target);
    let cuda_include = build_support::paths::find_cuda_include_dir(cuda_root, cuda_link_target_dir);
    let optix_include = optix_include_dir(&manifest_dir);
    if let Some(optix_include) = optix_include.as_ref() {
        println!(
            "cargo:warning=nerva-cuda using OptiX headers at {}",
            optix_include.display()
        );
        println!("cargo:rerun-if-changed={}", optix_include.display());
    }
    let host_cxx =
        cuda_nvcc_target_dir.and_then(|_| build_support::paths::find_host_cxx_for_target(target));
    if cuda_nvcc_target_dir.is_some() && host_cxx.is_none() {
        panic!(
            "nerva-cuda target {target} requires a CUDA host C++ compiler such as aarch64-linux-gnu-g++; set CXX_{} if it is installed elsewhere",
            build_support::paths::target_env_key(target)
        );
    }
    print_link_directives(
        out_dir,
        cuda_root,
        cuda_link_target_dir,
        target,
        cudnn.as_ref(),
    );
    let cuda_objects = compile_cuda_sources(
        sources,
        out_dir,
        nvcc,
        cuda_arches.as_slice(),
        cudnn.as_ref(),
        cuda_include.as_ref(),
        optix_include.as_ref(),
        cuda_nvcc_target_dir,
        host_cxx.as_ref(),
    );
    archive_cuda_objects(archive, cuda_objects.as_slice(), target);
}

fn should_use_native_cuda(
    host: &str,
    target: &str,
    nvcc: Option<&PathBuf>,
    cuda_target_dir: Option<&str>,
) -> bool {
    match env::var("NERVA_CUDA_BUILD")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "stub" | "stubs" | "off" | "none" => return false,
        "native" | "cuda" | "on" => {
            if nvcc.is_none() {
                panic!(
                    "NERVA_CUDA_BUILD=native requires nvcc; set CUDA_HOME/CUDA_PATH/CUDACXX or install CUDA"
                );
            }
            if host != target && cuda_target_dir.is_none() {
                let candidates =
                    build_support::paths::cuda_target_dir_candidates(target).join(", ");
                panic!(
                    "NERVA_CUDA_BUILD=native for target {target} requires CUDA target directory {candidates}"
                );
            }
            return true;
        }
        "auto" | "" => {}
        value => panic!("unsupported NERVA_CUDA_BUILD={value}; use auto, native, or stubs"),
    }

    if nvcc.is_none() {
        return false;
    }
    host == target || cuda_target_dir.is_some()
}

struct CudnnFrontendPaths {
    frontend_include: PathBuf,
    cudnn_include: PathBuf,
    cudnn_lib: PathBuf,
}

fn cudnn_frontend_paths(target: &str) -> Option<CudnnFrontendPaths> {
    let mut roots = Vec::new();
    if let Ok(root) = env::var("NERVA_CUDNN_FRONTEND_ROOT") {
        roots.push(PathBuf::from(root));
    }
    if target.starts_with("x86_64-") {
        roots.push(PathBuf::from(
            "/root/vllm/.venv/lib/python3.12/site-packages",
        ));
    }
    for root in roots {
        let frontend_include = root.join("include");
        let cudnn_include = root.join("nvidia/cudnn/include");
        let cudnn_lib = root.join("nvidia/cudnn/lib");
        if frontend_include.join("cudnn_frontend.h").is_file()
            && cudnn_include.join("cudnn.h").is_file()
            && cudnn_lib.join("libcudnn.so.9").is_file()
        {
            return Some(CudnnFrontendPaths {
                frontend_include,
                cudnn_include,
                cudnn_lib,
            });
        }
    }
    None
}

fn optix_include_dir(manifest_dir: &PathBuf) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(root) = env::var("NERVA_OPTIX_SDK_ROOT") {
        let root = PathBuf::from(root);
        candidates.push(root.join("include"));
        candidates.push(root);
    }
    candidates.push(manifest_dir.join("../../third_party/optix-sdk/include"));
    candidates.push(PathBuf::from("/usr/local/optix-sdk/include"));
    candidates.push(PathBuf::from("/opt/nvidia/optix-sdk/include"));
    candidates
        .into_iter()
        .find(|candidate| candidate.join("optix.h").is_file())
}

fn print_link_directives(
    out_dir: &PathBuf,
    cuda_root: Option<&PathBuf>,
    cuda_target_dir: Option<&str>,
    target: &str,
    cudnn: Option<&CudnnFrontendPaths>,
) {
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nerva_cuda_api");
    if let Some(cuda_lib_dir) =
        build_support::paths::find_cuda_lib_dir(cuda_root, cuda_target_dir, target)
    {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        if target.contains("linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib_dir.display());
        }
    }
    if let Some(cuda_stub_lib_dir) =
        build_support::paths::find_cuda_stub_lib_dir(cuda_root, cuda_target_dir)
    {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_stub_lib_dir.display()
        );
    }
    if let Some(cudnn) = cudnn {
        let link_name = out_dir.join("libcudnn.so");
        let real_name = cudnn.cudnn_lib.join("libcudnn.so.9");
        if !link_name.exists() && real_name.is_file() {
            #[cfg(unix)]
            {
                let _ = std::os::unix::fs::symlink(&real_name, &link_name);
            }
        }
        println!(
            "cargo:rustc-link-search=native={}",
            cudnn.cudnn_lib.display()
        );
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        if target.contains("linux") {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                cudnn.cudnn_lib.display()
            );
        }
        println!("cargo:rustc-link-lib=dylib=cudnn");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn compile_cuda_sources(
    sources: &build_support::sources::NativeCudaSources,
    out_dir: &PathBuf,
    nvcc: &PathBuf,
    cuda_arches: &[String],
    cudnn: Option<&CudnnFrontendPaths>,
    cuda_include: Option<&PathBuf>,
    optix_include: Option<&PathBuf>,
    cuda_target_dir: Option<&str>,
    host_cxx: Option<&PathBuf>,
) -> Vec<PathBuf> {
    let mut cuda_objects = Vec::with_capacity(sources.cuda_sources.len());
    for cuda_source in &sources.cuda_sources {
        let relative_source = cuda_source
            .strip_prefix(&sources.native_dir)
            .unwrap_or(cuda_source);
        let object_stem = relative_source
            .with_extension("")
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("__");
        let object_name = format!("{object_stem}.o");
        let cuda_object = out_dir.join(object_name);
        let mut command = Command::new(nvcc);
        command.arg("-std=c++17").arg("-Xcompiler").arg("-fPIC");
        if let Some(cuda_target_dir) = cuda_target_dir {
            command.arg("-target-dir").arg(cuda_target_dir);
        }
        if let Some(host_cxx) = host_cxx {
            command.arg("-ccbin").arg(host_cxx);
        }
        build_support::arch::add_cuda_arch_flags(&mut command, cuda_arches);
        if let Some(cudnn) = cudnn {
            command
                .arg("-D")
                .arg("NERVA_HAVE_CUDNN_FRONTEND=1")
                .arg("-I")
                .arg(&cudnn.frontend_include)
                .arg("-I")
                .arg(&cudnn.cudnn_include);
        }
        if let Some(optix_include) = optix_include {
            command.arg("-I").arg(optix_include);
            command.arg(format!(
                "-DNERVA_OPTIX_INCLUDE_DIR=\"{}\"",
                optix_include.display()
            ));
        }
        if let Some(cuda_include) = cuda_include {
            command.arg(format!(
                "-DNERVA_CUDA_INCLUDE_DIR=\"{}\"",
                cuda_include.display()
            ));
        }
        command
            .arg("-I")
            .arg(&sources.native_dir)
            .arg("-c")
            .arg(cuda_source)
            .arg("-o")
            .arg(&cuda_object);
        build_support::command::run(&mut command, &format!("compile {}", cuda_source.display()));
        cuda_objects.push(cuda_object);
    }
    cuda_objects
}

fn archive_cuda_objects(archive: &PathBuf, cuda_objects: &[PathBuf], target: &str) {
    let _ = std::fs::remove_file(archive);
    let ar = build_support::paths::find_ar_for_target(target);
    let mut ar_command = Command::new(&ar);
    ar_command.arg("crs").arg(archive);
    for cuda_object in cuda_objects {
        ar_command.arg(cuda_object);
    }
    build_support::command::run(&mut ar_command, &format!("archive {}", archive.display()));
}
