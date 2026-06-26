use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = manifest_dir.join("../../native/cuda");
    let cuda_source = native_dir.join("nerva_cuda_device_smoke.cu");
    let header = native_dir.join("nerva_cuda_api.h");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_object = out_dir.join("nerva_cuda_device_smoke.o");
    let archive = out_dir.join("libnerva_cuda_api.a");

    println!("cargo:rerun-if-changed={}", cuda_source.display());
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nerva_cuda_api");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/cuda/lib64");
    }

    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "nerva-cuda requires nvcc to build the native CUDA runtime smoke; set CUDA_HOME/CUDA_PATH or install CUDA"
        )
    });
    let cuda_arch = env::var("NERVA_CUDA_ARCH")
        .or_else(|_| env::var("CUDA_ARCH"))
        .unwrap_or_else(|_| "sm_120".to_string());
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-arch")
            .arg(&cuda_arch)
            .arg("-I")
            .arg(&native_dir)
            .arg("-c")
            .arg(&cuda_source)
            .arg("-o")
            .arg(&cuda_object),
        &format!("compile {}", cuda_source.display()),
    );

    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let mut ar_command = Command::new(&ar);
    ar_command.arg("crs").arg(&archive).arg(&cuda_object);
    run(&mut ar_command, &format!("archive {}", archive.display()));
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

fn find_nvcc() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cuda_home) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        candidates.push(PathBuf::from(cuda_home).join("bin/nvcc"));
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
