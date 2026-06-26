use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = manifest_dir.join("../../native/cuda");
    let source = native_dir.join("nerva_cuda_api.cpp");
    let header = native_dir.join("nerva_cuda_api.h");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let object = out_dir.join("nerva_cuda_api.o");
    let archive = out_dir.join("libnerva_cuda_api.a");

    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nerva_cuda_api");

    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    run(
        Command::new(&cxx)
            .arg("-std=c++17")
            .arg("-fPIC")
            .arg("-I")
            .arg(&native_dir)
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object),
        &format!("compile {}", source.display()),
    );

    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    run(
        Command::new(&ar).arg("crs").arg(&archive).arg(&object),
        &format!("archive {}", archive.display()),
    );
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
