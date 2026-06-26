mod build_support;

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sources = build_support::sources::NativeCudaSources::new(&manifest_dir);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archive = out_dir.join("libnerva_cuda_api.a");

    sources.print_rerun_directives();
    build_support::sources::print_env_rerun_directives();
    build_support::sources::print_build_support_rerun_directives(&manifest_dir);

    let cuda_root = build_support::paths::find_cuda_root();
    let nvcc = build_support::paths::find_nvcc(cuda_root.as_ref()).unwrap_or_else(|| {
        panic!(
            "nerva-cuda requires nvcc to build the native CUDA runtime smoke; set CUDA_HOME/CUDA_PATH/CUDACXX or install CUDA"
        )
    });
    let cuda_arches = build_support::arch::cuda_architectures(&nvcc);
    println!("cargo:warning=nerva-cuda using nvcc at {}", nvcc.display());
    println!(
        "cargo:warning=nerva-cuda CUDA architectures: {}",
        cuda_arches.join(";")
    );

    print_link_directives(&out_dir, cuda_root.as_ref());
    let cuda_objects = compile_cuda_sources(&sources, &out_dir, &nvcc, cuda_arches.as_slice());
    archive_cuda_objects(&archive, cuda_objects.as_slice());
}

fn print_link_directives(out_dir: &PathBuf, cuda_root: Option<&PathBuf>) {
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nerva_cuda_api");
    if let Some(cuda_lib_dir) = build_support::paths::find_cuda_lib_dir(cuda_root) {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib_dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn compile_cuda_sources(
    sources: &build_support::sources::NativeCudaSources,
    out_dir: &PathBuf,
    nvcc: &PathBuf,
    cuda_arches: &[String],
) -> Vec<PathBuf> {
    let mut cuda_objects = Vec::with_capacity(sources.cuda_sources.len());
    for cuda_source in &sources.cuda_sources {
        let object_name = cuda_source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(|stem| format!("{stem}.o"))
            .expect("CUDA source must have a valid file stem");
        let cuda_object = out_dir.join(object_name);
        let mut command = Command::new(nvcc);
        command.arg("-std=c++17").arg("-Xcompiler").arg("-fPIC");
        build_support::arch::add_cuda_arch_flags(&mut command, cuda_arches);
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

fn archive_cuda_objects(archive: &PathBuf, cuda_objects: &[PathBuf]) {
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let mut ar_command = Command::new(&ar);
    ar_command.arg("crs").arg(archive);
    for cuda_object in cuda_objects {
        ar_command.arg(cuda_object);
    }
    build_support::command::run(&mut ar_command, &format!("archive {}", archive.display()));
}
