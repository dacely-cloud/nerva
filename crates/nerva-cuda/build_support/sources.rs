use std::path::{Path, PathBuf};

pub(crate) struct NativeCudaSources {
    pub(crate) native_dir: PathBuf,
    pub(crate) cuda_sources: Vec<PathBuf>,
    header: PathBuf,
}

impl NativeCudaSources {
    pub(crate) fn new(manifest_dir: &Path) -> Self {
        let native_dir = manifest_dir.join("../../native/cuda");
        let cuda_sources = [
            "nerva_cuda_device_smoke.cu",
            "nerva_cuda_backend_contract.cu",
            "nerva_cuda_synthetic_graph.cu",
            "nerva_cuda_tiny_block.cu",
            "nerva_cuda_block_forward.cu",
            "nerva_cuda_greedy_sampler.cu",
            "nerva_cuda_hf_sampler.cu",
            "nerva_cuda_hf_decode_step.cu",
            "nerva_cuda_hf_decode_chain.cu",
            "nerva_cuda_hf_decode_sequence.cu",
            "nerva_cuda_tiny_decode.cu",
            "nerva_cuda_tiered_attention.cu",
            "nerva_cuda_projection_bench.cu",
            "nerva_cuda_experimental_rt.cu",
        ]
        .into_iter()
        .map(|source| native_dir.join(source))
        .collect::<Vec<_>>();
        let header = native_dir.join("nerva_cuda_api.h");
        Self {
            native_dir,
            cuda_sources,
            header,
        }
    }

    pub(crate) fn print_rerun_directives(&self) {
        for cuda_source in &self.cuda_sources {
            println!("cargo:rerun-if-changed={}", cuda_source.display());
        }
        println!("cargo:rerun-if-changed={}", self.header.display());
    }
}

pub(crate) fn print_env_rerun_directives() {
    for var in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDACXX",
        "NERVA_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "CMAKE_CUDA_ARCHITECTURES",
        "NERVA_CUDA_ARCH",
        "CUDA_ARCH",
        "NERVA_OPTIX_SDK_ROOT",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

pub(crate) fn print_build_support_rerun_directives(manifest_dir: &Path) {
    for source in [
        "build_support/mod.rs",
        "build_support/arch.rs",
        "build_support/command.rs",
        "build_support/paths.rs",
        "build_support/sources.rs",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join(source).display()
        );
    }
}
