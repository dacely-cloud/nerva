use std::path::{Path, PathBuf};

pub(crate) struct NativeCudaSources {
    pub(crate) native_dir: PathBuf,
    pub(crate) cuda_sources: Vec<PathBuf>,
    header: PathBuf,
    headers: Vec<PathBuf>,
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
            "hf_decode_sequence/attention_kernels.cu",
            "hf_decode_sequence/kernels.cu",
            "hf_decode_sequence/prefill_kernels.cu",
            "hf_decode_sequence/projection.cu",
            "hf_decode_sequence/sampler.cu",
            "hf_decode_sequence/weights.cu",
            "nerva_cuda_hf_decode_sequence.cu",
            "nerva_cuda_tiny_decode.cu",
            "nerva_cuda_tiered_attention.cu",
            "nerva_cuda_projection_bench.cu",
            "nerva_cuda_deepseek_quant.cu",
            "nerva_cuda_deepseek_router.cu",
            "nerva_cuda_deepseek_mla.cu",
            "nerva_cuda_deepseek_mhc.cu",
            "nerva_cuda_deepseek_mhc_fused.cu",
            "nerva_cuda_deepseek_kv.cu",
            "nerva_cuda_deepseek_moe.cu",
            "nerva_cuda_experimental_rt.cu",
        ]
        .into_iter()
        .map(|source| native_dir.join(source))
        .collect::<Vec<_>>();
        let header = native_dir.join("nerva_cuda_api.h");
        let headers = vec![
            native_dir.join("deepseek_quant.cuh"),
            native_dir.join("deepseek_router.cuh"),
            native_dir.join("hf_decode_sequence/device_ops.cuh"),
            native_dir.join("hf_decode_sequence/deepseek/kernels.cuh"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_helpers.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_prelude.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_residual_norm.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_sparse_moe.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_v3_mla.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_v32_indexer.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_v4.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_v4_final_norm.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_v4_mhc.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/kernels_v4_swa_dense.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/projection.cuh"),
            native_dir.join("hf_decode_sequence/deepseek/projection.inc.cu"),
            native_dir.join("hf_decode_sequence/deepseek/session_decode.inc.cu"),
            native_dir.join("hf_decode_sequence/kernels.cuh"),
            native_dir.join("hf_decode_sequence/projection.cuh"),
            native_dir.join("hf_decode_sequence/sampler.cuh"),
            native_dir.join("hf_decode_sequence/session_api_batch_fork_destroy.inc.cu"),
            native_dir.join("hf_decode_sequence/session_api_layer_projection_batch.inc.cu"),
            native_dir.join("hf_decode_sequence/session_api_lifecycle.inc.cu"),
            native_dir.join("hf_decode_sequence/session_api_oneshot.inc.cu"),
            native_dir.join("hf_decode_sequence/session_api_projection_batch.inc.cu"),
            native_dir.join("hf_decode_sequence/session_common.inc.cu"),
            native_dir.join("hf_decode_sequence/session_cudnn.inc.cu"),
            native_dir.join("hf_decode_sequence/session_decode_prefill.inc.cu"),
            native_dir.join("hf_decode_sequence/session_graph_result.inc.cu"),
            native_dir.join("hf_decode_sequence/session_prelude.inc.cu"),
            native_dir.join("hf_decode_sequence/session_profile.inc.cu"),
            native_dir.join("hf_decode_sequence/session_state.cuh"),
            native_dir.join("hf_decode_sequence/types.cuh"),
            native_dir.join("hf_decode_sequence/weights.cuh"),
        ];
        Self {
            native_dir,
            cuda_sources,
            header,
            headers,
        }
    }

    pub(crate) fn print_rerun_directives(&self) {
        for cuda_source in &self.cuda_sources {
            println!("cargo:rerun-if-changed={}", cuda_source.display());
        }
        println!("cargo:rerun-if-changed={}", self.header.display());
        for header in &self.headers {
            println!("cargo:rerun-if-changed={}", header.display());
        }
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
