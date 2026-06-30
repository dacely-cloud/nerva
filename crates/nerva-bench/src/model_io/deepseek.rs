use nerva_model::hf::architecture::HfArchitectureKind;
use nerva_model::hf::contract::validate_exact_runtime_contract;
use nerva_model::hf::deepseek::plan_deepseek_vllm_kv_cache;
use nerva_model::hf::metadata::{HfMlpLayerKind, HfModelMetadata};
use nerva_model::hf::parser::parse_hf_config_metadata;

use crate::json::{json_escape, json_string_array};

pub(crate) struct DeepSeekCudaPrimitiveReport<'a> {
    pub(crate) name: &'a str,
    pub(crate) status: &'a str,
    pub(crate) summary_json: &'a str,
}

pub(crate) fn run_deepseek_runtime_plan(config_path: Option<String>) -> Result<String, String> {
    let path =
        config_path.ok_or_else(|| "deepseek-runtime-plan requires config.json".to_string())?;
    let config =
        std::fs::read_to_string(&path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let metadata = parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    deepseek_runtime_plan_json(&metadata)
}

pub(crate) fn run_deepseek_cuda_readiness(config_path: Option<String>) -> Result<String, String> {
    let mla = nerva_cuda::deepseek_mla::probe::deepseek_mla_smoke();
    let moe = nerva_cuda::deepseek_moe::probe::deepseek_moe_smoke();
    let quant = nerva_cuda::deepseek_quant::probe::deepseek_quant_smoke();
    let router = nerva_cuda::deepseek_router::probe::deepseek_router_smoke();
    let mla_json = mla.to_json();
    let moe_json = moe.to_json();
    let quant_json = quant.to_json();
    let router_json = router.to_json();
    let primitives = [
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_mla_decode_mqa_smoke",
            status: smoke_status_label(&mla.status),
            summary_json: &mla_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_routed_moe_smoke",
            status: smoke_status_label(&moe.status),
            summary_json: &moe_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_quant_block_dequant_smoke",
            status: smoke_status_label(&quant.status),
            summary_json: &quant_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_router_smoke",
            status: smoke_status_label(&router.status),
            summary_json: &router_json,
        },
    ];
    deepseek_cuda_readiness_report_json(config_path, &primitives)
}

pub(crate) fn deepseek_cuda_readiness_report_json(
    config_path: Option<String>,
    primitives: &[DeepSeekCudaPrimitiveReport<'_>],
) -> Result<String, String> {
    let metadata = match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            if !metadata.architecture.is_deepseek() {
                return Err(format!(
                    "deepseek-cuda-readiness requires a DeepSeek architecture, got {}",
                    metadata.architecture.as_str()
                ));
            }
            Some(metadata)
        }
        None => None,
    };
    let metadata_ref = metadata.as_ref();
    let architecture = metadata_ref.map(|metadata| metadata.architecture.as_str());
    let implemented = metadata_ref.map_or_else(Vec::new, implemented_primitives);
    let required = metadata_ref.map_or_else(Vec::new, required_execution_units);
    let execution_unit_status = metadata_ref
        .map(execution_unit_coverage)
        .unwrap_or_default();
    let vllm_refs = metadata_ref.map_or_else(Vec::new, |metadata| {
        vllm_reference_units(metadata.architecture)
    });
    let vllm_kv_cache_plan = metadata_ref
        .map(|metadata| {
            let cache_dtype = match metadata.architecture {
                HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4 => "fp8_ds_mla",
                _ => "bfloat16",
            };
            plan_deepseek_vllm_kv_cache(metadata, cache_dtype).map(|plan| plan.to_json())
        })
        .transpose()
        .map_err(|err| format!("DeepSeek vLLM KV cache plan failed: {err:?}"))?;
    let passed = primitives
        .iter()
        .filter(|primitive| primitive.status == "ok")
        .count();
    let failed = primitives
        .iter()
        .filter(|primitive| primitive.status == "failed")
        .count();
    let unavailable = primitives
        .iter()
        .filter(|primitive| primitive.status == "unavailable")
        .count();
    let primitive_status = if failed > 0 {
        "failed"
    } else if unavailable > 0 {
        "unavailable"
    } else if passed == primitives.len() {
        "ok"
    } else {
        "incomplete"
    };
    let readiness_status = if primitive_status == "ok" {
        "primitive_smokes_ok"
    } else {
        "primitive_smokes_incomplete"
    };

    Ok(format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-cuda-readiness-v1\",\"architecture\":{},\"primitive_status\":\"{}\",\"primitive_smokes_passed\":{},\"primitive_smokes_total\":{},\"cuda_primitives\":{},\"implemented_primitives\":{},\"required_execution_units\":{},\"remaining_required_execution_units\":{},\"execution_unit_status\":{},\"vllm_reference_units\":{},\"vllm_kv_cache_plan\":{},\"runtime_parity_status\":\"not_verified\",\"performance_status\":\"not_benchmarked\",\"claim_allowed\":false}}",
        readiness_status,
        json_opt_architecture(architecture),
        primitive_status,
        passed,
        primitives.len(),
        cuda_primitives_json(primitives),
        json_string_array(&implemented),
        json_string_array(&required),
        json_string_array(&required),
        execution_unit_coverage_json(&execution_unit_status),
        json_string_array(&vllm_refs),
        vllm_kv_cache_plan.unwrap_or_else(|| "null".to_string()),
    ))
}

pub(crate) fn deepseek_runtime_plan_json(metadata: &HfModelMetadata) -> Result<String, String> {
    if !metadata.architecture.is_deepseek() {
        return Err(format!(
            "deepseek-runtime-plan requires a DeepSeek architecture, got {}",
            metadata.architecture.as_str()
        ));
    }

    let runtime_contract = match validate_exact_runtime_contract(metadata) {
        Ok(()) => RuntimeContractReport {
            status: "supported",
            reason: "exact runtime contract accepts this DeepSeek config".to_string(),
        },
        Err(err) => RuntimeContractReport {
            status: "unsupported",
            reason: format!("{err:?}"),
        },
    };
    let implemented = implemented_primitives(metadata);
    let units = required_execution_units(metadata);
    let execution_unit_status = execution_unit_coverage(metadata);
    let vllm_refs = vllm_reference_units(metadata.architecture);
    let layer_report = layer_report(metadata);
    let claim_allowed = runtime_contract.status == "supported";

    Ok(format!(
        "{{\"status\":\"ok\",\"schema\":\"nerva-deepseek-runtime-plan-v1\",\"architecture\":\"{}\",\"layers\":{},\"hidden_size\":{},\"heads\":{},\"head_dim\":{},\"moe_layers\":{},\"dense_mlp_layers\":{},\"mla_layers\":{},\"v4_swa_layers\":{},\"v4_c4_layers\":{},\"v4_c128_layers\":{},\"v4_indexer_layers\":{},\"v4_hash_router_layers\":{},\"runtime_status\":\"{}\",\"runtime_reason\":\"{}\",\"implemented_primitives\":{},\"required_execution_units\":{},\"execution_unit_status\":{},\"vllm_reference_units\":{},\"claim_allowed\":{}}}",
        metadata.architecture.as_str(),
        metadata.num_hidden_layers,
        metadata.hidden_size,
        metadata.num_attention_heads,
        metadata.head_dim(),
        layer_report.moe_layers,
        layer_report.dense_mlp_layers,
        metadata.num_hidden_layers,
        layer_report.v4_swa_layers,
        layer_report.v4_c4_layers,
        layer_report.v4_c128_layers,
        layer_report.v4_indexer_layers,
        layer_report.v4_hash_router_layers,
        runtime_contract.status,
        json_escape(&runtime_contract.reason),
        json_string_array(&implemented),
        json_string_array(&units),
        execution_unit_coverage_json(&execution_unit_status),
        json_string_array(&vllm_refs),
        claim_allowed,
    ))
}

struct RuntimeContractReport {
    status: &'static str,
    reason: String,
}

struct DeepSeekLayerReport {
    moe_layers: usize,
    dense_mlp_layers: usize,
    v4_swa_layers: usize,
    v4_c4_layers: usize,
    v4_c128_layers: usize,
    v4_indexer_layers: usize,
    v4_hash_router_layers: usize,
}

fn layer_report(metadata: &HfModelMetadata) -> DeepSeekLayerReport {
    let moe_layers = metadata
        .mlp_layer_types
        .iter()
        .filter(|kind| **kind == HfMlpLayerKind::SparseMoe)
        .count();
    let dense_mlp_layers = metadata.mlp_layer_types.len().saturating_sub(moe_layers);
    let mut v4_swa_layers = 0usize;
    let mut v4_c4_layers = 0usize;
    let mut v4_c128_layers = 0usize;
    let mut v4_indexer_layers = 0usize;
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        for layer in 0..metadata.num_hidden_layers {
            match metadata.compress_ratios.get(layer).copied().unwrap_or(0) {
                4 => {
                    v4_c4_layers += 1;
                    v4_indexer_layers += 1;
                }
                128 => v4_c128_layers += 1,
                _ => v4_swa_layers += 1,
            }
        }
    }
    DeepSeekLayerReport {
        moe_layers,
        dense_mlp_layers,
        v4_swa_layers,
        v4_c4_layers,
        v4_c128_layers,
        v4_indexer_layers,
        v4_hash_router_layers: metadata.num_hash_layers.unwrap_or(0),
    }
}

fn required_execution_units(metadata: &HfModelMetadata) -> Vec<String> {
    match metadata.architecture {
        HfArchitectureKind::DeepSeekV3 => vec![
            "deepseek_v3_mla_prefill_decode".to_string(),
            "deepseek_v3_block_fp8_projection_gemm".to_string(),
            "deepseek_v3_grouped_moe_router_noaux_tc".to_string(),
            "deepseek_v3_split_fp8_expert_moe".to_string(),
            "deepseek_v3_mtp_optional".to_string(),
        ],
        HfArchitectureKind::DeepSeekV32 => vec![
            "deepseek_v3_mla_prefill_decode".to_string(),
            "deepseek_v3_block_fp8_projection_gemm".to_string(),
            "deepseek_v32_sparse_attention_indexer".to_string(),
            "deepseek_v3_grouped_moe_router_noaux_tc".to_string(),
            "deepseek_v3_split_fp8_expert_moe".to_string(),
        ],
        HfArchitectureKind::DeepSeekV4 => vec![
            "deepseek_v4_mhc_pre_post_head".to_string(),
            "deepseek_v4_mla_swa_cache".to_string(),
            "deepseek_v4_fp8_ds_mla_cache".to_string(),
            "deepseek_v4_c4_c128_compressor".to_string(),
            "deepseek_v4_sparse_indexer".to_string(),
            "deepseek_v4_parallel_attention_gemm_streams".to_string(),
            "deepseek_v4_hash_and_bias_router".to_string(),
            "deepseek_v4_megamoe_int8_fp4_experts".to_string(),
        ],
        _ => Vec::new(),
    }
}

fn implemented_primitives(metadata: &HfModelMetadata) -> Vec<String> {
    if !metadata.architecture.is_deepseek() {
        return Vec::new();
    }

    let mut primitives = vec![
        "deepseek_weight_manifest_v3_v32_v4".to_string(),
        "fp8_e4m3fn_decode_matches_torch".to_string(),
        "e8m0_scale_upcast_matches_vllm_raw_exponent_path".to_string(),
        "fp8_e4m3fn_e8m0_block_dequant_reference".to_string(),
        "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke".to_string(),
        "deepseek_vllm_kv_cache_spec_planner".to_string(),
        "deepseek_mla_decode_mqa_reference".to_string(),
        "cuda_deepseek_mla_decode_mqa_smoke".to_string(),
        "deepseek_routed_moe_reference".to_string(),
        "cuda_deepseek_routed_moe_smoke".to_string(),
        "deepseek_v3_grouped_sigmoid_router_reference".to_string(),
        "precision_moe_deepseek_v3_grouped_router".to_string(),
        "precision_moe_deepseek_router_correction_bias_load".to_string(),
        "cuda_deepseek_router_route_api".to_string(),
        "cuda_deepseek_v3_grouped_sigmoid_router_smoke".to_string(),
    ];

    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        primitives.push("deepseek_v4_mhc_compressor_indexer_manifest".to_string());
        primitives.push("deepseek_v4_hash_router_manifest".to_string());
        primitives.push("mxfp4_e2m1_e8m0_block_dequant_reference".to_string());
        primitives.push("cuda_mxfp4_e2m1_e8m0_block_dequant_smoke".to_string());
        primitives.push("deepseek_v4_sqrtsoftplus_hash_router_reference".to_string());
        primitives.push("precision_moe_deepseek_v4_sqrtsoftplus_router".to_string());
        primitives.push("deepseek_v4_hash_route_table_i64_loader".to_string());
        primitives.push("precision_moe_deepseek_v4_hash_route_table".to_string());
        primitives.push("cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke".to_string());
    }

    primitives
}

#[derive(Clone, Debug, PartialEq)]
struct DeepSeekExecutionUnitCoverage {
    unit: String,
    status: &'static str,
    validated_primitives: Vec<String>,
    remaining_gaps: Vec<String>,
}

fn execution_unit_coverage(metadata: &HfModelMetadata) -> Vec<DeepSeekExecutionUnitCoverage> {
    required_execution_units(metadata)
        .into_iter()
        .map(|unit| coverage_for_unit(metadata.architecture, unit))
        .collect()
}

fn coverage_for_unit(
    architecture: HfArchitectureKind,
    unit: String,
) -> DeepSeekExecutionUnitCoverage {
    let (status, validated_primitives, remaining_gaps): (&'static str, &[&str], &[&str]) =
        match (architecture, unit.as_str()) {
            (
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
                "deepseek_v3_mla_prefill_decode",
            ) => (
                "partial",
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "deepseek_mla_decode_mqa_reference",
                    "cuda_deepseek_mla_decode_mqa_smoke",
                ],
                &[
                    "integrate MLA prefill/decode into exact runtime",
                    "commit vLLM-compatible MLA KV cache pages during decode",
                    "match vLLM DeepseekV2MLAAttention output numerics",
                ],
            ),
            (
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
                "deepseek_v3_block_fp8_projection_gemm",
            ) => (
                "partial",
                &[
                    "fp8_e4m3fn_decode_matches_torch",
                    "e8m0_scale_upcast_matches_vllm_raw_exponent_path",
                    "fp8_e4m3fn_e8m0_block_dequant_reference",
                    "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                ],
                &[
                    "fuse block-FP8 dequant with projection GEMM",
                    "wire DeepSeek q_a/kv_a/q_b/kv_b/o projection scales into decode",
                    "benchmark projection throughput against vLLM fused kernels",
                ],
            ),
            (HfArchitectureKind::DeepSeekV32, "deepseek_v32_sparse_attention_indexer") => (
                "partial",
                &["deepseek_vllm_kv_cache_spec_planner"],
                &[
                    "implement V3.2 sparse indexer query/key/weights runtime",
                    "store vLLM-compatible indexer cache pages",
                    "verify selected sparse blocks against vLLM",
                ],
            ),
            (
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
                "deepseek_v3_grouped_moe_router_noaux_tc",
            ) => (
                "partial",
                &[
                    "deepseek_v3_grouped_sigmoid_router_reference",
                    "precision_moe_deepseek_v3_grouped_router",
                    "precision_moe_deepseek_router_correction_bias_load",
                    "cuda_deepseek_router_route_api",
                    "cuda_deepseek_v3_grouped_sigmoid_router_smoke",
                ],
                &[
                    "integrate grouped sigmoid router into CUDA exact runtime decode layers",
                    "verify full-layer routed outputs against vLLM",
                ],
            ),
            (
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
                "deepseek_v3_split_fp8_expert_moe",
            ) => (
                "partial",
                &[
                    "deepseek_routed_moe_reference",
                    "cuda_deepseek_routed_moe_smoke",
                    "fp8_e4m3fn_e8m0_block_dequant_reference",
                    "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                ],
                &[
                    "run routed expert gate/up/down GEMMs with checkpoint FP8 scales",
                    "integrate shared experts and routed output accumulation",
                    "benchmark fused MoE against vLLM FusedMoE",
                ],
            ),
            (HfArchitectureKind::DeepSeekV3, "deepseek_v3_mtp_optional") => (
                "optional_missing",
                &[],
                &["optional MTP draft layers are not implemented"],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mhc_pre_post_head") => (
                "partial",
                &["deepseek_v4_mhc_compressor_indexer_manifest"],
                &[
                    "implement MHC pre/post-head transforms",
                    "verify MHC head/attention/FFN scale handling against vLLM",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mla_swa_cache") => (
                "partial",
                &["deepseek_vllm_kv_cache_spec_planner"],
                &[
                    "allocate and commit vLLM DeepseekV4 SWA cache pages",
                    "enforce sliding-window sparse attention semantics",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_fp8_ds_mla_cache") => (
                "partial",
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "fp8_e4m3fn_e8m0_block_dequant_reference",
                    "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                ],
                &[
                    "write 584-byte/token fp8_ds_mla pages in decode",
                    "match vLLM DeepseekV4 FlashMLA cache alignment",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_c4_c128_compressor") => (
                "partial",
                &["deepseek_v4_mhc_compressor_indexer_manifest"],
                &[
                    "implement C4/C128 compressor kernels",
                    "verify compressed-token cache selection against vLLM",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_sparse_indexer") => (
                "partial",
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "deepseek_v4_mhc_compressor_indexer_manifest",
                ],
                &[
                    "implement DeepseekV4 indexer runtime",
                    "verify C4/C128 indexer page writes and sparse block choices",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_parallel_attention_gemm_streams") => (
                "missing",
                &[],
                &[
                    "parallelize attention GEMM/compressor/indexer streams like vLLM",
                    "measure stream overlap against vLLM DeepseekV4 attention",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_hash_and_bias_router") => (
                "partial",
                &[
                    "deepseek_v4_hash_router_manifest",
                    "deepseek_v4_sqrtsoftplus_hash_router_reference",
                    "precision_moe_deepseek_v4_sqrtsoftplus_router",
                    "deepseek_v4_hash_route_table_i64_loader",
                    "precision_moe_deepseek_v4_hash_route_table",
                    "cuda_deepseek_router_route_api",
                    "cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke",
                ],
                &[
                    "integrate hash and bias routing into CUDA exact runtime decode layers",
                    "verify full-layer routed outputs against vLLM",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_megamoe_int8_fp4_experts") => (
                "partial",
                &[
                    "mxfp4_e2m1_e8m0_block_dequant_reference",
                    "cuda_mxfp4_e2m1_e8m0_block_dequant_smoke",
                    "deepseek_routed_moe_reference",
                    "cuda_deepseek_routed_moe_smoke",
                ],
                &[
                    "implement V4 MegaMoE int8/fp4 expert kernels",
                    "support expert-parallel physical/logical expert mapping",
                    "benchmark MegaMoE against vLLM deep_gemm_mega_moe/FusedMoE",
                ],
            ),
            _ => (
                "missing",
                &[],
                &["no DeepSeek coverage mapping exists for this unit"],
            ),
        };

    DeepSeekExecutionUnitCoverage {
        unit,
        status,
        validated_primitives: validated_primitives
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        remaining_gaps: remaining_gaps
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    }
}

fn execution_unit_coverage_json(units: &[DeepSeekExecutionUnitCoverage]) -> String {
    let mut out = String::from("[");
    for (index, unit) in units.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"unit\":\"{}\",\"status\":\"{}\",\"validated_primitives\":{},\"remaining_gaps\":{}}}",
            json_escape(&unit.unit),
            unit.status,
            json_string_array(&unit.validated_primitives),
            json_string_array(&unit.remaining_gaps),
        ));
    }
    out.push(']');
    out
}

fn smoke_status_label(status: &nerva_cuda::smoke::status::SmokeStatus) -> &'static str {
    match status {
        nerva_cuda::smoke::status::SmokeStatus::Ok => "ok",
        nerva_cuda::smoke::status::SmokeStatus::Unavailable => "unavailable",
        nerva_cuda::smoke::status::SmokeStatus::Failed => "failed",
    }
}

fn cuda_primitives_json(primitives: &[DeepSeekCudaPrimitiveReport<'_>]) -> String {
    let mut out = String::from("[");
    for (index, primitive) in primitives.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"status\":\"{}\",\"summary\":{}}}",
            json_escape(primitive.name),
            json_escape(primitive.status),
            primitive.summary_json,
        ));
    }
    out.push(']');
    out
}

fn json_opt_architecture(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn vllm_reference_units(architecture: HfArchitectureKind) -> Vec<String> {
    match architecture {
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32 => vec![
            "/root/vllm/vllm/model_executor/models/deepseek_v2.py".to_string(),
            "/root/vllm/vllm/v1/attention/backends/mla/indexer.py".to_string(),
            "/root/vllm/vllm/model_executor/layers/fused_moe".to_string(),
        ],
        HfArchitectureKind::DeepSeekV4 => vec![
            "/root/vllm/vllm/models/deepseek_v4/nvidia/model.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/attention.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/compressor.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/sparse_mla.py".to_string(),
            "/root/vllm/vllm/v1/kv_cache_interface.py".to_string(),
        ],
        _ => Vec::new(),
    }
}
