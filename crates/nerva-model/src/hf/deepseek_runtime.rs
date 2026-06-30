use crate::hf::architecture::HfArchitectureKind;
use crate::hf::metadata::{HfMlpLayerKind, HfModelMetadata};

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekExecutionUnitCoverage {
    pub unit: String,
    pub status: &'static str,
    pub validated_primitives: Vec<String>,
    pub remaining_gaps: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLayerReport {
    pub moe_layers: usize,
    pub dense_mlp_layers: usize,
    pub v4_swa_layers: usize,
    pub v4_c4_layers: usize,
    pub v4_c128_layers: usize,
    pub v4_indexer_layers: usize,
    pub v4_hash_router_layers: usize,
}

pub fn deepseek_layer_report(metadata: &HfModelMetadata) -> DeepSeekLayerReport {
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

pub fn deepseek_required_execution_units(metadata: &HfModelMetadata) -> Vec<String> {
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

pub fn deepseek_implemented_primitives(metadata: &HfModelMetadata) -> Vec<String> {
    if !metadata.architecture.is_deepseek() {
        return Vec::new();
    }

    let mut primitives = vec![
        "deepseek_weight_manifest_v3_v32_v4".to_string(),
        "fp8_e4m3fn_decode_matches_torch".to_string(),
        "e8m0_scale_upcast_matches_vllm_raw_exponent_path".to_string(),
        "fp8_e4m3fn_e8m0_block_dequant_reference".to_string(),
        "cuda_fp8_e4m3fn_e8m0_dequant_api".to_string(),
        "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke".to_string(),
        "deepseek_vllm_kv_cache_spec_planner".to_string(),
        "deepseek_mla_decode_mqa_reference".to_string(),
        "cuda_deepseek_mla_decode_api".to_string(),
        "cuda_deepseek_mla_decode_mqa_smoke".to_string(),
        "deepseek_routed_moe_reference".to_string(),
        "cuda_deepseek_routed_moe_api".to_string(),
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
        primitives.push("cuda_mxfp4_e2m1_e8m0_dequant_api".to_string());
        primitives.push("cuda_mxfp4_e2m1_e8m0_block_dequant_smoke".to_string());
        primitives.push("deepseek_v4_sqrtsoftplus_hash_router_reference".to_string());
        primitives.push("precision_moe_deepseek_v4_sqrtsoftplus_router".to_string());
        primitives.push("deepseek_v4_hash_route_table_i64_loader".to_string());
        primitives.push("precision_moe_deepseek_v4_hash_route_table".to_string());
        primitives.push("cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke".to_string());
        primitives.push("cuda_deepseek_qkv_rmsnorm_api".to_string());
        primitives.push("cuda_deepseek_qkv_rmsnorm_smoke".to_string());
        primitives.push("cuda_deepseek_fused_inv_rope_fp8_quant_api".to_string());
        primitives.push("cuda_deepseek_fused_inv_rope_fp8_quant_smoke".to_string());
        primitives.push("cuda_deepseek_fp8_ds_mla_kv_pack_api".to_string());
        primitives.push("cuda_deepseek_fp8_ds_mla_kv_pack_smoke".to_string());
    }
    if matches!(
        metadata.architecture,
        HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4
    ) {
        primitives.push("cuda_deepseek_compressed_slot_mapping_api".to_string());
        primitives.push("cuda_deepseek_compressed_slot_mapping_smoke".to_string());
    }
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        primitives.push("cuda_deepseek_c128_topk_metadata_api".to_string());
        primitives.push("cuda_deepseek_c128_topk_metadata_smoke".to_string());
        primitives.push("cuda_deepseek_save_partial_states_api".to_string());
        primitives.push("cuda_deepseek_save_partial_states_smoke".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_fp8_cache_api".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_fp8_cache_smoke".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_mxfp4_cache_api".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke".to_string());
    }

    primitives
}

pub fn deepseek_execution_unit_coverage(
    metadata: &HfModelMetadata,
) -> Vec<DeepSeekExecutionUnitCoverage> {
    deepseek_required_execution_units(metadata)
        .into_iter()
        .map(|unit| coverage_for_unit(metadata.architecture, unit))
        .collect()
}

pub fn deepseek_runtime_blocking_units(
    metadata: &HfModelMetadata,
) -> Vec<DeepSeekExecutionUnitCoverage> {
    deepseek_execution_unit_coverage(metadata)
        .into_iter()
        .filter(|unit| unit.status != "complete" && unit.status != "optional_missing")
        .collect()
}

pub fn deepseek_exact_runtime_blocked_reason(metadata: &HfModelMetadata) -> String {
    let blockers = deepseek_runtime_blocking_units(metadata);
    let blocker_summary = if blockers.is_empty() {
        "no blocking execution units are recorded".to_string()
    } else {
        blockers
            .iter()
            .map(|unit| {
                format!(
                    "{} is {}; remaining gaps: {}",
                    unit.unit,
                    unit.status,
                    unit.remaining_gaps.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };

    format!(
        "HF architecture {} is recognized and its DeepSeek weight/KV layout is planned, but the exact runtime has not completed DeepSeek MLA attention, block-quantized FP8/FP4 projection GEMM, compressed KV/indexer state, and DeepSeek grouped MoE routing. Blocking execution units: {blocker_summary}",
        metadata.architecture.as_str(),
    )
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
                    "cuda_deepseek_mla_decode_api",
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
                    "cuda_fp8_e4m3fn_e8m0_dequant_api",
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
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "cuda_deepseek_compressed_slot_mapping_api",
                    "cuda_deepseek_compressed_slot_mapping_smoke",
                ],
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
                    "cuda_deepseek_routed_moe_api",
                    "cuda_deepseek_routed_moe_smoke",
                    "fp8_e4m3fn_e8m0_block_dequant_reference",
                    "cuda_fp8_e4m3fn_e8m0_dequant_api",
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
                &[
                    "deepseek_v4_mhc_compressor_indexer_manifest",
                    "cuda_deepseek_qkv_rmsnorm_api",
                    "cuda_deepseek_qkv_rmsnorm_smoke",
                    "cuda_deepseek_fused_inv_rope_fp8_quant_api",
                    "cuda_deepseek_fused_inv_rope_fp8_quant_smoke",
                ],
                &[
                    "integrate fused Q/KV RMSNorm into MHC pre-head runtime",
                    "integrate fused inverse RoPE FP8 quant into O projection runtime",
                    "implement remaining MHC pre/post-head transforms",
                    "verify MHC head/attention/FFN scale handling against vLLM",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mla_swa_cache") => (
                "partial",
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "deepseek_mla_decode_mqa_reference",
                    "cuda_deepseek_mla_decode_api",
                ],
                &[
                    "allocate and commit vLLM DeepseekV4 SWA cache pages",
                    "enforce sliding-window sparse attention semantics",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_fp8_ds_mla_cache") => (
                "partial",
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "deepseek_mla_decode_mqa_reference",
                    "cuda_deepseek_mla_decode_api",
                    "fp8_e4m3fn_e8m0_block_dequant_reference",
                    "cuda_fp8_e4m3fn_e8m0_dequant_api",
                    "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                    "cuda_deepseek_fp8_ds_mla_kv_pack_api",
                    "cuda_deepseek_fp8_ds_mla_kv_pack_smoke",
                ],
                &[
                    "integrate 584-byte/token fp8_ds_mla page writes into decode",
                    "match vLLM DeepseekV4 FlashMLA cache alignment",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_c4_c128_compressor") => (
                "partial",
                &[
                    "deepseek_v4_mhc_compressor_indexer_manifest",
                    "cuda_deepseek_qkv_rmsnorm_api",
                    "cuda_deepseek_qkv_rmsnorm_smoke",
                    "cuda_deepseek_save_partial_states_api",
                    "cuda_deepseek_save_partial_states_smoke",
                    "cuda_deepseek_compress_norm_rope_fp8_cache_api",
                    "cuda_deepseek_compress_norm_rope_fp8_cache_smoke",
                    "cuda_deepseek_compress_norm_rope_mxfp4_cache_api",
                    "cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke",
                    "cuda_deepseek_compressed_slot_mapping_api",
                    "cuda_deepseek_compressed_slot_mapping_smoke",
                ],
                &[
                    "integrate C4/C128 compressor kernels into exact runtime",
                    "verify compressor cache insert inside full DeepSeekV4 attention runtime",
                ],
            ),
            (HfArchitectureKind::DeepSeekV4, "deepseek_v4_sparse_indexer") => (
                "partial",
                &[
                    "deepseek_vllm_kv_cache_spec_planner",
                    "deepseek_v4_mhc_compressor_indexer_manifest",
                    "cuda_deepseek_compressed_slot_mapping_api",
                    "cuda_deepseek_compressed_slot_mapping_smoke",
                    "cuda_deepseek_c128_topk_metadata_api",
                    "cuda_deepseek_c128_topk_metadata_smoke",
                ],
                &[
                    "implement DeepseekV4 indexer runtime",
                    "verify C4 indexer page writes and sparse block choices",
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
                    "cuda_mxfp4_e2m1_e8m0_dequant_api",
                    "cuda_mxfp4_e2m1_e8m0_block_dequant_smoke",
                    "deepseek_routed_moe_reference",
                    "cuda_deepseek_routed_moe_api",
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
