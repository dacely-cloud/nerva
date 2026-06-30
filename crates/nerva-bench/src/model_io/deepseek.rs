use nerva_model::hf::architecture::HfArchitectureKind;
use nerva_model::hf::contract::validate_exact_runtime_contract;
use nerva_model::hf::metadata::{HfMlpLayerKind, HfModelMetadata};
use nerva_model::hf::parser::parse_hf_config_metadata;

use crate::json::{json_escape, json_string_array};

pub(crate) fn run_deepseek_runtime_plan(config_path: Option<String>) -> Result<String, String> {
    let path =
        config_path.ok_or_else(|| "deepseek-runtime-plan requires config.json".to_string())?;
    let config =
        std::fs::read_to_string(&path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let metadata = parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    deepseek_runtime_plan_json(&metadata)
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
    let units = required_execution_units(metadata);
    let vllm_refs = vllm_reference_units(metadata.architecture);
    let layer_report = layer_report(metadata);
    let claim_allowed = runtime_contract.status == "supported";

    Ok(format!(
        "{{\"status\":\"ok\",\"schema\":\"nerva-deepseek-runtime-plan-v1\",\"architecture\":\"{}\",\"layers\":{},\"hidden_size\":{},\"heads\":{},\"head_dim\":{},\"moe_layers\":{},\"dense_mlp_layers\":{},\"mla_layers\":{},\"v4_swa_layers\":{},\"v4_c4_layers\":{},\"v4_c128_layers\":{},\"v4_indexer_layers\":{},\"v4_hash_router_layers\":{},\"runtime_status\":\"{}\",\"runtime_reason\":\"{}\",\"required_execution_units\":{},\"vllm_reference_units\":{},\"claim_allowed\":{}}}",
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
        json_string_array(&units),
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
