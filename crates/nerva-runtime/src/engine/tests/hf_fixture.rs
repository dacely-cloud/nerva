use std::path::{Path, PathBuf};

use nerva_core::types::dtype::DType;
use nerva_model::hf::parser::parse_hf_config_metadata;
use nerva_model::precision::bits::f32_to_f16_bits;
use nerva_model::weights::layout::entry::WeightBlockRole;
use nerva_model::weights::layout::plan::plan_hf_weight_layout;
use nerva_model::weights::manifest::build_hf_tensor_manifest;
use nerva_model::weights::manifest::{HfTensorManifest, HfTensorManifestEntry};
use nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest;

pub(crate) fn write_cycle_hf_checkpoint_dir(prefix: &str, layers: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = fixture_config(layers);
    std::fs::write(dir.join("config.json"), &config).unwrap();
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors(&dir, &manifest);
    dir
}

pub(crate) fn write_kv_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = fixture_config(1);
    std::fs::write(dir.join("config.json"), &config).unwrap();
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors_with(&dir, &manifest, values_for_kv_entry);
    dir
}

pub(crate) fn write_qwen3_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = qwen3_fixture_config();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors(&dir, &manifest);
    dir
}

pub(crate) fn write_qwen3_moe_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = qwen3_moe_fixture_config();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors_with(&dir, &manifest, values_for_qwen3_moe_entry);
    dir
}

pub(crate) fn write_qwen35_moe_full_attention_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = qwen35_moe_full_attention_fixture_config();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors_with(&dir, &manifest, values_for_qwen3_moe_entry);
    dir
}

pub(crate) fn write_qwen35_moe_linear_attention_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = qwen35_moe_linear_attention_fixture_config();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors_with(&dir, &manifest, values_for_qwen35_moe_linear_entry);
    dir
}

pub(crate) fn write_qwen2_moe_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = qwen2_moe_fixture_config();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors_with(&dir, &manifest, values_for_qwen3_moe_entry);
    dir
}

pub(crate) fn write_mixtral_moe_hf_checkpoint_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = mixtral_moe_fixture_config();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors_with(&dir, &manifest, values_for_qwen3_moe_entry);
    dir
}

pub(crate) fn remove_hf_checkpoint_dir(dir: &Path) {
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}

fn fixture_config(layers: usize) -> String {
    format!(
        r#"{{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": {layers},
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "rms_norm_eps": 0.00001,
            "torch_dtype": "float16"
        }}"#,
    )
}

fn qwen3_fixture_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 4,
        "rms_norm_eps": 0.00001,
        "rope_theta": 1000000.0,
        "torch_dtype": "float16"
    }"#
}

fn qwen3_moe_fixture_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "norm_topk_prob": true,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 4,
        "rms_norm_eps": 0.00001,
        "rope_theta": 1000000.0,
        "torch_dtype": "float16"
    }"#
}

fn qwen35_moe_full_attention_fixture_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": false,
            "dtype": "float16",
            "hidden_act": "silu",
            "hidden_size": 4,
            "intermediate_size": 8,
            "layer_types": ["full_attention"],
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 3,
            "norm_topk_prob": true,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.00001,
            "shared_expert_intermediate_size": 0,
            "tie_word_embeddings": false,
            "use_qk_norm": true,
            "vocab_size": 4
        },
        "tie_word_embeddings": false
    }"#
}

fn qwen35_moe_linear_attention_fixture_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "dtype": "float16",
            "hidden_act": "silu",
            "hidden_size": 4,
            "intermediate_size": 8,
            "layer_types": ["linear_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 3,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 3,
            "norm_topk_prob": true,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "shared_expert_intermediate_size": 0,
            "tie_word_embeddings": false,
            "vocab_size": 4
        },
        "tie_word_embeddings": false
    }"#
}

fn qwen2_moe_fixture_config() -> &'static str {
    r#"{
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 3,
        "shared_expert_intermediate_size": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "norm_topk_prob": false,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "rms_norm_eps": 0.00001,
        "rope_theta": 1000000.0,
        "torch_dtype": "float16"
    }"#
}

fn mixtral_moe_fixture_config() -> &'static str {
    r#"{
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": 4,
        "intermediate_size": 3,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 4,
        "rms_norm_eps": 0.00001,
        "rope_theta": 1000000.0,
        "torch_dtype": "float16"
    }"#
}

fn write_safetensors(dir: &Path, manifest: &HfTensorManifest) {
    write_safetensors_with(dir, manifest, values_for_entry);
}

fn write_safetensors_with(
    dir: &Path,
    manifest: &HfTensorManifest,
    values: fn(&HfTensorManifestEntry) -> Vec<u16>,
) {
    let header = synthetic_safetensors_header_for_manifest(manifest).unwrap();
    let payload = payload_for_manifest(manifest, values);
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

fn payload_for_manifest(
    manifest: &HfTensorManifest,
    values: fn(&HfTensorManifestEntry) -> Vec<u16>,
) -> Vec<u8> {
    let mut payload = Vec::new();
    for entry in &manifest.entries {
        for value in values(entry) {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    payload
}

fn values_for_kv_entry(entry: &HfTensorManifestEntry) -> Vec<u16> {
    let elements = entry.bytes / 2;
    match entry.role {
        WeightBlockRole::TokenEmbedding => {
            encode_values(&[1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        }
        WeightBlockRole::LmHead => encode_values(&[0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0]),
        WeightBlockRole::QueryProjection => encoded_identity(entry.rows, entry.cols, -1.0),
        WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection => encoded_identity(entry.rows, entry.cols, 1.0),
        WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => vec![0; elements],
        role => values_for_entry_role(role, elements),
    }
}

fn values_for_entry(entry: &HfTensorManifestEntry) -> Vec<u16> {
    if entry.dtype == DType::F32 {
        return match entry.role {
            WeightBlockRole::AttentionNorm
            | WeightBlockRole::QueryNorm
            | WeightBlockRole::KeyNorm
            | WeightBlockRole::LinearNorm
            | WeightBlockRole::MlpNorm
            | WeightBlockRole::FinalNorm => repeated_f32_slots(1.0, entry.elements),
            _ => repeated_f32_slots(0.0, entry.elements),
        };
    }
    let elements = entry.bytes / 2;
    values_for_entry_role(entry.role, elements)
}

fn values_for_qwen3_moe_entry(entry: &HfTensorManifestEntry) -> Vec<u16> {
    let elements = entry.elements;
    match entry.role {
        WeightBlockRole::TokenEmbedding => encode_values(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]),
        WeightBlockRole::LmHead => encode_values(&[
            0.7, -0.2, -0.1, 0.0, //
            -0.1, 0.8, -0.1, 0.0, //
            -0.1, -0.2, 0.9, 0.0, //
            0.0, -0.1, -0.2, 0.8,
        ]),
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::DeepSeekQALoraNorm
        | WeightBlockRole::DeepSeekKvANorm
        | WeightBlockRole::DeepSeekIndexerKeyNorm
        | WeightBlockRole::DeepSeekV4QNorm
        | WeightBlockRole::DeepSeekV4KvNorm
        | WeightBlockRole::DeepSeekV4CompressorNorm
        | WeightBlockRole::DeepSeekV4IndexerCompressorNorm
        | WeightBlockRole::LinearNorm
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::FinalNorm => vec![f32_to_f16_bits(1.0); elements],
        WeightBlockRole::RouterProjection => encode_values(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]),
        WeightBlockRole::ExpertGateProjection => qwen3_moe_expert_gate_values(entry),
        WeightBlockRole::ExpertUpProjection => qwen3_moe_expert_up_values(entry),
        WeightBlockRole::ExpertGateUpProjection => qwen3_moe_expert_gate_up_values(entry),
        WeightBlockRole::ExpertDownProjection => qwen3_moe_expert_down_values(entry),
        _ => vec![0; elements],
    }
}

fn values_for_qwen35_moe_linear_entry(entry: &HfTensorManifestEntry) -> Vec<u16> {
    let slots = entry.bytes / 2;
    match entry.role {
        WeightBlockRole::LinearNorm => repeated_f32_slots(1.0, entry.elements),
        WeightBlockRole::LinearALog => vec![0; slots],
        WeightBlockRole::LinearConvProjection
        | WeightBlockRole::LinearQkvProjection
        | WeightBlockRole::LinearZProjection
        | WeightBlockRole::LinearBProjection
        | WeightBlockRole::LinearAProjection
        | WeightBlockRole::LinearDtBias
        | WeightBlockRole::LinearOutputProjection => vec![0; slots],
        _ => qwen3_moe_values_with_slots(entry, slots),
    }
}

fn qwen3_moe_values_with_slots(entry: &HfTensorManifestEntry, slots: usize) -> Vec<u16> {
    let mut values = values_for_qwen3_moe_entry(entry);
    values.resize(slots, 0);
    values
}

fn repeated_f32_slots(value: f32, count: usize) -> Vec<u16> {
    let bits = value.to_bits();
    let slots = [(bits & 0xffff) as u16, (bits >> 16) as u16];
    (0..count).flat_map(|_| slots).collect()
}

fn qwen3_moe_expert_gate_values(entry: &HfTensorManifestEntry) -> Vec<u16> {
    qwen3_moe_split_expert_values(entry, 0, 0.8)
}

fn qwen3_moe_expert_up_values(entry: &HfTensorManifestEntry) -> Vec<u16> {
    qwen3_moe_split_expert_values(entry, entry.rows, 0.5)
}

fn qwen3_moe_split_expert_values(
    entry: &HfTensorManifestEntry,
    row_offset: usize,
    value: f32,
) -> Vec<u16> {
    let expert = entry.expert.unwrap_or(0) as usize;
    let mut values = vec![0.0f32; entry.elements];
    for row in 0..entry.rows {
        let col = (expert + row_offset + row) % entry.cols;
        values[row * entry.cols + col] = value;
    }
    encode_values(&values)
}

fn qwen3_moe_expert_gate_up_values(entry: &HfTensorManifestEntry) -> Vec<u16> {
    let depth = entry.depth.unwrap_or(1);
    let mut values = vec![0.0f32; entry.elements];
    for expert in 0..depth {
        let base = expert * entry.rows * entry.cols;
        for row in 0..entry.rows {
            let col = (expert + row) % entry.cols;
            let value = if row < entry.rows / 2 { 0.8 } else { 0.5 };
            values[base + row * entry.cols + col] = value;
        }
    }
    encode_values(&values)
}

fn qwen3_moe_expert_down_values(entry: &HfTensorManifestEntry) -> Vec<u16> {
    if let Some(expert) = entry.expert {
        let expert = expert as usize;
        let mut values = vec![0.0f32; entry.elements];
        for row in 0..entry.rows {
            let col = (row + expert) % entry.cols;
            values[row * entry.cols + col] = 0.6;
        }
        return encode_values(&values);
    }
    let depth = entry.depth.unwrap_or(1);
    let mut values = vec![0.0f32; entry.elements];
    for expert in 0..depth {
        let base = expert * entry.rows * entry.cols;
        for row in 0..entry.rows {
            let col = (row + expert) % entry.cols;
            values[base + row * entry.cols + col] = 0.6;
        }
    }
    encode_values(&values)
}

fn values_for_entry_role(role: WeightBlockRole, elements: usize) -> Vec<u16> {
    match role {
        WeightBlockRole::TokenEmbedding => {
            encode_values(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        }
        WeightBlockRole::LmHead => encode_values(&[0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::DeepSeekQALoraNorm
        | WeightBlockRole::DeepSeekKvANorm
        | WeightBlockRole::DeepSeekIndexerKeyNorm
        | WeightBlockRole::DeepSeekV4QNorm
        | WeightBlockRole::DeepSeekV4KvNorm
        | WeightBlockRole::DeepSeekV4CompressorNorm
        | WeightBlockRole::DeepSeekV4IndexerCompressorNorm
        | WeightBlockRole::LinearNorm
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::FinalNorm => vec![f32_to_f16_bits(1.0); elements],
        WeightBlockRole::QueryBias
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias
        | WeightBlockRole::DeepSeekIndexerKeyNormBias
        | WeightBlockRole::LinearConvProjection
        | WeightBlockRole::LinearQkvProjection
        | WeightBlockRole::LinearZProjection
        | WeightBlockRole::LinearBProjection
        | WeightBlockRole::LinearAProjection
        | WeightBlockRole::LinearDtBias
        | WeightBlockRole::LinearALog
        | WeightBlockRole::LinearOutputProjection
        | WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::DeepSeekQALoraProjection
        | WeightBlockRole::DeepSeekQALoraScaleInv
        | WeightBlockRole::DeepSeekQBProjection
        | WeightBlockRole::DeepSeekQBScaleInv
        | WeightBlockRole::DeepSeekKvAProjection
        | WeightBlockRole::DeepSeekKvAScaleInv
        | WeightBlockRole::DeepSeekKvBProjection
        | WeightBlockRole::DeepSeekKvBScaleInv
        | WeightBlockRole::DeepSeekOutputScaleInv
        | WeightBlockRole::DeepSeekIndexerQueryProjection
        | WeightBlockRole::DeepSeekIndexerQueryScaleInv
        | WeightBlockRole::DeepSeekIndexerKeyProjection
        | WeightBlockRole::DeepSeekIndexerKeyScaleInv
        | WeightBlockRole::DeepSeekIndexerWeightsProjection
        | WeightBlockRole::DeepSeekV4HcHeadBase
        | WeightBlockRole::DeepSeekV4HcHeadFn
        | WeightBlockRole::DeepSeekV4HcHeadScale
        | WeightBlockRole::DeepSeekV4HcAttnBase
        | WeightBlockRole::DeepSeekV4HcAttnFn
        | WeightBlockRole::DeepSeekV4HcAttnScale
        | WeightBlockRole::DeepSeekV4HcFfnBase
        | WeightBlockRole::DeepSeekV4HcFfnFn
        | WeightBlockRole::DeepSeekV4HcFfnScale
        | WeightBlockRole::DeepSeekV4AttentionSink
        | WeightBlockRole::DeepSeekV4WqAProjection
        | WeightBlockRole::DeepSeekV4WqAScale
        | WeightBlockRole::DeepSeekV4WqBProjection
        | WeightBlockRole::DeepSeekV4WqBScale
        | WeightBlockRole::DeepSeekV4WkvProjection
        | WeightBlockRole::DeepSeekV4WkvScale
        | WeightBlockRole::DeepSeekV4WoAProjection
        | WeightBlockRole::DeepSeekV4WoAScale
        | WeightBlockRole::DeepSeekV4WoBProjection
        | WeightBlockRole::DeepSeekV4WoBScale
        | WeightBlockRole::DeepSeekV4CompressorApe
        | WeightBlockRole::DeepSeekV4CompressorWkvProjection
        | WeightBlockRole::DeepSeekV4CompressorWkvScale
        | WeightBlockRole::DeepSeekV4CompressorWgateProjection
        | WeightBlockRole::DeepSeekV4CompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWqBProjection
        | WeightBlockRole::DeepSeekV4IndexerWqBScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorApe
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWeightsProjection
        | WeightBlockRole::DeepSeekV4IndexerWeightsScale
        | WeightBlockRole::DeepSeekV4HashRouteTable
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection
        | WeightBlockRole::GateScaleInv
        | WeightBlockRole::UpScaleInv
        | WeightBlockRole::DownScaleInv
        | WeightBlockRole::RouterProjection
        | WeightBlockRole::RouterCorrectionBias
        | WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::ExpertGateScaleInv
        | WeightBlockRole::ExpertUpScaleInv
        | WeightBlockRole::ExpertDownScaleInv
        | WeightBlockRole::DeepSeekV4ExpertGateScale
        | WeightBlockRole::DeepSeekV4ExpertUpScale
        | WeightBlockRole::DeepSeekV4ExpertDownScale
        | WeightBlockRole::SharedExpertGateProjection
        | WeightBlockRole::SharedExpertUpProjection
        | WeightBlockRole::SharedExpertDownProjection
        | WeightBlockRole::SharedExpertGateScaleInv
        | WeightBlockRole::SharedExpertUpScaleInv
        | WeightBlockRole::SharedExpertDownScaleInv
        | WeightBlockRole::DeepSeekV4SharedExpertGateScale
        | WeightBlockRole::DeepSeekV4SharedExpertUpScale
        | WeightBlockRole::DeepSeekV4SharedExpertDownScale
        | WeightBlockRole::SharedExpertRouterProjection => vec![0; elements],
    }
}

fn encoded_identity(rows: usize, cols: usize, diagonal: f32) -> Vec<u16> {
    let mut values = vec![0u16; rows * cols];
    let encoded = f32_to_f16_bits(diagonal);
    for index in 0..rows.min(cols) {
        values[index * cols + index] = encoded;
    }
    values
}

fn encode_values(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}
