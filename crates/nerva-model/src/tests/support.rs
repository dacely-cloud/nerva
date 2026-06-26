use crate::common::json::format::json_escape;
use crate::common::math::dot;
use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::plan_hf_weight_layout;
use crate::weights::manifest::{HfTensorManifest, HfTensorManifestEntry, build_hf_tensor_manifest};
use crate::weights::safetensors::header::synthetic_safetensors_header_for_manifest;

pub(crate) const SHARD_ONE: &str = "model-00001-of-00002.safetensors";
pub(crate) const SHARD_TWO: &str = "model-00002-of-00002.safetensors";

pub(crate) fn tiny_llama_manifest(tie_word_embeddings: bool) -> HfTensorManifest {
    let config = format!(
        r#"{{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "tie_word_embeddings": {tie_word_embeddings},
                "torch_dtype": "float16"
            }}"#,
    );
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();
    build_hf_tensor_manifest(&plan).unwrap()
}

pub(crate) fn synthetic_header_for_entries(
    architecture: HfArchitectureKind,
    entries: &[HfTensorManifestEntry],
) -> String {
    let total_weight_bytes = entries.iter().map(|entry| entry.bytes).sum();
    let manifest = HfTensorManifest {
        architecture,
        entries: entries.to_vec(),
        total_weight_bytes,
        manifest_hash: 0,
    };
    synthetic_safetensors_header_for_manifest(&manifest).unwrap()
}

pub(crate) fn synthetic_sharded_index_json(manifest: &HfTensorManifest, split_at: usize) -> String {
    let mut out = format!(
        "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
        manifest.total_weight_bytes
    );
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(&entry.name));
        out.push_str("\":\"");
        out.push_str(if index < split_at {
            SHARD_ONE
        } else {
            SHARD_TWO
        });
        out.push('"');
    }
    out.push_str("}}");
    out
}

pub(crate) fn dense_attention_reference(
    shape: TransformerBlockShape,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    token_count: usize,
) -> Vec<f32> {
    let head_dim = shape.head_dim();
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; shape.hidden];
    for head in 0..shape.heads {
        let head_start = head * head_dim;
        let head_end = head_start + head_dim;
        let mut max_score = f32::NEG_INFINITY;
        let mut scores = Vec::with_capacity(token_count);
        for token_index in 0..token_count {
            let token_start = token_index * shape.hidden + head_start;
            let token_end = token_start + head_dim;
            let score = dot(&query[head_start..head_end], &keys[token_start..token_end]) * scale;
            max_score = max_score.max(score);
            scores.push(score);
        }
        let mut denom = 0.0f32;
        for (token_index, score) in scores.iter().copied().enumerate() {
            let weight = (score - max_score).exp();
            denom += weight;
            let token_start = token_index * shape.hidden + head_start;
            let token_end = token_start + head_dim;
            for (out, value) in output[head_start..head_end]
                .iter_mut()
                .zip(values[token_start..token_end].iter().copied())
            {
                *out += weight * value;
            }
        }
        for out in &mut output[head_start..head_end] {
            *out /= denom;
        }
    }
    output
}
