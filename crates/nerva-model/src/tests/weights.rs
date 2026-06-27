use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::hf::probe::hf_metadata_probe;
use crate::tests::support::tiny_llama_manifest;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::layout::probe::hf_weight_layout_probe;
use crate::weights::layout::summary::HfWeightLayoutProbeStatus;
use crate::weights::manifest::{
    HfTensorManifestProbeStatus, build_hf_tensor_manifest, hf_tensor_manifest_probe,
};
use crate::weights::safetensors::header::synthetic_safetensors_header_for_manifest;
use crate::weights::safetensors::validation::validate_safetensors_header_for_manifest;
use nerva_core::types::dtype::DType;

#[test]
fn plans_hf_weight_layout_from_metadata() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();

    assert_eq!(plan.blocks.len(), 21);
    assert_eq!(plan.static_weight_bytes, 168);
    assert_eq!(plan.per_layer_weight_bytes, 304);
    assert_eq!(plan.total_weight_bytes, 776);
    assert_eq!(plan.blocks[0].role, WeightBlockRole::TokenEmbedding);
    assert_eq!(plan.blocks[2].role, WeightBlockRole::QueryProjection);
    assert_eq!(plan.blocks[2].rows, 4);
    assert_eq!(plan.blocks[2].cols, 4);
    assert_eq!(plan.blocks[3].role, WeightBlockRole::KeyProjection);
    assert_eq!(plan.blocks[3].rows, 2);
    assert_eq!(plan.blocks[3].cols, 4);
}

#[test]
fn tied_word_embedding_layout_omits_lm_head_block() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "tie_word_embeddings": true,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();

    assert_eq!(plan.blocks.len(), 20);
    assert_eq!(plan.static_weight_bytes, 88);
    assert_eq!(plan.per_layer_weight_bytes, 304);
    assert_eq!(plan.total_weight_bytes, 696);
    assert_eq!(plan.blocks[0].role, WeightBlockRole::TokenEmbedding);
    assert_eq!(plan.blocks.last().unwrap().role, WeightBlockRole::FinalNorm);
    assert!(plan.to_json().contains("\"tie_word_embeddings\":true"));
}

#[test]
fn hf_weight_layout_probe_reports_llama_scale_counts() {
    let summary = hf_weight_layout_probe().unwrap();

    assert_eq!(summary.status, HfWeightLayoutProbeStatus::Ok);
    assert_eq!(summary.plan.blocks.len(), 291);
    assert_eq!(summary.plan.static_weight_bytes, 524_296_192);
    assert_eq!(summary.plan.per_layer_weight_bytes, 354_435_072);
    assert_eq!(summary.plan.total_weight_bytes, 11_866_218_496);
    assert_eq!(summary.plan.dtype, DType::BF16);
    assert_ne!(summary.layout_hash, 0);
    assert!(summary.to_json().contains("\"blocks\":291"));
}

#[test]
fn weight_layout_requires_exact_declared_dtype() {
    let mut metadata = hf_metadata_probe().unwrap().metadata;
    metadata.torch_dtype = None;
    assert!(plan_hf_weight_layout(&metadata).is_err());

    metadata.torch_dtype = Some(DType::U8);
    assert!(plan_hf_weight_layout(&metadata).is_err());
}

#[test]
fn builds_canonical_hf_tensor_manifest_names() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();

    assert_eq!(manifest.entries.len(), plan.blocks.len());
    assert_eq!(manifest.total_weight_bytes, plan.total_weight_bytes);
    assert_eq!(manifest.entries[0].name, "model.embed_tokens.weight");
    assert_eq!(
        manifest.entries[1].name,
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(manifest.entries[1].rank, 1);
    assert_eq!(
        manifest.entries[2].name,
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(manifest.entries[2].rank, 2);
    assert_eq!(
        manifest.entries[9].name,
        "model.layers.0.mlp.down_proj.weight"
    );
    assert_eq!(
        manifest.entries[10].name,
        "model.layers.1.input_layernorm.weight"
    );
    assert_eq!(
        manifest.entries[18].name,
        "model.layers.1.mlp.down_proj.weight"
    );
    assert_eq!(manifest.entries[19].name, "model.norm.weight");
    assert_eq!(manifest.entries[19].rank, 1);
    assert_eq!(manifest.entries.last().unwrap().name, "lm_head.weight");
    assert_ne!(manifest.manifest_hash, 0);
}

#[test]
fn tied_word_embedding_manifest_omits_lm_head_tensor() {
    let manifest = tiny_llama_manifest(true);
    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();

    assert_eq!(manifest.entries.len(), 20);
    assert_eq!(manifest.total_weight_bytes, 696);
    assert_eq!(manifest.entries[0].name, "model.embed_tokens.weight");
    assert_eq!(manifest.entries.last().unwrap().name, "model.norm.weight");
    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.name == "lm_head.weight")
    );
    assert!(!header.contains("lm_head.weight"));
    assert_eq!(validation.validated_tensors, 20);
    assert_eq!(validation.total_data_bytes, 696);
}

#[test]
fn tensor_manifest_rejects_unsupported_architecture_names() {
    let mut metadata = hf_metadata_probe().unwrap().metadata;
    metadata.architecture = HfArchitectureKind::Gemma;

    assert!(plan_hf_weight_layout(&metadata).is_err());
}

#[test]
fn hf_tensor_manifest_probe_reports_llama_manifest() {
    let summary = hf_tensor_manifest_probe().unwrap();

    assert_eq!(summary.status, HfTensorManifestProbeStatus::Ok);
    assert_eq!(summary.manifest.entries.len(), 291);
    assert_eq!(summary.manifest.total_weight_bytes, 11_866_218_496);
    assert_eq!(
        summary.manifest.entries.first().unwrap().name,
        "model.embed_tokens.weight"
    );
    assert_eq!(
        summary.manifest.entries.last().unwrap().name,
        "lm_head.weight"
    );
    assert_ne!(summary.manifest.manifest_hash, 0);
    assert!(summary.to_json().contains("\"entries\":291"));
}
