use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::exact_blockwise_attention_into;
use crate::attention::scratch::BlockwiseAttentionScratch;
use crate::attention::smoke::{BlockwiseAttentionSmokeStatus, blockwise_attention_smoke};
use crate::common::json::json_escape;
use crate::common::math::{dot, silu};
use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::hf::probe::{HfMetadataProbeStatus, hf_metadata_probe};
use crate::precision::bits::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits,
};
use crate::precision::block::PrecisionTransformerBlock;
use crate::precision::file_smoke::{
    PrecisionSafetensorsBlockSmokeStatus, precision_block_from_safetensors_smoke,
};
use crate::precision::scratch::PrecisionTransformerBlockScratch;
use crate::precision::smoke::{PrecisionBlockSmokeStatus, precision_block_smoke};
use crate::reference::block::ReferenceTransformerBlock;
use crate::reference::scratch::TransformerBlockScratch;
use crate::reference::smoke::{ReferenceBlockSmokeStatus, reference_block_smoke};
use crate::tiny::output::TinyGreedyDecodeStatus;
use crate::tiny::scratch::TinyGreedyDecodeScratch;
use crate::tiny::smoke::{tiny_cycle_model, tiny_greedy_decode_smoke};
use crate::warm_compute::probe::warm_compute_probe;
use crate::warm_compute::strategy::WarmComputeStrategy;
use crate::warm_compute::summary::WarmComputeProbeStatus;
use crate::weights::file::{read_safetensors_header_file, read_safetensors_header_file_with_limit};
use crate::weights::layout::{
    HfWeightLayoutProbeStatus, WeightBlockRole, hf_weight_layout_probe, plan_hf_weight_layout,
};
use crate::weights::manifest::{
    HfTensorManifest, HfTensorManifestEntry, HfTensorManifestProbeStatus, build_hf_tensor_manifest,
    hf_tensor_manifest_probe,
};
use crate::weights::safetensors::{
    SafetensorsShardHeader, SafetensorsValidationStatus, plan_safetensors_shards_for_manifest,
    required_safetensors_shards_for_manifest, safetensors_header_from_bytes,
    safetensors_header_probe, synthetic_safetensors_header_for_manifest,
    validate_safetensors_header_for_manifest,
};
use nerva_core::types::dtype::DType;
use nerva_core::types::id::TokenId;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::TokenLedger;

const SHARD_ONE: &str = "model-00001-of-00002.safetensors";
const SHARD_TWO: &str = "model-00002-of-00002.safetensors";

fn tiny_llama_manifest(tie_word_embeddings: bool) -> HfTensorManifest {
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

fn synthetic_header_for_entries(
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

fn synthetic_sharded_index_json(manifest: &HfTensorManifest, split_at: usize) -> String {
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

#[test]
fn parses_llama_hf_config_metadata() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "torch_dtype": "bfloat16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Llama);
    assert_eq!(
        metadata.block_shape(),
        TransformerBlockShape::new(4096, 32, 11008)
    );
    assert_eq!(metadata.head_dim(), 128);
    assert_eq!(metadata.kv_groups(), 4);
    assert_eq!(metadata.torch_dtype, Some(DType::BF16));
    assert!(!metadata.tie_word_embeddings);
    assert!(metadata.to_json().contains("\"architecture\":\"llama\""));
}

#[test]
fn parses_model_type_and_defaults_kv_heads_to_attention_heads() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "mistral",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Mistral);
    assert_eq!(metadata.num_key_value_heads, 32);
    assert_eq!(metadata.kv_groups(), 1);
    assert_eq!(metadata.torch_dtype, Some(DType::F16));
}

#[test]
fn parses_tied_word_embedding_config() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "qwen2",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 12,
                "tie_word_embeddings": true,
                "torch_dtype": "bfloat16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen2);
    assert!(metadata.tie_word_embeddings);
    assert!(metadata.to_json().contains("\"tie_word_embeddings\":true"));
}

#[test]
fn rejects_invalid_hf_metadata_shapes_and_dtypes() {
    let bad_heads = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4097,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000
            }"#,
    );
    let bad_dtype = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "torch_dtype": "int4"
            }"#,
    );
    let bad_tie_word_embeddings = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "tie_word_embeddings": "yes"
            }"#,
    );

    assert!(bad_heads.is_err());
    assert!(bad_dtype.is_err());
    assert!(bad_tie_word_embeddings.is_err());
}

#[test]
fn hf_metadata_probe_reports_valid_shape() {
    let summary = hf_metadata_probe().unwrap();

    assert_eq!(summary.status, HfMetadataProbeStatus::Ok);
    assert_eq!(summary.metadata.architecture, HfArchitectureKind::Llama);
    assert_eq!(summary.metadata.hidden_size, 4096);
    assert_eq!(summary.metadata.num_attention_heads, 32);
    assert_eq!(summary.metadata.num_key_value_heads, 8);
    assert_eq!(summary.metadata.head_dim(), 128);
    assert_eq!(summary.metadata.kv_groups(), 4);
    assert_ne!(summary.metadata_hash, 0);
    assert!(summary.to_json().contains("\"metadata\""));
}

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

    assert_eq!(plan.blocks.len(), 20);
    assert_eq!(plan.static_weight_bytes, 160);
    assert_eq!(plan.per_layer_weight_bytes, 304);
    assert_eq!(plan.total_weight_bytes, 768);
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

    assert_eq!(plan.blocks.len(), 19);
    assert_eq!(plan.static_weight_bytes, 80);
    assert_eq!(plan.per_layer_weight_bytes, 304);
    assert_eq!(plan.total_weight_bytes, 688);
    assert_eq!(plan.blocks[0].role, WeightBlockRole::TokenEmbedding);
    assert_eq!(
        plan.blocks.last().unwrap().role,
        WeightBlockRole::DownProjection
    );
    assert!(plan.to_json().contains("\"tie_word_embeddings\":true"));
}

#[test]
fn hf_weight_layout_probe_reports_llama_scale_counts() {
    let summary = hf_weight_layout_probe().unwrap();

    assert_eq!(summary.status, HfWeightLayoutProbeStatus::Ok);
    assert_eq!(summary.plan.blocks.len(), 290);
    assert_eq!(summary.plan.static_weight_bytes, 524_288_000);
    assert_eq!(summary.plan.per_layer_weight_bytes, 354_435_072);
    assert_eq!(summary.plan.total_weight_bytes, 11_866_210_304);
    assert_eq!(summary.plan.dtype, DType::BF16);
    assert_ne!(summary.layout_hash, 0);
    assert!(summary.to_json().contains("\"blocks\":290"));
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
    assert_eq!(manifest.entries.last().unwrap().name, "lm_head.weight");
    assert_ne!(manifest.manifest_hash, 0);
}

#[test]
fn tied_word_embedding_manifest_omits_lm_head_tensor() {
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
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();

    assert_eq!(manifest.entries.len(), 19);
    assert_eq!(manifest.total_weight_bytes, 688);
    assert_eq!(manifest.entries[0].name, "model.embed_tokens.weight");
    assert_eq!(
        manifest.entries.last().unwrap().name,
        "model.layers.1.mlp.down_proj.weight"
    );
    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.name == "lm_head.weight")
    );
    assert!(!header.contains("lm_head.weight"));
    assert_eq!(validation.validated_tensors, 19);
    assert_eq!(validation.total_data_bytes, 688);
}

#[test]
fn tensor_manifest_rejects_unsupported_architecture_names() {
    let mut metadata = hf_metadata_probe().unwrap().metadata;
    metadata.architecture = HfArchitectureKind::Gemma;
    let plan = plan_hf_weight_layout(&metadata).unwrap();

    assert!(build_hf_tensor_manifest(&plan).is_err());
}

#[test]
fn hf_tensor_manifest_probe_reports_llama_manifest() {
    let summary = hf_tensor_manifest_probe().unwrap();

    assert_eq!(summary.status, HfTensorManifestProbeStatus::Ok);
    assert_eq!(summary.manifest.entries.len(), 290);
    assert_eq!(summary.manifest.total_weight_bytes, 11_866_210_304);
    assert_eq!(
        summary.manifest.entries.first().unwrap().name,
        "model.embed_tokens.weight"
    );
    assert_eq!(
        summary.manifest.entries.last().unwrap().name,
        "lm_head.weight"
    );
    assert_ne!(summary.manifest.manifest_hash, 0);
    assert!(summary.to_json().contains("\"entries\":290"));
}

#[test]
fn validates_synthetic_safetensors_header_against_manifest() {
    let manifest = hf_tensor_manifest_probe().unwrap().manifest;
    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();

    assert_eq!(validation.status, SafetensorsValidationStatus::Ok);
    assert_eq!(validation.manifest_entries, manifest.entries.len());
    assert_eq!(validation.validated_tensors, manifest.entries.len());
    assert_eq!(validation.total_data_bytes, manifest.total_weight_bytes);
    assert_eq!(validation.manifest_hash, manifest.manifest_hash);
    assert_ne!(validation.header_hash, 0);
}

#[test]
fn extracts_safetensors_header_from_file_bytes() {
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0xaa, 0xbb]);

    assert_eq!(safetensors_header_from_bytes(&bytes).unwrap(), header);
    assert!(safetensors_header_from_bytes(&bytes[..4]).is_err());
}

#[test]
fn reads_safetensors_file_header_without_payload_scan() {
    let dir = std::env::temp_dir().join(format!("nerva-model-header-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
    std::fs::write(&path, bytes).unwrap();

    let file_header = read_safetensors_header_file(&path).unwrap();

    assert_eq!(file_header.header_json, header);
    assert_eq!(file_header.header_bytes, header.len());
    assert_eq!(file_header.data_start, 8 + header.len());
    assert_eq!(file_header.payload_bytes, 4);
    assert!(file_header.require_payload_bytes(4).is_ok());
    assert!(file_header.require_payload_bytes(5).is_err());
    assert!(
        file_header
            .require_file_offset_end(8 + header.len() + 4)
            .is_ok()
    );
    assert!(
        file_header
            .require_file_offset_end(8 + header.len() + 5)
            .is_err()
    );
    assert!(file_header.to_json().contains("\"payload_bytes\":4"));

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn safetensors_file_header_rejects_oversized_header_limit() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-model-header-limit-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    std::fs::write(&path, bytes).unwrap();

    assert!(read_safetensors_header_file_with_limit(&path, header.len() - 1).is_err());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn safetensors_validation_rejects_missing_and_mismatched_tensors() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    let valid = synthetic_safetensors_header_for_manifest(&manifest).unwrap();

    assert!(validate_safetensors_header_for_manifest("{}", &manifest).is_err());

    let first = &manifest.entries[0];
    let bad_dtype = format!(
        "{{\"{}\":{{\"dtype\":\"F32\",\"shape\":[{},{}],\"data_offsets\":[0,{}]}}}}",
        first.name, first.rows, first.cols, first.bytes
    );
    assert!(validate_safetensors_header_for_manifest(&bad_dtype, &manifest).is_err());

    let bad_shape = valid.replacen(
        &format!("\"shape\":[{},{}]", first.rows, first.cols),
        "\"shape\":[1,1]",
        1,
    );
    assert!(validate_safetensors_header_for_manifest(&bad_shape, &manifest).is_err());
}

#[test]
fn safetensors_header_probe_reports_manifest_parity() {
    let summary = safetensors_header_probe().unwrap();

    assert_eq!(summary.status, SafetensorsValidationStatus::Ok);
    assert_eq!(summary.validation.manifest_entries, 290);
    assert_eq!(summary.validation.validated_tensors, 290);
    assert_eq!(summary.validation.total_data_bytes, 11_866_210_304);
    assert_ne!(summary.validation.header_hash, 0);
    assert!(summary.to_json().contains("\"validated_tensors\":290"));
}

#[test]
fn plans_safetensors_shards_from_index_and_headers() {
    let manifest = tiny_llama_manifest(false);
    let index = synthetic_sharded_index_json(&manifest, 10);
    let header_one = synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    let header_two = synthetic_header_for_entries(manifest.architecture, &manifest.entries[10..]);

    let required = required_safetensors_shards_for_manifest(&index, &manifest).unwrap();
    let plan = plan_safetensors_shards_for_manifest(
        &index,
        &[
            SafetensorsShardHeader::new(SHARD_ONE, &header_one),
            SafetensorsShardHeader::new(SHARD_TWO, &header_two),
        ],
        &manifest,
    )
    .unwrap();

    assert_eq!(required, vec![SHARD_ONE.to_string(), SHARD_TWO.to_string()]);
    assert_eq!(plan.entries.len(), 20);
    assert_eq!(plan.shards.len(), 2);
    assert_eq!(plan.total_weight_bytes, 768);
    assert_eq!(plan.index_total_size, Some(768));
    assert_eq!(plan.shards[0].file_name, SHARD_ONE);
    assert_eq!(plan.shards[0].tensor_count, 10);
    assert_eq!(plan.shards[0].payload_bytes, 384);
    assert_eq!(plan.entries[0].tensor_name, "model.embed_tokens.weight");
    assert_eq!(plan.entries[0].file_offset_begin, 8 + header_one.len());
    assert_eq!(
        plan.entries[0].file_offset_end,
        8 + header_one.len() + plan.entries[0].bytes
    );
    assert_eq!(plan.entries[10].shard_file, SHARD_TWO);
    assert_ne!(plan.plan_hash, 0);
    assert!(plan.to_json().contains("\"shards\":2"));
}

#[test]
fn safetensors_shard_plan_supports_tied_embedding_manifest() {
    let manifest = tiny_llama_manifest(true);
    let index = synthetic_sharded_index_json(&manifest, 10);
    let header_one = synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    let header_two = synthetic_header_for_entries(manifest.architecture, &manifest.entries[10..]);
    let plan = plan_safetensors_shards_for_manifest(
        &index,
        &[
            SafetensorsShardHeader::new(SHARD_ONE, &header_one),
            SafetensorsShardHeader::new(SHARD_TWO, &header_two),
        ],
        &manifest,
    )
    .unwrap();

    assert_eq!(plan.entries.len(), 19);
    assert_eq!(plan.total_weight_bytes, 688);
    assert_eq!(
        plan.entries.last().unwrap().tensor_name,
        "model.layers.1.mlp.down_proj.weight"
    );
    assert!(
        !plan
            .entries
            .iter()
            .any(|entry| entry.tensor_name == "lm_head.weight")
    );
}

#[test]
fn safetensors_shard_plan_rejects_missing_index_or_header() {
    let manifest = tiny_llama_manifest(false);
    let index = synthetic_sharded_index_json(&manifest, 10);
    let header_one = synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    let missing_lm_head_index = index.replace(
        "\"lm_head.weight\":\"model-00002-of-00002.safetensors\"",
        "\"unused.weight\":\"model-00002-of-00002.safetensors\"",
    );

    assert!(required_safetensors_shards_for_manifest(&missing_lm_head_index, &manifest).is_err());
    assert!(
        plan_safetensors_shards_for_manifest(
            &index,
            &[SafetensorsShardHeader::new(SHARD_ONE, &header_one)],
            &manifest,
        )
        .is_err()
    );
    assert!(
        plan_safetensors_shards_for_manifest(
            &index,
            &[
                SafetensorsShardHeader::new(SHARD_ONE, &header_one),
                SafetensorsShardHeader::new(SHARD_ONE, &header_one),
            ],
            &manifest,
        )
        .is_err()
    );
}

#[test]
fn zero_block_preserves_residual() {
    let shape = TransformerBlockShape::new(4, 2, 8);
    let block = ReferenceTransformerBlock::zero_for_shape(shape).unwrap();
    let mut scratch = TransformerBlockScratch::new(shape).unwrap();
    let mut output = [0.0; 4];
    let input = [1.0, -2.0, 3.0, -4.0];
    let mut ledger = TokenLedger::new(0);

    block
        .forward_into(&input, &mut scratch, &mut output, &mut ledger)
        .unwrap();

    assert_eq!(output, input);
    assert_eq!(ledger.hot_path_allocations, 0);
    assert!(ledger.require_zero_hot_path_allocations().is_ok());
}

#[test]
fn nontrivial_block_matches_hand_reference() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )
    .unwrap();
    let mut scratch = TransformerBlockScratch::new(shape).unwrap();
    let mut output = [0.0; 2];
    let input = [1.0, 2.0];
    let mut ledger = TokenLedger::new(7);

    block
        .forward_into(&input, &mut scratch, &mut output, &mut ledger)
        .unwrap();

    let attn_norm_scale = ((1.0_f32 + 4.0) / 2.0 + 1e-5).sqrt().recip();
    let attn = [input[0] * attn_norm_scale, input[1] * attn_norm_scale];
    let residual = [input[0] + attn[0], input[1] + attn[1]];
    let mlp_norm_scale = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0 + 1e-5)
        .sqrt()
        .recip();
    let mlp_norm = [residual[0] * mlp_norm_scale, residual[1] * mlp_norm_scale];
    let expected = [
        residual[0] + silu(0.5 * mlp_norm[0]) * mlp_norm[0],
        residual[1] + silu(0.5 * mlp_norm[1]) * mlp_norm[1],
    ];

    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(ledger.hot_path_allocations, 0);
}

#[test]
fn rejects_bad_shapes_and_scratch_mismatch() {
    assert!(TransformerBlockShape::new(3, 2, 4).validate().is_err());
    let block =
        ReferenceTransformerBlock::zero_for_shape(TransformerBlockShape::new(4, 2, 8)).unwrap();
    let mut scratch = TransformerBlockScratch::new(TransformerBlockShape::new(2, 1, 2)).unwrap();
    let mut ledger = TokenLedger::new(0);
    let mut output = [0.0; 4];
    assert!(
        block
            .forward_into(&[0.0; 4], &mut scratch, &mut output, &mut ledger)
            .is_err()
    );
}

#[test]
fn reference_block_smoke_reports_hash_and_no_allocations() {
    let summary = reference_block_smoke().unwrap();
    assert_eq!(summary.status, ReferenceBlockSmokeStatus::Ok);
    assert_eq!(summary.hidden, 2);
    assert_eq!(summary.heads, 1);
    assert_eq!(summary.intermediate, 2);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.output_hash, 3_850_145_622_605_741_247);
    assert!(summary.to_json().contains("\"status\":\"ok\""));
}

#[test]
fn f16_and_bf16_conversions_round_known_values() {
    assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
    assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
    assert_eq!(f32_to_f16_bits(0.5), 0x3800);
    assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
    assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
    assert_eq!(f16_bits_to_f32(0xc000), -2.0);

    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
    assert_eq!(f32_to_bf16_bits(-2.0), 0xc000);
    assert_eq!(f32_to_bf16_bits(0.5), 0x3f00);
    assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_bits_to_f32(0xc000), -2.0);
}

#[test]
fn precision_block_smoke_reports_f16_and_bf16_bit_parity() {
    let summary = precision_block_smoke().unwrap();

    assert_eq!(summary.status, PrecisionBlockSmokeStatus::Ok);
    assert!(summary.passed());
    assert!(summary.f16.bit_parity);
    assert!(summary.bf16.bit_parity);
    assert_eq!(summary.f16.hot_path_allocations, 0);
    assert_eq!(summary.bf16.hot_path_allocations, 0);
    assert_eq!(summary.f16.output_hash, summary.f16.expected_hash);
    assert_eq!(summary.bf16.output_hash, summary.bf16.expected_hash);
    assert!(summary.to_json().contains("\"dtype\":\"float16\""));
    assert!(summary.to_json().contains("\"dtype\":\"bfloat16\""));
}

#[test]
fn precision_block_loads_weights_from_safetensors_payload() {
    let summary = precision_block_from_safetensors_smoke().unwrap();

    assert_eq!(summary.status, PrecisionSafetensorsBlockSmokeStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.tensors_loaded, 9);
    assert_eq!(summary.bytes_loaded, 64);
    assert_ne!(summary.data_hash, 0);
    assert_eq!(summary.output_hash, summary.expected_hash);
    assert!(summary.bit_parity);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"tensors_loaded\":9"));
}

#[test]
fn precision_block_rejects_non_16_bit_dtypes() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let rms = [1.0, 1.0];
    let identity = [1.0, 0.0, 0.0, 1.0];

    assert!(
        PrecisionTransformerBlock::new_from_f32(
            DType::F32,
            shape,
            &rms,
            &rms,
            &identity,
            &identity,
            &identity,
            &identity,
            &identity,
            &identity,
            &identity,
            1e-5,
        )
        .is_err()
    );
}

#[test]
fn precision_block_rejects_scratch_shape_mismatch() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let rms = [1.0, 1.0];
    let identity = [1.0, 0.0, 0.0, 1.0];
    let block = PrecisionTransformerBlock::new_from_f32(
        DType::F16,
        shape,
        &rms,
        &rms,
        &identity,
        &identity,
        &identity,
        &identity,
        &identity,
        &identity,
        &identity,
        1e-5,
    )
    .unwrap();
    let mut scratch =
        PrecisionTransformerBlockScratch::new(TransformerBlockShape::new(4, 2, 4)).unwrap();
    let input = [f32_to_f16_bits(1.0), f32_to_f16_bits(2.0)];
    let mut output = [0u16; 2];
    let mut ledger = TokenLedger::new(0);

    assert!(
        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .is_err()
    );
}

#[test]
fn blockwise_attention_matches_dense_reference_across_tiers() {
    let shape = TransformerBlockShape::new(4, 2, 4);
    let query = [0.5, -1.0, 0.25, 0.75];
    let keys = [0.1, 0.2, 0.3, 0.4, 0.0, -0.5, 0.6, 0.2, 0.7, 0.1, -0.2, 0.3];
    let values = [
        1.0, 0.0, 0.5, -0.5, -1.0, 2.0, 0.25, 0.75, 0.3, -0.8, 1.5, 0.2,
    ];
    let blocks = [
        KvAttentionBlock::new(&keys[..4], &values[..4], 1, MemoryTier::Dram),
        KvAttentionBlock::new(&keys[4..], &values[4..], 2, MemoryTier::Vram),
    ];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 4];
    let mut ledger = TokenLedger::new(11);

    exact_blockwise_attention_into(
        shape,
        &query,
        &blocks,
        &mut scratch,
        &mut output,
        &mut ledger,
    )
    .unwrap();

    let expected = dense_attention_reference(shape, &query, &keys, &values, 3);
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(ledger.event_count(LedgerEventKind::CpuActivity), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
    assert_eq!(ledger.total_latency_ns(), 3);
    assert_eq!(ledger.hot_path_allocations, 0);
}

#[test]
fn blockwise_attention_rejects_empty_and_malformed_blocks() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let query = [1.0, 0.0];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);

    assert!(
        exact_blockwise_attention_into(shape, &query, &[], &mut scratch, &mut output, &mut ledger)
            .is_err()
    );

    let bad_block = [KvAttentionBlock::new(
        &[1.0],
        &[1.0, 0.0],
        1,
        MemoryTier::Dram,
    )];
    assert!(
        exact_blockwise_attention_into(
            shape,
            &query,
            &bad_block,
            &mut scratch,
            &mut output,
            &mut ledger,
        )
        .is_err()
    );
}

#[test]
fn blockwise_attention_smoke_reports_tier_events() {
    let summary = blockwise_attention_smoke().unwrap();
    assert_eq!(summary.status, BlockwiseAttentionSmokeStatus::Ok);
    assert_eq!(summary.hidden, 2);
    assert_eq!(summary.heads, 1);
    assert_eq!(summary.blocks, 2);
    assert_eq!(summary.tokens, 4);
    assert_eq!(summary.cpu_block_events, 1);
    assert_eq!(summary.device_block_events, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"device_block_events\":1"));
}

#[test]
fn warm_compute_probe_compares_all_exact_strategies() {
    let summary = warm_compute_probe().unwrap();

    assert_eq!(summary.status, WarmComputeProbeStatus::Ok);
    assert_eq!(summary.rows, 4);
    assert_eq!(summary.cols, 4);
    assert_eq!(summary.candidates.len(), 4);
    assert_eq!(summary.selected_strategy, WarmComputeStrategy::GpuResident);
    assert!(summary.parity);
    assert!(summary.cpu_beats_staged);
    assert_eq!(summary.execution_decisions, 1);
    assert_eq!(summary.cpu_events, 2);
    assert_eq!(summary.device_events, 3);
    assert_eq!(summary.copy_events, 3);
    assert_eq!(summary.copy_bytes, 104);
    assert_eq!(summary.total_latency_ns, 138);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .candidates
            .iter()
            .all(|candidate| candidate.output_hash == summary.output_hash)
    );
    assert!(
        summary
            .to_json()
            .contains("\"selected_strategy\":\"gpu-resident\"")
    );
}

#[test]
fn tiny_greedy_model_matches_expected_token_cycle() {
    let model = tiny_cycle_model().unwrap();
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();
    let output = model.decode_greedy(TokenId(0), 8, &mut scratch).unwrap();

    assert_eq!(
        output.tokens,
        vec![
            TokenId(1),
            TokenId(2),
            TokenId(3),
            TokenId(0),
            TokenId(1),
            TokenId(2),
            TokenId(3),
            TokenId(0),
        ]
    );
    assert_eq!(output.ledgers.len(), 8);
    assert_eq!(
        output
            .ledgers
            .iter()
            .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
            .sum::<u64>(),
        8
    );
    assert_eq!(
        output
            .ledgers
            .iter()
            .map(|ledger| ledger.hot_path_allocations)
            .sum::<u64>(),
        0
    );
}

#[test]
fn tiny_greedy_model_rejects_bad_decode_inputs() {
    let model = tiny_cycle_model().unwrap();
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();

    assert!(model.decode_greedy(TokenId(0), 0, &mut scratch).is_err());
    assert!(model.decode_greedy(TokenId(99), 1, &mut scratch).is_err());

    let mut wrong_scratch =
        TinyGreedyDecodeScratch::new(TransformerBlockShape::new(4, 2, 4), model.vocab_size())
            .unwrap();
    assert!(
        model
            .decode_greedy(TokenId(0), 1, &mut wrong_scratch)
            .is_err()
    );
}

#[test]
fn tiny_greedy_decode_smoke_reports_parity_and_ledger() {
    let summary = tiny_greedy_decode_smoke(8).unwrap();

    assert_eq!(summary.status, TinyGreedyDecodeStatus::Ok);
    assert_eq!(summary.seed_token, TokenId(0));
    assert_eq!(summary.steps, 8);
    assert_eq!(summary.vocab_size, 4);
    assert!(summary.parity);
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_eq!(summary.ledger_count, 8);
    assert_eq!(summary.device_events, 8);
    assert_eq!(summary.total_latency_ns, 8);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"parity\":true"));
}

fn dense_attention_reference(
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
