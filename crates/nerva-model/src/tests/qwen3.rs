use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::linear_attention::{ConvStateLayout, Qwen35GatedDeltaNetSpec};
use crate::hf::metadata::{HfAttentionLayerKind, HfMlpLayerKind};
use crate::hf::parser::parse_hf_config_metadata;
use crate::precision::block::gdn::{PrecisionGatedDeltaNetConfig, PrecisionGatedDeltaNetMoeBlock};
use crate::precision::block::moe::PrecisionMoeConfig;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;
use crate::weights::safetensors::header::synthetic_safetensors_header_for_manifest;
use crate::weights::safetensors::validation::validate_safetensors_header_for_manifest;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::NervaError;

#[test]
fn qwen3_dense_config_requires_qk_norm_tensors() {
    let metadata = parse_hf_config_metadata(qwen3_dense_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3);
    assert!(metadata.qk_norm);
    assert!(metadata.to_json().contains("\"qk_norm\":true"));

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert_eq!(plan.blocks.len(), 14);
    assert!(manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::QueryNorm
            && entry.name == "model.layers.0.self_attn.q_norm.weight"
            && entry.rows == metadata.head_dim
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::KeyNorm
            && entry.name == "model.layers.0.self_attn.k_norm.weight"
            && entry.rows == metadata.head_dim
    }));
}

#[test]
fn qwen3_5_config_preserves_attention_layer_types() {
    let metadata = parse_hf_config_metadata(qwen3_5_hybrid_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen35);
    assert_eq!(metadata.hidden_size, 2560);
    assert_eq!(metadata.num_hidden_layers, 4);
    assert_eq!(metadata.head_dim, 256);
    assert_eq!(metadata.vocab_size, 248320);
    assert_eq!(metadata.torch_dtype, Some(DType::BF16));
    assert!(metadata.tie_word_embeddings);
    assert!(metadata.qk_norm);
    assert!(metadata.has_linear_attention_layers());
    assert_eq!(metadata.linear_conv_kernel_dim, Some(4));
    assert_eq!(metadata.linear_key_head_dim, Some(128));
    assert_eq!(metadata.linear_value_head_dim, Some(128));
    assert_eq!(metadata.linear_num_key_heads, Some(16));
    assert_eq!(metadata.linear_num_value_heads, Some(32));
    assert_eq!(
        metadata.attention_layer_types,
        vec![
            HfAttentionLayerKind::Linear,
            HfAttentionLayerKind::Linear,
            HfAttentionLayerKind::Linear,
            HfAttentionLayerKind::Full,
        ]
    );
    assert!(metadata.to_json().contains("\"attention_linear_layers\":3"));
    assert!(metadata.to_json().contains("\"linear_conv_kernel_dim\":4"));

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::LinearQkvProjection
            && entry.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            && (entry.rows, entry.cols) == (8192, 2560)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::LinearConvProjection
            && entry.name == "model.language_model.layers.0.linear_attn.conv1d.weight"
            && (entry.rows, entry.cols) == (8192, 4)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::LinearNorm
            && entry.name == "model.language_model.layers.0.linear_attn.norm.weight"
            && entry.rows == 128
            && entry.dtype == DType::F32
    }));

    let err = crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap_err();
    let NervaError::InvalidArgument { reason } = err else {
        panic!("expected invalid argument, got {err:?}");
    };
    assert!(reason.contains("Qwen3.5 linear_attention"));
    assert!(reason.contains("GatedDeltaNet"));
}

#[test]
fn qwen3_5_real_4b_config_uses_language_model_prefix() {
    let metadata = parse_hf_config_metadata(qwen3_5_4b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen35);
    assert_eq!(metadata.hidden_size, 2560);
    assert_eq!(metadata.intermediate_size, 9216);
    assert_eq!(metadata.num_hidden_layers, 32);
    assert_eq!(metadata.num_attention_heads, 16);
    assert_eq!(metadata.num_key_value_heads, 4);
    assert!(metadata.tie_word_embeddings);
    assert!(metadata.qk_norm);
    assert!(metadata.has_linear_attention_layers());

    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert_eq!(manifest.entries.len(), 426);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.embed_tokens.weight"
            && entry.role == WeightBlockRole::TokenEmbedding
            && (entry.rows, entry.cols) == (248320, 2560)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.norm.weight"
            && entry.role == WeightBlockRole::FinalNorm
            && entry.rows == 2560
    }));
    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.role == WeightBlockRole::LmHead)
    );
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            && entry.role == WeightBlockRole::LinearQkvProjection
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (8192, 2560)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.3.self_attn.q_norm.weight"
            && entry.role == WeightBlockRole::QueryNorm
            && entry.layer == Some(3)
            && entry.rows == 256
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.31.mlp.down_proj.weight"
            && entry.role == WeightBlockRole::DownProjection
            && entry.layer == Some(31)
            && (entry.rows, entry.cols) == (2560, 9216)
    }));

    let err = crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap_err();
    let NervaError::InvalidArgument { reason } = err else {
        panic!("expected invalid argument, got {err:?}");
    };
    assert!(reason.contains("Qwen3.5 linear_attention"));
    assert!(reason.contains("GatedDeltaNet"));
}

#[test]
fn qwen3_5_defaults_attention_schedule_from_interval() {
    let metadata = parse_hf_config_metadata(
        r#"{
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "vocab_size": 248320,
                "full_attention_interval": 2,
                "dtype": "bfloat16"
            }
        }"#,
    )
    .unwrap();

    assert_eq!(
        metadata.attention_layer_types,
        vec![
            HfAttentionLayerKind::Linear,
            HfAttentionLayerKind::Full,
            HfAttentionLayerKind::Linear,
            HfAttentionLayerKind::Full,
        ]
    );
    assert_eq!(metadata.linear_conv_kernel_dim, Some(4));
    assert_eq!(metadata.linear_key_head_dim, Some(128));
    assert_eq!(metadata.linear_value_head_dim, Some(128));
    assert_eq!(metadata.linear_num_key_heads, Some(16));
    assert_eq!(metadata.linear_num_value_heads, Some(32));
}

#[test]
fn qwen3_moe_manifest_uses_router_and_real_per_expert_tensors() {
    let metadata = parse_hf_config_metadata(qwen3_moe_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3Moe);
    assert_eq!(metadata.num_experts, Some(4));
    assert_eq!(metadata.num_experts_per_tok, Some(2));
    assert_eq!(metadata.moe_intermediate_size, Some(3));
    assert!(metadata.qk_norm);
    assert!(metadata.has_moe_layers());
    assert_eq!(
        metadata.mlp_layer_types,
        vec![HfMlpLayerKind::SparseMoe, HfMlpLayerKind::SparseMoe]
    );

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    assert_eq!(plan.blocks.len(), 25);
    assert_eq!(plan.static_weight_bytes, 264);
    assert_eq!(plan.per_layer_weight_bytes, 440);
    assert_eq!(plan.total_weight_bytes, 1144);

    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert_eq!(manifest.entries.len(), 45);
    let router = manifest
        .entries
        .iter()
        .find(|entry| entry.role == WeightBlockRole::RouterProjection && entry.layer == Some(0))
        .unwrap();
    assert_eq!(router.name, "model.layers.0.mlp.gate.weight");
    assert_eq!(router.rank, 2);
    assert_eq!((router.rows, router.cols), (4, 4));

    let gate = manifest
        .entries
        .iter()
        .find(|entry| {
            entry.role == WeightBlockRole::ExpertGateProjection
                && entry.layer == Some(0)
                && entry.expert == Some(0)
        })
        .unwrap();
    assert_eq!(gate.name, "model.layers.0.mlp.experts.0.gate_proj.weight");
    assert_eq!(gate.rank, 2);
    assert_eq!(gate.depth, None);
    assert_eq!((gate.rows, gate.cols), (3, 4));

    let up = manifest
        .entries
        .iter()
        .find(|entry| {
            entry.role == WeightBlockRole::ExpertUpProjection
                && entry.layer == Some(0)
                && entry.expert == Some(0)
        })
        .unwrap();
    assert_eq!(up.name, "model.layers.0.mlp.experts.0.up_proj.weight");
    assert_eq!(up.rank, 2);
    assert_eq!(up.depth, None);
    assert_eq!((up.rows, up.cols), (3, 4));

    let down = manifest
        .entries
        .iter()
        .find(|entry| {
            entry.role == WeightBlockRole::ExpertDownProjection
                && entry.layer == Some(0)
                && entry.expert == Some(0)
        })
        .unwrap();
    assert_eq!(down.name, "model.layers.0.mlp.experts.0.down_proj.weight");
    assert_eq!(down.rank, 2);
    assert_eq!(down.depth, None);
    assert_eq!((down.rows, down.cols), (4, 3));

    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    assert!(header.contains(
        "\"model.layers.0.mlp.experts.0.gate_proj.weight\":{\"dtype\":\"F16\",\"shape\":[3,4]"
    ));
    assert!(header.contains(
        "\"model.layers.0.mlp.experts.0.up_proj.weight\":{\"dtype\":\"F16\",\"shape\":[3,4]"
    ));
    assert!(header.contains(
        "\"model.layers.0.mlp.experts.0.down_proj.weight\":{\"dtype\":\"F16\",\"shape\":[4,3]"
    ));
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();
    assert_eq!(validation.validated_tensors, manifest.entries.len());
    assert_eq!(validation.total_data_bytes, manifest.total_weight_bytes);

    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();
}

#[test]
fn qwen3_moe_mlp_only_layers_remain_dense() {
    let metadata = parse_hf_config_metadata(
        r#"{
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 4,
            "intermediate_size": 8,
            "moe_intermediate_size": 3,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [1],
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 16,
            "torch_dtype": "float16"
        }"#,
    )
    .unwrap();

    assert_eq!(
        metadata.mlp_layer_types,
        vec![HfMlpLayerKind::SparseMoe, HfMlpLayerKind::Dense]
    );
    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.expert == Some(0)
            && entry.role == WeightBlockRole::ExpertGateProjection
            && entry.name == "model.layers.0.mlp.experts.0.gate_proj.weight"
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(1)
            && entry.role == WeightBlockRole::GateProjection
            && entry.name == "model.layers.1.mlp.gate_proj.weight"
    }));
}

#[test]
fn qwen3_moe_manifest_includes_shared_expert_tensors() {
    let metadata = parse_hf_config_metadata(
        r#"{
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 4,
            "intermediate_size": 8,
            "moe_intermediate_size": 3,
            "shared_expert_intermediate_size": 3,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 16,
            "torch_dtype": "float16"
        }"#,
    )
    .unwrap();
    assert_eq!(metadata.shared_expert_intermediate_size, Some(3));
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::SharedExpertGateProjection
            && entry.name == "model.layers.0.mlp.shared_expert.gate_proj.weight"
            && (entry.rows, entry.cols) == (3, 4)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::SharedExpertUpProjection
            && entry.name == "model.layers.0.mlp.shared_expert.up_proj.weight"
            && (entry.rows, entry.cols) == (3, 4)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::SharedExpertDownProjection
            && entry.name == "model.layers.0.mlp.shared_expert.down_proj.weight"
            && (entry.rows, entry.cols) == (4, 3)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::SharedExpertRouterProjection
            && entry.name == "model.layers.0.mlp.shared_expert_gate.weight"
            && (entry.rows, entry.cols) == (1, 4)
    }));
}

#[test]
fn qwen3_moe_contract_rejects_native_expert_limit_overflow() {
    let config = qwen3_moe_config().replace("\"num_experts\": 4", "\"num_experts\": 257");
    let metadata = parse_hf_config_metadata(&config).unwrap();

    let err = crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap_err();
    let NervaError::InvalidArgument { reason } = err else {
        panic!("expected invalid argument, got {err:?}");
    };
    assert!(reason.contains("num_experts 257"));
    assert!(reason.contains("limit 256"));
}

#[test]
fn qwen3_moe_contract_rejects_native_topk_limit_overflow() {
    let config = qwen3_moe_config()
        .replace("\"num_experts\": 4", "\"num_experts\": 32")
        .replace("\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": 17");
    let metadata = parse_hf_config_metadata(&config).unwrap();

    let err = crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap_err();
    let NervaError::InvalidArgument { reason } = err else {
        panic!("expected invalid argument, got {err:?}");
    };
    assert!(reason.contains("num_experts_per_tok 17"));
    assert!(reason.contains("top-k limit 16"));
}

#[test]
fn qwen3_5_moe_config_supports_gated_delta_net_moe_contract() {
    let metadata = parse_hf_config_metadata(qwen3_5_moe_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen35Moe);
    assert!(metadata.has_linear_attention_layers());
    assert!(metadata.has_moe_layers());
    assert_eq!(metadata.linear_conv_kernel_dim, Some(4));
    assert_eq!(metadata.linear_key_head_dim, Some(128));
    assert_eq!(metadata.linear_value_head_dim, Some(128));
    assert_eq!(metadata.linear_num_key_heads, Some(16));
    assert_eq!(metadata.linear_num_value_heads, Some(32));

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::LinearOutputProjection
            && entry.name == "model.language_model.layers.0.linear_attn.out_proj.weight"
            && (entry.rows, entry.cols) == (2560, 4096)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::LinearDtBias
            && entry.name == "model.language_model.layers.0.linear_attn.dt_bias"
            && entry.rows == 32
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.layer == Some(0)
            && entry.role == WeightBlockRole::LinearALog
            && entry.name == "model.language_model.layers.0.linear_attn.A_log"
            && entry.rows == 32
            && entry.dtype == DType::F32
    }));
    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    assert!(header.contains(
        "\"model.language_model.layers.0.linear_attn.conv1d.weight\":{\"dtype\":\"BF16\",\"shape\":[8192,1,4]"
    ));
    assert!(header.contains(
        "\"model.language_model.layers.0.linear_attn.in_proj_qkv.weight\":{\"dtype\":\"BF16\",\"shape\":[8192,2560]"
    ));
    assert!(header.contains(
        "\"model.language_model.layers.0.linear_attn.A_log\":{\"dtype\":\"F32\",\"shape\":[32]"
    ));
    assert!(header.contains(
        "\"model.language_model.layers.0.linear_attn.norm.weight\":{\"dtype\":\"F32\",\"shape\":[128]"
    ));
    assert!(header.contains(
        "\"model.language_model.layers.0.linear_attn.out_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[2560,4096]"
    ));
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();
    assert_eq!(validation.validated_tensors, manifest.entries.len());

    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();
}

#[test]
fn qwen3_5_moe_real_35b_a3b_config_uses_language_model_prefix() {
    let metadata = parse_hf_config_metadata(qwen3_5_moe_35b_a3b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen35Moe);
    assert_eq!(metadata.hidden_size, 2048);
    assert_eq!(metadata.intermediate_size, 512);
    assert_eq!(metadata.num_hidden_layers, 40);
    assert_eq!(metadata.num_experts, Some(256));
    assert_eq!(metadata.num_experts_per_tok, Some(8));
    assert_eq!(metadata.moe_intermediate_size, Some(512));
    assert_eq!(metadata.shared_expert_intermediate_size, Some(512));
    assert!(metadata.qk_norm);
    assert!(metadata.has_linear_attention_layers());
    assert!(metadata.has_moe_layers());
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert_eq!(manifest.entries.len(), 693);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.embed_tokens.weight"
            && entry.role == WeightBlockRole::TokenEmbedding
            && (entry.rows, entry.cols) == (248320, 2048)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            && entry.role == WeightBlockRole::LinearQkvProjection
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (8192, 2048)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
            && entry.role == WeightBlockRole::ExpertGateUpProjection
            && entry.layer == Some(0)
            && entry.depth == Some(256)
            && (entry.rows, entry.cols) == (1024, 2048)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.39.mlp.experts.down_proj"
            && entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(39)
            && entry.depth == Some(256)
            && (entry.rows, entry.cols) == (2048, 512)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.3.self_attn.q_norm.weight"
            && entry.role == WeightBlockRole::QueryNorm
            && entry.layer == Some(3)
            && entry.rows == 256
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.39.self_attn.k_norm.weight"
            && entry.role == WeightBlockRole::KeyNorm
            && entry.layer == Some(39)
            && entry.rows == 256
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.language_model.layers.0.mlp.shared_expert_gate.weight"
            && entry.role == WeightBlockRole::SharedExpertRouterProjection
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (1, 2048)
    }));
}

#[test]
fn qwen3_5_moe_gated_delta_net_state_shape_matches_vllm_formula() {
    let metadata = parse_hf_config_metadata(qwen3_5_moe_config()).unwrap();
    let spec = Qwen35GatedDeltaNetSpec::from_metadata(&metadata)
        .unwrap()
        .unwrap();

    assert_eq!(spec.conv_dim().unwrap(), 8192);

    let single_gpu = spec
        .state_shape(1, 0, ConvStateLayout::StateLenDim)
        .unwrap();
    assert_eq!(single_gpu.conv_state, (3, 8192));
    assert_eq!(single_gpu.recurrent_state, (32, 128, 128));
    assert_eq!(single_gpu.conv_elements(), 24_576);
    assert_eq!(single_gpu.recurrent_elements(), 524_288);
    assert_eq!(single_gpu.total_elements(), 548_864);

    let tp2 = spec
        .state_shape(2, 0, ConvStateLayout::StateLenDim)
        .unwrap();
    assert_eq!(tp2.conv_state, (3, 4096));
    assert_eq!(tp2.recurrent_state, (16, 128, 128));

    let dim_first = spec
        .state_shape(1, 2, ConvStateLayout::DimStateLen)
        .unwrap();
    assert_eq!(dim_first.conv_state, (8192, 5));
    assert_eq!(dim_first.recurrent_state, (32, 128, 128));
}

#[test]
fn precision_gated_delta_net_moe_block_validates_and_exposes_view() {
    let block = precision_gdn_moe_block_with_linear_out(12).unwrap();
    let view = block.encoded_view();

    assert_eq!(view.dtype, DType::BF16);
    assert_eq!(view.shape.hidden, 4);
    assert_eq!(view.gdn.conv_dim().unwrap(), 7);
    assert_eq!(view.gdn.conv_kernel, 4);
    assert_eq!(view.moe.num_experts, 4);
    assert_eq!(view.moe.experts_per_token, 2);
    assert!(view.moe.norm_topk_prob);
    assert_eq!(view.linear_conv.len(), 28);
    assert_eq!(view.linear_qkv.len(), 28);
    assert_eq!(view.linear_z.len(), 12);
    assert_eq!(view.linear_norm.len(), 3);
    assert_eq!(view.linear_norm_bits.len(), 6);
    assert_eq!(view.linear_out.len(), 12);
    assert_eq!(view.router.len(), 16);
    assert_eq!(view.expert_gate_up.len(), 96);
    assert_eq!(view.expert_down.len(), 48);
    assert_eq!(view.shared_expert_router.len(), 4);
    assert_eq!(view.rms_eps, 1e-5);
}

#[test]
fn precision_gated_delta_net_moe_block_rejects_bad_linear_shape() {
    let err = precision_gdn_moe_block_with_linear_out(11).unwrap_err();
    let reason = format!("{err:?}");
    assert!(reason.contains("GatedDeltaNet output projection"));
    assert!(reason.contains("expected 12"));
}

#[test]
fn qwen3_5_moe_architecture_requires_moe_metadata() {
    let err = parse_hf_config_metadata(
        r#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "dtype": "bfloat16",
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "layer_types": ["full_attention"],
                "model_type": "qwen3_5_moe_text",
                "num_attention_heads": 16,
                "num_hidden_layers": 1,
                "num_key_value_heads": 4,
                "vocab_size": 248320
            }
        }"#,
    )
    .unwrap_err();

    let NervaError::InvalidArgument { reason } = err else {
        panic!("expected invalid argument, got {err:?}");
    };
    assert!(reason.contains("missing required field num_experts"));
}

#[test]
fn qwen3_5_moe_full_attention_layout_uses_packed_expert_tensors() {
    let config = qwen3_5_moe_config().replace("\"linear_attention\"", "\"full_attention\"");
    let metadata = parse_hf_config_metadata(&config).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen35Moe);
    assert!(!metadata.has_linear_attention_layers());
    assert!(metadata.has_moe_layers());

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    let query = manifest
        .entries
        .iter()
        .find(|entry| entry.role == WeightBlockRole::QueryProjection && entry.layer == Some(0))
        .unwrap();
    assert_eq!(
        query.name,
        "model.language_model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!((query.rows, query.cols), (8192, 2560));

    let gate_up = manifest
        .entries
        .iter()
        .find(|entry| {
            entry.role == WeightBlockRole::ExpertGateUpProjection && entry.layer == Some(0)
        })
        .unwrap();
    assert_eq!(
        gate_up.name,
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    );
    assert_eq!(gate_up.rank, 3);
    assert_eq!(gate_up.depth, Some(128));
    assert_eq!((gate_up.rows, gate_up.cols), (1536, 2560));

    let down = manifest
        .entries
        .iter()
        .find(|entry| entry.role == WeightBlockRole::ExpertDownProjection && entry.layer == Some(0))
        .unwrap();
    assert_eq!(
        down.name,
        "model.language_model.layers.0.mlp.experts.down_proj"
    );
    assert_eq!(down.rank, 3);
    assert_eq!(down.depth, Some(128));
    assert_eq!((down.rows, down.cols), (2560, 768));

    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();
}

#[test]
fn qwen3_moe_real_30b_a3b_config_uses_split_expert_manifest() {
    let metadata = parse_hf_config_metadata(qwen3_moe_30b_a3b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3Moe);
    assert_eq!(metadata.hidden_size, 2048);
    assert_eq!(metadata.num_hidden_layers, 48);
    assert_eq!(metadata.num_experts, Some(128));
    assert_eq!(metadata.num_experts_per_tok, Some(8));
    assert_eq!(metadata.moe_intermediate_size, Some(768));
    assert!(metadata.qk_norm);
    assert!(metadata.has_moe_layers());
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert_eq!(manifest.entries.len(), 18_867);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.mlp.experts.0.gate_proj.weight"
            && entry.role == WeightBlockRole::ExpertGateProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && (entry.rows, entry.cols) == (768, 2048)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.47.mlp.experts.127.down_proj.weight"
            && entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(47)
            && entry.expert == Some(127)
            && (entry.rows, entry.cols) == (2048, 768)
    }));
}

#[test]
fn qwen3_coder_30b_a3b_config_uses_split_expert_manifest() {
    let metadata = parse_hf_config_metadata(qwen3_coder_30b_a3b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3Moe);
    assert_eq!(metadata.hidden_size, 2048);
    assert_eq!(metadata.num_hidden_layers, 48);
    assert_eq!(metadata.num_experts, Some(128));
    assert_eq!(metadata.num_experts_per_tok, Some(8));
    assert_eq!(metadata.moe_intermediate_size, Some(768));
    assert_eq!(metadata.max_position_embeddings, Some(262144));
    assert!(metadata.qk_norm);
    assert!(metadata.has_moe_layers());
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert_eq!(manifest.entries.len(), 18_867);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.mlp.experts.0.gate_proj.weight"
            && entry.role == WeightBlockRole::ExpertGateProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && (entry.rows, entry.cols) == (768, 2048)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.47.mlp.experts.127.down_proj.weight"
            && entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(47)
            && entry.expert == Some(127)
            && (entry.rows, entry.cols) == (2048, 768)
    }));
}

#[test]
fn qwen3_moe_real_235b_a22b_config_uses_split_expert_manifest() {
    let metadata = parse_hf_config_metadata(qwen3_moe_235b_a22b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3Moe);
    assert_eq!(metadata.hidden_size, 4096);
    assert_eq!(metadata.num_hidden_layers, 94);
    assert_eq!(metadata.num_experts, Some(128));
    assert_eq!(metadata.num_experts_per_tok, Some(8));
    assert_eq!(metadata.moe_intermediate_size, Some(1536));
    assert!(metadata.qk_norm);
    assert!(metadata.has_moe_layers());
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert_eq!(manifest.entries.len(), 36_945);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.mlp.experts.0.up_proj.weight"
            && entry.role == WeightBlockRole::ExpertUpProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && (entry.rows, entry.cols) == (1536, 4096)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.93.mlp.experts.127.down_proj.weight"
            && entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(93)
            && entry.expert == Some(127)
            && (entry.rows, entry.cols) == (4096, 1536)
    }));
}

#[test]
fn qwen2_moe_real_a27b_config_uses_split_and_shared_expert_manifest() {
    let metadata = parse_hf_config_metadata(qwen2_moe_a27b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen2Moe);
    assert_eq!(metadata.hidden_size, 2048);
    assert_eq!(metadata.num_hidden_layers, 24);
    assert_eq!(metadata.num_experts, Some(60));
    assert_eq!(metadata.num_experts_per_tok, Some(4));
    assert_eq!(metadata.moe_intermediate_size, Some(1408));
    assert_eq!(metadata.shared_expert_intermediate_size, Some(5632));
    assert!(metadata.attention_bias);
    assert!(metadata.attention_qkv_bias);
    assert!(!metadata.attention_output_bias);
    assert!(!metadata.qk_norm);
    assert!(metadata.has_moe_layers());
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    assert_eq!(plan.blocks.len(), 387);
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert_eq!(manifest.entries.len(), 4_659);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.self_attn.q_proj.bias"
            && entry.role == WeightBlockRole::QueryBias
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (2048, 1)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.self_attn.k_proj.bias"
            && entry.role == WeightBlockRole::KeyBias
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (2048, 1)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.self_attn.v_proj.bias"
            && entry.role == WeightBlockRole::ValueBias
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (2048, 1)
    }));
    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.name == "model.layers.0.self_attn.o_proj.bias")
    );
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.mlp.experts.0.gate_proj.weight"
            && entry.role == WeightBlockRole::ExpertGateProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && (entry.rows, entry.cols) == (1408, 2048)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.23.mlp.experts.59.down_proj.weight"
            && entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(23)
            && entry.expert == Some(59)
            && (entry.rows, entry.cols) == (2048, 1408)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.mlp.shared_expert_gate.weight"
            && entry.role == WeightBlockRole::SharedExpertRouterProjection
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (1, 2048)
    }));
}

#[test]
fn mixtral_real_8x7b_config_uses_block_sparse_moe_manifest() {
    let metadata = parse_hf_config_metadata(mixtral_8x7b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::MixtralMoe);
    assert_eq!(metadata.hidden_size, 4096);
    assert_eq!(metadata.num_hidden_layers, 32);
    assert_eq!(metadata.num_experts, Some(8));
    assert_eq!(metadata.num_experts_per_tok, Some(2));
    assert_eq!(metadata.moe_intermediate_size, Some(14336));
    assert_eq!(metadata.shared_expert_intermediate_size, None);
    assert!(!metadata.qk_norm);
    assert!(metadata.has_moe_layers());
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    assert_eq!(plan.blocks.len(), 291);
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert_eq!(manifest.entries.len(), 995);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.block_sparse_moe.gate.weight"
            && entry.role == WeightBlockRole::RouterProjection
            && entry.layer == Some(0)
            && (entry.rows, entry.cols) == (8, 4096)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
            && entry.role == WeightBlockRole::ExpertGateProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && (entry.rows, entry.cols) == (14336, 4096)
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.31.block_sparse_moe.experts.7.w2.weight"
            && entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(31)
            && entry.expert == Some(7)
            && (entry.rows, entry.cols) == (4096, 14336)
    }));
}

#[test]
fn qwen3_coder_480b_a35b_config_uses_explicit_use_qk_norm() {
    let metadata = parse_hf_config_metadata(qwen3_coder_480b_a35b_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3Moe);
    assert_eq!(metadata.hidden_size, 6144);
    assert_eq!(metadata.num_hidden_layers, 62);
    assert_eq!(metadata.num_experts, Some(160));
    assert_eq!(metadata.num_experts_per_tok, Some(8));
    assert_eq!(metadata.moe_intermediate_size, Some(2560));
    assert!(metadata.qk_norm);
    crate::hf::contract::validate_exact_runtime_contract(&metadata).unwrap();

    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();
    assert_eq!(manifest.entries.len(), 30_321);
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.0.self_attn.q_norm.weight"
            && entry.role == WeightBlockRole::QueryNorm
            && entry.rows == 128
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.name == "model.layers.61.mlp.experts.159.up_proj.weight"
            && entry.role == WeightBlockRole::ExpertUpProjection
            && entry.layer == Some(61)
            && entry.expert == Some(159)
            && (entry.rows, entry.cols) == (2560, 6144)
    }));
}

fn precision_gdn_moe_block_with_linear_out(
    linear_out_len: usize,
) -> nerva_core::types::error::Result<PrecisionGatedDeltaNetMoeBlock> {
    let shape = TransformerBlockShape::new_with_kv_heads_and_head_dim(4, 2, 1, 2, 8);
    let gdn = PrecisionGatedDeltaNetConfig {
        key_heads: 1,
        value_heads: 1,
        key_head_dim: 2,
        value_head_dim: 3,
        conv_kernel: 4,
    };
    let moe = PrecisionMoeConfig {
        moe_intermediate: 3,
        shared_expert_intermediate: 2,
        num_experts: 4,
        experts_per_token: 2,
        norm_topk_prob: true,
    };
    PrecisionGatedDeltaNetMoeBlock::new_from_encoded(
        DType::BF16,
        shape,
        gdn,
        moe,
        u16_values(4),
        u16_values(28),
        u16_values(28),
        u16_values(12),
        u16_values(4),
        u16_values(4),
        u16_values(1),
        vec![0.0],
        vec![1.0, 1.0, 1.0],
        u16_values(linear_out_len),
        u16_values(4),
        u16_values(16),
        u16_values(96),
        u16_values(48),
        u16_values(8),
        u16_values(8),
        u16_values(8),
        u16_values(4),
        1e-5,
    )
}

fn u16_values(len: usize) -> Vec<u16> {
    (0..len).map(|value| value as u16).collect()
}

fn qwen3_5_moe_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": false,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 768,
            "norm_topk_prob": true,
            "num_attention_heads": 16,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 0.000001,
            "shared_expert_intermediate_size": 0,
            "tie_word_embeddings": true,
            "use_qk_norm": true,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000
            }
        },
        "tie_word_embeddings": true
    }"#
}

fn qwen3_5_moe_35b_a3b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": false,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "rms_norm_eps": 0.000001,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000
            }
        },
        "tie_word_embeddings": false
    }"#
}

fn qwen3_5_4b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": false,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 0.000001,
            "tie_word_embeddings": true,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000
            }
        },
        "tie_word_embeddings": true
    }"#
}

fn qwen3_5_hybrid_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": false,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 0.000001,
            "tie_word_embeddings": true,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000
            }
        },
        "tie_word_embeddings": true
    }"#
}

fn qwen3_moe_30b_a3b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": false,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "max_position_embeddings": 40960,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "vocab_size": 151936
    }"#
}

fn qwen3_coder_30b_a3b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "vocab_size": 151936
    }"#
}

fn qwen3_moe_235b_a22b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 40960,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 1536,
        "norm_topk_prob": true,
        "num_attention_heads": 64,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 94,
        "num_key_value_heads": 4,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "vocab_size": 151936
    }"#
}

fn qwen3_coder_480b_a35b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "intermediate_size": 8192,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 2560,
        "norm_topk_prob": true,
        "num_attention_heads": 96,
        "num_experts": 160,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 62,
        "num_key_value_heads": 8,
        "qkv_bias": false,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000000,
        "shared_expert_intermediate_size": 0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_qk_norm": true,
        "vocab_size": 151936
    }"#
}

fn qwen2_moe_a27b_config() -> &'static str {
    r#"{
        "architectures": ["Qwen2MoeForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "max_position_embeddings": 8192,
        "max_window_layers": 21,
        "model_type": "qwen2_moe",
        "moe_intermediate_size": 1408,
        "norm_topk_prob": false,
        "num_attention_heads": 16,
        "num_experts": 60,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 24,
        "num_key_value_heads": 16,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000.0,
        "shared_expert_intermediate_size": 5632,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#
}

fn mixtral_8x7b_config() -> &'static str {
    r#"{
        "architectures": ["MixtralForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mixtral",
        "num_attention_heads": 32,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "rms_norm_eps": 0.00001,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "vocab_size": 32000
    }"#
}

fn qwen3_moe_config() -> &'static str {
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
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "hidden_act": "silu",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "mlp_bias": false,
        "torch_dtype": "float16"
    }"#
}

fn qwen3_dense_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 16,
        "hidden_act": "silu",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "mlp_bias": false,
        "torch_dtype": "bfloat16"
    }"#
}
