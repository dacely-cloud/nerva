use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmModel;
use nerva_model::hf::architecture::HfArchitectureKind;
use nerva_model::weights::layout::entry::WeightBlockRole;

use crate::engine::hf_cuda_decode::run::run_loaded_hf_causal_lm_cuda_prompt_decode;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::hf_fixture::{remove_hf_checkpoint_dir, write_qwen3_hf_checkpoint_dir};

#[test]
fn cuda_loaded_qwen3_prompt_decode_consumes_qk_norm_weights() {
    let dir = write_qwen3_hf_checkpoint_dir("nerva-hf-cuda-qwen3");
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let prompt = [TokenId(0)];
    let summary = run_loaded_hf_causal_lm_cuda_prompt_decode(&runtime, &loaded, &prompt, 4, None);
    remove_hf_checkpoint_dir(&dir);

    let summary = match summary {
        Ok(summary) => summary,
        Err(_) => return,
    };
    assert_eq!(
        loaded.model.metadata().architecture,
        HfArchitectureKind::Qwen3
    );
    assert!(loaded.model.metadata().qk_norm);
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::QueryNorm
            && entry.name == "model.layers.0.self_attn.q_norm.weight"
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::KeyNorm
            && entry.name == "model.layers.0.self_attn.k_norm.weight"
    }));
    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_eq!(summary.host_causality_edges, 0);
    assert_eq!(summary.resident_weights.plan_descriptor_blocks, 14);
}
