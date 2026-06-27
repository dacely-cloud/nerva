use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmModel;

use crate::engine::hf_cuda_decode::run::run_loaded_hf_causal_lm_cuda_prompt_decode;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::hf_fixture::{remove_hf_checkpoint_dir, write_cycle_hf_checkpoint_dir};

#[test]
fn loaded_hf_cuda_decode_passes_resident_weight_contract_to_native() {
    let dir = write_cycle_hf_checkpoint_dir("nerva-hf-cuda-contract", 1);
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let prompt = [TokenId(0), TokenId(1)];
    let summary =
        run_loaded_hf_causal_lm_cuda_prompt_decode(&runtime, &loaded, &prompt, 3, Some(120))
            .unwrap();
    remove_hf_checkpoint_dir(&dir);

    if summary.status != nerva_cuda::smoke::status::SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.resident_weights.plan_steps, 12);
    assert_eq!(
        summary.resident_weights.cuda_contract_blocks,
        summary.resident_weights.plan_steps,
    );
    assert_eq!(
        summary.resident_weights.cuda_contract_weight_bytes,
        summary.resident_weights.plan_weight_bytes,
    );
    assert_eq!(
        summary.resident_weights.cuda_contract_descriptor_blocks,
        summary.resident_weights.plan_descriptor_blocks,
    );
    assert_eq!(
        summary.resident_weights.cuda_contract_descriptor_hash,
        summary.resident_weights.plan_descriptor_hash,
    );
    assert!(summary.resident_weights.cuda_contract_matched);
    assert!(summary.resident_weights.plan_gpu_resident_weight_bytes > 0);
    assert!(summary.resident_weights.plan_gpu_staged_weight_bytes > 0);
    assert_eq!(summary.resident_weights.plan_fallback_weight_bytes, 0);
}
