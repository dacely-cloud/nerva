use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmModel;
use nerva_runtime::engine::hf_cuda_decode::run::run_loaded_hf_causal_lm_cuda_prompt_decode;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::acceptance::cuda::hf_kv::{remove_checkpoint_dir, write_checkpoint_dir};
use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_loaded_hf_prompt_kv_decode(report: &mut AcceptanceReport) {
    let dir = write_checkpoint_dir();
    let prompt = [TokenId(0), TokenId(1)];
    let summary = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))
        .and_then(|runtime| {
            HfCausalLmModel::load_from_hf_dir(&dir)
                .map_err(|err| format!("failed to load HF prompt fixture: {err:?}"))
                .and_then(|loaded| {
                    run_loaded_hf_causal_lm_cuda_prompt_decode(&runtime, &loaded, &prompt, 3, None)
                        .map_err(|err| {
                            format!("failed to execute HF prompt decode on CUDA: {err:?}")
                        })
                })
        });
    remove_checkpoint_dir(&dir);

    match summary {
        Ok(summary) => report.push(
            "cuda_loaded_hf_prompt_kv_decode",
            summary.passed()
                && summary.tokens == summary.expected_tokens
                && summary.graph_replays == 4
                && summary.graph_replay_events == summary.steps_requested as u64
                && summary.resident_kv_bytes > 0
                && summary.kv_tokens == 4
                && summary.resident_weights.plan_steps == 12
                && summary.resident_weights.run_steps == 12
                && summary.resident_weights.hotset_promoted_blocks > 0
                && summary.resident_weights.hotset_kept_dram_blocks > 0
                && summary.resident_weights.plan_gpu_resident_steps > 0
                && summary.resident_weights.plan_gpu_staged_steps > 0
                && summary.resident_weights.plan_gpu_resident_weight_bytes > 0
                && summary.resident_weights.plan_gpu_staged_weight_bytes > 0
                && summary.resident_weights.plan_fallback_weight_bytes == 0
                && summary.resident_weights.cuda_contract_blocks
                    == summary.resident_weights.plan_steps
                && summary.resident_weights.cuda_contract_weight_bytes
                    == summary.resident_weights.plan_weight_bytes
                && summary.resident_weights.cuda_contract_descriptor_blocks
                    == summary.resident_weights.plan_descriptor_blocks
                && summary.resident_weights.cuda_contract_descriptor_hash
                    == summary.resident_weights.plan_descriptor_hash
                && summary.resident_weights.cuda_contract_matched
                && summary.resident_weights.run_gpu_resident_steps
                    == summary.resident_weights.plan_gpu_resident_steps
                && summary.resident_weights.run_gpu_staged_steps
                    == summary.resident_weights.plan_gpu_staged_steps
                && summary.host_causality_edges == 0
                && summary.hot_path_allocations == 0,
            format!(
                "status={:?} steps={} parity={} tokens={} expected={} graph_replays={} graph_replay_events={} resident_kv_bytes={} kv_tokens={} hotset_promoted={} hotset_kept_dram={} plan_steps={} run_steps={} plan_gpu_resident={} plan_gpu_staged={} contract_blocks={} contract_bytes={} descriptor_blocks={} descriptor_hash={} contract_matched={} run_gpu_resident={} run_gpu_staged={} host_causality_edges={} hot_path_allocations={} output_hash={} expected_hash={} error={}",
                summary.status,
                summary.steps_requested,
                summary.parity,
                summary.tokens.len(),
                summary.expected_tokens.len(),
                summary.graph_replays,
                summary.graph_replay_events,
                summary.resident_kv_bytes,
                summary.kv_tokens,
                summary.resident_weights.hotset_promoted_blocks,
                summary.resident_weights.hotset_kept_dram_blocks,
                summary.resident_weights.plan_steps,
                summary.resident_weights.run_steps,
                summary.resident_weights.plan_gpu_resident_steps,
                summary.resident_weights.plan_gpu_staged_steps,
                summary.resident_weights.cuda_contract_blocks,
                summary.resident_weights.cuda_contract_weight_bytes,
                summary.resident_weights.cuda_contract_descriptor_blocks,
                summary.resident_weights.cuda_contract_descriptor_hash,
                summary.resident_weights.cuda_contract_matched,
                summary.resident_weights.run_gpu_resident_steps,
                summary.resident_weights.run_gpu_staged_steps,
                summary.host_causality_edges,
                summary.hot_path_allocations,
                summary.output_hash,
                summary.expected_hash,
                summary.error.as_deref().unwrap_or("none"),
            ),
        ),
        Err(err) => report.push("cuda_loaded_hf_prompt_kv_decode", false, err),
    }
}
