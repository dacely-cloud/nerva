use nerva_core::types::id::token::TokenId;
use nerva_runtime::engine::hf_cuda_decode::file_backed::run::run_hf_causal_lm_cuda_shard_backed_device_only;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::cli::model::causal_lm_cuda_json::{HfCudaDecodeJson, hf_cuda_decode_json};
use crate::cli::model::causal_lm_text::generated_text_json;

pub(crate) fn hf_causal_lm_cuda_device_only_with_tokens_json(
    path: String,
    input_mode: &'static str,
    prompt_token_ids: Vec<u32>,
    prompt_text: Option<String>,
    steps: usize,
) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let prompt_tokens = prompt_token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let output = run_hf_causal_lm_cuda_shard_backed_device_only(
        &runtime,
        &path,
        &prompt_tokens,
        steps,
        None,
    )
    .map_err(|err| format!("HF CUDA causal LM decode failed: {err:?}"))?;
    let dtype = nerva_model::precision::bits::dtype_label(output.dtype)
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let generated_text = generated_text_json(&path, &output.summary.tokens)?;
    Ok(hf_cuda_decode_json(HfCudaDecodeJson {
        path: &path,
        input_mode,
        prompt_text: prompt_text.as_deref(),
        prompt_token_ids: &prompt_token_ids,
        prompt_tokens_len: prompt_tokens.len(),
        seed_token: prompt_tokens.last().map(|token| token.0).unwrap_or(0),
        steps,
        dtype,
        layers: output.metadata.num_hidden_layers,
        hidden: output.metadata.hidden_size,
        vocab_size: output.metadata.vocab_size,
        manifest_entries: output.manifest_entries,
        shard_plan_entries: output.shard_plan_entries,
        tensors_loaded: output.tensors_loaded,
        bytes_loaded: output.bytes_loaded,
        data_hash: output.data_hash,
        data_hash_available: output.data_hash_available,
        generated_text: &generated_text,
        summary: &output.summary,
    }))
}
