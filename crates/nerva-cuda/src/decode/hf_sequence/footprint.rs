use crate::decode::hf_sequence::request::CudaHfDecodeSequenceRequest;
use crate::decode::hf_sequence::weight_plan::CudaHfDecodeSequenceWeightPlan;

const U16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const LAYER_LAYOUT_BYTES: u64 = 13 * 8;
const TOKEN_SLOT_BYTES: u64 = 40;
const DESCRIPTOR_STREAM_STAGING_BYTES: u64 = 64 * 1024 * 1024;
const KV_CACHE_BLOCK_TOKENS: u64 = 16;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceFootprint {
    pub context_tokens: u64,
    pub resident_weight_bytes: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub layout_bytes: u64,
    pub scratch_bytes: u64,
    pub resident_kv_bytes: u64,
    pub token_slot_bytes: u64,
    pub prompt_bytes: u64,
}

impl CudaHfDecodeSequenceFootprint {
    pub fn to_json(self) -> String {
        format!(
            "{{\"context_tokens\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"layout_bytes\":{},\"scratch_bytes\":{},\"resident_kv_bytes\":{},\"token_slot_bytes\":{},\"prompt_bytes\":{}}}",
            self.context_tokens,
            self.resident_weight_bytes,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.layout_bytes,
            self.scratch_bytes,
            self.resident_kv_bytes,
            self.token_slot_bytes,
            self.prompt_bytes,
        )
    }
}

pub fn estimate_sequence_footprint(
    request: &CudaHfDecodeSequenceRequest<'_>,
) -> Result<CudaHfDecodeSequenceFootprint, String> {
    let hidden = as_u64("hidden", request.hidden)?;
    let heads = as_u64("heads", request.heads)?;
    let kv_heads = as_u64("KV heads", request.kv_heads)?;
    let head_dim = as_u64("head dim", request.head_dim)?;
    let intermediate = as_u64("intermediate", request.intermediate)?;
    let vocab_size = as_u64("vocab size", request.vocab_size)?;
    let layer_count = as_u64("layer count", request.layers.len())?;
    let prompt_count = as_u64("prompt token count", request.prompt_tokens.len())?;
    let steps = as_u64("steps", request.steps)?;
    let context_tokens = checked_add(prompt_count, steps, "context token count")?
        .checked_sub(1)
        .ok_or_else(|| "CUDA HF decode context token count underflow".to_string())?;
    let attention_hidden = checked_mul(heads, head_dim, "attention hidden")?;
    let kv_hidden = checked_mul(kv_heads, head_dim, "KV hidden")?;
    let arena_elements = arena_elements(
        request,
        hidden,
        attention_hidden,
        kv_hidden,
        head_dim,
        intermediate,
    )?;
    let arena_bytes = checked_mul(arena_elements, U16_BYTES, "arena bytes")?;
    let scratch_gap_bytes = checked_mul(hidden, U16_BYTES * 2, "scratch gap bytes")?;
    let resident_weight_bytes = arena_bytes
        .checked_sub(scratch_gap_bytes)
        .ok_or_else(|| "CUDA HF decode resident weight byte underflow".to_string())?;
    if request
        .weight_plan
        .is_some_and(|plan| plan.is_declared() && plan.weight_bytes != resident_weight_bytes)
    {
        return Err("CUDA HF decode declared weight bytes do not match packed layout".to_string());
    }
    let layout_bytes = checked_mul(layer_count, LAYER_LAYOUT_BYTES, "layout bytes")?;
    let scratch_bytes = scratch_bytes(
        hidden,
        attention_hidden,
        kv_hidden,
        intermediate,
        vocab_size,
    )?;
    let kv_block_count = context_tokens.div_ceil(KV_CACHE_BLOCK_TOKENS);
    let kv_token_capacity = checked_mul(
        kv_block_count,
        KV_CACHE_BLOCK_TOKENS,
        "KV token capacity",
    )?;
    let resident_kv_bytes = checked_mul(
        checked_mul(layer_count, kv_token_capacity, "KV layer tokens")?,
        checked_mul(kv_hidden, U16_BYTES * 2, "KV token bytes")?,
        "resident KV bytes",
    )?;
    let kv_block_table_bytes = checked_mul(kv_block_count, 4, "KV block table bytes")?;
    let token_slot_bytes = checked_mul(context_tokens, TOKEN_SLOT_BYTES, "token slot bytes")?;
    let prompt_bytes = checked_mul(prompt_count, 4, "prompt bytes")?;
    let device_arena_bytes = sum_bytes(&[
        arena_bytes,
        layout_bytes,
        scratch_bytes,
        resident_kv_bytes,
        kv_block_table_bytes,
        prompt_bytes,
        token_slot_bytes,
        4,
    ])?;
    let host_weight_bytes = if request.weight_plan.is_some_and(|plan| plan.is_declared()) {
        descriptor_host_staging_bytes(request, resident_weight_bytes)
    } else {
        arena_bytes
    };
    Ok(CudaHfDecodeSequenceFootprint {
        context_tokens,
        resident_weight_bytes,
        device_arena_bytes,
        pinned_host_bytes: checked_add(host_weight_bytes, token_slot_bytes, "pinned bytes")?,
        layout_bytes,
        scratch_bytes,
        resident_kv_bytes,
        token_slot_bytes,
        prompt_bytes,
    })
}

fn descriptor_host_staging_bytes(
    request: &CudaHfDecodeSequenceRequest<'_>,
    resident_weight_bytes: u64,
) -> u64 {
    if request
        .weight_blocks
        .iter()
        .any(|block| !block.source_file.is_null() && block.source_file_len != 0)
    {
        resident_weight_bytes.min(DESCRIPTOR_STREAM_STAGING_BYTES)
    } else {
        U16_BYTES
    }
}

fn arena_elements(
    request: &CudaHfDecodeSequenceRequest<'_>,
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    intermediate: u64,
) -> Result<u64, String> {
    let mut elements = checked_mul(as_u64("vocab", request.vocab_size)?, hidden, "embeddings")?;
    elements = checked_add(elements, hidden, "input buffer")?;
    elements = checked_add(elements, hidden, "scratch buffer")?;
    let declared_weight_plan = request
        .weight_plan
        .is_some_and(CudaHfDecodeSequenceWeightPlan::is_declared);
    for layer in request.layers {
        elements = checked_add(
            elements,
            crate::decode::hf_sequence::footprint_layers::layer_elements(
                layer,
                hidden,
                attention_hidden,
                kv_hidden,
                head_dim,
                intermediate,
                declared_weight_plan,
            )?,
            "layer weights",
        )?;
    }
    elements = checked_add(elements, hidden, "final norm")?;
    checked_add(
        elements,
        checked_mul(as_u64("vocab", request.vocab_size)?, hidden, "LM head")?,
        "arena elements",
    )
}

fn scratch_bytes(
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    intermediate: u64,
    vocab_size: u64,
) -> Result<u64, String> {
    let block = hidden * 5 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 3;
    let final_pass = hidden * 2 + vocab_size;
    checked_mul(block.max(final_pass), F32_BYTES, "scratch bytes")
}

fn sum_bytes(values: &[u64]) -> Result<u64, String> {
    values
        .iter()
        .try_fold(0u64, |sum, value| checked_add(sum, *value, "byte sum"))
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("CUDA HF decode {label} overflow"))
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("CUDA HF decode {label} overflow"))
}

fn as_u64(label: &str, value: usize) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("CUDA HF decode {label} does not fit u64"))
}
