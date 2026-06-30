use crate::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_DEEPSEEK_MLA, CUDA_HF_ATTENTION_LINEAR_GDN, CudaHfDecodeChainLayer,
};
use crate::decode::hf_sequence::request::CudaHfDecodeSequenceRequest;
use crate::decode::hf_sequence::weight_plan::CudaHfDecodeSequenceWeightPlan;

const U16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const LAYER_LAYOUT_BYTES: u64 = 584;
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
    let kv_cache_width = max_kv_cache_width(request.layers, kv_hidden)?;
    let scratch_bytes = scratch_bytes(
        request.layers,
        hidden,
        attention_hidden,
        kv_hidden,
        head_dim,
        intermediate,
        vocab_size,
    )?;
    let linear_gdn_state_bytes = linear_gdn_state_bytes(request.layers)?;
    let kv_block_count = context_tokens.div_ceil(KV_CACHE_BLOCK_TOKENS);
    let kv_token_capacity =
        checked_mul(kv_block_count, KV_CACHE_BLOCK_TOKENS, "KV token capacity")?;
    let resident_kv_bytes = checked_mul(
        checked_mul(layer_count, kv_token_capacity, "KV layer tokens")?,
        checked_mul(kv_cache_width, U16_BYTES * 2, "KV token bytes")?,
        "resident KV bytes",
    )?;
    let kv_block_table_bytes = checked_mul(kv_block_count, 4, "KV block table bytes")?;
    let token_slot_bytes = checked_mul(context_tokens, TOKEN_SLOT_BYTES, "token slot bytes")?;
    let prompt_bytes = checked_mul(prompt_count, 4, "prompt bytes")?;
    let device_arena_bytes = sum_bytes(&[
        arena_bytes,
        layout_bytes,
        scratch_bytes,
        linear_gdn_state_bytes,
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
    elements = checked_add(
        elements,
        crate::decode::hf_sequence::footprint_layers::deepseek_static_elements(
            request.layers,
            hidden,
        )?,
        "DeepSeek static weights",
    )?;
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
                as_u64("vocab", request.vocab_size)?,
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
    layers: &[CudaHfDecodeChainLayer<'_>],
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    intermediate: u64,
    vocab_size: u64,
) -> Result<u64, String> {
    let block = layers.iter().try_fold(
        full_attention_scratch_elements(hidden, attention_hidden, kv_hidden, intermediate)?,
        |max_scratch, layer| -> Result<u64, String> {
            Ok(max_scratch.max(layer_scratch_elements(
                layer,
                hidden,
                attention_hidden,
                kv_hidden,
                head_dim,
                intermediate,
            )?))
        },
    )?;
    let final_pass = hidden * 2 + vocab_size;
    checked_mul(block.max(final_pass), F32_BYTES, "scratch bytes")
}

fn layer_scratch_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    intermediate: u64,
) -> Result<u64, String> {
    if let Some(deepseek) = layer
        .deepseek
        .filter(|_| layer.attention_kind == CUDA_HF_ATTENTION_DEEPSEEK_MLA)
        .and_then(|deepseek| deepseek.v3_mla_shape((attention_hidden / head_dim) as usize))
    {
        let attention_rows = deepseek
            .q_rows
            .max(deepseek.kv_b_rows)
            .max(deepseek.value_rows)
            .max(attention_hidden as usize) as u64;
        return full_attention_scratch_elements(
            hidden,
            attention_rows,
            deepseek.kv_cache_width as u64,
            intermediate,
        );
    }
    if layer.attention_kind != CUDA_HF_ATTENTION_LINEAR_GDN {
        return full_attention_scratch_elements(hidden, attention_hidden, kv_hidden, intermediate);
    }
    let Some(gdn) = layer.linear_gdn else {
        return Err("CUDA HF decode linear GDN layer is missing layout metadata".to_string());
    };
    let value_dim = checked_mul(
        gdn.value_heads as u64,
        gdn.value_head_dim as u64,
        "GDN value scratch dim",
    )?;
    let key_dim = checked_mul(
        gdn.key_heads as u64,
        gdn.key_head_dim as u64,
        "GDN key scratch dim",
    )?;
    let conv_dim = checked_add(
        checked_mul(key_dim, 2, "GDN conv scratch key dim")?,
        value_dim,
        "GDN conv scratch dim",
    )?;
    sum_bytes(&[
        checked_mul(hidden, 5, "GDN hidden scratch")?,
        checked_mul(conv_dim, 2, "GDN conv scratch")?,
        checked_mul(value_dim, 3, "GDN value scratch")?,
        checked_mul(gdn.value_heads as u64, 2, "GDN scalar scratch")?,
        checked_mul(intermediate, 3, "GDN MLP scratch")?,
    ])
}

fn full_attention_scratch_elements(
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    intermediate: u64,
) -> Result<u64, String> {
    sum_bytes(&[
        checked_mul(hidden, 5, "full attention hidden scratch")?,
        checked_mul(attention_hidden, 3, "full attention scratch")?,
        checked_mul(kv_hidden, 2, "full attention KV scratch")?,
        checked_mul(intermediate, 3, "full attention MLP scratch")?,
    ])
}

fn linear_gdn_state_bytes(layers: &[CudaHfDecodeChainLayer<'_>]) -> Result<u64, String> {
    layers.iter().try_fold(0, |total, layer| {
        if layer.attention_kind != CUDA_HF_ATTENTION_LINEAR_GDN {
            return Ok(total);
        }
        let Some(gdn) = layer.linear_gdn else {
            return Err("CUDA HF decode linear GDN layer is missing layout metadata".to_string());
        };
        let value_dim = checked_mul(
            gdn.value_heads as u64,
            gdn.value_head_dim as u64,
            "GDN state value dim",
        )?;
        let key_dim = checked_mul(
            gdn.key_heads as u64,
            gdn.key_head_dim as u64,
            "GDN state key dim",
        )?;
        let conv_dim = checked_add(
            checked_mul(key_dim, 2, "GDN state key conv dim")?,
            value_dim,
            "GDN state conv dim",
        )?;
        let conv_state = checked_mul(
            conv_dim,
            (gdn.conv_kernel as u64).saturating_sub(1),
            "GDN conv state",
        )?;
        let recurrent_state =
            checked_mul(value_dim, gdn.key_head_dim as u64, "GDN recurrent state")?;
        checked_add(
            total,
            checked_mul(
                checked_add(conv_state, recurrent_state, "GDN state elements")?,
                F32_BYTES,
                "GDN state bytes",
            )?,
            "GDN state total bytes",
        )
    })
}

fn max_kv_cache_width(
    layers: &[CudaHfDecodeChainLayer<'_>],
    kv_hidden: u64,
) -> Result<u64, String> {
    layers.iter().try_fold(kv_hidden, |width, layer| {
        let Some(deepseek) = layer.deepseek else {
            return Ok(width);
        };
        if layer.attention_kind != CUDA_HF_ATTENTION_DEEPSEEK_MLA || !deepseek.is_v3_mla() {
            return Ok(width);
        }
        let kv_cache_width = checked_add(
            deepseek.kv_lora_rank as u64,
            deepseek.qk_rope_head_dim as u64,
            "DeepSeek V3 MLA KV cache width",
        )?;
        Ok(width.max(kv_cache_width))
    })
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
