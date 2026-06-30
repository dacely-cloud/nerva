use crate::decode::hf_chain::layer::{
    CudaHfDecodeChainLayer, CudaHfDeepSeekLayer, CudaHfLinearGdnLayer,
    CUDA_HF_ATTENTION_DEEPSEEK_MLA, CUDA_HF_ATTENTION_FULL, CUDA_HF_ATTENTION_LINEAR_GDN,
    CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR, CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER,
    CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS, CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER,
    CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER, CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED,
    CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER, CUDA_HF_DEEPSEEK_MODE_V4_SWA, CUDA_HF_MLP_DENSE,
    CUDA_HF_MLP_SPARSE_MOE,
};

const DEEPSEEK_V32_PACKED_KV_BLOCK_TOKENS: u64 = 64;
const DEEPSEEK_V32_PACKED_KV_NOPE_BYTES: u64 = 512;
const DEEPSEEK_V32_PACKED_KV_SCALE_BYTES: u64 = 16;
const DEEPSEEK_V32_PACKED_KV_ROPE_VALUES: u64 = 64;
const DEEPSEEK_V32_INDEXER_KV_BLOCK_TOKENS: u64 = 64;
const DEEPSEEK_V4_PACKED_KV_DEFAULT_BLOCK_TOKENS: u64 = 64;
const DEEPSEEK_V4_PACKED_KV_C128_BLOCK_TOKENS: u64 = 2;
const DEEPSEEK_V4_PACKED_KV_ALIGNMENT_BYTES: u64 = 576;

pub(crate) fn deepseek_static_elements(
    layers: &[CudaHfDecodeChainLayer<'_>],
    hidden: u64,
) -> Result<u64, String> {
    let Some(layer) = layers
        .iter()
        .find(|layer| layer.attention_kind == CUDA_HF_ATTENTION_DEEPSEEK_MLA)
    else {
        return Ok(0);
    };
    let deepseek = deepseek_metadata(layer)?;
    if !deepseek_is_v4(deepseek.mode) {
        return Ok(0);
    }
    let hc_mult = deepseek.hc_mult as u64;
    if hc_mult == 0 {
        return Err("CUDA HF decode DeepSeek V4 hc_mult must be non-zero".to_string());
    }
    let hc_dim = checked_mul(hidden, hc_mult, "DeepSeek V4 HC head dim")?;
    let mut total = f32_slots(hc_mult, 1, "DeepSeek V4 HC head base")?;
    total = checked_add(
        total,
        f32_slots(hc_mult, hc_dim, "DeepSeek V4 HC head fn")?,
        "DeepSeek V4 HC head fn",
    )?;
    checked_add(
        total,
        f32_slots(1, 1, "DeepSeek V4 HC head scale")?,
        "DeepSeek V4 HC head scale",
    )
}

pub(crate) fn deepseek_v4_mhc_runtime_bytes(
    layers: &[CudaHfDecodeChainLayer<'_>],
    hidden: u64,
    context_tokens: u64,
) -> Result<u64, String> {
    let hc_mult = layers
        .iter()
        .filter(|layer| layer.attention_kind == CUDA_HF_ATTENTION_DEEPSEEK_MLA)
        .filter_map(|layer| layer.deepseek)
        .filter(|deepseek| deepseek_is_v4(deepseek.mode))
        .map(|deepseek| deepseek.hc_mult as u64)
        .max()
        .unwrap_or(0);
    if hc_mult == 0 || hidden == 0 || context_tokens == 0 {
        return Ok(0);
    }
    let hc_hidden = checked_mul(hidden, hc_mult, "DeepSeek V4 mHC residual width")?;
    let residual = checked_mul(
        checked_mul(context_tokens, hc_hidden, "DeepSeek V4 mHC residual")?,
        4,
        "DeepSeek V4 mHC residual bytes",
    )?;
    let post_mix = checked_mul(
        checked_mul(context_tokens, hc_mult, "DeepSeek V4 mHC post mix")?,
        4,
        "DeepSeek V4 mHC post mix bytes",
    )?;
    let comb_mix = checked_mul(
        checked_mul(
            context_tokens,
            checked_mul(hc_mult, hc_mult, "DeepSeek V4 mHC comb mix width")?,
            "DeepSeek V4 mHC comb mix",
        )?,
        4,
        "DeepSeek V4 mHC comb mix bytes",
    )?;
    checked_add(
        checked_add(residual, post_mix, "DeepSeek V4 mHC runtime bytes")?,
        comb_mix,
        "DeepSeek V4 mHC runtime bytes",
    )
}

pub(crate) fn deepseek_runtime_kv_bytes(
    layers: &[CudaHfDecodeChainLayer<'_>],
    context_tokens: u64,
    kv_token_capacity: u64,
) -> Result<u64, String> {
    layers.iter().try_fold(0u64, |total, layer| {
        let Some(deepseek) = layer
            .deepseek
            .filter(|_| layer.attention_kind == CUDA_HF_ATTENTION_DEEPSEEK_MLA)
        else {
            return Ok(total);
        };
        let layer_bytes = if deepseek.is_v3_mla() {
            deepseek_v32_runtime_kv_bytes(deepseek, context_tokens)?
        } else if deepseek.is_v4_mla() {
            deepseek_v4_runtime_kv_bytes(layer, deepseek, context_tokens, kv_token_capacity)?
        } else {
            0
        };
        checked_add(total, layer_bytes, "DeepSeek runtime KV bytes")
    })
}

pub(crate) fn layer_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    intermediate: u64,
    vocab_size: u64,
    declared_weight_plan: bool,
) -> Result<u64, String> {
    if layer.attention_kind == CUDA_HF_ATTENTION_DEEPSEEK_MLA {
        return deepseek_layer_elements(
            layer,
            hidden,
            attention_hidden,
            head_dim,
            intermediate,
            vocab_size,
        );
    }

    let mut total = hidden;
    total = checked_add(
        total,
        attention_elements(
            layer,
            hidden,
            attention_hidden,
            kv_hidden,
            head_dim,
            declared_weight_plan,
        )?,
        "attention",
    )?;
    total = checked_add(total, hidden, "MLP norm")?;
    total = checked_add(
        total,
        mlp_elements(layer, hidden, intermediate, declared_weight_plan)?,
        "MLP",
    )?;
    optional_elements(
        layer,
        total,
        attention_hidden,
        kv_hidden,
        head_dim,
        hidden,
        declared_weight_plan,
    )
}

fn deepseek_layer_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    attention_hidden: u64,
    head_dim: u64,
    intermediate: u64,
    vocab_size: u64,
) -> Result<u64, String> {
    let deepseek = deepseek_metadata(layer)?;
    let norm = deepseek_norm_slots(deepseek.mode, hidden)?;
    let mut total = norm;
    total = checked_add(
        total,
        deepseek_attention_elements(layer, deepseek, hidden, attention_hidden, head_dim)?,
        "DeepSeek attention",
    )?;
    total = checked_add(total, norm, "DeepSeek MLP norm")?;
    checked_add(
        total,
        deepseek_mlp_elements(layer, deepseek, hidden, intermediate, vocab_size)?,
        "DeepSeek MLP",
    )
}

fn attention_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    _head_dim: u64,
    declared_weight_plan: bool,
) -> Result<u64, String> {
    match layer.attention_kind {
        CUDA_HF_ATTENTION_FULL => {
            let mut total = checked_mul(attention_hidden, hidden, "Q weight")?;
            total = checked_add(total, checked_mul(kv_hidden, hidden, "K weight")?, "K")?;
            total = checked_add(total, checked_mul(kv_hidden, hidden, "V weight")?, "V")?;
            checked_add(
                total,
                checked_mul(hidden, attention_hidden, "O weight")?,
                "O",
            )
        }
        CUDA_HF_ATTENTION_LINEAR_GDN => linear_gdn_elements(layer.linear_gdn, hidden),
        other => Err(format!("CUDA HF decode unsupported attention kind {other}")),
    }
    .and_then(|total| {
        if layer.attention_kind == CUDA_HF_ATTENTION_FULL {
            Ok(total)
        } else if declared_weight_plan || layer.linear_gdn.is_some() {
            Ok(total)
        } else {
            Err("CUDA HF decode linear GDN layer is missing layout metadata".to_string())
        }
    })
}

fn deepseek_attention_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    deepseek: CudaHfDeepSeekLayer,
    hidden: u64,
    attention_hidden: u64,
    head_dim: u64,
) -> Result<u64, String> {
    if deepseek_is_v4(deepseek.mode) {
        deepseek_v4_attention_elements(deepseek, hidden, attention_hidden, head_dim)
    } else {
        deepseek_v3_attention_elements(layer, deepseek, hidden, attention_hidden, head_dim)
    }
}

fn deepseek_v3_attention_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    deepseek: CudaHfDeepSeekLayer,
    hidden: u64,
    attention_hidden: u64,
    head_dim: u64,
) -> Result<u64, String> {
    let heads = checked_div_exact(attention_hidden, head_dim, "DeepSeek V3 attention heads")?;
    let q_lora_rank = deepseek.q_lora_rank as u64;
    let kv_lora_rank = deepseek.kv_lora_rank as u64;
    let qk_nope = deepseek.qk_nope_head_dim as u64;
    let qk_rope = deepseek.qk_rope_head_dim as u64;
    let v_head = deepseek.v_head_dim as u64;
    let q_rows = checked_mul(
        heads,
        checked_add(qk_nope, qk_rope, "DeepSeek V3 Q rows")?,
        "DeepSeek V3 Q rows",
    )?;
    let kv_a_rows = checked_add(kv_lora_rank, qk_rope, "DeepSeek V3 KV-A rows")?;
    let kv_b_rows = checked_mul(
        heads,
        checked_add(qk_nope, v_head, "DeepSeek V3 KV-B rows")?,
        "DeepSeek V3 KV-B rows",
    )?;
    let value_hidden = checked_mul(heads, v_head, "DeepSeek V3 value hidden")?;
    let norm = |rows| deepseek_norm_slots(deepseek.mode, rows);

    let mut total = fp8_slots(q_lora_rank, hidden, "DeepSeek V3 qa")?;
    total = checked_add(
        total,
        scale_f32_slots(q_lora_rank, hidden, "DeepSeek V3 qa scale")?,
        "DeepSeek V3 qa scale",
    )?;
    total = checked_add(total, norm(q_lora_rank)?, "DeepSeek V3 qa norm")?;
    total = checked_add(
        total,
        fp8_slots(q_rows, q_lora_rank, "DeepSeek V3 qb")?,
        "DeepSeek V3 qb",
    )?;
    total = checked_add(
        total,
        scale_f32_slots(q_rows, q_lora_rank, "DeepSeek V3 qb scale")?,
        "DeepSeek V3 qb scale",
    )?;
    total = checked_add(
        total,
        fp8_slots(kv_a_rows, hidden, "DeepSeek V3 kv_a")?,
        "DeepSeek V3 kv_a",
    )?;
    total = checked_add(
        total,
        scale_f32_slots(kv_a_rows, hidden, "DeepSeek V3 kv_a scale")?,
        "DeepSeek V3 kv_a scale",
    )?;
    total = checked_add(total, norm(kv_lora_rank)?, "DeepSeek V3 kv_a norm")?;
    total = checked_add(
        total,
        fp8_slots(kv_b_rows, kv_lora_rank, "DeepSeek V3 kv_b")?,
        "DeepSeek V3 kv_b",
    )?;
    total = checked_add(
        total,
        scale_f32_slots(kv_b_rows, kv_lora_rank, "DeepSeek V3 kv_b scale")?,
        "DeepSeek V3 kv_b scale",
    )?;
    total = checked_add(
        total,
        fp8_slots(hidden, value_hidden, "DeepSeek V3 output")?,
        "DeepSeek V3 output",
    )?;
    total = checked_add(
        total,
        scale_f32_slots(hidden, value_hidden, "DeepSeek V3 output scale")?,
        "DeepSeek V3 output scale",
    )?;

    if layer
        .deepseek
        .is_some_and(|value| value.flags & CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER != 0)
    {
        let index_n_heads = deepseek.index_n_heads as u64;
        let index_head_dim = deepseek.index_head_dim as u64;
        let query_rows = checked_mul(
            index_n_heads,
            index_head_dim,
            "DeepSeek V3.2 indexer query rows",
        )?;
        total = checked_add(
            total,
            fp8_slots(query_rows, q_lora_rank, "DeepSeek V3.2 indexer query")?,
            "DeepSeek V3.2 indexer query",
        )?;
        total = checked_add(
            total,
            scale_f32_slots(query_rows, q_lora_rank, "DeepSeek V3.2 indexer query scale")?,
            "DeepSeek V3.2 indexer query scale",
        )?;
        total = checked_add(
            total,
            fp8_slots(index_head_dim, hidden, "DeepSeek V3.2 indexer key")?,
            "DeepSeek V3.2 indexer key",
        )?;
        total = checked_add(
            total,
            scale_f32_slots(index_head_dim, hidden, "DeepSeek V3.2 indexer key scale")?,
            "DeepSeek V3.2 indexer key scale",
        )?;
        total = checked_add(
            total,
            f32_slots(index_head_dim, 1, "DeepSeek V3.2 indexer key norm")?,
            "DeepSeek V3.2 indexer key norm",
        )?;
        total = checked_add(
            total,
            f32_slots(index_head_dim, 1, "DeepSeek V3.2 indexer key norm bias")?,
            "DeepSeek V3.2 indexer key norm bias",
        )?;
        total = checked_add(
            total,
            bf16_slots(index_n_heads, hidden, "DeepSeek V3.2 indexer weights")?,
            "DeepSeek V3.2 indexer weights",
        )?;
    }
    Ok(total)
}

fn deepseek_v4_attention_elements(
    deepseek: CudaHfDeepSeekLayer,
    hidden: u64,
    attention_hidden: u64,
    head_dim: u64,
) -> Result<u64, String> {
    let heads = checked_div_exact(attention_hidden, head_dim, "DeepSeek V4 attention heads")?;
    let hc_mult = deepseek.hc_mult as u64;
    if hc_mult == 0 {
        return Err("CUDA HF decode DeepSeek V4 hc_mult must be non-zero".to_string());
    }
    let hc_dim = checked_mul(hidden, hc_mult, "DeepSeek V4 HC dim")?;
    let mix_hc = checked_mul(
        hc_mult,
        checked_add(hc_mult, 2, "DeepSeek V4 HC mix add")?,
        "DeepSeek V4 HC mix",
    )?;
    let q_lora_rank = deepseek.q_lora_rank as u64;
    let q_rows = attention_hidden;
    let o_groups = deepseek.o_groups as u64;
    let o_lora_rank = deepseek.o_lora_rank as u64;
    let wo_a_rows = checked_mul(o_groups, o_lora_rank, "DeepSeek V4 wo_a rows")?;
    let wo_a_cols = checked_div_exact(q_rows, o_groups, "DeepSeek V4 wo_a cols")?;

    let mut total = f32_slots(mix_hc, 1, "DeepSeek V4 HC attn base")?;
    total = checked_add(
        total,
        f32_slots(mix_hc, hc_dim, "DeepSeek V4 HC attn fn")?,
        "DeepSeek V4 HC attn fn",
    )?;
    total = checked_add(
        total,
        f32_slots(3, 1, "DeepSeek V4 HC attn scale")?,
        "DeepSeek V4 HC attn scale",
    )?;
    total = checked_add(
        total,
        f32_slots(mix_hc, 1, "DeepSeek V4 HC ffn base")?,
        "DeepSeek V4 HC ffn base",
    )?;
    total = checked_add(
        total,
        f32_slots(mix_hc, hc_dim, "DeepSeek V4 HC ffn fn")?,
        "DeepSeek V4 HC ffn fn",
    )?;
    total = checked_add(
        total,
        f32_slots(3, 1, "DeepSeek V4 HC ffn scale")?,
        "DeepSeek V4 HC ffn scale",
    )?;
    total = checked_add(
        total,
        f32_slots(heads, 1, "DeepSeek V4 attention sink")?,
        "DeepSeek V4 attention sink",
    )?;
    total = checked_add(
        total,
        fp8_slots(q_lora_rank, hidden, "DeepSeek V4 wq_a")?,
        "DeepSeek V4 wq_a",
    )?;
    total = checked_add(
        total,
        scale_e8m0_slots(q_lora_rank, hidden, "DeepSeek V4 wq_a scale")?,
        "DeepSeek V4 wq_a scale",
    )?;
    total = checked_add(
        total,
        fp8_slots(q_rows, q_lora_rank, "DeepSeek V4 wq_b")?,
        "DeepSeek V4 wq_b",
    )?;
    total = checked_add(
        total,
        scale_e8m0_slots(q_rows, q_lora_rank, "DeepSeek V4 wq_b scale")?,
        "DeepSeek V4 wq_b scale",
    )?;
    total = checked_add(
        total,
        bf16_slots(q_lora_rank, 1, "DeepSeek V4 q norm")?,
        "DeepSeek V4 q norm",
    )?;
    total = checked_add(
        total,
        fp8_slots(head_dim, hidden, "DeepSeek V4 wkv")?,
        "DeepSeek V4 wkv",
    )?;
    total = checked_add(
        total,
        scale_e8m0_slots(head_dim, hidden, "DeepSeek V4 wkv scale")?,
        "DeepSeek V4 wkv scale",
    )?;
    total = checked_add(
        total,
        bf16_slots(head_dim, 1, "DeepSeek V4 kv norm")?,
        "DeepSeek V4 kv norm",
    )?;
    total = checked_add(
        total,
        fp8_slots(wo_a_rows, wo_a_cols, "DeepSeek V4 wo_a")?,
        "DeepSeek V4 wo_a",
    )?;
    total = checked_add(
        total,
        scale_e8m0_slots(wo_a_rows, wo_a_cols, "DeepSeek V4 wo_a scale")?,
        "DeepSeek V4 wo_a scale",
    )?;
    total = checked_add(
        total,
        fp8_slots(hidden, wo_a_rows, "DeepSeek V4 wo_b")?,
        "DeepSeek V4 wo_b",
    )?;
    total = checked_add(
        total,
        scale_e8m0_slots(hidden, wo_a_rows, "DeepSeek V4 wo_b scale")?,
        "DeepSeek V4 wo_b scale",
    )?;

    if deepseek.flags & CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR != 0 && deepseek.compress_ratio > 1 {
        total = checked_add(
            total,
            deepseek_v4_compressor_elements(deepseek.compress_ratio as u64, hidden, head_dim)?,
            "DeepSeek V4 compressor",
        )?;
    }
    if deepseek.compress_ratio == 4 {
        let index_n_heads = deepseek.index_n_heads as u64;
        let index_head_dim = deepseek.index_head_dim as u64;
        let index_rows = checked_mul(index_n_heads, index_head_dim, "DeepSeek V4 indexer rows")?;
        total = checked_add(
            total,
            fp8_slots(index_rows, q_lora_rank, "DeepSeek V4 indexer wq_b")?,
            "DeepSeek V4 indexer wq_b",
        )?;
        total = checked_add(
            total,
            scale_e8m0_slots(index_rows, q_lora_rank, "DeepSeek V4 indexer wq_b scale")?,
            "DeepSeek V4 indexer wq_b scale",
        )?;
        total = checked_add(
            total,
            deepseek_v4_compressor_elements(4, hidden, index_head_dim)?,
            "DeepSeek V4 indexer compressor",
        )?;
        total = checked_add(
            total,
            bf16_slots(index_n_heads, hidden, "DeepSeek V4 indexer weights")?,
            "DeepSeek V4 indexer weights",
        )?;
    }
    Ok(total)
}

fn deepseek_v4_compressor_elements(
    compress_ratio: u64,
    hidden: u64,
    head_dim: u64,
) -> Result<u64, String> {
    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let rows = checked_mul(head_dim, coff, "DeepSeek V4 compressor rows")?;
    let mut total = f32_slots(compress_ratio, rows, "DeepSeek V4 compressor ape")?;
    total = checked_add(
        total,
        bf16_slots(rows, hidden, "DeepSeek V4 compressor wkv")?,
        "DeepSeek V4 compressor wkv",
    )?;
    total = checked_add(
        total,
        bf16_slots(rows, hidden, "DeepSeek V4 compressor wgate")?,
        "DeepSeek V4 compressor wgate",
    )?;
    checked_add(
        total,
        bf16_slots(head_dim, 1, "DeepSeek V4 compressor norm")?,
        "DeepSeek V4 compressor norm",
    )
}

fn linear_gdn_elements(gdn: Option<CudaHfLinearGdnLayer<'_>>, hidden: u64) -> Result<u64, String> {
    let Some(gdn) = gdn else {
        return Err("CUDA HF decode linear GDN layer is missing layout metadata".to_string());
    };
    let key_dim = checked_mul(gdn.key_heads as u64, gdn.key_head_dim as u64, "GDN key dim")?;
    let value_dim = checked_mul(
        gdn.value_heads as u64,
        gdn.value_head_dim as u64,
        "GDN value dim",
    )?;
    let conv_dim = checked_add(
        checked_mul(key_dim, 2, "GDN key conv dim")?,
        value_dim,
        "GDN conv dim",
    )?;
    let mut total = checked_mul(conv_dim, gdn.conv_kernel as u64, "GDN conv")?;
    total = checked_add(total, checked_mul(conv_dim, hidden, "GDN qkv")?, "GDN qkv")?;
    total = checked_add(total, checked_mul(value_dim, hidden, "GDN z")?, "GDN z")?;
    total = checked_add(
        total,
        checked_mul(gdn.value_heads as u64, hidden, "GDN b")?,
        "GDN b",
    )?;
    total = checked_add(
        total,
        checked_mul(gdn.value_heads as u64, hidden, "GDN a")?,
        "GDN a",
    )?;
    total = checked_add(total, gdn.value_heads as u64, "GDN dt bias")?;
    total = checked_add(
        total,
        checked_mul(gdn.value_heads as u64, 2, "GDN A_log f32 slots")?,
        "GDN A_log",
    )?;
    total = checked_add(total, gdn.value_head_dim as u64, "GDN norm")?;
    checked_add(total, checked_mul(hidden, value_dim, "GDN out")?, "GDN out")
}

fn mlp_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    intermediate: u64,
    declared_weight_plan: bool,
) -> Result<u64, String> {
    if layer.mlp_kind == CUDA_HF_MLP_SPARSE_MOE {
        let moe_intermediate = layer.moe_intermediate as u64;
        let num_experts = layer.num_experts as u64;
        let mut total = checked_mul(num_experts, hidden, "MoE router weight")?;
        total = checked_add(
            total,
            checked_mul(
                checked_mul(num_experts, 2, "MoE gate/up expert count")?,
                checked_mul(moe_intermediate, hidden, "MoE gate/up expert shape")?,
                "MoE gate/up weight",
            )?,
            "MoE gate/up",
        )?;
        total = checked_add(
            total,
            checked_mul(
                num_experts,
                checked_mul(hidden, moe_intermediate, "MoE down expert shape")?,
                "MoE down weight",
            )?,
            "MoE down",
        )?;
        let shared_intermediate = layer.shared_expert_intermediate as u64;
        if shared_intermediate != 0 {
            total = checked_add(
                total,
                checked_mul(shared_intermediate, hidden, "MoE shared gate weight")?,
                "MoE shared gate",
            )?;
            total = checked_add(
                total,
                checked_mul(shared_intermediate, hidden, "MoE shared up weight")?,
                "MoE shared up",
            )?;
            total = checked_add(
                total,
                checked_mul(hidden, shared_intermediate, "MoE shared down weight")?,
                "MoE shared down",
            )?;
            total = checked_add(total, hidden, "MoE shared gate router")?;
        }
        return Ok(total);
    }
    let rows = if declared_weight_plan {
        intermediate
    } else {
        as_u64("gate rows", layer.w_gate.len())?
            .checked_div(hidden)
            .ok_or_else(|| "CUDA HF decode layer hidden is zero".to_string())?
    };
    let mut total = checked_mul(rows, hidden, "gate weight")?;
    total = checked_add(total, checked_mul(rows, hidden, "up weight")?, "up")?;
    checked_add(total, checked_mul(hidden, rows, "down weight")?, "down")
}

fn deepseek_mlp_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    deepseek: CudaHfDeepSeekLayer,
    hidden: u64,
    intermediate: u64,
    vocab_size: u64,
) -> Result<u64, String> {
    if layer.mlp_kind != CUDA_HF_MLP_SPARSE_MOE {
        return deepseek_v3_dense_mlp_elements(hidden, intermediate);
    }
    if deepseek_is_v4(deepseek.mode) {
        deepseek_v4_moe_elements(layer, deepseek, hidden, vocab_size)
    } else {
        deepseek_v3_moe_elements(layer, deepseek, hidden)
    }
}

fn deepseek_v3_dense_mlp_elements(hidden: u64, intermediate: u64) -> Result<u64, String> {
    let mut total = fp8_slots(intermediate, hidden, "DeepSeek dense gate")?;
    total = checked_add(
        total,
        scale_f32_slots(intermediate, hidden, "DeepSeek dense gate scale")?,
        "DeepSeek dense gate scale",
    )?;
    total = checked_add(
        total,
        fp8_slots(intermediate, hidden, "DeepSeek dense up")?,
        "DeepSeek dense up",
    )?;
    total = checked_add(
        total,
        scale_f32_slots(intermediate, hidden, "DeepSeek dense up scale")?,
        "DeepSeek dense up scale",
    )?;
    total = checked_add(
        total,
        fp8_slots(hidden, intermediate, "DeepSeek dense down")?,
        "DeepSeek dense down",
    )?;
    checked_add(
        total,
        scale_f32_slots(hidden, intermediate, "DeepSeek dense down scale")?,
        "DeepSeek dense down scale",
    )
}

fn deepseek_v3_moe_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    deepseek: CudaHfDeepSeekLayer,
    hidden: u64,
) -> Result<u64, String> {
    let num_experts = layer.num_experts as u64;
    let moe_intermediate = layer.moe_intermediate as u64;
    let shared_intermediate = layer.shared_expert_intermediate as u64;
    let mut total = bf16_slots(num_experts, hidden, "DeepSeek V3 router")?;
    if deepseek.flags & CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS != 0 {
        total = checked_add(
            total,
            f32_slots(num_experts, 1, "DeepSeek V3 router bias")?,
            "DeepSeek V3 router bias",
        )?;
    }
    if shared_intermediate != 0 {
        for (rows, cols, label) in [
            (shared_intermediate, hidden, "DeepSeek V3 shared gate"),
            (shared_intermediate, hidden, "DeepSeek V3 shared up"),
            (hidden, shared_intermediate, "DeepSeek V3 shared down"),
        ] {
            total = checked_add(total, fp8_slots(rows, cols, label)?, label)?;
            total = checked_add(total, scale_f32_slots(rows, cols, label)?, label)?;
        }
    }
    for (rows, cols, label) in [
        (moe_intermediate, hidden, "DeepSeek V3 expert gate"),
        (moe_intermediate, hidden, "DeepSeek V3 expert up"),
        (hidden, moe_intermediate, "DeepSeek V3 expert down"),
    ] {
        total = checked_add(
            total,
            rank3_slots(num_experts, rows, cols, 1, label)?,
            label,
        )?;
        total = checked_add(
            total,
            rank3_f32_slots(num_experts, scale_dim(rows), scale_dim(cols), label)?,
            label,
        )?;
    }
    Ok(total)
}

fn deepseek_v4_moe_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    deepseek: CudaHfDeepSeekLayer,
    hidden: u64,
    vocab_size: u64,
) -> Result<u64, String> {
    let num_experts = layer.num_experts as u64;
    let top_k = layer.experts_per_token as u64;
    let moe_intermediate = layer.moe_intermediate as u64;
    let shared_intermediate = layer.shared_expert_intermediate as u64;
    let mut total = bf16_slots(num_experts, hidden, "DeepSeek V4 router")?;
    if deepseek.flags & CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER != 0 {
        total = checked_add(
            total,
            i64_slots(vocab_size, top_k, "DeepSeek V4 hash route table")?,
            "DeepSeek V4 hash route table",
        )?;
    } else {
        total = checked_add(
            total,
            f32_slots(num_experts, 1, "DeepSeek V4 router bias")?,
            "DeepSeek V4 router bias",
        )?;
    }
    if shared_intermediate != 0 {
        for (rows, cols, label) in [
            (shared_intermediate, hidden, "DeepSeek V4 shared gate"),
            (shared_intermediate, hidden, "DeepSeek V4 shared up"),
            (hidden, shared_intermediate, "DeepSeek V4 shared down"),
        ] {
            total = checked_add(total, fp8_slots(rows, cols, label)?, label)?;
            total = checked_add(total, scale_e8m0_slots(rows, cols, label)?, label)?;
        }
    }
    let half_hidden = checked_div_exact(hidden, 2, "DeepSeek V4 routed expert hidden")?;
    let half_intermediate = checked_div_exact(
        moe_intermediate,
        2,
        "DeepSeek V4 routed expert intermediate",
    )?;
    for (rows, cols, label) in [
        (moe_intermediate, half_hidden, "DeepSeek V4 expert gate"),
        (moe_intermediate, half_hidden, "DeepSeek V4 expert up"),
        (hidden, half_intermediate, "DeepSeek V4 expert down"),
    ] {
        total = checked_add(
            total,
            rank3_slots(num_experts, rows, cols, 1, label)?,
            label,
        )?;
        total = checked_add(
            total,
            rank3_slots(num_experts, rows, cols.div_ceil(16), 1, label)?,
            label,
        )?;
    }
    Ok(total)
}

fn optional_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    total: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    hidden: u64,
    declared_weight_plan: bool,
) -> Result<u64, String> {
    let mut total = total;
    if declared_weight_plan {
        total = checked_add(total, marker(layer.q_norm_weight, head_dim), "Q norm")?;
        total = checked_add(total, marker(layer.k_norm_weight, head_dim), "K norm")?;
        total = checked_add(
            total,
            marker(layer.w_q_gate, attention_hidden * hidden),
            "Q gate",
        )?;
        total = checked_add(total, marker(layer.q_bias, attention_hidden), "Q bias")?;
        total = checked_add(total, marker(layer.k_bias, kv_hidden), "K bias")?;
        total = checked_add(total, marker(layer.v_bias, kv_hidden), "V bias")?;
        return checked_add(total, marker(layer.o_bias, hidden), "O bias");
    }
    total = checked_add(total, optional_len(layer.q_norm_weight)?, "Q norm")?;
    total = checked_add(total, optional_len(layer.k_norm_weight)?, "K norm")?;
    total = checked_add(total, optional_len(layer.w_q_gate)?, "Q gate")?;
    total = checked_add(total, optional_len(layer.q_bias)?, "Q bias")?;
    total = checked_add(total, optional_len(layer.k_bias)?, "K bias")?;
    total = checked_add(total, optional_len(layer.v_bias)?, "V bias")?;
    checked_add(total, optional_len(layer.o_bias)?, "O bias")
}

fn deepseek_metadata(layer: &CudaHfDecodeChainLayer<'_>) -> Result<CudaHfDeepSeekLayer, String> {
    layer
        .deepseek
        .ok_or_else(|| "CUDA HF decode DeepSeek MLA layer is missing metadata".to_string())
}

fn deepseek_v32_runtime_kv_bytes(
    deepseek: CudaHfDeepSeekLayer,
    context_tokens: u64,
) -> Result<u64, String> {
    let mut total = 0u64;
    if deepseek.kv_lora_rank as u64 == DEEPSEEK_V32_PACKED_KV_NOPE_BYTES
        && deepseek.qk_rope_head_dim as u64 == DEEPSEEK_V32_PACKED_KV_ROPE_VALUES
    {
        let token_bytes = checked_add(
            checked_add(
                DEEPSEEK_V32_PACKED_KV_NOPE_BYTES,
                DEEPSEEK_V32_PACKED_KV_SCALE_BYTES,
                "DeepSeek V3.2 packed MLA token bytes",
            )?,
            checked_mul(
                DEEPSEEK_V32_PACKED_KV_ROPE_VALUES,
                2,
                "DeepSeek V3.2 packed MLA rope bytes",
            )?,
            "DeepSeek V3.2 packed MLA token bytes",
        )?;
        total = checked_add(
            total,
            packed_page_bytes(
                context_tokens,
                DEEPSEEK_V32_PACKED_KV_BLOCK_TOKENS,
                token_bytes,
                1,
                "DeepSeek V3.2 packed MLA KV",
            )?,
            "DeepSeek V3.2 runtime KV bytes",
        )?;
    }
    if deepseek.flags & CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER != 0 && deepseek.index_head_dim != 0 {
        let index_head_dim = deepseek.index_head_dim as u64;
        let token_bytes = checked_add(
            index_head_dim,
            checked_mul(
                index_head_dim.div_ceil(128),
                4,
                "DeepSeek V3.2 indexer KV scale",
            )?,
            "DeepSeek V3.2 indexer KV token bytes",
        )?;
        total = checked_add(
            total,
            packed_page_bytes(
                context_tokens,
                DEEPSEEK_V32_INDEXER_KV_BLOCK_TOKENS,
                token_bytes,
                1,
                "DeepSeek V3.2 indexer KV",
            )?,
            "DeepSeek V3.2 runtime KV bytes",
        )?;
        if deepseek.q_lora_rank != 0 && deepseek.index_n_heads != 0 {
            let query_bytes = checked_mul(
                deepseek.index_n_heads as u64,
                index_head_dim,
                "DeepSeek V3.2 indexer query bytes",
            )?;
            let q_scale_offset = round_up(query_bytes, 4)?;
            let token_bytes = checked_add(
                checked_add(
                    q_scale_offset,
                    checked_mul(
                        deepseek.index_n_heads as u64,
                        4,
                        "DeepSeek V3.2 indexer query scale bytes",
                    )?,
                    "DeepSeek V3.2 indexer query state bytes",
                )?,
                checked_mul(
                    deepseek.index_n_heads as u64,
                    4,
                    "DeepSeek V3.2 indexer weights state bytes",
                )?,
                "DeepSeek V3.2 indexer query state bytes",
            )?;
            total = checked_add(
                total,
                checked_mul(
                    context_tokens,
                    token_bytes,
                    "DeepSeek V3.2 indexer query state layer bytes",
                )?,
                "DeepSeek V3.2 runtime KV bytes",
            )?;
        }
    }
    Ok(total)
}

fn deepseek_v4_runtime_kv_bytes(
    layer: &CudaHfDecodeChainLayer<'_>,
    deepseek: CudaHfDeepSeekLayer,
    context_tokens: u64,
    kv_token_capacity: u64,
) -> Result<u64, String> {
    if layer.mlp_kind != CUDA_HF_MLP_SPARSE_MOE && layer.mlp_kind != CUDA_HF_MLP_DENSE {
        return Ok(0);
    }
    let mut total = packed_page_bytes(
        context_tokens,
        DEEPSEEK_V4_PACKED_KV_DEFAULT_BLOCK_TOKENS,
        deepseek_v4_main_compressed_kv_token_bytes(deepseek)?,
        DEEPSEEK_V4_PACKED_KV_ALIGNMENT_BYTES,
        "DeepSeek V4 SWA KV",
    )?;

    if !matches!(
        deepseek.mode,
        CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED | CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER
    ) || deepseek.compress_ratio <= 1
    {
        return Ok(total);
    }

    let compressed_capacity =
        deepseek_v4_compressed_token_capacity(context_tokens, deepseek.compress_ratio as u64)?;
    let compressed_block_tokens = deepseek_v4_packed_kv_block_tokens(deepseek.compress_ratio);
    total = checked_add(
        total,
        packed_page_bytes(
            compressed_capacity,
            compressed_block_tokens,
            deepseek_v4_main_compressed_kv_token_bytes(deepseek)?,
            DEEPSEEK_V4_PACKED_KV_ALIGNMENT_BYTES,
            "DeepSeek V4 compressed KV",
        )?,
        "DeepSeek V4 runtime KV bytes",
    )?;
    total = checked_add(
        total,
        deepseek_v4_compressor_state_layer_bytes(deepseek, kv_token_capacity)?,
        "DeepSeek V4 runtime KV bytes",
    )?;

    if deepseek.mode == CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER && deepseek.index_head_dim != 0
    {
        total = checked_add(
            total,
            packed_page_bytes(
                compressed_capacity,
                compressed_block_tokens,
                deepseek_v4_indexer_kv_token_bytes(deepseek)?,
                DEEPSEEK_V4_PACKED_KV_ALIGNMENT_BYTES,
                "DeepSeek V4 indexer KV",
            )?,
            "DeepSeek V4 runtime KV bytes",
        )?;
        total = checked_add(
            total,
            deepseek_v4_indexer_state_layer_bytes(deepseek, kv_token_capacity)?,
            "DeepSeek V4 runtime KV bytes",
        )?;
    }

    Ok(total)
}

fn deepseek_v4_compressed_token_capacity(
    context_tokens: u64,
    compress_ratio: u64,
) -> Result<u64, String> {
    if context_tokens == 0 || compress_ratio == 0 {
        return Ok(0);
    }
    let compressed_tokens = context_tokens.div_ceil(compress_ratio);
    let block_tokens = deepseek_v4_packed_kv_block_tokens(compress_ratio as usize);
    let blocks = compressed_tokens.div_ceil(block_tokens);
    checked_mul(
        blocks,
        block_tokens,
        "DeepSeek V4 compressed token capacity",
    )
}

fn deepseek_v4_main_compressed_kv_token_bytes(
    deepseek: CudaHfDeepSeekLayer,
) -> Result<u64, String> {
    let nope = deepseek.qk_nope_head_dim as u64;
    let rope = deepseek.qk_rope_head_dim as u64;
    if nope == 0 && rope == 0 {
        return Ok(0);
    }
    checked_add(
        checked_add(
            nope,
            checked_mul(rope, 2, "DeepSeek V4 KV rope bytes")?,
            "DeepSeek V4 KV token stride",
        )?,
        nope / 64 + 1,
        "DeepSeek V4 KV token bytes",
    )
}

fn deepseek_v4_indexer_kv_token_bytes(deepseek: CudaHfDeepSeekLayer) -> Result<u64, String> {
    let index_head_dim = deepseek.index_head_dim as u64;
    checked_add(
        index_head_dim,
        checked_mul(
            index_head_dim.div_ceil(128),
            4,
            "DeepSeek V4 indexer KV scale bytes",
        )?,
        "DeepSeek V4 indexer KV token bytes",
    )
}

fn deepseek_v4_compressor_state_layer_bytes(
    deepseek: CudaHfDeepSeekLayer,
    kv_token_capacity: u64,
) -> Result<u64, String> {
    let state_width = checked_mul(
        deepseek_v4_compressor_coff(deepseek),
        checked_add(
            deepseek.qk_nope_head_dim as u64,
            deepseek.qk_rope_head_dim as u64,
            "DeepSeek V4 compressor state width",
        )?,
        "DeepSeek V4 compressor state width",
    )?;
    checked_mul(
        checked_mul(
            kv_token_capacity,
            checked_mul(state_width, 2, "DeepSeek V4 compressor state slots")?,
            "DeepSeek V4 compressor state",
        )?,
        4,
        "DeepSeek V4 compressor state bytes",
    )
}

fn deepseek_v4_indexer_state_layer_bytes(
    deepseek: CudaHfDeepSeekLayer,
    kv_token_capacity: u64,
) -> Result<u64, String> {
    let state_width = checked_mul(
        deepseek_v4_compressor_coff(deepseek),
        deepseek.index_head_dim as u64,
        "DeepSeek V4 indexer state width",
    )?;
    checked_mul(
        checked_mul(
            kv_token_capacity,
            checked_mul(state_width, 2, "DeepSeek V4 indexer state slots")?,
            "DeepSeek V4 indexer state",
        )?,
        4,
        "DeepSeek V4 indexer state bytes",
    )
}

fn deepseek_v4_compressor_coff(deepseek: CudaHfDeepSeekLayer) -> u64 {
    if deepseek.compress_ratio == 4 {
        2
    } else {
        1
    }
}

fn deepseek_v4_packed_kv_block_tokens(compress_ratio: usize) -> u64 {
    if compress_ratio >= 128 {
        DEEPSEEK_V4_PACKED_KV_C128_BLOCK_TOKENS
    } else {
        DEEPSEEK_V4_PACKED_KV_DEFAULT_BLOCK_TOKENS
    }
}

fn packed_page_bytes(
    token_capacity: u64,
    block_tokens: u64,
    token_bytes: u64,
    alignment: u64,
    label: &str,
) -> Result<u64, String> {
    if token_capacity == 0 || block_tokens == 0 || token_bytes == 0 {
        return Ok(0);
    }
    let blocks = token_capacity.div_ceil(block_tokens);
    let page_bytes = checked_mul(block_tokens, token_bytes, label)?;
    let aligned_page_bytes = round_up(page_bytes, alignment)?;
    checked_mul(blocks, aligned_page_bytes, label)
}

fn deepseek_norm_slots(mode: u32, rows: u64) -> Result<u64, String> {
    if mode == CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER {
        f32_slots(rows, 1, "DeepSeek V3.2 norm")
    } else {
        bf16_slots(rows, 1, "DeepSeek norm")
    }
}

fn deepseek_is_v4(mode: u32) -> bool {
    matches!(
        mode,
        CUDA_HF_DEEPSEEK_MODE_V4_SWA
            | CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED
            | CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER
    )
}

fn optional_len(value: Option<&[u16]>) -> Result<u64, String> {
    value.map_or(Ok(0), |slice| as_u64("optional weight", slice.len()))
}

fn marker(value: Option<&[u16]>, elements: u64) -> u64 {
    if value.is_some() {
        elements
    } else {
        0
    }
}

fn bf16_slots(rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    checked_mul(rows, cols, label)
}

fn f32_slots(rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    checked_mul(checked_mul(rows, cols, label)?, 2, label)
}

fn i64_slots(rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    checked_mul(checked_mul(rows, cols, label)?, 4, label)
}

fn fp8_slots(rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    byte_slots(rows, cols, 1, label)
}

fn scale_f32_slots(rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    f32_slots(scale_dim(rows), scale_dim(cols), label)
}

fn scale_e8m0_slots(rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    byte_slots(scale_dim(rows), scale_dim(cols), 1, label)
}

fn rank3_f32_slots(depth: u64, rows: u64, cols: u64, label: &str) -> Result<u64, String> {
    checked_mul(
        checked_mul(depth, checked_mul(rows, cols, label)?, label)?,
        2,
        label,
    )
}

fn rank3_slots(
    depth: u64,
    rows: u64,
    cols: u64,
    bytes_per_element: u64,
    label: &str,
) -> Result<u64, String> {
    let elements = checked_mul(depth, checked_mul(rows, cols, label)?, label)?;
    let bytes = checked_mul(elements, bytes_per_element, label)?;
    Ok(bytes.div_ceil(2))
}

fn byte_slots(rows: u64, cols: u64, bytes_per_element: u64, label: &str) -> Result<u64, String> {
    let elements = checked_mul(rows, cols, label)?;
    let bytes = checked_mul(elements, bytes_per_element, label)?;
    Ok(bytes.div_ceil(2))
}

fn scale_dim(value: u64) -> u64 {
    value.div_ceil(128)
}

fn round_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment <= 1 {
        return Ok(value);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    checked_add(value, alignment - remainder, "aligned byte count")
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("CUDA HF decode {label} overflow"))
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("CUDA HF decode {label} overflow"))
}

fn checked_div_exact(left: u64, right: u64, label: &str) -> Result<u64, String> {
    if right == 0 {
        return Err(format!("CUDA HF decode {label} divisor is zero"));
    }
    if left % right != 0 {
        return Err(format!("CUDA HF decode {label} is not integral"));
    }
    Ok(left / right)
}

fn as_u64(label: &str, value: usize) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("CUDA HF decode {label} does not fit u64"))
}
