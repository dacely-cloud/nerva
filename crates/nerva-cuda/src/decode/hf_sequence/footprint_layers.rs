use crate::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_FULL, CUDA_HF_ATTENTION_LINEAR_GDN, CUDA_HF_MLP_SPARSE_MOE,
    CudaHfDecodeChainLayer, CudaHfLinearGdnLayer,
};

pub(crate) fn layer_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    intermediate: u64,
    declared_weight_plan: bool,
) -> Result<u64, String> {
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

fn optional_len(value: Option<&[u16]>) -> Result<u64, String> {
    value.map_or(Ok(0), |slice| as_u64("optional weight", slice.len()))
}

fn marker(value: Option<&[u16]>, elements: u64) -> u64 {
    if value.is_some() { elements } else { 0 }
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
