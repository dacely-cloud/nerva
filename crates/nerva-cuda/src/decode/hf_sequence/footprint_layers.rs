use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;

pub(crate) fn layer_elements(
    layer: &CudaHfDecodeChainLayer<'_>,
    hidden: u64,
    attention_hidden: u64,
    kv_hidden: u64,
    head_dim: u64,
    intermediate: u64,
    declared_weight_plan: bool,
) -> Result<u64, String> {
    let rows = if declared_weight_plan {
        intermediate
    } else {
        as_u64("gate rows", layer.w_gate.len())?
            .checked_div(hidden)
            .ok_or_else(|| "CUDA HF decode layer hidden is zero".to_string())?
    };
    let mut total = checked_mul(hidden, 2, "norm weights")?;
    total = checked_add(
        total,
        checked_mul(attention_hidden, hidden, "Q weight")?,
        "Q",
    )?;
    total = checked_add(total, checked_mul(kv_hidden, hidden, "K weight")?, "K")?;
    total = checked_add(total, checked_mul(kv_hidden, hidden, "V weight")?, "V")?;
    total = checked_add(
        total,
        checked_mul(hidden, attention_hidden, "O weight")?,
        "O",
    )?;
    total = checked_add(total, checked_mul(rows, hidden, "gate weight")?, "gate")?;
    total = checked_add(total, checked_mul(rows, hidden, "up weight")?, "up")?;
    total = checked_add(total, checked_mul(hidden, rows, "down weight")?, "down")?;
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
        total = checked_add(total, marker(layer.q_bias, attention_hidden), "Q bias")?;
        total = checked_add(total, marker(layer.k_bias, kv_hidden), "K bias")?;
        total = checked_add(total, marker(layer.v_bias, kv_hidden), "V bias")?;
        return checked_add(total, marker(layer.o_bias, hidden), "O bias");
    }
    total = checked_add(total, optional_len(layer.q_norm_weight)?, "Q norm")?;
    total = checked_add(total, optional_len(layer.k_norm_weight)?, "K norm")?;
    total = checked_add(total, optional_len(layer.q_bias)?, "Q bias")?;
    total = checked_add(total, optional_len(layer.k_bias)?, "K bias")?;
    total = checked_add(total, optional_len(layer.v_bias)?, "V bias")?;
    checked_add(total, optional_len(layer.o_bias)?, "O bias")
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
