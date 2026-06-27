use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_precision_block_layout(
    dtype: DType,
    shape: TransformerBlockShape,
    rms_attn_weight_len: usize,
    rms_mlp_weight_len: usize,
    w_q_len: usize,
    w_k_len: usize,
    w_v_len: usize,
    w_o_len: usize,
    w_gate_len: usize,
    w_up_len: usize,
    w_down_len: usize,
    rms_eps: f32,
) -> Result<()> {
    shape.validate()?;
    validate_dtype(dtype)?;
    require_len("rms_attn_weight", rms_attn_weight_len, shape.hidden)?;
    require_len("rms_mlp_weight", rms_mlp_weight_len, shape.hidden)?;
    require_len("w_q", w_q_len, shape.attention_hidden() * shape.hidden)?;
    require_len("w_k", w_k_len, shape.kv_hidden() * shape.hidden)?;
    require_len("w_v", w_v_len, shape.kv_hidden() * shape.hidden)?;
    require_len("w_o", w_o_len, shape.hidden * shape.attention_hidden())?;
    require_len("w_gate", w_gate_len, shape.intermediate * shape.hidden)?;
    require_len("w_up", w_up_len, shape.intermediate * shape.hidden)?;
    require_len("w_down", w_down_len, shape.hidden * shape.intermediate)?;
    if rms_eps <= 0.0 || !rms_eps.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "rms epsilon must be positive and finite".to_string(),
        });
    }
    Ok(())
}

fn validate_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::F16 | DType::BF16 => Ok(()),
        _ => Err(NervaError::InvalidArgument {
            reason: "precision block supports only FP16 and BF16".to_string(),
        }),
    }
}
