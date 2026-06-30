use nerva_core::types::error::{NervaError, Result};

use crate::common::validate::require_len;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepSeekMlaDecodeShape {
    pub heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl DeepSeekMlaDecodeShape {
    pub const fn new(
        heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
    ) -> Self {
        Self {
            heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.heads == 0
            || self.kv_lora_rank == 0
            || self.qk_nope_head_dim == 0
            || self.qk_rope_head_dim == 0
            || self.v_head_dim == 0
        {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek MLA shape dimensions must be non-zero".to_string(),
            });
        }
        Ok(())
    }

    pub fn q_nope_len(self) -> Result<usize> {
        checked_mul(
            self.heads,
            self.qk_nope_head_dim,
            "DeepSeek MLA q_nope length",
        )
    }

    pub fn q_pe_len(self) -> Result<usize> {
        checked_mul(
            self.heads,
            self.qk_rope_head_dim,
            "DeepSeek MLA q_pe length",
        )
    }

    pub fn latent_len(self) -> Result<usize> {
        checked_mul(self.heads, self.kv_lora_rank, "DeepSeek MLA latent length")
    }

    pub fn output_len(self) -> Result<usize> {
        checked_mul(self.heads, self.v_head_dim, "DeepSeek MLA output length")
    }

    pub fn w_uk_len(self) -> Result<usize> {
        checked_mul3(
            self.kv_lora_rank,
            self.heads,
            self.qk_nope_head_dim,
            "DeepSeek MLA W_UK length",
        )
    }

    pub fn w_uv_len(self) -> Result<usize> {
        checked_mul3(
            self.kv_lora_rank,
            self.heads,
            self.v_head_dim,
            "DeepSeek MLA W_UV length",
        )
    }
}

#[derive(Clone, Debug)]
pub struct DeepSeekMlaDecodeScratch {
    shape: DeepSeekMlaDecodeShape,
    ql_nope: Vec<f32>,
    latent_output: Vec<f32>,
}

impl DeepSeekMlaDecodeScratch {
    pub fn new(shape: DeepSeekMlaDecodeShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            ql_nope: vec![0.0; shape.latent_len()?],
            latent_output: vec![0.0; shape.latent_len()?],
        })
    }

    fn require_shape(&self, shape: DeepSeekMlaDecodeShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "DeepSeek MLA scratch shape mismatch".to_string(),
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeepSeekMlaPrefillScratch {
    shape: DeepSeekMlaDecodeShape,
    max_tokens: usize,
    ql_nope: Vec<f32>,
    latent_output: Vec<f32>,
}

impl DeepSeekMlaPrefillScratch {
    pub fn new(shape: DeepSeekMlaDecodeShape, max_tokens: usize) -> Result<Self> {
        shape.validate()?;
        if max_tokens == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek MLA prefill scratch requires at least one token".to_string(),
            });
        }
        Ok(Self {
            shape,
            max_tokens,
            ql_nope: vec![
                0.0;
                checked_mul(
                    max_tokens,
                    shape.latent_len()?,
                    "DeepSeek MLA prefill latent scratch length"
                )?
            ],
            latent_output: vec![0.0; shape.latent_len()?],
        })
    }

    fn require_shape_and_tokens(&self, shape: DeepSeekMlaDecodeShape, tokens: usize) -> Result<()> {
        if self.shape != shape {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek MLA prefill scratch shape mismatch".to_string(),
            });
        }
        if tokens == 0 || tokens > self.max_tokens {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek MLA prefill token count exceeds scratch capacity".to_string(),
            });
        }
        Ok(())
    }
}

pub fn exact_deepseek_mla_decode_mqa_into(
    shape: DeepSeekMlaDecodeShape,
    q_nope: &[f32],
    q_pe: &[f32],
    kv_c: &[f32],
    k_pe: &[f32],
    w_uk_lnp: &[f32],
    w_uv_lnv: &[f32],
    softmax_scale: f32,
    scratch: &mut DeepSeekMlaDecodeScratch,
    output: &mut [f32],
) -> Result<()> {
    shape.validate()?;
    scratch.require_shape(shape)?;
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek MLA softmax scale must be finite and positive".to_string(),
        });
    }

    require_len("DeepSeek MLA q_nope", q_nope.len(), shape.q_nope_len()?)?;
    require_len("DeepSeek MLA q_pe", q_pe.len(), shape.q_pe_len()?)?;
    require_len("DeepSeek MLA W_UK", w_uk_lnp.len(), shape.w_uk_len()?)?;
    require_len("DeepSeek MLA W_UV", w_uv_lnv.len(), shape.w_uv_len()?)?;
    require_len("DeepSeek MLA output", output.len(), shape.output_len()?)?;

    if kv_c.is_empty() || !kv_c.len().is_multiple_of(shape.kv_lora_rank) {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek MLA kv_c length must be a non-empty multiple of kv_lora_rank"
                .to_string(),
        });
    }
    let tokens = kv_c.len() / shape.kv_lora_rank;
    require_len(
        "DeepSeek MLA k_pe",
        k_pe.len(),
        checked_mul(tokens, shape.qk_rope_head_dim, "DeepSeek MLA k_pe length")?,
    )?;

    project_nope_query_to_latent(shape, q_nope, w_uk_lnp, &mut scratch.ql_nope);
    output.fill(0.0);

    for head in 0..shape.heads {
        let latent_start = head * shape.kv_lora_rank;
        let latent_end = latent_start + shape.kv_lora_rank;
        let ql_nope = &scratch.ql_nope[latent_start..latent_end];
        let q_pe = &q_pe[head * shape.qk_rope_head_dim..][..shape.qk_rope_head_dim];
        let latent_output = &mut scratch.latent_output[latent_start..latent_end];
        latent_output.fill(0.0);

        let mut local_m = f32::NEG_INFINITY;
        let mut local_l = 0.0f32;
        for token in 0..tokens {
            let kv = &kv_c[token * shape.kv_lora_rank..][..shape.kv_lora_rank];
            let k_pe = &k_pe[token * shape.qk_rope_head_dim..][..shape.qk_rope_head_dim];
            let score = (dot(ql_nope, kv) + dot(q_pe, k_pe)) * softmax_scale;
            let next_m = local_m.max(score);
            let old_scale = if local_l == 0.0 {
                0.0
            } else {
                (local_m - next_m).exp()
            };
            let new_scale = (score - next_m).exp();
            for (out, value) in latent_output.iter_mut().zip(kv.iter().copied()) {
                *out = *out * old_scale + value * new_scale;
            }
            local_l = local_l * old_scale + new_scale;
            local_m = next_m;
        }
        if local_l == 0.0 || !local_l.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek MLA produced invalid normalizer".to_string(),
            });
        }
        for value in latent_output.iter_mut() {
            *value /= local_l;
        }

        let output_head = &mut output[head * shape.v_head_dim..][..shape.v_head_dim];
        for v_index in 0..shape.v_head_dim {
            let mut sum = 0.0f32;
            for latent in 0..shape.kv_lora_rank {
                let weight_index = (latent * shape.heads + head) * shape.v_head_dim + v_index;
                sum += latent_output[latent] * w_uv_lnv[weight_index];
            }
            output_head[v_index] = sum;
        }
    }

    Ok(())
}

pub fn exact_deepseek_mla_prefill_causal_mqa_into(
    shape: DeepSeekMlaDecodeShape,
    tokens: usize,
    q_nope: &[f32],
    q_pe: &[f32],
    kv_c: &[f32],
    k_pe: &[f32],
    w_uk_lnp: &[f32],
    w_uv_lnv: &[f32],
    softmax_scale: f32,
    scratch: &mut DeepSeekMlaPrefillScratch,
    output: &mut [f32],
) -> Result<()> {
    shape.validate()?;
    scratch.require_shape_and_tokens(shape, tokens)?;
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek MLA softmax scale must be finite and positive".to_string(),
        });
    }

    let q_nope_per_token = shape.q_nope_len()?;
    let q_pe_per_token = shape.q_pe_len()?;
    let output_per_token = shape.output_len()?;
    require_len(
        "DeepSeek MLA prefill q_nope",
        q_nope.len(),
        checked_mul(
            tokens,
            q_nope_per_token,
            "DeepSeek MLA prefill q_nope length",
        )?,
    )?;
    require_len(
        "DeepSeek MLA prefill q_pe",
        q_pe.len(),
        checked_mul(tokens, q_pe_per_token, "DeepSeek MLA prefill q_pe length")?,
    )?;
    require_len(
        "DeepSeek MLA prefill kv_c",
        kv_c.len(),
        checked_mul(
            tokens,
            shape.kv_lora_rank,
            "DeepSeek MLA prefill kv_c length",
        )?,
    )?;
    require_len(
        "DeepSeek MLA prefill k_pe",
        k_pe.len(),
        checked_mul(
            tokens,
            shape.qk_rope_head_dim,
            "DeepSeek MLA prefill k_pe length",
        )?,
    )?;
    require_len("DeepSeek MLA W_UK", w_uk_lnp.len(), shape.w_uk_len()?)?;
    require_len("DeepSeek MLA W_UV", w_uv_lnv.len(), shape.w_uv_len()?)?;
    require_len(
        "DeepSeek MLA prefill output",
        output.len(),
        checked_mul(
            tokens,
            output_per_token,
            "DeepSeek MLA prefill output length",
        )?,
    )?;

    let latent_per_token = shape.latent_len()?;
    project_nope_query_sequence_to_latent(
        shape,
        tokens,
        q_nope,
        w_uk_lnp,
        &mut scratch.ql_nope[..tokens * latent_per_token],
    );
    output.fill(0.0);

    for query_token in 0..tokens {
        let token_q_pe = &q_pe[query_token * q_pe_per_token..][..q_pe_per_token];
        let token_output = &mut output[query_token * output_per_token..][..output_per_token];
        for head in 0..shape.heads {
            let latent_start = head * shape.kv_lora_rank;
            let latent_end = latent_start + shape.kv_lora_rank;
            let ql_token_start = query_token * latent_per_token + latent_start;
            let ql_nope = &scratch.ql_nope[ql_token_start..ql_token_start + shape.kv_lora_rank];
            let q_pe = &token_q_pe[head * shape.qk_rope_head_dim..][..shape.qk_rope_head_dim];
            let latent_output = &mut scratch.latent_output[latent_start..latent_end];
            latent_output.fill(0.0);

            let mut local_m = f32::NEG_INFINITY;
            let mut local_l = 0.0f32;
            for key_token in 0..=query_token {
                let kv = &kv_c[key_token * shape.kv_lora_rank..][..shape.kv_lora_rank];
                let k_pe = &k_pe[key_token * shape.qk_rope_head_dim..][..shape.qk_rope_head_dim];
                let score = (dot(ql_nope, kv) + dot(q_pe, k_pe)) * softmax_scale;
                let next_m = local_m.max(score);
                let old_scale = if local_l == 0.0 {
                    0.0
                } else {
                    (local_m - next_m).exp()
                };
                let new_scale = (score - next_m).exp();
                for (out, value) in latent_output.iter_mut().zip(kv.iter().copied()) {
                    *out = *out * old_scale + value * new_scale;
                }
                local_l = local_l * old_scale + new_scale;
                local_m = next_m;
            }
            if local_l == 0.0 || !local_l.is_finite() {
                return Err(NervaError::InvalidArgument {
                    reason: "DeepSeek MLA prefill produced invalid normalizer".to_string(),
                });
            }
            for value in latent_output.iter_mut() {
                *value /= local_l;
            }

            let output_head = &mut token_output[head * shape.v_head_dim..][..shape.v_head_dim];
            for v_index in 0..shape.v_head_dim {
                let mut sum = 0.0f32;
                for latent in 0..shape.kv_lora_rank {
                    let weight_index = (latent * shape.heads + head) * shape.v_head_dim + v_index;
                    sum += latent_output[latent] * w_uv_lnv[weight_index];
                }
                output_head[v_index] = sum;
            }
        }
    }

    Ok(())
}

fn project_nope_query_to_latent(
    shape: DeepSeekMlaDecodeShape,
    q_nope: &[f32],
    w_uk_lnp: &[f32],
    ql_nope: &mut [f32],
) {
    ql_nope.fill(0.0);
    for head in 0..shape.heads {
        let q_head = &q_nope[head * shape.qk_nope_head_dim..][..shape.qk_nope_head_dim];
        let out_head = &mut ql_nope[head * shape.kv_lora_rank..][..shape.kv_lora_rank];
        for (latent, out) in out_head.iter_mut().enumerate() {
            let weight_start = (latent * shape.heads + head) * shape.qk_nope_head_dim;
            let weights = &w_uk_lnp[weight_start..][..shape.qk_nope_head_dim];
            *out = dot(q_head, weights);
        }
    }
}

fn project_nope_query_sequence_to_latent(
    shape: DeepSeekMlaDecodeShape,
    tokens: usize,
    q_nope: &[f32],
    w_uk_lnp: &[f32],
    ql_nope: &mut [f32],
) {
    ql_nope.fill(0.0);
    let q_nope_per_token = shape.heads * shape.qk_nope_head_dim;
    let latent_per_token = shape.heads * shape.kv_lora_rank;
    for token in 0..tokens {
        let q_token = &q_nope[token * q_nope_per_token..][..q_nope_per_token];
        let out_token = &mut ql_nope[token * latent_per_token..][..latent_per_token];
        project_nope_query_to_latent(shape, q_token, w_uk_lnp, out_token);
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn checked_mul(left: usize, right: usize, label: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("{label} overflow"),
        })
}

fn checked_mul3(first: usize, second: usize, third: usize, label: &'static str) -> Result<usize> {
    checked_mul(first, second, label).and_then(|value| checked_mul(value, third, label))
}
