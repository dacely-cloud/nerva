use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::{sigmoid, silu};
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::precision::bits::decode_f32_for_dtype;
use crate::precision::block::moe::PrecisionMoeConfig;
use crate::precision::block::ops::{
    decode_vec_into, encode_vec_into, mat_vec_encoded_row_major, rms_norm_encoded_into,
};
use crate::precision::scratch::{
    PrecisionTransformerBlockKvScratch, PrecisionTransformerBlockScratch,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrecisionGatedDeltaNetConfig {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

impl PrecisionGatedDeltaNetConfig {
    pub fn key_dim(self) -> Result<usize> {
        checked_mul(self.key_heads, self.key_head_dim, "GatedDeltaNet key dim")
    }

    pub fn value_dim(self) -> Result<usize> {
        checked_mul(
            self.value_heads,
            self.value_head_dim,
            "GatedDeltaNet value dim",
        )
    }

    pub fn conv_dim(self) -> Result<usize> {
        checked_add(
            checked_mul(self.key_dim()?, 2, "GatedDeltaNet key conv dim")?,
            self.value_dim()?,
            "GatedDeltaNet conv dim",
        )
    }
}

#[derive(Clone, Debug)]
pub struct PrecisionGatedDeltaNetMoeBlock {
    dtype: DType,
    shape: TransformerBlockShape,
    gdn: PrecisionGatedDeltaNetConfig,
    moe: PrecisionMoeConfig,
    rms_attn_weight: Vec<u16>,
    linear_conv: Vec<u16>,
    linear_qkv: Vec<u16>,
    linear_z: Vec<u16>,
    linear_b: Vec<u16>,
    linear_a: Vec<u16>,
    linear_dt_bias: Vec<u16>,
    linear_a_log: Vec<f32>,
    linear_a_log_bits: Vec<u16>,
    linear_norm: Vec<f32>,
    linear_norm_bits: Vec<u16>,
    linear_out: Vec<u16>,
    rms_mlp_weight: Vec<u16>,
    router: Vec<u16>,
    expert_gate_up: Vec<u16>,
    expert_down: Vec<u16>,
    shared_expert_gate: Vec<u16>,
    shared_expert_up: Vec<u16>,
    shared_expert_down: Vec<u16>,
    shared_expert_router: Vec<u16>,
    rms_eps: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PrecisionGatedDeltaNetMoeEncodedView<'a> {
    pub dtype: DType,
    pub shape: TransformerBlockShape,
    pub gdn: PrecisionGatedDeltaNetConfig,
    pub moe: PrecisionMoeConfig,
    pub rms_attn_weight: &'a [u16],
    pub linear_conv: &'a [u16],
    pub linear_qkv: &'a [u16],
    pub linear_z: &'a [u16],
    pub linear_b: &'a [u16],
    pub linear_a: &'a [u16],
    pub linear_dt_bias: &'a [u16],
    pub linear_a_log: &'a [f32],
    pub linear_a_log_bits: &'a [u16],
    pub linear_norm: &'a [f32],
    pub linear_norm_bits: &'a [u16],
    pub linear_out: &'a [u16],
    pub rms_mlp_weight: &'a [u16],
    pub router: &'a [u16],
    pub expert_gate_up: &'a [u16],
    pub expert_down: &'a [u16],
    pub shared_expert_gate: &'a [u16],
    pub shared_expert_up: &'a [u16],
    pub shared_expert_down: &'a [u16],
    pub shared_expert_router: &'a [u16],
    pub rms_eps: f32,
}

impl PrecisionGatedDeltaNetMoeBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_encoded(
        dtype: DType,
        shape: TransformerBlockShape,
        gdn: PrecisionGatedDeltaNetConfig,
        moe: PrecisionMoeConfig,
        rms_attn_weight: Vec<u16>,
        linear_conv: Vec<u16>,
        linear_qkv: Vec<u16>,
        linear_z: Vec<u16>,
        linear_b: Vec<u16>,
        linear_a: Vec<u16>,
        linear_dt_bias: Vec<u16>,
        linear_a_log: Vec<f32>,
        linear_norm: Vec<f32>,
        linear_out: Vec<u16>,
        rms_mlp_weight: Vec<u16>,
        router: Vec<u16>,
        expert_gate_up: Vec<u16>,
        expert_down: Vec<u16>,
        shared_expert_gate: Vec<u16>,
        shared_expert_up: Vec<u16>,
        shared_expert_down: Vec<u16>,
        shared_expert_router: Vec<u16>,
        rms_eps: f32,
    ) -> Result<Self> {
        validate_gdn_moe_layout(
            dtype,
            shape,
            gdn,
            moe,
            rms_attn_weight.len(),
            linear_conv.len(),
            linear_qkv.len(),
            linear_z.len(),
            linear_b.len(),
            linear_a.len(),
            linear_dt_bias.len(),
            linear_a_log.len(),
            linear_norm.len(),
            linear_out.len(),
            rms_mlp_weight.len(),
            router.len(),
            expert_gate_up.len(),
            expert_down.len(),
            shared_expert_gate.len(),
            shared_expert_up.len(),
            shared_expert_down.len(),
            shared_expert_router.len(),
            rms_eps,
        )?;
        let linear_a_log_bits = f32_values_to_u16_slots(&linear_a_log);
        let linear_norm_bits = f32_values_to_u16_slots(&linear_norm);
        Ok(Self {
            dtype,
            shape,
            gdn,
            moe,
            rms_attn_weight,
            linear_conv,
            linear_qkv,
            linear_z,
            linear_b,
            linear_a,
            linear_dt_bias,
            linear_a_log,
            linear_a_log_bits,
            linear_norm,
            linear_norm_bits,
            linear_out,
            rms_mlp_weight,
            router,
            expert_gate_up,
            expert_down,
            shared_expert_gate,
            shared_expert_up,
            shared_expert_down,
            shared_expert_router,
            rms_eps,
        })
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub const fn gdn_config(&self) -> PrecisionGatedDeltaNetConfig {
        self.gdn
    }

    pub const fn moe_config(&self) -> PrecisionMoeConfig {
        self.moe
    }

    pub fn forward_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        let conv_dim = self.gdn.conv_dim()?;
        let recurrent_len = self.recurrent_state_len()?;
        let mut conv_state = vec![0.0f32; conv_dim * self.gdn.conv_kernel.saturating_sub(1)];
        let mut recurrent_state = vec![0.0f32; recurrent_len];
        self.forward_with_gdn_state(
            input,
            &mut conv_state,
            &mut recurrent_state,
            scratch,
            output,
        )
    }

    pub fn forward_prefill_sequence_into(
        &self,
        input: &[u16],
        token_count: usize,
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let values = self.require_sequence_io(input, output, token_count, scratch)?;
        let conv_dim = self.gdn.conv_dim()?;
        scratch.ensure_gated_delta_net_state(
            conv_dim,
            self.gdn.conv_kernel,
            self.gdn.value_heads,
            self.gdn.value_head_dim,
            self.gdn.key_head_dim,
        )?;
        scratch.reset();
        for row in 0..token_count {
            let start = row * self.shape.hidden;
            let gdn_state = scratch
                .gdn
                .as_mut()
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: "precision GatedDeltaNet scratch state is missing".to_string(),
                })?;
            self.forward_with_gdn_state(
                &input[start..start + self.shape.hidden],
                &mut gdn_state.conv_state,
                &mut gdn_state.recurrent_state,
                &mut scratch.token,
                &mut output[start..start + self.shape.hidden],
            )?;
            scratch.set_len(row + 1);
        }
        let _ = ledger;
        require_len(
            "precision GatedDeltaNet-MoE prefill output",
            output.len(),
            values,
        )
    }

    pub fn forward_decode_with_kv_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        require_len(
            "precision GatedDeltaNet-MoE decode input",
            input.len(),
            self.shape.hidden,
        )?;
        require_len(
            "precision GatedDeltaNet-MoE decode output",
            output.len(),
            self.shape.hidden,
        )?;
        let next_len = scratch
            .len()
            .checked_add(1)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "precision GatedDeltaNet-MoE context length overflow".to_string(),
            })?;
        scratch.require_capacity(self.shape, next_len)?;
        let conv_dim = self.gdn.conv_dim()?;
        scratch.ensure_gated_delta_net_state(
            conv_dim,
            self.gdn.conv_kernel,
            self.gdn.value_heads,
            self.gdn.value_head_dim,
            self.gdn.key_head_dim,
        )?;
        let gdn_state = scratch
            .gdn
            .as_mut()
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "precision GatedDeltaNet scratch state is missing".to_string(),
            })?;
        self.forward_with_gdn_state(
            input,
            &mut gdn_state.conv_state,
            &mut gdn_state.recurrent_state,
            &mut scratch.token,
            output,
        )?;
        scratch.set_len(next_len);
        Ok(())
    }

    pub fn encoded_view(&self) -> PrecisionGatedDeltaNetMoeEncodedView<'_> {
        PrecisionGatedDeltaNetMoeEncodedView {
            dtype: self.dtype,
            shape: self.shape,
            gdn: self.gdn,
            moe: self.moe,
            rms_attn_weight: &self.rms_attn_weight,
            linear_conv: &self.linear_conv,
            linear_qkv: &self.linear_qkv,
            linear_z: &self.linear_z,
            linear_b: &self.linear_b,
            linear_a: &self.linear_a,
            linear_dt_bias: &self.linear_dt_bias,
            linear_a_log: &self.linear_a_log,
            linear_a_log_bits: &self.linear_a_log_bits,
            linear_norm: &self.linear_norm,
            linear_norm_bits: &self.linear_norm_bits,
            linear_out: &self.linear_out,
            rms_mlp_weight: &self.rms_mlp_weight,
            router: &self.router,
            expert_gate_up: &self.expert_gate_up,
            expert_down: &self.expert_down,
            shared_expert_gate: &self.shared_expert_gate,
            shared_expert_up: &self.shared_expert_up,
            shared_expert_down: &self.shared_expert_down,
            shared_expert_router: &self.shared_expert_router,
            rms_eps: self.rms_eps,
        }
    }

    fn require_sequence_io(
        &self,
        input: &[u16],
        output: &[u16],
        token_count: usize,
        scratch: &PrecisionTransformerBlockKvScratch,
    ) -> Result<usize> {
        if token_count == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "precision GatedDeltaNet-MoE prefill requires at least one token"
                    .to_string(),
            });
        }
        scratch.require_capacity(self.shape, token_count)?;
        let values = token_count.checked_mul(self.shape.hidden).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "precision GatedDeltaNet-MoE prefill token count overflow".to_string(),
            }
        })?;
        require_len(
            "precision GatedDeltaNet-MoE prefill input",
            input.len(),
            values,
        )?;
        require_len(
            "precision GatedDeltaNet-MoE prefill output",
            output.len(),
            values,
        )?;
        Ok(values)
    }

    fn recurrent_state_len(&self) -> Result<usize> {
        checked_mul(
            checked_mul(
                self.gdn.value_heads,
                self.gdn.value_head_dim,
                "GatedDeltaNet recurrent state value dim",
            )?,
            self.gdn.key_head_dim,
            "GatedDeltaNet recurrent state",
        )
    }

    fn forward_with_gdn_state(
        &self,
        input: &[u16],
        conv_state: &mut [f32],
        recurrent_state: &mut [f32],
        scratch: &mut PrecisionTransformerBlockScratch,
        output: &mut [u16],
    ) -> Result<()> {
        require_len(
            "precision GatedDeltaNet-MoE input",
            input.len(),
            self.shape.hidden,
        )?;
        require_len(
            "precision GatedDeltaNet-MoE output",
            output.len(),
            self.shape.hidden,
        )?;
        scratch.require_shape(self.shape)?;
        let conv_dim = self.gdn.conv_dim()?;
        let key_dim = self.gdn.key_dim()?;
        let value_dim = self.gdn.value_dim()?;
        require_len(
            "precision GatedDeltaNet conv state",
            conv_state.len(),
            conv_dim * self.gdn.conv_kernel.saturating_sub(1),
        )?;
        require_len(
            "precision GatedDeltaNet recurrent state",
            recurrent_state.len(),
            self.recurrent_state_len()?,
        )?;

        let mut mixed_qkv = vec![0.0f32; conv_dim];
        let mut convolved_qkv = vec![0.0f32; conv_dim];
        let mut z = vec![0.0f32; value_dim];
        let mut b = vec![0.0f32; self.gdn.value_heads];
        let mut a = vec![0.0f32; self.gdn.value_heads];
        let mut core = vec![0.0f32; value_dim];
        let mut normed_core = vec![0.0f32; value_dim];

        decode_vec_into(self.dtype, input, &mut scratch.input)?;
        rms_norm_encoded_into(
            self.dtype,
            &scratch.input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.linear_qkv,
            &scratch.attn_norm,
            &mut mixed_qkv,
        )?;
        mat_vec_encoded_row_major(self.dtype, &self.linear_z, &scratch.attn_norm, &mut z)?;
        mat_vec_encoded_row_major(self.dtype, &self.linear_b, &scratch.attn_norm, &mut b)?;
        mat_vec_encoded_row_major(self.dtype, &self.linear_a, &scratch.attn_norm, &mut a)?;

        self.causal_conv1d_update(&mixed_qkv, conv_state, &mut convolved_qkv)?;
        self.gated_delta_rule_single(
            &convolved_qkv[..key_dim],
            &convolved_qkv[key_dim..key_dim * 2],
            &convolved_qkv[key_dim * 2..],
            &a,
            &b,
            recurrent_state,
            &mut core,
        )?;
        self.rms_norm_gated(&core, &z, &mut normed_core)?;

        mat_vec_encoded_row_major(
            self.dtype,
            &self.linear_out,
            &normed_core,
            &mut scratch.residual,
        )?;
        for (out, residual_value) in scratch
            .residual
            .iter_mut()
            .zip(scratch.input.iter().copied())
        {
            *out += residual_value;
        }
        rms_norm_encoded_into(
            self.dtype,
            &scratch.residual,
            &self.rms_mlp_weight,
            self.rms_eps,
            &mut scratch.mlp_norm,
        )?;
        self.add_sparse_moe_into(
            &scratch.mlp_norm,
            &mut scratch.gate,
            &mut scratch.up,
            &mut scratch.ff,
            &mut scratch.down,
            &mut scratch.residual,
        )?;
        encode_vec_into(self.dtype, &scratch.residual, output)
    }

    fn causal_conv1d_update(
        &self,
        mixed_qkv: &[f32],
        conv_state: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        let conv_dim = self.gdn.conv_dim()?;
        let state_len = self.gdn.conv_kernel.saturating_sub(1);
        require_len("GatedDeltaNet conv input", mixed_qkv.len(), conv_dim)?;
        require_len("GatedDeltaNet conv output", output.len(), conv_dim)?;
        require_len(
            "GatedDeltaNet conv state",
            conv_state.len(),
            conv_dim * state_len,
        )?;
        for dim in 0..conv_dim {
            let weight_start = dim * self.gdn.conv_kernel;
            let state_start = dim * state_len;
            let mut acc = 0.0f32;
            for tap in 0..state_len {
                acc += decode_f32_for_dtype(self.linear_conv[weight_start + tap], self.dtype)?
                    * conv_state[state_start + tap];
            }
            acc += decode_f32_for_dtype(self.linear_conv[weight_start + state_len], self.dtype)?
                * mixed_qkv[dim];
            output[dim] = silu(acc);
            if state_len > 0 {
                for tap in 1..state_len {
                    conv_state[state_start + tap - 1] = conv_state[state_start + tap];
                }
                conv_state[state_start + state_len - 1] = mixed_qkv[dim];
            }
        }
        Ok(())
    }

    fn gated_delta_rule_single(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        a: &[f32],
        b: &[f32],
        recurrent_state: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        let key_dim = self.gdn.key_dim()?;
        let value_dim = self.gdn.value_dim()?;
        require_len("GatedDeltaNet query", query.len(), key_dim)?;
        require_len("GatedDeltaNet key", key.len(), key_dim)?;
        require_len("GatedDeltaNet value", value.len(), value_dim)?;
        require_len("GatedDeltaNet a", a.len(), self.gdn.value_heads)?;
        require_len("GatedDeltaNet b", b.len(), self.gdn.value_heads)?;
        require_len("GatedDeltaNet output", output.len(), value_dim)?;
        require_len(
            "GatedDeltaNet recurrent state",
            recurrent_state.len(),
            self.recurrent_state_len()?,
        )?;
        if !self.gdn.value_heads.is_multiple_of(self.gdn.key_heads) {
            return Err(NervaError::InvalidArgument {
                reason: "GatedDeltaNet value heads must be a multiple of key heads".to_string(),
            });
        }

        let mut query_norm = query.to_vec();
        let mut key_norm = key.to_vec();
        for head in 0..self.gdn.key_heads {
            let start = head * self.gdn.key_head_dim;
            let end = start + self.gdn.key_head_dim;
            l2_norm_in_place(&mut query_norm[start..end]);
            l2_norm_in_place(&mut key_norm[start..end]);
        }
        let query_scale = (self.gdn.key_head_dim as f32).sqrt().recip();
        for value in &mut query_norm {
            *value *= query_scale;
        }

        let value_heads_per_key = self.gdn.value_heads / self.gdn.key_heads;
        for value_head in 0..self.gdn.value_heads {
            let key_head = value_head / value_heads_per_key;
            let q_start = key_head * self.gdn.key_head_dim;
            let q_slice = &query_norm[q_start..q_start + self.gdn.key_head_dim];
            let k_slice = &key_norm[q_start..q_start + self.gdn.key_head_dim];
            let v_start = value_head * self.gdn.value_head_dim;
            let v_slice = &value[v_start..v_start + self.gdn.value_head_dim];
            let out_slice = &mut output[v_start..v_start + self.gdn.value_head_dim];
            let state_start = value_head * self.gdn.value_head_dim * self.gdn.key_head_dim;
            let state = &mut recurrent_state
                [state_start..state_start + self.gdn.value_head_dim * self.gdn.key_head_dim];

            let decay = (-self.linear_a_log[value_head].exp()
                * softplus(
                    a[value_head]
                        + decode_f32_for_dtype(self.linear_dt_bias[value_head], self.dtype)?,
                ))
            .exp();
            let beta = sigmoid(b[value_head]);

            for value_offset in 0..self.gdn.value_head_dim {
                let row_start = value_offset * self.gdn.key_head_dim;
                let row = &mut state[row_start..row_start + self.gdn.key_head_dim];
                for item in row.iter_mut() {
                    *item *= decay;
                }
                let previous = row
                    .iter()
                    .zip(k_slice.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                let delta = (v_slice[value_offset] - previous) * beta;
                for (state_value, key_value) in row.iter_mut().zip(k_slice.iter().copied()) {
                    *state_value += delta * key_value;
                }
                out_slice[value_offset] = row
                    .iter()
                    .zip(q_slice.iter())
                    .map(|(left, right)| left * right)
                    .sum();
            }
        }
        Ok(())
    }

    fn rms_norm_gated(&self, core: &[f32], z: &[f32], output: &mut [f32]) -> Result<()> {
        let value_dim = self.gdn.value_dim()?;
        require_len("GatedDeltaNet core output", core.len(), value_dim)?;
        require_len("GatedDeltaNet z gate", z.len(), value_dim)?;
        require_len("GatedDeltaNet gated output", output.len(), value_dim)?;
        for value_head in 0..self.gdn.value_heads {
            let start = value_head * self.gdn.value_head_dim;
            let end = start + self.gdn.value_head_dim;
            let core_slice = &core[start..end];
            let mean_square = core_slice.iter().map(|value| value * value).sum::<f32>()
                / self.gdn.value_head_dim as f32;
            let scale = (mean_square + self.rms_eps).sqrt().recip();
            for (((out, value), gate), weight) in output[start..end]
                .iter_mut()
                .zip(core_slice.iter().copied())
                .zip(z[start..end].iter().copied())
                .zip(self.linear_norm.iter().copied())
            {
                *out = value * scale * weight * silu(gate);
            }
        }
        Ok(())
    }

    fn add_sparse_moe_into(
        &self,
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        ff: &mut [f32],
        down: &mut [f32],
        residual: &mut [f32],
    ) -> Result<()> {
        let mut router_logits = vec![0.0f32; self.moe.num_experts];
        mat_vec_encoded_row_major(self.dtype, &self.router, input, &mut router_logits)?;
        let router_probs = softmax(&router_logits);
        let mut selected = top_k(&router_probs, self.moe.experts_per_token);
        if self.moe.norm_topk_prob {
            let sum = selected.iter().map(|(_, weight)| *weight).sum::<f32>();
            if sum > 0.0 {
                for (_, weight) in &mut selected {
                    *weight /= sum;
                }
            }
        }

        let expert_gate_up_stride = 2 * self.moe.moe_intermediate * self.shape.hidden;
        let expert_down_stride = self.shape.hidden * self.moe.moe_intermediate;
        for (expert, weight) in selected {
            let gate_up_start = expert * expert_gate_up_stride;
            let gate_start = gate_up_start;
            let gate_end = gate_start + self.moe.moe_intermediate * self.shape.hidden;
            let up_start = gate_end;
            let up_end = up_start + self.moe.moe_intermediate * self.shape.hidden;
            let down_start = expert * expert_down_stride;
            let down_end = down_start + expert_down_stride;

            mat_vec_encoded_row_major(
                self.dtype,
                &self.expert_gate_up[gate_start..gate_end],
                input,
                &mut gate[..self.moe.moe_intermediate],
            )?;
            mat_vec_encoded_row_major(
                self.dtype,
                &self.expert_gate_up[up_start..up_end],
                input,
                &mut up[..self.moe.moe_intermediate],
            )?;
            for ((ff, gate), up) in ff[..self.moe.moe_intermediate]
                .iter_mut()
                .zip(gate[..self.moe.moe_intermediate].iter().copied())
                .zip(up[..self.moe.moe_intermediate].iter().copied())
            {
                *ff = silu(gate) * up;
            }
            mat_vec_encoded_row_major(
                self.dtype,
                &self.expert_down[down_start..down_end],
                &ff[..self.moe.moe_intermediate],
                down,
            )?;
            for (out, expert_value) in residual.iter_mut().zip(down.iter().copied()) {
                *out += weight * expert_value;
            }
        }
        self.add_shared_expert_into(input, gate, up, ff, down, residual)
    }

    fn add_shared_expert_into(
        &self,
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        ff: &mut [f32],
        down: &mut [f32],
        residual: &mut [f32],
    ) -> Result<()> {
        if self.moe.shared_expert_intermediate == 0 {
            return Ok(());
        }
        let mut gate_weight = [0.0f32; 1];
        mat_vec_encoded_row_major(
            self.dtype,
            &self.shared_expert_router,
            input,
            &mut gate_weight,
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.shared_expert_gate,
            input,
            &mut gate[..self.moe.shared_expert_intermediate],
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.shared_expert_up,
            input,
            &mut up[..self.moe.shared_expert_intermediate],
        )?;
        for ((ff, gate), up) in ff[..self.moe.shared_expert_intermediate]
            .iter_mut()
            .zip(gate[..self.moe.shared_expert_intermediate].iter().copied())
            .zip(up[..self.moe.shared_expert_intermediate].iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_encoded_row_major(
            self.dtype,
            &self.shared_expert_down,
            &ff[..self.moe.shared_expert_intermediate],
            down,
        )?;
        let scale = sigmoid(gate_weight[0]);
        for (out, shared_value) in residual.iter_mut().zip(down.iter().copied()) {
            *out += scale * shared_value;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gdn_moe_layout(
    dtype: DType,
    shape: TransformerBlockShape,
    gdn: PrecisionGatedDeltaNetConfig,
    moe: PrecisionMoeConfig,
    rms_attn_len: usize,
    linear_conv_len: usize,
    linear_qkv_len: usize,
    linear_z_len: usize,
    linear_b_len: usize,
    linear_a_len: usize,
    linear_dt_bias_len: usize,
    linear_a_log_len: usize,
    linear_norm_len: usize,
    linear_out_len: usize,
    rms_mlp_len: usize,
    router_len: usize,
    expert_gate_up_len: usize,
    expert_down_len: usize,
    shared_expert_gate_len: usize,
    shared_expert_up_len: usize,
    shared_expert_down_len: usize,
    shared_expert_router_len: usize,
    rms_eps: f32,
) -> Result<()> {
    shape.validate()?;
    match dtype {
        DType::F16 | DType::BF16 => {}
        _ => {
            return Err(NervaError::InvalidArgument {
                reason: "precision GatedDeltaNet-MoE block supports only FP16 and BF16 weights"
                    .to_string(),
            });
        }
    }
    if !rms_eps.is_finite() || rms_eps <= 0.0 {
        return Err(NervaError::InvalidArgument {
            reason: "precision GatedDeltaNet-MoE RMS epsilon must be positive".to_string(),
        });
    }
    if gdn.key_heads == 0
        || gdn.value_heads == 0
        || gdn.key_head_dim == 0
        || gdn.value_head_dim == 0
        || gdn.conv_kernel == 0
    {
        return Err(NervaError::InvalidArgument {
            reason: "precision GatedDeltaNet dimensions must be non-zero".to_string(),
        });
    }
    if moe.moe_intermediate == 0
        || moe.num_experts == 0
        || moe.experts_per_token == 0
        || moe.experts_per_token > moe.num_experts
    {
        return Err(NervaError::InvalidArgument {
            reason: "precision GatedDeltaNet-MoE expert dimensions must be non-zero and top-k must fit expert count".to_string(),
        });
    }
    if moe.moe_intermediate > shape.intermediate
        || moe.shared_expert_intermediate > shape.intermediate
    {
        return Err(NervaError::InvalidArgument {
            reason: "precision GatedDeltaNet-MoE expert intermediate exceeds scratch capacity"
                .to_string(),
        });
    }

    let value_dim = gdn.value_dim()?;
    let conv_dim = gdn.conv_dim()?;
    require_len("GatedDeltaNet attention norm", rms_attn_len, shape.hidden)?;
    require_len(
        "GatedDeltaNet conv projection",
        linear_conv_len,
        checked_mul(conv_dim, gdn.conv_kernel, "GatedDeltaNet conv projection")?,
    )?;
    require_len(
        "GatedDeltaNet qkv projection",
        linear_qkv_len,
        checked_mul(conv_dim, shape.hidden, "GatedDeltaNet qkv projection")?,
    )?;
    require_len(
        "GatedDeltaNet z projection",
        linear_z_len,
        checked_mul(value_dim, shape.hidden, "GatedDeltaNet z projection")?,
    )?;
    require_len(
        "GatedDeltaNet B projection",
        linear_b_len,
        checked_mul(gdn.value_heads, shape.hidden, "GatedDeltaNet B projection")?,
    )?;
    require_len(
        "GatedDeltaNet A projection",
        linear_a_len,
        checked_mul(gdn.value_heads, shape.hidden, "GatedDeltaNet A projection")?,
    )?;
    require_len("GatedDeltaNet dt bias", linear_dt_bias_len, gdn.value_heads)?;
    require_len("GatedDeltaNet A log", linear_a_log_len, gdn.value_heads)?;
    require_len("GatedDeltaNet norm", linear_norm_len, gdn.value_head_dim)?;
    require_len(
        "GatedDeltaNet output projection",
        linear_out_len,
        checked_mul(shape.hidden, value_dim, "GatedDeltaNet output projection")?,
    )?;
    require_len("GatedDeltaNet MLP norm", rms_mlp_len, shape.hidden)?;
    require_len(
        "GatedDeltaNet-MoE router",
        router_len,
        checked_mul(moe.num_experts, shape.hidden, "GatedDeltaNet-MoE router")?,
    )?;
    require_len(
        "GatedDeltaNet-MoE expert gate/up",
        expert_gate_up_len,
        checked_mul(
            checked_mul(moe.num_experts, 2, "GatedDeltaNet-MoE gate/up experts")?,
            checked_mul(
                moe.moe_intermediate,
                shape.hidden,
                "GatedDeltaNet-MoE gate/up expert",
            )?,
            "GatedDeltaNet-MoE expert gate/up",
        )?,
    )?;
    require_len(
        "GatedDeltaNet-MoE expert down",
        expert_down_len,
        checked_mul(
            moe.num_experts,
            checked_mul(
                shape.hidden,
                moe.moe_intermediate,
                "GatedDeltaNet-MoE down expert",
            )?,
            "GatedDeltaNet-MoE expert down",
        )?,
    )?;
    let shared = moe.shared_expert_intermediate;
    require_len(
        "GatedDeltaNet-MoE shared expert gate",
        shared_expert_gate_len,
        checked_mul(shared, shape.hidden, "GatedDeltaNet-MoE shared expert gate")?,
    )?;
    require_len(
        "GatedDeltaNet-MoE shared expert up",
        shared_expert_up_len,
        checked_mul(shared, shape.hidden, "GatedDeltaNet-MoE shared expert up")?,
    )?;
    require_len(
        "GatedDeltaNet-MoE shared expert down",
        shared_expert_down_len,
        checked_mul(shape.hidden, shared, "GatedDeltaNet-MoE shared expert down")?,
    )?;
    require_len(
        "GatedDeltaNet-MoE shared expert router",
        shared_expert_router_len,
        if shared == 0 { 0 } else { shape.hidden },
    )?;
    Ok(())
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: left,
            reason: format!("{label} size overflow"),
        })
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: left,
            reason: format!("{label} size overflow"),
        })
}

fn f32_values_to_u16_slots(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            [(bits & 0xffff) as u16, (bits >> 16) as u16]
        })
        .collect()
}

fn l2_norm_in_place(values: &mut [f32]) {
    let scale = (values.iter().map(|value| value * value).sum::<f32>() + 1e-6)
        .sqrt()
        .recip();
    for value in values {
        *value *= scale;
    }
}

fn softplus(value: f32) -> f32 {
    if value <= 20.0 {
        (1.0 + value.exp()).ln()
    } else {
        value
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |left, right| left.max(right));
    let mut values = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for logit in logits {
        let value = (*logit - max).exp();
        sum += value;
        values.push(value);
    }
    if sum > 0.0 {
        for value in &mut values {
            *value /= sum;
        }
    }
    values
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    indexed.truncate(k.min(indexed.len()));
    indexed
}
