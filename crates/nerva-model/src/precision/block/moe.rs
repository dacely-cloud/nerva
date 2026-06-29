use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::run::exact_blockwise_attention_into;
use crate::common::math::{sigmoid, silu, single_token_attention};
use crate::common::rope::{
    apply_rotary_to_key, apply_rotary_to_query, apply_rotary_to_query_key, validate_rope,
};
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::precision::block::ops::{
    add_encoded_bias_into, decode_vec_into, encode_vec_into, mat_vec_encoded_row_major,
    per_head_rms_norm_encoded_in_place, rms_norm_encoded_into,
};
use crate::precision::scratch::{
    PrecisionTransformerBlockKvScratch, PrecisionTransformerBlockScratch,
};

#[derive(Clone, Debug)]
pub struct PrecisionMoeTransformerBlock {
    dtype: DType,
    shape: TransformerBlockShape,
    moe_intermediate: usize,
    shared_expert_intermediate: usize,
    num_experts: usize,
    experts_per_token: usize,
    norm_topk_prob: bool,
    rms_attn_weight: Vec<u16>,
    rms_mlp_weight: Vec<u16>,
    w_q: Vec<u16>,
    w_q_gate: Option<Vec<u16>>,
    w_k: Vec<u16>,
    q_norm_weight: Option<Vec<u16>>,
    k_norm_weight: Option<Vec<u16>>,
    w_v: Vec<u16>,
    w_o: Vec<u16>,
    q_bias: Option<Vec<u16>>,
    k_bias: Option<Vec<u16>>,
    v_bias: Option<Vec<u16>>,
    o_bias: Option<Vec<u16>>,
    router: Vec<u16>,
    expert_gate_up: Vec<u16>,
    expert_down: Vec<u16>,
    shared_expert_gate: Vec<u16>,
    shared_expert_up: Vec<u16>,
    shared_expert_down: Vec<u16>,
    shared_expert_router: Vec<u16>,
    rms_eps: f32,
    rope_theta: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct PrecisionMoeConfig {
    pub moe_intermediate: usize,
    pub shared_expert_intermediate: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub norm_topk_prob: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PrecisionMoeTransformerBlockEncodedView<'a> {
    pub dtype: DType,
    pub shape: TransformerBlockShape,
    pub moe_intermediate: usize,
    pub shared_expert_intermediate: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub norm_topk_prob: bool,
    pub rms_attn_weight: &'a [u16],
    pub rms_mlp_weight: &'a [u16],
    pub w_q: &'a [u16],
    pub w_q_gate: Option<&'a [u16]>,
    pub w_k: &'a [u16],
    pub q_norm_weight: Option<&'a [u16]>,
    pub k_norm_weight: Option<&'a [u16]>,
    pub w_v: &'a [u16],
    pub w_o: &'a [u16],
    pub q_bias: Option<&'a [u16]>,
    pub k_bias: Option<&'a [u16]>,
    pub v_bias: Option<&'a [u16]>,
    pub o_bias: Option<&'a [u16]>,
    pub router: &'a [u16],
    pub expert_gate_up: &'a [u16],
    pub expert_down: &'a [u16],
    pub shared_expert_gate: &'a [u16],
    pub shared_expert_up: &'a [u16],
    pub shared_expert_down: &'a [u16],
    pub shared_expert_router: &'a [u16],
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
}

impl PrecisionMoeTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_encoded(
        dtype: DType,
        shape: TransformerBlockShape,
        config: PrecisionMoeConfig,
        rms_attn_weight: Vec<u16>,
        rms_mlp_weight: Vec<u16>,
        w_q: Vec<u16>,
        w_k: Vec<u16>,
        w_v: Vec<u16>,
        w_o: Vec<u16>,
        router: Vec<u16>,
        expert_gate_up: Vec<u16>,
        expert_down: Vec<u16>,
        shared_expert_gate: Vec<u16>,
        shared_expert_up: Vec<u16>,
        shared_expert_down: Vec<u16>,
        shared_expert_router: Vec<u16>,
        rms_eps: f32,
    ) -> Result<Self> {
        validate_moe_block_layout(
            dtype,
            shape,
            config,
            rms_attn_weight.len(),
            rms_mlp_weight.len(),
            w_q.len(),
            w_k.len(),
            w_v.len(),
            w_o.len(),
            router.len(),
            expert_gate_up.len(),
            expert_down.len(),
            shared_expert_gate.len(),
            shared_expert_up.len(),
            shared_expert_down.len(),
            shared_expert_router.len(),
            rms_eps,
        )?;
        Ok(Self {
            dtype,
            shape,
            moe_intermediate: config.moe_intermediate,
            shared_expert_intermediate: config.shared_expert_intermediate,
            num_experts: config.num_experts,
            experts_per_token: config.experts_per_token,
            norm_topk_prob: config.norm_topk_prob,
            rms_attn_weight,
            rms_mlp_weight,
            w_q,
            w_q_gate: None,
            w_k,
            q_norm_weight: None,
            k_norm_weight: None,
            w_v,
            w_o,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            router,
            expert_gate_up,
            expert_down,
            shared_expert_gate,
            shared_expert_up,
            shared_expert_down,
            shared_expert_router,
            rms_eps,
            rope_theta: None,
        })
    }

    pub fn with_rope_theta(mut self, rope_theta: Option<f32>) -> Result<Self> {
        if let Some(theta) = rope_theta {
            validate_rope(self.shape, theta)?;
        }
        self.rope_theta = rope_theta;
        Ok(self)
    }

    pub fn with_qk_norm(
        mut self,
        q_norm_weight: Vec<u16>,
        k_norm_weight: Vec<u16>,
    ) -> Result<Self> {
        require_len("q_norm.weight", q_norm_weight.len(), self.shape.head_dim())?;
        require_len("k_norm.weight", k_norm_weight.len(), self.shape.head_dim())?;
        self.q_norm_weight = Some(q_norm_weight);
        self.k_norm_weight = Some(k_norm_weight);
        Ok(self)
    }

    pub fn with_query_gate_projection(mut self, w_q_gate: Vec<u16>) -> Result<Self> {
        require_len(
            "q_proj gate",
            w_q_gate.len(),
            self.shape.attention_hidden() * self.shape.hidden,
        )?;
        if self.shape.intermediate < self.shape.attention_hidden() {
            return Err(NervaError::InvalidArgument {
                reason: "q_proj gate requires attention-hidden scratch capacity".to_string(),
            });
        }
        self.w_q_gate = Some(w_q_gate);
        Ok(self)
    }

    pub fn with_attention_biases(
        self,
        q_bias: Vec<u16>,
        k_bias: Vec<u16>,
        v_bias: Vec<u16>,
        o_bias: Vec<u16>,
    ) -> Result<Self> {
        self.with_optional_attention_biases(Some(q_bias), Some(k_bias), Some(v_bias), Some(o_bias))
    }

    pub fn with_optional_attention_biases(
        mut self,
        q_bias: Option<Vec<u16>>,
        k_bias: Option<Vec<u16>>,
        v_bias: Option<Vec<u16>>,
        o_bias: Option<Vec<u16>>,
    ) -> Result<Self> {
        if let Some(q_bias) = q_bias.as_deref() {
            require_len("q_proj.bias", q_bias.len(), self.shape.attention_hidden())?;
        }
        if let Some(k_bias) = k_bias.as_deref() {
            require_len("k_proj.bias", k_bias.len(), self.shape.kv_hidden())?;
        }
        if let Some(v_bias) = v_bias.as_deref() {
            require_len("v_proj.bias", v_bias.len(), self.shape.kv_hidden())?;
        }
        if let Some(o_bias) = o_bias.as_deref() {
            require_len("o_proj.bias", o_bias.len(), self.shape.hidden)?;
        }
        self.q_bias = q_bias;
        self.k_bias = k_bias;
        self.v_bias = v_bias;
        self.o_bias = o_bias;
        Ok(self)
    }

    pub const fn rope_theta(&self) -> Option<f32> {
        self.rope_theta
    }

    fn apply_query_gate_to_attention(
        &self,
        attn_norm: &[f32],
        attn: &mut [f32],
        scratch_gate: &mut [f32],
    ) -> Result<()> {
        let Some(w_q_gate) = self.w_q_gate.as_deref() else {
            return Ok(());
        };
        let attention_hidden = self.shape.attention_hidden();
        require_len("q_proj gate input", attn_norm.len(), self.shape.hidden)?;
        require_len("q_proj gate attention", attn.len(), attention_hidden)?;
        if scratch_gate.len() < attention_hidden {
            return Err(NervaError::InvalidArgument {
                reason: "q_proj gate scratch is smaller than attention hidden".to_string(),
            });
        }
        let gate = &mut scratch_gate[..attention_hidden];
        mat_vec_encoded_row_major(self.dtype, w_q_gate, attn_norm, gate)?;
        for (attn, gate) in attn.iter_mut().zip(gate.iter().copied()) {
            *attn *= sigmoid(gate);
        }
        Ok(())
    }

    pub fn encoded_view(&self) -> PrecisionMoeTransformerBlockEncodedView<'_> {
        PrecisionMoeTransformerBlockEncodedView {
            dtype: self.dtype,
            shape: self.shape,
            moe_intermediate: self.moe_intermediate,
            shared_expert_intermediate: self.shared_expert_intermediate,
            num_experts: self.num_experts,
            experts_per_token: self.experts_per_token,
            norm_topk_prob: self.norm_topk_prob,
            rms_attn_weight: &self.rms_attn_weight,
            rms_mlp_weight: &self.rms_mlp_weight,
            w_q: &self.w_q,
            w_q_gate: self.w_q_gate.as_deref(),
            w_k: &self.w_k,
            q_norm_weight: self.q_norm_weight.as_deref(),
            k_norm_weight: self.k_norm_weight.as_deref(),
            w_v: &self.w_v,
            w_o: &self.w_o,
            q_bias: self.q_bias.as_deref(),
            k_bias: self.k_bias.as_deref(),
            v_bias: self.v_bias.as_deref(),
            o_bias: self.o_bias.as_deref(),
            router: &self.router,
            expert_gate_up: &self.expert_gate_up,
            expert_down: &self.expert_down,
            shared_expert_gate: &self.shared_expert_gate,
            shared_expert_up: &self.shared_expert_up,
            shared_expert_down: &self.shared_expert_down,
            shared_expert_router: &self.shared_expert_router,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
        }
    }

    pub fn forward_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        require_len("precision MoE input", input.len(), self.shape.hidden)?;
        require_len("precision MoE output", output.len(), self.shape.hidden)?;
        scratch.require_shape(self.shape)?;

        decode_vec_into(self.dtype, input, &mut scratch.input)?;
        self.project_qkv_from_scratch(scratch)?;
        self.apply_qk_norm(&mut scratch.q, &mut scratch.k)?;
        if let Some(theta) = self.rope_theta {
            apply_rotary_to_query_key(self.shape, 0, theta, &mut scratch.q, &mut scratch.k)?;
        }
        single_token_attention(
            self.shape,
            &scratch.q,
            &scratch.k,
            &scratch.v,
            &mut scratch.attn,
        );
        self.apply_query_gate_to_attention(
            &scratch.attn_norm,
            &mut scratch.attn,
            &mut scratch.gate,
        )?;
        self.finish_attention_and_moe(
            &scratch.input,
            &scratch.attn,
            &mut scratch.residual,
            &mut scratch.mlp_norm,
            &mut scratch.gate,
            &mut scratch.up,
            &mut scratch.ff,
            &mut scratch.down,
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
        scratch.reset();
        for row in 0..token_count {
            let start = row * self.shape.hidden;
            self.append_kv_from_input(&input[start..start + self.shape.hidden], row, scratch)?;
        }
        for row in 0..token_count {
            let start = row * self.shape.hidden;
            self.forward_with_visible_kv(
                &input[start..start + self.shape.hidden],
                row + 1,
                row,
                scratch,
                &mut output[start..start + self.shape.hidden],
                ledger,
            )?;
        }
        scratch.set_len(token_count);
        require_len("precision MoE prefill output", output.len(), values)
    }

    pub fn forward_decode_with_kv_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        require_len("precision MoE decode input", input.len(), self.shape.hidden)?;
        require_len(
            "precision MoE decode output",
            output.len(),
            self.shape.hidden,
        )?;
        let next_len = scratch
            .len()
            .checked_add(1)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "precision MoE KV length overflow".to_string(),
            })?;
        scratch.require_capacity(self.shape, next_len)?;
        let position = scratch.len();
        self.append_kv_from_input(input, position, scratch)?;
        self.forward_with_visible_kv(input, next_len, position, scratch, output, ledger)
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
                reason: "precision MoE prefill requires at least one token".to_string(),
            });
        }
        scratch.require_capacity(self.shape, token_count)?;
        let values = token_count.checked_mul(self.shape.hidden).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "precision MoE prefill token count overflow".to_string(),
            }
        })?;
        require_len("precision MoE prefill input", input.len(), values)?;
        require_len("precision MoE prefill output", output.len(), values)?;
        Ok(values)
    }

    fn append_kv_from_input(
        &self,
        input: &[u16],
        position: usize,
        scratch: &mut PrecisionTransformerBlockKvScratch,
    ) -> Result<()> {
        let start = scratch.len() * self.shape.kv_hidden();
        let end = start + self.shape.kv_hidden();
        decode_vec_into(self.dtype, input, &mut scratch.token.input)?;
        rms_norm_encoded_into(
            self.dtype,
            &scratch.token.input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.token.attn_norm,
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.w_k,
            &scratch.token.attn_norm,
            &mut scratch.token.k,
        )?;
        if let Some(bias) = self.k_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, &mut scratch.token.k)?;
        }
        if let Some(weight) = self.k_norm_weight.as_deref() {
            per_head_rms_norm_encoded_in_place(
                self.dtype,
                weight,
                self.shape.head_dim(),
                &mut scratch.token.k,
                self.rms_eps,
            )?;
        }
        mat_vec_encoded_row_major(
            self.dtype,
            &self.w_v,
            &scratch.token.attn_norm,
            &mut scratch.token.v,
        )?;
        if let Some(bias) = self.v_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, &mut scratch.token.v)?;
        }
        if let Some(theta) = self.rope_theta {
            apply_rotary_to_key(self.shape, position, theta, &mut scratch.token.k)?;
        }
        scratch.keys[start..end].copy_from_slice(&scratch.token.k);
        scratch.values[start..end].copy_from_slice(&scratch.token.v);
        scratch.set_len(scratch.len() + 1);
        Ok(())
    }

    fn forward_with_visible_kv(
        &self,
        input: &[u16],
        visible_tokens: usize,
        position: usize,
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        decode_vec_into(self.dtype, input, &mut scratch.token.input)?;
        rms_norm_encoded_into(
            self.dtype,
            &scratch.token.input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.token.attn_norm,
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.w_q,
            &scratch.token.attn_norm,
            &mut scratch.token.q,
        )?;
        if let Some(bias) = self.q_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, &mut scratch.token.q)?;
        }
        if let Some(weight) = self.q_norm_weight.as_deref() {
            per_head_rms_norm_encoded_in_place(
                self.dtype,
                weight,
                self.shape.head_dim(),
                &mut scratch.token.q,
                self.rms_eps,
            )?;
        }
        if let Some(theta) = self.rope_theta {
            apply_rotary_to_query(self.shape, position, theta, &mut scratch.token.q)?;
        }
        let values = visible_tokens * self.shape.kv_hidden();
        let kv = [KvAttentionBlock::new(
            &scratch.keys[..values],
            &scratch.values[..values],
            visible_tokens,
            MemoryTier::Dram,
        )];
        exact_blockwise_attention_into(
            self.shape,
            &scratch.token.q,
            &kv,
            &mut scratch.attention,
            &mut scratch.token.attn,
            ledger,
        )?;
        self.apply_query_gate_to_attention(
            &scratch.token.attn_norm,
            &mut scratch.token.attn,
            &mut scratch.token.gate,
        )?;
        self.finish_attention_and_moe(
            &scratch.token.input,
            &scratch.token.attn,
            &mut scratch.token.residual,
            &mut scratch.token.mlp_norm,
            &mut scratch.token.gate,
            &mut scratch.token.up,
            &mut scratch.token.ff,
            &mut scratch.token.down,
            output,
        )
    }

    fn project_qkv_from_scratch(
        &self,
        scratch: &mut PrecisionTransformerBlockScratch,
    ) -> Result<()> {
        rms_norm_encoded_into(
            self.dtype,
            &scratch.input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        )?;
        mat_vec_encoded_row_major(self.dtype, &self.w_q, &scratch.attn_norm, &mut scratch.q)?;
        mat_vec_encoded_row_major(self.dtype, &self.w_k, &scratch.attn_norm, &mut scratch.k)?;
        mat_vec_encoded_row_major(self.dtype, &self.w_v, &scratch.attn_norm, &mut scratch.v)?;
        if let Some(bias) = self.q_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, &mut scratch.q)?;
        }
        if let Some(bias) = self.k_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, &mut scratch.k)?;
        }
        if let Some(bias) = self.v_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, &mut scratch.v)?;
        }
        Ok(())
    }

    fn apply_qk_norm(&self, q: &mut [f32], k: &mut [f32]) -> Result<()> {
        if let Some(weight) = self.q_norm_weight.as_deref() {
            per_head_rms_norm_encoded_in_place(
                self.dtype,
                weight,
                self.shape.head_dim(),
                q,
                self.rms_eps,
            )?;
        }
        if let Some(weight) = self.k_norm_weight.as_deref() {
            per_head_rms_norm_encoded_in_place(
                self.dtype,
                weight,
                self.shape.head_dim(),
                k,
                self.rms_eps,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_attention_and_moe(
        &self,
        input: &[f32],
        attn: &[f32],
        residual: &mut [f32],
        mlp_norm: &mut [f32],
        gate: &mut [f32],
        up: &mut [f32],
        ff: &mut [f32],
        down: &mut [f32],
        output: &mut [u16],
    ) -> Result<()> {
        mat_vec_encoded_row_major(self.dtype, &self.w_o, attn, residual)?;
        if let Some(bias) = self.o_bias.as_deref() {
            add_encoded_bias_into(self.dtype, bias, residual)?;
        }
        for (out, residual_value) in residual.iter_mut().zip(input.iter().copied()) {
            *out += residual_value;
        }
        rms_norm_encoded_into(
            self.dtype,
            residual,
            &self.rms_mlp_weight,
            self.rms_eps,
            mlp_norm,
        )?;
        self.add_sparse_moe_into(mlp_norm, gate, up, ff, down, residual)?;
        encode_vec_into(self.dtype, residual, output)
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
        let mut router_logits = vec![0.0f32; self.num_experts];
        mat_vec_encoded_row_major(self.dtype, &self.router, input, &mut router_logits)?;
        let router_probs = softmax(&router_logits);
        let mut selected = top_k(&router_probs, self.experts_per_token);
        if self.norm_topk_prob {
            let sum = selected.iter().map(|(_, weight)| *weight).sum::<f32>();
            if sum > 0.0 {
                for (_, weight) in &mut selected {
                    *weight /= sum;
                }
            }
        }

        let expert_gate_up_stride = 2 * self.moe_intermediate * self.shape.hidden;
        let expert_down_stride = self.shape.hidden * self.moe_intermediate;
        let gate_len = self.moe_intermediate;
        let up_len = self.moe_intermediate;
        for (expert, weight) in selected {
            let gate_up_start = expert * expert_gate_up_stride;
            let gate_start = gate_up_start;
            let gate_end = gate_start + gate_len * self.shape.hidden;
            let up_start = gate_end;
            let up_end = up_start + up_len * self.shape.hidden;
            let down_start = expert * expert_down_stride;
            let down_end = down_start + expert_down_stride;

            mat_vec_encoded_row_major(
                self.dtype,
                &self.expert_gate_up[gate_start..gate_end],
                input,
                &mut gate[..gate_len],
            )?;
            mat_vec_encoded_row_major(
                self.dtype,
                &self.expert_gate_up[up_start..up_end],
                input,
                &mut up[..up_len],
            )?;
            for ((ff, gate), up) in ff[..self.moe_intermediate]
                .iter_mut()
                .zip(gate[..gate_len].iter().copied())
                .zip(up[..up_len].iter().copied())
            {
                *ff = silu(gate) * up;
            }
            mat_vec_encoded_row_major(
                self.dtype,
                &self.expert_down[down_start..down_end],
                &ff[..self.moe_intermediate],
                down,
            )?;
            for (out, expert_value) in residual.iter_mut().zip(down.iter().copied()) {
                *out += weight * expert_value;
            }
        }
        self.add_shared_expert_into(input, gate, up, ff, down, residual)?;
        Ok(())
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
        if self.shared_expert_intermediate == 0 {
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
            &mut gate[..self.shared_expert_intermediate],
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.shared_expert_up,
            input,
            &mut up[..self.shared_expert_intermediate],
        )?;
        for ((ff, gate), up) in ff[..self.shared_expert_intermediate]
            .iter_mut()
            .zip(gate[..self.shared_expert_intermediate].iter().copied())
            .zip(up[..self.shared_expert_intermediate].iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_encoded_row_major(
            self.dtype,
            &self.shared_expert_down,
            &ff[..self.shared_expert_intermediate],
            down,
        )?;
        let scale = sigmoid(gate_weight[0]);
        for (out, shared_value) in residual.iter_mut().zip(down.iter().copied()) {
            *out += scale * shared_value;
        }
        Ok(())
    }
}

fn validate_moe_block_layout(
    dtype: DType,
    shape: TransformerBlockShape,
    config: PrecisionMoeConfig,
    rms_attn_weight_len: usize,
    rms_mlp_weight_len: usize,
    w_q_len: usize,
    w_k_len: usize,
    w_v_len: usize,
    w_o_len: usize,
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
                reason: "precision MoE block supports only FP16 and BF16".to_string(),
            });
        }
    }
    if config.moe_intermediate == 0
        || config.num_experts == 0
        || config.experts_per_token == 0
        || config.experts_per_token > config.num_experts
    {
        return Err(NervaError::InvalidArgument {
            reason:
                "precision MoE expert dimensions must be non-zero and top-k must fit expert count"
                    .to_string(),
        });
    }
    if config.moe_intermediate > shape.intermediate {
        return Err(NervaError::InvalidArgument {
            reason: "precision MoE intermediate cannot exceed scratch intermediate capacity"
                .to_string(),
        });
    }
    if config.shared_expert_intermediate > shape.intermediate {
        return Err(NervaError::InvalidArgument {
            reason: "precision MoE shared expert intermediate cannot exceed scratch intermediate capacity"
                .to_string(),
        });
    }
    require_len("rms_attn_weight", rms_attn_weight_len, shape.hidden)?;
    require_len("rms_mlp_weight", rms_mlp_weight_len, shape.hidden)?;
    require_len("w_q", w_q_len, shape.attention_hidden() * shape.hidden)?;
    require_len("w_k", w_k_len, shape.kv_hidden() * shape.hidden)?;
    require_len("w_v", w_v_len, shape.kv_hidden() * shape.hidden)?;
    require_len("w_o", w_o_len, shape.hidden * shape.attention_hidden())?;
    require_len("router", router_len, config.num_experts * shape.hidden)?;
    require_len(
        "expert_gate_up",
        expert_gate_up_len,
        config.num_experts * 2 * config.moe_intermediate * shape.hidden,
    )?;
    require_len(
        "expert_down",
        expert_down_len,
        config.num_experts * shape.hidden * config.moe_intermediate,
    )?;
    if config.shared_expert_intermediate == 0 {
        require_len("shared_expert_gate", shared_expert_gate_len, 0)?;
        require_len("shared_expert_up", shared_expert_up_len, 0)?;
        require_len("shared_expert_down", shared_expert_down_len, 0)?;
        require_len("shared_expert_gate_weight", shared_expert_router_len, 0)?;
    } else {
        require_len(
            "shared_expert_gate",
            shared_expert_gate_len,
            config.shared_expert_intermediate * shape.hidden,
        )?;
        require_len(
            "shared_expert_up",
            shared_expert_up_len,
            config.shared_expert_intermediate * shape.hidden,
        )?;
        require_len(
            "shared_expert_down",
            shared_expert_down_len,
            shape.hidden * config.shared_expert_intermediate,
        )?;
        require_len(
            "shared_expert_gate_weight",
            shared_expert_router_len,
            shape.hidden,
        )?;
    }
    if rms_eps <= 0.0 || !rms_eps.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "rms epsilon must be positive and finite".to_string(),
        });
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |left, right| left.max(right));
    let mut probs = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = probs.iter().sum::<f32>();
    if sum > 0.0 && sum.is_finite() {
        for value in &mut probs {
            *value /= sum;
        }
    }
    probs
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    indexed.truncate(k);
    indexed
}
