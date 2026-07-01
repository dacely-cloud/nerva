use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::{
    NervaCudaHfDecodeSequenceLayoutPlanRequest, NervaCudaHfDecodeSequenceLayoutPlanResult,
    plan_hf_decode_sequence_layout,
};

pub const CUDA_HF_SEQUENCE_MISSING_OFFSET: u64 = u64::MAX;

#[derive(Clone, Debug)]
pub struct CudaHfDecodeSequenceLayoutPlanRequest<'a> {
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub layers: &'a [CudaHfDecodeChainLayer<'a>],
    pub layer_index: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CudaHfDecodeSequenceLayoutPlan {
    pub status: i32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub layer_count: u32,
    pub layer_index: u32,
    pub attention_kind: u32,
    pub deepseek_mode: u32,
    pub deepseek_flags: u32,
    pub deepseek_hc_mult: u32,
    pub deepseek_hc_sinkhorn_iters: u32,
    pub deepseek_qk_head_dim: u32,
    pub deepseek_q_rows: u32,
    pub deepseek_kv_cache_width: u32,
    pub deepseek_kv_b_rows: u32,
    pub deepseek_value_rows: u32,
    pub resident_weight_bytes: u64,
    pub layout_bytes: u64,
    pub rms_attn: u64,
    pub rms_mlp: u64,
    pub w_q: u64,
    pub q_norm: u64,
    pub w_k: u64,
    pub k_norm: u64,
    pub w_v: u64,
    pub w_o: u64,
    pub w_router: u64,
    pub w_expert_gate_up: u64,
    pub w_expert_down: u64,
    pub deepseek_q_a_scale: u64,
    pub deepseek_q_b: u64,
    pub deepseek_q_b_scale: u64,
    pub deepseek_kv_a_scale: u64,
    pub deepseek_kv_b_scale: u64,
    pub deepseek_o_a_scale: u64,
    pub deepseek_o_b: u64,
    pub deepseek_o_b_scale: u64,
    pub deepseek_attention_sink: u64,
    pub deepseek_indexer_q: u64,
    pub deepseek_indexer_q_scale: u64,
    pub deepseek_indexer_k: u64,
    pub deepseek_indexer_k_scale: u64,
    pub deepseek_indexer_k_norm: u64,
    pub deepseek_indexer_k_norm_bias: u64,
    pub deepseek_indexer_weights: u64,
    pub deepseek_compressor_ape: u64,
    pub deepseek_compressor_wkv: u64,
    pub deepseek_compressor_wgate: u64,
    pub deepseek_compressor_norm: u64,
    pub deepseek_indexer_compressor_ape: u64,
    pub deepseek_indexer_compressor_wkv: u64,
    pub deepseek_indexer_compressor_wgate: u64,
    pub deepseek_indexer_compressor_norm: u64,
    pub deepseek_hc_head_base: u64,
    pub deepseek_hc_head_fn: u64,
    pub deepseek_hc_head_scale: u64,
    pub deepseek_hc_attn_base: u64,
    pub deepseek_hc_attn_fn: u64,
    pub deepseek_hc_attn_scale: u64,
    pub deepseek_hc_ffn_base: u64,
    pub deepseek_hc_ffn_fn: u64,
    pub deepseek_hc_ffn_scale: u64,
    pub deepseek_hc_eps: f32,
    pub deepseek_hc_post_alpha: f32,
    pub deepseek_compress_rope_theta: f32,
    pub deepseek_swiglu_limit: f32,
}

impl<'a> CudaHfDecodeSequenceLayoutPlanRequest<'a> {
    pub fn plan(&self) -> Result<CudaHfDecodeSequenceLayoutPlan, String> {
        if self.layers.is_empty() {
            return Err("layout plan requires at least one layer".to_string());
        }
        if self.layer_index as usize >= self.layers.len() {
            return Err("layout plan layer_index is outside layer_count".to_string());
        }
        let ffi_layers = self
            .layers
            .iter()
            .map(CudaHfDecodeChainLayer::to_descriptor_layout_ffi)
            .collect::<Vec<_>>();
        let request = NervaCudaHfDecodeSequenceLayoutPlanRequest {
            hidden: self.hidden,
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            intermediate: self.intermediate,
            vocab_size: self.vocab_size,
            layer_count: ffi_layers.len() as u32,
            layer_index: self.layer_index,
            layers: ffi_layers.as_ptr(),
        };
        let mut out = NervaCudaHfDecodeSequenceLayoutPlanResult::default();
        let code = plan_hf_decode_sequence_layout(&request, &mut out);
        if code != 0 || out.status != 0 {
            return Err(format!(
                "native sequence layout planning failed: code={code} status={}",
                out.status
            ));
        }
        Ok(out.into())
    }
}

impl From<NervaCudaHfDecodeSequenceLayoutPlanResult> for CudaHfDecodeSequenceLayoutPlan {
    fn from(value: NervaCudaHfDecodeSequenceLayoutPlanResult) -> Self {
        Self {
            status: value.status,
            hidden: value.hidden,
            heads: value.heads,
            kv_heads: value.kv_heads,
            head_dim: value.head_dim,
            intermediate: value.intermediate,
            vocab_size: value.vocab_size,
            layer_count: value.layer_count,
            layer_index: value.layer_index,
            attention_kind: value.attention_kind,
            deepseek_mode: value.deepseek_mode,
            deepseek_flags: value.deepseek_flags,
            deepseek_hc_mult: value.deepseek_hc_mult,
            deepseek_hc_sinkhorn_iters: value.deepseek_hc_sinkhorn_iters,
            deepseek_qk_head_dim: value.deepseek_qk_head_dim,
            deepseek_q_rows: value.deepseek_q_rows,
            deepseek_kv_cache_width: value.deepseek_kv_cache_width,
            deepseek_kv_b_rows: value.deepseek_kv_b_rows,
            deepseek_value_rows: value.deepseek_value_rows,
            resident_weight_bytes: value.resident_weight_bytes,
            layout_bytes: value.layout_bytes,
            rms_attn: value.rms_attn,
            rms_mlp: value.rms_mlp,
            w_q: value.w_q,
            q_norm: value.q_norm,
            w_k: value.w_k,
            k_norm: value.k_norm,
            w_v: value.w_v,
            w_o: value.w_o,
            w_router: value.w_router,
            w_expert_gate_up: value.w_expert_gate_up,
            w_expert_down: value.w_expert_down,
            deepseek_q_a_scale: value.deepseek_q_a_scale,
            deepseek_q_b: value.deepseek_q_b,
            deepseek_q_b_scale: value.deepseek_q_b_scale,
            deepseek_kv_a_scale: value.deepseek_kv_a_scale,
            deepseek_kv_b_scale: value.deepseek_kv_b_scale,
            deepseek_o_a_scale: value.deepseek_o_a_scale,
            deepseek_o_b: value.deepseek_o_b,
            deepseek_o_b_scale: value.deepseek_o_b_scale,
            deepseek_attention_sink: value.deepseek_attention_sink,
            deepseek_indexer_q: value.deepseek_indexer_q,
            deepseek_indexer_q_scale: value.deepseek_indexer_q_scale,
            deepseek_indexer_k: value.deepseek_indexer_k,
            deepseek_indexer_k_scale: value.deepseek_indexer_k_scale,
            deepseek_indexer_k_norm: value.deepseek_indexer_k_norm,
            deepseek_indexer_k_norm_bias: value.deepseek_indexer_k_norm_bias,
            deepseek_indexer_weights: value.deepseek_indexer_weights,
            deepseek_compressor_ape: value.deepseek_compressor_ape,
            deepseek_compressor_wkv: value.deepseek_compressor_wkv,
            deepseek_compressor_wgate: value.deepseek_compressor_wgate,
            deepseek_compressor_norm: value.deepseek_compressor_norm,
            deepseek_indexer_compressor_ape: value.deepseek_indexer_compressor_ape,
            deepseek_indexer_compressor_wkv: value.deepseek_indexer_compressor_wkv,
            deepseek_indexer_compressor_wgate: value.deepseek_indexer_compressor_wgate,
            deepseek_indexer_compressor_norm: value.deepseek_indexer_compressor_norm,
            deepseek_hc_head_base: value.deepseek_hc_head_base,
            deepseek_hc_head_fn: value.deepseek_hc_head_fn,
            deepseek_hc_head_scale: value.deepseek_hc_head_scale,
            deepseek_hc_attn_base: value.deepseek_hc_attn_base,
            deepseek_hc_attn_fn: value.deepseek_hc_attn_fn,
            deepseek_hc_attn_scale: value.deepseek_hc_attn_scale,
            deepseek_hc_ffn_base: value.deepseek_hc_ffn_base,
            deepseek_hc_ffn_fn: value.deepseek_hc_ffn_fn,
            deepseek_hc_ffn_scale: value.deepseek_hc_ffn_scale,
            deepseek_hc_eps: value.deepseek_hc_eps,
            deepseek_hc_post_alpha: value.deepseek_hc_post_alpha,
            deepseek_compress_rope_theta: value.deepseek_compress_rope_theta,
            deepseek_swiglu_limit: value.deepseek_swiglu_limit,
        }
    }
}
