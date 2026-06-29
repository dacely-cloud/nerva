use core::ptr;

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;

pub const CUDA_HF_MLP_DENSE: u32 = 0;
pub const CUDA_HF_MLP_SPARSE_MOE: u32 = 1;
pub const CUDA_HF_ATTENTION_FULL: u32 = 0;
pub const CUDA_HF_ATTENTION_LINEAR_GDN: u32 = 1;
pub const CUDA_HF_MOE_EXPERTS_MAX: usize = 256;
pub const CUDA_HF_MOE_TOP_K_MAX: usize = 16;

#[derive(Clone, Debug)]
pub struct CudaHfDecodeChainLayer<'a> {
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
    pub w_gate: &'a [u16],
    pub w_up: &'a [u16],
    pub w_down: &'a [u16],
    pub w_router: Option<&'a [u16]>,
    pub w_expert_gate_up: Option<&'a [u16]>,
    pub w_expert_down: Option<&'a [u16]>,
    pub w_shared_expert_gate: Option<&'a [u16]>,
    pub w_shared_expert_up: Option<&'a [u16]>,
    pub w_shared_expert_down: Option<&'a [u16]>,
    pub w_shared_expert_router: Option<&'a [u16]>,
    pub linear_gdn: Option<CudaHfLinearGdnLayer<'a>>,
    pub mlp_kind: u32,
    pub moe_intermediate: usize,
    pub shared_expert_intermediate: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub norm_topk_prob: bool,
    pub attention_kind: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaHfLinearGdnLayer<'a> {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    pub w_conv: &'a [u16],
    pub w_qkv: &'a [u16],
    pub w_z: &'a [u16],
    pub w_b: &'a [u16],
    pub w_a: &'a [u16],
    pub dt_bias: &'a [u16],
    pub a_log: &'a [f32],
    pub norm_weight: &'a [u16],
    pub w_out: &'a [u16],
}

impl<'a> CudaHfDecodeChainLayer<'a> {
    pub(crate) fn validate(
        &self,
        hidden: usize,
        attention_hidden: usize,
        kv_hidden: usize,
        head_dim: usize,
        intermediate: usize,
    ) -> Option<String> {
        let attention_error = match self.attention_kind {
            CUDA_HF_ATTENTION_FULL => {
                self.validate_full_attention(hidden, attention_hidden, kv_hidden, head_dim)
            }
            CUDA_HF_ATTENTION_LINEAR_GDN => self.validate_linear_gdn(hidden),
            other => Some(format!(
                "CUDA HF decode chain unsupported attention kind {other}"
            )),
        };
        let mlp_error = match self.mlp_kind {
            CUDA_HF_MLP_DENSE => self.validate_dense_mlp(hidden, intermediate),
            CUDA_HF_MLP_SPARSE_MOE => self.validate_sparse_moe_mlp(hidden, intermediate),
            other => Some(format!("CUDA HF decode chain unsupported MLP kind {other}")),
        };
        attention_error
            .or(mlp_error)
            .or_else(|| validate_optional("q_bias", self.q_bias, attention_hidden))
            .or_else(|| validate_optional("k_bias", self.k_bias, kv_hidden))
            .or_else(|| validate_optional("v_bias", self.v_bias, kv_hidden))
            .or_else(|| validate_optional("o_bias", self.o_bias, hidden))
            .or_else(|| validate_optional("q_norm_weight", self.q_norm_weight, head_dim))
            .or_else(|| validate_optional("k_norm_weight", self.k_norm_weight, head_dim))
            .or_else(|| validate_optional("w_q_gate", self.w_q_gate, attention_hidden * hidden))
    }

    pub(crate) fn to_ffi(&self) -> NervaCudaHfDecodeChainLayer {
        NervaCudaHfDecodeChainLayer {
            rms_attn_weight: self.rms_attn_weight.as_ptr(),
            rms_mlp_weight: self.rms_mlp_weight.as_ptr(),
            w_q: self.w_q.as_ptr(),
            w_q_gate: optional_ptr(self.w_q_gate),
            w_k: self.w_k.as_ptr(),
            q_norm_weight: optional_ptr(self.q_norm_weight),
            k_norm_weight: optional_ptr(self.k_norm_weight),
            w_v: self.w_v.as_ptr(),
            w_o: self.w_o.as_ptr(),
            q_bias: optional_ptr(self.q_bias),
            k_bias: optional_ptr(self.k_bias),
            v_bias: optional_ptr(self.v_bias),
            o_bias: optional_ptr(self.o_bias),
            w_gate: self.w_gate.as_ptr(),
            w_up: self.w_up.as_ptr(),
            w_down: self.w_down.as_ptr(),
            w_router: optional_ptr(self.w_router),
            w_expert_gate_up: optional_ptr(self.w_expert_gate_up),
            w_expert_down: optional_ptr(self.w_expert_down),
            w_shared_expert_gate: optional_ptr(self.w_shared_expert_gate),
            w_shared_expert_up: optional_ptr(self.w_shared_expert_up),
            w_shared_expert_down: optional_ptr(self.w_shared_expert_down),
            w_shared_expert_router: optional_ptr(self.w_shared_expert_router),
            linear_key_heads: self.linear_gdn.map_or(0, |gdn| gdn.key_heads as u32),
            linear_value_heads: self.linear_gdn.map_or(0, |gdn| gdn.value_heads as u32),
            linear_key_head_dim: self.linear_gdn.map_or(0, |gdn| gdn.key_head_dim as u32),
            linear_value_head_dim: self.linear_gdn.map_or(0, |gdn| gdn.value_head_dim as u32),
            linear_conv_kernel: self.linear_gdn.map_or(0, |gdn| gdn.conv_kernel as u32),
            w_linear_conv: self
                .linear_gdn
                .map_or(ptr::null(), |gdn| gdn.w_conv.as_ptr()),
            w_linear_qkv: self
                .linear_gdn
                .map_or(ptr::null(), |gdn| gdn.w_qkv.as_ptr()),
            w_linear_z: self.linear_gdn.map_or(ptr::null(), |gdn| gdn.w_z.as_ptr()),
            w_linear_b: self.linear_gdn.map_or(ptr::null(), |gdn| gdn.w_b.as_ptr()),
            w_linear_a: self.linear_gdn.map_or(ptr::null(), |gdn| gdn.w_a.as_ptr()),
            w_linear_dt_bias: self
                .linear_gdn
                .map_or(ptr::null(), |gdn| gdn.dt_bias.as_ptr()),
            w_linear_a_log: self
                .linear_gdn
                .map_or(ptr::null(), |gdn| gdn.a_log.as_ptr()),
            w_linear_norm: self
                .linear_gdn
                .map_or(ptr::null(), |gdn| gdn.norm_weight.as_ptr()),
            w_linear_out: self
                .linear_gdn
                .map_or(ptr::null(), |gdn| gdn.w_out.as_ptr()),
            mlp_kind: self.mlp_kind,
            moe_intermediate: self.moe_intermediate as u32,
            shared_expert_intermediate: self.shared_expert_intermediate as u32,
            num_experts: self.num_experts as u32,
            experts_per_token: self.experts_per_token as u32,
            norm_topk_prob: self.norm_topk_prob as u32,
            attention_kind: self.attention_kind,
        }
    }

    pub(crate) fn to_descriptor_layout_ffi(&self) -> NervaCudaHfDecodeChainLayer {
        NervaCudaHfDecodeChainLayer {
            rms_attn_weight: ptr::null(),
            rms_mlp_weight: ptr::null(),
            w_q: ptr::null(),
            w_q_gate: optional_ptr(self.w_q_gate),
            w_k: ptr::null(),
            q_norm_weight: optional_ptr(self.q_norm_weight),
            k_norm_weight: optional_ptr(self.k_norm_weight),
            w_v: ptr::null(),
            w_o: ptr::null(),
            q_bias: optional_ptr(self.q_bias),
            k_bias: optional_ptr(self.k_bias),
            v_bias: optional_ptr(self.v_bias),
            o_bias: optional_ptr(self.o_bias),
            w_gate: ptr::null(),
            w_up: ptr::null(),
            w_down: ptr::null(),
            w_router: ptr::null(),
            w_expert_gate_up: ptr::null(),
            w_expert_down: ptr::null(),
            w_shared_expert_gate: ptr::null(),
            w_shared_expert_up: ptr::null(),
            w_shared_expert_down: ptr::null(),
            w_shared_expert_router: ptr::null(),
            linear_key_heads: self.linear_gdn.map_or(0, |gdn| gdn.key_heads as u32),
            linear_value_heads: self.linear_gdn.map_or(0, |gdn| gdn.value_heads as u32),
            linear_key_head_dim: self.linear_gdn.map_or(0, |gdn| gdn.key_head_dim as u32),
            linear_value_head_dim: self.linear_gdn.map_or(0, |gdn| gdn.value_head_dim as u32),
            linear_conv_kernel: self.linear_gdn.map_or(0, |gdn| gdn.conv_kernel as u32),
            w_linear_conv: ptr::null(),
            w_linear_qkv: ptr::null(),
            w_linear_z: ptr::null(),
            w_linear_b: ptr::null(),
            w_linear_a: ptr::null(),
            w_linear_dt_bias: ptr::null(),
            w_linear_a_log: ptr::null(),
            w_linear_norm: ptr::null(),
            w_linear_out: ptr::null(),
            mlp_kind: self.mlp_kind,
            moe_intermediate: self.moe_intermediate as u32,
            shared_expert_intermediate: self.shared_expert_intermediate as u32,
            num_experts: self.num_experts as u32,
            experts_per_token: self.experts_per_token as u32,
            norm_topk_prob: self.norm_topk_prob as u32,
            attention_kind: self.attention_kind,
        }
    }

    fn validate_full_attention(
        &self,
        hidden: usize,
        attention_hidden: usize,
        kv_hidden: usize,
        head_dim: usize,
    ) -> Option<String> {
        if self.linear_gdn.is_some() {
            return Some(
                "CUDA HF decode chain full attention cannot carry linear GDN weights".to_string(),
            );
        }
        for (name, actual, expected) in
            self.full_attention_required_lengths(hidden, attention_hidden, kv_hidden)
        {
            if actual != expected {
                return Some(format!(
                    "CUDA HF decode chain {name} length {actual} != {expected}"
                ));
            }
        }
        validate_optional("q_bias", self.q_bias, attention_hidden)
            .or_else(|| validate_optional("k_bias", self.k_bias, kv_hidden))
            .or_else(|| validate_optional("v_bias", self.v_bias, kv_hidden))
            .or_else(|| validate_optional("o_bias", self.o_bias, hidden))
            .or_else(|| validate_optional("q_norm_weight", self.q_norm_weight, head_dim))
            .or_else(|| validate_optional("k_norm_weight", self.k_norm_weight, head_dim))
            .or_else(|| validate_optional("w_q_gate", self.w_q_gate, attention_hidden * hidden))
    }

    fn validate_linear_gdn(&self, hidden: usize) -> Option<String> {
        if self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len() != 0 {
            return Some(
                "CUDA HF decode chain linear GDN layer cannot carry full-attention QKV/O weights"
                    .to_string(),
            );
        }
        let Some(gdn) = self.linear_gdn else {
            return Some("CUDA HF decode chain linear GDN layer is missing weights".to_string());
        };
        if gdn.key_heads == 0
            || gdn.value_heads == 0
            || gdn.key_head_dim == 0
            || gdn.value_head_dim == 0
            || gdn.conv_kernel == 0
        {
            return Some("CUDA HF decode chain linear GDN dimensions must be non-zero".to_string());
        }
        let key_dim = match checked_mul(gdn.key_heads, gdn.key_head_dim, "linear GDN key dim") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let value_dim =
            match checked_mul(gdn.value_heads, gdn.value_head_dim, "linear GDN value dim") {
                Ok(value) => value,
                Err(error) => return Some(error),
            };
        let key_conv_dim = match checked_mul(key_dim, 2, "linear GDN key conv dim") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let conv_dim = match checked_add(key_conv_dim, value_dim, "linear GDN conv dim") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let linear_conv = match checked_mul(conv_dim, gdn.conv_kernel, "linear GDN conv weight") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let linear_qkv = match checked_mul(conv_dim, hidden, "linear GDN qkv weight") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let linear_z = match checked_mul(value_dim, hidden, "linear GDN z weight") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let linear_ba = match checked_mul(gdn.value_heads, hidden, "linear GDN BA weight") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        let linear_out = match checked_mul(hidden, value_dim, "linear GDN output weight") {
            Ok(value) => value,
            Err(error) => return Some(error),
        };
        for (name, actual, expected) in [
            ("linear_conv", gdn.w_conv.len(), linear_conv),
            ("linear_qkv", gdn.w_qkv.len(), linear_qkv),
            ("linear_z", gdn.w_z.len(), linear_z),
            ("linear_b", gdn.w_b.len(), linear_ba),
            ("linear_a", gdn.w_a.len(), linear_ba),
            ("linear_dt_bias", gdn.dt_bias.len(), gdn.value_heads),
            ("linear_a_log", gdn.a_log.len(), gdn.value_heads),
            ("linear_norm", gdn.norm_weight.len(), gdn.value_head_dim * 2),
            ("linear_out", gdn.w_out.len(), linear_out),
        ] {
            if actual != expected {
                return Some(format!(
                    "CUDA HF decode chain {name} length {actual} != {expected}"
                ));
            }
        }
        None
    }

    fn full_attention_required_lengths(
        &self,
        hidden: usize,
        attention_hidden: usize,
        kv_hidden: usize,
    ) -> [(&'static str, usize, usize); 6] {
        [
            ("rms_attn_weight", self.rms_attn_weight.len(), hidden),
            ("rms_mlp_weight", self.rms_mlp_weight.len(), hidden),
            ("w_q", self.w_q.len(), attention_hidden * hidden),
            ("w_k", self.w_k.len(), kv_hidden * hidden),
            ("w_v", self.w_v.len(), kv_hidden * hidden),
            ("w_o", self.w_o.len(), hidden * attention_hidden),
        ]
    }

    fn validate_dense_mlp(&self, hidden: usize, intermediate: usize) -> Option<String> {
        for (name, actual, expected) in [
            ("w_gate", self.w_gate.len(), intermediate * hidden),
            ("w_up", self.w_up.len(), intermediate * hidden),
            ("w_down", self.w_down.len(), hidden * intermediate),
        ] {
            if actual != expected {
                return Some(format!(
                    "CUDA HF decode chain {name} length {actual} != {expected}"
                ));
            }
        }
        None
    }

    fn validate_sparse_moe_mlp(&self, hidden: usize, intermediate: usize) -> Option<String> {
        if self.moe_intermediate == 0
            || self.num_experts == 0
            || self.experts_per_token == 0
            || self.experts_per_token > self.num_experts
            || self.num_experts > CUDA_HF_MOE_EXPERTS_MAX
            || self.experts_per_token > CUDA_HF_MOE_TOP_K_MAX
        {
            return Some(
                "CUDA HF decode chain sparse MoE dimensions must be non-zero and fit native expert/top-k limits"
                    .to_string(),
            );
        }
        if self.moe_intermediate > intermediate {
            return Some(
                "CUDA HF decode chain sparse MoE intermediate exceeds scratch capacity".to_string(),
            );
        }
        if self.shared_expert_intermediate > intermediate {
            return Some(
                "CUDA HF decode chain shared expert intermediate exceeds scratch capacity"
                    .to_string(),
            );
        }
        validate_optional("w_router", self.w_router, self.num_experts * hidden)
            .or_else(|| {
                validate_optional(
                    "w_expert_gate_up",
                    self.w_expert_gate_up,
                    self.num_experts * 2 * self.moe_intermediate * hidden,
                )
            })
            .or_else(|| {
                validate_optional(
                    "w_expert_down",
                    self.w_expert_down,
                    self.num_experts * hidden * self.moe_intermediate,
                )
            })
            .or_else(|| {
                validate_optional(
                    "w_shared_expert_gate",
                    self.w_shared_expert_gate,
                    self.shared_expert_intermediate * hidden,
                )
            })
            .or_else(|| {
                validate_optional(
                    "w_shared_expert_up",
                    self.w_shared_expert_up,
                    self.shared_expert_intermediate * hidden,
                )
            })
            .or_else(|| {
                validate_optional(
                    "w_shared_expert_down",
                    self.w_shared_expert_down,
                    hidden * self.shared_expert_intermediate,
                )
            })
            .or_else(|| {
                validate_optional(
                    "w_shared_expert_router",
                    self.w_shared_expert_router,
                    usize::from(self.shared_expert_intermediate != 0) * hidden,
                )
            })
            .or_else(|| {
                if self.w_router.is_none()
                    || self.w_expert_gate_up.is_none()
                    || self.w_expert_down.is_none()
                {
                    Some(
                        "CUDA HF decode chain sparse MoE layer is missing router/expert weights"
                            .to_string(),
                    )
                } else {
                    None
                }
            })
            .or_else(|| {
                if self.shared_expert_intermediate != 0
                    && (self.w_shared_expert_gate.is_none()
                        || self.w_shared_expert_up.is_none()
                        || self.w_shared_expert_down.is_none()
                        || self.w_shared_expert_router.is_none())
                {
                    Some(
                        "CUDA HF decode chain sparse MoE layer is missing shared expert weights"
                            .to_string(),
                    )
                } else {
                    None
                }
            })
    }
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("CUDA HF decode chain {label} overflow"))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("CUDA HF decode chain {label} overflow"))
}

fn validate_optional(name: &'static str, value: Option<&[u16]>, expected: usize) -> Option<String> {
    match value {
        Some(slice) if slice.len() != expected => Some(format!(
            "CUDA HF decode chain {name} length {} != {expected}",
            slice.len()
        )),
        _ => None,
    }
}

fn optional_ptr(slice: Option<&[u16]>) -> *const u16 {
    slice.map_or(ptr::null(), <[u16]>::as_ptr)
}
