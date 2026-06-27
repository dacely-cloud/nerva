use core::ptr;

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;

#[derive(Clone, Debug)]
pub struct CudaHfDecodeChainLayer<'a> {
    pub rms_attn_weight: &'a [u16],
    pub rms_mlp_weight: &'a [u16],
    pub w_q: &'a [u16],
    pub w_k: &'a [u16],
    pub w_v: &'a [u16],
    pub w_o: &'a [u16],
    pub q_bias: Option<&'a [u16]>,
    pub k_bias: Option<&'a [u16]>,
    pub v_bias: Option<&'a [u16]>,
    pub o_bias: Option<&'a [u16]>,
    pub w_gate: &'a [u16],
    pub w_up: &'a [u16],
    pub w_down: &'a [u16],
}

impl<'a> CudaHfDecodeChainLayer<'a> {
    pub(crate) fn validate(
        &self,
        hidden: usize,
        attention_hidden: usize,
        kv_hidden: usize,
        intermediate: usize,
    ) -> Option<String> {
        for (name, actual, expected) in
            self.required_lengths(hidden, attention_hidden, kv_hidden, intermediate)
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
    }

    pub(crate) fn to_ffi(&self) -> NervaCudaHfDecodeChainLayer {
        NervaCudaHfDecodeChainLayer {
            rms_attn_weight: self.rms_attn_weight.as_ptr(),
            rms_mlp_weight: self.rms_mlp_weight.as_ptr(),
            w_q: self.w_q.as_ptr(),
            w_k: self.w_k.as_ptr(),
            w_v: self.w_v.as_ptr(),
            w_o: self.w_o.as_ptr(),
            q_bias: optional_ptr(self.q_bias),
            k_bias: optional_ptr(self.k_bias),
            v_bias: optional_ptr(self.v_bias),
            o_bias: optional_ptr(self.o_bias),
            w_gate: self.w_gate.as_ptr(),
            w_up: self.w_up.as_ptr(),
            w_down: self.w_down.as_ptr(),
        }
    }

    pub(crate) fn to_descriptor_layout_ffi(&self) -> NervaCudaHfDecodeChainLayer {
        NervaCudaHfDecodeChainLayer {
            rms_attn_weight: ptr::null(),
            rms_mlp_weight: ptr::null(),
            w_q: ptr::null(),
            w_k: ptr::null(),
            w_v: ptr::null(),
            w_o: ptr::null(),
            q_bias: optional_ptr(self.q_bias),
            k_bias: optional_ptr(self.k_bias),
            v_bias: optional_ptr(self.v_bias),
            o_bias: optional_ptr(self.o_bias),
            w_gate: ptr::null(),
            w_up: ptr::null(),
            w_down: ptr::null(),
        }
    }

    fn required_lengths(
        &self,
        hidden: usize,
        attention_hidden: usize,
        kv_hidden: usize,
        intermediate: usize,
    ) -> [(&'static str, usize, usize); 9] {
        [
            ("rms_attn_weight", self.rms_attn_weight.len(), hidden),
            ("rms_mlp_weight", self.rms_mlp_weight.len(), hidden),
            ("w_q", self.w_q.len(), attention_hidden * hidden),
            ("w_k", self.w_k.len(), kv_hidden * hidden),
            ("w_v", self.w_v.len(), kv_hidden * hidden),
            ("w_o", self.w_o.len(), hidden * attention_hidden),
            ("w_gate", self.w_gate.len(), intermediate * hidden),
            ("w_up", self.w_up.len(), intermediate * hidden),
            ("w_down", self.w_down.len(), hidden * intermediate),
        ]
    }
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
