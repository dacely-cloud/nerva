use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::{decode_f32_for_dtype, dtype_label, encode_f32_for_dtype, hash_u16s};
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::scratch::PrecisionTransformerBlockScratch;
use crate::reference::block::ReferenceTransformerBlock;
use crate::reference::scratch::TransformerBlockScratch;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrecisionBlockSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PrecisionDTypeBlockSmokeSummary {
    pub dtype: DType,
    pub bit_parity: bool,
    pub output_bits: [u16; 2],
    pub expected_bits: [u16; 2],
    pub output_hash: u64,
    pub expected_hash: u64,
    pub max_abs_error: f32,
    pub hot_path_allocations: u64,
}

impl PrecisionDTypeBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let dtype = dtype_label(self.dtype).unwrap_or("unsupported");
        format!(
            "{{\"dtype\":\"{}\",\"bit_parity\":{},\"output_bits\":[{},{}],\"expected_bits\":[{},{}],\"output_hash\":{},\"expected_hash\":{},\"max_abs_error\":{},\"hot_path_allocations\":{}}}",
            dtype,
            self.bit_parity,
            self.output_bits[0],
            self.output_bits[1],
            self.expected_bits[0],
            self.expected_bits[1],
            self.output_hash,
            self.expected_hash,
            self.max_abs_error,
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PrecisionBlockSmokeSummary {
    pub status: PrecisionBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub f16: PrecisionDTypeBlockSmokeSummary,
    pub bf16: PrecisionDTypeBlockSmokeSummary,
}

impl PrecisionBlockSmokeSummary {
    pub fn passed(self) -> bool {
        self.f16.bit_parity
            && self.bf16.bit_parity
            && self.f16.hot_path_allocations == 0
            && self.bf16.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            PrecisionBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"f16\":{},\"bf16\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.f16.to_json(),
            self.bf16.to_json(),
        )
    }
}

pub fn precision_block_smoke() -> Result<PrecisionBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let f16 = run_dtype_smoke(DType::F16, shape)?;
    let bf16 = run_dtype_smoke(DType::BF16, shape)?;
    let summary = PrecisionBlockSmokeSummary {
        status: PrecisionBlockSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        intermediate: shape.intermediate,
        f16,
        bf16,
    };
    if summary.passed() {
        Ok(summary)
    } else {
        Err(NervaError::InvalidArgument {
            reason: "FP16/BF16 precision block bit parity failed".to_string(),
        })
    }
}

fn run_dtype_smoke(
    dtype: DType,
    shape: TransformerBlockShape,
) -> Result<PrecisionDTypeBlockSmokeSummary> {
    let weights = smoke_weights();
    let block = PrecisionTransformerBlock::new_from_f32(
        dtype,
        shape,
        &weights.rms_attn_weight,
        &weights.rms_mlp_weight,
        &weights.w_q,
        &weights.w_k,
        &weights.w_v,
        &weights.w_o,
        &weights.w_gate,
        &weights.w_up,
        &weights.w_down,
        weights.rms_eps,
    )?;
    let reference = ReferenceTransformerBlock::new(
        shape,
        weights.rms_attn_weight.clone(),
        weights.rms_mlp_weight.clone(),
        weights.w_q.clone(),
        weights.w_k.clone(),
        weights.w_v.clone(),
        weights.w_o.clone(),
        weights.w_gate.clone(),
        weights.w_up.clone(),
        weights.w_down.clone(),
        weights.rms_eps,
    )?;

    let input_f32 = [1.0, 2.0];
    let input = [
        encode_f32_for_dtype(input_f32[0], dtype)?,
        encode_f32_for_dtype(input_f32[1], dtype)?,
    ];
    let mut scratch = PrecisionTransformerBlockScratch::new(shape)?;
    let mut output_bits = [0u16; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output_bits, &mut ledger)?;
    ledger.require_zero_hot_path_allocations()?;

    let mut reference_scratch = TransformerBlockScratch::new(shape)?;
    let mut reference_output = [0.0f32; 2];
    let mut reference_ledger = TokenLedger::new(0);
    reference.forward_into(
        &input_f32,
        &mut reference_scratch,
        &mut reference_output,
        &mut reference_ledger,
    )?;
    let expected_bits = [
        encode_f32_for_dtype(reference_output[0], dtype)?,
        encode_f32_for_dtype(reference_output[1], dtype)?,
    ];
    let output_f32 = [
        decode_f32_for_dtype(output_bits[0], dtype)?,
        decode_f32_for_dtype(output_bits[1], dtype)?,
    ];
    let max_abs_error = output_f32
        .iter()
        .zip(reference_output.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);

    Ok(PrecisionDTypeBlockSmokeSummary {
        dtype,
        bit_parity: output_bits == expected_bits,
        output_bits,
        expected_bits,
        output_hash: hash_u16s(&output_bits),
        expected_hash: hash_u16s(&expected_bits),
        max_abs_error,
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

#[derive(Clone, Debug)]
struct SmokeWeights {
    rms_attn_weight: Vec<f32>,
    rms_mlp_weight: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    rms_eps: f32,
}

fn smoke_weights() -> SmokeWeights {
    SmokeWeights {
        rms_attn_weight: vec![1.0, 1.0],
        rms_mlp_weight: vec![1.0, 1.0],
        w_q: vec![1.0, 0.0, 0.0, 1.0],
        w_k: vec![1.0, 0.0, 0.0, 1.0],
        w_v: vec![1.0, 0.0, 0.0, 1.0],
        w_o: vec![1.0, 0.0, 0.0, 1.0],
        w_gate: vec![0.5, 0.0, 0.0, 0.5],
        w_up: vec![1.0, 0.0, 0.0, 1.0],
        w_down: vec![1.0, 0.0, 0.0, 1.0],
        rms_eps: 1e-5,
    }
}
