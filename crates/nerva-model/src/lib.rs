#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{BlockKind, DType, MemoryTier, NervaError, Result};
use nerva_ledger::TokenLedger;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransformerBlockShape {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
}

impl TransformerBlockShape {
    pub const fn new(hidden: usize, heads: usize, intermediate: usize) -> Self {
        Self {
            hidden,
            heads,
            intermediate,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.hidden == 0 || self.heads == 0 || self.intermediate == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transformer block dimensions must be non-zero".to_string(),
            });
        }
        if !self.hidden.is_multiple_of(self.heads) {
            return Err(NervaError::InvalidArgument {
                reason: "hidden size must be divisible by head count".to_string(),
            });
        }
        Ok(())
    }

    pub const fn head_dim(self) -> usize {
        self.hidden / self.heads
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceTransformerBlock {
    shape: TransformerBlockShape,
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

impl ReferenceTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shape: TransformerBlockShape,
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
    ) -> Result<Self> {
        shape.validate()?;
        require_len("rms_attn_weight", rms_attn_weight.len(), shape.hidden)?;
        require_len("rms_mlp_weight", rms_mlp_weight.len(), shape.hidden)?;
        require_len("w_q", w_q.len(), shape.hidden * shape.hidden)?;
        require_len("w_k", w_k.len(), shape.hidden * shape.hidden)?;
        require_len("w_v", w_v.len(), shape.hidden * shape.hidden)?;
        require_len("w_o", w_o.len(), shape.hidden * shape.hidden)?;
        require_len("w_gate", w_gate.len(), shape.intermediate * shape.hidden)?;
        require_len("w_up", w_up.len(), shape.intermediate * shape.hidden)?;
        require_len("w_down", w_down.len(), shape.hidden * shape.intermediate)?;
        if rms_eps <= 0.0 || !rms_eps.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "rms epsilon must be positive and finite".to_string(),
            });
        }
        Ok(Self {
            shape,
            rms_attn_weight,
            rms_mlp_weight,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            rms_eps,
        })
    }

    pub fn zero_for_shape(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Self::new(
            shape,
            vec![1.0; shape.hidden],
            vec![1.0; shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.hidden * shape.intermediate],
            1e-5,
        )
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub fn forward_into(
        &self,
        input: &[f32],
        scratch: &mut TransformerBlockScratch,
        output: &mut [f32],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        let shape = self.shape;
        require_len("input", input.len(), shape.hidden)?;
        require_len("output", output.len(), shape.hidden)?;
        scratch.require_shape(shape)?;

        rms_norm_into(
            input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        );
        mat_vec_row_major(&self.w_q, &scratch.attn_norm, &mut scratch.q);
        mat_vec_row_major(&self.w_k, &scratch.attn_norm, &mut scratch.k);
        mat_vec_row_major(&self.w_v, &scratch.attn_norm, &mut scratch.v);

        single_token_attention(shape, &scratch.q, &scratch.k, &scratch.v, &mut scratch.attn);
        mat_vec_row_major(&self.w_o, &scratch.attn, output);
        for (out, residual) in output.iter_mut().zip(input.iter().copied()) {
            *out += residual;
        }

        rms_norm_into(
            output,
            &self.rms_mlp_weight,
            self.rms_eps,
            &mut scratch.mlp_norm,
        );
        mat_vec_row_major(&self.w_gate, &scratch.mlp_norm, &mut scratch.gate);
        mat_vec_row_major(&self.w_up, &scratch.mlp_norm, &mut scratch.up);
        for ((ff, gate), up) in scratch
            .ff
            .iter_mut()
            .zip(scratch.gate.iter().copied())
            .zip(scratch.up.iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_row_major(&self.w_down, &scratch.ff, &mut scratch.down);
        for (out, mlp) in output.iter_mut().zip(scratch.down.iter().copied()) {
            *out += mlp;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TransformerBlockScratch {
    shape: TransformerBlockShape,
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ff: Vec<f32>,
    down: Vec<f32>,
}

impl TransformerBlockScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            attn_norm: vec![0.0; shape.hidden],
            mlp_norm: vec![0.0; shape.hidden],
            q: vec![0.0; shape.hidden],
            k: vec![0.0; shape.hidden],
            v: vec![0.0; shape.hidden],
            attn: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            ff: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
        })
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    fn require_shape(&self, shape: TransformerBlockShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "transformer block scratch shape does not match block shape".to_string(),
            })
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ModelBlockContract {
    pub block_kind: BlockKind,
    pub weight_dtype: DType,
    pub activation_dtype: DType,
    pub weight_tier: MemoryTier,
    pub activation_tier: MemoryTier,
}

impl ModelBlockContract {
    pub const fn reference_f32() -> Self {
        Self {
            block_kind: BlockKind::Weight,
            weight_dtype: DType::F32,
            activation_dtype: DType::F32,
            weight_tier: MemoryTier::Dram,
            activation_tier: MemoryTier::Dram,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceBlockSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReferenceBlockSmokeSummary {
    pub status: ReferenceBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub hot_path_allocations: u64,
}

impl ReferenceBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            ReferenceBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"output\":[{},{}],\"output_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.hot_path_allocations,
        )
    }
}

pub fn reference_block_smoke() -> Result<ReferenceBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )?;
    let input = [1.0, 2.0];
    let mut scratch = TransformerBlockScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output, &mut ledger)?;
    Ok(ReferenceBlockSmokeSummary {
        status: ReferenceBlockSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        intermediate: shape.intermediate,
        output,
        output_hash: hash_f32s(&output),
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

fn require_len(label: &'static str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("{label} length {got} does not match expected {expected}"),
        })
    }
}

fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for ((out, value), weight) in output
        .iter_mut()
        .zip(input.iter().copied())
        .zip(weight.iter().copied())
    {
        *out = value * scale * weight;
    }
}

fn mat_vec_row_major(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    let cols = input.len();
    for (row, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        *out = row
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
}

fn single_token_attention(
    shape: TransformerBlockShape,
    _q: &[f32],
    _k: &[f32],
    v: &[f32],
    output: &mut [f32],
) {
    let head_dim = shape.head_dim();
    for head in 0..shape.heads {
        let start = head * head_dim;
        let end = start + head_dim;
        output[start..end].copy_from_slice(&v[start..end]);
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn hash_f32s(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_block_preserves_residual() {
        let shape = TransformerBlockShape::new(4, 2, 8);
        let block = ReferenceTransformerBlock::zero_for_shape(shape).unwrap();
        let mut scratch = TransformerBlockScratch::new(shape).unwrap();
        let mut output = [0.0; 4];
        let input = [1.0, -2.0, 3.0, -4.0];
        let mut ledger = TokenLedger::new(0);

        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .unwrap();

        assert_eq!(output, input);
        assert_eq!(ledger.hot_path_allocations, 0);
        assert!(ledger.require_zero_hot_path_allocations().is_ok());
    }

    #[test]
    fn nontrivial_block_matches_hand_reference() {
        let shape = TransformerBlockShape::new(2, 1, 2);
        let block = ReferenceTransformerBlock::new(
            shape,
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.0, 0.0, 0.5],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            1e-5,
        )
        .unwrap();
        let mut scratch = TransformerBlockScratch::new(shape).unwrap();
        let mut output = [0.0; 2];
        let input = [1.0, 2.0];
        let mut ledger = TokenLedger::new(7);

        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .unwrap();

        let attn_norm_scale = ((1.0_f32 + 4.0) / 2.0 + 1e-5).sqrt().recip();
        let attn = [input[0] * attn_norm_scale, input[1] * attn_norm_scale];
        let residual = [input[0] + attn[0], input[1] + attn[1]];
        let mlp_norm_scale = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0 + 1e-5)
            .sqrt()
            .recip();
        let mlp_norm = [residual[0] * mlp_norm_scale, residual[1] * mlp_norm_scale];
        let expected = [
            residual[0] + silu(0.5 * mlp_norm[0]) * mlp_norm[0],
            residual[1] + silu(0.5 * mlp_norm[1]) * mlp_norm[1],
        ];

        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(ledger.hot_path_allocations, 0);
    }

    #[test]
    fn rejects_bad_shapes_and_scratch_mismatch() {
        assert!(TransformerBlockShape::new(3, 2, 4).validate().is_err());
        let block =
            ReferenceTransformerBlock::zero_for_shape(TransformerBlockShape::new(4, 2, 8)).unwrap();
        let mut scratch =
            TransformerBlockScratch::new(TransformerBlockShape::new(2, 1, 2)).unwrap();
        let mut ledger = TokenLedger::new(0);
        let mut output = [0.0; 4];
        assert!(
            block
                .forward_into(&[0.0; 4], &mut scratch, &mut output, &mut ledger)
                .is_err()
        );
    }

    #[test]
    fn reference_block_smoke_reports_hash_and_no_allocations() {
        let summary = reference_block_smoke().unwrap();
        assert_eq!(summary.status, ReferenceBlockSmokeStatus::Ok);
        assert_eq!(summary.hidden, 2);
        assert_eq!(summary.heads, 1);
        assert_eq!(summary.intermediate, 2);
        assert_eq!(summary.hot_path_allocations, 0);
        assert_eq!(summary.output_hash, 3_850_145_622_605_741_247);
        assert!(summary.to_json().contains("\"status\":\"ok\""));
    }
}
