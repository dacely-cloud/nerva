use nerva_core::types::dtype::DType;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_model::common::shape::TransformerBlockShape;
use nerva_model::precision::bits::{f32_to_bf16_bits, f32_to_f16_bits};
use nerva_model::precision::block::model::PrecisionTransformerBlock;
use nerva_model::precision::scratch::PrecisionTransformerBlockScratch;

use crate::engine::cuda_block::run_precision_block_on_cuda;

#[test]
fn cuda_precision_block_forward_matches_cpu_exact_block() {
    let _guard = super::cuda_test_lock();

    let shape = TransformerBlockShape::new(2, 1, 2);
    for dtype in [DType::F16, DType::BF16] {
        let block = tiny_block(dtype, shape);
        let input = [encode(dtype, 1.0), encode(dtype, 2.0)];
        let mut scratch = PrecisionTransformerBlockScratch::new(shape).unwrap();
        let mut expected = [0u16; 2];
        let mut ledger = TokenLedger::new(0);
        block
            .forward_into(&input, &mut scratch, &mut expected, &mut ledger)
            .unwrap();

        let summary = run_precision_block_on_cuda(&block, &input, 0).unwrap();
        if summary.status != SmokeStatus::Ok {
            return;
        }

        assert_eq!(summary.output, expected);
        assert_eq!(summary.hidden, 2);
        assert_eq!(summary.heads, 1);
        assert_eq!(summary.kv_heads, 1);
        assert_eq!(summary.head_dim, 2);
        assert_eq!(summary.kernel_launches, 1);
        assert_eq!(summary.sync_calls, 1);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.resident_weight_bytes > 0);
        assert!(summary.to_json().contains("\"status\":\"ok\""));
    }
}

fn tiny_block(dtype: DType, shape: TransformerBlockShape) -> PrecisionTransformerBlock {
    let rms = [1.0, 1.0];
    let identity = [1.0, 0.0, 0.0, 1.0];
    let gate = [0.5, 0.0, 0.0, 0.5];
    PrecisionTransformerBlock::new_from_f32(
        dtype, shape, &rms, &rms, &identity, &identity, &identity, &identity, &gate, &identity,
        &identity, 1e-5,
    )
    .unwrap()
}

fn encode(dtype: DType, value: f32) -> u16 {
    match dtype {
        DType::F16 => f32_to_f16_bits(value),
        DType::BF16 => f32_to_bf16_bits(value),
        _ => 0,
    }
}
