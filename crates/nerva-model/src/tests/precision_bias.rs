use nerva_core::types::dtype::DType;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::f32_to_f16_bits;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::scratch::PrecisionTransformerBlockScratch;

#[test]
fn precision_block_applies_attention_output_bias() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let rms = [1.0, 1.0];
    let zero_matrix = [0.0, 0.0, 0.0, 0.0];
    let mut block = PrecisionTransformerBlock::new_from_f32(
        DType::F16,
        shape,
        &rms,
        &rms,
        &zero_matrix,
        &zero_matrix,
        &zero_matrix,
        &zero_matrix,
        &zero_matrix,
        &zero_matrix,
        &zero_matrix,
        1e-5,
    )
    .unwrap();
    block = block
        .with_attention_biases(
            vec![0; 2],
            vec![0; 2],
            vec![0; 2],
            vec![f32_to_f16_bits(0.25), f32_to_f16_bits(-0.5)],
        )
        .unwrap();

    let input = [f32_to_f16_bits(1.0), f32_to_f16_bits(2.0)];
    let mut output = [0u16; 2];
    let mut scratch = PrecisionTransformerBlockScratch::new(shape).unwrap();
    let mut ledger = TokenLedger::new(0);

    block
        .forward_into(&input, &mut scratch, &mut output, &mut ledger)
        .unwrap();

    assert_eq!(output, [f32_to_f16_bits(1.25), f32_to_f16_bits(1.5)]);
    assert_eq!(ledger.hot_path_allocations, 0);
}
