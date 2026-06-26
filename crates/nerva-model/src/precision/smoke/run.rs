use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::precision::smoke::dtype::run_dtype_smoke;
use crate::precision::smoke::status::PrecisionBlockSmokeStatus;
use crate::precision::smoke::summary::PrecisionBlockSmokeSummary;

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
