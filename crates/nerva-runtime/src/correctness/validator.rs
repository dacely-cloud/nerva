use nerva_core::types::error::{NervaError, Result};

use crate::correctness::case::CorrectnessCase;
use crate::correctness::exactness::ExactnessClass;
use crate::correctness::outcome::CorrectnessOutcome;

pub fn validate_correctness_case(case: CorrectnessCase) -> Result<CorrectnessOutcome> {
    if !case.exactness.accepted_for_core_runtime() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "exactness class '{}' is not accepted for core runtime validation",
                case.exactness.as_str()
            ),
        });
    }

    match case.exactness {
        ExactnessClass::BitExact => validate_bit_exact(case)?,
        ExactnessClass::ReferenceEquivalentWithinDeclaredFpTolerance
        | ExactnessClass::DistributionPreserving => validate_with_tolerance(case)?,
        ExactnessClass::Approximate => unreachable!("approximate exactness rejected above"),
    }

    Ok(CorrectnessOutcome {
        name: case.name,
        exactness: case.exactness,
        accepted: true,
    })
}

fn validate_bit_exact(case: CorrectnessCase) -> Result<()> {
    if case.expected_hash != case.observed_hash || case.max_abs_error_micros != 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!("bit-exact correctness case '{}' mismatched", case.name),
        });
    }
    Ok(())
}

fn validate_with_tolerance(case: CorrectnessCase) -> Result<()> {
    if case.max_abs_error_micros > case.tolerance_micros {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "correctness case '{}' exceeded declared tolerance",
                case.name
            ),
        });
    }
    Ok(())
}
