use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::correctness::case::CorrectnessCase;
use crate::correctness::exactness::ExactnessClass;
use crate::correctness::outcome::CorrectnessOutcome;
use crate::correctness::summary::{CorrectnessValidationStatus, CorrectnessValidationSummary};
use crate::correctness::validator::validate_correctness_case;

pub fn run_correctness_validation_probe() -> Result<CorrectnessValidationSummary> {
    let ledger = TokenLedger::new(0);
    ledger.require_zero_hot_path_allocations()?;

    let accepted_outcomes = validate_accepted_cases()?;
    let approximate_rejections = rejection_count(approximate_case());
    let bit_exact_mismatch_rejections = rejection_count(bit_exact_mismatch_case());
    let tolerance_rejections = rejection_count(tolerance_exceeded_case());

    let bit_exact_cases = count_exactness(&accepted_outcomes, ExactnessClass::BitExact);
    let fp_tolerance_cases = count_exactness(
        &accepted_outcomes,
        ExactnessClass::ReferenceEquivalentWithinDeclaredFpTolerance,
    );
    let distribution_preserving_cases =
        count_exactness(&accepted_outcomes, ExactnessClass::DistributionPreserving);

    Ok(CorrectnessValidationSummary {
        status: CorrectnessValidationStatus::Ok,
        accepted_cases: accepted_outcomes.len() as u64,
        bit_exact_cases,
        fp_tolerance_cases,
        distribution_preserving_cases,
        approximate_rejections,
        bit_exact_mismatch_rejections,
        tolerance_rejections,
        exactness_classes_declared: bit_exact_cases
            + fp_tolerance_cases
            + distribution_preserving_cases,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}

fn validate_accepted_cases() -> Result<Vec<CorrectnessOutcome>> {
    accepted_cases()
        .into_iter()
        .map(validate_correctness_case)
        .collect()
}

fn accepted_cases() -> [CorrectnessCase; 3] {
    [
        CorrectnessCase {
            name: "greedy_token_stream_hash",
            exactness: ExactnessClass::BitExact,
            expected_hash: 0x1111_2222_3333_4444,
            observed_hash: 0x1111_2222_3333_4444,
            max_abs_error_micros: 0,
            tolerance_micros: 0,
        },
        CorrectnessCase {
            name: "fp16_logits_tolerance",
            exactness: ExactnessClass::ReferenceEquivalentWithinDeclaredFpTolerance,
            expected_hash: 0x2222_3333_4444_5555,
            observed_hash: 0xaaaa_bbbb_cccc_dddd,
            max_abs_error_micros: 25,
            tolerance_micros: 50,
        },
        CorrectnessCase {
            name: "sampling_distribution_window",
            exactness: ExactnessClass::DistributionPreserving,
            expected_hash: 0x3333_4444_5555_6666,
            observed_hash: 0xbbbb_cccc_dddd_eeee,
            max_abs_error_micros: 40,
            tolerance_micros: 100,
        },
    ]
}

fn approximate_case() -> CorrectnessCase {
    CorrectnessCase {
        name: "approximate_core_runtime_claim",
        exactness: ExactnessClass::Approximate,
        expected_hash: 1,
        observed_hash: 1,
        max_abs_error_micros: 0,
        tolerance_micros: 0,
    }
}

fn bit_exact_mismatch_case() -> CorrectnessCase {
    CorrectnessCase {
        name: "bit_exact_mismatch",
        exactness: ExactnessClass::BitExact,
        expected_hash: 1,
        observed_hash: 2,
        max_abs_error_micros: 0,
        tolerance_micros: 0,
    }
}

fn tolerance_exceeded_case() -> CorrectnessCase {
    CorrectnessCase {
        name: "fp_tolerance_exceeded",
        exactness: ExactnessClass::ReferenceEquivalentWithinDeclaredFpTolerance,
        expected_hash: 1,
        observed_hash: 2,
        max_abs_error_micros: 101,
        tolerance_micros: 100,
    }
}

fn rejection_count(case: CorrectnessCase) -> u64 {
    u64::from(validate_correctness_case(case).is_err())
}

fn count_exactness(outcomes: &[CorrectnessOutcome], exactness: ExactnessClass) -> u64 {
    outcomes
        .iter()
        .filter(|outcome| outcome.exactness == exactness && outcome.accepted)
        .count() as u64
}
