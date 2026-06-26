use crate::correctness::exactness::ExactnessClass;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CorrectnessCase {
    pub name: &'static str,
    pub exactness: ExactnessClass,
    pub expected_hash: u64,
    pub observed_hash: u64,
    pub max_abs_error_micros: u64,
    pub tolerance_micros: u64,
}
