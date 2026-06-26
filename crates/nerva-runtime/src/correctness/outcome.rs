use crate::correctness::exactness::ExactnessClass;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CorrectnessOutcome {
    pub name: &'static str,
    pub exactness: ExactnessClass,
    pub accepted: bool,
}
