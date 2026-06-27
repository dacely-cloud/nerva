#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HfCausalLmLoadOptions {
    pub compute_data_hash: bool,
}

impl HfCausalLmLoadOptions {
    pub const fn full_verification() -> Self {
        Self {
            compute_data_hash: true,
        }
    }

    pub const fn skip_payload_hash() -> Self {
        Self {
            compute_data_hash: false,
        }
    }
}

impl Default for HfCausalLmLoadOptions {
    fn default() -> Self {
        Self::full_verification()
    }
}
