use nerva_core::types::error::{NervaError, Result};

pub(crate) fn require_len(label: &'static str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("{label} length {got} does not match expected {expected}"),
        })
    }
}
