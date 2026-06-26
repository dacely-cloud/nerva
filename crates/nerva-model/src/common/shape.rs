use nerva_core::types::error::{NervaError, Result};

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
