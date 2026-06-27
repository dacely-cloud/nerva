use nerva_core::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransformerBlockShape {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
}

impl TransformerBlockShape {
    pub const fn new(hidden: usize, heads: usize, intermediate: usize) -> Self {
        let head_dim = if heads == 0 || hidden % heads != 0 {
            0
        } else {
            hidden / heads
        };
        Self {
            hidden,
            heads,
            kv_heads: heads,
            head_dim,
            intermediate,
        }
    }

    pub const fn new_with_kv_heads(
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        intermediate: usize,
    ) -> Self {
        let head_dim = if heads == 0 || hidden % heads != 0 {
            0
        } else {
            hidden / heads
        };
        Self::new_with_kv_heads_and_head_dim(hidden, heads, kv_heads, head_dim, intermediate)
    }

    pub const fn new_with_kv_heads_and_head_dim(
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
    ) -> Self {
        Self {
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.hidden == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.intermediate == 0
        {
            return Err(NervaError::InvalidArgument {
                reason: "transformer block dimensions must be non-zero".to_string(),
            });
        }
        if self.kv_heads > self.heads {
            return Err(NervaError::InvalidArgument {
                reason: "KV head count cannot exceed attention head count".to_string(),
            });
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            return Err(NervaError::InvalidArgument {
                reason: "attention head count must be divisible by KV head count".to_string(),
            });
        }
        Ok(())
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn attention_hidden(self) -> usize {
        self.heads * self.head_dim
    }

    pub const fn kv_groups(self) -> usize {
        self.heads / self.kv_heads
    }

    pub const fn kv_hidden(self) -> usize {
        self.kv_heads * self.head_dim()
    }

    pub const fn kv_head_for_attention_head(self, head: usize) -> usize {
        head / self.kv_groups()
    }
}
