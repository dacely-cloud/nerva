use nerva_core::types::error::{NervaError, Result};

pub const DEFAULT_PROJECTION_BLOCK_TOKENS: usize = 8;
pub const MAX_PROJECTION_BLOCK_TOKENS: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfCudaProjectionMode {
    Token,
    BlockVerify { block_tokens: usize },
}

impl HfCudaProjectionMode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::BlockVerify { .. } => "block-verify",
        }
    }

    pub const fn block_tokens(self) -> usize {
        match self {
            Self::Token => 1,
            Self::BlockVerify { block_tokens } => block_tokens,
        }
    }
}

impl Default for HfCudaProjectionMode {
    fn default() -> Self {
        Self::Token
    }
}

pub fn block_verify_mode(block_tokens: usize) -> Result<HfCudaProjectionMode> {
    if block_tokens < 2 {
        return Err(NervaError::InvalidArgument {
            reason: "block verification requires at least 2 projection block tokens".to_string(),
        });
    }
    if block_tokens > MAX_PROJECTION_BLOCK_TOKENS {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "block verification supports at most {MAX_PROJECTION_BLOCK_TOKENS} projection block tokens"
            ),
        });
    }
    Ok(HfCudaProjectionMode::BlockVerify { block_tokens })
}
