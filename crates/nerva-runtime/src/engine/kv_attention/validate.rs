use nerva_core::types::error::{NervaError, Result};

use crate::engine::kv_attention::config::TieredKvAttentionProbeConfig;

pub(super) fn validate_config(config: TieredKvAttentionProbeConfig) -> Result<()> {
    if config.tokens_per_page != 2 {
        return Err(NervaError::InvalidArgument {
            reason: "tiered KV attention probe currently requires two tokens per page".to_string(),
        });
    }
    let required_page_bytes = config.tokens_per_page as usize * 2 * core::mem::size_of::<f32>() * 2;
    if config.page_bytes < required_page_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "tiered KV attention probe page bytes cannot hold keys and values".to_string(),
        });
    }
    Ok(())
}
