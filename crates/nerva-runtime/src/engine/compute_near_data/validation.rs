use nerva_core::types::error::{NervaError, Result};

use crate::engine::compute_near_data::config::ComputeNearDataProbeConfig;

pub(crate) fn validate_config(config: ComputeNearDataProbeConfig) -> Result<()> {
    if config.rows != 4 || config.cols != 3 || config.split_row != 2 {
        return Err(NervaError::InvalidArgument {
            reason: "compute-near-data probe currently requires rows=4 cols=3 split_row=2"
                .to_string(),
        });
    }
    Ok(())
}
