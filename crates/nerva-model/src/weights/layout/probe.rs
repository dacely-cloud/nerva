use nerva_core::types::error::Result;

use crate::hf::probe::hf_metadata_probe;
use crate::weights::hash::hash_weight_layout;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::layout::summary::{HfWeightLayoutProbeStatus, HfWeightLayoutProbeSummary};

pub fn hf_weight_layout_probe() -> Result<HfWeightLayoutProbeSummary> {
    let metadata = hf_metadata_probe()?.metadata;
    let plan = plan_hf_weight_layout(&metadata)?;
    Ok(HfWeightLayoutProbeSummary {
        layout_hash: hash_weight_layout(&plan),
        status: HfWeightLayoutProbeStatus::Ok,
        plan,
    })
}
