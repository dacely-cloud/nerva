use nerva_core::types::id::ResidentBlockId;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::weights::probe::ResidentWeightProbeStatus;

#[test]
fn resident_weight_probe_reports_manifest_materialization() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_resident_weight_probe().unwrap();

    assert_eq!(summary.status, ResidentWeightProbeStatus::Ok);
    assert_eq!(summary.blocks, 290);
    assert_eq!(summary.total_weight_bytes, 11_866_210_304);
    assert_eq!(summary.dram_used_bytes, summary.total_weight_bytes);
    assert_eq!(summary.vram_used_bytes, 0);
    assert_eq!(summary.residency_decisions, 290);
    assert_eq!(summary.first_block_id, Some(ResidentBlockId(1)));
    assert_eq!(summary.last_block_id, Some(ResidentBlockId(290)));
    assert_eq!(
        summary.first_tensor.as_deref(),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(summary.last_tensor.as_deref(), Some("lm_head.weight"));
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"blocks\":290"));
}
