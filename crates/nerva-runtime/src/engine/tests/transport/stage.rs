use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::summary::StagePipelineStatus;

#[test]
fn stage_pipeline_probe_moves_activations_without_moving_weights() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_stage_pipeline_probe(StagePipelineConfig::reference_decode())
        .unwrap();

    assert_eq!(summary.status, StagePipelineStatus::Ok);
    assert_eq!(summary.stages, 4);
    assert_eq!(summary.boundaries, 3);
    assert_eq!(summary.activation_bytes_per_boundary, 32 * 1024);
    assert_eq!(summary.total_activation_tx_bytes, 96 * 1024);
    assert_eq!(summary.activation_only_boundaries, 3);
    assert_eq!(summary.inter_stage_weight_bytes, 0);
    assert_eq!(summary.all_reduce_bytes, 0);
    assert_eq!(summary.transport_events, 3);
    assert_eq!(summary.phase_handoff_syncs, 3);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.stage_local_weight_bytes > 0);
    assert!(summary.stage_local_kv_bytes > 0);
    assert!(summary.passed());
}

#[test]
fn stage_pipeline_rejects_invalid_stage_counts() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut config = StagePipelineConfig::reference_decode();
    config.stages = 1;

    assert!(runtime.run_stage_pipeline_probe(config).is_err());
}
