use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::plan::plan_stage_pipeline;
use crate::transport::stage::route::{
    planned_stage_routes, probe_stage_route_validation, validate_stage_route,
};
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
    assert_eq!(summary.stage_route_validations, 3);
    assert!(summary.stage_route_identity_checks >= 6);
    assert!(summary.wrong_source_stage_rejections > 0);
    assert!(summary.wrong_destination_stage_rejections > 0);
    assert!(summary.non_adjacent_route_rejections > 0);
    assert!(summary.endpoint_identity_rejections > 0);
    assert!(summary.activation_size_rejections > 0);
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
fn stage_route_validation_rejects_wrong_stage_identity() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let capabilities = runtime.discover_capabilities();
    let plan = plan_stage_pipeline(
        StagePipelineConfig::reference_decode(),
        runtime.config().device,
        &capabilities,
    )
    .unwrap();
    let routes = planned_stage_routes(&plan);
    assert_eq!(routes.len(), 3);
    assert!(validate_stage_route(&plan, routes[0]).is_ok());

    let report = probe_stage_route_validation(&plan).unwrap();
    assert_eq!(report.route_validations, 3);
    assert!(report.route_identity_checks >= 6);
    assert_eq!(report.wrong_source_stage_rejections, 1);
    assert_eq!(report.wrong_destination_stage_rejections, 1);
    assert_eq!(report.non_adjacent_route_rejections, 1);
    assert_eq!(report.endpoint_identity_rejections, 1);
    assert_eq!(report.activation_size_rejections, 1);
    assert_eq!(report.route_rejections(), 5);

    let mut wrong_route = routes[0];
    wrong_route.source.stage_id = wrong_route.source.stage_id.saturating_add(1);
    assert!(validate_stage_route(&plan, wrong_route).is_err());
}

#[test]
fn stage_pipeline_rejects_invalid_stage_counts() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut config = StagePipelineConfig::reference_decode();
    config.stages = 1;

    assert!(runtime.run_stage_pipeline_probe(config).is_err());
}
