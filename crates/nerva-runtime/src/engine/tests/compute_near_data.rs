use nerva_core::types::error::NervaError;

use crate::engine::compute_near_data::config::ComputeNearDataProbeConfig;
use crate::engine::compute_near_data::summary::ComputeNearDataProbeStatus;
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn compute_near_data_probe_executes_exact_resident_split_matvec() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_compute_near_data_probe(ComputeNearDataProbeConfig::default())
        .unwrap();

    assert_eq!(summary.status, ComputeNearDataProbeStatus::Ok);
    assert_eq!(summary.rows, 4);
    assert_eq!(summary.cols, 3);
    assert_eq!(summary.split_row, 2);
    assert_eq!(summary.blocks, 2);
    assert_eq!(summary.dram_blocks, 1);
    assert_eq!(summary.vram_blocks, 1);
    assert_eq!(summary.output, vec![3.0, -5.0, 0.5, 6.5]);
    assert_eq!(summary.reference, summary.output);
    assert!(summary.parity);
    assert_eq!(summary.max_abs_error, 0.0);
    assert_eq!(summary.execution_decisions, 2);
    assert_eq!(summary.block_version_dependencies, 2);
    assert_eq!(summary.cpu_events, 1);
    assert_eq!(summary.device_events, 1);
    assert_eq!(summary.copy_events, 1);
    assert_eq!(summary.merge_bytes, 8);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"parity\":true"));
    assert!(
        summary
            .to_json()
            .contains("\"block_version_dependencies\":2")
    );
}

#[test]
fn compute_near_data_probe_rejects_unsupported_fixture_shape() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let err = runtime
        .run_compute_near_data_probe(ComputeNearDataProbeConfig {
            rows: 8,
            ..ComputeNearDataProbeConfig::default()
        })
        .unwrap_err();

    assert!(matches!(err, NervaError::InvalidArgument { .. }));
}
