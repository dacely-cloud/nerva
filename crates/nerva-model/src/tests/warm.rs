use crate::warm_compute::probe::run::warm_compute_probe;
use crate::warm_compute::strategy::WarmComputeStrategy;
use crate::warm_compute::summary::WarmComputeProbeStatus;

#[test]
fn warm_compute_probe_compares_all_exact_strategies() {
    let summary = warm_compute_probe().unwrap();

    assert_eq!(summary.status, WarmComputeProbeStatus::Ok);
    assert_eq!(summary.rows, 4);
    assert_eq!(summary.cols, 4);
    assert_eq!(summary.candidates.len(), 4);
    assert_eq!(summary.selected_strategy, WarmComputeStrategy::GpuResident);
    assert!(summary.parity);
    assert!(summary.cpu_beats_staged);
    assert_eq!(summary.execution_decisions, 1);
    assert_eq!(summary.runtime_timestamp_decisions, 1);
    assert_eq!(summary.measured_candidate_costs, 4);
    assert_eq!(summary.estimated_candidate_costs, 4);
    assert_eq!(summary.cpu_events, 2);
    assert_eq!(summary.device_events, 3);
    assert_eq!(summary.copy_events, 3);
    assert_eq!(summary.copy_bytes, 104);
    assert_eq!(summary.total_latency_ns, 138);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .candidates
            .iter()
            .all(|candidate| candidate.output_hash == summary.output_hash
                && candidate.measured_ns > 0)
    );
    assert!(
        summary
            .to_json()
            .contains("\"selected_strategy\":\"gpu-resident\"")
    );
    assert!(summary.to_json().contains("\"measured_candidate_costs\":4"));
}
