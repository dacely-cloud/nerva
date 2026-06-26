use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn critical_path_probe_reports_host_wait_and_gpu_idle_separately() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let report = runtime.run_critical_path_probe().unwrap();

    assert_eq!(report.token_index, 0);
    assert_eq!(report.graph_replay_ns, 1);
    assert_eq!(report.device_activity_event_ns, 3);
    assert_eq!(report.copy_ns, 1);
    assert_eq!(report.host_event_wait_ns, 1);
    assert_eq!(report.device_timeline_active_ns, 3);
    assert_eq!(report.gpu_idle_ns, 0);
    assert_eq!(report.host_wait_events, 1);
    assert_eq!(report.device_timeline_spans, 1);
    assert!(report.host_wait_gpu_idle_sources_separate);
    assert!(report.proves_host_wait_not_gpu_idle());

    let json = report.to_json();
    assert!(json.contains("\"host_event_wait_ns\":1"));
    assert!(json.contains("\"gpu_idle_ns\":0"));
    assert!(json.contains("\"estimated_presented_as_measured\":false"));
}
