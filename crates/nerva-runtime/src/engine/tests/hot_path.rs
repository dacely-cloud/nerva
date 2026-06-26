use nerva_memory::arena::kind::ArenaKind;

use crate::engine::hot_path::guard::{HotPathGuard, allocation_event_count};
use crate::engine::hot_path::status::HotPathGuardStatus;
use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn hot_path_guard_clean_scope_records_zero_allocation_attempts() {
    let mut guard = HotPathGuard::new(7);
    {
        let scope = guard.enter("clean").unwrap();
        assert_eq!(scope.label(), "clean");
    }

    assert_eq!(guard.entered_scopes(), 1);
    assert_eq!(guard.exited_scopes(), 1);
    assert_eq!(guard.active_scopes(), 0);
    assert_eq!(allocation_event_count(&guard), 0);
    assert_eq!(guard.ledger().hot_path_allocations, 0);
}

#[test]
fn hot_path_guard_rejects_nested_scopes() {
    let mut guard = HotPathGuard::new(0);
    let scope = guard.enter("outer").unwrap();
    let _ = scope.label();

    assert!(scope.reject_nested_scope().is_err());
}

#[test]
fn hot_path_guard_rejects_arena_reservations_and_preserves_usage() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut arenas = runtime.static_arenas(ResidencyBudget::new(1024, 2048, 4096));
    let before = arenas.device().used();
    let mut guard = HotPathGuard::new(0);

    {
        let mut scope = guard.enter("violation").unwrap();
        assert!(
            scope
                .reject_arena_reservation(
                    &mut arenas,
                    ArenaKind::Device,
                    "hot-path-device",
                    64,
                    64,
                )
                .is_err()
        );
    }

    assert_eq!(arenas.device().used(), before);
    assert_eq!(guard.forbidden_allocation_attempts(), 1);
    assert_eq!(guard.rejected_allocation_attempts(), 1);
    assert_eq!(guard.ledger().hot_path_allocations, 1);
    assert_eq!(allocation_event_count(&guard), 1);
    assert!(guard.usage_preserved_after_rejections());
}

#[test]
fn hot_path_guard_probe_reports_clean_scope_and_deliberate_rejections() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_hot_path_guard_probe(ResidencyBudget::new(1024, 2048, 4096))
        .unwrap();

    assert_eq!(summary.status, HotPathGuardStatus::Ok);
    assert!(summary.passed(), "{summary:?}");
    assert_eq!(summary.entered_scopes, 2);
    assert_eq!(summary.exited_scopes, 2);
    assert_eq!(summary.active_scopes_after_probe, 0);
    assert_eq!(summary.clean_scope_allocation_attempts, 0);
    assert_eq!(summary.deliberate_allocation_attempts, 3);
    assert_eq!(summary.deliberate_rejections, 3);
    assert_eq!(summary.ledger_allocation_events, 3);
    assert_eq!(summary.ledger_hot_path_allocations, 3);
    assert_eq!(summary.attempted_bytes, 192);
    assert_eq!(summary.release_to_system_calls, 0);
    assert!(summary.usage_preserved_after_rejections);

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"clean_scope_allocation_attempts\":0"));
    assert!(json.contains("\"deliberate_rejections\":3"));
    assert!(json.contains("\"release_to_system_calls\":0"));
}
