use nerva_core::types::error::NervaError;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::graph::layout::GraphKey;
use crate::token::ring::{DeviceTokenRef, TokenInputSource};

#[test]
fn synthetic_launch_collect_records_token_and_ledger() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let step = engine
        .launch(RequestId(1), SequenceId(1), 0, TokenId(41))
        .unwrap();
    let output = step.collect().unwrap();

    assert_eq!(output.token, TokenId(42));
    assert_eq!(output.input_source, TokenInputSource::Seed);
    assert_eq!(output.device_token_ref.token_index, 0);
    assert_eq!(output.ledger.hot_path_allocations, 0);
    assert_eq!(output.ledger.events.len(), 4);
    assert_eq!(output.ledger.event_count(LedgerEventKind::GraphReplay), 1);
    assert_eq!(
        output.ledger.event_count(LedgerEventKind::DeviceActivity),
        1
    );
    assert_eq!(output.ledger.event_count(LedgerEventKind::Copy), 1);
    assert_eq!(output.ledger.event_count(LedgerEventKind::Sync), 1);
    assert_eq!(
        output.ledger.sync_count_for(SyncClass::SoftVisibilitySync),
        1
    );
    assert_eq!(output.ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 3);
    assert_eq!(output.ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 0);
    assert!(output.ledger.require_classified_syncs().is_ok());
    assert_eq!(
        engine
            .token_ring()
            .consume_device_input(RequestId(1), SequenceId(1), 0)
            .unwrap(),
        TokenId(42)
    );
    assert_eq!(
        engine
            .graph_pool()
            .replay_count(GraphKey {
                bucket: 1,
                max_blocks: 1
            })
            .unwrap(),
        1
    );
}

#[test]
fn synthetic_next_step_must_use_device_token() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let output = engine
        .launch(RequestId(2), SequenceId(9), 0, TokenId(10))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(11));

    let err = engine
        .launch(RequestId(2), SequenceId(9), 1, TokenId(99))
        .unwrap_err();
    assert!(matches!(err, NervaError::ResidencyViolation { .. }));

    let output = engine
        .launch_device_next(RequestId(2), SequenceId(9), 1, TokenId(10))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(12));
    assert!(matches!(
        output.input_source,
        TokenInputSource::DeviceRing(DeviceTokenRef { token_index: 0, .. })
    ));
}
