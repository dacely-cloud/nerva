use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::token::ring::DeviceTokenRing;

#[test]
fn device_token_ring_rejects_stale_reads() {
    let mut ring = DeviceTokenRing::new(2).unwrap();
    ring.publish(RequestId(1), SequenceId(1), 0, TokenId(7))
        .unwrap();
    assert!(
        ring.consume_device_input(RequestId(1), SequenceId(2), 0)
            .is_err()
    );
    assert_eq!(
        ring.consume_device_input(RequestId(1), SequenceId(1), 0)
            .unwrap(),
        TokenId(7)
    );
}

#[test]
fn device_token_ring_blocks_slot_reuse_until_host_observation() {
    let mut ring = DeviceTokenRing::new(1).unwrap();
    ring.publish(RequestId(1), SequenceId(1), 0, TokenId(7))
        .unwrap();

    assert!(
        ring.publish(RequestId(1), SequenceId(1), 1, TokenId(8))
            .is_err()
    );

    assert_eq!(
        ring.host_observe(RequestId(1), SequenceId(1), 0).unwrap(),
        TokenId(7)
    );
    let token_ref = ring
        .publish(RequestId(1), SequenceId(1), 1, TokenId(8))
        .unwrap();

    assert_eq!(token_ref.slot_index, 0);
    assert_eq!(token_ref.token_index, 1);
    assert_eq!(token_ref.version, 2);
    assert!(
        ring.consume_device_input(RequestId(1), SequenceId(1), 0)
            .is_err()
    );
    assert_eq!(
        ring.consume_device_input(RequestId(1), SequenceId(1), 1)
            .unwrap(),
        TokenId(8)
    );
}
