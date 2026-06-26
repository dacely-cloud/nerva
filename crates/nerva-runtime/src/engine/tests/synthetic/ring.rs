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
