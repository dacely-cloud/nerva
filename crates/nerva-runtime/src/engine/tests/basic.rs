use nerva_core::types::id::device::DeviceOrdinal;

use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn runtime_uses_device_zero_by_default() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    assert_eq!(runtime.config().device, DeviceOrdinal(0));
    assert_eq!(runtime.empty_token_ledger(9).token_index, 9);
}
