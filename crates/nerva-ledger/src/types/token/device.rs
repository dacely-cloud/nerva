use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;

use crate::types::token::ledger::TokenLedger;
use crate::types::token::timeline;

impl TokenLedger {
    pub fn device_active_ns(&self, device: DeviceOrdinal) -> Result<u64> {
        let (active_ns, _) = timeline::device_timeline_totals(&self.device_timeline, device)?;
        Ok(active_ns)
    }

    pub fn device_idle_ns(&self, device: DeviceOrdinal) -> Result<u64> {
        let (_, idle_ns) = timeline::device_timeline_totals(&self.device_timeline, device)?;
        Ok(idle_ns)
    }
}
