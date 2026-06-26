use nerva_core::types::error::{NervaError, Result};

pub(crate) fn div_ceil_usize(value: usize, divisor: usize) -> Result<u32> {
    let count = value
        .checked_add(divisor - 1)
        .and_then(|sum| sum.checked_div(divisor))
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP chunk count overflowed".to_string(),
        })?;
    u32::try_from(count).map_err(|_| NervaError::InvalidArgument {
        reason: "DPDK UDP chunk count exceeds u32".to_string(),
    })
}

pub(crate) fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}
