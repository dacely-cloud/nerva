use nerva_core::types::id::{RequestId, SequenceId, TokenId};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use nerva_runtime::engine::synthetic::SyntheticDecodeConfig;

pub(crate) fn run_synthetic(steps: u64, ring_capacity: usize) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(steps, ring_capacity, TokenId(1)))
        .map_err(|err| format!("synthetic decode failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_synthetic_ledger_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut engine = runtime
        .synthetic_engine(4)
        .map_err(|err| format!("synthetic engine init failed: {err:?}"))?;
    let output = engine
        .launch_device_next(RequestId(1), SequenceId(1), 0, TokenId(1))
        .map_err(|err| format!("synthetic ledger launch failed: {err:?}"))?
        .collect()
        .map_err(|err| format!("synthetic ledger collect failed: {err:?}"))?;
    Ok(output.ledger.to_json())
}
