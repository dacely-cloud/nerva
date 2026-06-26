use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_backend_contract_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    Ok(runtime.run_backend_contract_probe().to_json())
}
