use crate::acceptance::cuda;
use crate::acceptance::report::AcceptanceReport;
use nerva_core::types::dtype::DType;

pub(crate) fn push_precision_and_cuda_blocks(report: &mut AcceptanceReport) {
    match nerva_model::precision::smoke::run::precision_block_smoke() {
        Ok(summary) => report.push(
            "fp16_bf16_precision_block",
            summary.passed(),
            format!(
                "f16_hash={} bf16_hash={} f16_expected_hash={} bf16_expected_hash={} f16_max_abs_error={} bf16_max_abs_error={} f16_hot_path_allocations={} bf16_hot_path_allocations={}",
                summary.f16.output_hash,
                summary.bf16.output_hash,
                summary.f16.expected_hash,
                summary.bf16.expected_hash,
                summary.f16.max_abs_error,
                summary.bf16.max_abs_error,
                summary.f16.hot_path_allocations,
                summary.bf16.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("fp16_bf16_precision_block", false, format!("{err:?}")),
    }

    match nerva_model::precision::file_smoke::run::precision_block_from_safetensors_smoke() {
        Ok(summary) => {
            report.push(
                "safetensors_precision_block",
                summary.passed(),
                format!(
                    "tensors_loaded={} bytes_loaded={} data_hash={} output_hash={} expected_hash={} bit_parity={} hot_path_allocations={}",
                    summary.tensors_loaded,
                    summary.bytes_loaded,
                    summary.data_hash,
                    summary.output_hash,
                    summary.expected_hash,
                    summary.bit_parity,
                    summary.hot_path_allocations,
                ),
            );
            cuda::block::push_precision_checks(report, &summary);
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("safetensors_precision_block", false, details.clone());
            cuda::block::push_prerequisite_failure(report, &details);
        }
    }

    match (
        nerva_model::tiny::precision::smoke::tiny_precision_greedy_decode_smoke(DType::F16, 8),
        nerva_model::tiny::precision::smoke::tiny_precision_greedy_decode_smoke(DType::BF16, 8),
    ) {
        (Ok(f16), Ok(bf16)) => report.push(
            "precision_tiny_model_greedy_parity",
            f16.passed() && bf16.passed() && f16.output_hash == bf16.output_hash,
            format!(
                "f16_parity={} bf16_parity={} f16_hash={} bf16_hash={} f16_ledgers={} bf16_ledgers={} f16_cpu_events={} bf16_cpu_events={} f16_decisions={} bf16_decisions={} f16_hot_path_allocations={} bf16_hot_path_allocations={}",
                f16.parity,
                bf16.parity,
                f16.output_hash,
                bf16.output_hash,
                f16.ledger_count,
                bf16.ledger_count,
                f16.cpu_events,
                bf16.cpu_events,
                f16.execution_decisions,
                bf16.execution_decisions,
                f16.hot_path_allocations,
                bf16.hot_path_allocations,
            ),
        ),
        (Err(err), _) | (_, Err(err)) => report.push(
            "precision_tiny_model_greedy_parity",
            false,
            format!("{err:?}"),
        ),
    }

    match nerva_model::causal_lm::smoke::hf_causal_lm_safetensors_smoke(4) {
        Ok(summary) => report.push(
            "hf_causal_lm_safetensors_greedy_decode",
            summary.passed(),
            format!(
                "layers={} hidden={} vocab={} manifest_entries={} shard_plan_entries={} tensors_loaded={} bytes_loaded={} final_norm_loaded={} tied_lm_head={} parity={} ledger_count={} cpu_events={} execution_decisions={} output_hash={} data_hash={} hot_path_allocations={}",
                summary.layers,
                summary.hidden,
                summary.vocab_size,
                summary.manifest_entries,
                summary.shard_plan_entries,
                summary.tensors_loaded,
                summary.bytes_loaded,
                summary.final_norm_loaded,
                summary.tied_lm_head,
                summary.parity,
                summary.ledger_count,
                summary.cpu_events,
                summary.execution_decisions,
                summary.output_hash,
                summary.data_hash,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "hf_causal_lm_safetensors_greedy_decode",
            false,
            format!("{err:?}"),
        ),
    }

    cuda::hf_layer::push_loaded_hf_layer_forward(report);
}
