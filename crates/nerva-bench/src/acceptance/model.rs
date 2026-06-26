use crate::acceptance::cuda;
use crate::acceptance::files;
use crate::acceptance::manifest;
use crate::acceptance::report::AcceptanceReport;
use crate::acceptance::vllm;
use nerva_core::types::dtype::DType;

pub(crate) fn push_reference_block(report: &mut AcceptanceReport) {
    match nerva_model::reference::smoke::reference_block_smoke() {
        Ok(summary) => report.push(
            "reference_block",
            summary.hot_path_allocations == 0,
            format!(
                "hidden={} heads={} output_hash={} hot_path_allocations={}",
                summary.hidden, summary.heads, summary.output_hash, summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("reference_block", false, format!("{err:?}")),
    }
}

pub(crate) fn push_precision_and_cuda_blocks(report: &mut AcceptanceReport) {
    match nerva_model::precision::smoke::precision_block_smoke() {
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
}

pub(crate) fn push_tiny_model_and_cuda_decode(report: &mut AcceptanceReport) {
    match nerva_model::tiny::smoke::tiny_greedy_decode_smoke(8) {
        Ok(summary) => {
            report.push(
                "tiny_model_greedy_parity",
                summary.parity
                    && summary.ledger_count == summary.steps as u64
                    && summary.hot_path_allocations == 0,
                format!(
                    "steps={} parity={} ledger_count={} device_events={} hot_path_allocations={} output_hash={}",
                    summary.steps,
                    summary.parity,
                    summary.ledger_count,
                    summary.device_events,
                    summary.hot_path_allocations,
                    summary.output_hash,
                ),
            );
            cuda::decode::push_tiny_decode_check(report, &summary);
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("tiny_model_greedy_parity", false, details.clone());
            cuda::decode::push_prerequisite_failure(report, &details);
        }
    }
}

pub(crate) fn push_manifest_and_file_checks(report: &mut AcceptanceReport) {
    match vllm::vllm_token_identity_acceptance() {
        Ok((passed, details)) => report.push("vllm_token_identity_parity", passed, details),
        Err(err) => report.push("vllm_token_identity_parity", false, err),
    }

    match manifest::model_manifest_acceptance() {
        Ok((passed, details)) => report.push("hf_model_manifest", passed, details),
        Err(err) => report.push("hf_model_manifest", false, err),
    }

    match files::safetensors_file_header_acceptance() {
        Ok((passed, details)) => report.push("safetensors_file_header", passed, details),
        Err(err) => report.push("safetensors_file_header", false, err),
    }

    match files::safetensors_file_prefetch_acceptance() {
        Ok((passed, details)) => report.push("safetensors_file_prefetch", passed, details),
        Err(err) => report.push("safetensors_file_prefetch", false, err),
    }
}

pub(crate) fn push_tiered_attention_and_cuda(report: &mut AcceptanceReport) {
    match nerva_model::attention::smoke::blockwise_attention_smoke() {
        Ok(summary) => {
            report.push(
                "tiered_blockwise_attention",
                summary.cpu_block_events > 0
                    && summary.device_block_events > 0
                    && summary.hot_path_allocations == 0,
                format!(
                    "blocks={} tokens={} cpu_block_events={} device_block_events={} hot_path_allocations={} output_hash={}",
                    summary.blocks,
                    summary.tokens,
                    summary.cpu_block_events,
                    summary.device_block_events,
                    summary.hot_path_allocations,
                    summary.output_hash,
                ),
            );
            cuda::attention::push_tiered_check(report, &summary);
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("tiered_blockwise_attention", false, details.clone());
            cuda::attention::push_prerequisite_failure(report, &details);
        }
    }
}

pub(crate) fn push_warm_compute(report: &mut AcceptanceReport) {
    match nerva_model::warm_compute::probe::run::warm_compute_probe() {
        Ok(summary) => report.push(
            "warm_compute_selection",
            summary.parity
                && summary.cpu_beats_staged
                && summary.execution_decisions > 0
                && summary.hot_path_allocations == 0,
            format!(
                "selected_strategy={} parity={} cpu_beats_staged={} execution_decisions={} hot_path_allocations={} output_hash={}",
                summary.selected_strategy.label(),
                summary.parity,
                summary.cpu_beats_staged,
                summary.execution_decisions,
                summary.hot_path_allocations,
                summary.output_hash,
            ),
        ),
        Err(err) => report.push("warm_compute_selection", false, format!("{err:?}")),
    }
}

pub(crate) fn push_kernel_contracts(report: &mut AcceptanceReport) {
    match nerva_kernel_contracts::registry::probe::kernel_registry_probe() {
        Ok(summary) => report.push(
            "kernel_contract_fallbacks",
            summary.direct_plans > 0
                && summary.fallback_plans > 0
                && summary.rejected_plans > 0
                && summary.exact_fallbacks > 0,
            format!(
                "implementations={} fallbacks={} direct_plans={} fallback_plans={} rejected_plans={} exact_fallbacks={}",
                summary.implementations,
                summary.fallbacks,
                summary.direct_plans,
                summary.fallback_plans,
                summary.rejected_plans,
                summary.exact_fallbacks,
            ),
        ),
        Err(err) => report.push("kernel_contract_fallbacks", false, format!("{err:?}")),
    }
}
