use crate::acceptance::files;
use crate::acceptance::manifest;
use crate::acceptance::report::AcceptanceReport;
use crate::acceptance::vllm;

pub(crate) fn push_manifest_and_file_checks(report: &mut AcceptanceReport) {
    match vllm::vllm_token_identity_acceptance() {
        Ok((passed, details)) => report.push("vllm_token_identity_parity", passed, details),
        Err(err) => report.push("vllm_token_identity_parity", false, err),
    }

    match vllm::qwen3_vllm_nerva_token_acceptance() {
        Ok((passed, details)) => report.push("qwen3_vllm_nerva_token_parity", passed, details),
        Err(err) => report.push("qwen3_vllm_nerva_token_parity", false, err),
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
