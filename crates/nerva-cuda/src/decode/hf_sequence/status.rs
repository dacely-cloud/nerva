use crate::decode::ffi::CUDA_ERROR_NO_DEVICE;
use crate::decode::hf_sequence::ffi::NervaCudaHfDecodeSequenceResult;
use crate::smoke::status::SmokeStatus;

pub(super) fn sequence_status_from_result(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceResult,
) -> SmokeStatus {
    if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    }
}

pub(super) fn sequence_failure_reason(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceResult,
) -> String {
    format!(
        "CUDA HF decode sequence failed: return_code={return_code} status={} cuda_error={}",
        out.status, out.cuda_error,
    )
}
