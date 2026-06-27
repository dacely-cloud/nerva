use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::decode::hf_sequence::weight_plan::CudaHfDecodeSequenceWeightPlan;

use crate::engine::hf_cuda_decode::summary::HfCudaResidentWeightSummary;

pub(super) fn cuda_weight_plan(
    summary: &HfCudaResidentWeightSummary,
) -> Result<CudaHfDecodeSequenceWeightPlan> {
    Ok(CudaHfDecodeSequenceWeightPlan {
        blocks: u32_from_u64("weight plan blocks", summary.plan_steps)?,
        gpu_resident_blocks: u32_from_u64(
            "resident weight plan GPU-resident blocks",
            summary.plan_gpu_resident_steps,
        )?,
        gpu_staged_blocks: u32_from_u64(
            "resident weight plan GPU-staged blocks",
            summary.plan_gpu_staged_steps,
        )?,
        weight_bytes: summary.plan_weight_bytes,
        gpu_resident_weight_bytes: summary.plan_gpu_resident_weight_bytes,
        gpu_staged_weight_bytes: summary.plan_gpu_staged_weight_bytes,
    })
}

pub(super) fn attach_cuda_weight_contract(
    summary: &mut HfCudaResidentWeightSummary,
    sequence: &CudaHfDecodeSequenceSummary,
) -> Result<()> {
    if sequence.planned_weight_bytes != summary.plan_weight_bytes {
        return Err(contract_error(
            "planned weight byte count",
            summary.plan_weight_bytes,
        ));
    }
    if sequence.resident_weight_bytes != summary.plan_weight_bytes {
        return Err(contract_error(
            "CUDA resident weight byte count",
            summary.plan_weight_bytes,
        ));
    }
    summary.cuda_contract_blocks = sequence.planned_weight_blocks as u64;
    summary.cuda_contract_weight_bytes = sequence.planned_weight_bytes;
    summary.cuda_contract_matched = sequence.planned_weight_blocks as u64 == summary.plan_steps
        && sequence.planned_gpu_resident_weight_bytes == summary.plan_gpu_resident_weight_bytes
        && sequence.planned_gpu_staged_weight_bytes == summary.plan_gpu_staged_weight_bytes;
    Ok(())
}

fn u32_from_u64(label: &'static str, value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| NervaError::InvalidArgument {
        reason: format!("{label} does not fit u32"),
    })
}

fn contract_error(label: &'static str, expected: u64) -> NervaError {
    NervaError::InvalidArgument {
        reason: format!("CUDA HF decode {label} does not match resident plan {expected}"),
    }
}
