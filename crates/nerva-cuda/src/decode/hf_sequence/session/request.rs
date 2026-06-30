use core::ptr;

use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::NervaCudaHfDecodeSequenceResult;
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16, CudaHfDecodeSamplerConfig,
};
use crate::decode::hf_sequence::session::failures::{failed_create_summary, failed_run_summary};
use crate::decode::hf_sequence::session::ffi::{
    NervaCudaHfDecodeSequenceBatchAdvanceRequest, NervaCudaHfDecodeSequenceBatchAdvanceResult,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult,
    NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotRequest,
    NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotResult,
    NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotRequest,
    NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotResult,
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest,
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult,
    NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest,
    NervaCudaHfDecodeSequenceProjectionBatchExecuteResult,
    NervaCudaHfDecodeSequenceProjectionBatchPlanRequest,
    NervaCudaHfDecodeSequenceProjectionBatchPlanResult, NervaCudaHfDecodeSequenceSession,
    NervaCudaHfDecodeSequenceSessionCreateRequest, NervaCudaHfDecodeSequenceSessionCreateResult,
    NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest,
    NervaCudaHfDecodeSequenceSessionRunRequest, PROJECTION_BATCH_KIND_ATTENTION_OUTPUT,
    PROJECTION_BATCH_KIND_DOWN, PROJECTION_BATCH_KIND_GATE_UP, PROJECTION_BATCH_KIND_LM_HEAD,
    PROJECTION_BATCH_KIND_QKV, PROJECTION_BATCH_PLAN_INSUFFICIENT_COMPATIBLE_READY,
    PROJECTION_BATCH_PLAN_INSUFFICIENT_SCRATCH, PROJECTION_BATCH_PLAN_INVALID_LAYER,
    PROJECTION_BATCH_PLAN_INVALID_REQUEST, PROJECTION_BATCH_PLAN_NO_READY_SESSIONS,
    PROJECTION_BATCH_PLAN_NO_SESSIONS, PROJECTION_BATCH_PLAN_READY,
    PROJECTION_BATCH_PLAN_SHARED_WEIGHTS_UNPROVEN, PROJECTION_BATCH_PLAN_UNSUPPORTED_PROJECTION,
    batch_advance_one_hf_decode_sequence, create_hf_decode_sequence_session,
    destroy_hf_decode_sequence_session, execute_hf_decode_sequence_layer_projection_batch,
    execute_hf_decode_sequence_projection_batch, fork_shared_weights_hf_decode_sequence_session,
    plan_hf_decode_sequence_projection_batch, run_hf_decode_sequence_session,
    snapshot_deepseek_v3_mla_kv, snapshot_deepseek_v4_compressed_kv, snapshot_deepseek_v4_swa_kv,
};
use crate::decode::hf_sequence::session::helpers::{
    descriptor_ptr, planned_ptr, summary_from_run, validate_run,
};
use crate::decode::hf_sequence::session::summary::{
    CudaHfDecodeSequenceSessionCreateSummary, create_summary_from_result,
};
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceExperimentalRtConfig {
    pub enabled: bool,
    pub mode: u32,
    pub page_tokens: u32,
    pub pages: u32,
    pub local_window_tokens: u32,
    pub sink_tokens: u32,
}

pub const CUDA_HF_EXPERIMENTAL_RT_MODE_AUTO: u32 = 1;
pub const CUDA_HF_EXPERIMENTAL_RT_MODE_SHADOW: u32 = 2;
pub const CUDA_HF_EXPERIMENTAL_RT_MODE_SPARSE: u32 = 3;

impl Default for CudaHfDecodeSequenceExperimentalRtConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: CUDA_HF_EXPERIMENTAL_RT_MODE_AUTO,
            page_tokens: 64,
            pages: 512,
            local_window_tokens: 8192,
            sink_tokens: 128,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CudaHfDecodeSequenceSessionConfig<'a> {
    pub dtype: u32,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub max_context_tokens: usize,
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
    pub embeddings: &'a [u16],
    pub layers: &'a [CudaHfDecodeChainLayer<'a>],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
    pub weight_plan: Option<CudaHfDecodeSequenceWeightPlan>,
    pub weight_blocks: &'a [CudaHfDecodeSequenceWeightBlock],
    pub detailed_profile: bool,
    pub experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig,
}

pub struct CudaHfDecodeSequenceSession {
    handle: *mut NervaCudaHfDecodeSequenceSession,
    create_summary: CudaHfDecodeSequenceSessionCreateSummary,
}

pub struct CudaHfDecodeSequenceSessionCreateOutput {
    pub summary: CudaHfDecodeSequenceSessionCreateSummary,
    pub session: Option<CudaHfDecodeSequenceSession>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaHfDecodeSequenceProjectionBatchPlanSummary {
    pub status: SmokeStatus,
    pub reason: &'static str,
    pub exact: bool,
    pub requested_session_count: u32,
    pub eligible_session_count: u32,
    pub block_tokens: u32,
    pub target_block_tokens: u32,
    pub min_block_tokens: u32,
    pub dtype: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub layer_count: u32,
    pub max_context_tokens: u32,
    pub planned_weight_descriptor_hash: u64,
    pub resident_weight_bytes: u64,
    pub qkv_rows: u64,
    pub gate_up_rows: u64,
    pub qkv_input_bytes: u64,
    pub qkv_output_bytes: u64,
    pub attention_output_input_bytes: u64,
    pub attention_output_output_bytes: u64,
    pub gate_up_input_bytes: u64,
    pub gate_up_output_bytes: u64,
    pub down_input_bytes: u64,
    pub down_output_bytes: u64,
    pub lm_head_input_bytes: u64,
    pub lm_head_output_bytes: u64,
    pub pack_input_bytes: u64,
    pub max_projection_output_bytes: u64,
    pub hot_path_allocations: u64,
    pub cuda_error: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaHfDecodeSequenceProjectionBatchExecuteSummary {
    pub status: SmokeStatus,
    pub reason: &'static str,
    pub exact: bool,
    pub projection_kind: u32,
    pub layer_index: u32,
    pub requested_session_count: u32,
    pub eligible_session_count: u32,
    pub block_tokens: u32,
    pub target_block_tokens: u32,
    pub min_block_tokens: u32,
    pub dtype: u32,
    pub rows: u32,
    pub cols: u32,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub elapsed_ns: u64,
    pub pack_kernel_launches: u64,
    pub projection_kernel_launches: u64,
    pub scatter_kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub cuda_error: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaHfDecodeSequenceLayerProjectionBatchExecuteSummary {
    pub status: SmokeStatus,
    pub reason: &'static str,
    pub exact: bool,
    pub layer_index: u32,
    pub requested_session_count: u32,
    pub eligible_session_count: u32,
    pub block_tokens: u32,
    pub target_block_tokens: u32,
    pub min_block_tokens: u32,
    pub dtype: u32,
    pub qkv_rows: u32,
    pub attention_output_rows: u32,
    pub gate_up_rows: u32,
    pub down_rows: u32,
    pub hidden_cols: u32,
    pub attention_output_cols: u32,
    pub down_cols: u32,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub elapsed_ns: u64,
    pub qkv_elapsed_ns: u64,
    pub attention_output_elapsed_ns: u64,
    pub gate_up_elapsed_ns: u64,
    pub down_elapsed_ns: u64,
    pub pack_kernel_launches: u64,
    pub projection_kernel_launches: u64,
    pub scatter_kernel_launches: u64,
    pub dependency_kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub cuda_error: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaHfDecodeSequenceBatchAdvanceSummary {
    pub status: SmokeStatus,
    pub reason: &'static str,
    pub exact: bool,
    pub requested_session_count: u32,
    pub eligible_session_count: u32,
    pub block_tokens: u32,
    pub target_block_tokens: u32,
    pub min_block_tokens: u32,
    pub dtype: u32,
    pub layer_count: u32,
    pub observed_tokens: u32,
    pub last_token: u32,
    pub observed_token_hash: u64,
    pub tokens: Vec<u32>,
    pub d2h_bytes: u64,
    pub projection_elapsed_ns: u64,
    pub qkv_elapsed_ns: u64,
    pub attention_output_elapsed_ns: u64,
    pub gate_up_elapsed_ns: u64,
    pub down_elapsed_ns: u64,
    pub lm_head_elapsed_ns: u64,
    pub pack_kernel_launches: u64,
    pub projection_kernel_launches: u64,
    pub scatter_kernel_launches: u64,
    pub dependency_kernel_launches: u64,
    pub sampling_kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub cuda_error: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaHfDecodeSequenceDeepSeekV4SwaKvSnapshot {
    pub status: SmokeStatus,
    pub cuda_error: i32,
    pub layer_index: u32,
    pub block_count: u32,
    pub layer_offset_bytes: u64,
    pub layer_bytes: u64,
    pub page_bytes: u64,
    pub copied_bytes: u64,
    pub output_hash: u64,
    pub bytes: Vec<u8>,
    pub error: Option<String>,
}

pub type CudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshot =
    CudaHfDecodeSequenceDeepSeekV4SwaKvSnapshot;

pub type CudaHfDecodeSequenceDeepSeekV3MlaKvSnapshot = CudaHfDecodeSequenceDeepSeekV4SwaKvSnapshot;

impl<'a> CudaHfDecodeSequenceSessionConfig<'a> {
    pub fn create(&self) -> CudaHfDecodeSequenceSessionCreateOutput {
        if let Some(error) = validate_config(self) {
            return CudaHfDecodeSequenceSessionCreateOutput {
                summary: failed_create_summary(self, error),
                session: None,
            };
        }
        let plan = self.weight_plan.unwrap_or_default();
        let use_descriptors = plan.is_declared();
        let ffi_layers = self
            .layers
            .iter()
            .map(|layer| {
                if use_descriptors {
                    layer.to_descriptor_layout_ffi()
                } else {
                    layer.to_ffi()
                }
            })
            .collect::<Vec<_>>();
        let mut handle = ptr::null_mut();
        let request = self.to_ffi(ffi_layers.as_ptr());
        let mut out = NervaCudaHfDecodeSequenceSessionCreateResult::default();
        let admission_memory = crate::smoke::probe::smoke();
        let return_code = create_hf_decode_sequence_session(&request, &mut out, &mut handle);
        let summary = create_summary_from_result(
            return_code,
            &out,
            admission_memory.device_total_memory_bytes,
            admission_memory.device_free_memory_bytes,
        );
        let session = (summary.status == SmokeStatus::Ok && !handle.is_null()).then(|| {
            CudaHfDecodeSequenceSession {
                handle,
                create_summary: summary.clone(),
            }
        });
        CudaHfDecodeSequenceSessionCreateOutput { summary, session }
    }

    fn to_ffi(
        &self,
        layers: *const crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer,
    ) -> NervaCudaHfDecodeSequenceSessionCreateRequest {
        let plan = self.weight_plan.unwrap_or_default();
        NervaCudaHfDecodeSequenceSessionCreateRequest {
            dtype: self.dtype,
            hidden: self.hidden as u32,
            heads: self.heads as u32,
            kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            intermediate: self.intermediate as u32,
            vocab_size: self.vocab_size as u32,
            layer_count: self.layers.len() as u32,
            max_context_tokens: self.max_context_tokens as u32,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta.unwrap_or(0.0),
            embeddings: planned_ptr(self.embeddings, plan),
            layers,
            final_norm_weight: planned_ptr(self.final_norm_weight, plan),
            lm_head: planned_ptr(self.lm_head, plan),
            planned_weight_blocks: plan.blocks,
            planned_gpu_resident_blocks: plan.gpu_resident_blocks,
            planned_gpu_staged_blocks: plan.gpu_staged_blocks,
            planned_weight_bytes: plan.weight_bytes,
            planned_gpu_resident_weight_bytes: plan.gpu_resident_weight_bytes,
            planned_gpu_staged_weight_bytes: plan.gpu_staged_weight_bytes,
            planned_weight_descriptors: descriptor_ptr(self.weight_blocks),
            planned_weight_descriptor_count: self.weight_blocks.len() as u32,
            planned_weight_descriptor_hash: plan.descriptor_hash,
            detailed_profile: self.detailed_profile as u32,
            experimental_rt_decode: self.experimental_rt.enabled as u32,
            experimental_rt_mode: self.experimental_rt.mode,
            experimental_rt_page_tokens: self.experimental_rt.page_tokens,
            experimental_rt_pages: self.experimental_rt.pages,
            experimental_rt_local_window_tokens: self.experimental_rt.local_window_tokens,
            experimental_rt_sink_tokens: self.experimental_rt.sink_tokens,
        }
    }
}

impl CudaHfDecodeSequenceSession {
    pub fn create_summary(&self) -> &CudaHfDecodeSequenceSessionCreateSummary {
        &self.create_summary
    }

    pub fn fork_shared_weights(
        &mut self,
        detailed_profile: bool,
    ) -> CudaHfDecodeSequenceSessionCreateOutput {
        let request = NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest {
            parent: self.handle,
            detailed_profile: detailed_profile as u32,
        };
        let mut handle = ptr::null_mut();
        let mut out = NervaCudaHfDecodeSequenceSessionCreateResult::default();
        let admission_memory = crate::smoke::probe::smoke();
        let return_code =
            fork_shared_weights_hf_decode_sequence_session(&request, &mut out, &mut handle);
        let summary = create_summary_from_result(
            return_code,
            &out,
            admission_memory.device_total_memory_bytes,
            admission_memory.device_free_memory_bytes,
        );
        let session = (summary.status == SmokeStatus::Ok && !handle.is_null()).then(|| {
            CudaHfDecodeSequenceSession {
                handle,
                create_summary: summary.clone(),
            }
        });
        CudaHfDecodeSequenceSessionCreateOutput { summary, session }
    }

    pub(super) fn raw_handle(&mut self) -> *mut NervaCudaHfDecodeSequenceSession {
        self.handle
    }

    pub fn run(
        &mut self,
        prompt_tokens: &[u32],
        steps: usize,
        eos_token: Option<u32>,
    ) -> CudaHfDecodeSequenceSummary {
        self.run_with_sampler(
            prompt_tokens,
            steps,
            eos_token,
            CudaHfDecodeSamplerConfig::greedy(),
        )
    }

    pub fn run_with_sampler(
        &mut self,
        prompt_tokens: &[u32],
        steps: usize,
        eos_token: Option<u32>,
        sampler: CudaHfDecodeSamplerConfig,
    ) -> CudaHfDecodeSequenceSummary {
        if let Some(error) = validate_run(prompt_tokens, steps, self.create_summary.vocab_size) {
            return failed_run_summary(&self.create_summary, steps, 0, error);
        }
        if let Some(error) = sampler.validate() {
            return failed_run_summary(&self.create_summary, steps, 0, error);
        }
        let mut tokens = vec![0u32; steps];
        let seed = *prompt_tokens.last().unwrap();
        let request = NervaCudaHfDecodeSequenceSessionRunRequest {
            session: self.handle,
            steps: steps as u32,
            seed_token: seed,
            prompt_tokens: prompt_tokens.as_ptr(),
            prompt_token_count: prompt_tokens.len() as u32,
            has_eos_token: eos_token.is_some() as u32,
            eos_token: eos_token.unwrap_or(0),
            output_tokens: tokens.as_mut_ptr(),
            output_token_capacity: steps as u32,
            sampler: sampler.to_ffi(),
        };
        let mut out = NervaCudaHfDecodeSequenceResult::default();
        let return_code = run_hf_decode_sequence_session(&request, &mut out);
        tokens.truncate(out.observed_tokens.min(steps as u32) as usize);
        summary_from_run(return_code, &out, tokens, &self.create_summary)
    }

    pub fn projection_batch_plan(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchPlanSummary {
        let mut handles = sessions
            .iter_mut()
            .map(|session| session.raw_handle())
            .collect::<Vec<_>>();
        let request = NervaCudaHfDecodeSequenceProjectionBatchPlanRequest {
            sessions: handles.as_mut_ptr(),
            session_count: handles.len() as u32,
            target_block_tokens,
            min_block_tokens,
        };
        let mut out = NervaCudaHfDecodeSequenceProjectionBatchPlanResult::default();
        let return_code = plan_hf_decode_sequence_projection_batch(&request, &mut out);
        projection_batch_plan_summary(return_code, &out)
    }

    pub fn execute_qkv_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
        layer_index: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        Self::execute_projection_batch(
            sessions,
            target_block_tokens,
            min_block_tokens,
            PROJECTION_BATCH_KIND_QKV,
            layer_index,
        )
    }

    pub fn execute_attention_output_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
        layer_index: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        Self::execute_projection_batch(
            sessions,
            target_block_tokens,
            min_block_tokens,
            PROJECTION_BATCH_KIND_ATTENTION_OUTPUT,
            layer_index,
        )
    }

    pub fn execute_gate_up_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
        layer_index: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        Self::execute_projection_batch(
            sessions,
            target_block_tokens,
            min_block_tokens,
            PROJECTION_BATCH_KIND_GATE_UP,
            layer_index,
        )
    }

    pub fn execute_down_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
        layer_index: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        Self::execute_projection_batch(
            sessions,
            target_block_tokens,
            min_block_tokens,
            PROJECTION_BATCH_KIND_DOWN,
            layer_index,
        )
    }

    pub fn execute_lm_head_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        Self::execute_projection_batch(
            sessions,
            target_block_tokens,
            min_block_tokens,
            PROJECTION_BATCH_KIND_LM_HEAD,
            0,
        )
    }

    pub fn execute_layer_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
        layer_index: u32,
    ) -> CudaHfDecodeSequenceLayerProjectionBatchExecuteSummary {
        let mut handles = sessions
            .iter_mut()
            .map(|session| session.raw_handle())
            .collect::<Vec<_>>();
        let request = NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest {
            sessions: handles.as_mut_ptr(),
            session_count: handles.len() as u32,
            target_block_tokens,
            min_block_tokens,
            layer_index,
        };
        let mut out = NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult::default();
        let return_code = execute_hf_decode_sequence_layer_projection_batch(&request, &mut out);
        layer_projection_batch_execute_summary(return_code, &out)
    }

    pub fn batch_advance_one(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
    ) -> CudaHfDecodeSequenceBatchAdvanceSummary {
        let mut handles = sessions
            .iter_mut()
            .map(|session| session.raw_handle())
            .collect::<Vec<_>>();
        let mut tokens = vec![u32::MAX; handles.len()];
        let request = NervaCudaHfDecodeSequenceBatchAdvanceRequest {
            sessions: handles.as_mut_ptr(),
            session_count: handles.len() as u32,
            target_block_tokens,
            min_block_tokens,
            output_tokens: tokens.as_mut_ptr(),
            output_token_capacity: tokens.len() as u32,
        };
        let mut out = NervaCudaHfDecodeSequenceBatchAdvanceResult::default();
        let return_code = batch_advance_one_hf_decode_sequence(&request, &mut out);
        batch_advance_summary(return_code, &out, tokens)
    }

    pub fn deepseek_v3_mla_kv_snapshot(
        &mut self,
        layer_index: u32,
        byte_capacity: usize,
    ) -> CudaHfDecodeSequenceDeepSeekV3MlaKvSnapshot {
        if byte_capacity == 0 {
            return CudaHfDecodeSequenceDeepSeekV3MlaKvSnapshot {
                status: SmokeStatus::Failed,
                cuda_error: 0,
                layer_index,
                block_count: 0,
                layer_offset_bytes: 0,
                layer_bytes: 0,
                page_bytes: 0,
                copied_bytes: 0,
                output_hash: 0,
                bytes: Vec::new(),
                error: Some("DeepSeek V3 MLA KV snapshot requires a byte capacity".to_string()),
            };
        }
        let mut bytes = vec![0u8; byte_capacity];
        let request = NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest {
            session: self.handle,
            layer_index,
            output_bytes: bytes.as_mut_ptr(),
            output_byte_capacity: byte_capacity as u64,
        };
        let mut out = NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult::default();
        let return_code = snapshot_deepseek_v3_mla_kv(&request, &mut out);
        let copied = out.copied_bytes.min(byte_capacity as u64) as usize;
        bytes.truncate(copied);
        let status = if return_code == 0 && out.status == 0 {
            SmokeStatus::Ok
        } else {
            SmokeStatus::Failed
        };
        let error = (status != SmokeStatus::Ok).then(|| {
            format!(
                "DeepSeek V3 MLA KV snapshot failed: return_code={return_code} status={} cuda_error={}",
                out.status, out.cuda_error
            )
        });
        CudaHfDecodeSequenceDeepSeekV3MlaKvSnapshot {
            status,
            cuda_error: out.cuda_error,
            layer_index: out.layer_index,
            block_count: out.block_count,
            layer_offset_bytes: out.layer_offset_bytes,
            layer_bytes: out.layer_bytes,
            page_bytes: out.page_bytes,
            copied_bytes: out.copied_bytes,
            output_hash: out.output_hash,
            bytes,
            error,
        }
    }

    pub fn deepseek_v4_swa_kv_snapshot(
        &mut self,
        layer_index: u32,
        byte_capacity: usize,
    ) -> CudaHfDecodeSequenceDeepSeekV4SwaKvSnapshot {
        if byte_capacity == 0 {
            return CudaHfDecodeSequenceDeepSeekV4SwaKvSnapshot {
                status: SmokeStatus::Failed,
                cuda_error: 0,
                layer_index,
                block_count: 0,
                layer_offset_bytes: 0,
                layer_bytes: 0,
                page_bytes: 0,
                copied_bytes: 0,
                output_hash: 0,
                bytes: Vec::new(),
                error: Some("DeepSeek V4 SWA KV snapshot requires a byte capacity".to_string()),
            };
        }
        let mut bytes = vec![0u8; byte_capacity];
        let request = NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotRequest {
            session: self.handle,
            layer_index,
            output_bytes: bytes.as_mut_ptr(),
            output_byte_capacity: byte_capacity as u64,
        };
        let mut out = NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotResult::default();
        let return_code = snapshot_deepseek_v4_swa_kv(&request, &mut out);
        let copied = out.copied_bytes.min(byte_capacity as u64) as usize;
        bytes.truncate(copied);
        let status = if return_code == 0 && out.status == 0 {
            SmokeStatus::Ok
        } else {
            SmokeStatus::Failed
        };
        let error = (status != SmokeStatus::Ok).then(|| {
            format!(
                "DeepSeek V4 SWA KV snapshot failed: return_code={return_code} status={} cuda_error={}",
                out.status, out.cuda_error
            )
        });
        CudaHfDecodeSequenceDeepSeekV4SwaKvSnapshot {
            status,
            cuda_error: out.cuda_error,
            layer_index: out.layer_index,
            block_count: out.block_count,
            layer_offset_bytes: out.layer_offset_bytes,
            layer_bytes: out.layer_bytes,
            page_bytes: out.page_bytes,
            copied_bytes: out.copied_bytes,
            output_hash: out.output_hash,
            bytes,
            error,
        }
    }

    pub fn deepseek_v4_compressed_kv_snapshot(
        &mut self,
        layer_index: u32,
        byte_capacity: usize,
    ) -> CudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshot {
        if byte_capacity == 0 {
            return CudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshot {
                status: SmokeStatus::Failed,
                cuda_error: 0,
                layer_index,
                block_count: 0,
                layer_offset_bytes: 0,
                layer_bytes: 0,
                page_bytes: 0,
                copied_bytes: 0,
                output_hash: 0,
                bytes: Vec::new(),
                error: Some(
                    "DeepSeek V4 compressed KV snapshot requires a byte capacity".to_string(),
                ),
            };
        }
        let mut bytes = vec![0u8; byte_capacity];
        let request = NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotRequest {
            session: self.handle,
            layer_index,
            output_bytes: bytes.as_mut_ptr(),
            output_byte_capacity: byte_capacity as u64,
        };
        let mut out = NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotResult::default();
        let return_code = snapshot_deepseek_v4_compressed_kv(&request, &mut out);
        let copied = out.copied_bytes.min(byte_capacity as u64) as usize;
        bytes.truncate(copied);
        let status = if return_code == 0 && out.status == 0 {
            SmokeStatus::Ok
        } else {
            SmokeStatus::Failed
        };
        let error = (status != SmokeStatus::Ok).then(|| {
            format!(
                "DeepSeek V4 compressed KV snapshot failed: return_code={return_code} status={} cuda_error={}",
                out.status, out.cuda_error
            )
        });
        CudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshot {
            status,
            cuda_error: out.cuda_error,
            layer_index: out.layer_index,
            block_count: out.block_count,
            layer_offset_bytes: out.layer_offset_bytes,
            layer_bytes: out.layer_bytes,
            page_bytes: out.page_bytes,
            copied_bytes: out.copied_bytes,
            output_hash: out.output_hash,
            bytes,
            error,
        }
    }

    fn execute_projection_batch(
        sessions: &mut [&mut CudaHfDecodeSequenceSession],
        target_block_tokens: u32,
        min_block_tokens: u32,
        projection_kind: u32,
        layer_index: u32,
    ) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        let mut handles = sessions
            .iter_mut()
            .map(|session| session.raw_handle())
            .collect::<Vec<_>>();
        let request = NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest {
            sessions: handles.as_mut_ptr(),
            session_count: handles.len() as u32,
            target_block_tokens,
            min_block_tokens,
            projection_kind,
            layer_index,
        };
        let mut out = NervaCudaHfDecodeSequenceProjectionBatchExecuteResult::default();
        let return_code = execute_hf_decode_sequence_projection_batch(&request, &mut out);
        projection_batch_execute_summary(return_code, &out)
    }
}

impl Drop for CudaHfDecodeSequenceSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut out = NervaCudaHfDecodeSequenceSessionCreateResult::default();
            let _ = destroy_hf_decode_sequence_session(self.handle, &mut out);
            self.handle = ptr::null_mut();
        }
    }
}

fn validate_config(request: &CudaHfDecodeSequenceSessionConfig<'_>) -> Option<String> {
    if request.hidden == 0 || request.heads == 0 || request.kv_heads == 0 || request.head_dim == 0 {
        return Some("CUDA HF decode sequence session dimensions must be non-zero".to_string());
    }
    if request.layers.is_empty() || request.max_context_tokens == 0 {
        return Some("CUDA HF decode sequence session requires layers and capacity".to_string());
    }
    if request.dtype > CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16 {
        return Some("CUDA HF decode sequence session dtype is unsupported".to_string());
    }
    request.weight_plan.and_then(|plan| {
        plan.validate()
            .or_else(|| plan.validate_descriptors(request.weight_blocks))
    })
}

fn projection_batch_plan_summary(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceProjectionBatchPlanResult,
) -> CudaHfDecodeSequenceProjectionBatchPlanSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else {
        SmokeStatus::Failed
    };
    CudaHfDecodeSequenceProjectionBatchPlanSummary {
        status,
        reason: projection_batch_reason_name(out.reason),
        exact: out.exact != 0,
        requested_session_count: out.requested_session_count,
        eligible_session_count: out.eligible_session_count,
        block_tokens: out.block_tokens,
        target_block_tokens: out.target_block_tokens,
        min_block_tokens: out.min_block_tokens,
        dtype: out.dtype,
        hidden: out.hidden,
        heads: out.heads,
        kv_heads: out.kv_heads,
        head_dim: out.head_dim,
        intermediate: out.intermediate,
        vocab_size: out.vocab_size,
        layer_count: out.layer_count,
        max_context_tokens: out.max_context_tokens,
        planned_weight_descriptor_hash: out.planned_weight_descriptor_hash,
        resident_weight_bytes: out.resident_weight_bytes,
        qkv_rows: out.qkv_rows,
        gate_up_rows: out.gate_up_rows,
        qkv_input_bytes: out.qkv_input_bytes,
        qkv_output_bytes: out.qkv_output_bytes,
        attention_output_input_bytes: out.attention_output_input_bytes,
        attention_output_output_bytes: out.attention_output_output_bytes,
        gate_up_input_bytes: out.gate_up_input_bytes,
        gate_up_output_bytes: out.gate_up_output_bytes,
        down_input_bytes: out.down_input_bytes,
        down_output_bytes: out.down_output_bytes,
        lm_head_input_bytes: out.lm_head_input_bytes,
        lm_head_output_bytes: out.lm_head_output_bytes,
        pack_input_bytes: out.pack_input_bytes,
        max_projection_output_bytes: out.max_projection_output_bytes,
        hot_path_allocations: out.hot_path_allocations,
        cuda_error: out.cuda_error,
    }
}

fn projection_batch_execute_summary(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceProjectionBatchExecuteResult,
) -> CudaHfDecodeSequenceProjectionBatchExecuteSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else {
        SmokeStatus::Failed
    };
    CudaHfDecodeSequenceProjectionBatchExecuteSummary {
        status,
        reason: projection_batch_reason_name(out.reason),
        exact: out.exact != 0,
        projection_kind: out.projection_kind,
        layer_index: out.layer_index,
        requested_session_count: out.requested_session_count,
        eligible_session_count: out.eligible_session_count,
        block_tokens: out.block_tokens,
        target_block_tokens: out.target_block_tokens,
        min_block_tokens: out.min_block_tokens,
        dtype: out.dtype,
        rows: out.rows,
        cols: out.cols,
        input_bytes: out.input_bytes,
        output_bytes: out.output_bytes,
        elapsed_ns: out.elapsed_ns,
        pack_kernel_launches: out.pack_kernel_launches,
        projection_kernel_launches: out.projection_kernel_launches,
        scatter_kernel_launches: out.scatter_kernel_launches,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        cuda_error: out.cuda_error,
    }
}

fn layer_projection_batch_execute_summary(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult,
) -> CudaHfDecodeSequenceLayerProjectionBatchExecuteSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else {
        SmokeStatus::Failed
    };
    CudaHfDecodeSequenceLayerProjectionBatchExecuteSummary {
        status,
        reason: projection_batch_reason_name(out.reason),
        exact: out.exact != 0,
        layer_index: out.layer_index,
        requested_session_count: out.requested_session_count,
        eligible_session_count: out.eligible_session_count,
        block_tokens: out.block_tokens,
        target_block_tokens: out.target_block_tokens,
        min_block_tokens: out.min_block_tokens,
        dtype: out.dtype,
        qkv_rows: out.qkv_rows,
        attention_output_rows: out.attention_output_rows,
        gate_up_rows: out.gate_up_rows,
        down_rows: out.down_rows,
        hidden_cols: out.hidden_cols,
        attention_output_cols: out.attention_output_cols,
        down_cols: out.down_cols,
        input_bytes: out.input_bytes,
        output_bytes: out.output_bytes,
        elapsed_ns: out.elapsed_ns,
        qkv_elapsed_ns: out.qkv_elapsed_ns,
        attention_output_elapsed_ns: out.attention_output_elapsed_ns,
        gate_up_elapsed_ns: out.gate_up_elapsed_ns,
        down_elapsed_ns: out.down_elapsed_ns,
        pack_kernel_launches: out.pack_kernel_launches,
        projection_kernel_launches: out.projection_kernel_launches,
        scatter_kernel_launches: out.scatter_kernel_launches,
        dependency_kernel_launches: out.dependency_kernel_launches,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        cuda_error: out.cuda_error,
    }
}

fn batch_advance_summary(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceBatchAdvanceResult,
    tokens: Vec<u32>,
) -> CudaHfDecodeSequenceBatchAdvanceSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else {
        SmokeStatus::Failed
    };
    CudaHfDecodeSequenceBatchAdvanceSummary {
        status,
        reason: projection_batch_reason_name(out.reason),
        exact: out.exact != 0,
        requested_session_count: out.requested_session_count,
        eligible_session_count: out.eligible_session_count,
        block_tokens: out.block_tokens,
        target_block_tokens: out.target_block_tokens,
        min_block_tokens: out.min_block_tokens,
        dtype: out.dtype,
        layer_count: out.layer_count,
        observed_tokens: out.observed_tokens,
        last_token: out.last_token,
        observed_token_hash: out.observed_token_hash,
        tokens,
        d2h_bytes: out.d2h_bytes,
        projection_elapsed_ns: out.projection_elapsed_ns,
        qkv_elapsed_ns: out.qkv_elapsed_ns,
        attention_output_elapsed_ns: out.attention_output_elapsed_ns,
        gate_up_elapsed_ns: out.gate_up_elapsed_ns,
        down_elapsed_ns: out.down_elapsed_ns,
        lm_head_elapsed_ns: out.lm_head_elapsed_ns,
        pack_kernel_launches: out.pack_kernel_launches,
        projection_kernel_launches: out.projection_kernel_launches,
        scatter_kernel_launches: out.scatter_kernel_launches,
        dependency_kernel_launches: out.dependency_kernel_launches,
        sampling_kernel_launches: out.sampling_kernel_launches,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        cuda_error: out.cuda_error,
    }
}

fn projection_batch_reason_name(reason: u32) -> &'static str {
    match reason {
        PROJECTION_BATCH_PLAN_READY => "ready",
        PROJECTION_BATCH_PLAN_INVALID_REQUEST => "invalid_request",
        PROJECTION_BATCH_PLAN_NO_SESSIONS => "no_sessions",
        PROJECTION_BATCH_PLAN_NO_READY_SESSIONS => "no_ready_sessions",
        PROJECTION_BATCH_PLAN_SHARED_WEIGHTS_UNPROVEN => "shared_weights_unproven",
        PROJECTION_BATCH_PLAN_INSUFFICIENT_COMPATIBLE_READY => "insufficient_compatible_ready",
        PROJECTION_BATCH_PLAN_UNSUPPORTED_PROJECTION => "unsupported_projection",
        PROJECTION_BATCH_PLAN_INVALID_LAYER => "invalid_layer",
        PROJECTION_BATCH_PLAN_INSUFFICIENT_SCRATCH => "insufficient_scratch",
        _ => "unknown",
    }
}
