use crate::decode::hf_sequence::ffi::NervaCudaHfDecodeSequenceResult;
use crate::decode::hf_sequence::session::failures::failed_run_summary;
use crate::decode::hf_sequence::session::ffi::{
    NervaCudaHfDecodeSequenceSessionAdvanceRequest, NervaCudaHfDecodeSequenceSessionStartRequest,
    advance_hf_decode_sequence_session, start_hf_decode_sequence_session,
};
use crate::decode::hf_sequence::session::helpers::{summary_from_run, validate_run};
use crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSession;
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::smoke::status::SmokeStatus;

pub struct CudaHfDecodeSequenceLoopStart<'a> {
    pub summary: CudaHfDecodeSequenceSummary,
    pub loop_state: Option<CudaHfDecodeSequenceLoop<'a>>,
}

pub struct CudaHfDecodeSequenceLoop<'a> {
    session: &'a mut CudaHfDecodeSequenceSession,
    eos_token: Option<u32>,
    finished: bool,
}

impl<'a> CudaHfDecodeSequenceLoop<'a> {
    pub fn start(
        session: &'a mut CudaHfDecodeSequenceSession,
        prompt_tokens: &[u32],
        eos_token: Option<u32>,
    ) -> CudaHfDecodeSequenceLoopStart<'a> {
        let create_summary = session.create_summary().clone();
        if let Some(error) = validate_loop_start(prompt_tokens, create_summary.vocab_size) {
            return CudaHfDecodeSequenceLoopStart {
                summary: failed_run_summary(&create_summary, 0, 0, error),
                loop_state: None,
            };
        }
        let request = NervaCudaHfDecodeSequenceSessionStartRequest {
            session: session.raw_handle(),
            prompt_tokens: prompt_tokens.as_ptr(),
            prompt_token_count: prompt_tokens.len() as u32,
            has_eos_token: eos_token.is_some() as u32,
            eos_token: eos_token.unwrap_or(0),
        };
        let mut out = NervaCudaHfDecodeSequenceResult::default();
        let return_code = start_hf_decode_sequence_session(&request, &mut out);
        let summary = summary_from_run(return_code, &out, Vec::new());
        let loop_state = (summary.status == SmokeStatus::Ok).then_some(Self {
            session,
            eos_token,
            finished: false,
        });
        CudaHfDecodeSequenceLoopStart {
            summary,
            loop_state,
        }
    }

    pub fn advance(&mut self, steps: usize) -> CudaHfDecodeSequenceSummary {
        let create_summary = self.session.create_summary().clone();
        if self.finished {
            return failed_run_summary(
                &create_summary,
                steps,
                0,
                "CUDA HF decode sequence loop is already finished".to_string(),
            );
        }
        if let Some(error) = validate_run(&[0], steps, create_summary.vocab_size) {
            return failed_run_summary(&create_summary, steps, 0, error);
        }
        let mut tokens = vec![0u32; steps];
        let request = NervaCudaHfDecodeSequenceSessionAdvanceRequest {
            session: self.session.raw_handle(),
            steps: steps as u32,
            output_tokens: tokens.as_mut_ptr(),
            output_token_capacity: steps as u32,
        };
        let mut out = NervaCudaHfDecodeSequenceResult::default();
        let return_code = advance_hf_decode_sequence_session(&request, &mut out);
        tokens.truncate(out.observed_tokens.min(steps as u32) as usize);
        let summary = summary_from_run(return_code, &out, tokens);
        self.finished = summary.status == SmokeStatus::Ok
            && (summary.tokens.len() < steps
                || self
                    .eos_token
                    .is_some_and(|eos| summary.tokens.last().is_some_and(|token| *token == eos)));
        summary
    }
}

fn validate_loop_start(prompt_tokens: &[u32], vocab_size: u32) -> Option<String> {
    if prompt_tokens.is_empty() {
        return Some("CUDA HF decode sequence loop start requires prompt".to_string());
    }
    if prompt_tokens.iter().any(|token| *token >= vocab_size) {
        return Some("CUDA HF decode sequence loop prompt token is outside vocabulary".to_string());
    }
    None
}
