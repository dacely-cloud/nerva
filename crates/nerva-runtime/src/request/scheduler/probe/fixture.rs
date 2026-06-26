use nerva_core::types::error::Result;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::scheduler::admission::RequestAdmission;
use crate::request::scheduler::bounded::BoundedRequestScheduler;

pub(super) struct RequestSchedulerProbeFixture {
    pub scheduler: BoundedRequestScheduler,
    pub slot_b: usize,
    pub reuse_admission: RequestAdmission,
    pub full_rejections: u64,
    pub duplicate_rejections: u64,
}

impl RequestSchedulerProbeFixture {
    pub fn new() -> Result<Self> {
        let mut scheduler = BoundedRequestScheduler::new(2)?;
        let prompt_a = vec![TokenId(0)];
        let prompt_b = vec![TokenId(1)];
        let slot_b = scheduler.admit(admission(2, 2, prompt_b, 2, Some(TokenId(2))))?;
        scheduler.admit(admission(1, 1, prompt_a.clone(), 3, Some(TokenId(3))))?;
        let reuse_admission = admission(3, 3, vec![TokenId(2)], 1, Some(TokenId(3)));
        let full_rejections = scheduler.admit(reuse_admission.clone()).is_err() as u64;
        let duplicate_rejections =
            scheduler.admit(admission(1, 9, prompt_a, 1, None)).is_err() as u64;

        Ok(Self {
            scheduler,
            slot_b,
            reuse_admission,
            full_rejections,
            duplicate_rejections,
        })
    }
}

fn admission(
    request: u64,
    sequence: u64,
    prompt_tokens: Vec<TokenId>,
    max_new_tokens: usize,
    eos_token: Option<TokenId>,
) -> RequestAdmission {
    RequestAdmission {
        request_id: RequestId(request),
        sequence_id: SequenceId(sequence),
        prompt_tokens,
        max_new_tokens,
        eos_token,
    }
}
