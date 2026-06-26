use nerva_core::types::id::TokenId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct TokenComparison {
    pub(crate) matched_tokens: usize,
    pub(crate) mismatched_tokens: usize,
    pub(crate) missing_tokens: usize,
    pub(crate) extra_tokens: usize,
    pub(crate) first_mismatch_index: Option<usize>,
}

pub(crate) fn compare_token_slices(vllm: &[TokenId], nerva: &[TokenId]) -> TokenComparison {
    let shared = vllm.len().min(nerva.len());
    let mut matched_tokens = 0usize;
    let mut mismatched_tokens = 0usize;
    let mut first_mismatch_index = None;

    for index in 0..shared {
        if vllm[index] == nerva[index] {
            matched_tokens += 1;
        } else {
            mismatched_tokens += 1;
            first_mismatch_index.get_or_insert(index);
        }
    }
    let missing_tokens = nerva.len().saturating_sub(vllm.len());
    let extra_tokens = vllm.len().saturating_sub(nerva.len());
    if first_mismatch_index.is_none() && (missing_tokens > 0 || extra_tokens > 0) {
        first_mismatch_index = Some(shared);
    }

    TokenComparison {
        matched_tokens,
        mismatched_tokens,
        missing_tokens,
        extra_tokens,
        first_mismatch_index,
    }
}
