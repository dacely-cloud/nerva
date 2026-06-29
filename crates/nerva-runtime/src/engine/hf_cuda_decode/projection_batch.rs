use nerva_core::types::dtype::DType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionBatchConfig {
    pub target_block_tokens: usize,
    pub min_block_tokens: usize,
}

impl ProjectionBatchConfig {
    pub fn new(target_block_tokens: usize, min_block_tokens: usize) -> Self {
        Self {
            target_block_tokens: target_block_tokens.max(1),
            min_block_tokens: min_block_tokens.max(1),
        }
    }
}

impl Default for ProjectionBatchConfig {
    fn default() -> Self {
        Self {
            target_block_tokens: 32,
            min_block_tokens: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionBatchModelKey {
    pub data_hash: u64,
    pub data_hash_available: bool,
    pub dtype: DType,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub layer_count: usize,
}

impl ProjectionBatchModelKey {
    pub fn proves_same_weights_as(&self, other: &Self) -> bool {
        self.data_hash_available
            && other.data_hash_available
            && self.data_hash == other.data_hash
            && self.dtype == other.dtype
            && self.hidden_size == other.hidden_size
            && self.attention_heads == other.attention_heads
            && self.kv_heads == other.kv_heads
            && self.head_dim == other.head_dim
            && self.intermediate_size == other.intermediate_size
            && self.vocab_size == other.vocab_size
            && self.layer_count == other.layer_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionBatchCandidate {
    pub request_id: u64,
    pub model: ProjectionBatchModelKey,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub remaining_tokens: usize,
    pub max_context_tokens: usize,
    pub ready: bool,
    pub stopped: bool,
}

impl ProjectionBatchCandidate {
    pub fn can_decode_one_token(&self) -> bool {
        self.ready
            && !self.stopped
            && self.remaining_tokens > 0
            && self.prompt_tokens.saturating_add(self.generated_tokens) < self.max_context_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionBatchPlanReason {
    Ready,
    NoCandidates,
    NoReadyCandidates,
    SharedWeightsUnproven,
    InsufficientCompatibleReady,
}

impl ProjectionBatchPlanReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NoCandidates => "no_candidates",
            Self::NoReadyCandidates => "no_ready_candidates",
            Self::SharedWeightsUnproven => "shared_weights_unproven",
            Self::InsufficientCompatibleReady => "insufficient_compatible_ready",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionBatchPlan {
    pub reason: ProjectionBatchPlanReason,
    pub exact: bool,
    pub block_tokens: usize,
    pub selected_request_ids: Vec<u64>,
    pub model: Option<ProjectionBatchModelKey>,
}

impl ProjectionBatchPlan {
    fn empty(reason: ProjectionBatchPlanReason) -> Self {
        Self {
            reason,
            exact: false,
            block_tokens: 0,
            selected_request_ids: Vec::new(),
            model: None,
        }
    }
}

pub fn plan_exact_projection_batch(
    candidates: &[ProjectionBatchCandidate],
    config: ProjectionBatchConfig,
) -> ProjectionBatchPlan {
    if candidates.is_empty() {
        return ProjectionBatchPlan::empty(ProjectionBatchPlanReason::NoCandidates);
    }

    let ready = candidates
        .iter()
        .filter(|candidate| candidate.can_decode_one_token())
        .collect::<Vec<_>>();
    if ready.is_empty() {
        return ProjectionBatchPlan::empty(ProjectionBatchPlanReason::NoReadyCandidates);
    }
    if ready
        .iter()
        .all(|candidate| !candidate.model.data_hash_available)
    {
        return ProjectionBatchPlan::empty(ProjectionBatchPlanReason::SharedWeightsUnproven);
    }

    let mut best_start = 0usize;
    let mut best_len = 0usize;
    for (index, candidate) in ready.iter().enumerate() {
        if !candidate.model.data_hash_available {
            continue;
        }
        let len = ready
            .iter()
            .filter(|other| candidate.model.proves_same_weights_as(&other.model))
            .count();
        if len > best_len {
            best_start = index;
            best_len = len;
        }
    }

    if best_len < config.min_block_tokens {
        return ProjectionBatchPlan::empty(ProjectionBatchPlanReason::InsufficientCompatibleReady);
    }

    let model = ready[best_start].model;
    let selected_request_ids = ready
        .iter()
        .filter(|candidate| model.proves_same_weights_as(&candidate.model))
        .take(config.target_block_tokens)
        .map(|candidate| candidate.request_id)
        .collect::<Vec<_>>();

    ProjectionBatchPlan {
        reason: ProjectionBatchPlanReason::Ready,
        exact: true,
        block_tokens: selected_request_ids.len(),
        selected_request_ids,
        model: Some(model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(data_hash: u64) -> ProjectionBatchModelKey {
        ProjectionBatchModelKey {
            data_hash,
            data_hash_available: true,
            dtype: DType::BF16,
            hidden_size: 4096,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            vocab_size: 151936,
            layer_count: 36,
        }
    }

    fn candidate(request_id: u64, model: ProjectionBatchModelKey) -> ProjectionBatchCandidate {
        ProjectionBatchCandidate {
            request_id,
            model,
            prompt_tokens: 128,
            generated_tokens: 16,
            remaining_tokens: 64,
            max_context_tokens: 4096,
            ready: true,
            stopped: false,
        }
    }

    #[test]
    fn exact_projection_batch_groups_same_resident_weights() {
        let candidates = [
            candidate(10, model(7)),
            candidate(11, model(7)),
            candidate(12, model(7)),
        ];
        let plan = plan_exact_projection_batch(&candidates, ProjectionBatchConfig::default());

        assert_eq!(plan.reason, ProjectionBatchPlanReason::Ready);
        assert!(plan.exact);
        assert_eq!(plan.block_tokens, 3);
        assert_eq!(plan.selected_request_ids, [10, 11, 12]);
    }

    #[test]
    fn exact_projection_batch_caps_to_target_block_tokens() {
        let candidates = (0..12)
            .map(|index| candidate(index, model(9)))
            .collect::<Vec<_>>();
        let plan = plan_exact_projection_batch(&candidates, ProjectionBatchConfig::new(4, 2));

        assert_eq!(plan.reason, ProjectionBatchPlanReason::Ready);
        assert_eq!(plan.block_tokens, 4);
        assert_eq!(plan.selected_request_ids, [0, 1, 2, 3]);
    }

    #[test]
    fn exact_projection_batch_rejects_unproven_weights() {
        let mut unknown = model(0);
        unknown.data_hash_available = false;
        let candidates = [candidate(1, unknown), candidate(2, unknown)];
        let plan = plan_exact_projection_batch(&candidates, ProjectionBatchConfig::default());

        assert_eq!(
            plan.reason,
            ProjectionBatchPlanReason::SharedWeightsUnproven
        );
        assert!(!plan.exact);
    }

    #[test]
    fn exact_projection_batch_selects_largest_compatible_group() {
        let candidates = [
            candidate(1, model(1)),
            candidate(2, model(2)),
            candidate(3, model(2)),
            candidate(4, model(2)),
            candidate(5, model(1)),
        ];
        let plan = plan_exact_projection_batch(&candidates, ProjectionBatchConfig::default());

        assert_eq!(plan.reason, ProjectionBatchPlanReason::Ready);
        assert_eq!(plan.block_tokens, 3);
        assert_eq!(plan.selected_request_ids, [2, 3, 4]);
    }

    #[test]
    fn exact_projection_batch_filters_not_ready_or_full_context() {
        let mut stopped = candidate(2, model(7));
        stopped.stopped = true;
        let mut no_context = candidate(3, model(7));
        no_context.prompt_tokens = 100;
        no_context.generated_tokens = 28;
        no_context.max_context_tokens = 128;
        let candidates = [candidate(1, model(7)), stopped, no_context];
        let plan = plan_exact_projection_batch(&candidates, ProjectionBatchConfig::default());

        assert_eq!(
            plan.reason,
            ProjectionBatchPlanReason::InsufficientCompatibleReady
        );
        assert!(!plan.exact);
    }
}
