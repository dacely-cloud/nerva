use std::collections::BTreeSet;

use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;

use crate::engine::hf_cuda_decode::batch_advance::{
    advance_decode_loops_once, advance_decode_loops_sequential_once, CudaDecodeBatchAdvanceConfig,
    CudaDecodeBatchAdvanceMode, CudaDecodeBatchAdvanceOutput,
};
use crate::engine::hf_cuda_decode::projection_batch::{
    plan_exact_projection_batch, ProjectionBatchCandidate, ProjectionBatchConfig,
    ProjectionBatchModelKey, ProjectionBatchPlan, ProjectionBatchPlanReason,
};

const NOT_SELECTED_REASON: &str = "not_selected_for_projection_batch";

pub struct CudaDecodeLoopBatchEntry<'a, 'session> {
    pub candidate: ProjectionBatchCandidate,
    pub loop_state: &'a mut CudaHfDecodeSequenceLoop<'session>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuousProjectionBatchPlan {
    pub projection: ProjectionBatchPlan,
    pub selected_indices: Vec<usize>,
    pub selected_groups: Vec<Vec<usize>>,
    pub fallback_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct CudaContinuousDecodeLoopRecord {
    pub input_index: usize,
    pub request_id: u64,
    pub tokens: Vec<u32>,
    pub mode: CudaDecodeBatchAdvanceMode,
}

#[derive(Clone, Debug)]
pub struct CudaContinuousDecodeBatchOutput {
    pub plan: ContinuousProjectionBatchPlan,
    pub records: Vec<CudaContinuousDecodeLoopRecord>,
    pub selected: Vec<CudaDecodeBatchAdvanceOutput>,
    pub fallback: Option<CudaDecodeBatchAdvanceOutput>,
}

impl CudaContinuousDecodeBatchOutput {
    pub fn observed_tokens(&self) -> usize {
        self.records.iter().map(|record| record.tokens.len()).sum()
    }

    pub fn used_batched_projection(&self) -> bool {
        self.selected
            .iter()
            .any(CudaDecodeBatchAdvanceOutput::used_batched_projection)
    }
}

pub fn plan_continuous_projection_batch(
    candidates: &[ProjectionBatchCandidate],
    config: ProjectionBatchConfig,
) -> ContinuousProjectionBatchPlan {
    let projection = plan_exact_projection_batch(candidates, config);
    let selected_groups = if projection.reason == ProjectionBatchPlanReason::Ready {
        continuous_projection_groups(candidates, config)
    } else {
        Vec::new()
    };
    let selected_index_set = selected_groups
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut selected_indices = Vec::new();
    let mut fallback_indices = Vec::new();
    for index in 0..candidates.len() {
        if selected_index_set.contains(&index) {
            selected_indices.push(index);
        } else {
            fallback_indices.push(index);
        }
    }
    ContinuousProjectionBatchPlan {
        projection,
        selected_indices,
        selected_groups,
        fallback_indices,
    }
}

pub fn advance_continuous_decode_batch_once<'a, 'session>(
    entries: Vec<CudaDecodeLoopBatchEntry<'a, 'session>>,
    config: ProjectionBatchConfig,
) -> CudaContinuousDecodeBatchOutput {
    let candidates = entries
        .iter()
        .map(|entry| entry.candidate)
        .collect::<Vec<_>>();
    let plan = plan_continuous_projection_batch(&candidates, config);
    let mut entry_slots = entries
        .into_iter()
        .enumerate()
        .map(|(input_index, entry)| Some(IndexedLoopEntry { input_index, entry }))
        .collect::<Vec<_>>();

    let mut records = Vec::new();
    let mut selected = Vec::new();
    for group in &plan.selected_groups {
        let mut group_entries = Vec::with_capacity(group.len());
        for &input_index in group {
            if let Some(slot) = entry_slots.get_mut(input_index) {
                if let Some(entry) = slot.take() {
                    group_entries.push(entry);
                }
            }
        }
        if let Some(output) =
            advance_selected_entries_once(&mut group_entries, config, &mut records)
        {
            let failed = matches!(output.mode, CudaDecodeBatchAdvanceMode::BatchFailed { .. });
            selected.push(output);
            if failed {
                return CudaContinuousDecodeBatchOutput {
                    plan,
                    records,
                    selected,
                    fallback: None,
                };
            }
        }
    }

    let mut fallback_entries = entry_slots.into_iter().flatten().collect::<Vec<_>>();
    let fallback_reason = fallback_reason(plan.projection.reason);
    let fallback =
        advance_fallback_entries_once(&mut fallback_entries, fallback_reason, &mut records);
    records.sort_by_key(|record| record.input_index);
    CudaContinuousDecodeBatchOutput {
        plan,
        records,
        selected,
        fallback,
    }
}

struct IndexedLoopEntry<'a, 'session> {
    input_index: usize,
    entry: CudaDecodeLoopBatchEntry<'a, 'session>,
}

fn continuous_projection_groups(
    candidates: &[ProjectionBatchCandidate],
    config: ProjectionBatchConfig,
) -> Vec<Vec<usize>> {
    let mut model_groups: Vec<(ProjectionBatchModelKey, Vec<usize>)> = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.can_decode_one_token() || !candidate.model.data_hash_available {
            continue;
        }
        if let Some((_, group)) = model_groups
            .iter_mut()
            .find(|(model, _)| model.proves_same_weights_as(&candidate.model))
        {
            group.push(index);
        } else {
            model_groups.push((candidate.model, vec![index]));
        }
    }

    let mut groups = Vec::new();
    for (_, indices) in model_groups {
        groups.extend(split_projection_group(indices, config));
    }
    groups
}

fn split_projection_group(indices: Vec<usize>, config: ProjectionBatchConfig) -> Vec<Vec<usize>> {
    let target = config.effective_target_block_tokens();
    let min = config.effective_min_block_tokens();
    if indices.len() < min {
        return Vec::new();
    }
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < indices.len() {
        let remaining = indices.len() - start;
        if remaining < min {
            break;
        }
        let mut take = remaining.min(target);
        let tail = remaining.saturating_sub(take);
        if tail > 0 && tail < min {
            take = take.saturating_sub(min - tail).max(min);
        }
        groups.push(indices[start..start + take].to_vec());
        start += take;
    }
    groups
}

fn advance_selected_entries_once(
    entries: &mut [IndexedLoopEntry<'_, '_>],
    config: ProjectionBatchConfig,
    records: &mut Vec<CudaContinuousDecodeLoopRecord>,
) -> Option<CudaDecodeBatchAdvanceOutput> {
    if entries.is_empty() {
        return None;
    }
    let mut loop_refs = entries
        .iter_mut()
        .map(|entry| &mut *entry.entry.loop_state)
        .collect::<Vec<_>>();
    let output = advance_decode_loops_once(
        &mut loop_refs,
        CudaDecodeBatchAdvanceConfig::new(
            u32::try_from(config.effective_target_block_tokens()).unwrap_or(u32::MAX),
            u32::try_from(config.effective_min_block_tokens()).unwrap_or(u32::MAX),
        ),
    );
    push_records(entries, &output, records);
    Some(output)
}

fn advance_fallback_entries_once(
    entries: &mut [IndexedLoopEntry<'_, '_>],
    reason: &'static str,
    records: &mut Vec<CudaContinuousDecodeLoopRecord>,
) -> Option<CudaDecodeBatchAdvanceOutput> {
    if entries.is_empty() {
        return None;
    }
    let mut loop_refs = entries
        .iter_mut()
        .map(|entry| &mut *entry.entry.loop_state)
        .collect::<Vec<_>>();
    let output = advance_decode_loops_sequential_once(&mut loop_refs, reason);
    push_records(entries, &output, records);
    Some(output)
}

fn push_records(
    entries: &[IndexedLoopEntry<'_, '_>],
    output: &CudaDecodeBatchAdvanceOutput,
    records: &mut Vec<CudaContinuousDecodeLoopRecord>,
) {
    for (index, entry) in entries.iter().enumerate() {
        records.push(CudaContinuousDecodeLoopRecord {
            input_index: entry.input_index,
            request_id: entry.entry.candidate.request_id,
            tokens: output
                .tokens_by_loop
                .get(index)
                .cloned()
                .unwrap_or_default(),
            mode: output.mode,
        });
    }
}

fn fallback_reason(reason: ProjectionBatchPlanReason) -> &'static str {
    if reason == ProjectionBatchPlanReason::Ready {
        NOT_SELECTED_REASON
    } else {
        reason.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nerva_core::types::dtype::DType;

    fn model(
        data_hash: u64,
    ) -> crate::engine::hf_cuda_decode::projection_batch::ProjectionBatchModelKey {
        crate::engine::hf_cuda_decode::projection_batch::ProjectionBatchModelKey {
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

    fn candidate(request_id: u64, data_hash: u64) -> ProjectionBatchCandidate {
        ProjectionBatchCandidate {
            request_id,
            model: model(data_hash),
            prompt_tokens: 128,
            generated_tokens: 8,
            remaining_tokens: 16,
            max_context_tokens: 4096,
            ready: true,
            stopped: false,
        }
    }

    #[test]
    fn continuous_plan_selects_compatible_group_and_marks_fallbacks() {
        let candidates = [
            candidate(10, 1),
            candidate(11, 2),
            candidate(12, 2),
            candidate(13, 2),
            candidate(14, 3),
        ];
        let plan = plan_continuous_projection_batch(&candidates, ProjectionBatchConfig::default());

        assert_eq!(plan.projection.reason, ProjectionBatchPlanReason::Ready);
        assert_eq!(plan.projection.selected_request_ids, [11, 12, 13]);
        assert_eq!(plan.selected_indices, [1, 2, 3]);
        assert_eq!(plan.selected_groups, vec![vec![1, 2, 3]]);
        assert_eq!(plan.fallback_indices, [0, 4]);
    }

    #[test]
    fn continuous_plan_batches_all_full_compatible_groups() {
        let candidates = (0..6)
            .map(|index| candidate(index as u64, 7))
            .collect::<Vec<_>>();
        let plan = plan_continuous_projection_batch(&candidates, ProjectionBatchConfig::new(4, 2));

        assert_eq!(plan.projection.reason, ProjectionBatchPlanReason::Ready);
        assert_eq!(plan.selected_indices, [0, 1, 2, 3, 4, 5]);
        assert_eq!(plan.selected_groups, vec![vec![0, 1, 2, 3], vec![4, 5]]);
        assert!(plan.fallback_indices.is_empty());
    }

    #[test]
    fn continuous_plan_rebalances_tiny_tail_into_batch_group() {
        let candidates = (0..17)
            .map(|index| candidate(index as u64, 7))
            .collect::<Vec<_>>();
        let plan = plan_continuous_projection_batch(&candidates, ProjectionBatchConfig::new(16, 2));

        assert_eq!(
            plan.selected_groups,
            vec![(0..15).collect::<Vec<_>>(), vec![15, 16]]
        );
        assert!(plan.fallback_indices.is_empty());
    }

    #[test]
    fn continuous_plan_caps_groups_to_native_width() {
        let candidates = (0..64)
            .map(|index| candidate(index as u64, 7))
            .collect::<Vec<_>>();
        let plan = plan_continuous_projection_batch(&candidates, ProjectionBatchConfig::new(64, 2));

        assert_eq!(
            plan.selected_groups,
            vec![(0..32).collect::<Vec<_>>(), (32..64).collect::<Vec<_>>()]
        );
        assert!(plan.fallback_indices.is_empty());
    }

    #[test]
    fn continuous_plan_falls_back_all_when_batch_is_not_exact() {
        let mut unknown = model(0);
        unknown.data_hash_available = false;
        let candidates = [
            ProjectionBatchCandidate {
                model: unknown,
                ..candidate(1, 0)
            },
            ProjectionBatchCandidate {
                model: unknown,
                ..candidate(2, 0)
            },
        ];
        let plan = plan_continuous_projection_batch(&candidates, ProjectionBatchConfig::default());

        assert_eq!(
            plan.projection.reason,
            ProjectionBatchPlanReason::SharedWeightsUnproven
        );
        assert!(plan.selected_indices.is_empty());
        assert!(plan.selected_groups.is_empty());
        assert_eq!(plan.fallback_indices, [0, 1]);
    }

    #[test]
    fn ready_plan_leftovers_use_not_selected_reason() {
        assert_eq!(
            fallback_reason(ProjectionBatchPlanReason::Ready),
            NOT_SELECTED_REASON
        );
        assert_eq!(
            fallback_reason(ProjectionBatchPlanReason::NoReadyCandidates),
            "no_ready_candidates"
        );
    }
}
