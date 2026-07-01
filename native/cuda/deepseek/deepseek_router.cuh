#pragma once

#include <math.h>
#include <stdint.h>

namespace nerva {
namespace deepseek {
namespace router {

constexpr uint32_t kMaxTopK = 16;
constexpr uint32_t kMaxGroups = 64;

enum Kind : uint32_t {
  kV3GroupedSigmoid = 1,
  kV4SqrtSoftplus = 2,
  kV4Hash = 3,
};

__host__ __device__ inline float sigmoid_score(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__host__ __device__ inline float softplus_score(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return log1pf(expf(value));
}

__host__ __device__ inline float sqrtsoftplus_score(float value) {
  return sqrtf(softplus_score(value));
}

__host__ __device__ inline bool better_score(float lhs_score,
                                             uint32_t lhs_id,
                                             float rhs_score,
                                             uint32_t rhs_id) {
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_id < rhs_id);
}

__host__ __device__ inline void insert_topk(float score,
                                            uint32_t id,
                                            float *top_scores,
                                            uint32_t *top_ids,
                                            uint32_t k) {
  for (uint32_t slot = 0; slot < k; ++slot) {
    if (better_score(score, id, top_scores[slot], top_ids[slot])) {
      for (uint32_t shift = k - 1; shift > slot; --shift) {
        top_scores[shift] = top_scores[shift - 1];
        top_ids[shift] = top_ids[shift - 1];
      }
      top_scores[slot] = score;
      top_ids[slot] = id;
      return;
    }
  }
}

__host__ __device__ inline float route_scale(float weight_sum,
                                             uint32_t norm_topk_prob,
                                             float routed_scaling_factor) {
  if (norm_topk_prob != 0) {
    return weight_sum == 0.0f ? 0.0f : routed_scaling_factor / weight_sum;
  }
  return routed_scaling_factor;
}

__host__ __device__ inline bool validate_common(uint32_t num_experts,
                                                uint32_t top_k) {
  return num_experts > 0 && top_k > 0 && top_k <= kMaxTopK &&
         top_k <= num_experts;
}

__host__ __device__ inline int route_v4_sqrtsoftplus(
    const float *logits,
    const float *correction_bias,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    float routed_scaling_factor,
    uint32_t *expert_ids,
    float *weights) {
  if (logits == nullptr || expert_ids == nullptr || weights == nullptr ||
      !validate_common(num_experts, top_k)) {
    return -1;
  }

  float top_scores[kMaxTopK];
  uint32_t top_ids[kMaxTopK];
  for (uint32_t i = 0; i < top_k; ++i) {
    top_scores[i] = -INFINITY;
    top_ids[i] = i;
  }

  for (uint32_t expert = 0; expert < num_experts; ++expert) {
    const float raw = sqrtsoftplus_score(logits[expert]);
    const float choice =
        raw + (correction_bias == nullptr ? 0.0f : correction_bias[expert]);
    insert_topk(choice, expert, top_scores, top_ids, top_k);
  }

  float weight_sum = 0.0f;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    weights[rank] = sqrtsoftplus_score(logits[top_ids[rank]]);
    weight_sum += weights[rank];
  }
  const float scale = route_scale(weight_sum, norm_topk_prob, routed_scaling_factor);
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    expert_ids[rank] = top_ids[rank];
    weights[rank] *= scale;
  }
  return 0;
}

__host__ __device__ inline int route_v4_hash(
    const float *logits,
    const uint32_t *hash_route_table,
    uint32_t hash_route_table_len,
    uint32_t route_token,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    float routed_scaling_factor,
    uint32_t *expert_ids,
    float *weights) {
  if (logits == nullptr || hash_route_table == nullptr ||
      expert_ids == nullptr || weights == nullptr ||
      !validate_common(num_experts, top_k)) {
    return -1;
  }
  const uint64_t start = static_cast<uint64_t>(route_token) * top_k;
  const uint64_t end = start + top_k;
  if (end > hash_route_table_len) {
    return -2;
  }

  float weight_sum = 0.0f;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = hash_route_table[start + rank];
    if (expert >= num_experts) {
      return -3;
    }
    expert_ids[rank] = expert;
    weights[rank] = sqrtsoftplus_score(logits[expert]);
    weight_sum += weights[rank];
  }
  const float scale = route_scale(weight_sum, norm_topk_prob, routed_scaling_factor);
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    weights[rank] *= scale;
  }
  return 0;
}

__host__ __device__ inline float top2_sum_group(const float *logits,
                                                const float *correction_bias,
                                                uint32_t start,
                                                uint32_t len) {
  float first = -INFINITY;
  float second = -INFINITY;
  for (uint32_t i = 0; i < len; ++i) {
    const uint32_t expert = start + i;
    const float raw = sigmoid_score(logits[expert]);
    const float choice =
        raw + (correction_bias == nullptr ? 0.0f : correction_bias[expert]);
    if (choice > first) {
      second = first;
      first = choice;
    } else if (choice > second) {
      second = choice;
    }
  }
  return first + second;
}

__host__ __device__ inline bool group_selected(uint32_t group,
                                               const uint32_t *group_ids,
                                               uint32_t top_k_groups) {
  for (uint32_t i = 0; i < top_k_groups; ++i) {
    if (group_ids[i] == group) {
      return true;
    }
  }
  return false;
}

__host__ __device__ inline int route_v3_grouped_sigmoid(
    const float *logits,
    const float *correction_bias,
    uint32_t num_experts,
    uint32_t num_groups,
    uint32_t top_k_groups,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    float routed_scaling_factor,
    uint32_t *expert_ids,
    float *weights) {
  if (logits == nullptr || expert_ids == nullptr || weights == nullptr ||
      !validate_common(num_experts, top_k) || num_groups == 0 ||
      num_groups > kMaxGroups || top_k_groups == 0 ||
      top_k_groups > num_groups || num_experts % num_groups != 0) {
    return -1;
  }

  const uint32_t experts_per_group = num_experts / num_groups;
  float group_scores[kMaxGroups];
  uint32_t group_ids[kMaxGroups];
  for (uint32_t i = 0; i < top_k_groups; ++i) {
    group_scores[i] = -INFINITY;
    group_ids[i] = i;
  }
  for (uint32_t group = 0; group < num_groups; ++group) {
    const float score = top2_sum_group(
        logits, correction_bias, group * experts_per_group, experts_per_group);
    insert_topk(score, group, group_scores, group_ids, top_k_groups);
  }

  float top_scores[kMaxTopK];
  uint32_t top_ids[kMaxTopK];
  for (uint32_t i = 0; i < top_k; ++i) {
    top_scores[i] = -INFINITY;
    top_ids[i] = i;
  }
  for (uint32_t group = 0; group < num_groups; ++group) {
    if (!group_selected(group, group_ids, top_k_groups)) {
      continue;
    }
    const uint32_t start = group * experts_per_group;
    for (uint32_t i = 0; i < experts_per_group; ++i) {
      const uint32_t expert = start + i;
      const float raw = sigmoid_score(logits[expert]);
      const float choice =
          raw + (correction_bias == nullptr ? 0.0f : correction_bias[expert]);
      insert_topk(choice, expert, top_scores, top_ids, top_k);
    }
  }

  float weight_sum = 0.0f;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    weights[rank] = sigmoid_score(logits[top_ids[rank]]);
    weight_sum += weights[rank];
  }
  const float scale = route_scale(weight_sum, norm_topk_prob, routed_scaling_factor);
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    expert_ids[rank] = top_ids[rank];
    weights[rank] *= scale;
  }
  return 0;
}

}  // namespace router
}  // namespace deepseek
}  // namespace nerva
