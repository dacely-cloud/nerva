#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <chrono>
#include <new>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kRequestId = 1u;
constexpr uint32_t kSequenceId = 1u;
constexpr uint32_t kCompletionDeviceComplete = 1u;
constexpr uint32_t kHidden = 2u;
constexpr uint32_t kVocabSize = 4u;
constexpr uint32_t kEmbeddingFloats = kVocabSize * kHidden;
constexpr uint32_t kLmHeadFloats = kVocabSize * kHidden;
constexpr uint32_t kLmHeadOffset = kEmbeddingFloats;
constexpr uint32_t kModelFloats = kEmbeddingFloats + kLmHeadFloats;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;

struct NervaCudaTinyDecodeState {
  uint32_t token;
  uint64_t next_index;
};

__global__ void tiny_decode_step_kernel(
    NervaCudaTinyDecodeState *state,
    const float *model,
    NervaCudaSyntheticTokenSlot *ring,
    NervaCudaSyntheticTokenSlot *observation,
    uint32_t ring_capacity) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || ring_capacity == 0) {
    return;
  }

  const uint32_t current_token = state->token;
  const float *embedding = model + current_token * kHidden;
  const float hidden0 = embedding[0];
  const float hidden1 = embedding[1];
  const float *lm_head = model + kLmHeadOffset;

  uint32_t best_token = 0u;
  float best_value = lm_head[0] * hidden0 + lm_head[1] * hidden1;
  for (uint32_t token = 1u; token < kVocabSize; ++token) {
    const float *row = lm_head + token * kHidden;
    const float value = row[0] * hidden0 + row[1] * hidden1;
    if (value > best_value) {
      best_value = value;
      best_token = token;
    }
  }

  const uint64_t token_index = state->next_index;
  const uint32_t slot_index = static_cast<uint32_t>(token_index % ring_capacity);
  NervaCudaSyntheticTokenSlot *slot = &ring[slot_index];
  slot->request_id = kRequestId;
  slot->sequence_id = kSequenceId;
  slot->token_index = token_index;
  slot->token = best_token;
  slot->version = slot->version + 1ull;
  slot->completion = kCompletionDeviceComplete;
  slot->host_copied = 0u;

  state->token = best_token;
  state->next_index = token_index + 1ull;
  *observation = *slot;
}

void clear_result(NervaCudaTinyDecodeResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->vocab_size = kVocabSize;
  out->hidden = kHidden;
  out->resident_weight_bytes = sizeof(float) * kModelFloats;
}

int fail(NervaCudaTinyDecodeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point stop) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
          .count());
}

uint64_t elapsed_event_ns(cudaEvent_t start, cudaEvent_t stop) {
  float elapsed_ms = 0.0f;
  const cudaError_t err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    return 0ull;
  }
  const uint64_t elapsed =
      static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  return elapsed == 0ull ? 1ull : elapsed;
}

uint64_t hash_token(uint64_t current, uint32_t token) {
  uint64_t hash = current;
  for (uint32_t shift = 0u; shift < 32u; shift += 8u) {
    hash ^= static_cast<uint64_t>((token >> shift) & 0xffu);
    hash *= kFnvPrime;
  }
  return hash;
}

uint32_t expected_token(uint32_t seed_token, uint32_t step) {
  return (seed_token + step + 1u) % kVocabSize;
}

uint64_t expected_slot_version(uint64_t token_index, uint32_t ring_capacity) {
  return token_index / static_cast<uint64_t>(ring_capacity) + 1ull;
}

void fill_host_model(float *model) {
  const float embeddings[kEmbeddingFloats] = {
      1.0f, 0.0f,
      0.0f, 1.0f,
      -1.0f, 0.0f,
      0.0f, -1.0f,
  };
  const float lm_head[kLmHeadFloats] = {
      0.0f, -1.0f,
      1.0f, 0.0f,
      0.0f, 1.0f,
      -1.0f, 0.0f,
  };
  memcpy(model, embeddings, sizeof(embeddings));
  memcpy(model + kLmHeadOffset, lm_head, sizeof(lm_head));
}

}  // namespace

extern "C" int nerva_cuda_tiny_decode_smoke(
    uint32_t steps,
    uint32_t ring_capacity,
    uint32_t seed_token,
    NervaCudaTinyDecodeResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  out->steps = steps;
  out->ring_capacity = ring_capacity;
  out->seed_token = seed_token;
  out->last_token = seed_token;

  if (steps == 0 || ring_capacity == 0 || seed_token >= kVocabSize) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  NervaCudaTinyDecodeState *device_state = nullptr;
  float *device_model = nullptr;
  NervaCudaSyntheticTokenSlot *device_ring = nullptr;
  NervaCudaSyntheticTokenSlot *device_observation = nullptr;
  float *host_model = nullptr;
  NervaCudaSyntheticTokenSlot *host_observation = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaEvent_t active_start = nullptr;
  cudaEvent_t active_stop = nullptr;

  const uint64_t ring_bytes =
      static_cast<uint64_t>(ring_capacity) * sizeof(NervaCudaSyntheticTokenSlot);
  out->device_arena_bytes =
      sizeof(NervaCudaTinyDecodeState) + out->resident_weight_bytes +
      ring_bytes + sizeof(NervaCudaSyntheticTokenSlot);
  out->pinned_host_bytes =
      out->resident_weight_bytes + sizeof(NervaCudaSyntheticTokenSlot);

  err = cudaMalloc(reinterpret_cast<void **>(&device_state),
                   sizeof(NervaCudaTinyDecodeState));
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_model),
                   static_cast<size_t>(out->resident_weight_bytes));
  if (err != cudaSuccess) {
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_ring),
                   static_cast<size_t>(ring_bytes));
  if (err != cudaSuccess) {
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_observation),
                   sizeof(NervaCudaSyntheticTokenSlot));
  if (err != cudaSuccess) {
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_model),
                      static_cast<size_t>(out->resident_weight_bytes),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_observation),
                      sizeof(NervaCudaSyntheticTokenSlot),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }
  fill_host_model(host_model);
  memset(host_observation, 0, sizeof(*host_observation));

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }

  NervaCudaTinyDecodeState initial_state{seed_token, 0ull};
  err = cudaMemsetAsync(device_ring, 0, static_cast<size_t>(ring_bytes), stream);
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_observation, 0,
                          sizeof(NervaCudaSyntheticTokenSlot), stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_model, host_model,
                          static_cast<size_t>(out->resident_weight_bytes),
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += out->resident_weight_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_state, &initial_state,
                          sizeof(NervaCudaTinyDecodeState),
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += sizeof(NervaCudaTinyDecodeState);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }

  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err == cudaSuccess) {
    tiny_decode_step_kernel<<<1, 1, 0, stream>>>(
        device_state, device_model, device_ring, device_observation,
        ring_capacity);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_observation, device_observation,
                          sizeof(NervaCudaSyntheticTokenSlot),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamEndCapture(stream, &graph);
  } else {
    cudaStreamEndCapture(stream, &graph);
  }
  if (err != cudaSuccess) {
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }

  size_t graph_nodes = 0;
  err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }
  out->graph_nodes = static_cast<uint64_t>(graph_nodes);

  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaEventCreateWithFlags(&active_start, cudaEventDefault);
  if (err == cudaSuccess) {
    err = cudaEventCreateWithFlags(&active_stop, cudaEventDefault);
  }
  if (err != cudaSuccess) {
    if (active_start != nullptr) {
      cudaEventDestroy(active_start);
    }
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, err);
  }

  out->observed_token_hash = kFnvOffset;
  bool *slot_seen = new (std::nothrow) bool[ring_capacity]();
  if (slot_seen == nullptr) {
    cudaEventDestroy(active_stop);
    cudaEventDestroy(active_start);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFreeHost(host_model);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_model);
    cudaFree(device_state);
    return fail(out, cudaErrorMemoryAllocation);
  }

  for (uint32_t step = 0; step < steps; ++step) {
    const auto wall_start = std::chrono::steady_clock::now();
    uint64_t wait_ns = 0ull;
    err = cudaEventRecord(active_start, stream);
    if (err == cudaSuccess) {
      err = cudaGraphLaunch(graph_exec, stream);
    }
    if (err == cudaSuccess) {
      out->graph_launches += 1ull;
      out->kernel_launches += 1ull;
      err = cudaEventRecord(active_stop, stream);
    }
    if (err == cudaSuccess) {
      const auto wait_start = std::chrono::steady_clock::now();
      err = cudaStreamSynchronize(stream);
      const auto wait_stop = std::chrono::steady_clock::now();
      wait_ns = elapsed_ns(wait_start, wait_stop);
      out->sync_calls += 1ull;
    }
    const auto wall_stop = std::chrono::steady_clock::now();
    const uint64_t wall_ns = elapsed_ns(wall_start, wall_stop);
    out->wall_latency_ns += wall_ns;
    out->host_event_wait_ns += wait_ns;
    if (err != cudaSuccess) {
      delete[] slot_seen;
      cudaEventDestroy(active_stop);
      cudaEventDestroy(active_start);
      cudaGraphExecDestroy(graph_exec);
      cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
      cudaFreeHost(host_observation);
      cudaFreeHost(host_model);
      cudaFree(device_observation);
      cudaFree(device_ring);
      cudaFree(device_model);
      cudaFree(device_state);
      return fail(out, err);
    }
    out->gpu_active_ns += elapsed_event_ns(active_start, active_stop);
    out->token_ledgers += 1ull;
    out->graph_replay_events += 1ull;
    out->device_activity_events += 1ull;
    out->copy_events += 1ull;
    out->soft_visibility_syncs += 1ull;

    const uint64_t token_index = host_observation->token_index;
    const uint32_t slot_index =
        static_cast<uint32_t>(token_index % ring_capacity);
    const uint32_t expected = expected_token(seed_token, step);
    const uint64_t expected_version =
        expected_slot_version(token_index, ring_capacity);

    out->graph_replays += 1ull;
    out->observed_tokens += 1ull;
    out->d2h_bytes += sizeof(NervaCudaSyntheticTokenSlot);
    out->last_token = host_observation->token;
    out->observed_token_hash =
        hash_token(out->observed_token_hash, host_observation->token);

    if (!slot_seen[slot_index]) {
      slot_seen[slot_index] = true;
      out->token_ring_slots_touched += 1ull;
    }
    if (host_observation->version > 1ull) {
      out->token_ring_reuses += 1ull;
    }
    if (host_observation->version > out->token_ring_max_slot_version) {
      out->token_ring_max_slot_version = host_observation->version;
    }
    if (token_index < step) {
      out->stale_tokens += 1ull;
    } else if (token_index > step) {
      out->extra_tokens += 1ull;
    }
    if (host_observation->request_id != kRequestId ||
        host_observation->sequence_id != kSequenceId ||
        host_observation->completion != kCompletionDeviceComplete ||
        host_observation->token != expected ||
        host_observation->version != expected_version) {
      out->mismatched_tokens += 1ull;
    }
  }

  out->missing_tokens = out->observed_tokens <= static_cast<uint64_t>(steps)
                            ? static_cast<uint64_t>(steps) - out->observed_tokens
                            : 0ull;
  delete[] slot_seen;

  cudaEventDestroy(active_stop);
  cudaEventDestroy(active_start);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(host_observation);
  cudaFreeHost(host_model);
  cudaFree(device_observation);
  cudaFree(device_ring);
  cudaFree(device_model);
  cudaFree(device_state);

  if (out->observed_tokens != static_cast<uint64_t>(steps) ||
      out->stale_tokens != 0 || out->missing_tokens != 0 ||
      out->extra_tokens != 0 || out->mismatched_tokens != 0 ||
      out->host_causality_edges != 0) {
    out->status = -1;
    return -1;
  }

  out->status = 0;
  return 0;
}
