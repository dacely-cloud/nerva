#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kRequestId = 1u;
constexpr uint32_t kSequenceId = 1u;
constexpr uint32_t kCompletionDeviceComplete = 1u;
constexpr uint64_t kHashSeed = 0xcbf29ce484222325ull;

struct NervaCudaSyntheticState {
  uint32_t token;
  uint64_t next_index;
};

__device__ __forceinline__ uint32_t wrapping_add_one(uint32_t value) {
  return value + 1u;
}

__global__ void nerva_synthetic_graph_step_kernel(
    NervaCudaSyntheticState *state,
    NervaCudaSyntheticTokenSlot *ring,
    uint32_t *history,
    uint32_t history_capacity,
    uint32_t ring_capacity) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || ring_capacity == 0) {
    return;
  }

  const uint64_t token_index = state->next_index;
  const uint32_t slot_index = static_cast<uint32_t>(token_index % ring_capacity);
  NervaCudaSyntheticTokenSlot *slot = &ring[slot_index];
  const uint32_t token = wrapping_add_one(state->token);

  slot->request_id = kRequestId;
  slot->sequence_id = kSequenceId;
  slot->token_index = token_index;
  slot->token = token;
  slot->version = slot->version + 1ull;
  slot->completion = kCompletionDeviceComplete;
  slot->host_copied = 0u;

  state->token = token;
  state->next_index = token_index + 1ull;
  if (history != nullptr && token_index < history_capacity) {
    history[token_index] = token;
  }
}

void clear_result(NervaCudaSyntheticGraphResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail(NervaCudaSyntheticGraphResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

uint64_t hash_observed_token(uint64_t current,
                             uint64_t token_index,
                             uint32_t token) {
  uint64_t hash = current ^ (token_index * 0x9e3779b97f4a7c15ull);
  hash = ((hash << 13) | (hash >> (64 - 13))) ^ static_cast<uint64_t>(token);
  return hash * 0xff51afd7ed558ccdull;
}

uint64_t expected_slot_version(uint64_t token_index, uint32_t ring_capacity) {
  return token_index / static_cast<uint64_t>(ring_capacity) + 1ull;
}

}  // namespace

extern "C" int nerva_cuda_synthetic_graph_smoke(
    uint32_t steps,
    uint32_t ring_capacity,
    uint32_t seed_token,
    NervaCudaSyntheticGraphResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  out->steps = steps;
  out->ring_capacity = ring_capacity;
  out->seed_token = seed_token;

  if (steps == 0 || ring_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  NervaCudaSyntheticState *device_state = nullptr;
  NervaCudaSyntheticState *host_state = nullptr;
  NervaCudaSyntheticTokenSlot *device_ring = nullptr;
  NervaCudaSyntheticTokenSlot *host_ring = nullptr;
  uint32_t *device_history = nullptr;
  uint32_t *host_history = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  const uint64_t ring_bytes =
      static_cast<uint64_t>(ring_capacity) * sizeof(NervaCudaSyntheticTokenSlot);
  const uint64_t history_bytes =
      static_cast<uint64_t>(steps) * sizeof(uint32_t);
  out->device_arena_bytes =
      sizeof(NervaCudaSyntheticState) + ring_bytes + history_bytes;
  out->pinned_host_bytes =
      sizeof(NervaCudaSyntheticState) + ring_bytes + history_bytes;

  err = cudaMalloc(reinterpret_cast<void **>(&device_state),
                   sizeof(NervaCudaSyntheticState));
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_ring),
                   static_cast<size_t>(ring_bytes));
  if (err != cudaSuccess) {
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_history),
                   static_cast<size_t>(history_bytes));
  if (err != cudaSuccess) {
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_state),
                      sizeof(NervaCudaSyntheticState),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_ring),
                      static_cast<size_t>(ring_bytes),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_history),
                      static_cast<size_t>(history_bytes),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  memset(host_state, 0, sizeof(*host_state));
  memset(host_ring, 0, static_cast<size_t>(ring_bytes));
  memset(host_history, 0, static_cast<size_t>(history_bytes));

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_history);
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  NervaCudaSyntheticState initial_state{seed_token, 0ull};
  err = cudaMemsetAsync(device_ring, 0, static_cast<size_t>(ring_bytes), stream);
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_history, 0,
                          static_cast<size_t>(history_bytes), stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_state, &initial_state,
                          sizeof(NervaCudaSyntheticState),
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    cudaStreamDestroy(stream);
    cudaFreeHost(host_history);
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err == cudaSuccess) {
    nerva_synthetic_graph_step_kernel<<<1, 1, 0, stream>>>(
        device_state, device_ring, device_history, steps, ring_capacity);
    err = cudaGetLastError();
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
    cudaFreeHost(host_history);
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  size_t graph_nodes = 0;
  err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_history);
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  out->graph_nodes = static_cast<uint64_t>(graph_nodes);

  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_history);
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  out->observed_token_hash = kHashSeed;

  for (uint32_t step = 0; step < steps; ++step) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err == cudaSuccess) {
      out->graph_launches += 1ull;
    }
    if (err != cudaSuccess) {
      cudaGraphExecDestroy(graph_exec);
      cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
      cudaFreeHost(host_history);
      cudaFreeHost(host_ring);
      cudaFreeHost(host_state);
      cudaFree(device_history);
      cudaFree(device_ring);
      cudaFree(device_state);
      return fail(out, err);
    }
  }

  err = cudaMemcpyAsync(host_history, device_history,
                        static_cast<size_t>(history_bytes),
                        cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) {
    out->d2h_bytes += history_bytes;
    err = cudaMemcpyAsync(host_ring, device_ring,
                          static_cast<size_t>(ring_bytes),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    out->d2h_bytes += ring_bytes;
    err = cudaMemcpyAsync(host_state, device_state,
                          sizeof(NervaCudaSyntheticState),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    out->d2h_bytes += sizeof(NervaCudaSyntheticState);
    err = cudaStreamSynchronize(stream);
    out->sync_calls += 1ull;
  }
  if (err != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_history);
    cudaFreeHost(host_ring);
    cudaFreeHost(host_state);
    cudaFree(device_history);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  out->graph_replays = static_cast<uint64_t>(steps);
  out->observed_tokens = static_cast<uint64_t>(steps);
  out->last_token = host_state->token;

  if (host_state->next_index < static_cast<uint64_t>(steps)) {
    out->missing_tokens =
        static_cast<uint64_t>(steps) - host_state->next_index;
  } else if (host_state->next_index > static_cast<uint64_t>(steps)) {
    out->extra_tokens = host_state->next_index - static_cast<uint64_t>(steps);
  }

  for (uint32_t step = 0; step < steps; ++step) {
    const uint32_t expected_token = seed_token + step + 1u;
    const uint32_t observed_token = host_history[step];
    out->observed_token_hash =
        hash_observed_token(out->observed_token_hash, step, observed_token);
    if (observed_token != expected_token) {
      out->mismatched_tokens += 1ull;
    }
  }

  for (uint32_t slot_index = 0; slot_index < ring_capacity; ++slot_index) {
    if (slot_index >= steps) {
      continue;
    }
    out->token_ring_slots_touched += 1ull;
    const uint64_t token_index =
        static_cast<uint64_t>(slot_index) +
        ((static_cast<uint64_t>(steps - 1u - slot_index) /
          static_cast<uint64_t>(ring_capacity)) *
         static_cast<uint64_t>(ring_capacity));
    const uint32_t expected_token =
        seed_token + static_cast<uint32_t>(token_index) + 1u;
    const uint64_t expected_version =
        expected_slot_version(token_index, ring_capacity);
    const NervaCudaSyntheticTokenSlot *slot = &host_ring[slot_index];
    if (slot->version > out->token_ring_max_slot_version) {
      out->token_ring_max_slot_version = slot->version;
    }
    if (slot->request_id != kRequestId || slot->sequence_id != kSequenceId ||
        slot->completion != kCompletionDeviceComplete ||
        slot->token_index != token_index || slot->token != expected_token ||
        slot->version != expected_version) {
      out->mismatched_tokens += 1ull;
    }
  }

  if (steps > ring_capacity) {
    out->token_ring_reuses =
        static_cast<uint64_t>(steps - ring_capacity);
  }

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(host_history);
  cudaFreeHost(host_ring);
  cudaFreeHost(host_state);
  cudaFree(device_history);
  cudaFree(device_ring);
  cudaFree(device_state);

  if (out->stale_tokens != 0 || out->missing_tokens != 0 ||
      out->extra_tokens != 0 || out->mismatched_tokens != 0 ||
      out->host_causality_edges != 0) {
    out->status = -1;
    return -1;
  }

  out->status = 0;
  return 0;
}
