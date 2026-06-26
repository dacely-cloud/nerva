#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <new>
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
    NervaCudaSyntheticTokenSlot *observation,
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
  *observation = *slot;
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
  NervaCudaSyntheticTokenSlot *device_ring = nullptr;
  NervaCudaSyntheticTokenSlot *device_observation = nullptr;
  NervaCudaSyntheticTokenSlot *host_observation = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  const uint64_t ring_bytes =
      static_cast<uint64_t>(ring_capacity) * sizeof(NervaCudaSyntheticTokenSlot);
  out->device_arena_bytes =
      sizeof(NervaCudaSyntheticState) + ring_bytes +
      sizeof(NervaCudaSyntheticTokenSlot);
  out->pinned_host_bytes = sizeof(NervaCudaSyntheticTokenSlot);

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
  err = cudaMalloc(reinterpret_cast<void **>(&device_observation),
                   sizeof(NervaCudaSyntheticTokenSlot));
  if (err != cudaSuccess) {
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_observation),
                      sizeof(NervaCudaSyntheticTokenSlot),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  memset(host_observation, 0, sizeof(*host_observation));

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_observation);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  NervaCudaSyntheticState initial_state{seed_token, 0ull};
  err = cudaMemsetAsync(device_ring, 0, static_cast<size_t>(ring_bytes), stream);
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_observation, 0,
                          sizeof(NervaCudaSyntheticTokenSlot), stream);
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
    cudaFreeHost(host_observation);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err == cudaSuccess) {
    nerva_synthetic_graph_step_kernel<<<1, 1, 0, stream>>>(
        device_state, device_ring, device_observation, ring_capacity);
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
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  size_t graph_nodes = 0;
  err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }
  out->graph_nodes = static_cast<uint64_t>(graph_nodes);

  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, err);
  }

  out->observed_token_hash = kHashSeed;
  bool *slot_seen = new (std::nothrow) bool[ring_capacity]();
  if (slot_seen == nullptr) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_observation);
    cudaFree(device_observation);
    cudaFree(device_ring);
    cudaFree(device_state);
    return fail(out, cudaErrorMemoryAllocation);
  }

  for (uint32_t step = 0; step < steps; ++step) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err == cudaSuccess) {
      out->graph_launches += 1ull;
      err = cudaStreamSynchronize(stream);
      out->sync_calls += 1ull;
    }
    if (err != cudaSuccess) {
      delete[] slot_seen;
      cudaGraphExecDestroy(graph_exec);
      cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
      cudaFreeHost(host_observation);
      cudaFree(device_observation);
      cudaFree(device_ring);
      cudaFree(device_state);
      return fail(out, err);
    }

    const uint64_t token_index = host_observation->token_index;
    const uint32_t slot_index =
        static_cast<uint32_t>(token_index % ring_capacity);
    const uint32_t expected_token = seed_token + step + 1u;
    const uint64_t expected_version =
        expected_slot_version(token_index, ring_capacity);

    out->graph_replays += 1ull;
    out->observed_tokens += 1ull;
    out->d2h_bytes += sizeof(NervaCudaSyntheticTokenSlot);
    out->last_token = host_observation->token;
    out->observed_token_hash = hash_observed_token(
        out->observed_token_hash, token_index, host_observation->token);

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
        host_observation->token != expected_token ||
        host_observation->version != expected_version) {
      out->mismatched_tokens += 1ull;
    }
  }

  out->missing_tokens = static_cast<uint64_t>(steps) - out->observed_tokens;
  delete[] slot_seen;

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(host_observation);
  cudaFree(device_observation);
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
