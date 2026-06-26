#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NervaCudaDeviceSmokeResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t value;
  int32_t device_count;
  int32_t device_ordinal;
  int32_t driver_version;
  int32_t runtime_version;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  uint64_t total_global_mem;
  char gpu_name[128];
  char pci_bus_id[32];
} NervaCudaDeviceSmokeResult;

typedef struct NervaCudaSyntheticTokenSlot {
  uint32_t request_id;
  uint32_t sequence_id;
  uint64_t token_index;
  uint32_t token;
  uint64_t version;
  uint32_t completion;
  uint32_t host_copied;
} NervaCudaSyntheticTokenSlot;

typedef struct NervaCudaSyntheticGraphResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t steps;
  uint32_t ring_capacity;
  uint32_t seed_token;
  uint32_t last_token;
  uint64_t graph_replays;
  uint64_t graph_nodes;
  uint64_t observed_tokens;
  uint64_t observed_token_hash;
  uint64_t token_ring_slots_touched;
  uint64_t token_ring_reuses;
  uint64_t token_ring_max_slot_version;
  uint64_t stale_tokens;
  uint64_t missing_tokens;
  uint64_t extra_tokens;
  uint64_t mismatched_tokens;
  uint64_t host_causality_edges;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t graph_launches;
  uint64_t sync_calls;
  uint64_t d2h_bytes;
  uint64_t hot_path_allocations;
} NervaCudaSyntheticGraphResult;

typedef struct NervaCudaTinyBlockResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t hidden;
  uint32_t intermediate;
  uint16_t output[2];
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t d2h_bytes;
  uint64_t hot_path_allocations;
} NervaCudaTinyBlockResult;

typedef struct NervaCudaLoadedTinyBlockResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t hidden;
  uint32_t intermediate;
  uint16_t output[2];
  uint64_t output_hash;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaLoadedTinyBlockResult;

typedef struct NervaCudaGreedySamplerResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t vocab_size;
  uint64_t token_index;
  uint32_t token;
  uint64_t slot_version;
  uint32_t completion;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaGreedySamplerResult;

int nerva_cuda_device_smoke(NervaCudaDeviceSmokeResult *out);
int nerva_cuda_synthetic_graph_smoke(uint32_t steps,
                                     uint32_t ring_capacity,
                                     uint32_t seed_token,
                                     NervaCudaSyntheticGraphResult *out);
int nerva_cuda_tiny_block_smoke(NervaCudaTinyBlockResult *out);
int nerva_cuda_loaded_tiny_block_smoke(NervaCudaLoadedTinyBlockResult *out);
int nerva_cuda_greedy_sampler_smoke(NervaCudaGreedySamplerResult *out);

#ifdef __cplusplus
}
#endif
