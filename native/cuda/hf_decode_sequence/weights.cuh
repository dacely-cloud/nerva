#pragma once

#include "types.cuh"

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

uint64_t push(uint64_t &cursor, uint64_t len);
uint64_t push_optional(uint64_t &cursor, uint64_t len, const uint16_t *ptr);
uint64_t hash_tokens(const uint32_t *tokens, uint32_t count);

bool descriptor_has_memory_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor);
bool descriptor_has_file_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor);
bool descriptor_has_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor);

bool has_declared_weight_plan(const NervaCudaHfDecodeSequenceRequest *request);
bool has_declared_weight_plan(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request);
uint64_t pinned_weight_staging_bytes(
    const NervaCudaHfDecodeSequenceRequest *request,
    uint64_t full_weight_bytes);
uint64_t pinned_weight_staging_bytes(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    uint64_t full_weight_bytes);

bool valid_layer(const NervaCudaHfDecodeChainLayer &layer, bool require_sources);
bool valid_request(const NervaCudaHfDecodeSequenceRequest *request);
void clear_result(const NervaCudaHfDecodeSequenceRequest *request,
                  NervaCudaHfDecodeSequenceResult *out);
void clear_session_create_result(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out);

bool validate_weight_descriptors(const NervaCudaHfDecodeSequenceRequest *request,
                                 uint64_t resident_weight_bytes,
                                 NervaCudaHfDecodeSequenceResult *out);
bool validate_weight_descriptors(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    uint64_t resident_weight_bytes,
    NervaCudaHfDecodeSequenceSessionCreateResult *out);

bool should_pack_cublas_weights(uint32_t hidden, uint32_t attention_hidden);
PackedProjectionShape packed_projection_shape(uint64_t hidden,
                                              uint64_t attention_hidden,
                                              uint64_t kv_hidden,
                                              uint64_t intermediate);
void pack_layer(SequenceLayerLayout &layout, uint64_t &cursor,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden,
                uint64_t head_dim, uint64_t intermediate);
void copy_layer(uint16_t *arena, const SequenceLayerLayout &layout,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden,
                uint64_t head_dim, uint64_t intermediate);
void assign_linear_gdn_state_offsets(std::vector<SequenceLayerLayout> &layouts,
                                     uint64_t *conv_state_elements,
                                     uint64_t *recurrent_state_elements);

cudaError_t copy_weight_descriptors_to_device(
    uint16_t *device_arena, uint16_t *staging, uint64_t staging_bytes,
    const NervaCudaHfDecodeSequenceRequest *request, uint64_t arena_bytes,
    uint64_t embedding_bytes, uint64_t scratch_gap_bytes, cudaStream_t stream,
    NervaCudaHfDecodeSequenceResult *out, uint64_t *setup_sync_calls);
cudaError_t copy_weight_descriptors_to_device(
    uint16_t *device_arena, uint16_t *staging, uint64_t staging_bytes,
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    uint64_t arena_bytes, uint64_t embedding_bytes, uint64_t scratch_gap_bytes,
    cudaStream_t stream, NervaCudaHfDecodeSequenceSessionCreateResult *out,
    uint64_t *setup_sync_calls);
