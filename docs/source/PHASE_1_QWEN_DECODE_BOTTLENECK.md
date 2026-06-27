# Phase 1 - Qwen Decode Bottleneck Source Analysis

Date: 2026-06-27
Scope: Qwen3-8B, BF16, batch 1, fully GPU-resident decode on RTX 5090.

## Source Map

The hot runtime source is:

```text
native/cuda/nerva_cuda_hf_decode_sequence.cu
```

The decode graph capture/replay path is built from:

```text
launch_cublas_layer_session_step
profile_cublas_layer_session_step
encoded_row_major_gemv_lt
hf_layer_qkv_attention_encode_kernel
hf_layer_mlp_norm_encode_kernel
hf_layer_ff_encode_kernel
hf_decode_final_head_reduce_kernel
```

The current batch-1 projection primitive is:

```text
encoded_row_major_gemv_lt
```

It wraps cuBLASLt matmul for a row-major matrix-vector operation:

```text
output = matrix[rows x cols] * input[cols x 1]
```

This primitive is used for:

```text
QKV projection
attention output projection
MLP gate/up projection
MLP down projection
lm_head projection
```

The source-level loop is:

```text
for each layer:
    RMSNorm + encode projection input
    qkv GEMV
    attention + KV append
    attention output GEMV
    residual + RMSNorm + encode projection input
    gate/up GEMV
    SiLU/gate activation
    down GEMV
    residual + next RMSNorm or final RMSNorm

final:
    lm_head GEMV
    argmax/sample
```

## Measured Baseline

Baseline artifact:

```text
docs/source/perf/qwen3_8b_nerva_cuda_generate.json
```

The replay critical path is about:

```text
10.25 ms/token
97.56 tokens/s
```

The profile-pass accounting says projection dominates:

```text
profile_accounted_ns_per_token:     11.96 ms
profile_projection_ns_per_token:    10.43 ms
profile_attention_ns_per_token:      0.29 ms
profile_mlp_ns_per_token:            0.14 ms
profile_norm_ns_per_token:           1.00 ms
profile_sampling_ns_per_token:       0.10 ms
```

Projection is roughly 87 percent of accounted token time.

The dominant projection buckets are:

```text
gate_up_projection_ns_per_token: ~4.58 ms
down_projection_ns_per_token:    ~2.58 ms
lm_head_projection_ns_per_token: ~0.79 ms
```

This means the current single-GPU bottleneck is not PCIe, disk, token copy,
host output sync, graph replay overhead, attention, or sampling.

The bottleneck is resident dense batch-1 projection over large weight matrices.

## First Owned GEMV Candidate

A source-level candidate was tested that grouped multiple output rows per CUDA
block and reused the input vector from shared memory across those rows.

The intended fix was to improve the old simple custom GEMV pattern:

```text
one output row per block
same input vector reread for every row
one reduction per output row
```

The candidate preserved token identity:

```text
baseline tokens:  [50994, 67]
candidate tokens: [50994, 67]
```

But it lost the replay critical path:

```text
baseline:  ~10.25 ms/token
candidate: ~18.70 ms/token
```

Candidate profile buckets:

```text
gate_up_projection_ns_per_token: 6.98 ms
down_projection_ns_per_token:    7.61 ms
lm_head_projection_ns_per_token: 1.07 ms
```

Conclusion:

```text
The naive owned GEMV candidate is correct but slower.
It must not be selected in the runtime.
```

## Why The Candidate Lost

The candidate reduced repeated input reads, but it still used scalar per-thread
dot products and shared-memory reductions. For these Qwen shapes, cuBLASLt's
internal GEMV/matmul path is much better at scheduling memory traffic and
reduction work.

The failed candidate proves the problem is not just:

```text
input vector reuse
```

The real projection redesign has to attack:

```text
weight-read bandwidth
row/column layout for batch-1 GEMV
reduction shape
MLP intermediate writes and rereads
gate/up activation/down dataflow
kernel launch count and graph node count
executor choice when weights are not VRAM-resident
```

## Next Performance Work

The next source-level target is not another one-row GEMV variant.

The next candidates should be:

```text
1. cuBLASLt descriptor/algo reuse for fixed Qwen projection shapes.
2. MLP dataflow redesign around gate/up, activation, and down projection.
3. Packed weight layouts that are explicitly optimized for batch-1 decode.
4. CPU/DRAM compute-near-data for nonresident weight shards.
5. Hybrid partial output merge when only part of W is hot in VRAM.
```

For the fully resident Qwen3-8B run, CPU/DRAM placement is not expected to beat
VRAM because all weights fit and the bottleneck is local dense projection.

For the larger NERVA architecture, the same source-level operation must become
a planner decision:

```text
GPU resident W x
GPU staged W x
CPU DRAM W x
CPU/GPU split W x with partial-output merge
```

The runtime should select the path that wins visible token critical-path time,
not the path that looks fastest by ideology.
