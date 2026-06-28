# cuDNN Decode SDPA Notes

Date: 2026-06-28
Scope: Qwen3-8B BF16, batch-1 decode, RTX 5090 / SM120.

## Current Finding

cuDNN decode SDPA is faster than NERVA's native fallback for long-context
batch-1 attention.

Measured on the same 3965-token prompt and 512-token decode budget:

```text
cuDNN decode on:
  critical path      97.10 tok/s
  post-load          90.34 tok/s
  decode wall        5272.98 ms
  attention profile   548.60 ms
  projection profile 5246.84 ms

cuDNN decode off:
  critical path      87.93 tok/s
  post-load          82.13 tok/s
  decode wall        5822.96 ms
  attention profile  1016.73 ms
  projection profile 5259.36 ms
```

The output tokens were unchanged. Projection was effectively unchanged. The
gain came from attention: cuDNN reduced the profiled attention time by about
468 ms over 511 graph replays.

Short answer:

```text
cuDNN is faster because it turns decode attention into one backend-scheduled
SDPA operation with runtime sequence lengths, GQA-aware head mapping, and no
NERVA-visible partial-scratch/reduce stage.
```

It is not faster because it changes the attention math. It is the same exact
softmax(QK^T / sqrt(d))V result, but with a better execution contract.

## Why cuDNN Wins Here

The native fallback does chunked paged attention as two explicit stages:

```text
1. per-chunk attention kernel
   writes partial values, partial max, partial sum-exp

2. per-head reduce kernel
   rereads partial state and produces the final attention output
```

That structure is correct, but it pays extra global-memory traffic:

```text
partial_values[head, chunk, dim]
partial_m[head, chunk]
partial_l[head, chunk]
```

For a long context, every token and every layer writes those partials and then
reads them again. It also adds another graph node for the reduce stage.

cuDNN builds the decode operation as SDPA over Q, K, V with runtime sequence
lengths:

```text
Q: {batch, heads, 1, head_dim}
K: {batch, kv_heads, kv_tokens, head_dim}
V: {batch, kv_heads, kv_tokens, head_dim}
O: {batch, heads, 1, head_dim}
seq_len_q:  1
seq_len_kv: current context length
```

The cuDNN frontend validates GQA directly when query heads are a multiple of
KV heads. For Qwen3-8B this is:

```text
heads    32
kv_heads  8
group      4
head_dim 128
```

The important pattern is not just "use cuDNN". It is:

```text
single fused SDPA operation
runtime seq-len masking
GQA-aware head mapping
backend-selected tiling
no explicit partial-value global scratch/reduce stage in NERVA code
small reusable workspace
graph-capturable execution
```

NVIDIA's cuDNN frontend documentation describes this exact surface for current
SDPA: FP16/BF16 SDPA uses a FlashAttention-2 style implementation, supports
decode, supports MHA/MQA/GQA, accepts runtime `seq_len_q` and `seq_len_kv`
tensors for padded variable-length inputs, and exposes paged-attention K/V
table attributes. That combination is why cuDNN can keep the captured op shape
stable while making the live work depend on device-side sequence length data.

In the current NERVA code, that maps to two very different execution shapes.
The native decode path launches a chunk kernel over `{head or kv_head, chunk}`,
writes `partial_values`, `partial_m`, and `partial_l`, then launches a reduce
kernel over heads. The cuDNN path builds one SDPA graph op over:

```text
Q: {1, heads, 1, head_dim}
K: {1, kv_heads, kv_token_capacity, head_dim}
V: {1, kv_heads, kv_token_capacity, head_dim}
O: {1, heads, 1, head_dim}
```

and passes the live lengths as device tensors:

```text
seq_len_q  = 1
seq_len_kv = current decoded context length
```

That is why cuDNN can keep a stable captured graph while the visible context
length grows. The graph shape is capacity-sized, but the actual work is masked
by runtime sequence length data.

cuDNN's frontend also rewrites K for backend SDPA requirements internally. The
frontend API accepts K as:

```text
{batch, kv_heads, seq_kv, head_dim}
```

and maps the backend matmul view to:

```text
{batch, kv_heads, head_dim, seq_kv}
```

That is the same layout lesson vLLM's paged attention kernels follow in spirit:
make K/V reads coalesced and vectorized for the attention tile rather than
treating each token/head as scalar work.

NVIDIA's current cuDNN frontend SDPA documentation describes the same surface:
SDPA is implemented with the FlashAttention-2 algorithm, supports MHA/MQA/GQA,
supports sequence length tensors for variable/padded inputs, and exposes paged
attention K/V page-table options. NERVA currently uses the dense K/V tensor
entry point for decode. That is acceptable while the block table is identity,
but the better long-term design is to make cuDNN consume NERVA's KV block table
directly through its paged-attention inputs.

## vLLM Cross-Check

The same design shows up in vLLM, even when vLLM is not literally routing the
ordinary text decode path through cuDNN.

vLLM's FlashAttention backend builds attention metadata around:

```text
seq_lens
query_start_loc
block_table
slot_mapping
optional scheduler_metadata
max_num_splits
```

That is the serving-side version of the cuDNN lesson: the kernel receives a
packed description of the active batch and the paged KV layout, rather than
making shape changes or host control decisions inside the hot token loop.

The vLLM paged decode kernels also show the lower-level CUDA pattern:

```text
Q is loaded once into registers/shared memory.
K/V are read through a block table.
K/V movement is vectorized around 16-byte chunks.
The kernel performs online softmax state updates.
Long contexts split KV into independent partitions and merge partial states.
GQA maps multiple query heads to one KV head inside the kernel.
```

So the portable rule is:

```text
Attention metadata is a device-side schedule.
KV cache is paged memory.
Decode attention is an online reduction over pages.
```

NERVA already has the start of this: 16-token KV blocks, a block table, and a
cuDNN decode SDPA path. The missing part is making the block table a first-class
backend ABI across cuDNN/native attention instead of treating the cuDNN path as
a dense identity-table shortcut.

## Exact CUDA Pattern Lessons

The useful patterns to copy are concrete:

```text
1. Keep graph shape fixed and pass live lengths as device tensors.
   cuDNN decode builds capacity-shaped K/V descriptors and passes
   seq_len_q/seq_len_kv to mask actual work. Native kernels should do the same
   instead of selecting graph shape from current context length.

2. Use paged KV as the attention ABI.
   The logical block table must be passed to every attention backend. Dense
   physical K/V can be a special case where the table is identity.

3. Move K/V in vector units.
   vLLM chooses vector types so each thread group moves 16 bytes at a time.
   NERVA's fallback still has scalar-looking BF16-to-FP32 decode loops in the
   generic paths. The native fast path should use vector loads and convert in
   registers.

4. Specialize the hot Qwen shape.
   For Qwen3-8B BF16 on this GPU, the stable decode shape is:
     heads = 32
     kv_heads = 8
     group = 4
     head_dim = 128
   The native fallback should have a dedicated kernel for this shape, not only
   a generic paged attention kernel.

5. Avoid visible partial scratch unless the context requires split-KV.
   For medium contexts, prefer one online-softmax pass that writes final O.
   For long contexts, split KV, but keep partial state compact and merge with
   the same online-softmax algebra.

6. Cache the attention policy per GPU/model shape.
   The runtime should autotune once between:
     cuDNN SDPA
     native one-pass GQA paged attention
     native split-KV GQA paged attention
   Then cache the winner by SM, dtype, heads, kv_heads, head_dim, page size,
   and context range.
```

The pattern is not "copy cuDNN internals". The pattern is to give the backend a
clean enough problem that it can select the right kernel: fixed descriptors,
runtime lengths, page tables, GQA shape, no extra scratch outputs, and no
per-token host decision.

## What To Copy Into Native Kernels

The native path should move toward these patterns:

```text
1. Fuse chunk scan and final reduction when possible.
   Avoid writing per-chunk partial vectors to global memory unless the context
   is too large for a single pass.

2. Use vectorized K/V loads.
   vLLM's paged attention uses vector types so each thread group moves 16-byte
   chunks. The current NERVA native fallback converts scalar BF16 values to
   float from shared/global memory more often.

3. Keep the 16-token KV page size for this model.
   It matches the current NERVA page table and vLLM's common block-size path.

4. Specialize for Qwen GQA:
   heads / kv_heads = 4 and head_dim = 128 are fixed for this target. A kernel
   specialized to that shape can remove generic branches and use predictable
   thread-group mapping.

5. Keep runtime sequence length as data, not as a graph shape.
   cuDNN accepts seq_len_q and seq_len_kv tensors. This is why one captured
   graph can replay across decode steps without recapturing for every context
   length.

6. Prefer one graph node for decode attention.
   Splitting attention into chunk kernels plus reduce increases global traffic
   and graph-node count.

7. Make paged KV a backend contract, not a native-only detail.
   NERVA already stores KV in 16-token blocks and initializes an identity block
   table. cuDNN now has paged-attention K/V table inputs, so the cuDNN path
   should use the same logical block table instead of assuming dense physical
   order forever.

8. Keep statistics optional.
   cuDNN decode sets generate_stats=false. Native decode should avoid producing
   log-sum-exp or max scratch unless a later operation actually needs it.

9. Autotune at the attention-operation level.
   cuDNN chooses a backend implementation through frontend heuristics. The
   native path should benchmark chunk size, head-thread mapping, shared-K/V
   staging, and one-pass versus split-reduce variants per GPU/model shape, then
   cache that choice.
```

## Implementation Checklist

The next code changes should be ordered like this:

```text
1. Promote paged cuDNN decode.
   Replace the dense decode K/V descriptors with cuDNN paged-attention K/V
   table descriptors when the installed cuDNN frontend supports them. Keep the
   dense identity-table path as a fallback only.

2. Add a native Qwen3 GQA decode-attention microbench.
   Benchmark one-pass and split-KV kernels independently of the full model:
     heads=32, kv_heads=8, head_dim=128, page=16, BF16
   Sweep context lengths and block/thread layouts.

3. Add a native attention policy cache.
   Store the winning kernel choice in the session after warmup/autotune. The
   decode loop should only consume the cached policy.

4. Make split-KV threshold data-driven.
   The current chunk threshold is static. It should be selected by the
   attention microbench, because the best split point varies by GPU.

5. Keep profiling separated from the hot graph.
   Detailed profiling should measure attention variants during warmup or
   explicit `--profiling`, not add per-token syncs to normal decode.
```

This does not solve projection. It removes attention waste and gives NERVA the
same attention-shape contract that cuDNN and vLLM optimize around.

## Source Pointers

NERVA:

```text
native/cuda/nerva_cuda_hf_decode_sequence.cu
  hf_layer_*attention*_chunk_kernel
  hf_layer_attention_reduce_kernel
  ensure_cudnn_decode_sdpa_plan
  execute_cudnn_decode_sdpa
  launch_cublas_layer_session_step
```

vLLM:

```text
/root/vllm/vllm/v1/attention/backends/flash_attn.py
  FlashAttentionMetadata
  FlashAttentionMetadataBuilder

/root/vllm/vllm/v1/attention/ops/triton_decode_attention.py
  _fwd_kernel_stage1
  _fwd_grouped_kernel_stage1

/root/vllm/csrc/libtorch_stable/attention/paged_attention_v1.cu
  paged_attention_v1_launcher

/root/vllm/csrc/libtorch_stable/attention/attention_kernels.cuh
  paged_attention_kernel
```

External references:

```text
NVIDIA cuDNN frontend Attention documentation
https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html

NVIDIA cuDNN frontend graph API
https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html
```

## Current NERVA Controls

Default behavior keeps cuDNN decode enabled when cuDNN frontend is available
and the shape is supported.

For A/B tests:

```text
NERVA_CUDNN_DECODE=0
```

forces the native fallback.

For diagnosing plan build/capture issues:

```text
NERVA_CUDNN_DECODE_DEBUG=1
```

prints cuDNN decode gate/build/capture failures to stderr. Normal CLI output is
unchanged unless this variable is set.

## Current Bottleneck After cuDNN

cuDNN attention helps, but decode remains projection-bound:

```text
projection profile 5246.84 ms
attention profile   548.60 ms
```

So a perfect attention kernel cannot solve the whole decode wall. It can still
matter: the current cuDNN path is about 10 percent faster than the native
fallback on the measured long-context run.

The next native performance lesson is to treat attention like cuDNN does:
fused, vectorized, shape-specialized, and graph-stable. The next major wall is
still dense batch-1 projection.
