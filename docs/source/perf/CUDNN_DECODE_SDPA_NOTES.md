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
