# Block Projection Architecture

Date: 2026-06-28
Scope: exact BF16 projection for Qwen3-8B decode hot shapes on RTX 5090 / SM120.

## Problem

The current decode hot path is batch-1 dense projection:

```text
y = W * x
```

For an arbitrary dense matrix `W`, exact single-vector projection has to read
the active matrix. That is the bandwidth wall.

The exact escape hatch is to change the unit of projection work:

```text
Y = W * X
```

where `X` holds multiple hidden vectors. The weight matrix is streamed once and
used for several token states. This does not approximate the model and does not
change the projection math.

## Implemented Measurement Surface

`projection-bench` now accepts a sixth argument:

```text
projection-bench [rows] [cols] [dtype] [iterations] [warmup_iterations] [block_tokens]
```

When `block_tokens > 1`, the benchmark measures an exact cuBLASLt block GEMM
path in addition to the existing single-vector GEMV path. The JSON output
reports:

```text
block_tokens
block_cublaslt_avg_ns
block_cublaslt_per_token_ns
block_cublaslt_graph_avg_ns
block_cublaslt_graph_per_token_ns
block_cublaslt_speedup_x1000
block_cublaslt_graph_speedup_x1000
block_cublaslt_effective_bandwidth_bps
```

The speedup fields are scaled by 1000. For example, `7750` means `7.750x`.

## Measured Qwen Hot Shapes

Command shape:

```text
target/release/nerva-bench projection-bench rows cols 1 iterations 4 8
```

Results:

```text
name       rows    cols    gemv graph ns   block8 graph ns   block8 ns/token   speedup
qkv        6144    4096         12314             20500             2562        4.806x
gate_up   24576    4096        120932            124728            15591        7.756x
down       4096   12288         22550             69686             8710        2.588x
lm_head  151936    4096        733912            751808            93976        7.809x
```

The important result is not total operation latency. The block operation does
more math. The important result is per-token projection cost:

```text
single vector: one matrix stream per token
block8:        one matrix stream amortized over eight token states
```

## Architectural Consequence

This proves the first exact lower-level primitive needed to attack projection
bandwidth. The runtime should not try to make arbitrary `W * x` cheap. It
should create situations where the target model can run `W * X`.

The runtime integration paths are:

```text
1. Continuous batching:
   multiple active requests advance together, so each layer projects a block of
   hidden vectors.

2. Exact target verification:
   draft candidate future states and verify a block with the target model,
   committing only the accepted prefix. For greedy decode, accepted tokens must
   match target argmax exactly.

3. Structure planner:
   detect exact matrix structure at load time. Use special paths only when the
   matrix proves exact sparsity, exact duplicate rows, exact block diagonal
   structure, or exact low rank. Fall back to dense projection otherwise.
```

Continuous batching is the most production-compatible general path. Exact
target verification can help a single request, but only if the draft acceptance
rate is high enough. Structure planning is safe but should be expected to fail
for normal dense transformer matrices.

## Runtime Wiring

The CUDA session path now has a single projection dispatch interface:

```text
project_encoded_rows(
    rows,
    cols,
    tokens,
    input,
    output,
)
```

For `tokens == 1`, this preserves the existing autotuned GEMV path. For
`tokens > 1`, it routes through the cached token-GEMM plan:

```text
single-token decode: W * x
block projection:    W * X
```

Prefill and single-token decode both call this same dispatcher now. This does
not make single-request decode faster by itself, because single-request decode
still only has one exact hidden vector at each projection point. It removes the
projection-kernel split so the next runtime patch can focus on creating valid
multi-vector decode blocks.

The remaining runtime integration target is:

```text
continuous batched decode over multiple resident sessions
```

The single-request fallback remains `W * x`. The block path should only be
selected when there are multiple exact hidden vectors available for the same
projection matrix.

## Continuous Projection Batch Planner

The runtime now has a planner surface for exact continuous decode batching:

```text
crates/nerva-runtime/src/engine/hf_cuda_decode/projection_batch.rs
```

The planner does not pretend that unrelated requests are safe to merge. It only
forms a projection batch when all selected sequences prove:

```text
same resident weight hash
same dtype
same transformer shape
same vocabulary shape
ready to decode one token
not stopped
has context capacity remaining
```

That is the exactness contract for turning multiple single-token states:

```text
x0, x1, x2, ...
```

into one projection block:

```text
X = [x0 x1 x2 ...]
Y = W * X
```

The default target block is `8` because the measured Qwen3-8B projection bench
shows strong per-token wins at `block_tokens=8`. The planner caps the selected
request count to that target and returns no batch when fewer than two compatible
ready sequences exist.

The planner can be exercised from the bench CLI:

```text
cargo run -p nerva-bench -- projection-batch-plan 8 8 8 2
```

The output reports the selected request ids and the ideal projection weight
stream reuse factor. For example, `block_tokens = 8` exposes an ideal `8x`
weight-stream reuse opportunity before kernel overhead and non-projection work.

This is the next runtime seam for the native executor:

```text
1. scheduler collects ready resident sessions
2. planner selects an exact compatible projection batch
3. native executor packs per-session hidden vectors into X
4. project_encoded_rows(..., tokens = batch_size) runs W * X
5. per-session attention/KV/sample state remains isolated
```

That preserves greedy/sampling semantics for each request. The only shared work
is the dense projection read over identical weights.

## Hardware-Backed Batch Execution Probe

The bench CLI also exposes a stricter probe that combines the exact scheduler
batch plan with an actual CUDA block projection measurement:

```text
cargo run -p nerva-bench --release -- projection-batch-exec-probe \
  8 8 6144 4096 1 16 2 8 2
```

Arguments:

```text
ready_requests compatible_requests rows cols dtype iterations warmups target_block min_block
```

For the Qwen3-8B QKV shape on the RTX 5090, the measured artifact is:

```text
single graph projection     12364 ns/token
block8 graph projection     20442 ns total
block8 graph per token       2555 ns/token
per-token speedup           4.839x
mismatches                  0
hot-path allocations        0
```

The artifact is recorded at:

```text
docs/source/perf/qwen3_8b_projection_batch_exec_probe.json
```

This probe is still not a full decode executor. Its purpose is to fail unless
both halves of the next architecture step are true:

```text
1. the runtime planner proves an exact same-weight batch
2. CUDA measures the matching W * X block projection on the current GPU
```

That makes the remaining executor work concrete: collect compatible resident
sessions, pack their hidden vectors into the block input, call the existing
token projection dispatcher with `tokens = batch_size`, and scatter the output
columns back to isolated per-session state before attention, sampling, and stop
policy continue.

## Native Session Batch Contract

The native CUDA session API now exposes the metadata contract needed before a
real multi-session executor can run:

```text
nerva_cuda_hf_decode_sequence_projection_batch_plan(...)
```

This function accepts actual opaque session handles, not synthetic request ids.
It reports `ready` only when active sessions prove:

```text
same dtype
same transformer shape
same planned weight descriptor hash
same resident weight byte count
not finished
context capacity still available
at least min_block_tokens compatible sessions
```

When the contract is exact, the result also reports the staging sizes for the
real executor:

```text
pack_input_bytes
max_projection_output_bytes
qkv_input_bytes / qkv_output_bytes
attention_output_input_bytes / attention_output_output_bytes
gate_up_input_bytes / gate_up_output_bytes
down_input_bytes / down_output_bytes
lm_head_input_bytes / lm_head_output_bytes
```

This is the native handoff point from the Rust planner to CUDA execution.

The native execution hook now covers the decode projection family:

```text
nerva_cuda_hf_decode_sequence_projection_batch_execute(...)
projection_kind = QKV | W_O | GATE_UP | DOWN | LM_HEAD
```

Current behavior:

```text
1. select active compatible sessions with the same descriptor hash and shape
2. pack each session's encoded projection input into a row-major token block
3. run one `project_encoded_rows(..., tokens = block_tokens)` GEMM
4. scatter each output row block back to that session's private scratch/logits
```

The implementation uses existing per-session scratch as staging and reports
zero hot-path allocations. The focused CUDA test creates two descriptor-backed
sessions, starts both, verifies the native plan is exact, executes a block-2
projection for every supported stage, and checks each stage:

```text
pack launches        2
projection launches  1
scatter launches     2
hot allocations      0
```

This still is not full batched decode. The next step is to move these stage
calls inside the decode scheduler so compatible resident sessions advance
together through the block projection path instead of exposing it only as a
session primitive.

## Layer Projection Batch Primitive

The native API also exposes a layer-level scheduler primitive:

```text
nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(...)
```

It runs a semantically valid one-layer decode transaction over the same
compatible session block. The dense projections are shared as `W * X`, while
the dependency kernels that produce the next projection input remain isolated
per session:

```text
optional layer-0 input/RMSNorm prepare
batched QKV projection
per-session QKV publish + decode attention
batched W_O projection
per-session residual + MLP RMSNorm
batched gate/up projection
per-session SiLU gate activation
batched down projection
per-session residual finish + next/final RMSNorm
```

and returns aggregate accounting:

```text
block_tokens
qkv_rows / attention_output_rows / gate_up_rows / down_rows
input_bytes / output_bytes
qkv_elapsed_ns / attention_output_elapsed_ns / gate_up_elapsed_ns / down_elapsed_ns
pack_kernel_launches / projection_kernel_launches / scatter_kernel_launches
dependency_kernel_launches
hot_path_allocations
```

This is the scheduler-facing unit needed for continuous decode batching. The
current implementation still reuses the proven per-projection executor for each
dense stage, so it is a correctness and integration contract first. The
remaining performance work is to fuse the selection, stream synchronization,
and event timing now that the layer dataflow itself is correct.

## One-Token Batch Advance

The CUDA session API now has a token-level batch executor:

```text
nerva_cuda_hf_decode_sequence_batch_advance_one(...)
```

This is the first primitive that composes the projection-batched layer
transaction into a complete decode step for multiple active sessions:

```text
for each transformer layer:
    run layer projection batch transaction

run batched LM head projection
run per-session argmax/sample reduction
copy one completed token slot per selected session
advance each selected session cursor
```

The eligibility contract intentionally excludes the first post-prefill token
because `session_start` already computes that slot. A session becomes eligible
for this batch path after that first slot has been consumed by the normal
stateful loop. From that point, compatible resident sessions can advance one
decode token together while sharing projection weight reads through `W * X`.

The result reports:

```text
block_tokens / observed_tokens
projection_elapsed_ns
qkv_elapsed_ns / attention_output_elapsed_ns / gate_up_elapsed_ns / down_elapsed_ns / lm_head_elapsed_ns
pack_kernel_launches / projection_kernel_launches / scatter_kernel_launches
dependency_kernel_launches / sampling_kernel_launches
sync_calls / hot_path_allocations
```

The Rust stateful loop exposes this without requiring scheduler code to unwrap
raw session handles:

```text
CudaHfDecodeSequenceLoop::batch_advance_one(...)
```

Schedulers can keep owning `CudaHfDecodeSequenceLoop` values, consume the
prefill-produced first token through the normal `advance(1)` path, and then
submit compatible active loops to the batch advance wrapper.

The runtime now has the scheduler-facing bridge:

```text
crates/nerva-runtime/src/engine/hf_cuda_decode/batch_advance.rs

advance_decode_loops_once(...)
```

It attempts the CUDA one-token batch advance when at least `min_block_tokens`
loops are present. A successful exact batch returns tokens in original loop
order and marks `used_batched_projection() == true`. If the native planner
rejects the batch before touching device state (`no_ready_sessions`,
`shared_weights_unproven`, or `insufficient_compatible_ready`), the bridge
falls back to one sequential `advance(1)` per loop and keeps the rejection
summary attached for scheduler telemetry. If a batch attempt fails after the
execution path may have started, the bridge returns `BatchFailed` instead of
silently replaying the loops sequentially.

This is still not the full production queue scheduler. The remaining runtime
work is to connect the real request queue to this grouping/execution path and
prove Qwen throughput with multiple live requests.

## Shared-Weight Sessions

Continuous batching is not useful if every request duplicates the resident
model weights. The native session layer now supports shared-weight forks:

```text
nerva_cuda_hf_decode_sequence_session_fork_shared_weights(...)

CudaHfDecodeSequenceSession::fork_shared_weights(...)
```

A fork borrows the parent session's immutable device weight arena, device
layout table, packed QKV projection replicas, and packed gate/up projection
replicas through a reference-counted native shared-weight block. The fork owns
its own KV cache, prompt buffer, token slots, scratch buffers, prefill scratch,
CUDA stream, cuBLAS/cuDNN handles, CUDA events, graph cache, and projection
plan descriptors.

This changes the memory shape needed for production continuous batching:

```text
before:
  request N = weights + packed projection replicas + KV/session state

after:
  first request = weights + packed projection replicas + KV/session state
  forked request = KV/session state only
```

The cublas decode prepare path no longer writes the first hidden vector into
the shared arena scratch slot, so forked decode sessions do not corrupt each
other while sharing immutable weight storage.

The runtime also has a continuous decode batch scheduler surface:

```text
crates/nerva-runtime/src/engine/hf_cuda_decode/continuous_batch.rs

plan_continuous_projection_batch(...)
advance_continuous_decode_batch_once(...)
```

`plan_continuous_projection_batch` wraps the exact projection planner and
returns selected input indices plus fallback input indices. The executor
consumes `(ProjectionBatchCandidate, CudaHfDecodeSequenceLoop)` entries,
partitions the compatible group, calls `advance_decode_loops_once` for the
selected loops, and advances leftovers sequentially with an explicit
`not_selected_for_projection_batch` reason. If the selected batch fails after
the execution path may have started, leftovers are not advanced silently.

The bench CLI exposes a synthetic end-to-end probe for this primitive:

```text
cargo run -p nerva-bench --release -- projection-batch-advance-probe 2 2 2 2
```

The current RTX 5090 synthetic artifact is:

```text
docs/source/perf/synthetic_projection_batch_advance_probe.json
```

It verifies matching tokens through shared-weight session forks and the
continuous runtime scheduler, five batched projection launches for a one-layer
model (`QKV`, `W_O`, `gate/up`, `down`, `LM head`), zero hot-path allocations,
and a small wall-clock win over two sequential one-token loop advances on the
tiny synthetic model. This is not a Qwen throughput claim; it is the first
hardware-backed proof that the full batched decode-step composition is callable
through scheduler-facing runtime code and preserves token output for compatible
sessions without duplicating resident weights.

## File-Backed Shared-Fork Probe

The bench CLI now also exposes a real checkpoint probe:

```text
cargo run -p nerva-bench --release -- hf-cuda-shared-fork-batch \
  <checkpoint_dir> [request_count] [max_context] [max_new_tokens] \
  [target_block_tokens] [min_block_tokens] prompt_text|@prompt.txt \
  [compute_capability]
```

This path loads the checkpoint once, creates `request_count - 1` CUDA session
forks that share the parent's resident weights, starts each request from the
same prompt, drains the first prefill-produced token, then runs subsequent
tokens through `advance_continuous_decode_batch_once`.

The current tiny Qwen3-8B proof run is:

```text
docs/source/perf/qwen3_8b_shared_fork_batch_probe.json

requests             2
max context          1024
max new tokens       4/request
aggregate decode     188.03 tok/s
batched steps        3/3
fallback steps       0
hot-path allocations 0
tokens/request       [11, 358, 2776, 4460]
```

This proves the real file-backed Qwen path can execute through shared-weight
forks and projection-batched decode steps. It is deliberately a tiny functional
probe, not a final throughput benchmark. The next benchmark needs longer
multi-request decode budgets and a comparison against sequential same-session
decode under the same prompt, context, and request count.

There is also a comparison command:

```text
cargo run -p nerva-bench --release -- hf-cuda-shared-fork-batch-compare \
  <checkpoint_dir> [request_count] [max_context] [max_new_tokens] \
  [target_block_tokens] [min_block_tokens] prompt_text|@prompt.txt \
  [compute_capability]
```

It runs the same prompt/request count twice: once with `min_block_tokens`
forced above the request count so the continuous scheduler takes the sequential
fallback path, then once with the requested batch settings.

The current tiny Qwen3-8B comparison artifact is:

```text
docs/source/perf/qwen3_8b_shared_fork_batch_compare.json

sequential fallback  130.58 tok/s
batched projection   187.32 tok/s
decode speedup       1.43x
token match          true
```

This is the first real checkpoint evidence that shared-weight continuous decode
batching can lower projection weight-stream pressure for multiple live requests
without changing tokens. The run is intentionally short; longer prompts,
larger request groups, and production queue integration are still required
before claiming final serving performance.
