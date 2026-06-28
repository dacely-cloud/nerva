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
