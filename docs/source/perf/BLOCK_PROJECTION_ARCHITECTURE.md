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

## Next Runtime Patch

The next patch should introduce a projection execution interface that can carry
`tokens > 1` through decode:

```text
input:  cols * block_tokens
output: rows * block_tokens
```

Then integrate it first where correctness is simplest:

```text
continuous batched decode over multiple resident sessions
```

The single-request fallback remains `W * x`. The block path should only be
selected when there are multiple exact hidden vectors available for the same
projection matrix.
