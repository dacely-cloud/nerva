# Phase 2 - Single-GPU Latency Ledger

Date: 2026-06-26
Active scope: one CPU, one DRAM pool, one disk/NVMe, one GPU, one VRAM pool,
one LLM inference runtime.

This supersedes the earlier Phase 2 networking/RDMA direction. Networking,
multi-GPU, RDMA, DPDK, ConnectX behavior, and distributed scheduling are out of
scope until the single-GPU runtime is understood.

## Thesis

Current inference engines usually behave as if:

```text
GPU is the main machine.
CPU is the feeder.
VRAM is where the model should live.
DRAM is fallback.
Disk is storage.
```

The redesign starts from a different model:

```text
VRAM is not the model.
VRAM is the hot working set.
The model is not loaded.
The model is scheduled.
```

Token latency is a runtime property:

```text
token latency =
    memory residency
  + kernel launches
  + synchronization
  + HBM traffic
  + DRAM traffic
  + PCIe movement
  + CPU scheduling
  + KV-cache layout
  + sampling
  + allocation/page faults
  + actual math
```

The actual math is only one component. The ledger must expose the rest.

## Non-Negotiable Runtime Rules

```text
Every byte must justify its trip.
Every kernel launch must justify its existence.
Every synchronization must justify its stall.
Every tensor must have a residency reason.
```

During decode:

```text
no malloc
no free
no cudaMalloc
no cudaFree
no mmap
no page fault
no memory registration
no dynamic tensor allocation
no surprise synchronization
```

## Runtime Responsibilities

CPU owns the latency control plane:

```text
tokenization and detokenization
request state
decode loop ownership
scheduler decisions
memory residency planning
KV metadata and block/page tables
prefix cache metadata
prefetch and eviction decisions
pinned buffer management
telemetry and token ledger
sampling when cheaper than GPU sampling
small reductions and branch-heavy work
CPU compute for host-resident shards when profitable
```

GPU owns the hot throughput data plane:

```text
large dense matrix operations
prefill
resident decode projections
resident MLP blocks
hot attention over VRAM KV
FlashAttention/Flash-Decoding-style tiled work
large reductions
fused kernels
coalesced memory scans
```

VRAM is a managed hot cache:

```text
hot model weights
currently active layer/tile
hot KV pages
current activations
GPU workspaces
persistent graph buffers
prefetch slots
```

DRAM is the planned warm tier:

```text
full model backing store when model > VRAM
warm weights
CPU-computable weight shards
warm/cold KV before disk
prefix cache
tokenizer data
scheduler metadata
page tables
pinned staging pools
```

Disk is cold storage only. Disk is allowed before decode. Disk is not allowed
during decode wait.

## Core Components To Build

```text
static memory arenas
residency planner
persistent decode executor
fused kernel layer
tiered KV cache
CPU/GPU operation selector
prefetch engine
token latency ledger
```

The first implementation is not a full LLM engine. It is the measurement and
execution skeleton that proves which decisions are real.

## Phase 2 Build Order

### Step 1 - Baseline vLLM On One GPU

Measure one existing engine:

```text
time to first token
time per output token
kernel launches/token
GPU active/idle time
CPU blocked time
VRAM usage
PCIe bytes
page faults
malloc/cudaMalloc calls
sampling time
sync count
```

This establishes current reality.

### Step 2 - Synthetic Transformer Block

Build a fake transformer block with:

```text
RMSNorm
QKV projection
attention-like memory scan
MLP projection
residual
KV append
```

This tests runtime design without model complexity.

### Step 3 - Static Arenas

Implement:

```text
CPU arena
pinned arena
GPU arena
KV page pool
workspace allocator
```

Target: zero allocation during decode.

### Step 4 - Operation Placement Benchmark

Test the core question:

```text
GPU resident W x
DRAM W copied to GPU then W x
DRAM W computed on CPU
DRAM W prefetched ahead then W x on GPU
```

For each operation, decide whether it is cheaper to move the data to the GPU or
compute where the data already is.

### Step 5 - Persistent Decode Graph

Capture/replay or prebuild as much decode work as possible.

Measure:

```text
launch count reduction
GPU idle gap reduction
latency reduction
sync count reduction
```

### Step 6 - Tiered KV Prototype

Implement:

```text
VRAM KV pages
DRAM KV pages
exact blockwise attention partials
online softmax merge
```

This must remain exact except for normal floating-point ordering differences.

### Step 7 - Minimal Real Model

Only after synthetic results are clean, run a real small model through the new
execution skeleton. Then scale.

## Token Ledger Schema

Every generated token should eventually produce:

```text
token_id
total_latency_us

CPU:
    scheduler_us
    sampling_us
    blocked_us
    page_faults
    malloc_calls

GPU:
    active_us
    idle_us
    kernel_launches
    sync_count
    hbm_read_mb
    hbm_write_mb

PCIe:
    h2d_bytes
    d2h_bytes
    visible_us

DRAM:
    read_mb
    write_mb

KV:
    vram_pages
    dram_pages
    moved_pages

Runtime:
    prefetched_blocks
    evicted_blocks
    graph_replays
```

Without this ledger, runtime design is guessing.

## Workspace

The active single-GPU workspace is:

```text
single-gpu-ledger workspace
```

The prior `latency-ledger workspace` RDMA/network workspace is historical and
not part of the active Phase 2 scope.

