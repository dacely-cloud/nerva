<div align="center">

<img src="./images/logo.svg" alt="NERVA" width="380" />

<h3>AI inference beyond the VRAM wall</h3>

<h4>An inference operating system for large models</h4>

<p>
<sub>
Memory residency, device-first token state, heterogeneous CPU/GPU execution, and token-level observability, all built into the runtime itself rather than bolted on afterward.
</sub>
</p>

<p>
<img src="https://img.shields.io/badge/stage-runtime_foundation-2563ff?labelColor=0e1520" alt="stage: runtime foundation" />
<img src="https://img.shields.io/badge/runtime-Rust-b7410e?labelColor=0e1520" alt="runtime: Rust" />
<img src="https://img.shields.io/badge/backend-CUDA_12.x_%2F_13.x-76b900?labelColor=0e1520" alt="backend: CUDA 12.x / 13.x" />
<img src="https://img.shields.io/badge/platform-Linux_x86__64_%2F_aarch64-4b5563?labelColor=0e1520" alt="platform: Linux x86_64 / aarch64" />
<img src="https://img.shields.io/badge/profile-single--GPU_%28NERVA--SG%29-7c3aed?labelColor=0e1520" alt="profile: single-GPU (NERVA-SG)" />
<img src="https://img.shields.io/badge/math-exact-22e3ab?labelColor=0e1520" alt="math: exact" />
</p>

</div>

---

## Table of contents

- [Overview](#overview)
  - [What NERVA is](#what-nerva-is)
  - [The thesis](#the-thesis)
  - [Measured Qwen3-8B performance](#measured-qwen3-8b-performance)
- [The architecture](#the-architecture)
  - [Why inference needs a new machine](#why-inference-needs-a-new-machine)
  - [What changes](#what-changes)
  - [ResidentBlocks](#residentblocks)
  - [Memory residency](#memory-residency)
  - [CPU and GPU roles](#cpu-and-gpu-roles)
  - [Device-first decoding](#device-first-decoding)
  - [KV cache as virtual memory](#kv-cache-as-virtual-memory)
  - [Static arenas and synchronization](#static-arenas-and-synchronization)
  - [Token ledgers](#token-ledgers)
- [Hardware and the road ahead](#hardware-and-the-road-ahead)
  - [Hardware model](#hardware-model)
  - [Coherent shared memory](#coherent-shared-memory)
  - [Future transport and distributed inference](#future-transport-and-distributed-inference)
    - [Move activations, not weights](#move-activations-not-weights)
    - [The VRAM-to-NIC problem](#the-vram-to-nic-problem)
    - [Choosing a transport path](#choosing-a-transport-path)
    - [Topology-aware egress](#topology-aware-egress)
    - [Backends, fabrics, and the engineering boundary](#backends-fabrics-and-the-engineering-boundary)
- [Positioning](#positioning)
  - [Relationship to vLLM and rvLLM](#relationship-to-vllm-and-rvllm)
  - [What NERVA is not](#what-nerva-is-not)
- [Experimental work](#experimental-work)
  - [RT-core candidate selection](#rt-core-candidate-selection)
  - [MoE support](#moe-support)
  - [Long-context residency](#long-context-residency)
- [Development state](#development-state)
  - [Current implementation status](#current-implementation-status)
  - [Long-term goal](#long-term-goal)
- [Implementation and running it](#implementation-and-running-it)
  - [Current implementation](#current-implementation)
  - [Requirements](#requirements)
  - [Server mode](#server-mode)
  - [Running the checks](#running-the-checks)

---

## Overview

### What NERVA is

NERVA means **Neural Execution & Residency Virtual Architecture**. It is an inference operating system for AI models, a Rust-first runtime with a CUDA-first device backend, built to rebuild LLM inference around memory residency rather than treating the GPU as one monolithic place where the whole model has to fit.

The Transformer math stays exact. NERVA does not start by changing the architecture, quantizing weights, pruning layers, dropping context, approximating attention, or swapping in a smaller target model. The model stays the model, and what changes is the execution machine around it.

So instead of treating inference as a sequence of framework calls that launch GPU kernels, NERVA treats it as a live scheduling problem that spans memory tiers, compute devices, synchronization phases, and token-causality state. Weights, KV cache, activations, tokens, sampler state, temporary workspaces, and future transport buffers all become explicit runtime objects, and each one carries its own location, ownership, lifetime, and next-use semantics. The point is simple but deep, which is that **the model is not loaded, the model is scheduled.**

### The thesis

Current inference engines usually start from a GPU-first assumption. They put the weights in VRAM, put the KV cache in VRAM, let the CPU feed the GPU, and make the CPU observe every token before the next step can proceed. That works when the model fits comfortably, the KV cache fits comfortably, batching hides overhead, and the workload is shaped for the hardware. It breaks down the moment the model is larger than VRAM, the context is long, the batch is small, the hardware is old, memory movement is hidden behind framework abstractions, or token latency matters more than aggregate throughput.

NERVA starts from a different assumption. The runtime should know where every meaningful block of data lives, who owns it, when it is needed next, whether it is hot or cold, whether it is cheaper to move it or compute beside it, and whether a synchronization is truly required for correctness. From there, inference gets scheduled around the critical path rather than blindly executed as a GPU command loop.

### Measured Qwen3-8B performance

The current real-model baseline is Qwen3-8B BF16 on an RTX 5090. The checked-in artifacts compare NERVA's resident CUDA decode path against a vLLM short-decode latency run on the same local safetensors snapshot. This is a narrow batch-one decode comparison, not a production serving claim.

| Engine | Artifact | Output tokens | Throughput | p99/token | Notes |
|---|---|---:|---:|---:|---|
| NERVA | `docs/source/perf/qwen3_8b_nerva_cuda_generate.json` | 2 | 97.56 tok/s | 10.30 ms | CUDA graph replay, 291 graph nodes/token, `hot_path_allocations: 0` |
| vLLM | `docs/source/perf/qwen3_8b_vllm_latency.json` | 2 | 89.33 tok/s | 11.66 ms | `vllm bench latency`, request avg 22.39 ms |

The token-parity artifact also checks that the short greedy decode emits the same tokens as vLLM:

| Artifact | Matched | Mismatched | Missing | Extra |
|---|---:|---:|---:|---:|
| `docs/source/parity/qwen3_8b_token_parity.json` | 2 | 0 | 0 | 0 |

The Qwen3-8B profile currently says the batch-one bottleneck is projection, not sampling or host traffic:

| NERVA profile bucket | ns/token |
|---|---:|
| QKV / output / MLP / LM-head projection | 10,426,479 |
| Norm kernels | 996,512 |
| Attention | 292,699 |
| MLP elementwise | 141,472 |
| Sampling | 104,704 |

The other checked-in Qwen3-8B result is about amortizing projection work across compatible requests. With 32 short requests, target block size 32, and token identity preserved, the shared-fork batch comparison reaches 1184.47 tok/s versus 127.76 tok/s for the sequential path, a 9.27x decode-wall speedup in that probe.

| Artifact | Sequential | Batched | Speedup | Token match |
|---|---:|---:|---:|---|
| `docs/source/perf/qwen3_8b_shared_fork_batch_compare_32req_target32.json` | 127.76 tok/s | 1184.47 tok/s | 9.27x | true |

The block projection microbench explains why batching matters. At `block_tokens = 8`, exact BF16 block projection improves graph replay per-token cost by 2.59x to 7.81x on the measured Qwen3-8B hot shapes:

| Shape | Per-token graph speedup |
|---|---:|
| QKV | 4.81x |
| Gate/up | 7.76x |
| Down | 2.59x |
| LM head | 7.81x |

The practical read is simple: NERVA's current Qwen3-8B path works, matches the vLLM short-token artifact, and is already competitive on a narrow latency check. The next performance work is to make the projection batching path usable for real continuous batching and to prove whether sparse long-context attention can reduce full decode latency without unacceptable quality loss.

---

## The architecture

At the top level, NERVA replaces the GPU-centric command loop with an inference virtual machine, where a CPU control plane drives policy and scheduling, a memory operating system owns where every block lives, a GPU hot plane executes a prebuilt decode transaction, and a transport layer moves named block versions between domains. The device token ring is what closes the decode loop, and the host observes it asynchronously rather than gating every step.

```mermaid
flowchart TB
    REQ["Request and prompt"] --> CPU
    subgraph CONTROL["CPU, latency and control plane"]
        CPU["Request and token state machine"]
        SCHED["Execution planner and scheduler"]
        HILO["HILO residency planner"]
        LEDGER["Token and stall ledger"]
    end
    subgraph MEMOS["Memory operating system"]
        BT["Global block table of ResidentBlocks"]
        KVM["KV virtual-memory manager"]
        ARENA["Static arena manager"]
        PFE["Prefetch and eviction engine"]
    end
    subgraph HOT["GPU, hot tensor plane"]
        EXEC["Prebuilt decode graph or transaction"]
        RING["Device token ring"]
    end
    TM["Transport manager, RDMA, DPDK, pinned-host"]
    CPU --> SCHED --> HILO --> BT
    BT --> KVM
    BT --> ARENA
    HILO --> PFE
    SCHED --> EXEC
    EXEC --> RING
    RING -->|next step consumes slot| EXEC
    RING -.async observe.-> CPU
    EXEC --> LEDGER
    PFE <--> TM
    BT <--> TM
```

### Why inference needs a new machine

LLM inference is not one workload. It contains dense matrix math, attention over a growing KV cache, token sampling, request scheduling, memory allocation, device synchronization, CPU-visible output, host-to-device transfer, cache management, and sometimes disk or network staging. Treating all of that as "GPU work" hides the actual problem.

A GPU is excellent at hot, parallel, throughput-heavy tensor work, and a CPU is excellent at branchy, cache-resident, latency-sensitive control. DRAM is not just an emergency fallback, VRAM is not the model, disk is not token-time memory, and PCIe and network links are transfer fabrics rather than magic shared memory. So the real performance question is not whether a device is fast. It is whether the right data is in the right place at the right time, with the right owner, without forcing the rest of the pipeline to wait, and NERVA is built to make that question explicit.

### What changes

NERVA shifts the center of gravity of inference away from "load the model into the device" and toward "decide residency, ownership, movement, compute placement, synchronization, and observability before and during execution."

| Traditional inference | NERVA inference |
|---|---|
| The model is loaded into VRAM if possible. | The model is split into scheduled resident blocks. |
| CPU feeds GPU and waits for tokens. | CPU controls policy, memory, and metadata while GPU owns hot execution. |
| VRAM is treated as the model container. | VRAM is a managed hot cache. |
| DRAM is fallback or offload. | DRAM is a warm tier and a possible CPU-compute tier. |
| KV cache is a tensor allocation. | KV cache is virtual memory. |
| Decode is CPU-mediated token by token. | Decode is a device-resident transaction with asynchronous host observation. |
| Runtime overhead is discovered after profiling. | Token ledgers are part of the runtime contract. |
| Fallbacks may be implicit. | Fallbacks must be explicit and measurable. |

### ResidentBlocks

The core NERVA object is the **ResidentBlock**, which is any meaningful unit of data whose location and execution relationship matter. ResidentBlocks cover weights, weight tiles, KV pages, activations, logits, device tokens, host-visible tokens, sampler state, metadata, workspaces, staging buffers, and future transport buffers. Each one is tracked with enough information to answer practical execution questions, including what it is, how large it is, where it lives, who owns it, whether it is hot or cold, when it is needed next, whether it can be evicted, whether it can be prefetched, whether the CPU should compute against it directly, and whether the GPU should own the next phase.

This is the conceptual difference between NERVA and a normal tensor runtime. A tensor says "here are values," whereas a ResidentBlock says "here are values, here is where they live, here is who owns them, here is when they matter, and here is the cost of moving or computing against them." That richer object is the foundation of everything else.

Every block moves through an explicit residency state machine, and no executor is allowed to consume a block until its required replica is `Ready` and its version satisfies the execution dependency, which is what keeps a moved or evicted block from being read before its transfer finishes.

```mermaid
stateDiagram-v2
    [*] --> Unmapped
    Unmapped --> Allocated: reserve in arena
    Allocated --> Prefetching: stage toward executor
    Prefetching --> Ready: transfer complete
    Ready --> InUse: executor consumes
    InUse --> Ready: phase released
    Ready --> Draining: scheduled to move
    Draining --> Evicting: copy out complete
    Evicting --> Unmapped: slot reclaimed
    InUse --> Invalid: freed or lifetime end
    Invalid --> [*]
```

### Memory residency

NERVA treats memory as a hierarchy of roles rather than a binary capacity wall.

```mermaid
flowchart TB
    DISK["Disk / NVMe, cold tier<br/>model files, cold KV snapshots, persistent prefix cache"]
    CXL["CXL / coherent fabric, expandable warm or cold tier"]
    DRAM["DRAM, warm tier and CPU-compute tier<br/>warm weights, warm KV, prefix cache, computable shards"]
    PIN["Pinned DRAM, staging tier<br/>registered transfer and transport rings"]
    VRAM["VRAM / local HBM, hot tier<br/>active weights, hot KV, activations, device token ring"]
    DISK -->|planned prefetch| DRAM
    CXL --- DRAM
    DRAM -->|stage| PIN
    PIN -->|async H2D| VRAM
    VRAM -->|demote or evict| DRAM
    GPU["GPU, hot tensor plane"] --- VRAM
    CPUc["CPU, control and warm compute"] --- DRAM
    NIC["NIC / fabric, explicit transport device"] --- PIN
```

**VRAM** is the hot tier, and it should hold active weights, hot KV pages, current activations, graph workspaces, sampler state, prefetch slots, and the device token ring. It should not fill up with cold KV, duplicated prefixes, dead temporaries, or layout-conversion garbage.

**Pinned DRAM** is an explicit staging tier, allocated deliberately and reused rather than churned. It exists for controlled host-to-device movement, future RDMA fallback buffers, DPDK packet buffers, mapped host output, and overlap-friendly transfer paths.

**DRAM** is the warm tier, and it holds model backing data, warm weights, warm KV, metadata, prefix cache, scheduler state, CPU-computable shards, and prefetch targets. DRAM is not just slow VRAM, and in some cases it is better to compute against DRAM-resident data on the CPU than to move a huge block to the GPU.

**Disk or NVMe** is the cold tier, storing model files, cold weights, cold KV snapshots, persistent prefix and session cache, and pre-transformed layouts. Disk must never surprise the decode critical path, so if it is involved at all it enters through planned prefetch and cold staging.

**CXL or coherent shared memory** is a future expansion tier, and NERVA is designed so it can eventually target coherent memory fabrics without being rewritten around a CUDA-only, discrete-memory assumption. Across all of these tiers the runtime's job is to keep the critical path resident, not to keep everything resident.

### CPU and GPU roles

NERVA does not treat the CPU as a weak GPU, and it does not treat the GPU as a fast CPU.

The CPU is the latency control plane. It owns request state, scheduler state, stop policy, complex sampling policy, grammar and tool constraints, residency decisions, KV metadata, prefix metadata, weight-block metadata, disk IO planning, pinned-memory management, telemetry, token ledgers, and warm-compute work whenever that is cheaper than moving data. The GPU is the hot throughput plane. It owns resident GEMV and GEMM, prefill dense compute, hot decode projections, hot MLP blocks, attention over hot KV, device-side sampling fast paths, device token state, fused kernels, and persistent decode graph execution.

The deciding policy is compute-near-data versus move-data-to-compute. For batch-one decode, many operations look like a huge weight matrix applied to a tiny current activation, and if that weight block already lives in DRAM then copying the whole thing to the GPU may cost more than letting the CPU compute a partial result and merge the smaller output. NERVA is designed to measure and decide that explicitly, which makes CPU computation a planned mode rather than a desperate fallback.

```mermaid
flowchart TB
    OP["Decode operation: large weight block, tiny activation"] --> Q{"Where does the weight block already live?"}
    Q -->|VRAM, hot| G["GPU executes on device"]
    Q -->|DRAM, warm| D{"Move block to GPU, or compute beside it on the CPU?"}
    D -->|moving is cheaper| MV["Stage block to GPU, execute on GPU"]
    D -->|compute-near-data is cheaper| CN["CPU computes partial result, merge smaller output"]
    G --> MERGE["Merged exact result"]
    MV --> MERGE
    CN --> MERGE
```

### Device-first decoding

Decode is the heart of the runtime. Traditional decode often forces a CPU-visible boundary on every token, so the GPU computes, sampling happens, a token is copied or exposed to the host, the CPU updates state, and only then does the next step proceed.

NERVA separates device token state from host token state. The device token state is what the GPU needs to continue generation, while the host token state is what the server, user, logger, stop policy, or streaming interface observes. Those two are related, but they should not always share the same synchronization point. In the NERVA model the GPU writes the sampled token into a device token ring, the next decode transaction consumes that device token directly, and the CPU observes the token asynchronously unless a correctness policy demands a hard synchronization.

This is not a license to ignore correctness. It is a way to classify correctness boundaries precisely, separating hard syncs, soft host-visibility syncs, debug-only syncs, and policy syncs required by complex stop strings, grammar constraints, or tool-call boundaries. The result is a decode loop designed as a device-resident transaction rather than a CPU-mediated token loop.

```mermaid
sequenceDiagram
    participant G as GPU hot loop
    participant R as Device token ring
    participant C as CPU control loop
    Note over G,R: decode step t
    G->>R: write sampled token t into versioned slot
    R-->>G: step t+1 consumes the same slot directly
    R--)C: async copy of token t to host
    Note over C: host-side stream, stop policy and metadata, kept off the decode path
    C-->>G: explicit policy barrier only when a policy demands it
```

### KV cache as virtual memory

KV cache is not just a tensor, it is virtual memory. Each KV page carries a layer, a head or group, a token range, a size, a dtype, a location, a hotness, an owner, reuse information, and predicted next-use behavior. With that, the runtime can keep recent hot KV in VRAM, move warm KV to DRAM, retain cold KV outside the hot set, and eventually compute attention across multiple tiers at once.

The long-term design supports exact blockwise attention, where hot KV blocks run on the GPU, warm KV blocks may run on the CPU when that is cheaper than staging them, and partial attention results merge through the same online-softmax logic used by IO-aware attention algorithms. The constraint that matters here is exactness, because this tiered KV design never drops context, approximates attention, quantizes KV, or changes model semantics. It only changes where KV lives and where partial work executes.

```mermaid
flowchart TB
    subgraph PAGES["KV pages, each tracks layer, head, token range, hotness, owner"]
        HOTKV["Hot KV pages in VRAM"]
        WARMKV["Warm KV pages in DRAM"]
        COLDKV["Cold KV pages retained off the hot set"]
    end
    HOTKV --> AG["GPU blockwise attention"]
    WARMKV --> AC["CPU blockwise attention when cheaper than staging"]
    AG --> MERGE["Online-softmax merge"]
    AC --> MERGE
    MERGE --> OUT["Exact attention output, context never dropped"]
```

### Static arenas and synchronization

NERVA allocates before the hot path, not during it. The runtime preallocates CPU arenas, pinned DRAM arenas, GPU arenas, KV page pools, sampler buffers, graph buffers, telemetry buffers, and staging regions before decode begins. Once decode is running it does not allocate, free, map, unmap, pin, unpin, register memory, or create hidden page faults, and any allocation that does happen inside a supposedly hot phase counts as a runtime bug.

Synchronization gets the same treatment, because every sync has to justify itself. The runtime distinguishes hard correctness syncs, soft host-observation syncs, debug syncs, and policy syncs, and any sync that exists only because the framework structure forced it should be removed, overlapped, or redesigned. NERVA never hides these costs, since they are measured by the token ledger.

### Token ledgers

NERVA does not accept "tokens per second" as a sufficient explanation, so every generated token produces a ledger. A token ledger records wall latency, GPU active time, GPU idle gaps, CPU active time, CPU blocked time, graph launches, kernel count, runtime API calls, synchronization count, host-to-device bytes, device-to-host bytes, device-to-device bytes, memset bytes, allocator calls, page faults, scheduler time, and token-ring time, and it will eventually break out attention, MLP, norm, KV-write, sampling, and logits timings as well.

The ledger is not an external profiler artifact, it is part of the runtime contract, which is why NERVA can always answer the one question that profiling-by-mythology never quite does, namely why this token took this long.

The per-token record groups its fields by where the time and traffic actually go, so a slow token can be attributed to a specific plane rather than to a vague "GPU is busy."

```text
per token:
    token_id, total_latency_us
    CPU:     scheduler_us, sampling_us, blocked_us, page_faults, malloc_calls
    GPU:     active_us, idle_us, kernel_launches, sync_count, hbm_read_mb, hbm_write_mb
    PCIe:    h2d_bytes, d2h_bytes, visible_us
    DRAM:    read_mb, write_mb
    KV:      vram_pages, dram_pages, moved_pages
    Runtime: prefetched_blocks, evicted_blocks, graph_replays
```

The ledger exists to enforce four non-negotiable runtime rules, that every byte must justify its trip, every kernel launch must justify its existence, every synchronization must justify its stall, and every tensor must have a residency reason. Because CPU blocked time and GPU idle time are recorded separately, a CPU waiting on an event while the GPU does useful work is never miscounted as the GPU sitting idle.

---

## Hardware and the road ahead

### Hardware model

NERVA starts on Linux with one NVIDIA GPU and a CUDA backend, and that is an implementation starting point rather than a philosophical limit. The runtime is designed around backend capabilities, so CUDA-specific raw types stay inside the CUDA backend and core runtime structures avoid hardcoding NVIDIA-only assumptions, which leaves room for future HIP and ROCm support under the same backend contract.

CUDA comes first only because it is the immediate development environment, and the architecture has to allow HIP later by treating both as implementations of one broader device model built from device memory, pinned host memory, streams, events, graph-like execution where available, kernel launch, and device-visible state. In short, NERVA is CUDA-first, not CUDA-only.

### Coherent shared memory

The long-term hardware target is stronger than today's discrete CPU-plus-GPU machine. NERVA is designed for heterogeneous coherent shared-memory systems with a unified virtual address map, shared physical HBM or LPDDR, a coherent fabric or NoC, CPU cores for branchy latency work, GPU SMs or CUs for tensor throughput, hardware queues, and phase-based synchronization. On such machines the CPU and GPU may address the same physical memory, but that capability should not be abused, because random CPU-GPU ping-pong over the same cache lines can destroy performance.

NERVA answers this with phase-owned coherence, where a block can be CPU-owned, GPU-owned, shared read-only, or in a handoff phase, so coherence serves zero-copy visibility and clean ownership transfer rather than uncontrolled fine-grained sharing. On today's discrete systems NERVA emulates this model with explicit residency, staging, prefetch, and ownership tracking, and on future coherent machines the same model maps more directly onto the hardware.

The runtime discovers which kind of memory fabric it is running on at startup rather than guessing from the product name, and it treats four cases explicitly. `DiscreteExplicit` is today's separate CPU DRAM and GPU VRAM with explicit transfers, and it is the first target. `UnifiedVirtualManaged` shares a virtual address model but pages may fault or migrate, so it is used only after characterization and never with reactive migration on the decode path. `CoherentSharedPhysical` shares physical memory and hardware coherence, which turns block location into a locality preference and ownership state rather than a visibility boundary. `CxlCoherentFabric` extends capacity and coherence across a CXL-class fabric, treated as another memory domain with its own measured latency and bandwidth rather than as more local HBM.

```mermaid
flowchart TB
    UVA["Unified virtual address map"] --> FAB["Coherent fabric, crossbar or NoC"]
    FAB --> CPUcore["CPU cores<br/>branchy, latency-sensitive work"]
    FAB --> GPUsm["GPU SMs or CUs<br/>tensor throughput"]
    CPUcore --> SHM["Shared physical HBM / LPDDR / SRAM"]
    GPUsm --> SHM
```

On such a machine the same block changes owner over time rather than being copied, and the runtime tracks that ownership as an explicit phase so a writer and a reader never collide on the same memory.

```mermaid
stateDiagram-v2
    [*] --> CpuOwned
    CpuOwned --> Handoff: phase boundary
    Handoff --> GpuOwned: ownership transferred
    GpuOwned --> Handoff: phase boundary
    Handoff --> CpuOwned: ownership returned
    CpuOwned --> SharedReadOnly: publish read-only
    GpuOwned --> SharedReadOnly: publish read-only
    SharedReadOnly --> Handoff: reclaim for a writer
```

### Future transport and distributed inference

Networking, multi-host scheduling, RDMA, and DPDK are deliberately out of scope for the current single-GPU runtime, and they stay future work until the single-GPU critical path is fully understood and measured. The architecture is nonetheless built so a future NERVA can grow into a multi-host inference system without a rewrite, and this section records that design.

#### Move activations, not weights

The distributed rule is simple, which is to move activations, not weights. If a system owns a range of model layers then its weights and KV stay local, and the next system only needs the boundary activation, which makes inference across several machines possible without pretending every GPU belongs to one giant memory pool.

```mermaid
flowchart LR
    IN["Prompt / hidden state"] --> S1
    subgraph S1["System 1, stage A"]
        W1["Weights A + KV A, local"]
    end
    subgraph S2["System 2, stage B"]
        W2["Weights B + KV B, local"]
    end
    subgraph S3["System 3, stage C"]
        W3["Weights C + KV C, local"]
    end
    subgraph S4["System 4, stage D"]
        W4["Weights D + KV D, local"]
    end
    S1 -->|boundary activation| S2
    S2 -->|boundary activation| S3
    S3 -->|boundary activation| S4
    S4 --> LOGITS["Logits, then next token"]
```

This pattern solves capacity rather than active-weight bandwidth, since a dense exact model still has to touch its active weights on each pass, and single-request decode stays sequential across stages, so pipeline utilization improves with multiple requests or chunked prefill rather than making serial token latency free.

#### The VRAM-to-NIC problem

Moving activations instead of weights is correct, but it hides a detail that decides whether the whole scheme is fast, which is where the activation actually lives at the moment it has to be sent. If the stage output sits in GPU VRAM and the NIC cannot read GPU memory directly, the path becomes VRAM, then pinned host DRAM, then NIC, then network, then remote DRAM, then remote VRAM, which crosses PCIe twice, touches DRAM, needs staging buffers, and adds synchronization. The direct path of VRAM, then NIC, then remote VRAM avoids the host bounce entirely.

The sizes explain why this is still workable even on the slow path. A decode boundary activation is tiny, since a hidden size of 16,384 at two bytes per element is about 32 KB per token, or roughly 96 KB across three system boundaries, while a prefill boundary is large, on the order of 8,192 by 16,384 by two bytes, about 256 MB, which is why prefill needs chunking and overlap. A ConnectX-6-class NIC runs around 200 Gb/s, roughly 25 GB/s raw, so for small decode activations the network serialization cost is negligible next to model weights, and the real killer is synchronization rather than raw bytes.

```mermaid
flowchart TB
    subgraph BEST["Best case, GPU-direct RDMA or PeerDirect verified"]
        direction LR
        GV1["Source GPU VRAM"] --> N1["NIC reads VRAM directly"] --> NET1["InfiniBand or RoCE"] --> N2["Remote NIC writes VRAM"] --> GV2["Remote GPU VRAM"]
    end
    subgraph FB["Fallback, NIC cannot read VRAM"]
        direction LR
        GV3["Source GPU VRAM"] --> P1["async D2H into pinned ring"] --> N3["NIC RDMA send"] --> NET2["network"] --> P2["remote pinned ring"] --> H1["async H2D"] --> GV4["Remote GPU VRAM"]
    end
```

#### Choosing a transport path

The runtime detects the best available path at startup, records the decision in the ledger, and degrades deterministically rather than silently picking an arbitrary slow route.

```mermaid
flowchart TB
    START["Move a boundary activation"] --> CAP{"GPU-direct peer DMA verified at startup?"}
    CAP -->|yes| A["Path A, true GPU-direct RDMA<br/>GPU VRAM to NIC to remote GPU VRAM"]
    CAP -->|no| MAP{"Mapped pinned host output measured faster?"}
    MAP -->|yes, small decode activation| Dp["Path D, GPU kernel writes mapped pinned host memory, NIC sends it"]
    MAP -->|no| LAST{"Last stage produced on the CPU?"}
    LAST -->|yes| Cp["Path C, CPU writes boundary into pinned send buffer, NIC sends it"]
    LAST -->|no| B["Path B, async D2H into pinned ring, RDMA or DPDK, remote pinned ring, H2D<br/>universal discrete-GPU fallback"]
    A --> LEDGER["Path choice recorded in the ledger"]
    Dp --> LEDGER
    Cp --> LEDGER
    B --> LEDGER
```

Path A is the direct GPU-to-NIC-to-GPU route used when peer DMA, topology, and ordering are verified. Path B is the optimized pinned-host bounce, the universal fallback for discrete GPUs, which still uses host DRAM but avoids any CPU memcpy by going through pre-registered pinned rings. Path C is used when the final stage work is already CPU-resident, so the boundary is written straight into a pinned send buffer and never makes a pointless trip to the GPU and back. Path D has the GPU's final kernel write the boundary directly into mapped pinned host memory, which removes a separate D2H copy and can win for small decode activations even though it still crosses PCIe. Decode and prefill then use different modes, where decode pre-posts receives onto fixed registered rings with no per-token handshake or allocation, and prefill streams chunks and overlaps stage compute with network transfer because its boundary tensors are large.

#### Topology-aware egress

It is not enough to partition layers, because the runtime also has to partition egress responsibility. GPU-direct RDMA performs best when the GPU and the NIC sit under the same PCIe root complex, so a system should produce its stage-boundary tensor on the GPU nearest the NIC rather than wherever the computation happens to land. The rule is to ask not which GPU is fastest but which GPU is closest to where the data has to go next, which usually means assigning the final local layer of a stage to the NIC-near GPU and letting it own the egress packing.

```mermaid
flowchart TB
    subgraph SYS["One system, one stage"]
        G0["GPUs 0 to 6<br/>compute most local layers"]
        G7["GPU 7, NIC-near<br/>owns final layer and egress packing"]
        G0 -->|internal handoff| G7
    end
    G7 --> NIC["ConnectX-6 under the same PCIe root complex"]
    NIC --> NET["boundary activation to next system"]
```

#### Backends, fabrics, and the engineering boundary

The transport abstraction is meant to sit over several backends rather than one vendor path. On NVIDIA the direct route is GPUDirect RDMA through either Linux DMA-BUF or the legacy `nvidia-peermem` module, on AMD it is ROCm PeerDirect, and both ride UCX or libibverbs as the portable layer, with NCCL and NVSHMEM as later options for collectives and GPU-initiated communication, GPUDirect Storage reserved for cold model staging rather than token-time transfer, DPDK UDP as a future custom packet data plane, and TCP confined to control, discovery, and debugging because it is not a production tensor data plane. The fabric itself can be InfiniBand, which is the lower-friction HPC path with the best latency, or RoCE over Ethernet, which is cheaper and more flexible but needs careful lossless-Ethernet and congestion tuning.

The hardware caveat is real and is treated as such. GPUDirect RDMA is documented for data-center and workstation-class GPUs, so it must not be assumed on consumer GeForce parts like the RTX 2080 Ti, which means the runtime tests the direct path at startup and uses the pinned-host route when it is unavailable rather than designing around a capability that may silently fall back. This sets a clear engineering boundary, that NERVA builds the inference system, the residency and execution planner, the stage pipeline, the transport abstraction, and a custom protocol, and it integrates documented vendor peer-memory paths where they exist, but it does not try to fabricate unsupported GPU peer-memory mappings or hand-replace a vendor peer-memory driver, because the GPU driver owns the memory manager and reverse-engineering its page tables would be fragile and unmaintainable. The design therefore never requires proprietary enterprise memory pooling, only an honest transport abstraction with multiple backends, topology awareness, pre-registered buffers, and correct asynchronous ownership.

---

## Positioning

### Relationship to vLLM and rvLLM

NERVA learns from both vLLM and rvLLM without being a fork of either. vLLM is the production ecosystem reference, strong on model compatibility, serving infrastructure, paged KV ideas, attention backend structure, scheduler behavior, benchmark discipline, and real-world deployment pressure, and NERVA studies it closely without inheriting a Python or PyTorch-centered hot path.

rvLLM is the Rust and CUDA architecture reference, strong on Rust-owned execution, explicit kernels, CUDA graph execution, and engine ownership with no Python in the serving path, and NERVA studies it just as closely without blindly inheriting model-family-specific, FP8-first, H100-only, or narrow serving assumptions. What NERVA builds instead is a new runtime organized around ResidentBlock scheduling and memory residency.

### What NERVA is not

NERVA is not a quantizer, a pruner, a distillation pipeline, or an attention approximation. It does not shrink the model, drop context, or trade accuracy for speed, because the Transformer math stays exact.

NERVA is also not a Python or PyTorch wrapper, and it is not a thin scheduler bolted onto an existing engine, because the hot path is Rust-owned and the device backend is explicit rather than hidden behind a framework. Finally, at this stage NERVA is not a finished serving system, a multi-GPU engine, or a network transport, and those are designed-for futures rather than current claims, which the runtime is honest about.

---

## Experimental work

### RT-core candidate selection

NERVA has an experimental RT path for long-context page candidate selection. It uses OptiX hardware traversal to select candidate KV pages, then uses CUDA for exact rerank and attention-stage work. The point is not to run the Transformer on RT cores. The point is to use RT cores for the search-like part of far-context selection, then feed the selected pages back into the normal decode pipeline.

Current knobs:

```bash
cargo run -p nerva-bench -- experimental-rt 524288 8 512 64 64 36
NERVA_EXPERIMENTAL_RT_SEMANTIC_OPTIX=1 \
  cargo run -p nerva-bench -- experimental-rt 524288 8 512 64 64 36
cargo run -p nerva-bench -- experimental-rt-sweep 524288 8 1024 64 64 1 36
cargo run -p nerva-bench -- experimental-rt-matrix 16 64 36
cargo run -p nerva-bench -- experimental-rt-cold-kv 9437184 4 64 8
cargo run -p nerva -- -m qwen3-8b -p "Tell me a story" -c 32768 -o 2048 \
  --rt-mode sparse --rt-far-pages 14 --rt-local-window 4096 --rt-sink-tokens 128
NERVA_EXPERIMENTAL_RT_QK_SELECTOR=1 NERVA_EXPERIMENTAL_RT_QK_FUSED=1 \
  cargo run -p nerva -- -m qwen3-8b -p @/tmp/nerva-long-prompt-32k-o2048.txt \
  -c 32768 -o 2048 --raw --rt-mode sparse --rt-far-pages 14 \
  --rt-local-window 4096 --rt-sink-tokens 128
```

The current Qwen decode path has three selector policies. The default sparse RT policy uses OptiX traversal to produce a sink/local/far page pattern and then runs the normal CUDA selected-page attention path. For semantic experiments, `NERVA_EXPERIMENTAL_RT_QK_SELECTOR=1` switches the Qwen decode integration to a CUDA query/key-aware page selector that scores real KV keys with the live query and still reuses the existing decode, attention, KV, graph, and sampler machinery. `NERVA_EXPERIMENTAL_RT_QK_FUSED=1` folds that Q/K far-page choice into the shared-warp attention kernel, removing the extra selector kernel launches. Both Q/K policies are semantic CUDA experiments, not semantic RT-core retrieval; JSON labels them as `cuda_qk_representative_page_selector` and `cuda_qk_fused_attention_page_selector`.

The RT microbench also has a query-derived OptiX selector behind `NERVA_EXPERIMENTAL_RT_SEMANTIC_OPTIX=1`. That path passes the synthetic query descriptors into the OptiX raygen program and derives the target page from those descriptors instead of hashing `query_id`. It is a measured building block for semantic RT page selection, not a Qwen decode integration yet. Qwen JSON now reports `rt_core_page_selector`, `semantic_page_selection`, and `semantic_rt_retrieval` separately so the synthetic RT path cannot be mistaken for semantic RT retrieval.

Important correction: current Qwen `--rt-mode sparse` is not semantic RT retrieval. The decode integration calls OptiX with page/count metadata only, so it selects sink pages, local pages, and synthetic far pages before handing those page ids back to the existing CUDA selected-page attention path. The detailed evidence and viability notes are in [docs/source/RT_CORE_LLM_VIABILITY.md](docs/source/RT_CORE_LLM_VIABILITY.md).

The semantic integration boundary artifact at `docs/source/perf/rt_semantic_integration_boundary_summary.json` measures the 32k Qwen selector shape. OptiX selector plus CUDA rerank is about 13-14 us for 8 KV-head queries and also about 13-14 us for a synthetic 288-query lower bound covering 36 layers x 8 KV heads. If semantic RT has to launch once per layer, the selector/rerank estimate is about 0.47-0.49 ms/token before descriptor-update cost. That means RT traversal overhead is not the blocker; the missing piece is a real Qwen page-descriptor acceleration structure and a layer-aware launch path that feeds live queries into OptiX.

The measured 512k-token synthetic selector point on an RTX 5090 is:

| Field | Value |
|---|---:|
| Context tokens | 524,288 |
| Pages, page size | 8,192 pages, 64 tokens/page |
| Query count | 8 |
| Candidate pages/query | 512 |
| Dense selector | 28.664 us |
| OptiX selector | 9.485 us |
| OptiX selector + CUDA rerank | 13.644 us |
| Query-derived OptiX selector | 9.135 us |
| Query-derived OptiX selector + CUDA rerank | 13.279 us |
| Selector speedup | 3.02x |
| Selector + rerank speedup | 2.10x |
| Estimated RT attention KV fraction | 78,125 ppm |
| Attention-mass recall | 954,498 ppm avg |

The same run reports synthetic full dense attention at 5.238 ms/layer and the RT selected-page attention stage at 1.211 ms/layer, or 1.202 ms/layer with selector/local overlap modeled. That is a useful attention-stage result, but it is not yet a proven full Qwen decode win. Full decode still has projection, MLP, sampling, graph, and quality effects, and sparse selected-page decode can change outputs unless the candidate set preserves the relevant attention mass.

The 1,048,576-token synthetic sweep now has a summarized artifact at `docs/source/perf/rt_1m_synthetic_sweep_summary.json`. The high-signal points are 512 candidate pages/query at 3.9062% estimated KV bytes with 95.4528% attention-mass recall, and 1024 candidate pages/query at 7.0312% estimated KV bytes with 99.9937% attention-mass recall. This supports RT candidate selection as a viable hot/cold KV search primitive, but it is still synthetic and not semantic Qwen retrieval.

The fixed-1024-candidate synthetic scale artifact at `docs/source/perf/rt_context_scale_c1024_summary.json` extends that check to 8,388,608 tokens. RT selected attention stayed near 2.31 ms while dense synthetic attention grew to 109.02 ms; the estimated Qwen3-8B hot KV set stays about 10.125 GiB while dense BF16 KV would be about 1,152 GiB. This supports the hot/cold memory direction, not exact over-context Qwen inference.

The cold-KV staging artifact at `docs/source/perf/rt_cold_kv_staging_summary.json` measures pinned host-to-device transfer for Qwen3-8B BF16 KV pages. One 64-token page is 9 MiB across all layers. Staging one page took 0.689 ms, four pages took 2.743 ms, eight pages took 5.452 ms, and sixty-four pages took 65.663 ms. That means cold misses must be small, rare, or prefetched; fetching hundreds of pages from host every token would erase the RT selection win. Token-time hot/cold KV paging is still not integrated into Qwen decode.

On the 30,571-token Qwen3-8B prompt with 2,048 generated tokens, 80 selected pages, 4,096 local tokens, 128 sink tokens, and `NERVA_EXPERIMENTAL_PREFILL_LOCAL_WINDOW_TOKENS=4096`, the current decode comparison is:

| Selector policy | Decode throughput | Decode wall | Attention per 256-token chunk | Notes |
|---|---:|---:|---:|---|
| Dense no-RT | 78.24 tok/s | 26.25s | 890.92 ms | Exact full selected-chunk coverage. |
| `optix_synthetic_sink_local_far_page_pattern`, 67 pages | 87.57 tok/s | 23.41s | 516.51 ms | Fastest measured sparse setting; local + sink + one synthetic far page. |
| `optix_synthetic_sink_local_far_page_pattern`, 80 pages | 86.63 tok/s | 23.66s | 536.07 ms | Reproduced current RT path; local + sink + 14 synthetic far pages. |
| `cuda_qk_representative_page_selector` | 80.74 tok/s | 25.39s | 777.78 ms | Semantic page choice, separate selector kernels. |
| `cuda_qk_fused_attention_page_selector` | 81.65 tok/s | 25.11s | 723.46 ms | Semantic page choice with selector launch overhead removed. |

The sparse runs above generate different tokens from dense on this prompt. Dense versus synthetic RT and dense versus fused Q/K first diverge at generated token index 7. That makes current sparse RT a speed experiment, not an exact decode replacement. It also does not reduce VRAM yet: the current Qwen path still allocates the full resident KV cache, so 32k dense and sparse both use about 31.8 GiB on the RTX 5090. The standalone cold-KV staging probe exists; token-time hot/cold KV paging is still future work.

The current real-model integration has three modes:

| Mode | Purpose |
|---|---|
| `--rt-mode shadow` | Launch the selector and report counters while dense decode remains authoritative. |
| `--rt-mode sparse` | Use selected pages in the sparse attention path when the selected page count is below dense page count. |
| `--rt-mode auto` | Enable sparse mode only when the runtime can use it; otherwise fall back explicitly. |

The open performance question is whether RT-selected far pages plus local/sink attention reduce full per-token latency at 32k and above without losing quality. The selector result is promising; full decode improvement remains under measurement. A real semantic RT selector still needs a page-descriptor BVH and CUDA rerank path; the current semantic Q/K selectors are the correctness-facing CUDA baselines before that.

### MoE support

Qwen3-MoE support is being built as real MoE support, not as fake dense projection naming. The model parser recognizes Qwen3-MoE metadata, sparse MLP layer schedules, router weights, per-expert gate/up/down tensors, top-k routing settings, and optional shared-expert tensors.

Current implemented pieces:

| Area | Current state |
|---|---|
| Qwen3-MoE config parsing | `Qwen3MoeForCausalLM`, `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `mlp_only_layers`, QK norm, and shared-expert fields are parsed. |
| Manifest layout | Per-expert tensors are emitted as real HF names such as `model.layers.N.mlp.experts.E.gate_proj.weight`, `up_proj.weight`, and `down_proj.weight`. |
| Real configs covered by tests | Qwen3-30B-A3B produces 18,867 manifest entries; Qwen3-Coder-480B-A35B produces 30,321 manifest entries. |
| Runtime limits | Native exact-runtime contract currently caps experts at 256 and top-k at 16. |
| Shared experts | `shared_expert.gate_proj`, `shared_expert.up_proj`, `shared_expert.down_proj`, and `shared_expert_gate.weight` are represented. |

Qwen3.5 / Qwen3.5-MoE configs are recognized separately, but hybrid linear-attention models are rejected until the runtime implements the required Qwen3.5 attention pieces, including GatedDeltaNet state handling and the Qwen3.5 full-attention differences. That rejection is deliberate; it prevents a config parser match from becoming a false runtime support claim.

### Long-context residency

The long-context target is still exact inference with a managed hot/warm/cold KV hierarchy. Recent KV pages stay hot in VRAM, warm pages can move to DRAM, cold pages are retained outside the decode-critical set, and selection logic decides which far pages must return to the hot path. RT candidate selection is one candidate mechanism for that far-page search. It does not replace attention, projection, or sampling.

The planned correctness rule is unchanged: if NERVA claims exact mode, it must not silently drop context. Sparse or selected-page modes must be labeled as experimental unless they prove parity or a quality bound for the workload being run.

## Development state

### Current implementation status

NERVA is currently a single-GPU, CUDA-first runtime with a real Qwen3-8B CUDA path and a set of probes that lock down the runtime contracts before the system grows into a production server.

Implemented or actively wired:

| Area | Status |
|---|---|
| Qwen3-8B BF16 CUDA decode | Real local safetensors path, tokenizer path, CUDA graph decode, Qwen3 QK norm, and checked-in vLLM parity/perf artifacts. |
| Runtime contracts | Static arenas, device token state, token ledgers, graph replay, no hot-path allocation checks, classified syncs, and explicit host/device causality accounting. |
| Safetensors handling | Header validation, sharded manifest planning, bounded file-range reads, file-backed prefetch, descriptor hashes, and staged resident weights. |
| Exact small-model probes | f32, FP16, and BF16 block/model paths with token or bit parity against reference paths. |
| Attention and KV probes | Exact online-softmax blockwise attention, KV page residency decisions, prefetch, demotion, eviction, and stall ledger events. |
| Warm compute | CPU-resident, GPU-resident, GPU-staged, and hybrid matvec candidates are measured instead of assumed. |
| Transport groundwork | Fabric topology, RDMA/DPDK/backend capability classification, DPDK UDP chunk planning, registration-cache invariants, stage-pipeline planning, and same-node multi-GPU island planning. |
| Experimental RT | OptiX-backed synthetic candidate selection, CUDA selected-page attention, query/key-aware CUDA selector experiments, shadow/sparse/auto modes, synthetic attention-stage estimates, and cold-KV staging measurements. |
| MoE groundwork | Qwen3-MoE parser, manifest, shared-expert roles, loader wiring, native contract limits, and real-config manifest tests. |

Not finished:

| Area | Current limitation |
|---|---|
| Production serving | `nerva serve` exposes an OpenAI-compatible HTTP API; production-grade scheduling, admission control, and deployment hardening are still under development. |
| Full RT decode proof | Candidate selection is measured; full Qwen decode speedup, semantic RT selection, and quality bounds are still being tested. |
| Long-context overprovisioning | Hot/warm/cold KV design exists and cold-KV H2D staging is measured, but exact multi-tier long-context decode is not complete. |
| Qwen3.5 hybrid attention | Configs are recognized and intentionally rejected until the required attention runtime exists. |
| Distributed execution | Transport and stage probes exist; multi-host inference is still future work. |

### Long-term goal

The long-term goal is exact large-model inference that degrades gracefully beyond VRAM. NERVA should eventually support fully resident inference, VRAM-hot-cache inference, CPU/GPU hybrid inference, long-context tiered KV, models larger than VRAM, coherent shared-memory systems, AMD and HIP devices, RDMA transport, DPDK UDP transport, multi-GPU execution, distributed stage pipelines, and old hardware profiles. The aim is not to pretend that many devices are one giant GPU, but to coordinate many memory and compute domains as a single inference machine.

In that machine, weights stay where they are useful, KV stays local to the layers that own it, activations move only when needed, the CPU controls policy and computes near warm data when profitable, the GPU executes hot tensor math, and transports use direct paths when the hardware supports them and pinned fallbacks when it does not.

The final purpose of NERVA is to make AI inference less dependent on giant VRAM pools and vendor-blessed monolithic hardware assumptions. Training made models big, and inference systems decide who can actually run them, so NERVA is the attempt to rebuild that inference system from the ground up.

The build order reflects that single-GPU-first discipline, where each step has to produce clean, measured results before the next one is allowed to build on it, and networking only appears after the local runtime is trustworthy.

```mermaid
flowchart LR
    B1["Baseline<br/>profile a warm engine"] --> B2["Synthetic block<br/>and static arenas"]
    B2 --> B3["Placement benchmark<br/>move vs compute-near-data"]
    B3 --> B4["Persistent decode graph"]
    B4 --> B5["Tiered KV<br/>exact online-softmax merge"]
    B5 --> B6["Minimal real model<br/>then scale"]
    B6 -.future.-> B7["Same-node multi-GPU"]
    B7 -.future.-> B8["Distributed stage pipeline<br/>and transport layer"]
```

---

## Implementation and running it

### Current implementation

This repository is in the runtime foundation stage, so it is not a production model server yet. The current code exists to prove the first runtime contracts, and it is intentionally small because the goal is to lock those contracts down before a larger model path goes on top.

| Checkpoint | Current artifact |
|---|---|
| Device smoke | CUDA driver and runtime load, primary context setup, device allocation, pinned-host allocation, one kernel, and a JSON ledger summary. |
| Static arena | CPU, pinned-host, and GPU logical arenas are preallocated, and any hot-path arena allocation attempt is rejected and ledgered. |
| Synthetic transaction | A native CUDA graph captures a synthetic device-token step against preallocated device state/ring/pinned observation, and the Rust synthetic ledger counts graph replay separately from device activity, copies, and host-visibility waits. |
| Device token | 1,024 synthetic decode steps run on device-ring causality with zero stale, missing, extra, mismatched, or host-causality tokens. |
| Token policy | Device-fast, host-policy, and hybrid validation paths are classified; only host-policy steps create a `PolicySync` and host causality edge. |
| Phase handoff | CPU, GPU, and NIC ownership transfer requires a ready block, matching owner, valid version, legal edge, classified `PhaseHandoff` sync, and version publication. |
| Shared queue | A bounded preallocated descriptor/completion queue passes block handles between owners, rejects overflow and wrong endpoints, and keeps tensor payload bytes out of queue metadata. |
| Execution transaction | A runtime transaction planner validates resident block readiness/version dependencies, records execution decisions, classifies hard/soft/phase syncs, and separates host visibility from device activity. |
| Memory/fabric loop | A bounded ahead-of-use loop plans and executes disk-read, prefetch, staging, eviction, and transport-buffer preparation tasks with explicit residency decisions, copy events, phase handoffs, overlap accounting, and zero pageable copies or per-token registrations. |
| Real block | One exact f32 Transformer block runs through a preallocated scratch path with zero hot-path allocations. |
| Precision block | One exact FP16 and BF16 Transformer block path uses encoded 16-bit weights, inputs, outputs, and preallocated scratch with bit-level reference parity. |
| Safetensors header loader | Single-file and sharded safetensors probes read bounded headers, validate tensor metadata against the HF manifest, and avoid bulk payload reads during metadata inspection. |
| Safetensors file prefetch | Resident shard prefetch reads planned file ranges from safetensors shards, validates complete block coverage, records disk-read/copy events, and hashes read bytes. |
| Single model | One exact tiny f32 greedy decode path checks deterministic token parity and per-token ledgers. |
| Precision single model | One exact tiny FP16 and BF16 Transformer greedy decode path uses encoded weights, encoded embeddings, encoded LM head, reusable scratch, per-token ledgers, and token parity. |
| Qwen3-8B CUDA decode | The `qwen3-8b` alias resolves to the local Qwen/Qwen3-8B HF snapshot, loads BF16 safetensors, runs Qwen3 QK norm, and emits JSON decode metrics from the CUDA path. |
| Qwen3-8B vLLM comparison | Checked-in artifacts record NERVA/vLLM short-decode latency and exact token identity parity for the local Qwen3-8B snapshot. |
| vLLM token parity | A vLLM-style token artifact is compared against NERVA token IDs with exact mismatch, missing, extra, and hash accounting. |
| Tiered attention | Exact online-softmax blockwise attention merges warm DRAM and hot VRAM KV blocks without changing semantics. |
| Experimental RT selector | OptiX-backed synthetic page candidate selection, selected-page CUDA attention, Q/K-aware CUDA selector experiments, synthetic attention-stage estimates, and Qwen decode shadow/sparse/auto flags are wired for measurement. |
| Warm compute | Exact dense matvec candidates compare CPU-resident, GPU-resident, GPU-staged, and hybrid execution with selected-owner ledgering. |
| Qwen3-MoE groundwork | Qwen3-MoE parser, real per-expert manifest entries, shared-expert roles, native contract limits, and loader wiring are present; full production MoE decode remains under development. |
| Kernel contracts | Decode-kernel contract descriptors validate launch bounds, device-resident buffers, and zero hot-path allocation permission. |
| Residency probe | KV page placement across DRAM and VRAM produces explicit prefetch, demotion, eviction, copy, stall, and residency-decision ledger entries. |
| Fabric topology | Linux sysfs topology discovery reports GPU/RDMA PCI bus IDs, NUMA affinity, PCI root-complex affinity, IOMMU mode, peer-memory module state, and explicit GPUDirect-to-pinned-host degradation. |
| Fabric backends | RDMA, DPDK, kernel UDP, and TCP-control backend readiness is classified from real sysfs, module, hugepage, shim-source, and `pkg-config libdpdk` evidence with zero false GPU-direct claims. |
| DPDK UDP protocol | A decode activation transfer is planned as bounded chunks with credit windows, sender retention, receiver bitmaps, NACK ranges, selective retransmit, no per-packet ACKs, preallocated mbufs/rings, and pinned-host fallback unless GPU buffers are verified. |
| Transport path | Direct, pinned-host, CPU-produced, and mapped-pinned transfer candidates are selected by measured capability state with explicit fallback and zero per-token registration. |
| Transport registration | A registration cache binds block replica, backend, address, and version evidence before hot-path entry, records cache hits, rejects stale mappings, and reports zero performed per-token registrations. |
| Stage pipeline | A Fabric-style stage planner keeps weights and KV stage-local while moving only decode activations across stage boundaries. |
| Same-node multi-GPU | A node-local planner treats each GPU as a separate memory island, keeps hot allocations within per-GPU VRAM, uses DRAM backing for cold owned weights, moves only activations across local boundaries, and avoids default all-reduce. |

### Requirements

NERVA currently builds on Linux only, and the first host targets are Ubuntu on `x86_64` and `aarch64`.

The CUDA backend supports **CUDA 12.x and CUDA 13.x only.** Older CUDA stacks are not supported, and newer CUDA major versions should be treated as unsupported until the loader and smoke checks are updated to match. The CUDA loader is written to probe platform-specific driver and runtime library names, but the runtime crates stay gated to Linux while the M0 runtime contracts are being built.

CUDA architecture selection uses explicit overrides first, local GPU detection second, and compiler-supported default architectures last. Use `NERVA_CUDA_ARCHITECTURES`, `CUDAARCHS`, or `CMAKE_CUDA_ARCHITECTURES` for a list such as `75;86;89;120`; use `NERVA_CUDA_ARCH` or `CUDA_ARCH` for one target such as `sm_120` or `12.0`. `CUDA_HOME`, `CUDA_PATH`, and `CUDACXX` select the CUDA toolkit and compiler when the default `nvcc` is not the right one.

### Server mode

`nerva serve` starts an OpenAI-compatible HTTP API backed by the same local runtime path as direct generation. The development form is:

```bash
cargo run -p nerva -- serve -m qwen3-8b
```

An installed or already-built binary uses the same arguments without `cargo`:

```bash
./target/release/nerva serve -m qwen3-8b --host 127.0.0.1 --port 8000
```

Bind to `127.0.0.1` for local-only access. Use `--host 0.0.0.0` only when the server should listen on the network and the host firewall or reverse proxy is configured appropriately.

Authentication is optional. When `--api-key` or `NERVA_OPENAI_API_KEY` is set, requests must include a matching bearer token:

```bash
NERVA_OPENAI_API_KEY=dev-secret cargo run -p nerva -- serve -m qwen3-8b
```

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer dev-secret"
```

Basic OpenAI-compatible requests:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Explain NERVA in one paragraph"}],
    "max_tokens": 128
  }'
```

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "input": "Write a concise status update.",
    "max_output_tokens": 128
  }'
```

Streaming works with the OpenAI-style `stream` flag:

```bash
curl -N http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "input": "Stream three short bullets.",
    "stream": true
  }'
```

OpenAI SDKs can point at the local server by overriding the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dev-secret")
response = client.responses.create(
    model="qwen3-8b",
    input="Say hello from NERVA.",
)
print(response.output_text)
```

Full server launch configuration:

| Option | Default | Meaning |
|---|---:|---|
| `-m`, `--model` | required | Model alias or local checkpoint path to serve. `qwen3-8b` resolves to the local Qwen3-8B path used by the runtime probes. |
| `--host` | `127.0.0.1` | Listen address. Use `0.0.0.0` for external network access. |
| `--port` | `8000` | Listen port. |
| `-c`, `--context` | model/runtime default | Context token budget. Counts accept plain integers, `k`, or `m`, such as `32768`, `32k`, or `1m`. |
| `-o`, `--output`, `--max-new-tokens` | `256` | Default generation output-token cap when a request does not provide its own cap. |
| `-q`, `--queue` | `1024` | Host-visible token queue capacity. Counts accept the same `k`/`m` suffixes. |
| `--compute-cap`, `--compute-capability` | detected | Override CUDA compute capability, for example `120` for `sm_120`. |
| `--max-concurrent-requests` | `1` | Maximum in-flight generation requests admitted by the server. Must be non-zero. |
| `--workers` | Actix default | HTTP worker thread count. Must be non-zero when set. |
| `--max-blocking-threads` | Actix default | Blocking thread-pool limit for runtime work that cannot run on async workers. Must be non-zero when set. |
| `--api-key` | unset | Required bearer token for HTTP requests. Can also be set with `NERVA_OPENAI_API_KEY`. |
| `-rt`, `--rt` | off | Enable the experimental RT candidate-selection path. |
| `--rt-mode auto` | `auto` | Enable sparse RT only when the runtime can use it; otherwise fall back explicitly. Setting any non-default RT mode enables RT. |
| `--rt-mode shadow` | `auto` | Run the selector and report counters while dense decode remains authoritative. |
| `--rt-mode sparse` | `auto` | Use selected pages in the sparse attention path when selected pages are fewer than dense pages. |
| `--rt-page-tokens` | runtime default | Tokens per RT page. Must be non-zero when set. |
| `--rt-pages` | runtime default | Total selected RT pages. Mutually exclusive with `--rt-far-pages`; must be non-zero when set. |
| `--rt-far-pages` | runtime default | Selected far-page count while sink/local pages are derived separately. Mutually exclusive with `--rt-pages`; must be non-zero when set. |
| `--rt-local-window` | runtime default | Recent-token local window preserved by RT selection. |
| `--rt-sink-tokens` | runtime default | Prefix/sink tokens preserved by RT selection. |
| `--profiling` | off | Emit the runtime profiling/reporting path for server generations. |

Any RT knob (`--rt-page-tokens`, `--rt-pages`, `--rt-far-pages`, `--rt-local-window`, or `--rt-sink-tokens`) enables RT even when `--rt` is not passed. `--rt-pages` and `--rt-far-pages` cannot be used together.

Request bodies can still override generation behavior per call. The chat and responses paths accept OpenAI-style fields including `max_tokens` or `max_output_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`, `stream`, `store`, `metadata`, `previous_response_id`, `conversation`, `session_id`, `cache_key`, `response_format`, `reasoning_effort`, `tools`, and `tool_choice`.

### Running the checks

```bash
cargo test --workspace
```

```bash
cargo run -p nerva-bench -- smoke
cargo run -p nerva-bench -- cuda-graph 1024 64 1
cargo run -p nerva-bench -- synthetic 1024 64
cargo run -p nerva-bench -- token-policy
cargo run -p nerva-bench -- phase-handoff
cargo run -p nerva-bench -- shared-queue
cargo run -p nerva-bench -- transaction
cargo run -p nerva-bench -- memory-loop
cargo run -p nerva-bench -- block
cargo run -p nerva-bench -- precision
cargo run -p nerva-bench -- model 8
cargo run -p nerva-bench -- precision-model 8
cargo run -p nerva-bench -- vllm-parity path/to/vllm_tokens.json 8
cargo run -p nerva-bench -- attention
cargo run -p nerva-bench -- warm
cargo run -p nerva-bench -- contracts
cargo run -p nerva-bench -- kv
cargo run -p nerva-bench -- fabric-topology
cargo run -p nerva-bench -- fabric-backends
cargo run -p nerva-bench -- dpdk-udp
cargo run -p nerva-bench -- transport
cargo run -p nerva-bench -- transport-matrix
cargo run -p nerva-bench -- transport-registration
cargo run -p nerva-bench -- stage-pipeline
cargo run -p nerva-bench -- multi-gpu
cargo run -p nerva-bench -- experimental-rt 524288 8 512 64 64 36
cargo run -p nerva-bench -- experimental-rt-matrix 16 64 36
cargo run -p nerva-bench -- experimental-rt-cold-kv 9437184 4 64 8
cargo run -p nerva-bench -- hf-cuda-shared-fork-batch-compare path/to/qwen3-8b 32 128 4 32 2 "Hello" 120
cargo run -p nerva -- -m qwen3-8b -p "Tell me a story" -c 32768 -o 2048
```

The benchmark commands emit single-line JSON summaries, and the acceptance fields that matter are `hot_path_allocations: 0`, exact token parity for the f32 model probe, exact FP16/BF16 token parity for the precision model probe, exact FP16/BF16 bit parity for the precision block probe, bounded safetensors `header_bytes` and `payload_bytes`, safetensors file-prefetch `disk_read_events`, `ready_blocks`, and `data_hash`, exact vLLM-style token identity parity, Qwen3-8B `token_match: true` for shared-fork comparisons, exact dense-reference parity for the attention tests, zero synthetic token audit failures, the graph, device, copy, and host-wait event counts, token-policy `policy_syncs`, token-policy `device_fast_host_dependencies: 0`, phase-handoff `phase_handoff_syncs`, phase-handoff `owner_mismatch_rejections`, shared-queue `queue_full_rejections`, shared-queue `payload_bytes_in_queue: 0`, transaction `block_version_dependencies`, transaction `hard_syncs`, transaction `soft_visibility_syncs`, transaction `phase_handoff_syncs`, memory-loop `queue_overflows: 0`, memory-loop `pageable_copies: 0`, memory-loop `per_token_registrations: 0`, memory-loop `page_faults: 0`, fabric-topology `false_direct_claims: 0`, fabric-topology `degraded_to_pinned_host` when GPUDirect is not verified, fabric-backend `false_direct_claims: 0`, fabric-backend `explicit_degradations`, fabric-backend `dpdk_pkg_config`, DPDK UDP `direct_gpu_memory_claimed: false`, DPDK UDP `ack_packets: 0`, DPDK UDP `selective_retransmits`, warm-compute `execution_decisions`, contract `device_resident_buffers`, explicit KV residency transfer and stall ledger events, experimental RT `real_rt_backend_available`, `candidate_parity_checked`, `candidate_parity_mismatches: 0`, `attention_mass_recall_*`, `full_decode_latency_measured: false` on derived estimates, cold-KV staging `effective_bandwidth_bps`, matching cold-KV allocation/free counters, transport `pageable_copies: 0`, transport `per_token_registrations: 0`, transport-registration `cache_hits`, transport-registration `stale_address_rejections`, transport-registration `per_token_registrations: 0`, stage-pipeline `inter_stage_weight_bytes: 0`, stage-pipeline `all_reduce_bytes: 0`, multi-gpu `aggregate_vram_pool_claimed: false`, multi-gpu `inter_gpu_weight_bytes: 0`, and multi-gpu `all_reduce_bytes: 0`.
