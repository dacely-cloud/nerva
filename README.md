<div align="center">

<img src="./images/logo.svg" alt="Nerva" width="500" />

### AI inference beyond the VRAM wall.

#### An inference operating system for large models.

<sub>
An inference operating system for AI models: memory residency, device-first token state, heterogeneous CPU/GPU execution, and token-level observability built into the runtime itself.
</sub>

![status](https://img.shields.io/badge/status-architecture_first-0e1520?labelColor=0e1520&color=2563ff)
![runtime](https://img.shields.io/badge/runtime-Rust-b7410e?labelColor=0e1520)
![device](https://img.shields.io/badge/device-CUDA_first-76b900?labelColor=0e1520)
![future](https://img.shields.io/badge/future-HIP_%2B_RDMA_%2B_DPDK-7c3aed?labelColor=0e1520)
![model](https://img.shields.io/badge/model_math-exact-22e3ab?labelColor=0e1520)

<br/>

</div>

---

<details>
<summary><b>Table of contents</b></summary>
<br/>

1. [What NERVA is](#what-nerva-is)
2. [Current implementation](#current-implementation)
3. [Requirements](#requirements)
4. [Run the checks](#run-the-checks)
5. [The thesis](#the-thesis)
6. [Why inference needs a new machine](#why-inference-needs-a-new-machine)
7. [What changes](#what-changes)
8. [ResidentBlocks](#residentblocks)
9. [Memory residency](#memory-residency)
10. [CPU and GPU roles](#cpu-and-gpu-roles)
11. [Device-first decoding](#device-first-decoding)
12. [KV cache as virtual memory](#kv-cache-as-virtual-memory)
13. [Static arenas and synchronization discipline](#static-arenas-and-synchronization-discipline)
14. [Token ledgers](#token-ledgers)
15. [Hardware model](#hardware-model)
16. [Coherent shared-memory target](#coherent-shared-memory-target)
17. [Future transport and distributed inference](#future-transport-and-distributed-inference)
18. [Relationship to vLLM and rvLLM](#relationship-to-vllm-and-rvllm)
19. [What NERVA is not](#what-nerva-is-not)
20. [Current stage](#current-stage)
21. [Long-term goal](#long-term-goal)

</details>

---

## What NERVA is

**NERVA means Neural Execution & Residency Virtual Architecture.**

NERVA is an inference operating system for AI models, a Rust-first runtime with a CUDA-first device backend, designed to rebuild LLM inference around memory residency rather than treating the GPU as one monolithic place where the whole model must fit.

The Transformer math stays exact. NERVA does not begin by changing the architecture, quantizing weights, pruning layers, dropping context, approximating attention, or swapping in a smaller target model. The model remains the model, and what changes is the execution machine around it.

Instead of treating inference as a sequence of framework calls that launch GPU kernels, NERVA treats it as a live scheduling problem that spans memory tiers, compute devices, synchronization phases, and token-causality state. Weights, KV cache, activations, tokens, sampler state, temporary workspaces, and future transport buffers all become explicit runtime objects, each carrying its own location, ownership, lifetime, and next-use semantics.

The point is simple but deep:

**The model is not loaded. The model is scheduled.**

---

## Current implementation

This repository is in the runtime foundation stage. It is not a production model server yet.

The current code proves the first runtime contracts:

| Checkpoint | Current artifact |
|---|---|
| Device smoke | CUDA driver/runtime load, primary context setup, device allocation, pinned-host allocation, one kernel, JSON ledger summary. |
| Static arena | CPU, pinned-host, and GPU logical arenas are preallocated; hot-path arena allocation attempts are rejected and ledgered. |
| Synthetic transaction | Captured synthetic decode graph replay is counted separately from device activity, copies, and host visibility waits. |
| Device token | 1,024 synthetic decode steps use device-ring causality with zero stale, missing, extra, mismatched, or host-causality tokens. |
| Real block | One exact f32 Transformer block runs through a preallocated scratch path with zero hot-path allocations. |
| Residency probe | KV page placement across DRAM and VRAM produces explicit prefetch, demotion, eviction, copy, stall, and residency-decision ledger entries. |

The implementation is intentionally small. It is meant to lock the runtime contracts before a larger model path is added.

---

## Requirements

NERVA currently builds on Linux only. The first host targets are Ubuntu on `x86_64` and `aarch64`.

**WARNING: the CUDA backend currently supports CUDA 12.x and CUDA 13.x only.** Older CUDA stacks are not supported. Newer CUDA major versions should be treated as unsupported until the loader and smoke checks are updated.

The CUDA loader is written to probe platform-specific CUDA driver and runtime library names, but the runtime crates are still gated to Linux while the M0 runtime contracts are being built.

---

## Run the checks

```bash
cargo test --workspace
```

```bash
cargo run -p nerva-bench -- smoke
cargo run -p nerva-bench -- synthetic 1024 64
cargo run -p nerva-bench -- block
cargo run -p nerva-bench -- kv
```

The benchmark commands emit single-line JSON summaries. The important acceptance fields are `hot_path_allocations: 0`, zero synthetic token audit failures, graph/device/copy/host-wait event counts, and explicit KV residency transfer/stall ledger events.

---

## The thesis

Current inference engines usually start from a GPU-first assumption. Put the weights in VRAM, put the KV cache in VRAM, let the CPU feed the GPU, and make the CPU observe every token before the next step can proceed.

That works when the model fits comfortably, the KV cache fits comfortably, batching hides overhead, and the workload is shaped for the hardware. It breaks down the moment the model is larger than VRAM, the context is long, the batch is small, the hardware is old, memory movement is hidden behind framework abstractions, or token latency matters more than aggregate throughput.

NERVA starts from a different assumption. The runtime should know where every meaningful block of data lives, who owns it, when it is needed next, whether it is hot or cold, whether it is cheaper to move it or compute beside it, and whether a synchronization is truly required for correctness. From there, inference should be scheduled around the critical path rather than blindly executed as a GPU command loop.

---

## Why inference needs a new machine

LLM inference is not one workload. It contains dense matrix math, attention over a growing KV cache, token sampling, request scheduling, memory allocation, device synchronization, CPU-visible output, host-to-device transfer, cache management, and sometimes disk or network staging. Treating all of that as "GPU work" hides the actual problem.

A GPU is excellent at hot, parallel, throughput-heavy tensor work. A CPU is excellent at branchy, cache-resident, latency-sensitive control. DRAM is not just an emergency fallback. VRAM is not the model. Disk is not token-time memory. PCIe and network links are transfer fabrics, not magic shared memory.

So the real performance question is not whether a device is "fast." It is whether the right data is in the right place at the right time, with the right owner, without forcing the rest of the pipeline to wait. NERVA is built to make that question explicit.

---

## What changes

NERVA changes the center of gravity of inference.

| Traditional inference | NERVA inference |
|---|---|
| The model is loaded into VRAM if possible. | The model is split into scheduled resident blocks. |
| CPU feeds GPU and waits for tokens. | CPU controls policy, memory, and metadata while GPU owns hot execution. |
| VRAM is treated as the model container. | VRAM is a managed hot cache. |
| DRAM is fallback/offload. | DRAM is a warm tier and possible CPU-compute tier. |
| KV cache is a tensor allocation. | KV cache is virtual memory. |
| Decode is CPU-mediated token-by-token. | Decode is a device-resident transaction with asynchronous host observation. |
| Runtime overhead is discovered after profiling. | Token ledgers are part of the runtime contract. |
| Fallbacks may be implicit. | Fallbacks must be explicit and measurable. |

The job is not only to execute kernels. It is to decide residency, ownership, movement, compute placement, synchronization, and observability before and during execution.

---

## ResidentBlocks

The core NERVA object is the **ResidentBlock**, any meaningful unit of data whose location and execution relationship matter.

ResidentBlocks cover weights, weight tiles, KV pages, activations, logits, device tokens, host-visible tokens, sampler state, metadata, workspaces, staging buffers, and future transport buffers. Each one is tracked with enough information to answer practical execution questions, including what it is, how large it is, where it lives, who owns it, whether it is hot or cold, when it is needed next, whether it can be evicted, whether it can be prefetched, whether the CPU should compute against it directly, and whether the GPU should own the next phase.

This is the conceptual difference between NERVA and a normal tensor runtime. A tensor says "here are values," whereas a ResidentBlock says "here are values, here is where they live, here is who owns them, here is when they matter, and here is the cost of moving or computing against them." That richer object is the foundation of the runtime.

---

## Memory residency

NERVA treats memory as a hierarchy of roles, not as a binary capacity wall.

**VRAM** is the hot tier, and it should hold active weights, hot KV pages, current activations, graph workspaces, sampler state, prefetch slots, and the device token ring. It should not fill up with cold KV, duplicated prefixes, dead temporaries, or layout-conversion garbage.

**Pinned DRAM** is an explicit staging tier, allocated deliberately and reused rather than churned. It exists for controlled host-to-device movement, future RDMA fallback buffers, DPDK packet buffers, mapped host output, and overlap-friendly transfer paths.

**DRAM** is the warm tier, holding model backing data, warm weights, warm KV, metadata, prefix cache, scheduler state, CPU-computable shards, and prefetch targets. DRAM is not just slow VRAM, and in some cases it is better to compute against DRAM-resident data on the CPU than to move a huge block to the GPU.

**Disk or NVMe** is the cold tier, storing model files, cold weights, cold KV snapshots, persistent prefix and session cache, and pre-transformed layouts. Disk must never surprise the decode critical path, so if it is involved at all, it should enter through planned prefetch and cold staging.

**CXL or coherent shared memory** is a future expansion tier. NERVA is designed so it can eventually target coherent memory fabrics without being rewritten around a CUDA-only, discrete-memory assumption.

Across all of this, the runtime's job is to keep the critical path resident, not to keep everything resident.

---

## CPU and GPU roles

NERVA does not treat the CPU as a weak GPU, and it does not treat the GPU as a fast CPU.

The CPU is the latency control plane. It owns request state, scheduler state, stop policy, complex sampling policy, grammar and tool constraints, residency decisions, KV metadata, prefix metadata, weight-block metadata, disk IO planning, pinned-memory management, telemetry, token ledgers, and warm-compute work whenever that is cheaper than moving data.

The GPU is the hot throughput plane. It owns resident GEMV and GEMM, prefill dense compute, hot decode projections, hot MLP blocks, attention over hot KV, device-side sampling fast paths, device token state, fused kernels, and persistent decode graph execution.

The deciding policy is compute-near-data versus move-data-to-compute. For batch-one decode, many operations look like a huge weight matrix applied to a tiny current activation, and if that weight block already lives in DRAM, copying the whole thing to the GPU may cost more than letting the CPU compute a partial result and merge the smaller output. NERVA is designed to measure and decide that explicitly, which makes CPU computation a planned mode rather than a desperate fallback.

---

## Device-first decoding

Decode is the heart of the runtime.

Traditional decode often forces a CPU-visible boundary on every token. The GPU computes, sampling happens, a token is copied or exposed to the host, the CPU updates state, and only then does the next step proceed.

NERVA separates device token state from host token state. The device token state is what the GPU needs to continue generation, while the host token state is what the server, user, logger, stop policy, or streaming interface observes. Those are related, but they should not always share the same synchronization point. In the NERVA model, the GPU writes the sampled token into a device token ring, the next decode transaction consumes that device token directly, and the CPU observes the token asynchronously unless a correctness policy demands a hard synchronization.

This is not a license to ignore correctness. It is a way to classify correctness boundaries precisely, separating hard syncs, soft host-visibility syncs, debug-only syncs, and policy syncs required by complex stop strings, grammar constraints, or tool-call boundaries. The result is a decode loop designed as a device-resident transaction, not a CPU-mediated token loop.

---

## KV cache as virtual memory

KV cache is not just a tensor. KV cache is virtual memory.

Each KV page carries a layer, a head or group, a token range, a size, a dtype, a location, a hotness, an owner, reuse information, and predicted next-use behavior. With that, the runtime can keep recent hot KV in VRAM, move warm KV to DRAM, retain cold KV outside the hot set, and eventually compute attention across multiple tiers at once.

The long-term design supports exact blockwise attention. Hot KV blocks run on the GPU, warm KV blocks may run on the CPU when that is cheaper than staging them, and partial attention results merge through the same online-softmax logic used by IO-aware attention algorithms. The constraint that matters here is exactness, because this tiered KV design never drops context, approximates attention, quantizes KV, or changes model semantics. It only changes where KV lives and where partial work executes.

---

## Static arenas and synchronization discipline

NERVA allocates before the hot path, not during it.

The runtime preallocates CPU arenas, pinned DRAM arenas, GPU arenas, KV page pools, sampler buffers, graph buffers, telemetry buffers, and staging regions before decode begins. Once decode is running, it does not allocate, free, map, unmap, pin, unpin, register memory, or create hidden page faults, and any allocation that does happen inside a supposedly hot phase counts as a runtime bug.

Synchronization gets the same treatment, since every sync has to justify itself. The runtime distinguishes hard correctness syncs, soft host-observation syncs, debug syncs, and policy syncs, and any sync that exists only because the framework structure forced it should be removed, overlapped, or redesigned. NERVA never hides these costs, because they are measured by the token ledger.

---

## Token ledgers

NERVA does not accept "tokens per second" as a sufficient explanation, so every generated token produces a ledger.

A token ledger records wall latency, GPU active time, GPU idle gaps, CPU active time, CPU blocked time, graph launches, kernel count, runtime API calls, synchronization count, host-to-device bytes, device-to-host bytes, device-to-device bytes, memset bytes, allocator calls, page faults, scheduler time, and token-ring time, and it will eventually break out attention, MLP, norm, KV-write, sampling, and logits timings as well.

The ledger is not an external profiler artifact. It is part of the runtime contract, which is why NERVA can always answer one question that profiling-by-mythology never quite does: why did this token take this long?

---

## Hardware model

NERVA starts on Linux with one NVIDIA GPU and a CUDA backend. That is an implementation starting point, not a philosophical limit.

The runtime is designed around backend capabilities. CUDA-specific raw types stay inside the CUDA backend, and core runtime structures avoid hardcoding NVIDIA-only assumptions so that future HIP and ROCm support can fit under the same backend contract. CUDA comes first only because it is the immediate development environment, and the architecture must allow HIP later by treating both as implementations of one broader device model built from device memory, pinned host memory, streams, events, graph-like execution where available, kernel launch, and device-visible state.

NERVA is CUDA-first, not CUDA-only.

---

## Coherent shared-memory target

The long-term hardware target is stronger than today's discrete CPU-plus-GPU machine.

NERVA is designed for heterogeneous coherent shared-memory systems with a unified virtual address map, shared physical HBM or LPDDR, a coherent fabric or NoC, CPU cores for branchy latency work, GPU SMs or CUs for tensor throughput, hardware queues, and phase-based synchronization. On such machines the CPU and GPU may address the same physical memory, but that capability should not be abused, since random CPU-GPU ping-pong over the same cache lines can destroy performance.

NERVA answers this with phase-owned coherence. A block can be CPU-owned, GPU-owned, shared read-only, or in a handoff phase, so coherence serves zero-copy visibility and clean ownership transfer rather than uncontrolled fine-grained sharing. On today's discrete systems NERVA emulates this model with explicit residency, staging, prefetch, and ownership tracking, and on future coherent machines the same model maps more directly onto the hardware.

---

## Future transport and distributed inference

Networking is not part of the initial runtime, but the architecture is built so that a future NERVA can grow into a multi-host inference system.

The distributed rule is simple. Move activations, not weights. If a system owns a range of model layers, its weights and KV stay local, and the next system only needs the boundary activation. That makes distributed inference possible without pretending every GPU belongs to one giant memory pool.

The location of that activation still matters, though. If it lives in GPU VRAM and the NIC cannot read GPU memory directly, the path has to bounce through pinned host memory, whereas GPU-direct RDMA or AMD PeerDirect lets the NIC read or write GPU memory directly. NERVA can build the runtime, the stage pipeline, the activation transport, the RDMA or DPDK data plane, the pinned fallback, the scheduler, and topology-aware routing, but it cannot fabricate peer-memory access when the GPU driver and hardware refuse to expose VRAM for peer DMA.

Future transport modes therefore include true GPU-direct RDMA, an optimized pinned-host bounce, CPU-produced boundary tensors written straight into pinned send buffers, and GPU kernels that write directly into mapped pinned host memory when that is the best available fallback. The backends behind them include RDMA, UCX, libibverbs, DPDK UDP, kernel UDP for testing, and TCP only for control or debugging, because TCP is not a production tensor data plane. DPDK belongs in that list as a future packet transport, not as a substitute for GPU-direct memory support, since it can bypass the kernel network stack and carry custom tensor datagrams but still cannot let a NIC read GPU VRAM without the required memory mappings and driver support.

---

## Relationship to vLLM and rvLLM

NERVA learns from both vLLM and rvLLM without being a fork of either.

vLLM is the production ecosystem reference, strong on model compatibility, serving infrastructure, paged KV ideas, attention backend structure, scheduler behavior, benchmark discipline, and real-world deployment pressure. NERVA studies it closely, but it does not inherit a Python or PyTorch-centered hot path.

rvLLM is the Rust and CUDA architecture reference, strong on Rust-owned execution, explicit kernels, CUDA graph execution, and engine ownership with no Python in the serving path. NERVA studies it just as closely, but it does not inherit model-family-specific, FP8-first, H100-only, or narrow serving assumptions blindly.

What NERVA builds instead is a new runtime organized around ResidentBlock scheduling and memory residency.

---

## What NERVA is not

NERVA is not a quantizer, a pruner, a distillation pipeline, or an attention approximation. It does not shrink the model, drop context, or trade accuracy for speed, because the Transformer math stays exact.

NERVA is not a Python or PyTorch wrapper, and it is not a thin scheduler bolted onto an existing engine. The hot path is Rust-owned, and the device backend is explicit rather than hidden behind a framework.

NERVA is not, at this stage, a finished serving system, a multi-GPU engine, or a network transport. Those are designed-for futures, not current claims, and the runtime is honest about the difference.

---

## Current stage

The current development stage is runtime foundation plus deterministic residency probes.

The first target is not a serving system. It is a runtime that proves it can initialize the device when one is visible, own memory, allocate static arenas, replay a synthetic decode graph, keep token state on device, emit token ledgers, avoid hot-path allocation, run an exact reference Transformer block, and make KV residency decisions visible.

Current verified probes:

```bash
cargo run -p nerva-bench -- smoke
cargo run -p nerva-bench -- synthetic
cargo run -p nerva-bench -- block
cargo run -p nerva-bench -- kv
```

The `kv` probe exercises a small KV page pool with prefetch, demotion, eviction, copy attribution, and visible-stall ledger events. That is still not real model execution. The next milestones are to connect these contracts to real FP16/BF16 model blocks, then to a small exact greedy decode path, and only after that to broader residency planning, CPU/GPU compute-near-data experiments, tiered KV attention, multi-GPU, or distributed execution.

---

## Long-term goal

The long-term goal is exact large-model inference that degrades gracefully beyond VRAM.

NERVA should eventually support fully resident inference, VRAM-hot-cache inference, CPU/GPU hybrid inference, long-context tiered KV, models larger than VRAM, coherent shared-memory systems, AMD and HIP devices, RDMA transport, DPDK UDP transport, multi-GPU execution, distributed stage pipelines, and old hardware profiles. The aim is not to pretend that many devices are one giant GPU, but to coordinate many memory and compute domains as a single inference machine.

In that machine, weights stay where they are useful, KV stays local to the layers that own it, activations move only when needed, the CPU controls policy and computes near warm data when profitable, the GPU executes hot tensor math, and transports use direct paths when the hardware supports them and pinned fallbacks when it does not.

The final purpose of NERVA is to make AI inference less dependent on giant VRAM pools and vendor-blessed monolithic hardware assumptions.

Training made models big. Inference systems decide who can actually run them.

NERVA is the attempt to rebuild that inference system from the ground up.
