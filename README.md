<div align="center">

<img src="./images/logo.svg" alt="Nerva" width="600" />

# NERVA

### AI inference, scheduled instead of loaded.

#### Neural Execution & Residency Virtual Architecture.

<sub>
An inference operating system for AI models: memory residency, device-first token state, heterogeneous CPU/GPU execution, and token-level observability built into the runtime itself.
</sub>

<br/>

**NERVA does not treat the GPU as a dumb execution target.**  
**NERVA treats inference as a memory, residency, scheduling, and causality problem.**

<br/>

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
2. [The thesis](#the-thesis)
3. [Why inference needs a new machine](#why-inference-needs-a-new-machine)
4. [What changes](#what-changes)
5. [ResidentBlocks](#residentblocks)
6. [Memory residency](#memory-residency)
7. [CPU and GPU roles](#cpu-and-gpu-roles)
8. [Device-first decoding](#device-first-decoding)
9. [KV cache as virtual memory](#kv-cache-as-virtual-memory)
10. [Static arenas and synchronization discipline](#static-arenas-and-synchronization-discipline)
11. [Token ledgers](#token-ledgers)
12. [Hardware model](#hardware-model)
13. [Coherent shared-memory target](#coherent-shared-memory-target)
14. [Future transport and distributed inference](#future-transport-and-distributed-inference)
15. [Relationship to vLLM and rvLLM](#relationship-to-vllm-and-rvllm)
16. [What NERVA is not](#what-nerva-is-not)
17. [Current stage](#current-stage)
18. [Long-term goal](#long-term-goal)

</details>

---

## What NERVA is

**NERVA means Neural Execution & Residency Virtual Architecture.**

NERVA is an inference operating system for AI models. It is a Rust-first runtime, CUDA-first at the device backend, designed to rebuild LLM inference around memory residency rather than treating the GPU as a single monolithic place where the model must fit.

The Transformer math stays exact. NERVA does not begin by changing the architecture, quantizing weights, pruning layers, dropping context, approximating attention, or replacing the target model with a smaller one. The model remains the model.

What changes is the execution machine around it.

Instead of treating inference as a sequence of framework calls that launch GPU kernels, NERVA treats inference as a live scheduling problem across memory tiers, compute devices, synchronization phases, and token-causality state. Model weights, KV cache, activations, tokens, sampler state, temporary workspaces, and future transport buffers are all represented as explicit runtime objects with location, ownership, lifetime, and next-use semantics.

The point is simple but deep:

**The model is not loaded. The model is scheduled.**

---

## The thesis

Current inference engines usually start from a GPU-first assumption: put the weights in VRAM, put the KV cache in VRAM, let the CPU feed the GPU, and make the CPU observe every token before the next step can proceed.

That works when the model fits comfortably, the KV cache fits comfortably, batching hides overhead, and the workload is shaped for the hardware. It breaks down when the model is larger than VRAM, the context is long, the batch is small, the hardware is old, memory movement is hidden behind framework abstractions, or token latency matters more than aggregate throughput.

NERVA starts from a different assumption.

The runtime should know where every meaningful block of data lives, who owns it, when it is needed next, whether it is hot or cold, whether it is cheaper to move it or compute near it, and whether a synchronization is truly required for correctness.

Inference should be scheduled around the critical path, not blindly executed as a GPU command loop.

---

## Why inference needs a new machine

LLM inference is not one workload.

It contains dense matrix math, attention over a growing KV cache, token sampling, request scheduling, memory allocation, device synchronization, CPU-visible output, host/device transfer, cache management, and sometimes disk or network staging. Treating all of that as “GPU work” hides the actual problem.

A GPU is excellent at hot, parallel, throughput-heavy tensor work. A CPU is excellent at branchy, cache-resident, latency-sensitive control. DRAM is not just an emergency fallback. VRAM is not the model. Disk is not token-time memory. PCIe and network links are transfer fabrics, not magic shared memory.

The real performance question is not whether a device is “fast.”

The real question is whether the right data is in the right place at the right time, with the right owner, without forcing the rest of the pipeline to wait.

NERVA is built to make that explicit.

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

NERVA’s job is not only to execute kernels. Its job is to decide residency, ownership, movement, compute placement, synchronization, and observability.

---

## ResidentBlocks

The core NERVA object is the **ResidentBlock**.

A ResidentBlock is any meaningful unit of data whose location and execution relationship matter.

ResidentBlocks include weights, weight tiles, KV pages, activations, logits, device tokens, host-visible tokens, sampler state, metadata, workspaces, staging buffers, and future transport buffers.

A ResidentBlock is tracked by the runtime with enough information to answer practical execution questions: what it is, how large it is, where it lives, who owns it, whether it is hot or cold, when it is needed next, whether it can be evicted, whether it can be prefetched, whether CPU should compute against it directly, and whether GPU should own the next phase.

This is the conceptual difference between NERVA and a normal tensor runtime. A tensor says “here are values.” A ResidentBlock says “here are values, here is where they live, here is who owns them, here is when they matter, and here is the cost of moving or computing against them.”

That is the foundation of the runtime.

---

## Memory residency

NERVA treats memory as a hierarchy of roles, not as a binary capacity wall.

**VRAM** is the hot tier. It should contain active weights, hot KV pages, current activations, graph workspaces, sampler state, prefetch slots, and the device token ring. It should not be filled with cold KV, duplicated prefixes, dead temporaries, or layout-conversion garbage.

**Pinned DRAM** is an explicit staging tier. It exists for controlled host/device movement, future RDMA fallback buffers, DPDK packet buffers, mapped host output, and overlap-friendly transfer paths. It is allocated deliberately and reused.

**DRAM** is the warm tier. It stores model backing data, warm weights, warm KV, metadata, prefix cache, scheduler state, CPU-computable shards, and prefetch targets. DRAM is not just slow VRAM. In some cases, it is better to compute against DRAM-resident data on CPU than to move a huge block to GPU.

**Disk or NVMe** is the cold tier. It stores model files, cold weights, cold KV snapshots, persistent prefix/session cache, and pre-transformed layouts. Disk must not surprise the decode critical path. If disk is involved, it should be involved through planned prefetch and cold staging.

**CXL or coherent shared memory** is a future expansion tier. NERVA is designed so it can eventually target coherent memory fabrics without being rewritten around a CUDA-only discrete-memory assumption.

The runtime’s job is to keep the critical path resident, not to keep everything resident.

---

## CPU and GPU roles

NERVA does not treat the CPU as a weak GPU, and it does not treat the GPU as a fast CPU.

The CPU is the latency control plane. It owns request state, scheduler state, stop policy, complex sampling policy, grammar or tool constraints, residency decisions, KV metadata, prefix metadata, weight-block metadata, disk IO planning, pinned-memory management, telemetry, token ledgers, and warm-compute work when that is cheaper than moving data.

The GPU is the hot throughput plane. It owns resident GEMV and GEMM, prefill dense compute, hot decode projections, hot MLP blocks, attention over hot KV, device-side sampling fast paths, device token state, fused kernels, and persistent decode graph execution.

The key policy is compute-near-data versus move-data-to-compute.

For batch-one decode, many operations look like a huge weight matrix applied to a tiny current activation. If the weight block already lives in DRAM, copying the whole block to GPU may be worse than letting the CPU compute a partial result and merging the smaller output. NERVA is designed to measure and decide that explicitly.

This makes CPU computation a planned mode, not a desperate fallback.

---

## Device-first decoding

Decode is the heart of the runtime.

Traditional decode often forces a CPU-visible boundary every token. The GPU computes, sampling happens, a token is copied or exposed to the host, the CPU updates state, and only then does the next step proceed.

NERVA separates device token state from host token state.

The device token state is what the GPU needs to continue generation. The host token state is what the server, user, logger, stop policy, or streaming interface observes. Those are related, but they should not always be the same synchronization point.

In the NERVA model, the GPU can write the sampled token into a device token ring. The next decode transaction can consume that device token directly. The CPU observes the token asynchronously unless a correctness policy requires a hard synchronization.

This does not mean ignoring correctness. It means classifying correctness boundaries precisely. Some syncs are hard syncs. Some are soft host-visibility syncs. Some exist only for debugging. Some are policy syncs required by complex stop strings, grammar constraints, or tool-call boundaries.

NERVA’s decode loop is designed to be a device-resident transaction, not a CPU-mediated token loop.

---

## KV cache as virtual memory

KV cache is not just a tensor.

KV cache is virtual memory.

Each KV page has a layer, head or group, token range, size, dtype, location, hotness, owner, reuse information, and predicted next-use behavior. The runtime can keep recent hot KV in VRAM, move warm KV to DRAM, retain cold KV outside the hot set, and eventually compute attention over multiple tiers.

The long-term design supports exact blockwise attention. Hot KV blocks can be processed on GPU. Warm KV blocks may be processed on CPU if that is cheaper than staging them. Partial attention results can be merged using the same online-softmax logic used by IO-aware attention algorithms.

The important constraint is exactness. NERVA’s tiered KV design does not require dropping context, approximating attention, quantizing KV, or changing model semantics. It changes where KV lives and where partial work executes.

---

## Static arenas and synchronization discipline

NERVA should allocate before the hot path.

The runtime should preallocate CPU arenas, pinned DRAM arenas, GPU arenas, KV page pools, sampler buffers, graph buffers, telemetry buffers, and staging regions before decode begins.

During the hot path, the runtime should not allocate, free, map, unmap, pin, unpin, register memory, or create hidden page faults. If allocation is required during a supposedly hot phase, that is a runtime bug.

Synchronization is treated the same way. Every synchronization must justify itself. The runtime distinguishes hard correctness syncs, soft host-observation syncs, debug syncs, and policy syncs. A sync that exists only because the framework structure forced it should be removed, overlapped, or redesigned.

NERVA does not hide these costs. They are measured by the token ledger.

---

## Token ledgers

NERVA does not accept “tokens per second” as a sufficient explanation.

Every generated token should produce a ledger.

A token ledger records wall latency, GPU active time, GPU idle gaps, CPU active time, CPU blocked time, graph launches, kernel count, runtime API calls, synchronization count, host-to-device bytes, device-to-host bytes, device-to-device bytes, memset bytes, allocator calls, page faults, scheduler time, token-ring time, and eventually attention, MLP, norm, KV-write, sampling, and logits timings.

The ledger is not an external profiler artifact. It is part of the runtime contract.

NERVA should always be able to answer: why did this token take this long?

That is how the project avoids optimizing by mythology.

---

## Hardware model

NERVA starts on Linux with one NVIDIA GPU and a CUDA backend.

That is an implementation starting point, not a philosophical limit.

The runtime is designed around backend capabilities. CUDA-specific raw types live inside the CUDA backend. Core runtime structures should not hardcode NVIDIA-only assumptions. Future HIP and ROCm support should fit under the same backend contract.

The first device backend is CUDA because it is the immediate development environment. The architecture must allow HIP later. The runtime should treat CUDA and HIP as implementations of the same broader device model: device memory, pinned host memory, streams, events, graph-like execution where available, kernel launch, and device-visible state.

NERVA is CUDA-first, not CUDA-only.

---

## Coherent shared-memory target

The long-term hardware target is stronger than today’s discrete CPU plus GPU machine.

NERVA is designed for heterogeneous coherent shared-memory machines: systems with a unified virtual address map, shared physical HBM or LPDDR, a coherent fabric or NoC, CPU cores for branchy latency work, GPU SMs or CUs for tensor throughput, hardware queues, and phase-based synchronization.

On such machines, the CPU and GPU may be able to address the same physical memory. That does not mean coherence should be abused. Random CPU/GPU ping-pong over the same cache lines can destroy performance.

NERVA uses phase-owned coherence. A block can be CPU-owned, GPU-owned, shared read-only, or in a handoff phase. Coherence is useful for zero-copy visibility and ownership transfer, not uncontrolled fine-grained sharing.

On today’s discrete systems, NERVA emulates this model with explicit residency, staging, prefetch, and ownership tracking. On future coherent machines, the same runtime model can map more directly to hardware.

---

## Future transport and distributed inference

Networking is not part of the initial runtime.

But the architecture is designed so that future NERVA can become a multi-host inference system.

The distributed rule is simple: move activations, not weights.

If a system owns a range of model layers, its weights and KV stay local. The next system only needs the boundary activation. That makes distributed inference possible without pretending every GPU is part of one giant memory pool.

However, the location of the activation matters. If the activation lives in GPU VRAM and the NIC cannot directly read GPU memory, then the path must bounce through pinned host memory. If GPU-direct RDMA or AMD PeerDirect works, the NIC may read or write GPU memory directly. If not, NERVA must use a preallocated pinned-host fallback.

NERVA can build the runtime, stage pipeline, activation transport, RDMA or DPDK data plane, pinned fallback, scheduler, and topology-aware routing. NERVA cannot fabricate unsupported GPU peer-memory access if the GPU driver and hardware refuse to expose VRAM for peer DMA.

Future transport modes include true GPU-direct RDMA, optimized pinned-host bounce, CPU-produced boundary tensors written directly into pinned send buffers, and GPU kernels that write directly into mapped pinned host memory when that is the best fallback.

Future transport backends include RDMA, UCX, libibverbs, DPDK UDP, kernel UDP for testing, and TCP only for control or debugging. TCP is not a production tensor data plane.

DPDK is a future packet transport backend, not a substitute for GPU-direct memory support. It can bypass the kernel network stack and support custom tensor datagrams, but it does not magically let a NIC read GPU VRAM without the required memory mappings and driver support.

---

## Relationship to vLLM and rvLLM

NERVA learns from both vLLM and rvLLM.

vLLM is the production ecosystem reference. It has model compatibility, serving infrastructure, paged KV ideas, attention backend structure, scheduler behavior, benchmark discipline, and real-world deployment pressure. NERVA should study vLLM carefully, but it should not inherit a Python or PyTorch-centered hot path.

rvLLM is the Rust/CUDA architecture reference. It has useful ideas around Rust-owned execution, explicit kernels, CUDA graph execution, engine ownership, and no Python in the serving path. NERVA should study rvLLM carefully, but it should not inherit model-family-specific, FP8-first, H100-only, or narrow serving assumptions blindly.

NERVA is not a fork of either.

NERVA is a new runtime built around ResidentBlock scheduling and memory residency.

## Current stage

The current development stage is runtime foundation.

The first target is not a real model. The first target is a runtime that proves it can initialize the device, own memory, allocate static arenas, replay a synthetic decode graph, keep token state on device, emit a token ledger, and avoid hot-path allocation.

Only after that foundation exists should NERVA run a real Transformer block. Only after one real block is correct should it run a small model. Only after a small model works should it begin serious residency planning, CPU/GPU compute-near-data experiments, tiered KV, multi-GPU, or distributed execution.

This order is intentional.

A bad runtime with a real model is still a bad runtime.

---

## Long-term goal

The long-term goal is exact large-model inference that degrades gracefully beyond VRAM.

NERVA should eventually support fully resident inference, VRAM-hot-cache inference, CPU/GPU hybrid inference, long-context tiered KV, models larger than VRAM, coherent shared-memory systems, AMD/HIP devices, RDMA transport, DPDK UDP transport, multi-GPU execution, distributed stage pipelines, and old hardware profiles.

The future target is not to pretend many devices are one giant GPU.

The target is to coordinate many memory and compute domains as one inference machine.

Weights should stay where they are useful. KV should stay local to the layers that own it. Activations should move when needed. CPU should control policy and compute near warm data when profitable. GPU should execute hot tensor math. Transports should use direct paths when hardware supports them and pinned fallbacks when it does not.

The final purpose of NERVA is to make AI inference less dependent on giant VRAM pools and vendor-blessed monolithic hardware assumptions.

Training made models big.

Inference systems decide who can actually run them.

NERVA is the attempt to rebuild that inference system from the ground up.

<div align="center">

<br/>

**NERVA makes AI inference scheduled, not loaded.**

<sub>Rust runtime. CUDA first. HIP later. RDMA and DPDK later. Memory residency always.</sub>

</div>
