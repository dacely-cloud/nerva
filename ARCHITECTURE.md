# NERVA Architecture

**Project:** NERVA — Neural Execution & Residency Virtual Architecture
**Category:** Inference operating system for AI models
**Document status:** Canonical architecture specification
**Version:** 1.0
**Date:** 2026-06-26

> **The model is not loaded. The model is scheduled.**

---

## 1. Definition

NERVA is an inference operating system for AI models. It rebuilds Transformer inference around explicit memory residency, heterogeneous CPU/GPU execution, device-first token state, static memory ownership, and measured critical-path scheduling. The mathematical model remains unchanged: weights, precision, attention semantics, normalization, MLP semantics, and output distribution are preserved. What changes is the execution machine around the model. Weights, KV pages, activations, tokens, sampler state, workspaces, and transport buffers become explicit runtime objects with known ownership, location, lifetime, next use, movement cost, coherence policy, and available execution paths. NERVA runs first on ordinary discrete CPU-DRAM/GPU-VRAM systems, but its logical machine is designed to map naturally onto heterogeneous coherent shared-memory systems with unified addressing, shared physical HBM or LPDDR, selective cache coherence, hardware work queues, CXL-class fabrics, and direct GPU/NIC transport when the hardware and driver expose it.

NERVA is not a model architecture, a serving wrapper, a faster Python loop, a vLLM fork, or a collection of isolated kernels. It is a runtime architecture that decides:

- where every block of state lives;
- which processor should execute against it;
- when data should move;
- what work can overlap;
- which synchronization is required for correctness;
- which latency is inherent and which latency is software-created;
- how the same model should execute on radically different hardware.

---

## 2. Architectural Thesis

Current inference engines commonly reduce the system to a GPU-centric loop:

```text
weights live on GPU
KV lives on GPU
CPU prepares and schedules
GPU executes
CPU receives the token
repeat
```

NERVA replaces that abstraction with an inference virtual machine:

```text
model
+ memory operating system
+ execution scheduler
+ residency planner
+ token state machine
+ backend/kernel layer
+ transport layer
+ per-token observability
```

The logical resource model is:

```text
VRAM / local HBM     = hot high-bandwidth tier
shared HBM / LPDDR   = coherent heterogeneous tier when available
pinned DRAM          = explicit DMA and transport staging tier
DRAM                 = warm storage and CPU-compute tier
CXL memory           = expandable coherent or semi-coherent warm/cold tier
disk / NVMe          = cold persistent tier
CPU                  = latency, control, metadata, and warm-compute plane
GPU                  = hot tensor-throughput plane
NIC / fabric         = explicit transport device, not fake memory
KV cache             = virtual memory
weights              = scheduled immutable blocks
tokens               = device/host state machine
```

NERVA does not assume that the fastest device is the device with the highest advertised FLOPs. The preferred executor is the device that has acceptable access to the required data, can perform the operation without exposing avoidable latency, and does not force unnecessary synchronization on the rest of the pipeline.

---

## 3. Scope

### 3.1 Initial implementation profile

The first implementation profile is **NERVA-SG**:

```text
one process
one CPU complex
one DRAM pool
one disk/NVMe backing store
one GPU
one VRAM pool
Rust host runtime
CUDA backend first
exact FP16/BF16 execution
batch 1 before general serving
```

The initial implementation exists to prove ownership, execution, memory, token causality, graph replay, and observability. It is not required to support broad model compatibility or production serving at bootstrap.

### 3.2 Future profiles

```text
NERVA-SG       single-GPU runtime
NERVA-HM       coherent heterogeneous-memory runtime
NERVA-MG       same-node multi-GPU runtime
NERVA-Fabric   multi-host distributed runtime
```

All profiles share the same core block, memory, execution, token, and ledger models.

### 3.3 Core research constraints

The baseline research profile MUST preserve model semantics and precision. Performance claims for the core architecture MUST NOT depend on:

- weight quantization;
- KV quantization;
- pruning;
- lossy compression;
- approximate attention;
- dropping context;
- distillation or replacing the target model;
- skipping dense work not skipped by the original model;
- lower precision than the reference execution path.

Optional lossy plugins may exist in the distant future, but they MUST be clearly separated from exact-runtime results and MUST NOT be used to substantiate NERVA's core research claims.

---

## 4. Non-Goals

NERVA is not initially responsible for:

- training;
- automatic model conversion for every Hugging Face architecture;
- an OpenAI-compatible server;
- continuous batching;
- multi-tenant fairness;
- speculative decoding;
- multi-GPU or networking in the bootstrap implementation;
- replacing CUDA or HIP kernel languages with Rust;
- defeating unsupported GPU drivers to expose peer DMA mappings;
- pretending discrete memory is physically coherent;
- silently falling back to unknown or slow code paths.

These may become integrations or later profiles only after the base runtime invariants are established.

---

## 5. Non-Negotiable Invariants

### 5.1 Hot-path invariants

During steady-state decode, the hot path MUST NOT perform:

```text
malloc / free
mmap / munmap
page pinning or unpinning
cudaMalloc / cudaFree
hipMalloc / hipFree
cudaHostAlloc / hipHostMalloc
pageable H2D or D2H transfers
reactive managed-memory migration
first-touch page faults
disk reads
Python calls
PyTorch calls
global mutex acquisition
unbounded queue operations
cudaDeviceSynchronize / hipDeviceSynchronize
per-token memory registration with a NIC
```

All required memory, descriptors, queues, graph resources, and transport registrations MUST be established before the hot path begins or prepared asynchronously ahead of use.

### 5.2 Correctness invariants

For deterministic greedy generation:

```text
next_input_token[t + 1] == sampled_token[t]
host_visible_token[t]   == sampled_token[t]
```

The device token state is authoritative for the next device step. Host visibility is a replicated observation path, not the source of decode causality.

The runtime MUST prevent:

- stale token-ring reads;
- slot reuse before completion;
- duplicate token rows;
- missing token rows;
- reuse of a moved or evicted block before transfer completion;
- simultaneous uncoordinated CPU/GPU writes to the same block;
- stale RDMA mappings after allocation lifetime ends;
- implicit precision or kernel fallback changes.

### 5.3 Observability invariants

Every generated token MUST be explainable. NERVA MUST be able to report the critical-path composition of a token without conflating host waiting with GPU idleness.

In particular:

```text
host_event_wait_us != gpu_idle_us
```

A CPU waiting on an event while the GPU performs useful work is not GPU idle time. GPU idle time is derived only from device-timeline gaps or equivalent hardware evidence.

### 5.4 Fallback invariants

Fallbacks MUST be explicit, named, measurable, and visible in the ledger. A missing optimized kernel or unsupported memory path MUST NOT silently select an arbitrary framework fallback.

---

## 6. Logical Machine

### 6.1 NERVA logical address machine

NERVA presents a unified **logical** address and ownership model even when the hardware has physically separate memories.

```text
                    ┌───────────────────────────────┐
                    │ NERVA global block address map│
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ residency + ownership planner │
                    └───────┬────────┬────────┬──────┘
                            │        │        │
                 ┌──────────▼──┐ ┌───▼────┐ ┌─▼──────────┐
                 │ CPU / DRAM  │ │GPU/VRAM│ │CXL / NVMe  │
                 └─────────────┘ └────────┘ └────────────┘
```

On discrete hardware, a logical address is not a universally dereferenceable raw pointer. It is a stable block handle resolved through a backend-specific mapping. On coherent shared-memory hardware, the mapping may resolve to a shared virtual address directly accessible by CPU and GPU.

### 6.2 Ideal coherent heterogeneous machine

NERVA's hardware end-state is a coherent heterogeneous shared-memory machine:

```text
          ┌────────────────────────────┐
          │ unified virtual address map│
          └─────────────┬──────────────┘
                        │
        ┌───────────────┴────────────────┐
        │coherent fabric / crossbar / NoC│
        └───────┬────────────┬───────────┘
                │            │
          ┌─────▼─────┐ ┌────▼─────┐
          │ CPU cores │ │ GPU SMs  │
          │ branchy   │ │ tensor   │
          │ latency   │ │throughput│
          └─────┬─────┘ └────┬─────┘
                │            │
        ┌───────▼────────────▼───────┐
        │ shared HBM / LPDDR / SRAM  │
        └────────────────────────────┘
```

The winning policy is not uncontrolled fine-grained sharing. It is:

```text
shared address space
shared physical memory when available
coherent but selectively used coherence
hardware-visible queues
phase-based ownership transfer
GPU-local execution for hot tensor work
CPU execution for branchy, irregular, latency-sensitive work
optional CXL / NVLink-class / Infinity-Fabric-class extension
```

---

## 7. Hardware and Memory-Fabric Modes

NERVA discovers hardware capabilities at startup and selects a fabric contract. It MUST NOT infer capability solely from product name.

```rust
pub enum MemoryFabricKind {
    DiscreteExplicit,
    UnifiedVirtualManaged,
    CoherentSharedPhysical,
    CxlCoherentFabric,
}
```

### 7.1 `DiscreteExplicit`

The CPU and GPU have physically separate primary memories. Transfers are explicit and ownership is clear.

```text
CPU DRAM ↔ pinned staging ↔ PCIe ↔ GPU VRAM
```

This is the first implementation target and covers conventional NVIDIA and AMD discrete GPUs.

### 7.2 `UnifiedVirtualManaged`

CPU and GPU share a virtual address model, but pages may migrate or fault. Managed memory is a compatibility mechanism, not the default performance mechanism. NERVA MUST NOT place reactive page migration on the decode critical path.

Managed memory MAY be used only after capability and latency characterization, with explicit prefetch and residency control where the backend supports them.

### 7.3 `CoherentSharedPhysical`

CPU and GPU share physical memory and hardware coherence. Block location becomes a locality preference and ownership state rather than a visibility boundary.

NERVA MUST still track:

- NUMA and memory-controller locality;
- CPU and GPU cache behavior;
- ownership phase;
- read-mostly versus mutable data;
- coherence traffic;
- write-sharing hazards.

Coherence is used for zero-copy visibility and ownership handoff, not as permission for arbitrary simultaneous mutation.

### 7.4 `CxlCoherentFabric`

CXL-class memory and accelerator fabrics extend capacity and coherence beyond a package or socket. NERVA treats these as additional memory domains with measured latency, bandwidth, topology, and coherence semantics. CXL capacity MUST NOT be treated as equal to local HBM capacity.

---

## 8. Core Data Model

### 8.1 Global block address

Core runtime code MUST NOT embed raw CUDA, HIP, CPU, or NIC pointers in general-purpose objects.

```rust
pub struct GlobalBlockAddress {
    pub domain: MemoryDomainId,
    pub allocation: AllocationId,
    pub offset: u64,
}
```

Backend-specific code resolves this address to a native pointer or registration handle.

### 8.2 `ResidentBlock`

`ResidentBlock` is NERVA's primary unit of memory, scheduling, ownership, and observability.

```rust
pub struct ResidentBlock {
    pub id: BlockId,
    pub kind: BlockKind,
    pub bytes: usize,
    pub dtype: DType,
    pub shape: BlockShape,
    pub layout: LayoutId,

    pub address: GlobalBlockAddress,
    pub residency: ResidencySet,
    pub authoritative_copy: ReplicaId,
    pub version: u64,

    pub memory_domain: MemoryDomainId,
    pub fabric: MemoryFabricKind,
    pub owner: ExecutionOwner,
    pub coherence: CoherencePolicy,
    pub access: AccessPolicy,
    pub semantics: MutationSemantics,

    pub lifetime: Lifetime,
    pub hotness: Hotness,
    pub next_use: Option<UseDistance>,
    pub reuse_distance: Option<UseDistance>,

    pub read_cost: CostEstimate,
    pub write_cost: CostEstimate,
    pub move_cost: CostEstimate,
    pub compute_near_data_cost: CostEstimate,

    pub state: ResidencyState,
    pub flags: BlockFlags,
}
```

### 8.3 Block kinds

```rust
pub enum BlockKind {
    Weight,
    KvPage,
    Activation,
    Logits,
    TokenState,
    SamplerState,
    Workspace,
    Queue,
    TransportBuffer,
    Metadata,
}
```

### 8.4 Mutation semantics

```rust
pub enum MutationSemantics {
    Immutable,
    AppendOnly,
    SingleWriter,
    Ephemeral,
    AtomicControl,
}
```

Weights are generally immutable and may have read-only replicas. KV pages are append-only or single-writer. Activations are ephemeral and single-owner. Shared atomic mutation is restricted to compact control data, never bulk tensor data.

### 8.5 Ownership and coherence

```rust
pub enum ExecutionOwner {
    Cpu,
    Gpu(DeviceId),
    Nic(TransportDeviceId),
    SharedReadOnly,
    PhaseTransition,
    None,
}

pub enum CoherencePolicy {
    ExplicitVersioned,
    CoherentReadMostly,
    CoherentPhaseOwned,
    AtomicControlOnly,
}

pub enum AccessPolicy {
    CpuOnly,
    GpuOnly,
    NicOnly,
    CpuGpuReadOnly,
    CpuThenGpu,
    GpuThenCpu,
    GpuThenNic,
    NicThenGpu,
    PhaseOwned,
}
```

Bulk tensor blocks SHOULD use `ExplicitVersioned`, `CoherentReadMostly`, or `CoherentPhaseOwned`. `AtomicControlOnly` is reserved for queue indices, completion flags, and similarly small control state.

### 8.6 Residency state

```rust
pub enum ResidencyState {
    Unmapped,
    Allocated,
    Prefetching,
    Ready,
    InUse,
    Draining,
    Evicting,
    Invalid,
}
```

No executor may consume a block unless its required replica is `Ready` and its version satisfies the execution dependency.

---

## 9. Runtime Components

```text
NERVA
├── capability discovery
├── request and token state machine
├── global block table
├── HILO residency planner
├── execution planner
├── CPU executor
├── GPU executor
├── KV virtual-memory manager
├── static arena manager
├── prefetch and eviction engine
├── kernel-contract registry
├── transport manager
├── topology manager
├── correctness validator
└── token/stall ledger
```

### 9.1 Capability discovery

At startup NERVA records:

- CPU topology, NUMA nodes, cache hierarchy, ISA features, and memory bandwidth;
- GPU backend, architecture, VRAM/HBM capacity, graph support, peer-access capabilities, and DMA capabilities;
- memory-fabric mode and coherence properties;
- PCIe topology and root-complex relationships;
- NICs, RDMA capabilities, DPDK drivers, DMA-BUF or PeerDirect availability;
- CXL and coherent-memory domains where exposed;
- disk and NVMe topology;
- measured transfer latency and bandwidth between relevant domains.

Capabilities are represented as data, not hardcoded SKU assumptions.

```rust
pub struct HardwareCapabilities {
    pub backends: Vec<BackendCapabilities>,
    pub fabric: MemoryFabricKind,
    pub unified_virtual_addressing: bool,
    pub managed_memory_supported: bool,
    pub coherent_cpu_gpu_memory: bool,
    pub shared_physical_memory: bool,
    pub gpu_direct_rdma: CapabilityState,
    pub amd_peerdirect: CapabilityState,
    pub dma_buf_export: CapabilityState,
    pub cxl: CapabilityState,
    pub topology: TopologyGraph,
}
```

### 9.2 HILO residency planner

HILO — Hierarchical Inference Layout Optimizer — decides:

- the authoritative location of each block;
- which read-only replicas are worthwhile;
- whether a block should remain hot, be prefetched, be computed in place, or be evicted;
- which backend should execute each operation;
- whether transfer can be hidden behind independent work;
- whether coherence or explicit copy is the cheaper contract;
- how capacity is divided among weights, KV, workspaces, queues, and staging buffers.

The planner starts with deterministic policies and measured tables. Machine-learned policy is not required.

### 9.3 Execution planner

The execution planner converts model operations and resident blocks into an execution transaction. It produces explicit dependencies, queues, streams, phase transitions, and copy/compute overlap.

### 9.4 Executors

The CPU executor handles latency-sensitive, branchy, irregular, DRAM-local, or control-heavy work. The GPU executor handles hot, parallel, coalesced tensor operations. Neither backend is selected by ideology; selection is a measured critical-path decision.

### 9.5 Prefetch and eviction engine

The engine operates ahead of use. It MUST NOT wait until a dependency is immediately required before initiating a known transfer.

It manages:

```text
disk → DRAM
DRAM → pinned staging
pinned staging → VRAM
VRAM → DRAM eviction
KV hot/warm transitions
transport-buffer preparation
RDMA registration caches
```

### 9.6 Static arenas

NERVA owns preallocated arenas for:

- CPU metadata and request state;
- CPU compute workspaces;
- pinned host staging;
- GPU activations and graph workspaces;
- KV page pools;
- token rings and completion queues;
- transport descriptors and registered buffers;
- ledger buffers.

Arena allocation occurs before hot-path entry. Workspace reset is constant-time and does not release memory to the system.

---

## 10. Execution Model

### 10.1 Concurrent loops

NERVA separates three logically concurrent loops.

#### GPU hot loop

```text
consume device input state
execute prebuilt transaction or graph
read hot weights and KV
append new KV
sample or produce next-token state
publish device completion
```

#### CPU control loop

```text
observe completions
stream or decode tokens
apply complex stop and grammar policy
update request metadata
schedule future work
perform CPU-local compute
```

#### Memory/fabric loop

```text
prefetch blocks
evict cold blocks
prepare registered buffers
advance transfer completions
update hotness and topology costs
```

Only unavoidable dependencies belong on the immediate token critical path.

### 10.2 Device-first token state

The next decode step MUST consume device-resident token state when the device produces it. Host output is an asynchronous replica.

```text
GPU step t
  → sampled token[t] written to device token slot
  → GPU step t+1 consumes the same slot/version
  → host asynchronously observes token[t]
```

Complex host policies may introduce an explicit policy barrier. The barrier must be visible and classified as a policy synchronization, not hidden inside a generic output call.

### 10.3 Token ring

The device token ring is bounded, versioned, and phase-owned. Each slot contains:

```text
request id
sequence id
token index
token value
version
completion state
host-copy state
```

A slot cannot be reused until all declared consumers complete.

---

## 11. Prefill, Decode, and Sampling

### 11.1 Prefill

Prefill processes many prompt tokens and generally exposes substantial parallelism. The default policy is GPU-heavy:

- chunk prompt processing;
- use large fused kernels;
- construct KV pages directly in their target layout;
- avoid materializing full attention matrices;
- overlap chunk compute with later chunk preparation;
- write final KV into page-managed storage without post-conversion.

### 11.2 Decode

Decode is the central latency problem:

```text
one token
serial dependency
small activation
large static weights
growing KV history
many skinny matrix operations
```

Decode SHOULD be represented as a prebuilt transaction with static addresses, bounded queues, and minimal host intervention. CUDA/HIP graph replay is one backend mechanism; it is not the architecture itself.

### 11.3 Sampling policies

NERVA provides distinct policies instead of forcing the slowest policy on every request:

```text
DeviceFastPath:
    greedy, temperature, top-k/top-p where suitable, EOS check

HostPolicyPath:
    complex grammar, regex, tools, custom constraints

HybridValidationPath:
    device candidate generation with asynchronous host validation
```

The policy declares whether host visibility is a hard dependency for the next token.

---

## 12. CPU and GPU Responsibilities

### 12.1 CPU

The CPU is the latency and control plane and MAY be a warm-compute device.

CPU-owned work includes:

- request, scheduler, and stop-policy state;
- tokenization and detokenization;
- complex sampling constraints;
- metadata and block tables;
- residency and prefetch decisions;
- disk and transport control;
- CPU-local matrix/vector work when data movement is more expensive;
- partial attention over DRAM-resident KV when profitable;
- ledger aggregation and asynchronous output.

### 12.2 GPU

The GPU is the hot throughput plane.

GPU-owned work includes:

- prefill GEMMs;
- resident decode GEMV/GEMM;
- hot attention and KV append;
- fused normalization/activation paths;
- device sampling fast paths;
- graph or command-buffer execution;
- transport packing/unpacking when data is GPU-local and direct access exists.

### 12.3 Compute-near-data

For an operation such as:

```text
y = W x
```

NERVA evaluates several exact strategies:

```text
A. W in VRAM; GPU computes.
B. W in DRAM; CPU computes.
C. W in DRAM; prefetch W; GPU computes.
D. W split across CPU/GPU; partial outputs merge.
```

The scheduler compares visible critical-path cost, not raw device peak throughput.

---

## 13. Memory Tiers

### 13.1 VRAM or local HBM

Holds the hot working set:

- hot weight blocks;
- active layer tiles;
- hot KV pages;
- current activations;
- graph workspaces;
- device token state;
- prefetch slots;
- direct transport buffers where supported.

VRAM is not synonymous with the model. It is a managed high-value cache.

### 13.2 Shared HBM or LPDDR

On coherent hardware, shared physical memory may back CPU and GPU work. NERVA still assigns locality and ownership phases. Shared visibility does not imply equal latency, equal bandwidth, or safe simultaneous mutation.

### 13.3 Pinned DRAM

Pinned DRAM is a deliberate DMA and transport tier. It holds long-lived staging rings and registered transport buffers. It MUST NOT be allocated, pinned, or registered per token.

### 13.4 DRAM

DRAM holds:

- full model backing state when needed;
- warm weights;
- CPU-computable shards;
- warm and cold KV;
- metadata and prefix state;
- prefetched disk blocks;
- non-pinned bulk storage.

### 13.5 CXL memory

CXL-attached memory is represented as one or more topology-aware domains. Capacity does not erase latency. Placement policies treat CXL as measured memory, not generic DRAM.

### 13.6 Disk and NVMe

Disk is cold persistence and staging only. It MUST NOT be touched synchronously on the decode critical path. Reads are large, sequential where possible, predicted, asynchronous, and directed into preallocated DRAM arenas.

---

## 14. Coherence and Phase Ownership

NERVA uses coherence selectively.

### 14.1 Phase model

A block moves among ownership phases:

```text
CPU-owned
GPU-owned
NIC-owned
shared read-only
handoff / transition
```

Only the declared owner may mutate a block. Handoff requires an explicit publication and acquisition event or backend-specific equivalent.

### 14.2 Why phase ownership matters

Random fine-grained CPU/GPU writes can create cache-line ping-pong, fence overhead, invalidations, and hidden stalls. The coherent machine is fastest when coherence removes copies and simplifies handoff while ownership remains coarse and predictable.

### 14.3 Hardware-visible queues

On coherent platforms, NERVA MAY use shared descriptor and completion queues:

```rust
pub struct SharedWorkQueue {
    pub descriptors: BlockId,
    pub completions: BlockId,
    pub producer: ExecutionOwner,
    pub consumer: ExecutionOwner,
    pub coherence: CoherencePolicy,
}
```

Queues SHOULD be single-producer/single-consumer where possible, bounded, preallocated, and cache-line padded. Bulk tensors are referenced by block handles, not copied into queue metadata.

---

## 15. KV Virtual Memory

KV cache is a virtual-memory system, not a monolithic tensor.

```rust
pub struct KvPage {
    pub block: BlockId,
    pub layer_id: u32,
    pub head_group_id: u32,
    pub token_start: u32,
    pub token_count: u32,
    pub layout: LayoutId,
    pub location: MemoryDomainId,
    pub hotness: Hotness,
    pub prefix_owner: Option<RequestId>,
    pub last_use: u64,
    pub next_use: Option<u64>,
}
```

### 15.1 Tiers

```text
hot KV    → VRAM or shared HBM
warm KV   → pinned DRAM / DRAM / coherent expanded memory
cold KV   → DRAM / CXL / optional persisted cache
```

### 15.2 Exact blockwise attention

Attention over KV blocks may be computed independently and merged using online softmax state. For block `i`:

```text
m_i = local maximum
l_i = local exponential sum
o_i = local weighted value sum
```

Merge:

```text
m = max_i(m_i)
l = Σ_i exp(m_i - m) · l_i
o = Σ_i exp(m_i - m) · o_i
result = o / l
```

This enables exact or reference-equivalent execution where:

- GPU computes hot KV blocks;
- CPU computes selected DRAM-local blocks;
- another accelerator computes remote/local blocks;
- partial states merge without dropping context.

The policy benefit is unproven until measured for each hardware class; the mathematical decomposition is the enabling primitive.

### 15.3 KV ownership

Each page has a single append owner. Read-only attention consumers may use versioned replicas. Prefix reuse is represented as shared read-only page ownership with reference-managed lifetime outside the decode hot path.

---

## 16. Weight Storage and Layout

Weights are immutable blocks divided by layer, matrix, tile, row range, column range, layout, and backend compatibility.

The loader records the canonical value representation separately from backend-specific packed layouts. A packed replica is valid only for a declared kernel contract and value version.

NERVA MUST support:

- fully resident execution;
- manually selected hot blocks;
- predicted prefetch;
- CPU execution against DRAM-resident blocks;
- split CPU/GPU output merge;
- stage-local ownership in multi-GPU and multi-host profiles.

Disk-backed model files SHOULD store pretransformed layouts when this eliminates repeated load-time conversion, while preserving a verifiable canonical representation.

---

## 17. Cost Model and Scheduling

For each candidate operation placement, NERVA estimates visible critical-path cost.

### 17.1 Transfer

```text
T_transfer_visible = max(0,
    T_setup
  + bytes / effective_bandwidth
  + topology_penalty
  + contention_penalty
  + required_sync
  - overlap_window)
```

### 17.2 GPU execution

```text
T_gpu = queue_delay
      + launch_or_graph_cost
      + max(bytes / effective_local_bandwidth,
            FLOPs / effective_compute)
      + dependency_stalls
      + required_handoff
```

### 17.3 CPU execution

```text
T_cpu = queue_delay
      + cache_and_NUMA_cost
      + max(bytes / effective_DRAM_bandwidth,
            FLOPs / effective_SIMD_compute)
      + merge_cost
```

The planner selects the legal candidate with the lowest predicted effect on the request critical path subject to capacity, correctness, latency policy, and fairness constraints.

### 17.4 Measurement-driven tables

NERVA builds per-machine tables for:

- small and large copies;
- pageable versus pinned transfers;
- CPU SIMD kernels;
- GPU kernel families;
- graph launch/replay;
- memory-domain latency and bandwidth;
- transport paths;
- merge operations;
- queue and synchronization cost.

No hardware path is assumed fast merely because it is newer or marketed as high performance.

---

## 18. Kernel and Backend Architecture

### 18.1 Language split

```text
Rust            runtime ownership, scheduling, memory, queues, ledger
C++/CUDA        NVIDIA kernels, graphs, streams, events, low-level memory
C++/HIP         AMD kernels, graphs/queues, events, low-level memory
C / DPDK        optional packet data-plane integration
Python          optional tooling, conversion, API compatibility outside hot path
```

### 18.2 Device backend contract

Core crates depend on an abstract backend contract.

```rust
pub trait DeviceBackend {
    type Device;
    type Queue;
    type Event;
    type GraphExec;
    type DeviceAllocation;
    type PinnedAllocation;

    fn discover() -> Result<Vec<BackendCapabilities>, BackendError>;
    fn create_device(id: DeviceId) -> Result<Self::Device, BackendError>;
    fn create_queue(device: &Self::Device) -> Result<Self::Queue, BackendError>;
    fn allocate_device(
        device: &Self::Device,
        bytes: usize,
        alignment: usize,
    ) -> Result<Self::DeviceAllocation, BackendError>;
    fn allocate_pinned(
        bytes: usize,
        alignment: usize,
    ) -> Result<Self::PinnedAllocation, BackendError>;
    fn capture(&self, transaction: &ExecutionTransaction)
        -> Result<Self::GraphExec, BackendError>;
    fn submit(&self, executable: &Self::GraphExec)
        -> Result<SubmissionId, BackendError>;
}
```

CUDA-specific types remain in `nerva-cuda`; HIP-specific types remain in `nerva-hip`.

### 18.3 Kernel contracts

Every kernel has an explicit contract:

```text
operation
backend
architecture range
dtypes
layouts
alignment
workspace
mutation semantics
graph safety
determinism/exactness class
expected output tolerance
fallback policy
```

Missing kernels cause an explicit planning failure or declared fallback. NERVA does not silently route through a general framework.

### 18.4 Old hardware

Kernel selection is capability-based. Tensor cores, FP8, or newest architecture features are optional accelerators, never prerequisites for the architecture. NERVA supports architecture-specific kernel families, including conventional CUDA-core/SIMD paths for older hardware.

---

## 19. Synchronization Model

NERVA classifies every synchronization:

```text
HardSync:
    required before dependent device work can be correct

SoftVisibilitySync:
    required for host or remote observation but not next device progress

PolicySync:
    required by grammar, stop, tool, or user policy

PhaseHandoff:
    transfers mutation ownership between CPU/GPU/NIC

DebugSync:
    forbidden in production hot path
```

A synchronization MUST identify the block versions and ownership transition it protects. Generic wait calls without declared dependency are architecture violations.

GPU barriers, atomics, fences, and occupancy costs are kernel-contract concerns and appear in the device ledger. CPU locks, futexes, atomics, page faults, and context switches appear in the host ledger.

---

## 20. Observability and the Token Ledger

NERVA treats observability as part of execution, not a one-off profiler exercise.

### 20.1 Per-token ledger

At minimum:

```text
request_id
sequence_id
token_index
wall_latency_us
cpu_active_us
cpu_blocked_us
gpu_active_us
gpu_idle_us
host_event_wait_us
graph_launches
kernel_count
runtime_api_calls
sync_calls
H2D_bytes
D2H_bytes
D2D_bytes
NIC_TX_bytes
NIC_RX_bytes
memset_bytes
allocator_calls
page_faults
context_switches
attention_us
mlp_us
norm_us
kv_write_us
sampling_us
scheduler_us
prefetch_visible_us
transport_visible_us
```

### 20.2 Residency-decision ledger

Each planner decision records:

```text
block id
old location
new location
executor selected
candidate costs
reason
predicted overlap
actual visible cost
```

### 20.3 Metric provenance

Metrics are marked by source:

```text
runtime timestamp
GPU event
hardware counter
Nsight / ROCm profiler
eBPF / perf
transport completion
estimated model
```

Estimated metrics cannot be presented as measured metrics.

### 20.4 Reproducibility artifacts

Every benchmark stores:

- git commit;
- build flags;
- host and kernel;
- CPU/GPU/NIC inventory;
- driver, CUDA/HIP, RDMA, DPDK versions;
- topology;
- command and environment;
- raw ledger;
- parsed summary;
- correctness hashes;
- profiler references where used.

---

## 21. Same-Node Multi-GPU Profile

NERVA-MG treats GPUs as separate memory and compute islands unless hardware proves otherwise.

Potential execution strategies include:

- layer pipeline;
- row-sharded projections;
- selected column sharding with explicit reductions;
- stable weight ownership per GPU;
- per-GPU KV ownership by layer;
- CPU warm-compute participation;
- topology-aware ingress and egress assignment.

The runtime MUST NOT pretend aggregate VRAM is one coherent pool. It may expose a logical block space while preserving physical ownership and transfer cost.

Global all-reduce on every layer is not the default. The planner prefers moving small activations or partial results over repeatedly moving large weight blocks.

---

## 22. Multi-Host Distributed Profile

NERVA-Fabric extends the same block and execution model across hosts.

### 22.1 Stage pipeline

A large model may be partitioned into stage-local layer ranges:

```text
System 1 owns stage A weights and KV
System 2 owns stage B weights and KV
System 3 owns stage C weights and KV
System 4 owns stage D weights and KV
```

Data flow:

```text
input / hidden state
  → stage A
  → activation
  → stage B
  → activation
  → stage C
  → activation
  → stage D
  → logits
```

Weights remain local. Each stage owns KV for its layers. Activations move.

### 22.2 Capacity versus bandwidth

This architecture solves capacity and avoids moving the full model between hosts. It does not erase the dense-model active-weight bandwidth requirement. For a dense exact model, active weights must still participate in each target pass. NERVA MUST report whether a workload is capacity-bound, local-memory-bandwidth-bound, CPU-compute-bound, PCIe-bound, or network-bound.

### 22.3 Pipeline utilization

Single-request autoregressive decode is sequential across stages. Pipeline utilization improves with multiple requests, chunked prefill, or exact speculative verification. NERVA does not claim that stage partitioning makes serial token latency free.

---

## 23. Transport Architecture

Transport is a pluggable backend that moves named block versions between memory domains and hosts.

```rust
pub trait TensorTransport {
    type Endpoint;
    type Registration;

    fn discover(&self) -> Result<Vec<TransportCapabilities>, TransportError>;
    fn register(
        &self,
        block: &ResidentBlock,
        replica: ReplicaId,
    ) -> Result<Self::Registration, TransportError>;
    fn send(
        &self,
        dst: &Self::Endpoint,
        transfer: TransferDescriptor,
    ) -> Result<TransferId, TransportError>;
    fn post_receive(
        &self,
        src: &Self::Endpoint,
        transfer: ReceiveDescriptor,
    ) -> Result<TransferId, TransportError>;
    fn poll(&self, completions: &mut [TransferCompletion])
        -> Result<usize, TransportError>;
}
```

### 23.1 Transport backends

```text
LocalSharedMemory
LocalGpuPeer
RdmaGpuDirect
RdmaPinnedHost
DpdkUdpGpu
DpdkUdpPinnedHost
KernelUdpTest
TcpControlOnly
```

TCP is not a production tensor data plane. It may be used for control, discovery, debugging, and administrative traffic.

### 23.2 Decode and prefill modes

Decode transfers are small and latency-sensitive. Prefill transfers are large and bandwidth-sensitive.

```text
Decode:
    preposted receives
    fixed registered rings
    no per-token handshake
    low synchronization count

Prefill:
    chunked streaming
    larger windows
    overlap compute and transport
    range completion
```

---

## 24. GPU/NIC Direct-Memory Boundary

NERVA can implement its own inference runtime, stage pipeline, scheduling, activation protocol, RDMA usage, pinned-buffer fallback, topology routing, and DPDK data plane. It cannot safely invent a true GPU-direct NIC path when the GPU memory manager and driver refuse to expose the required DMA mappings.

Direct NIC access to GPU VRAM requires:

- GPU memory pinning;
- virtual-address-to-BAR or DMA mapping;
- lifetime tracking and invalidation callbacks;
- IOMMU-compatible mappings;
- PCIe peer-access permission;
- NIC memory-region registration;
- ordering and coherence guarantees.

These mechanisms require cooperation from the GPU driver, kernel, NIC driver, and platform. NERVA MUST NOT make unsupported peer-memory reverse engineering a production dependency.

### 24.1 NVIDIA paths

Supported NVIDIA integration paths include:

```text
CUDA buffer
  → Linux DMA-BUF export where supported
  → or legacy nvidia-peermem
  → UCX / libibverbs / compatible middleware
  → ConnectX-class HCA/NIC
```

The runtime prefers DMA-BUF where supported and tested. `nvidia-peermem` is an explicit legacy backend. Capability is tested at runtime; GeForce model name alone does not establish support.

GPUDirect performance depends heavily on PCIe topology. GPU and NIC under the same PCIe switch/root complex are preferred. Cross-socket paths may be severely limited. IOMMU translation must be compatible with the selected peer-DMA path.

GPU memory registration is cached. Per-transfer pin/unpin is forbidden in the hot path.

### 24.2 AMD paths

Supported AMD integration paths include:

```text
HIP/ROCm buffer
  → AMD PeerDirect / RDMA-capable GPU memory
  → UCX / libfabric / libibverbs-compatible stack
  → InfiniBand or RoCE NIC
```

NERVA-HIP and transport backends share the same transport contract as CUDA. AMD is a first-class architecture target, not a compatibility afterthought.

### 24.3 Linux P2P

Linux PCI P2PDMA may be used where provider, client, topology, and driver lifetime requirements are satisfied. NERVA treats it as a capability, not a guarantee.

### 24.4 Custom kernel modules

NERVA MAY develop kernel modules that integrate with documented vendor and Linux APIs. It MUST NOT claim that an unsupported consumer GPU can be made safely peer-DMA capable without the vendor memory manager exposing valid mappings and revocation semantics.

---

## 25. Transport Path Selection

NERVA dynamically selects among four primary memory paths.

### Path A — true GPU-direct RDMA

```text
GPU VRAM → NIC → network → remote NIC → remote GPU VRAM
```

Use when peer-DMA capability, topology, registration, and ordering are verified.

### Path B — optimized pinned-host bounce

```text
GPU VRAM
  → asynchronous D2H into pre-registered pinned ring
  → RDMA or DPDK transmit
  → remote pre-registered pinned ring
  → asynchronous H2D
  → remote GPU VRAM
```

This avoids CPU memcpy and per-token registration. It is the required universal fallback for discrete GPUs.

### Path C — CPU-produced boundary

```text
CPU computes boundary result directly into registered pinned send buffer
  → NIC
```

Use when the last stage work is CPU-local or the result is much smaller than the host-resident input weights.

### Path D — GPU writes mapped pinned host memory

```text
GPU final kernel writes boundary output directly into mapped pinned memory
  → NIC transmits the same buffer
```

This still crosses PCIe and may be slower than VRAM writes, but it removes a separate explicit D2H copy and may be effective for small decode activations. It is selected only by measurement.

### 25.1 Selection inputs

The path planner considers:

- source and destination memory domain;
- tensor size and mode (decode/prefill);
- peer-DMA support;
- root-complex topology;
- current PCIe and DRAM pressure;
- queue availability;
- transfer setup and registration state;
- ability to overlap;
- destination executor;
- tail latency.

---

## 26. DPDK Support

DPDK is a first-class future transport backend, not the core inference runtime and not a substitute for GPU peer-memory support.

### 26.1 Purpose

NERVA uses DPDK to support:

- kernel-bypass userspace packet IO;
- poll-mode, bounded-latency data-plane loops;
- ConnectX `mlx5` PMD support;
- custom UDP-style activation transport;
- explicit packet pacing and flow control;
- optional GPU-memory packet buffers where DPDK gpudev and the device stack support direct DMA.

### 26.2 DPDK does not create GPU-direct capability

A DPDK application can only use GPU-resident packet buffers if the NIC and GPU memory can be validly DMA-mapped through the supported driver stack. If they cannot, NERVA uses pinned-host buffers.

### 26.3 Custom UDP data plane

The DPDK backend is message-oriented. A tensor transfer is split into bounded chunks with explicit identity:

```text
protocol version
request id
sequence id
token or prefill chunk id
source and destination stage
block id and version
chunk id and count
offset and length
flags
integrity field
```

Reliability for a private inference fabric uses:

- credit-based flow control;
- bounded sender retention;
- receiver bitmaps;
- NACK of missing ranges;
- fast selective retransmission;
- no ACK per packet unless measurement requires it;
- optional CRC for corruption diagnosis;
- preallocated mbufs and rings.

RoCEv2 and custom DPDK UDP are separate backends. RoCEv2 uses UDP encapsulation underneath an RDMA transport managed by the NIC/verbs stack; NERVA does not reimplement RoCE semantics in DPDK.

### 26.4 DPDK queues and CPU use

DPDK poll-mode workers consume CPU cores. The planner accounts for those cores and NUMA placement. Polling is isolated from CPU warm-compute workers and request-control threads.

---

## 27. Topology-Aware Stage Design

NERVA maintains a graph of CPUs, NUMA nodes, PCIe switches, GPUs, NICs, storage, and memory domains.

The stage-boundary tensor SHOULD be born as close as possible to the next transport device.

```text
If NIC can read GPU memory:
    produce boundary tensor in NIC-near GPU memory.

If NIC cannot read GPU memory:
    produce or copy into a pre-registered pinned egress ring with overlap.

If CPU owns the result:
    never move it to GPU merely to move it back to the NIC.
```

A multi-GPU stage may distinguish compute GPUs, KV owners, and an egress GPU. The final stage-local operation may be placed on the GPU closest to the NIC when that reduces total critical-path cost.

---

## 28. Correctness, Validation, and Exactness

### 28.1 Reference parity

NERVA validates against a reference implementation at progressively larger scopes:

- kernel output;
- single operation;
- Transformer block;
- prefill output;
- per-token logits;
- greedy token stream;
- sampling distribution where stochastic sampling is enabled.

### 28.2 Exactness classes

```text
BitExact
ReferenceEquivalentWithinDeclaredFpTolerance
DistributionPreserving
Approximate
```

Core NERVA results use the first three classes. Approximate results are isolated and labelled.

### 28.3 Block versioning

Every transfer and execution transaction names a block version. Mutable blocks increment version on publication. Read consumers reject stale versions.

### 28.4 Transport correctness

Transport completion does not imply execution visibility until backend ordering requirements and memory barriers are satisfied. GPU/NIC direct receive paths invoke the required backend memory barrier before GPU consumption.

---

## 29. Failure Handling

NERVA fails explicitly and locally.

Examples:

- unsupported direct GPU/NIC registration selects a declared pinned-host path;
- missing kernel contract prevents model plan creation or selects a named exact fallback;
- failed prefetch prevents dependent submission rather than exposing stale data;
- transport loss triggers bounded retransmission or request failure;
- device failure invalidates all replicas owned by that device;
- correctness mismatch halts benchmark acceptance.

Fallback decisions are included in artifacts and metrics.

---

## 30. vLLM and rvLLM Reference Policy

NERVA is a new runtime, but implementation MUST begin with a code audit of vLLM and rvLLM.

### 30.1 vLLM is the compatibility and production oracle

The audit identifies:

- process architecture and scheduler ownership;
- model runner and CUDA graph path;
- token sampling and output handoff;
- PagedAttention and KV block management;
- custom-op and attention backend structure;
- model loading and Hugging Face compatibility;
- allocation, copy, and synchronization callsites;
- distributed abstractions and AMD/HIP platform separation.

NERVA SHOULD reuse concepts and test behavior, not inherit the Python/PyTorch hot-path architecture.

### 30.2 rvLLM is the Rust/CUDA architecture reference

The audit identifies:

- crate structure;
- single engine-owner design;
- CUDA context, stream, and graph ownership;
- memory arenas and pinned buffers;
- token state and sampling path;
- explicit kernel contracts;
- model-, FP8-, and SM90-specific assumptions;
- which runtime patterns generalize to exact FP16/BF16 and older hardware.

NERVA SHOULD reuse proven ownership and graph patterns where legal and appropriate, without inheriting Gemma-, FP8-, H100-, or CUDA-only assumptions.

### 30.3 Mandatory audit output

Before real model execution is implemented, the agent MUST produce a report containing exact file paths, functions, call graphs, memory ownership tables, token-state ownership, graph behavior, KV management, kernel strategy, backend portability, and a decision table:

```text
Area | vLLM | rvLLM | NERVA decision
```

The audit is a design input, not permission to copy code blindly.

---

## 31. Research Status

### 31.1 Proven building blocks

The following concepts are already demonstrated independently in prior systems or hardware stacks:

- exact IO-aware attention;
- blockwise/online softmax attention;
- paged KV management;
- CUDA/HIP graph replay reducing host-launch overhead;
- explicit CPU/GPU/disk offload enabling models larger than VRAM;
- pipeline parallelism moving activations rather than weights;
- GPU-direct RDMA when hardware, topology, and drivers support it;
- AMD PeerDirect and ROCm-aware communication;
- DPDK `mlx5` and gpudev mechanisms;
- coherent CPU/GPU physical memory on integrated accelerator systems.

### 31.2 Novel NERVA contribution

The intended contribution is the integrated runtime model:

- `ResidentBlock` as the central abstraction across weights, KV, activations, tokens, queues, and transport;
- the same logical architecture across discrete explicit memory and coherent shared physical memory;
- exact CPU/GPU compute-near-data decisions;
- device-first token causality separated from host visibility;
- selective phase-owned coherence;
- residency, execution, and transport chosen by one critical-path planner;
- transport paths that understand source/destination memory location;
- token-level and block-decision observability as runtime primitives;
- graceful model-larger-than-VRAM operation without treating host memory as accidental overflow.

### 31.3 Unproven research questions

NERVA does not assume success in advance. The following require measurement:

- whether CPU compute against DRAM-resident dense shards beats staged GPU execution for useful shapes;
- whether exact CPU/GPU partial attention is beneficial on specific systems;
- whether large dense models can reach interactive batch-1 latency without full high-bandwidth residency;
- which coherent-memory policies avoid cache/coherence storms;
- whether mapped pinned output beats explicit D2H for small activations;
- whether DPDK UDP outperforms RDMA for selected message sizes;
- how well old multi-GPU systems can be utilized without fast collectives;
- how to fill distributed pipeline bubbles without compromising exactness.

### 31.4 Known weak or rejected directions

NERVA rejects as default architecture:

- naive synchronous layer offload;
- reactive managed-memory faults;
- disk page faults during decode;
- global tensor parallelism over weak interconnect on every layer;
- excessive atomics or shared writable cache lines;
- many host-submitted tiny kernels without graphing/fusion;
- shallow output-queue changes that break token causality;
- treating CPU event wait duration as GPU idle time;
- assuming DPDK creates GPU peer-DMA access;
- assuming a driver restriction can always be bypassed safely in userspace.

---

## 32. Repository Architecture

The eventual workspace is modular, but bootstrap creates only what is required.

```text
nerva/
├── crates/
│   ├── nerva-core
│   ├── nerva-ledger
│   ├── nerva-memory
│   ├── nerva-runtime
│   ├── nerva-backend
│   ├── nerva-cpu
│   ├── nerva-cuda
│   ├── nerva-hip
│   ├── nerva-kv
│   ├── nerva-kernel-contracts
│   ├── nerva-loader
│   ├── nerva-model
│   ├── nerva-transport
│   ├── nerva-transport-rdma
│   ├── nerva-transport-dpdk
│   └── nerva-bench
├── native/
│   ├── cuda
│   ├── hip
│   └── dpdk
├── docs/
└── tools/
```

Bootstrap SHOULD create only:

```text
nerva-core
nerva-ledger
nerva-memory
nerva-cuda
nerva-runtime
nerva-bench
```

Additional crates are created when their first real implementation and contract exist.

---

## 33. Implementation Checkpoints

No calendar schedule is prescribed. Progress is defined by accepted invariants.

### BOOT

- Rust workspace builds.
- architecture documents exist.
- vLLM/rvLLM audit is complete before real model code.

### DEVICE-SMOKE

- Rust initializes CUDA through a C ABI.
- device and pinned arenas allocate.
- one kernel executes.
- run artifact and ledger are produced.

### STATIC-ARENA

- CPU, pinned, and GPU arenas are preallocated.
- hot-path guard reports zero forbidden allocation.

### SYNTHETIC-TRANSACTION

- one synthetic decode transaction is captured and replayed.
- token ledger records graph launches, host wait, device activity, and copies.

### DEVICE-TOKEN

- device token state is authoritative.
- 1,024 synthetic steps produce no stale, missing, extra, or mismatched tokens.
- host observation does not force per-token device causality through the CPU.

### REAL-BLOCK

- one exact Transformer block matches reference output.
- no hot-path allocation.

### SINGLE-MODEL

- one small model performs exact greedy decode.
- token parity and ledger pass.

### RESIDENCY

- manual `ResidentBlock` placement works across VRAM and DRAM.
- transfer and execution decisions are visible.

### WARM-COMPUTE

- CPU, GPU-staged, GPU-resident, and hybrid exact strategies are compared.

### TIERED-KV

- exact blockwise KV execution across at least two memory tiers passes correctness.

### HIP

- the same runtime contracts execute on AMD through `nerva-hip`.

### FABRIC

- registered activation transfer works through RDMA pinned-host path.
- direct GPU path is capability-detected.
- DPDK backend is benchmarked independently.

---

## 34. Acceptance Criteria

An architecture milestone is accepted only when:

- correctness passes against the declared reference;
- all hot-path allocation counters are zero where required;
- synchronization is classified and attributable;
- measured GPU idle is not inferred from CPU wait;
- fallback paths are explicit;
- artifacts are reproducible;
- no core type leaks CUDA-only assumptions;
- AMD/HIP and transport backends remain implementable through declared contracts;
- the code does not rely on unsupported peer-memory access;
- performance results include both mean and tail latency;
- exactness class is stated.

---

## 35. Security and Isolation

NERVA manipulates raw device memory, pinned host memory, peer mappings, and network registrations. The runtime MUST:

- validate all block offsets and lengths;
- prevent stale handle reuse;
- revoke transport registrations before arena destruction;
- isolate request-visible data where multi-tenancy is enabled;
- clear or version-revoke sensitive buffers according to policy outside the hot path;
- validate transport identities and stage routing;
- avoid exposing raw peer-memory tokens to untrusted code;
- treat kernel modules and DMA mappings as privileged components.

---

## 36. Summary

NERVA is a virtual architecture for heterogeneous AI inference.

It maps the same exact model onto:

```text
ordinary CPU + discrete GPU
coherent CPU/GPU shared-memory packages
same-node multi-GPU systems
multi-host stage pipelines
CXL-expanded memory systems
GPU-direct or pinned-host network fabrics
NVIDIA and AMD backends
```

Its central idea is simple but far-reaching:

```text
Every important byte has an identity, an owner, a location, a lifetime,
a cost, a next use, and a reason for being where it is.
```

The runtime therefore does not ask only whether a model fits in VRAM. It asks which data must be hot, which processor should execute against each block, what can overlap, how coherence should be used, which transport path is legal, and where the token critical path actually waits.

**NERVA makes AI inference scheduled, not loaded.**

---


## 37. Reference Scaling Case: 800 GB on Commodity GPU Stages

A motivating NERVA-Fabric case is:

```text
model size: approximately 800 GB
hosts: 4
GPUs per host: 8
reference GPU class: RTX 2080 Ti, 11 GB each
aggregate VRAM: 352 GB
host DRAM and NVMe: sufficient for remaining state and staging
reference NIC: ConnectX-6 VPI / MCX653106A-class
```

This case is architecturally possible without presenting the 32 GPUs as one device. The model is divided into stable stage-local ownership ranges. Each host stores and executes its own weight range and owns KV for those layers. Stage-boundary activations move between hosts; weights do not move between hosts per token.

```text
prompt / hidden state
  → host 1 stage
  → activation
  → host 2 stage
  → activation
  → host 3 stage
  → activation
  → host 4 stage
  → logits
```

Within each host, the eight GPUs remain separate memory islands. The local planner may combine layer pipeline, row sharding, CPU warm compute, and topology-aware egress placement. It MUST NOT pretend that 88 GB of aggregate local VRAM is one coherent allocation.

For a decode hidden size of 16,384 in FP16/BF16, one stage-boundary activation is approximately 32 KB. This is small relative to the owned weight range. Prefill activations are much larger and require chunking and overlap.

This design solves model capacity and eliminates inter-host weight streaming. It does not eliminate the active-weight bandwidth requirement of an exact dense model. Interactive performance for an 800 GB dense batch-1 workload remains an open research question and MUST be reported honestly. MoE activation sparsity, batching, or exact speculative verification may improve amortization but are separate policies.

The reference case is successful when NERVA can demonstrate:

- stable stage-local weight and KV ownership;
- activation-only inter-host data movement;
- direct GPU/NIC path when genuinely supported;
- pinned-host fallback without pageable copies or per-token registration;
- no global all-reduce across all 32 GPUs per layer;
- explicit per-stage and per-token ledgers;
- graceful performance degradation rather than a fit/fail capacity wall.

---

## Appendix A. Mandatory vLLM and rvLLM Code Audit

No agent may implement real model execution in NERVA before completing this audit. The audit is stored as:

```text
docs/audits/VLLM_RVLLM_ARCHITECTURE_AUDIT.md
```

The report records repository paths, exact commits, file paths, functions, call graphs, ownership, allocation, synchronization, and conclusions. General README summaries are insufficient.

### A.1 vLLM audit

Reference repository:

```text
vLLM checkout
```

The agent MUST inspect at least:

```text
vllm/v1/engine/core.py
vllm/v1/engine/utils.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu/async_utils.py
vllm/v1/attention/
vllm/attention/
vllm/model_executor/
vllm/model_executor/custom_op.py
vllm/_custom_ops.py
vllm/csrc/
vllm/compilation/
vllm/platforms/cuda.py
vllm/platforms/rocm.py where present
vllm/distributed/
```

The vLLM audit MUST answer:

1. Which process and object own request scheduling?
2. Which process and object own GPU context, streams, graphs, and memory?
3. Where are weights loaded and input tensors prepared?
4. Where are CUDA/HIP graphs captured and replayed?
5. Which operations remain outside the graph?
6. Where is the sampled token first materialized?
7. Is the next token authoritative on the device, host, scheduler output object, or Python request state?
8. Where does host output synchronization occur?
9. What exact dependency is protected by `AsyncOutput.get_output()` and its copy-event synchronization?
10. Where do `aten::empty`, pinned-memory requests, D2D copies, H2D/D2H copies, and allocator-like events originate?
11. Which allocations are warm-up only and which are steady-state decode?
12. How are paged KV blocks represented, allocated, written, shared, and evicted?
13. Can KV reside outside VRAM without recomputing or staging it to the GPU?
14. How do CustomOps and attention backends dispatch kernels?
15. What silent fallbacks exist?
16. How are CUDA and ROCm separated, and which abstractions are reusable?
17. Which vLLM components should NERVA reuse conceptually, test against, or avoid?

The audit MUST include a hot-path call graph from engine step to graph submission, sampling, output handoff, and next-step scheduling.

### A.2 rvLLM audit

Reference repository:

```text
rvLLM checkout
```

If absent, clone the upstream repository and record the commit. The agent MUST inspect at least:

```text
v3/Cargo.toml
v3/crates/rvllm-core/
v3/crates/rvllm-mem/
v3/crates/rvllm-graph/
v3/crates/rvllm-runtime/
v3/crates/rvllm-sampling/
v3/crates/rvllm-kernels/
v3/crates/rvllm-cutlass/
v3/crates/rvllm-attention/
v3/crates/rvllm-fused/
v3/crates/rvllm-loader/
v3/crates/rvllm-metadata/
v3/crates/rvllm-serve/
```

Names may differ at the audited commit; the report maps actual crate names and paths.

The rvLLM audit MUST answer:

1. Where is the single engine-owner thread implemented?
2. Which object owns the CUDA context, stream, graph, and device arena?
3. Does steady-state decode allocate?
4. How are HBM/device and pinned-host arenas implemented and reset?
5. How are graphs captured, keyed, updated, and replayed?
6. Is token state device-resident between decode steps?
7. Where does sampling execute and when does the CPU observe the result?
8. How are greedy token hashes or correctness checks performed?
9. Does rvLLM have a concept equivalent to `ResidentBlock`?
10. Which memory structures can be generalized and which are model-specific?
11. Which kernels use cuBLASLt, CUTLASS, custom CUDA, or fused paths?
12. Which assumptions require FP8, SM90, H100, or Gemma-specific layouts?
13. Which paths remain useful for RTX 5090, RTX 4090/3090, or Turing-class GPUs?
14. Is there ROCm/HIP support or a clean backend boundary?
15. What should NERVA reuse conceptually, port carefully, or reject?

The audit MUST include the Rust ownership graph for engine, memory, graph, token state, and output.

### A.3 Required comparison table

```text
Area | vLLM | rvLLM | NERVA decision
```

Required rows:

```text
runtime language
hot-path owner
request scheduler
GPU context ownership
graph capture/replay
static arenas
hot-path allocation
token source of truth
sampling
host output handoff
KV representation
weight loading
kernel contracts
silent fallback behavior
CUDA portability
AMD/HIP portability
model coverage
old-hardware viability
exact FP16/BF16 path
DRAM warm compute
transport assumptions
ResidentBlock compatibility
```

The audit ends with concrete decisions, not a neutral summary.

---

## Appendix B. Transport Capability and Benchmark Matrix

The transport implementation MUST test actual capability. Product names and loaded modules are not proof of a direct memory path.

### B.1 Reference ConnectX-6 profile

The initial fabric target is a ConnectX-6 VPI / MCX653106A-class adapter operating through InfiniBand or RoCE and, separately, through the DPDK `mlx5` PMD in Ethernet mode where appropriate.

The topology report includes:

```text
nvidia-smi topo -m or AMD equivalent
lspci -tv
numactl --hardware
ibv_devinfo
ibstat
ibdev2netdev
IOMMU mode
PCIe root complexes and switches
GPU/NIC NUMA affinity
UCX-selected transports
DPDK PMD and firmware information
```

### B.2 Required transfer paths

```text
A. UCX/libibverbs with a GPU buffer and verified direct RDMA
B. UCX/libibverbs with a pre-registered pinned-host buffer
C. GPU final kernel writes mapped pinned-host output
D. asynchronous D2H into pinned ring followed by RDMA
E. DPDK UDP with GPU buffer when valid DMA mapping exists
F. DPDK UDP with pinned-host buffer
G. kernel UDP test baseline
H. TCP control/debug baseline only
```

A path is labelled `GPU_DIRECT` only when host DRAM traffic, UCX/verbs diagnostics, and device counters support that conclusion. Otherwise it is labelled `HOST_STAGED`.

### B.3 Transfer sizes

```text
32 KB       representative single-token activation
256 KB      grouped decode / small microbatch
1 MB        larger decode/control payload
16 MB       prefill chunk
64 MB       large prefill chunk
256 MB      stress and bandwidth characterization
```

### B.4 Required metrics

```text
P50/P95/P99 completion latency
effective payload bandwidth
CPU core consumption
DRAM read/write bandwidth
PCIe RX/TX bytes
GPU idle and host wait separated
NIC utilization
registration-cache hit rate
packet loss and retransmit rate for DPDK UDP
queue depth and credit stalls
visible non-overlapped transfer time
```

### B.5 Capability result

The startup capability table records:

```text
SUPPORTED_AND_VERIFIED
SUPPORTED_UNVERIFIED
UNSUPPORTED
DEGRADED_TO_PINNED_HOST
```

A failed or unsupported GPU-direct path is not a system failure. NERVA selects the pinned-host ring path and exposes the decision in the ledger.

---

## Appendix C. Direct-Memory Engineering Boundary

NERVA's engineering boundary is explicit:

```text
We build the inference system.
We build the residency and execution planner.
We build the stage pipeline.
We build the transport abstraction and custom protocol.
We integrate documented GPU/NIC peer-memory paths.
We build high-performance pinned-host fallbacks.

We do not rely on fabricating unsupported GPU physical mappings.
```

A custom kernel module is acceptable only when it uses documented Linux and vendor interfaces for pinning, mapping, invalidation, and lifetime management. A replacement for a vendor peer-memory module that depends on reverse-engineering opaque GPU page tables is not a maintainable NERVA requirement.

The practical rule is:

```text
if direct GPU/NIC access is verified:
    use it
else if mapped pinned output wins:
    use it
else:
    use asynchronous GPU ↔ pinned registered ring ↔ NIC
```

The architecture never requires proprietary enterprise memory pooling. It requires explicit capabilities, multiple transport backends, topology awareness, bounded queues, and correct asynchronous ownership.

---

## 38. External Technical References

- NVIDIA, *GPUDirect RDMA — Developing a Linux Kernel Module using GPUDirect RDMA*: https://docs.nvidia.com/cuda/gpudirect-rdma/
- NVIDIA, *GPUDirect RDMA and GPUDirect Storage — GPU Operator*: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html
- Linux kernel, *PCI Peer-to-Peer DMA Support*: https://docs.kernel.org/driver-api/pci/p2pdma.html
- AMD ROCm, *GPU-enabled Message Passing Interface*: https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html
- DPDK, *NVIDIA MLX5 Ethernet Driver*: https://doc.dpdk.org/guides/nics/mlx5.html
- DPDK, *General-Purpose GPU Library*: https://doc.dpdk.org/guides/prog_guide/gpudev.html
- vLLM, *Architecture Overview*: https://docs.vllm.ai/en/latest/design/arch_overview/
- rvLLM repository: https://github.com/m0at/rvllm
- Dao et al., *FlashAttention*: https://arxiv.org/abs/2205.14135
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*: https://arxiv.org/abs/2309.06180
