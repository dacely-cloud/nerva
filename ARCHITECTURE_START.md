The real project name should be:

# **NERVA**

```text
NERVA
Neural Execution & Residency Virtual Architecture
```

Tagline:

```text
NERVA is an inference operating system for AI models.
The model is not loaded. The model is scheduled.
```

What we are building:

```text
Not an LLM server.
Not a vLLM fork.
Not a Rust wrapper.
Not a faster kernel pack.

NERVA is a memory-residency operating system for Transformer inference.
```

This name fits the actual mission:

```text
Neural:
    AI inference

Execution:
    where compute happens

Residency:
    where data lives

Virtual Architecture:
    VRAM / DRAM / disk / CPU / GPU presented as a planned execution machine
```

Internal terms:

```text
Project:
    NERVA

Core runtime:
    nerva-runtime

Core planner:
    HILO
    Hierarchical Inference Layout Optimizer

Core object:
    ResidentBlock

Single-GPU engine:
    nerva-sg

Future multi-GPU engine:
    nerva-mg

Future distributed engine:
    nerva-fabric

CLI:
    nerva
```

Use this wording everywhere:

```text
NERVA is not another inference server.
NERVA is an inference operating system.
```

The ground truth of the project is still what we wrote before: LLM inference should be rebuilt as a **latency-first memory-residency runtime**, where Transformer math stays the same but the execution machine around it changes. 

---

# What We Are Actually Building

Current inference engines treat the world like this:

```text
weights live on GPU
KV lives on GPU
CPU feeds GPU
GPU runs model
CPU receives token
repeat
```

That is too primitive. 

NERVA treats the world like this:

```text
VRAM = hot tier
DRAM = warm tier
disk = cold tier
CPU = control plane + warm compute
GPU = hot throughput plane
KV = virtual memory
weights = scheduled blocks
tokens = device/host state machine
```

That is the redesign. 

The core idea:

```text
The model is not loaded.
The model is scheduled.
```

The runtime asks:

```text
Which bytes are needed now?
Which bytes are needed soon?
Which bytes deserve VRAM?
Which bytes stay in DRAM?
Which bytes should CPU compute against directly?
Which bytes should be prefetched?
Which bytes should never move?
Which syncs are actually necessary?
```



The core object is:

```text
ResidentBlock
```

A `ResidentBlock` is any model/runtime object with explicit location, owner, lifetime, hotness, next-use distance, movement cost, and compute-near-data cost. 

---

# Rename The Repo Plan

Use:

```text
dacely-cloud/nerva
```

Not:

```text
ember
hylix
vexil
```

Those were not strong enough.

The first README line should be:

```text
NERVA is an inference operating system for AI models.
```

Not:

```text
NERVA is an LLM inference engine.
```

Because “engine” undersells it.

The category we are creating is:

```text
Inference OS
```

---

# Required Repos

Keep three codebases locally:

```text
/root/vllm
    Purpose:
        production baseline
        compatibility oracle
        vLLM behavior reference

/root/rvllm
    Purpose:
        Rust/CUDA architecture reference
        graph/runtime reference
        memory ownership reference

/root/nerva
    Purpose:
        actual new runtime
```

Do **not** make `/root/nerva` a fork.

NERVA is new architecture.

vLLM and rvLLM are things the agent studies.

---

# Agent Task: Audit vLLM And rvLLM Before Coding NERVA

Send the following to the coding agent.

```text
Before writing NERVA code, audit both vLLM and rvLLM.

This is mandatory.

Do not copy code blindly.
Do not optimize anything.
Do not write NERVA yet except maybe empty repo docs.

The goal is to understand:
    what vLLM already solved,
    what rvLLM already solved,
    what neither solved,
    and which code patterns NERVA should reuse or avoid.

Produce one report:

/root/llms/nerva/AUDIT_VLLM_RVLLM_20260626.md

The report must be technical, code-path-based, and include exact file paths, function names, call graphs, and conclusions.
```

---

# Why This Audit Matters

vLLM has the ecosystem.

vLLM supports broad serving features: Hugging Face model integration, parallel sampling, beam search, tensor/pipeline/data/expert/context parallelism, streaming, structured outputs, tool calling, OpenAI-compatible API, Anthropic Messages API, and gRPC. It also claims support for 200+ Hugging Face model architectures, including decoder-only LLMs, MoE models, hybrid attention/state-space models, multimodal models, embedding/retrieval models, and reward/classification models. ([GitHub][1])

rvLLM has the architecture smell we like.

Its README says it is an LLM inference engine using Rust+CUDA on GPU, JAX+XLA on TPU, and a native Rust binary on GPU with **zero Python in the serving path**. ([GitHub][2])

rvLLM’s workspace also has the kind of crate split we care about: core, memory, kernels, cutlass, attention, fused, metadata, graph, loader, sampling, runtime, serve, bench, deployment, invariants, and Metal crates. ([GitHub][3])

But neither is NERVA.

vLLM is too Python/PyTorch-centered.

rvLLM is too specialized around Gemma/FP8/H100-class paths. Its README describes the GPU path as Rust+CUDA on H100 using FP8 weights, FP8 or F16 paged KV, FA3 SM90 attention, split-KV FP8 decode, and all 60 layers captured in one CUDA graph. ([GitHub][2])

NERVA must learn from both and inherit neither blindly.

---

# Agent Audit Instructions — vLLM

Tell the agent:

```text
Audit vLLM as the production ecosystem/reference engine.

Repo:
    /root/vllm

Branch/commit:
    record current git commit

Main question:
    What does vLLM solve that NERVA should not rebuild first?
    What architectural assumptions does vLLM have that NERVA must avoid?
```

## vLLM Audit Section 1 — Process Architecture

Agent must inspect:

```text
vllm/v1/engine/core.py
vllm/v1/engine/utils.py
vllm/v1/worker/
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/gpu_model_runner.py
vllm/entrypoints/openai/api_server.py
```

Report:

```text
How API server, engine core, and GPU worker communicate.
Where request scheduling happens.
Where KV cache management happens.
Where model execution is dispatched.
Where GPU memory is owned.
Where Python is in the critical path.
```

Reference behavior: vLLM V1 uses an API server process for HTTP/input/streaming, an engine core process for scheduling and KV-cache coordination, and GPU worker processes that load weights, execute forward passes, and manage GPU memory. ([vLLM][4])

Agent must answer:

```text
Can NERVA reuse this process model?
Which parts are useful?
Which parts violate "Python may configure, Python may not be the machine"?
```

---

## vLLM Audit Section 2 — Model Runner

Agent must inspect:

```text
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu/
vllm/model_executor/
```

Report:

```text
Where model weights are loaded.
Where input tensors are prepared.
Where CUDA graphs are captured.
Where CUDA graphs are replayed.
Where sampling happens.
Where output handoff happens.
Where D2H token copies happen.
Where cudaEventSynchronize or output event waits happen.
```

vLLM docs say every worker has a model runner responsible for loading/running the model, preparing input tensors, and capturing CUDA graphs, and the model object under it is a `torch.nn.Module`. ([vLLM][4])

Agent must answer:

```text
Is the next token source of truth on host or device?
Can the sampled token be consumed by the next graph without host synchronization?
What exact object owns decode-step causality?
```

This is critical because the previous queue-depth experiment failed correctness. We need to know if vLLM’s architecture fundamentally requires host-visible token state.

---

## vLLM Audit Section 3 — CUDA Graphs

Agent must inspect:

```text
vllm/v1/worker/gpu_model_runner.py
vllm/compilation/
vllm/platforms/cuda.py
vllm/attention/
```

Report:

```text
How CUDA graph warmup works.
How graph capture is triggered.
Which shapes/modes are graphable.
Which parts remain outside the graph.
How many graph launches happen per token.
Where eager fallback happens.
```

vLLM docs note that CUDA Graph warm-up is controlled directly by the GPU model runner. ([vLLM][5])

Agent must answer:

```text
What would NERVA do differently?
Can NERVA make the entire synthetic decode transaction graph-owned from the start?
```

---

## vLLM Audit Section 4 — CustomOps And Kernels

Agent must inspect:

```text
vllm/csrc/
vllm/_custom_ops.py
vllm/model_executor/custom_op.py
vllm/attention/backends/
vllm/v1/attention/backends/
```

Report:

```text
How CustomOp registration works.
Which operations are custom ops.
Where RMSNorm, RoPE, MoE, attention, sampling, and KV kernels live.
Which kernels are CUDA/C++.
Which are Triton.
Which are torch ops.
Which fallbacks exist.
```

vLLM’s docs explain that implementing a new `CustomOp` means subclassing `CustomOp`, registering it with `@CustomOp.register(...)`, and implementing the relevant `forward_xxx()` methods. ([vLLM][6])

Agent must answer:

```text
Which vLLM custom ops are worth studying for NERVA?
Which custom ops are too PyTorch-bound?
Which kernels should NERVA not rewrite initially?
```

---

## vLLM Audit Section 5 — KV Cache

Agent must inspect:

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/worker/kv_connector*
vllm/attention/
vllm/v1/attention/
vllm/csrc/cache*
```

Report:

```text
How paged KV is represented.
Where block tables live.
Where KV pages are allocated.
Where KV writes happen.
Where prefix caching happens.
Whether KV can live outside VRAM.
Whether CPU/DRAM KV compute exists.
```

Agent must answer:

```text
What should NERVA steal conceptually?
What is missing for NERVA's tiered KV design?
```

NERVA’s KV goal is different: KV cache is virtual memory, with hot KV in VRAM, warm KV in DRAM, and exact partial attention merge later. This is part of the redesign described in the project notes. 

---

## vLLM Audit Section 6 — Allocations And Hot Path

Agent must inspect:

```text
gpu_model_runner.py
async_utils.py
sampling paths
KV allocation paths
input tensor preparation
```

Report:

```text
Where torch.empty / aten::empty happens.
Where pin_memory happens.
Where cudaMemcpyAsync happens.
Where D2D copies happen.
Where cudaEventSynchronize happens.
Which allocations are warmup-only.
Which allocations are per-token.
```

The vLLM baseline already showed allocator-like events and output-handoff sync evidence; this audit must find exact code ownership.

Agent must answer:

```text
Can vLLM be made allocation-free in decode without rewriting the runner?
If not, which assumptions block it?
```

---

## vLLM Audit Section 7 — What To Reuse

The agent must produce a table:

```text
vLLM component
path
purpose
reuse directly?
study only?
avoid?
reason
```

Minimum rows:

```text
scheduler
model runner
KV manager
PagedAttention concepts
CustomOp system
attention backends
CUDA graph handling
serving API
tokenizer/HF integration
benchmark tooling
```

---

# Agent Audit Instructions — rvLLM

Tell the agent:

```text
Audit rvLLM as the Rust/CUDA architecture reference.

Repo:
    /root/rvllm or clone https://github.com/m0at/rvllm

Branch/commit:
    record current git commit

Main question:
    What has rvLLM already solved about a Rust-owned GPU inference hot path?
    Which assumptions are too specialized for NERVA?
```

---

## rvLLM Audit Section 1 — Workspace And Crate Map

Agent must inspect:

```text
v3/Cargo.toml
v3/crates/
```

Report every crate:

```text
crate name
purpose
important files
dependencies
whether useful for NERVA
```

rvLLM’s `v3/Cargo.toml` shows a workspace with crates for core, memory, kernels, cutlass, attention, fused, metadata, graph, loader, sampling, runtime, serve, bench, deployment, invariants, vision, image IO, and Metal. ([GitHub][3])

Agent must answer:

```text
Which crate structure should NERVA copy?
Which crates are too early for NERVA?
Which crates solve problems NERVA should postpone?
```

---

## rvLLM Audit Section 2 — Engine Owner Thread

Agent must find:

```text
Where is the single engine owner thread?
Where is CUDA context owned?
Where are streams owned?
Where is graph replay called?
Where does request input enter the engine?
Where does output leave the engine?
```

Report exact files/functions.

Agent must answer:

```text
Does rvLLM have a better ownership model than vLLM?
Can NERVA copy this pattern?
```

---

## rvLLM Audit Section 3 — Memory System

Agent must inspect:

```text
rvllm-mem
arena code
pinned buffer code
device allocation code
KV allocation code
loader memory path
```

Report:

```text
Does rvLLM use static arenas?
Does it have HBM/device arenas?
Does it checkpoint/restore arenas?
Does decode allocate?
Does it track hot-path allocation?
Does it use pinned host memory?
Does it mmap weights?
Does it pretransform weights?
```

The rvLLM README mentions a previous perplexity harness bug involving repeated multi-GB paged KV allocation and later fixing it with arena checkpoint/restore, which makes this especially important to inspect. ([GitHub][2])

Agent must answer:

```text
Which memory ownership ideas should NERVA steal?
Does rvLLM have anything like ResidentBlock?
If not, where would ResidentBlock fit?
```

---

## rvLLM Audit Section 4 — CUDA Graph Path

Agent must inspect:

```text
rvllm-graph
rvllm-runtime
graph_executor
decode graph capture
graph replay callsites
```

Report:

```text
How graphs are captured.
How many graphs exist.
Whether graph per chunk size exists.
How graph inputs are updated.
How token state flows between graph replays.
Whether host synchronizes per token.
```

rvLLM README claims its 31B GPU path captures all 60 layers in a single CUDA graph. ([GitHub][2])

Agent must answer:

```text
How does rvLLM avoid Python/PyTorch launch overhead?
What is the minimum graph-executor pattern NERVA should implement first?
```

---

## rvLLM Audit Section 5 — Token State And Sampling

Agent must inspect:

```text
rvllm-sampling
runtime decode loop
token output path
EOS/stop path
speculative decoding path if present
```

Report:

```text
Is sampling on GPU or CPU?
Where does sampled token live first?
Is next token device-resident?
Does host visibility block next decode?
How is greedy correctness verified?
How are token hashes checked?
How is speculative decode validated?
```

The README states that speculative decoding verifies multiple draft tokens in one forward, that greedy acceptance emits model argmax tokens, and that some graph/eager paths are token-hash checked. ([GitHub][2])

Agent must answer:

```text
Does rvLLM already implement the device-token-ring concept?
If yes, how?
If no, what is closest?
```

---

## rvLLM Audit Section 6 — Kernels And Vendor Libraries

Agent must inspect:

```text
rvllm-kernels
rvllm-cutlass
rvllm-attention
rvllm-fused
kernels/
cutlass submodule
```

Report:

```text
Which kernels are custom.
Which use cuBLASLt.
Which use CUTLASS.
Which are FP8-specific.
Which are SM90-specific.
Which would fail on RTX 5090 / 4090 / 3090 / 2080 Ti.
```

The rvLLM README explicitly says small-M GEMMs are routed through cuBLASLt, that custom FP8 GEMV lost to cuBLASLt, and that a megakernel attempt was slower. ([GitHub][2])

Agent must answer:

```text
Which kernel decisions are generally useful?
Which are H100/FP8/Gemma-specific?
What should NERVA avoid rewriting?
```

---

## rvLLM Audit Section 7 — Model Specificity

Agent must inspect:

```text
loader
metadata
Gemma-specific code
model configs
tokenizer handling
weight naming
RoPE/sliding attention assumptions
MoE assumptions
```

Report:

```text
Which parts are Gemma-specific?
Which parts generalize?
How hard would Qwen support be?
How hard would Llama support be?
Does rvLLM assume FP8?
Does it support BF16/FP16 exact paths?
```

Agent must answer:

```text
Is rvLLM a reusable base or a specialized reference?
```

Expected likely conclusion:

```text
rvLLM is the better architecture reference.
vLLM is the better compatibility reference.
NERVA should be new.
```

---

# Combined Audit Output Format

The final audit report must have these sections:

```text
# NERVA Audit: vLLM and rvLLM

## Executive Conclusion

## vLLM Summary
    What vLLM solves
    What vLLM assumes
    What NERVA should reuse
    What NERVA should avoid

## rvLLM Summary
    What rvLLM solves
    What rvLLM assumes
    What NERVA should reuse
    What NERVA should avoid

## Code Path Tables

## Hot Path Call Graphs

## Memory Ownership Comparison

## Token State Comparison

## CUDA Graph Comparison

## KV Cache Comparison

## Kernel Strategy Comparison

## Serving/Compatibility Comparison

## Final Recommendation
```

The report must include this table:

```text
Area | vLLM | rvLLM | NERVA decision
```

Rows:

```text
runtime language
hot path owner
GPU execution
CUDA graphs
memory arenas
KV cache
token state
sampling
model loading
kernel strategy
serving API
model coverage
old hardware viability
exact FP16/BF16 viability
DRAM warm-tier compute
ResidentBlock compatibility
```

---

# Agent Must Answer These Exact Questions

```text
1. Where does vLLM decide the next input token?
2. Is vLLM’s next token host-owned or device-owned?
3. Where exactly does vLLM synchronize host output?
4. Can vLLM decode continue without host-visible token state?
5. Where does vLLM allocate during decode?
6. Which vLLM allocations are real and which are profiler/ATen artifacts?
7. How does rvLLM keep Python out of the hot path?
8. How does rvLLM own CUDA streams/graphs?
9. Does rvLLM have a static arena model?
10. Does rvLLM have anything equivalent to ResidentBlock?
11. Does rvLLM’s graph design generalize beyond Gemma/FP8/H100?
12. Which rvLLM code is useful for RTX 5090?
13. Which rvLLM code would fail or underperform on 2080 Ti?
14. What should NERVA copy conceptually?
15. What should NERVA never inherit?
```

---

# Agent Must Not Do This During Audit

```text
Do not write NERVA implementation code yet.
Do not start with model loading.
Do not start with CUDA kernels.
Do not port rvLLM code blindly.
Do not patch vLLM.
Do not benchmark random things.
Do not make performance claims without artifact paths.
```

Only audit and report.

---

# After Audit: NERVA Bootstrap Instructions

Once the audit is done, the coding agent starts NERVA.

Use this final bootstrap instruction.

```text
Create /root/nerva.

NERVA means Neural Execution & Residency Virtual Architecture.

NERVA is an inference operating system for AI models.

The goal is to build a Rust/CUDA runtime skeleton for single-GPU inference where memory residency and token latency are first-class objects.

Do not fork vLLM.
Do not fork rvLLM.
Do not depend on PyTorch.
Do not use Python in the hot path.
Do not implement real model inference yet.

Initial structure:

nerva/
  README.md
  ARCHITECTURE.md
  ROADMAP.md
  CONTRIBUTING.md
  LICENSE
  Cargo.toml
  rust-toolchain.toml
  .gitignore

  crates/
    nerva-core/
    nerva-ledger/
    nerva-memory/
    nerva-cuda/
    nerva-runtime/
    nerva-bench/

  native/
    cuda/
      CMakeLists.txt
      nerva_cuda_api.h
      nerva_cuda_api.cpp
      smoke_kernel.cu
      graph_executor.cu
      token_ring_kernels.cu
      synthetic_decode.cu

  docs/
    design/
      resident_block.md
      memory_tiers.md
      token_state_machine.md
      static_arenas.md
      cuda_graph_executor.md
      token_ledger.md
      no_python_hot_path.md
      vllm_baseline.md
      rvllm_notes.md

  benches/
    single_gpu/
      README.md

  tools/
    compare_ledgers/

Initial command:

cargo run -p nerva-bench -- smoke

Smoke must:
    initialize CUDA device 0
    print GPU name
    print CUDA driver/runtime version
    allocate device arena
    allocate pinned host arena
    launch one smoke kernel
    emit summary JSON
    report hot_path_allocations = 0

Do not implement:
    real model loading
    networking
    multi-GPU
    OpenAI server
    quantization
    KV cache beyond placeholder types
    PyTorch integration
    vLLM integration

Commit:
    M0: bootstrap NERVA Rust CUDA smoke runtime
```

---

# Final Naming Decision

Use **NERVA**.

Full expansion:

```text
Neural Execution & Residency Virtual Architecture
```

Short description:

```text
NERVA is an inference operating system for AI models.
```

Long description:

```text
NERVA rebuilds LLM inference around memory residency, device-first token state, static arenas, and explicit CPU/GPU execution policy.
```

One-liner:

```text
NERVA makes inference scheduled, not loaded.
```

This name actually says what we are building.

[1]: https://github.com/vllm-project/vllm "GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs · GitHub"
[2]: https://raw.githubusercontent.com/m0at/rvllm/main/README.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/m0at/rvllm/main/v3/Cargo.toml "raw.githubusercontent.com"
[4]: https://docs.vllm.ai/en/latest/design/arch_overview/ "Architecture Overview - vLLM"
[5]: https://docs.vllm.ai/en/latest/design/cuda_graphs/ "CUDA Graphs - vLLM"
[6]: https://docs.vllm.ai/en/latest/design/custom_op/ "CustomOp - vLLM"

Good. **NERVA** is the name.

Now add AMD and DPDK support to the plan, but as **architecture requirements**, not first-code distractions.

The correct rule is:

```text id="l2zisu"
NERVA starts single-GPU NVIDIA/CUDA only so the base runtime exists.

But NERVA must be architected from day one so:
    AMD/HIP can be added without rewriting the runtime,
    DPDK can be added later without rewriting the transport/runtime model.
```

So: **design for AMD and DPDK now, implement later**.

---

# Add To NERVA Architecture

Add this section to `ARCHITECTURE.md`.

```text id="gglzf6"
## Backend Portability

NERVA is not a CUDA-only architecture.

The first implementation target is CUDA because the current benchmark machine is NVIDIA RTX 5090 and the existing vLLM baseline is CUDA-based.

However, the runtime architecture must not assume CUDA as the only execution backend.

NERVA has three layers:

1. Runtime layer:
   Rust-owned scheduling, memory residency, token state, arenas, ledgers.

2. Device backend layer:
   CUDA initially.
   HIP/ROCm later.
   Other backends later only if justified.

3. Transport/backend layer:
   local process only initially.
   DPDK, RDMA, UCX, and networking later.

The Rust runtime must depend on backend traits, not CUDA-specific concepts directly, except inside the `nerva-cuda` crate.
```

---

# Add Backend Trait Requirement

In `nerva-core` or later `nerva-runtime`, define the idea now, even if CUDA is the only implementation.

```rust id="dwz0y8"
pub trait DeviceBackend {
    type Device;
    type Stream;
    type Event;
    type GraphExec;
    type DeviceArena;
    type PinnedArena;

    fn init(device_id: u32) -> Result<Self::Device, BackendError>;

    fn alloc_device_arena(
        device: &Self::Device,
        bytes: usize,
    ) -> Result<Self::DeviceArena, BackendError>;

    fn alloc_pinned_arena(
        bytes: usize,
    ) -> Result<Self::PinnedArena, BackendError>;

    fn capture_synthetic_decode_graph(
        device: &Self::Device,
        arena: &Self::DeviceArena,
    ) -> Result<Self::GraphExec, BackendError>;

    fn replay_graph(
        graph: &Self::GraphExec,
    ) -> Result<(), BackendError>;
}
```

Initial implementation:

```text id="x5hn2h"
nerva-cuda implements DeviceBackend
```

Future implementation:

```text id="8t0upq"
nerva-hip implements DeviceBackend
```

Do **not** put CUDA types into `ResidentBlock`.

Bad:

```rust id="ileh5e"
Location::CudaDevicePtr(...)
```

Good:

```rust id="yi7bw2"
Location::Vram {
    backend: BackendKind,
    device_id: DeviceId,
    ptr: DevicePtr,
}
```

Where:

```rust id="feff4n"
pub enum BackendKind {
    Cuda,
    Hip,
    Cpu,
}
```

---

# Add AMD/HIP Support Requirement

Add this to `docs/design/backend_portability.md`.

```text id="75pm93"
# Backend Portability

NERVA must support AMD eventually.

CUDA is only the first backend.

Do not hardcode NVIDIA-only assumptions into core crates.

Allowed in `nerva-cuda`:
    CUDA streams
    CUDA events
    CUDA graphs
    cudaMalloc
    cudaHostAlloc
    CUDA driver/runtime queries

Forbidden outside `nerva-cuda`:
    raw cudaStream_t
    raw cudaEvent_t
    raw cudaGraphExec_t
    CUDA-specific pointer types
    CUDA-specific error codes
    NVIDIA-specific device properties

Future HIP backend:
    `nerva-hip`

HIP equivalent concepts:
    HIP device
    HIP stream
    HIP event
    HIP graph if available/suitable
    hipMalloc
    hipHostMalloc
    ROCm profiling hooks

The runtime must treat CUDA and HIP as implementations of the same abstract device contract.
```

---

# Add DPDK Support Requirement

DPDK should **not** be implemented during single-GPU bootstrap.

But we should reserve the architecture.

Add this to `docs/design/future_transport.md`.

```text id="05afsm"
# Future Transport Layer

NERVA starts as a single-process, single-GPU runtime.

Networking is out of scope for the initial implementation.

However, the runtime must be designed so future multi-node NERVA can support:

- RDMA
- UCX
- DPDK UDP
- pinned-host transport
- GPU-direct transport when hardware supports it

The hot transport path must not use TCP.

TCP may be used only for:
    debug
    control plane
    configuration
    non-hot administrative messages

The future tensor data plane should use:
    RDMA where available
    DPDK UDP where custom packet control is required
    kernel UDP only for testing
```

Add:

```text id="m4s9kt"
Future crate:
    nerva-transport

Future transport implementations:
    nerva-transport-rdma
    nerva-transport-dpdk
    nerva-transport-udp-test
```

But do not create these crates yet.

---

# DPDK Design Notes To Add

Add this to the future transport doc:

```text id="9i0sg6"
## DPDK Position

DPDK is not the first implementation target.

DPDK is a future transport backend for multi-node NERVA.

DPDK is useful because:
    it bypasses the kernel network stack,
    gives userspace poll-mode packet IO,
    allows custom UDP-style tensor transport,
    avoids TCP stream semantics,
    supports ConnectX NICs through mlx5 PMD.

DPDK does not automatically solve GPU-to-NIC memory movement.

For GPU-resident tensors, NERVA must still distinguish:
    GPU-direct NIC access,
    GPU -> pinned host -> NIC fallback,
    CPU-produced tensor directly in pinned host memory.

DPDK is a packet transport layer.
It is not a substitute for a memory residency planner.
```

---

# Add Transport Trait Design

Do not implement yet, but define future shape in docs.

```rust id="pgthhh"
pub trait TensorTransport {
    type Endpoint;
    type Buffer;

    fn register_buffer(
        &self,
        buffer: TransportBuffer,
    ) -> Result<RegisteredBuffer, TransportError>;

    fn send_tensor(
        &self,
        dst: &Self::Endpoint,
        tensor: TensorMessage,
    ) -> Result<(), TransportError>;

    fn poll_recv(
        &self,
    ) -> Result<Option<TensorMessage>, TransportError>;
}
```

Buffer types:

```rust id="s82r6b"
pub enum TransportBuffer {
    Device {
        backend: BackendKind,
        device_id: DeviceId,
        ptr: DevicePtr,
        bytes: usize,
    },
    PinnedHost {
        ptr: HostPinnedPtr,
        bytes: usize,
    },
}
```

Future transport backends:

```text id="ujvwlc"
RdmaGpuDirect
RdmaPinnedHost
DpdkUdpGpu
DpdkUdpPinnedHost
KernelUdpTest
TcpDebugOnly
```

Rules:

```text id="j5rhdj"
TCP is not a production tensor transport.
DPDK is allowed only for future multi-node data plane.
No networking in M0/M1/M2.
```

---

# Update Initial Repo Layout

Current initial repo should stay small.

Add docs only:

```text id="lah7tw"
docs/design/backend_portability.md
docs/design/future_transport.md
```

Do **not** add these crates yet:

```text id="yg4iry"
nerva-hip
nerva-transport
nerva-transport-dpdk
nerva-transport-rdma
```

Why?

Because the agent will waste time scaffolding fake crates. We need the CUDA single-GPU machine first.

The implementation order remains:

```text id="qgagwi"
BOOT
CUDA-SMOKE
ARENA
LEDGER
SYNTH-GRAPH
DEVICE-TOKEN
```

AMD and DPDK are architecture constraints until the base runtime exists.

---

# Updated Agent Audit Instructions

Add this to the audit brief for vLLM and rvLLM.

```text id="kvf9rh"
Additional audit requirements:

AMD/HIP:
    Check how vLLM abstracts CUDA vs HIP.
    Find platform/backend files responsible for CUDA/HIP selection.
    Identify which abstractions NERVA should copy.
    Identify CUDA assumptions NERVA must avoid.

    Check rvLLM for CUDA-only assumptions.
    Identify whether rvLLM has HIP/ROCm support.
    Identify which parts would be hard to port to AMD.

DPDK/transport:
    Do not implement networking.
    Only inspect whether either project has transport abstractions relevant to future multi-node inference.
    Check whether vLLM distributed code assumes NCCL/Ray/PyTorch distributed.
    Check whether rvLLM has swarm/network/distributed code.
    Report what should be ignored for now.
```

---

# Updated Bootstrap Agent Prompt

Use this final version.

```text id="cvztx0"
Create a new project called NERVA.

NERVA means Neural Execution & Residency Virtual Architecture.

NERVA is an inference operating system for AI models.

The model is not loaded. The model is scheduled.

Do not fork vLLM.
Do not fork rvLLM.
Do not depend on PyTorch.
Do not use Python in the hot path.
Do not implement real model inference yet.
Do not implement networking yet.
Do not implement AMD/HIP yet.
Do not implement DPDK yet.

But the architecture must not block future AMD/HIP or DPDK support.

Create this structure:

nerva/
  README.md
  ARCHITECTURE.md
  ROADMAP.md
  CONTRIBUTING.md
  LICENSE
  Cargo.toml
  rust-toolchain.toml
  .gitignore

  crates/
    nerva-core/
    nerva-ledger/
    nerva-memory/
    nerva-cuda/
    nerva-runtime/
    nerva-bench/

  native/
    cuda/
      CMakeLists.txt
      nerva_cuda_api.h
      nerva_cuda_api.cpp
      smoke_kernel.cu
      graph_executor.cu
      token_ring_kernels.cu
      synthetic_decode.cu

  docs/
    design/
      resident_block.md
      memory_tiers.md
      token_state_machine.md
      static_arenas.md
      cuda_graph_executor.md
      token_ledger.md
      no_python_hot_path.md
      backend_portability.md
      future_transport.md
      vllm_baseline.md
      rvllm_notes.md

  benches/
    single_gpu/
      README.md

  tools/
    compare_ledgers/

Architecture rules:

Rust owns:
- request state
- scheduler
- memory residency planner
- ResidentBlock table
- CPU arena
- pinned DRAM arena
- token state machine
- token latency ledger
- prefetch/eviction policy later

C++/CUDA owns:
- CUDA initialization
- CUDA streams/events/graphs
- device memory allocation
- pinned memory allocation
- GPU kernels
- device token ring
- graph replay

Core crates must not hardcode CUDA-only types.
CUDA-specific raw types must remain inside nerva-cuda.

The architecture must allow a future nerva-hip backend.

The architecture must allow a future nerva-transport backend with RDMA and DPDK UDP.

The hot path must not call:
- malloc
- free
- cudaMalloc
- cudaFree
- cudaHostAlloc
- mmap
- munmap
- pin/unpin registration
- Python
- PyTorch
- pageable H2D/D2H copies
- cudaDeviceSynchronize
- global locks
- unbounded queues

Initial deliverable:

Implement BOOT and CUDA-SMOKE.

Required command:

cargo run -p nerva-bench -- smoke

Smoke must:
- initialize CUDA device 0
- print GPU name
- print CUDA driver/runtime version
- allocate device arena
- allocate pinned host arena
- launch one smoke kernel
- emit summary JSON
- report hot_path_allocations = 0

Do not implement:
- real model loading
- networking
- multi-GPU
- OpenAI server
- quantization
- KV cache beyond placeholder types
- PyTorch integration
- vLLM integration
- HIP backend
- DPDK backend

Commit:
M0: bootstrap NERVA Rust CUDA smoke runtime
```

---

# Updated README Text

Use this:

```text id="inxt65"
# NERVA

NERVA means Neural Execution & Residency Virtual Architecture.

NERVA is an inference operating system for AI models.

The model is not loaded. The model is scheduled.

NERVA rebuilds inference around:
- memory residency
- static arenas
- device-first token state
- explicit CPU/GPU execution policy
- per-token latency ledgers

Initial scope:
- one CPU
- one DRAM pool
- one disk
- one GPU
- one VRAM pool
- Rust runtime
- CUDA backend

Future architecture:
- AMD/HIP backend
- DPDK UDP transport
- RDMA transport
- multi-GPU
- distributed inference

NERVA is not a vLLM fork.
NERVA is not an rvLLM fork.

vLLM is used as a baseline and compatibility oracle.
rvLLM is used as a Rust/CUDA architecture reference.
NERVA is a new runtime.
```

---

# Updated Architecture Statement

Add to `ARCHITECTURE.md`:

```text id="95o66f"
## Future Backend Requirements

NERVA starts with CUDA because the first benchmark machine is NVIDIA.

NERVA must not become CUDA-only.

The core runtime must be backend-neutral.

CUDA-specific details live in:
    nerva-cuda

Future AMD-specific details live in:
    nerva-hip

The runtime sees both through a device backend contract.

NERVA starts single-node and local-only.

NERVA must not become TCP-stream based.

Future tensor transport backends:
    RDMA
    DPDK UDP
    kernel UDP test path
    TCP debug/control only

No transport code is implemented in the bootstrap.
The architecture simply avoids blocking it.
```

---

# Final Position

NERVA is good.

The project definition is now:

```text id="u5y6mh"
NERVA is an inference operating system for AI models,
built around memory residency, device-first token state,
static arenas, backend portability, and token-level observability.
```

CUDA first.

AMD later.

No TCP hot path later.

DPDK later.

But first:

```text id="yy0cdx"
Rust owns CUDA.
Rust owns memory.
Rust owns token state.
Rust emits the ledger.
CUDA executes a graph.
No allocation in the hot path.
```

That is the base.
