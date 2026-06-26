# Addendum — Inter-System GPU/NIC Communication and the VRAM → NIC Problem

## Core problem

In the 4-system / 32-GPU design, we said:

```text
Move activations between systems.
Do not move weights.
```

That is correct.

But we missed a critical detail:

```text
Where does the activation live at the moment we need to send it?
```

If the stage output activation lives in GPU VRAM, and the NIC cannot directly read GPU memory, the path becomes:

```text
GPU VRAM
  → host pinned DRAM
  → NIC
  → network
  → remote host DRAM
  → remote GPU VRAM
```

That is much worse than:

```text
GPU VRAM
  → NIC
  → network
  → remote GPU VRAM
```

The first path consumes PCIe bandwidth twice, touches DRAM, may need staging buffers, and adds synchronization. The second path is direct GPU/NIC DMA.

So the communication section must distinguish:

```text
activation transfer size
activation source location
activation destination location
NIC/GPU direct-memory support
PCIe topology
driver path
fallback path
```

---

## The technology we want: GPU-direct RDMA

On NVIDIA, the relevant feature is **GPUDirect RDMA**.

NVIDIA describes GPUDirect RDMA as a direct path for data exchange between a GPU and a third-party PCIe peer device such as a network interface, storage adapter, or video device. NVIDIA’s docs say it allows the NIC/HCA to read or write peer GPU memory buffers without copying through host memory, and ConnectX-4 or later adapters support the capability. ([NVIDIA Docs][1])

So with GPUDirect RDMA, the ideal inter-system path is:

```text
stage N output tensor in GPU VRAM
    ↓
ConnectX NIC reads GPU memory directly
    ↓
InfiniBand or RoCE network
    ↓
remote ConnectX NIC writes directly to remote GPU memory
    ↓
next stage consumes tensor from VRAM
```

This avoids the CPU DRAM bounce.

NVIDIA also says best GPUDirect RDMA performance requires the GPU and HCA/NIC to be physically under the same PCIe IO root complex. ([NVIDIA Docs][2])

That topology constraint matters a lot for a machine with 8 GPUs and maybe one or two NICs.

---

## ConnectX-6 is a good target NIC

ConnectX-6 VPI is a realistic target for this project. NVIDIA’s ConnectX-6 VPI docs describe up to two ports of 200 Gb/s for InfiniBand and Ethernet, sub-600 ns latency, and roughly 200 million messages/sec. ([NVIDIA Docs][3])

That means, in raw bandwidth terms:

```text
200 Gb/s ≈ 25 GB/s raw
real application payload usually lower
```

For decode-stage activation transfer, this can be enough.

Example:

```text
hidden_size = 16,384
dtype = FP16/BF16 = 2 bytes

one token hidden activation:
16,384 × 2 = 32 KB
```

Even across three system boundaries:

```text
3 × 32 KB = 96 KB per generated token
```

The network serialization cost is tiny compared to moving model weights.

For prefill, the tensor is much bigger:

```text
sequence_length × hidden_size × dtype_size
```

Example:

```text
8192 × 16,384 × 2 bytes ≈ 256 MB per stage boundary
```

That is where chunking, overlap, and pipeline scheduling matter.

---

## InfiniBand vs RoCE

InfiniBand is the classic HPC RDMA fabric.

RoCE is RDMA over Converged Ethernet.

ConnectX-6 VPI supports both InfiniBand and Ethernet/RoCE-style deployments. NVIDIA’s GPUDirect RDMA user manual says GPUDirect RDMA works with ConnectX-4 and later InfiniBand adapters, and also works using RoCE with ConnectX-4 and later. ([NVIDIA Docs][4])

For our runtime:

```text
InfiniBand path:
    lower-friction HPC path
    best latency characteristics
    common with MPI/UCX/NCCL/NVSHMEM

RoCE path:
    Ethernet-based
    can be cheaper/flexible
    needs good switch/NIC configuration
    congestion control and lossless Ethernet tuning matter
```

Either is viable. InfiniBand is usually the cleaner engineering target if the cluster is dedicated.

---

## Important NVIDIA caveat: 2080 Ti / GeForce support

This is the painful part.

NVIDIA’s low-level GPUDirect RDMA CUDA documentation says GPUDirect RDMA is available on **Tesla and Quadro GPUs**. ([NVIDIA Docs][1]) Older NVIDIA/Mellanox GPUDirect RDMA material similarly lists Tesla/Quadro GPU families. ([NVIDIA][5])

Modern NVIDIA GPU Operator docs describe GPUDirect RDMA support through either DMA-BUF or legacy `nvidia-peermem`, and list GPU prerequisites including Turing data-center, Quadro RTX, and RTX-class GPUs or higher, depending on the path. ([NVIDIA Docs][6])

For **GeForce RTX 2080 Ti specifically**, I would not assume official GPUDirect RDMA support. It may fail, or it may silently fall back to a host-bounce path depending on the stack. This has to be tested directly.

Required validation:

```text
allocate CUDA buffer
register it with RDMA stack / UCX / ibverbs
send GPU buffer directly
verify whether transfer uses GDR path or host bounce
measure with:
    nvidia-smi topo -m
    ib_write_bw / perftest CUDA mode
    UCX logs
    NCCL_DEBUG=INFO
    PCIe counters
    DRAM bandwidth counters
```

Project rule:

```text
Do not design the 2080 Ti path assuming GPUDirect RDMA works.
Design with a direct path if available, and a fast pinned-host fallback if not.
```

---

## Modern NVIDIA paths to support

The NVIDIA communication stack has several relevant layers.

### GPUDirect RDMA

Purpose:

```text
NIC directly reads/writes GPU memory.
Avoid host DRAM bounce.
```

This is the main thing we want.

### CUDA-aware MPI / UCX

Open MPI documents CUDA-aware support as allowing MPI libraries to send and receive GPU buffers directly, commonly through UCX. ([Open MPI Documentation][7])

For our runtime, UCX is probably a better low-level target than hand-writing everything immediately, because it can select between:

```text
GPU-direct RDMA
CUDA IPC
host-staged transfer
shared memory
TCP fallback
```

### NCCL

NCCL is useful for collectives, but our design should avoid global collectives across all systems whenever possible. We mostly need point-to-point stage transfers, not all-reduce everywhere.

### NVSHMEM / GPU-initiated communication

NVSHMEM supports GPU-initiated communication, which can reduce communication and synchronization overhead. NVIDIA’s NVSHMEM docs say GPU-initiated communication can reduce communication/synchronization overhead and improve strong scaling. ([NVIDIA Docs][8])

This is interesting later, but it may be overkill for Phase 1. For the first implementation, host-orchestrated RDMA/UCX is simpler. Later, GPU-initiated communication could remove CPU from the fast path.

### GPUDirect Storage

Not the same as network RDMA, but relevant for model loading/offload. NVIDIA says GPUDirect Storage creates a direct path between local/remote storage and GPU memory, avoiding CPU bounce buffers. ([NVIDIA Developer][9])

For our 800 GB model design, GDS is useful for cold model staging, not for inter-system activation transfer.

---

## AMD equivalent

AMD does have a direct GPU/NIC RDMA path.

AMD ROCm documentation says the AMD kernel driver exposes RDMA through PeerDirect interfaces, allowing NICs to directly read and write RDMA-capable GPU device memory for high-speed DMA transfers between GPU and NIC. AMD also documents ROCm-aware Open MPI over InfiniBand and RoCE using UCX. ([ROCm Documentation][10])

So the portable design should support:

```text
NVIDIA:
    CUDA + GPUDirect RDMA + UCX/NCCL/NVSHMEM

AMD:
    ROCm/HIP + PeerDirect RDMA + UCX/RCCL/MPI

Fallback:
    pinned host DRAM bounce buffers
```

This is exactly why the transport abstraction matters.

---

## The fallback path if GPUDirect RDMA is unavailable

If GeForce 2080 Ti cannot do GPUDirect RDMA, the best fallback is not naive TCP and pageable memory.

The fallback should be:

```text
GPU VRAM
    → async D2H copy into pre-registered pinned host buffer
    → NIC RDMA send/write from pinned host buffer
    → remote pinned host buffer
    → async H2D copy into remote GPU VRAM
```

This still uses host DRAM, but avoids CPU memcpy.

Bad fallback:

```text
cudaMemcpy to pageable memory
normal socket send
kernel TCP stack
remote socket recv
cudaMemcpy to GPU
```

Better fallback:

```text
cudaMemcpyAsync to pinned ring buffer
RDMA from registered host memory
double/triple buffering
overlap transfer with local compute
remote async H2D into preallocated GPU buffer
```

The decode activation is small enough that this may still be acceptable.

The prefill activation is large enough that it must be chunked and overlapped.

---

## Transport decision matrix

The runtime should choose the transfer path dynamically.

```text
Path A: GPU → NIC → remote GPU
    Use when:
        GPUDirect RDMA / ROCm PeerDirect works
        GPU and NIC topology is good
        tensor is already in VRAM
    Best for:
        stage-boundary activations
        hot tensor transfer
        low-latency decode

Path B: GPU → pinned DRAM → NIC → remote pinned DRAM → GPU
    Use when:
        GPU-direct RDMA unavailable
        consumer GPU path blocked
        topology prevents GPU/NIC P2P
    Best for:
        fallback compatibility
        small decode activations
        overlapped prefill chunks

Path C: CPU computes boundary output directly into pinned DRAM
    Use when:
        final part of stage is CPU-resident
        result is small
        avoiding D2H is better
    Best for:
        host-resident weight shards
        CPU partial attention
        CPU-side logits/sampling paths

Path D: disk/storage → GPU direct
    Use when:
        GPUDirect Storage available
        preloading large cold weights
    Best for:
        model staging, not token-critical network transfer
```

---

## Topology rule for 8 GPUs + ConnectX-6

A machine with 8 GPUs should not blindly use one NIC for everything.

The runtime must inspect topology:

```text
nvidia-smi topo -m
lspci -tv
numactl --hardware
ibdev2netdev
UCX device locality
PCIe root complex
PCIe switch layout
```

The best layout is:

```text
GPU producing stage output
    close to NIC under same PCIe switch/root complex
```

Worst layout:

```text
GPU output produced on GPU behind CPU socket A
NIC attached to CPU socket B
transfer crosses inter-socket link
then host DRAM bounce
then NIC
```

That can destroy latency.

Runtime rule:

```text
The GPU that produces the inter-system boundary tensor should be topology-near the NIC.
```

This may mean assigning the final layer/tile of each stage to the GPU closest to the NIC.

---

## Key redesign addition

The distributed pipeline should not merely partition layers.

It should also partition **egress responsibility**.

Each stage should have:

```text
compute GPUs
KV GPUs
egress GPU
NIC-near staging buffers
host pinned fallback buffers
transport worker
```

Example:

```text
System 1:
    GPUs 0–6 compute most local layers
    GPU 7 is NIC-near and owns final stage output projection/packing
    ConnectX-6 sends activation to System 2
```

Why?

Because if GPU 2 computes the boundary tensor but GPU 7/NIC are topology-near, you may create an internal copy or P2P transfer. Better to schedule the last local layer so the boundary tensor is born near the NIC.

Rule:

```text
Do not only ask “which GPU is fastest?”
Ask “which GPU is closest to where the data must go next?”
```

---

## Communication-aware stage design

Add this to HILO-DP:

```text
Each stage has an ingress tensor and egress tensor.

Ingress:
    receive activation from previous system
    place it directly in the memory tier needed by first local layer

Egress:
    produce activation in NIC-near GPU memory if GDR is available
    otherwise produce/copy into pinned host egress ring
```

For decode:

```text
boundary tensor small
optimize latency and sync count
```

For prefill:

```text
boundary tensor large
optimize bandwidth and chunk overlap
```

So the runtime should use two different communication modes:

```text
decode mode:
    tiny activation
    low latency
    pre-post receives
    persistent registered buffers
    no allocation
    no handshake per token

prefill mode:
    chunked activation blocks
    streaming
    overlap stage compute with network transfer
    pipeline chunks across systems
```

---

## Updated distributed dataflow

### Best case: GPU-direct RDMA available

```text
System 1 GPU computes boundary activation in VRAM
    ↓
NIC directly reads GPU buffer via GPUDirect RDMA / PeerDirect
    ↓
InfiniBand/RoCE
    ↓
remote NIC writes directly into System 2 GPU buffer
    ↓
System 2 GPU starts next stage
```

### Fallback case: no GPU-direct RDMA

```text
System 1 GPU computes boundary activation in VRAM
    ↓ async copy
pinned host egress ring
    ↓ RDMA
remote pinned host ingress ring
    ↓ async copy
System 2 GPU VRAM
```

The fallback is worse, but not fatal for decode activations if overlapped and buffer sizes are small.

---

## Important implication for 2080 Ti cluster

For your exact 32x 2080 Ti idea:

```text
GPUDirect RDMA may not be officially supported.
ConnectX-6 itself is capable.
The NVIDIA software/GPU support may be the limiting factor.
```

So the runtime should be designed like this:

```text
if GDR works:
    use GPU → NIC direct path

if GDR does not work:
    use pinned-host RDMA bounce path

if CPU computes the boundary:
    send directly from pinned host memory

if tensor is large:
    chunk and overlap

if tensor is small:
    prioritize latency and avoid sync
```

This keeps the architecture viable even without enterprise GPU features.

---

## Phase 1 conclusion update

Add this to the final thesis:

```text
The multi-system pipeline is only efficient if communication is memory-location-aware.

Moving activations instead of weights solves the largest data-movement problem,
but the runtime must still avoid unnecessary VRAM → DRAM → NIC bounce copies.

Therefore HILO-DP needs a transport layer that supports:
    GPUDirect RDMA on NVIDIA when available,
    PeerDirect RDMA on AMD/ROCm when available,
    UCX/MPI/NCCL/RCCL-compatible GPU buffers,
    pinned-host RDMA fallback,
    topology-aware egress GPU selection,
    pre-registered communication buffers,
    separate decode and prefill transfer modes.
```

And the new rule:

```text
The stage boundary tensor must be born as close as possible to the next transport device.
```

That means:

```text
If NIC can read GPU memory:
    produce boundary tensor in NIC-near GPU VRAM.

If NIC cannot read GPU memory:
    produce or copy boundary tensor into pre-registered pinned DRAM with overlap.

If CPU already owns the result:
    never move it to GPU just to move it back to NIC.
```

This is a major part of the hybrid redesign.

[1]: https://docs.nvidia.com/cuda/gpudirect-rdma/ "1. Overview — GPUDirect RDMA 13.3 documentation"
[2]: https://docs.nvidia.com/networking/display/GPUDirectRDMAv18/System%2BRequirements%2Band%2BRecommendations "System Requirements and Recommendations - NVIDIA Docs"
[3]: https://docs.nvidia.com/networking/display/ConnectX6VPI/Introduction?utm_source=chatgpt.com "Introduction - NVIDIA Docs"
[4]: https://docs.nvidia.com/networking/display/GPUDirectRDMAv18 "NVIDIA GPUDirect RDMA User Manual | NVIDIA GPUDirect RDMA User Manual"
[5]: https://network.nvidia.com/products/GPUDirect-RDMA "Mellanox OFED GPUDirect RDMA"
[6]: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html "GPUDirect RDMA and GPUDirect Storage — NVIDIA GPU Operator"
[7]: https://docs.open-mpi.org/en/v5.0.7/tuning-apps/networking/cuda.html?utm_source=chatgpt.com "11.2.6. CUDA — Open MPI 5.0.7 documentation"
[8]: https://docs.nvidia.com/nvshmem/api/introduction.html?utm_source=chatgpt.com "Introduction — NVSHMEM 3.6.5 documentation"
[9]: https://developer.nvidia.com/gpudirect "GPUDirect | NVIDIA Developer"
[10]: https://rocm.docs.amd.com/en/docs-6.0.0/how-to/gpu-enabled-mpi.html "GPU-enabled Message Passing Interface — ROCm Documentation"