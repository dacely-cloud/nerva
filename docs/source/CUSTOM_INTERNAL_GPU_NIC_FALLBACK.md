Yes, but the answer splits into two layers:

```text id="rdivsy"
We can absolutely create our own distributed inference/runtime/transport software.

We probably cannot create our own true GPU-direct NIC access path
if the GPU driver/hardware refuses to expose VRAM for peer DMA.
```

That distinction matters.

## What we can build ourselves

We can build:

```text id="c5lq2b"
custom multi-host inference runtime
custom stage pipeline
custom activation transport
custom RDMA protocol over ConnectX-6
custom pinned-buffer fallback
custom scheduler
custom memory residency planner
custom GPU/CPU execution split
custom topology-aware routing
custom token latency ledger
```

That is fully possible.

The runtime does not need NVIDIA’s fancy distributed stack. It can use ordinary Linux processes, CUDA/HIP kernels, InfiniBand/RoCE, UCX, libibverbs, pinned memory, and custom scheduling.

## What we probably cannot fully build ourselves

The thing we cannot simply “code around” is this:

```text id="j30upp"
NIC directly reading/writing GPU VRAM requires the NIC to receive valid DMA mappings
for GPU memory.
```

That means the GPU driver must expose GPU memory pages/BAR mappings safely to the RDMA stack. NVIDIA’s GPUDirect RDMA documentation says peer devices access GPU memory through PCIe BAR address mappings, and because operating systems do not normally exchange those mappings between arbitrary drivers, the NVIDIA kernel driver exports functions for address translation and mapping. ([docs.nvidia.com](https://docs.nvidia.com/cuda/gpudirect-rdma/))

So if the GPU driver does not provide a supported path, we cannot just write normal userspace code and magically let the NIC DMA into VRAM.

The kernel-level pieces are real:

```text id="gid5lh"
GPU memory pinning
GPU virtual address → physical/BAR mapping
DMA address registration
memory lifetime tracking
free callback / invalidation
IOMMU mapping
PCIe P2P permission
NIC memory-region registration
```

Linux has a PCI peer-to-peer DMA framework, but it still requires proper provider/importer drivers, compatible PCIe topology, and revocation/lifetime handling. The Linux docs explicitly describe compatibility checks around PCI root ports and driver-managed P2P memory lifetimes. ([docs.kernel.org](https://docs.kernel.org/driver-api/pci/p2pdma.html))

So the real answer is:

```text id="czql1k"
We can build the inference system.
We can build the transport layer.
We can use official GPUDirect / PeerDirect paths if available.
We should not assume we can replace the GPU vendor’s peer-memory driver from scratch.
```

## Existing official paths

For NVIDIA, there are two supported GPUDirect RDMA kernel-mode paths today:

```text id="s6h8js"
DMA-BUF path
legacy nvidia-peermem path
```

NVIDIA’s current GPU Operator documentation says GPUDirect RDMA needs userspace CUDA APIs, and kernel support is provided either by Linux DMA-BUF or the legacy `nvidia-peermem` module; NVIDIA recommends DMA-BUF over `nvidia-peermem`. ([docs.nvidia.com](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html))

For AMD, ROCm exposes RDMA through PeerDirect interfaces, allowing NICs to directly read/write RDMA-capable GPU device memory; AMD documents ROCm-aware Open MPI over InfiniBand/RoCE using UCX. ([rocm.docs.amd.com](https://rocm.docs.amd.com/en/docs-6.0.0/how-to/gpu-enabled-mpi.html))

So the portable runtime should support:

```text id="eyu438"
NVIDIA direct path:
    CUDA buffer → DMA-BUF / nvidia-peermem → UCX/libibverbs → ConnectX-6

AMD direct path:
    HIP buffer → ROCm PeerDirect → UCX/libibverbs → ConnectX-6

Fallback path:
    GPU buffer → pinned host ring → RDMA → remote pinned host ring → remote GPU
```

## Could we write a kernel module?

Maybe, but only in the sense of integrating with the official APIs.

For NVIDIA, their GPUDirect RDMA documentation is literally an API guide for developing a Linux kernel module that connects third-party devices to NVIDIA GPUs. It explains pinning GPU memory, unpinning, free callbacks, buffer ID checks, and linking against `nvidia.ko`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/gpudirect-rdma/))

So yes, we could write a kernel module or userspace RDMA layer that uses the official NVIDIA path.

But if the question is:

```text id="k75qec"
Can we create our own replacement for nvidia-peermem
that forces unsupported GeForce VRAM to be RDMA-accessible?
```

Then the practical answer is:

```text id="u9iyz1"
Not realistically as a serious engineering plan.
```

Because the GPU driver owns the GPU memory manager. Without exported APIs, we do not safely know:

```text id="734fd6"
which physical GPU pages back the allocation
whether they can move
how long they remain valid
how to revoke mappings when freed
how IOMMU mappings are programmed
how BAR aperture mappings are managed
how cache coherency/order is guaranteed
```

Trying to bypass that would be fragile, driver-version-specific, and likely not maintainable.

## The better “create our own” solution

We create our own **communication abstraction**, not our own fake GPUDirect.

The runtime should have four transfer modes.

```text id="fllzz5"
Mode 1 — true GPU-direct RDMA:
    GPU VRAM → NIC → network → remote GPU VRAM

Mode 2 — optimized pinned-host bounce:
    GPU VRAM → pinned DRAM → NIC → network → pinned DRAM → remote GPU VRAM

Mode 3 — CPU-produced boundary:
    CPU computes final boundary output directly into pinned send buffer

Mode 4 — GPU writes directly to mapped host pinned memory:
    GPU kernel writes boundary activation into host-mapped pinned memory
    NIC sends that pinned buffer
```

Mode 4 is interesting.

Instead of:

```text id="w7r82e"
GPU computes activation in VRAM
cudaMemcpyAsync VRAM → pinned DRAM
NIC sends pinned DRAM
```

we can sometimes do:

```text id="r8as27"
GPU final kernel writes activation directly into mapped pinned host memory
NIC sends that same buffer
```

This avoids an explicit D2H copy kernel/copy call, but it does **not** avoid PCIe. The GPU is still writing over PCIe into host memory. It may be slower than VRAM writes, but for a decode activation like 32 KB, it may be acceptable and simpler.

That gives us a practical fallback even if RTX 2080 Ti GPUDirect is blocked.

## Why this may be enough

For decode, the stage boundary activation is small.

Example:

```text id="e1lpj4"
hidden_size = 16,384
dtype = fp16/bf16
activation = 16,384 × 2 bytes = 32 KB
```

Even with a host bounce, this is tiny compared to model weights.

The killer is not the raw bytes. The killer is synchronization.

So the fallback path must be:

```text id="825vgi"
preallocated
pre-pinned
pre-registered with NIC
double/triple buffered
async D2H
async RDMA send/write
async H2D on receive
no malloc
no registration per token
no pageable memory
no normal TCP send path
```

Bad fallback:

```text id="q3wh0w"
cudaMemcpy to pageable RAM
send() over TCP
recv()
cudaMemcpy to GPU
```

Good fallback:

```text id="bd7lxg"
cudaMemcpyAsync to pinned registered ring
ibv_post_send / UCX put
remote registered pinned ring
cudaMemcpyAsync to GPU
```

For prefill, activations are much larger, so we need chunking and overlap.

## Phase 1 addition

Add this to the final document:

```text id="to8mtv"
We can create our own distributed runtime and communication layer,
but we should not rely on inventing unsupported GPU peer-memory access.

The runtime must detect the best available memory path at startup:

    if GPU-direct RDMA works:
        use GPU → NIC → remote GPU

    else if mapped pinned host output is faster:
        produce boundary tensors into pinned host egress buffers

    else:
        use async GPU → pinned host → RDMA → pinned host → GPU

The design must never require proprietary enterprise memory pooling.
It only requires a transport abstraction with multiple backends.
```

## Practical recommendation

For your ConnectX-6 + 2080 Ti system, build this test matrix:

```text id="he02ti"
A. UCX/libibverbs with CUDA buffer directly
B. UCX/libibverbs with pinned host buffer
C. GPU writes boundary tensor into mapped pinned host buffer
D. cudaMemcpyAsync D2H into pinned buffer + RDMA
E. normal socket fallback, only as baseline
```

Then measure:

```text id="dyt61h"
latency for 32 KB
latency for 256 KB
latency for 1 MB
latency for 64 MB prefill chunks
CPU usage
DRAM bandwidth
PCIe RX/TX
GPU idle gap
NIC bandwidth
```

If true GPUDirect works on the 2080 Ti stack, great.

If it does not, the architecture is still viable because boundary activations are small and can use the optimized pinned-host path.

The key is this:

```text id="ake20c"
We do not need to defeat NVIDIA’s driver to build the system.
We need to make the runtime correct, topology-aware, async, and fallback-safe.
```
