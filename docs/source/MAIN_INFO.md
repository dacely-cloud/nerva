# Phase 1 Final Report

## Latency-First Hybrid LLM Inference on Commodity Hardware

Date: **June 25, 2026**
Scope: **LLM inference only**
Primary target: **exact or distribution-preserving inference without reducing model precision, shrinking the model, or requiring newer/fancier hardware**

---

## 1. Executive Thesis

The current mainstream LLM inference model is wrong or at least incomplete.

The common assumption is:

```text id="lrxx3m"
GPU fast.
CPU slow.
VRAM mandatory.
If model does not fit in VRAM, inference is impossible or terrible.
```

The better assumption is:

```text id="9sj913"
LLM inference is a memory-residency, scheduling, synchronization,
kernel-launch, driver, and dataflow problem.

GPU is a throughput engine.
CPU is a low-latency control/cache engine.
VRAM is a hot tier.
DRAM is a warm tier.
Disk is a cold tier.
PCIe/network are transfer fabrics.
Software decides whether the system is efficient or trash.
```

The core rule:

```text id="xr60x8"
Keep the critical path resident.
Not everything.
The critical path.
```

A model does **not** fundamentally need to be fully loaded into VRAM to run. But for a dense exact model, every generated token must still use the active weights somehow. So the real problem is not just capacity.

It is three separate problems:

```text id="6a3sn2"
Capacity:
    Can I store the model?

Bandwidth:
    Can I move or access the active weights/KV fast enough?

Latency:
    Can I do it without stalling the token critical path?
```

This distinction is everything.

---

## 2. Project Constraints

This project is not about making the model smaller.

Forbidden for this research direction:

```text id="57e4ej"
weight quantization
KV quantization
lossy compression
model pruning
dropping context
approximate attention
changing architecture
distillation as replacement
using a smaller model instead of the target model
precision downgrade
skipping dense work unless the model is exactly sparse
```

Allowed:

```text id="v8igsk"
exact IO-aware attention
exact blockwise attention
FlashAttention-style tiling
Flash-Decoding-style partial attention merge
kernel fusion
static arenas
persistent command graphs
CPU/GPU cooperative execution
distributed pipeline parallelism
explicit prefetch
KV paging
KV reuse
DRAM/VRAM tiering
disk as cold storage
speculative decoding with exact target verification
lossless compression if decompression is cheaper than raw movement
backend-neutral execution
```

FlashAttention is important because it proved exact attention can be faster by reducing memory traffic between GPU HBM and on-chip SRAM, not by changing the model. ([arXiv][2])

---

## 3. Current LLM Inference Design

A normal inference engine roughly does this:

```text id="po7t0e"
load weights
allocate KV cache
tokenize prompt
run prefill
store prompt KV
loop:
    run one-token decode
    read old KV
    append new KV
    compute logits
    sample token
    detokenize token
```

Most GPU-first engines assume:

```text id="34ycin"
weights should live in VRAM
KV cache should live in VRAM
activations should live in VRAM
GPU should execute almost all tensor ops
CPU should mostly feed the GPU
```

That works when:

```text id="x30p18"
model fits in VRAM
KV fits in VRAM
batch is large
work is throughput-oriented
kernel launches are amortized
GPU has enough parallelism
```

It fails when:

```text id="f7u7cd"
model exceeds VRAM
batch is 1
decode is serial
context is long
KV cache grows huge
offload is synchronous
runtime does too many small kernels
CPU/GPU sync is excessive
driver overhead is visible
PCIe/network transfer is on the hot path
```

The existing software world often creates a fake wall:

```text id="25bk0l"
fits in VRAM → possible
does not fit in VRAM → impossible / unusable
```

That wall is not physical law. It is mostly software design.

---

## 4. Prefill and Decode Are Different Problems

LLM inference has two different phases.

### Prefill

Prefill processes the prompt.

```text id="25j3om"
input = many prompt tokens
work = process all prompt tokens through all layers
output = logits + initial KV cache
```

Characteristics:

```text id="28uxdy"
parallel over sequence
larger GEMMs
higher arithmetic intensity
better GPU utilization
compute-heavy
```

### Decode

Decode generates one token at a time.

```text id="cwwgnx"
input = one new token + previous KV
work = run every layer
output = next token
```

Characteristics:

```text id="j0x4zm"
serial token dependency
tiny current activation
huge weight state
growing KV cache
low arithmetic intensity
memory-bound
launch/sync sensitive
```

This is why decode is where software inefficiency becomes obvious. Flash-Decoding exists because normal attention kernels underutilize the GPU during long-context decode; it splits K/V into chunks, processes them in parallel, then rescales and combines the partial outputs exactly. ([PyTorch][3])

---

## 5. CPU vs GPU: Correct Division of Labor

The CPU is not “slow GPU.” The GPU is not “fast CPU.” They are different machines.

### CPU should do

```text id="0wvzn6"
tokenization
detokenization
request ownership
decode-loop control
scheduler decisions
KV metadata management
prefix-cache lookup
page-table management
DRAM residency planning
prefetch scheduling
eviction decisions
sampling when GPU launch/transfer cost is higher
small reductions
branchy operations
irregular metadata traversal
CPU-side partial attention when data is host-resident
CPU-side dense shard compute when moving weights would be worse
```

CPU is strong when:

```text id="kpu1tk"
operation is small
operation is latency-sensitive
operation is branch-heavy
data is cache-resident
data is already in DRAM
GPU launch would dominate
GPU would need a synchronous transfer
```

### GPU should do

```text id="p8ibms"
large dense GEMM
prefill
hot decode projection
hot KV attention
FlashAttention-style kernels
MLP dense work
large parallel reductions
fused kernels
coalesced memory scans
```

GPU is strong when:

```text id="ywxk8o"
data is already in VRAM
parallelism is enough
memory access is coalesced
kernel launch is amortized
work is throughput-heavy
```

GPU is weak when:

```text id="dhzycu"
kernel is tiny
branch divergence is high
atomics serialize
barriers dominate
memory is uncoalesced
managed-memory page faults occur
host sync happens constantly
```

NVIDIA and AMD performance documentation both describe the core GPU rule: GPU memory latency is hidden by having enough independent warps/wavefronts ready to execute. If there are no eligible warps, the hardware stalls. ([arXiv][2])

---

## 6. VRAM, DRAM, Disk, PCIe, and Network

### VRAM should hold

```text id="d7c1vs"
hot weights
currently executing layer blocks
hot KV pages
current activations
GPU workspaces
staging rings
frequently reused prefix KV
```

VRAM should **not** be treated as:

```text id="hbuaey"
the whole model
a binary fit/fail boundary
cold KV storage
duplicated prefix storage
temporary-layout-conversion garbage
```

VRAM is the hot working set.

### DRAM should hold

```text id="ahr0lv"
warm weights
nonresident model segments
warm KV pages
cold-but-likely KV pages
prefix cache
CPU-side metadata
tokenizer/detokenizer structures
pinned transfer buffers
prefaulted arenas
mmap windows
CPU-computable weight shards
```

DRAM should not be a reactive dumping ground. It should be scheduled.

### Disk should hold

```text id="4h2r2j"
model files
cold weights
cold KV snapshots
persistent prefix/session cache
layout-transformed weight files
```

Disk should not be touched synchronously during token decode.

### PCIe and network should move

```text id="c2exdf"
activations between stages
prefetched weight tiles
evicted KV pages
small partial results
logits or hidden states when needed
```

PCIe/network should not be used as random-access fake VRAM.

Rule:

```text id="hmw59x"
Visible transfer cost = max(0, transfer_time - overlapped_compute_time)
```

If transfer is fully overlapped, it can nearly disappear. If transfer is synchronous, it becomes the bottleneck.

---

## 7. Hidden Stalls: OS Kernel, Driver, Runtime, GPU Kernel

The model code is not the whole program.

A runtime can stall because of:

```text id="1a6yqg"
syscalls
futex waits
locks
bad atomics
allocator locks
page faults
zero-fill-on-demand
mmap faults
pin/unpin registration
driver ioctl paths
GPU command submission
stream waits
event waits
implicit synchronization
GPU memory dependency stalls
GPU atomics
GPU barriers
GPU managed-memory faults
```

Normal flamegraphs often miss blocked time because the thread is not consuming CPU while it waits. Off-CPU profiling exists specifically to find wall-clock time lost while threads are blocked. ([arXiv][1])

The DPDK analogy matters. DPDK-style systems bypass parts of the generic kernel networking stack with userspace poll-mode drivers. io_uring similarly reduces traditional syscall/copy overhead with shared submission/completion rings. The lesson is not that LLMs are networking; the lesson is that generic kernel/runtime boundaries can destroy performance on the same hardware. ([arXiv][1])

For LLM inference, the hot decode path should avoid:

```text id="iyh0ri"
malloc/free
mmap/munmap
page faults
pin/unpin
global locks
global atomics
logging locks
metrics locks
dynamic GPU allocation
synchronous CPU↔GPU copies
driver launch storms
reactive managed-memory page migration
```

---

## 8. What Is Already Proven

### Proven: exact attention can be faster by redesigning memory IO

FlashAttention proves exact attention can be IO-aware and faster by reducing HBM/SRAM traffic. ([arXiv][2])

### Proven: KV cache is a memory-management problem

PagedAttention/vLLM treats KV cache like virtual memory, reducing fragmentation and enabling flexible sharing; the vLLM paper reports near-zero KV waste and 2–4x throughput improvement at the same latency level versus prior systems. ([arXiv][4])

### Proven: virtual KV layout matters

vAttention argues that PagedAttention’s non-contiguous virtual layout adds programming/performance overhead and proposes retaining virtual contiguity while decoupling physical allocation. ([arXiv][5])

### Proven: offload can run models larger than VRAM

FlexGen aggregates GPU, CPU, and disk memory and uses a placement search to run large models with limited GPU memory; its OPT-175B single-16GB-GPU result was throughput-oriented and used 4-bit compression, so not all of it fits our exactness constraint, but the memory-placement idea is proven. ([arXiv][6])

### Proven: CPU/NVMe offload can enable massive model inference

ZeRO-Inference explicitly targets massive model inference using non-GPU memory such as CPU/NVMe, making hundreds-of-billions-parameter inference possible with far less GPU memory. ([DeepSpeed][7])

### Proven: distributed consumer inference is possible

Petals demonstrated collaborative inference/fine-tuning by joining resources from multiple parties; its paper reports BLOOM-176B inference on consumer GPUs at about one step per second, enough for many interactive applications. ([arXiv][1])

A later distributed-inference paper reports Petals-style decentralized inference for Llama 2 70B and BLOOM 176B over the Internet, up to 10x faster than offloading for interactive generation. ([arXiv][8])

### Proven: pipeline parallelism across nodes is a known model

vLLM documentation describes using tensor parallelism within a node and pipeline parallelism across nodes, for example tensor-parallel size 8 and pipeline-parallel size 2 for 16 GPUs across 2 nodes. ([vLLM][9])

So your 4-system design is not impossible. What is missing is not physics. It is the right software.

---

## 9. What Has Been Tried and Usually Does Not Work Well

### Naive layer offload

Bad pattern:

```text id="b2gdtv"
put some layers on GPU
put rest in CPU RAM
copy missing layers during decode
```

Problem:

```text id="vip4i0"
dense decode needs every layer every token
synchronous streaming creates a bandwidth wall
```

### Reactive managed memory

Bad pattern:

```text id="v40y4v"
let CUDA/HIP/driver page data between CPU and GPU on demand
```

Problem:

```text id="y2cmvd"
GPU touches nonresident page
driver migrates page
kernel stalls
token latency spikes
```

### Disk on the hot path

Bad pattern:

```text id="32vy8z"
mmap huge model
allow decode to fault random pages from disk
```

Problem:

```text id="24zls6"
disk is cold storage, not token-time memory
```

### Pure tensor parallelism over weak interconnect

Bad pattern:

```text id="9vvub7"
split every layer across many GPUs/nodes
all GPUs participate every layer
all-reduce constantly
```

Problem:

```text id="74094k"
cross-device communication dominates
weak network/PCIe topology kills latency
```

### Too many tiny GPU kernels

Bad pattern:

```text id="7moi5u"
launch norm
launch qkv
launch rope
launch attention
launch residual
launch mlp
launch activation
launch projection
sync
repeat for every layer/token
```

Problem:

```text id="xk3yve"
driver/runtime launch overhead becomes visible
GPU has idle gaps
```

### Global locks and atomics

Bad pattern:

```text id="gef5tg"
global scheduler queue
global KV map lock
global token counter
global metrics atomics
shared reference counts
```

Problem:

```text id="08z6t8"
cache-line bouncing
futex waits
off-CPU stalls
multithreading slower than single-threading
```

---

## 10. What Works

Reliable pieces:

```text id="n25sha"
IO-aware attention
Flash-Decoding-style chunking
KV paging
KV prefix reuse
static memory arenas
kernel fusion
graph replay
persistent kernels where useful
pinned staging buffers
explicit prefetch
distributed pipeline parallelism
CPU/GPU hybrid execution
speculative decoding with exact verification
multi-backend execution
```

But no single one is enough.

The real redesign is the combination.

---

## 11. What Is Novel Here

The novelty is not “pipeline parallelism” alone. That exists.

The novelty is the full hybrid redesign:

```text id="37h8hb"
exact inference
latency-first
batch-1 aware
old-hardware aware
multi-host aware
VRAM as hot tier
DRAM as warm compute/storage tier
disk as cold tier
CPU as active compute/control device
GPU as local high-bandwidth throughput island
distributed sequential layer pipeline across machines
tiered KV cache with exact partial attention merge
speculative verification to amortize streamed dense weights
off-CPU/GPU-stall/driver-stall measurement built into runtime
backend-neutral execution
no precision loss
```

This is not a tiny kernel tweak. It is a runtime architecture change.

---

## 12. The 4-System / 32x 2080 Ti Architecture

Your proposed architecture should be a central Phase 1 design candidate.

You described:

```text id="j1mjdj"
800 GB model
4 physical systems
8x RTX 2080 Ti per system
32 GPUs total
large DRAM and disk available
no enterprise NVSwitch/NVLink cluster
custom software runtime
```

An RTX 2080 Ti has 4,352 CUDA cores, 11 GB GDDR6, and 616 GB/s memory bandwidth. Eight cards provide 88 GB VRAM per system and 32 cards provide 352 GB aggregate VRAM, but that VRAM is split into 32 separate memory islands, not one memory pool. ([IT Creations][10])

So for an 800 GB model:

```text id="w5if4i"
total VRAM = 32 × 11 GB = 352 GB
model size = 800 GB
remaining non-VRAM state = ~448 GB plus KV/runtime overhead
```

This does not make inference impossible. It means the runtime must not expect all weights to be VRAM-resident.

---

## 13. Correct Way to Use Four Systems

The correct distributed layout is **stage pipeline parallelism**, not fake unified GPU memory.

### Model split

```text id="28pvlv"
System 1:
    layers 0–N1

System 2:
    layers N1–N2

System 3:
    layers N2–N3

System 4:
    layers N3–final
```

Each system owns:

```text id="n7z33i"
its layer weights
its layer KV cache
its local GPU command graphs
its local CPU metadata
its local DRAM backing store
its local prefetch plan
```

The user prompt / hidden state flows:

```text id="3kw1aq"
tokens / hidden states
    → System 1
    → System 2
    → System 3
    → System 4
    → logits / sampled token
```

This is the important part:

```text id="ojqm2q"
Weights do not move between systems per token.
Activations move between systems.
```

That is a massive difference.

For decode, the activation crossing systems can be tiny compared to weights.

Example:

```text id="sapwb1"
hidden_size = 16384
dtype = bf16/fp16 = 2 bytes
one decode activation = 16384 × 2 = 32768 bytes = 32 KB
```

Across three inter-system boundaries:

```text id="ia95rm"
~96 KB per generated token
```

That is nothing compared to moving hundreds of GB of weights.

For prefill, the activation is larger:

```text id="nbmqtq"
sequence_length × hidden_size × bytes
```

Example:

```text id="zkxbte"
8192 tokens × 16384 hidden × 2 bytes ≈ 256 MB per boundary
```

Still much smaller than moving the full model.

So your idea is correct:

```text id="jdhdpf"
Move activations.
Do not move weights.
```

---

## 14. Why This Avoids the Fancy-Hardware Requirement

Enterprise systems use NVLink/NVSwitch to make multi-GPU communication fast and convenient.

But this design does not require pretending all GPUs are one memory pool.

It needs:

```text id="sya2el"
one runtime process per machine
one or more workers per GPU
local layer ownership
local KV ownership
network transport for activations
explicit scheduling
explicit prefetch
explicit backpressure
```

Possible transports:

```text id="d31rr9"
TCP
QUIC
RDMA if available
UCX
gRPC-like custom binary transport
shared memory within node
PCIe P2P where available
NCCL only where useful, not mandatory
```

The driver does not need to “allow” one 352 GB VRAM pool.

The runtime can orchestrate 32 independent GPUs.

The architecture is:

```text id="ywvl5t"
distributed actor system
not fake monolithic GPU
```

---

## 15. Critical Correction: It Helps Capacity More Than Single-Request Latency

Your statement is almost right:

```text id="6qpm0c"
user prompt goes system 1 → system 4
you barely lose latency if done correctly
```

The communication latency can be small because activations are small.

But single-request latency is still:

```text id="l86ynm"
T_token = T_stage1 + T_stage2 + T_stage3 + T_stage4 + network_between_stages
```

So for a **single user, one token at a time**, pipeline stages are sequential.

The design improves single-request latency only if:

```text id="tdylex"
each stage is much faster because it owns less model
weights/KV are local
communication is small
driver overhead is controlled
prefetch works
CPU/GPU split is good
```

It improves throughput much more naturally when:

```text id="eykewp"
multiple requests are in flight
prompt chunks are microbatched
speculative verification creates K-token blocks
pipeline bubbles are filled
```

So the correct claim is:

```text id="cjpc0v"
This design makes 800 GB inference possible and can be efficient.
It does not magically make autoregressive single-token latency free.
To become fast, it needs pipeline filling, speculative verification,
microbatching, or enough local stage acceleration.
```

---

## 16. Why “Do Not Hit All Cards at Once” Is Partly Correct

You are right if “hit all cards at once” means:

```text id="l0ugg3"
force all 32 GPUs to participate in every layer
do all-reduce over weak network every layer
make every token wait on global synchronization
```

That will kill performance.

But inside one machine, using 8 GPUs together can still be useful if designed correctly.

The right hierarchy is:

```text id="09f4kw"
across machines:
    pipeline parallelism by layer groups

inside each machine:
    local pipeline or careful tensor/row sharding
    avoid global all-reduce over network
    prefer moving small activations/results
    keep weights local
```

Bad architecture:

```text id="12ym8c"
32 GPUs all synchronize every layer
```

Good architecture:

```text id="501w27"
4 systems synchronize only at stage boundaries
8 local GPUs per system cooperate on local layer group
```

Even better:

```text id="w5kygp"
within a system:
    GPU 0 owns layer/tile group A
    GPU 1 owns layer/tile group B
    ...
    CPU owns warm DRAM shards
    local scheduler decides whether to compute or prefetch
```

---

## 17. Per-System Internal Design

Each of the 4 systems has 8 RTX 2080 Ti cards.

Do not treat each system as one fake 88 GB GPU. Treat it as:

```text id="76kw24"
8 local high-bandwidth islands
1 large DRAM pool
1 CPU NUMA topology
1 local disk/NVMe tier
1 local scheduler
```

Each system should have:

```text id="0lyueq"
local layer partition
local KV cache for those layers
GPU-resident hot weights
DRAM-resident warm weights
disk-resident cold weights
pinned staging rings
static CPU/GPU arenas
prefetch planner
token-stage executor
```

The stage executor should decide:

```text id="vilcpv"
compute on GPU if weight tile is hot in VRAM
compute on CPU if weight tile is in DRAM and result is smaller than moving weights
prefetch if tile will be needed soon and compute window can hide transfer
evict if reuse distance is large
```

For dense decode operation:

```text id="k3i0u3"
y = W x
```

If `W` is huge and `x` is tiny, moving `W` can be worse than computing where `W` already lives.

So for host-resident shards:

```text id="yn0sni"
CPU computes y_part = W_host_shard × x
send y_part to merge
```

Instead of:

```text id="69zj9k"
copy W_host_shard to GPU
GPU computes y_part
```

This is one of the most important hybrid ideas.

---

## 18. 800 GB Model: Do We Need 800 GB VRAM?

No, not fundamentally.

But we need to be honest.

For an exact dense 800 GB model:

```text id="unxk5v"
the active dense weights must participate in each token
```

So if the model is dense and active weight volume is 800 GB, at 1 token/s you need roughly:

```text id="wyk4t2"
800 GB/s effective weight access
```

At 5 token/s:

```text id="2zpndd"
4 TB/s effective weight access
```

At 10 token/s:

```text id="9gxoe8"
8 TB/s effective weight access
```

This does **not** mean 800 GB VRAM is required.

It means the runtime must get 800 GB/token of useful dense work from:

```text id="4q71ec"
VRAM bandwidth
DRAM bandwidth
CPU compute
GPU compute
prefetch
pipeline parallelism
batching
speculative verification
MoE sparsity if model has it
```

Capacity can be solved by DRAM/disk.

Bandwidth and latency require software design.

---

## 19. Cases Where 800 GB Without 800 GB VRAM Is Practical

### Case A: MoE model

If 800 GB is total model size but only a subset of experts is active per token:

```text id="7fknzt"
active weight volume << total model size
```

Then the design is much easier.

VRAM holds:

```text id="xryf4p"
hot experts
router
shared layers
hot KV
```

DRAM holds:

```text id="122k7a"
warm/cold experts
prefetch candidates
```

This is exactly where old multi-GPU hardware can shine.

### Case B: batched throughput

If many requests are processed together:

```text id="uqelth"
load/stage weights once
apply to many tokens/sequences
```

This amortizes weight movement.

FlexGen is mostly in this family: high-throughput, latency-insensitive batching with GPU/CPU/disk memory aggregation. ([arXiv][6])

### Case C: speculative verification

This is the big one for interactive use.

Naive dense decode:

```text id="14f1qj"
stream/use model once → generate/verify 1 token
```

Speculative verified decode:

```text id="nw5i00"
draft K tokens cheaply
stream/use target model once → verify K tokens
```

If accepted tokens per target pass > 1, weight access is amortized. The target distribution remains correct if verification is implemented properly. Speculative decoding was introduced specifically as a way to accelerate autoregressive generation while preserving the target model distribution. ([arXiv][1])

### Case D: layer pipeline across systems

Your proposed 4-system layer pipeline avoids moving weights over the network.

```text id="mjto8u"
weights stay local
KV stays local
activations move
```

This is exactly the right direction.

---

## 20. Cases Where It Will Be Hard

### Dense exact batch-1, no speculation, no batching

This is the worst case.

Every token needs the full dense model.

If a large fraction of the 800 GB is in DRAM/disk, and only one token is being generated, the system may be limited by:

```text id="fzrhuk"
DRAM bandwidth
CPU compute
PCIe staging
pipeline bubbles
network latency
GPU underutilization
```

It can run.

Fast interactive latency is not proven.

### Disk hot path

If disk is read per token, it will be awful.

Disk must be cold storage and prefetch only.

### Global tensor parallel over four systems

If every layer requires all four systems to communicate partial results, the network becomes the bottleneck.

Avoid this.

---

## 21. The Distributed Pipeline Architecture

Name it:

```text id="zu1jhr"
HILO-DP: Hierarchical Inference Layout Optimizer — Distributed Pipeline
```

Core structure:

```text id="e15wuj"
Client
  ↓
Coordinator
  ↓
Stage 1: System 1, GPUs 0–7, layers A
  ↓ activation
Stage 2: System 2, GPUs 0–7, layers B
  ↓ activation
Stage 3: System 3, GPUs 0–7, layers C
  ↓ activation
Stage 4: System 4, GPUs 0–7, layers D
  ↓ logits
Sampler
  ↓
token output
```

Each stage has:

```text id="k6qjrn"
local model shard
local KV cache
local CPU/GPU scheduler
local memory planner
local prefetch engine
local command graph cache
local static arenas
```

The coordinator does:

```text id="izc4ux"
request routing
backpressure
pipeline scheduling
failure detection
stage latency accounting
token ledger aggregation
speculative draft control
```

The coordinator does **not** do:

```text id="juijxp"
centralized tensor math
global locks on hot path
per-token heavyweight scheduling
```

---

## 22. Dataflow for Prefill

Prompt prefill:

```text id="ap6pss"
tokens enter stage 1
stage 1 processes its layers
activation block moves to stage 2
stage 2 processes its layers
activation block moves to stage 3
stage 3 processes its layers
activation block moves to stage 4
stage 4 outputs logits / final hidden state
```

KV storage:

```text id="12snm7"
stage 1 stores KV for layers it owns
stage 2 stores KV for layers it owns
stage 3 stores KV for layers it owns
stage 4 stores KV for layers it owns
```

Important:

```text id="cthd0l"
KV does not need to be centralized.
Each stage owns KV for its own layers.
```

This is huge. It avoids moving all KV everywhere.

---

## 23. Dataflow for Decode

For each generated token:

```text id="bvdpo7"
current token embedding / hidden state → stage 1
stage 1 runs local decode layers using local KV
stage 1 appends local KV
hidden state → stage 2
stage 2 runs local decode layers using local KV
stage 2 appends local KV
hidden state → stage 3
stage 3 runs local decode layers using local KV
stage 3 appends local KV
hidden state → stage 4
stage 4 runs local decode layers using local KV
stage 4 appends local KV
stage 4 computes logits
sampler chooses next token
```

Network traffic:

```text id="1nlhe9"
hidden state per stage boundary
not weights
not full KV
```

This is why the architecture is possible.

---

## 24. Pipeline Bubbles and How to Kill Them

For one request and one token at a time, pipeline utilization is poor:

```text id="0ftztg"
stage 1 active
stage 2 idle
stage 3 idle
stage 4 idle

then stage 1 idle
stage 2 active
stage 3 idle
stage 4 idle
...
```

Ways to fill the pipeline:

```text id="g6aiyv"
multiple concurrent users
microbatch prompt chunks
speculative K-token verification
continuous batching
chunked prefill
interleave requests
stage-local async prefetch
```

This is why speculative verification is extremely important for your design.

Instead of verifying one token:

```text id="lpbzem"
token t passes stage 1→2→3→4
sample
token t+1 passes stage 1→2→3→4
sample
```

Use draft K-token block:

```text id="i1a856"
draft proposes tokens t..t+K
target pipeline verifies block
accepted tokens amortize full model pass
```

This makes the distributed pipeline much more efficient.

---

## 25. What Is Novel About Your 4-System Idea

Pipeline parallelism exists.

Petals exists.

vLLM distributed inference exists.

What is novel here is the combination:

```text id="y3x13l"
consumer old GPUs
multi-host layer pipeline
local 8-GPU memory islands
DRAM as warm compute tier
CPU compute for host-resident shards
exact KV locality per stage
activation-only inter-system transfer
speculative verification to fill pipeline
no enterprise interconnect assumption
no fake unified VRAM
no precision loss
```

That exact combination is not mainstream.

The current ecosystem mostly optimizes for:

```text id="7n9zn4"
single big GPU
or datacenter GPU cluster
or throughput batching
or CUDA-specific paths
```

Your design targets:

```text id="dkpm2k"
commodity used hardware
large DRAM
many old GPUs
weak interconnect
exact inference
custom software scheduler
```

That is a different optimization problem.

---

## 26. Driver Limitation vs Software Limitation

You are right that “driver does not allow it” is not the real answer.

The driver may not give:

```text id="vgx0zy"
one giant transparent VRAM pool
NVSwitch-like collectives
easy peer-to-peer across all cards
enterprise memory pooling
```

But none of that is required for this architecture.

The runtime can use:

```text id="s6xdmt"
ordinary GPU contexts
ordinary device memory
ordinary host memory
ordinary network sockets/RDMA
ordinary process communication
custom tensor transport
custom scheduler
```

So the real limitation is:

```text id="ysbs4b"
the software does not exist yet in the form we want
```

Not:

```text id="4np60t"
physics forbids it
```

Not:

```text id="msucdl"
the driver must bless a giant fake GPU
```

---

## 27. Old Hardware: 32x RTX 2080 Ti

A single RTX 2080 Ti is old, but not useless.

Relevant approximate aggregate:

```text id="flnld1"
1x 2080 Ti:
    11 GB VRAM
    616 GB/s local memory bandwidth
    4,352 CUDA cores

8x 2080 Ti:
    88 GB aggregate VRAM
    ~4.9 TB/s aggregate local memory bandwidth
    34,816 CUDA cores

32x 2080 Ti:
    352 GB aggregate VRAM
    ~19.7 TB/s aggregate local memory bandwidth
    139,264 CUDA cores
```

The aggregate numbers are real, but they are not automatically usable as one device. The scheduler has to expose the parallelism without forcing global synchronization. The RTX 2080 Ti hardware specs support the premise that a pile of old cards has serious aggregate compute and memory bandwidth, even though each card has only 11 GB VRAM. ([IT Creations][10])

Compared with one RTX 5090, the old cluster has far more aggregate VRAM and local bandwidth, but the RTX 5090 has one coherent 32 GB device memory space, newer tensor cores, newer cache behavior, and much simpler scheduling. NVIDIA lists the RTX 5090 as a 32 GB GDDR7 Blackwell card; third-party summaries of NVIDIA specs list 21,760 CUDA cores and 1,792 GB/s bandwidth. ([NVIDIA][11])

Correct conclusion:

```text id="kn67n2"
32x 2080 Ti can be powerful.
But only if software treats them as distributed memory islands.
```

---

## 28. Exact Tiered KV Cache

KV cache must not be a giant naive tensor.

Each stage should own KV for its layers:

```text id="10jaga"
stage 1 owns KV for layers 0–N1
stage 2 owns KV for layers N1–N2
stage 3 owns KV for layers N2–N3
stage 4 owns KV for layers N3–final
```

Within each stage:

```text id="dv0stn"
hot KV → local GPU VRAM
warm KV → pinned DRAM
cold KV → DRAM/disk-backed cache
```

For attention over block `i`:

```text id="e2t9z9"
m_i = local max
l_i = local exp sum
o_i = local weighted value
```

Merge blocks:

```text id="iiu41j"
m = max(m_i)
l = Σ exp(m_i - m) × l_i
o = Σ exp(m_i - m) × o_i
result = o / l
```

This allows exact blockwise attention across tiers.

GPU can compute hot blocks.

CPU can compute warm/cold blocks if cheaper than staging them.

---

## 29. Residency Planner

The runtime needs a residency planner, not `n_gpu_layers`.

Bad knob:

```text id="jkghmy"
put 35 layers on GPU
put rest on CPU
```

Better planner:

```text id="mnqf7z"
for every tensor:
    where does it live?
    when is it next used?
    how expensive is movement?
    how expensive is local compute?
    is it hot/warm/cold?
    can transfer be overlapped?
    does it cause sync?
```

Placement targets:

```text id="c432zp"
GPU VRAM
pinned DRAM
normal DRAM
CPU cache-resident metadata
disk/mmap cold storage
remote system
```

The planner must decide:

```text id="tnl8ln"
compute near data
or move data near compute
```

Not always GPU.

---

## 30. Kernel and Runtime Rules

Canonical rules:

```text id="az9o0a"
Every byte must justify its trip.
Every sync must justify its stall.
Every kernel launch must justify its existence.
Every tensor must have a residency reason.
Every syscall must justify crossing into the kernel.
Every atomic must justify cache-line ownership transfer.
Every allocation must justify page-table and zeroing risk.
Every wait must be visible in wall-clock profiling.
Every GPU page migration must be scheduled, not reactive.
Every backend choice must be measured, not assumed.
```

Hot decode path must avoid:

```text id="8528s4"
malloc/free
mmap/munmap
page faults
global locks
global atomics
logging locks
metrics atomics
GPU malloc/free
managed-memory faults
unnecessary memset
layout conversion
tiny kernel launch storms
host-device sync
```

---

## 31. What Phase 1 Must Measure

The measurement harness must produce a token ledger.

For every token:

```text id="ar096w"
CPU active time
CPU blocked/off-CPU time
GPU active time per device
GPU idle time per device
kernel launches
driver API time
network transfer time
PCIe transfer time
DRAM bandwidth
VRAM bandwidth
page faults
futex waits
lock waits
atomic contention
memcpy bytes
memset bytes
KV movement
KV hit/miss
pipeline bubble time
stage imbalance
```

Example ledger:

```text id="cxgafa"
token 421:
    total latency: 3585 us
    stage 1: 820 us
    stage 2: 790 us
    stage 3: 840 us
    stage 4: 900 us
    network: 55 us
    CPU blocked: 120 us
    GPU idle bubble: 310 us
    page faults: 0
    allocator: 0
    memcpy visible: 0
```

Without this, we are guessing.

---

## 32. Phase 1 Acceptance Gate

Phase 1 is not accepted by showing that a model can run once.

Phase 1 is accepted only when the runtime can prove where token time goes and
can change placement or dataflow based on that proof.

Required evidence:

```text id="f3w81y"
per-token wall latency
per-token GPU active spans
per-token GPU idle gaps
per-token CPU active time
per-token CPU blocked/off-CPU time
per-kernel or per-graph replay cost
runtime API cost
driver wait cost
allocator calls
page faults
host/device copy bytes
DRAM/VRAM residency decisions
candidate CPU/GPU/DRAM costs
selected executor and reason
```

The runtime must distinguish:

```text id="h4qj2f"
hardware latency:
    unavoidable device, memory, and fabric latency

software latency:
    avoidable waits, redundant movement, bad layout, bad sync,
    weak kernel shape, launch storms, allocator work, page faults,
    poor cache behavior, and wrong executor choice
```

Phase 1 must not use hardware as an excuse until these software causes have
been measured and either removed or marked unavoidable.

### Current Qwen decode bottleneck rule

The current single-GPU Qwen decode profile shows that the hot replay path is
dominated by dense batch-1 projection work over resident weights.

The observed hot path is not dominated by:

```text id="wvrqf8"
disk reads
PCIe model loading
token D2H visibility copy
host output synchronization
graph replay overhead
attention
sampling
```

The observed hot path is dominated by:

```text id="tn0b04"
QKV projection
attention output projection
MLP gate/up projection
MLP down projection
lm_head projection
```

So the first real performance target is not "add more timing." It is:

```text id="pt558b"
reduce projection memory traffic
reduce projection kernel overhead
fuse MLP projection/activation/down dataflow where legal
test CPU/DRAM compute-near-data for nonresident shards
test GPU-resident, GPU-staged, CPU-resident, and hybrid candidates
select the lowest visible critical-path cost
```

### Required source-level analysis

For any hot operation, Phase 1 must record:

```text id="d2t65u"
source file and function
what memory it reads
what memory it writes
who owns each buffer
which synchronization protects correctness
which work is inside graph replay
which work happens during setup/capture
which memory movement is avoidable
which kernel shape limits occupancy or bandwidth
what exact replacement is being tested
```

For `y = W x`, the planner must compare at least:

```text id="gj8w90"
GPU reads W from VRAM and computes
GPU stages W from DRAM then computes
CPU computes against DRAM-resident W
CPU/GPU split W and merge partial y
```

The selected path must be justified by measured or explicitly estimated visible
critical-path cost, not by assuming that GPU always wins.

### Old-hardware requirement

Phase 1 must keep old hardware in the design loop.

That means:

```text id="lrv88w"
no dependency on newest tensor-core-only paths
no dependency on enterprise memory pooling
no assumption that aggregate VRAM is coherent
no global all-reduce as the default distributed strategy
no precision loss to make a result look fast
```

The old-GPU path is allowed to be slower than a new single GPU. It is not
allowed to be architecturally ignored.

---

## 33. Final Architecture Summary

The final proposed runtime is:

```text id="7a2ksh"
HILO:
    Hierarchical Inference Layout Optimizer

HILO-DP:
    Distributed Pipeline extension for multi-host old-GPU clusters
```

Core principles:

```text id="un6xnl"
do not require full VRAM residency
do not fake unified GPU memory
split model by stages across systems
keep weights local to stages
keep KV local to owning layers
move activations, not weights
use CPU as control and warm compute tier
use GPU as hot throughput tier
use DRAM as planned warm storage
use disk only as cold storage
use speculative verification to amortize dense passes
measure hidden stalls
support old hardware
avoid vendor lock-in where possible
```

---

## 34. Final Answer to the Big Question

Can you run an 800 GB model on 4 systems with 32x RTX 2080 Ti?

```text id="kgx1nu"
Yes, in principle.
```

Do you need 800 GB VRAM?

```text id="on4bab"
No.
```

Can it be efficient?

```text id="4l4bz4"
Yes for some workloads.
Not proven for worst-case dense exact batch-1 without speculation or batching.
Very plausible with:
    layer pipeline across systems
    local 8-GPU stage scheduling
    DRAM-resident warm weights
    CPU compute for host shards
    activation-only network transfer
    exact tiered KV
    speculative verification
    no disk on hot path
    no global all-reduce over all systems
```

The correct architecture is not:

```text id="6b8bto"
32 GPUs behave like one GPU
```

It is:

```text id="w7f51b"
32 GPUs behave like 32 local high-bandwidth compute islands
coordinated by a runtime that understands the model graph.
```

And your proposed execution flow is exactly the right starting point:

```text id="4u5yyn"
user prompt
    → system 1 owns part A
    → system 2 owns part B
    → system 3 owns part C
    → system 4 owns part D
    → final logits
    → sample
    → repeat
```

That is the missing software layer.

The final Phase 1 thesis is:

```text id="nl1g1v"
Large-model inference on commodity hardware is not blocked by the absence of enterprise memory pooling.
It is blocked by the absence of a latency-first distributed runtime that treats CPU, DRAM, VRAM, disk, PCIe, network, GPU kernels, and OS/driver stalls as one scheduling problem.
```

That is the redesign.

[1]: https://arxiv.org/pdf/2209.01188?utm_source=chatgpt.com "arXiv:2209.01188v2 [cs.LG] 2 Mar 2023"
[2]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[3]: https://pytorch.org/blog/flash-decoding/?utm_source=chatgpt.com "Flash-Decoding for long-context inference"
[4]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[5]: https://arxiv.org/abs/2405.04437?utm_source=chatgpt.com "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention"
[6]: https://arxiv.org/abs/2303.06865?utm_source=chatgpt.com "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"
[7]: https://www.deepspeed.ai/2022/09/09/zero-inference.html?utm_source=chatgpt.com "ZeRO-Inference: Democratizing massive model inference"
[8]: https://arxiv.org/abs/2312.08361?utm_source=chatgpt.com "Distributed Inference and Fine-tuning of Large Language Models Over The Internet"
[9]: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/?utm_source=chatgpt.com "Parallelism and Scaling - vLLM Documentation"
[10]: https://www.itcreations.com/nvidia-gpu/nvidia-geforce-rtx-2080-ti-gpu?srsltid=AfmBOor4EvaIX472lYRM01kNDu-J-vlG9tfjiST5K_CgIPDVnezoEi7H&utm_source=chatgpt.com "NVIDIA GEFORCE RTX 2080 TI GPU"
[11]: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/?utm_source=chatgpt.com "GeForce RTX 5090 Graphics Cards"
