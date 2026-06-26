# NERVA

NERVA means Neural Execution & Residency Virtual Architecture.

NERVA is an inference operating system for AI models: a Rust/CUDA-first runtime
that rebuilds LLM inference around memory residency instead of treating the GPU
as a dumb execution target.

The model is not loaded. The model is scheduled.

The Transformer math stays exact, but the execution machine changes: model
weights, KV cache, activations, tokens, and sampler state become explicit
ResidentBlocks that can live in VRAM, pinned DRAM, DRAM, or disk, with the
runtime deciding whether to compute near the data, prefetch it, evict it, or
keep it hot.

The CPU becomes the latency control plane and warm-compute tier. The GPU becomes
the hot throughput plane. VRAM becomes a managed hot cache rather than an
all-or-nothing model container. KV cache becomes virtual memory. Decode becomes
a device-resident transaction instead of a Python/CPU-mediated token loop. Every
token produces a ledger explaining latency, stalls, copies, syncs, allocations,
and kernel work.

In one sentence: NERVA makes AI inference scheduled, not loaded.

Initial project direction is documented in:

- `ARCHITECTURE_START.md`
- `MAIN_INFO.md`
- `CONTEXT.md`
- `INTER_SYSTEM_GPU_NIC_COM.md`
- `PHASE_2_LATENCY_LEDGER.md`
- `VLLM_LATENCY_RESULTS.md`
- `CUSTOM_INTERNAL_GPU_NIC_FALLBACK.md`
- `REDESIGN.md`

NERVA is not a vLLM fork.
NERVA is not an rvLLM fork.

vLLM is used as a baseline and compatibility oracle.
rvLLM is used as a Rust/CUDA architecture reference.
NERVA is a new runtime.
