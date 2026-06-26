# vLLM and rvLLM Porting Ledger

This ledger records source paths that are relevant to NERVA implementation and
the decision for each area. The default policy is to recode contracts and
invariants into NERVA's `ResidentBlock` architecture rather than bulk-copying
Python/Torch hot-path code or model-specific Rust/CUDA paths.

## Implemented Slice: Static Arenas

Source references:

- `rvllm/v3/crates/rvllm-mem/src/hbm.rs`
- `rvllm/v3/crates/rvllm-mem/src/pinned.rs`
- `rvllm/v3/crates/rvllm-mem/src/capture.rs`

NERVA decision:

- Recode the stable-slab, aligned-region, checkpoint/restore, and graph-safe
  allocation discipline in `nerva-memory`.
- Keep CUDA pointers out of core and memory metadata. Logical arena regions
  use `GlobalBlockAddress` until native backends attach real allocations.
- Reject arena reservation during `AllocationPhase::HotPath`.

Implemented in:

- `crates/nerva-memory/src/lib.rs`
- `crates/nerva-runtime/src/lib.rs`
- `crates/nerva-ledger/src/lib.rs`

## Implemented Slice: Synthetic Transaction Lifecycle

Source references:

- `rvllm/v3/crates/rvllm-runtime/src/engine.rs`
- `rvllm/v3/crates/rvllm-graph/src/pool.rs`

NERVA decision:

- Recode the launch/collect type-state pattern as a synthetic NERVA
  transaction path.
- Recode replay-time graph layout hash checks as backend-neutral graph
  descriptors.
- Add a device-first token ring so token `t + 1` must consume the device output
  from token `t`.
- Keep this synthetic path model-free. It validates ownership, graph replay
  accounting, token causality, and ledger plumbing before real model execution.

Implemented in:

- `crates/nerva-core/src/lib.rs`
- `crates/nerva-runtime/src/lib.rs`

## Near-Term Porting Targets

| Area | Source | NERVA action |
|---|---|---|
| KV block pool | `vllm/vllm/v1/core/block_pool.py` | Recode block-table, refcount, eviction, and prefix-cache concepts on top of `ResidentBlock` and tiered KV pages. |
| KV admission | `vllm/vllm/v1/core/kv_cache_manager.py` | Recode allocation/admission logic without Python objects or Torch tensors. |
| Engine ticket | `rvllm/v3/crates/rvllm-runtime/src/engine.rs` | Recode launch/collect type-state pattern for NERVA transactions. |
| Graph pool | `rvllm/v3/crates/rvllm-graph/src/pool.rs` | Recode graph handles keyed by stable layout hash and block addresses. |
| Kernel manifests | `rvllm/v3/crates/rvllm-kernels/src/manifest.rs` | Recode fail-closed kernel-contract registry. |
| Sampling | `rvllm/v3/crates/rvllm-sampling/` and vLLM sampler paths | Recode deterministic reference checks and move steady-state token causality to device token ring. |

## Do Not Import

- vLLM Python scheduler/request objects as hot-path runtime state.
- vLLM Torch tensors or PyTorch allocator as the memory model.
- rvLLM Gemma-, FP8-, H100-, SM90-, or GB10-specific assumptions as NERVA defaults.
- Per-request graph recapture as the steady-state production policy.
