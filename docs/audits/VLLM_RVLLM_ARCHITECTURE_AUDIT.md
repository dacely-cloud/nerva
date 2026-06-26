# NERVA Audit: vLLM and rvLLM

Date: 2026-06-26

Audited inputs:

- vLLM: `vLLM checkout`, commit `3903286619f507567637a412cc1187c44ddf3cb0`.
- rvLLM: `rvLLM checkout`, commit `17b1c85dff7cea3cc6259f19fce394d6cfea002e`.
- vLLM working tree was dirty during the audit. The audit therefore describes the current local tree, not pristine upstream. Modified local files include `vllm/v1/engine/core.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/async_utils.py`, and single-GPU ledger tools.

## Executive Conclusion

NERVA should use vLLM as the compatibility and baseline oracle, and rvLLM as a structural reference for Rust-owned CUDA execution. Neither codebase implements NERVA's core abstraction: explicit multi-tier `ResidentBlock`s for weights, KV, activations, token state, sampler state, and ledgers.

The key architectural result is:

- vLLM is a mature Python/Torch inference machine with excellent scheduling, model coverage, attention backends, and serving compatibility, but its decode loop is still host/Python-owned.
- rvLLM removes Python from the hot serving path and has useful static arenas, graph handles, kernel manifests, and Rust CUDA ownership, but its strongest path is specialized around Gemma 4, FP8, H100/SM90 and GB10/SM121 work.
- NERVA should not inherit host-visible token commits as the steady-state token loop. Decode should become a device-resident transaction, with host-visible output as a ledgered side effect.

## vLLM Summary

vLLM V1 process architecture is:

```text
OpenAI API server
-> AsyncLLM / EngineClient
-> EngineCore or EngineCoreProc
-> Scheduler
-> ModelExecutor
-> Worker
-> GPUModelRunner
-> PyTorch model + custom ops + attention backend
-> Sampler
-> ModelRunnerOutput / AsyncGPUModelRunnerOutput
-> Scheduler.update_from_output
```

Primary code paths:

- `vllm/vllm/entrypoints/openai/api_server.py:79`: `build_async_engine_client`.
- `vllm/vllm/entrypoints/openai/api_server.py:692`: `run_server`.
- `vllm/vllm/entrypoints/openai/api_server.py:708`: `run_server_worker`.
- `vllm/vllm/v1/engine/utils.py:1025`: `get_engine_zmq_addresses`.
- `vllm/vllm/v1/engine/core.py:487`: `EngineCore.step`.
- `vllm/vllm/v1/engine/core.py:942`: `EngineCoreProc`.
- `vllm/vllm/v1/engine/core.py:1305`: `EngineCoreProc.run_busy_loop`.
- `vllm/vllm/v1/engine/core.py:1346`: `_process_engine_step`.
- `vllm/vllm/v1/engine/core.py:1635`: `process_output_sockets`.

The scheduling step is explicitly host-owned. `EngineCore.step` calls `scheduler.schedule`, then `model_executor.execute_model`, then waits on `future.result()`, calls `sample_tokens` if needed, and finally advances request state with `scheduler.update_from_output`.

vLLM model execution path:

- `vllm/vllm/v1/worker/gpu_worker.py:250`: `Worker.init_device` sets device/distributed state and constructs `GPUModelRunner`.
- `vllm/vllm/v1/worker/gpu_worker.py:377`: `Worker.load_model`.
- `vllm/vllm/v1/worker/gpu_worker.py:400`: `Worker.determine_available_memory`.
- `vllm/vllm/v1/worker/gpu_worker.py:591`: `Worker.initialize_from_config`.
- `vllm/vllm/v1/worker/gpu_worker.py:835`: `Worker.execute_model`.
- `vllm/vllm/v1/worker/gpu_model_runner.py:4101`: `GPUModelRunner.execute_model`.
- `vllm/vllm/v1/worker/gpu_model_runner.py:4482`: `GPUModelRunner.sample_tokens`.

Token state in vLLM is still host-authoritative:

- `vllm/vllm/v1/worker/gpu_input_batch.py:92`: `InputBatch` owns request rows, CPU token tensors, pinned CPU metadata, and block tables.
- `vllm/vllm/v1/worker/gpu_input_batch.py:1002`: `set_async_sampled_token_ids` stores async CPU copy state when logits processors need previous output ids.
- `vllm/vllm/v1/worker/gpu_model_runner.py:3658`: `_bookkeeping_sync` copies or stages sampled ids, updates `InputBatch.token_ids_cpu`, `InputBatch.num_tokens_no_spec`, and `CachedRequestState.output_token_ids`.
- `vllm/vllm/v1/worker/gpu_model_runner.py:1758`: `_prepare_input_ids` can reuse `prev_sampled_token_ids` on GPU for async scheduling, but it falls back to CPU tensor copies and may allocate pinned index tensors on reorder/spec paths.

vLLM CUDA graph handling:

- `vllm/vllm/compilation/cuda_graph.py:145`: `CUDAGraphWrapper`.
- `vllm/vllm/compilation/cuda_graph.py:231`: `CUDAGraphWrapper.__call__`.
- `vllm/vllm/compilation/cuda_graph.py:283`: creates `torch.cuda.CUDAGraph`.
- `vllm/vllm/compilation/cuda_graph.py:360`: `entry.cudagraph.replay()`.
- `vllm/vllm/v1/cudagraph_dispatcher.py:235`: `CUDAGraphDispatcher.dispatch`.
- `vllm/vllm/v1/cudagraph_dispatcher.py:326`: `get_capture_descs`.

vLLM graph capture accelerates model-forward subgraphs inside a Python-owned engine step. It is not a device-resident decode transaction.

vLLM KV cache:

- `vllm/vllm/v1/core/kv_cache_manager.py:110`: `KVCacheManager`.
- `vllm/vllm/v1/core/kv_cache_manager.py:202`: `get_computed_blocks`.
- `vllm/vllm/v1/core/kv_cache_manager.py:244`: `allocate_slots`.
- `vllm/vllm/v1/core/kv_cache_manager.py:588`: `cache_blocks`.
- `vllm/vllm/v1/core/block_pool.py:144`: `BlockPool`.
- `vllm/vllm/v1/core/block_pool.py:542`: `get_new_blocks`.
- `vllm/vllm/v1/core/block_pool.py:597`: `touch`.
- `vllm/vllm/v1/core/block_pool.py:614`: `free_blocks`.
- `vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:161`: `allocate_uniform_kv_caches`.
- `vllm/vllm/v1/worker/gpu_model_runner.py:7094`: `_allocate_kv_cache_tensors`.

vLLM has a strong logical KV block manager and prefix cache. Physical KV is still Torch tensor storage allocated by workers, plus block tables consumed by attention kernels. It is not a multi-tier virtual memory system for KV.

vLLM custom ops and kernels:

- `vllm/vllm/model_executor/custom_op.py:86`: `CustomOp`.
- `vllm/vllm/model_executor/custom_op.py:140`: per-platform dispatch through `forward_cuda`, `forward_hip`, `forward_cpu`, etc.
- `vllm/vllm/_custom_ops.py`: wrappers for `torch.ops._C`, `_C_cache_ops`, `_moe_C`, `_C_custom_ar`, etc.
- `vllm/csrc/`: CUDA, ROCm, CPU, cache, attention, MoE, quantization, CUTLASS, and custom-all-reduce sources.
- `vllm/vllm/v1/attention/backends/registry.py`: attention backend registry.
- `vllm/vllm/v1/attention/backends/triton_attn.py`, `flash_attn.py`, `flashinfer.py`, `cpu_attn.py`, `mla/*`: attention implementations.

Conceptually useful pieces: PagedAttention, block-table contracts, kernel/backend registry, prefix cache policy, model coverage, OpenAI-compatible serving. Not directly reusable for NERVA's runtime because the control plane and tensor ownership are Torch/Python-centered.

## rvLLM Summary

rvLLM workspace root:

- `rvllm/v3/Cargo.toml`.

Crate map:

| Crate | Role found in audit |
|---|---|
| `rvllm-core` | Core ids, errors, dtypes, compile targets, config. |
| `rvllm-mem` | CUDA context, streams, HBM arena, unified arena, pinned host buffers, graph-safe handles, KV layout. |
| `rvllm-kernels` | Kernel manifest verification, PTX/module loading, kernel function handles, architecture dispatch helpers. |
| `rvllm-cutlass` | CUTLASS policy, variants, cuBLASLt wrappers, W4A8 and shared-library loading. |
| `rvllm-attention` | Paged decode/prefill launchers and backend enum. |
| `rvllm-fused` | Low-level `cuLaunchKernel` wrapper and fused-kernel launchers/references. |
| `rvllm-metadata` | Frozen per-bucket metadata layouts, hashes, packing plan. |
| `rvllm-graph` | Captured CUDA graph handles and graph pool keyed by bucket/max blocks. |
| `rvllm-loader` | HF safetensors loaders, Gemma 4 arch/weights, FP8 quantization, generic loader. |
| `rvllm-sampling` | Sampling params, deterministic host sampler, device top-k launcher, greedy speculative accept launcher/reference. |
| `rvllm-runtime` | Engine type-state API, scheduler, generic bring-up, Gemma 4 production path. |
| `rvllm-serve` | Serving crate. |
| `rvllm-mcp` | MCP integration crate. |
| `rvllm-bench` | Benchmark crate. |
| `rvllm-deploy` | Deploy scaffold. |
| `rvllm-invariants` | Invariant/test crate. |
| `rvllm-vision` | Vision support crate. |
| `rvllm-imageio` | Image IO crate. |
| `rvllm-metal` | Apple Metal path. |

rvLLM hot path ownership:

- `rvllm/v3/crates/rvllm-runtime/src/engine.rs:1`: type-state engine design.
- `rvllm/v3/crates/rvllm-runtime/src/engine.rs:31`: `Engine`.
- `rvllm/v3/crates/rvllm-runtime/src/engine.rs:43`: `step_launch`.
- `rvllm/v3/crates/rvllm-runtime/src/engine.rs:70`: `PendingStep::collect`.
- `rvllm/v3/crates/rvllm-runtime/src/scheduler.rs:19`: `BatchPlan`.
- `rvllm/v3/crates/rvllm-runtime/src/scheduler.rs:34`: `Scheduler`.
- `rvllm/v3/crates/rvllm-runtime/src/sched_state.rs:10`: `ReqState`.

The abstract engine is a good NERVA pattern: one launch ticket, borrow-checker prevents double launch, and collection is explicit. The production path is not fully generalized through this type-state API; the current fastest implementation is in `gemma4_bring_up.rs`.

rvLLM production Gemma path:

- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:249`: `Gemma4Bringup`.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:316`: `Gemma4Bringup::load`.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:3159`: `run_generate`.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:3188`: `run_generate_sampled`.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:3220`: `run_generate_inner`.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:5508`: greedy CUDA graph decode fast path.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:5320`: speculative decode loop.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:5844`: `ngram_draft`.

rvLLM memory system:

- `rvllm/v3/crates/rvllm-mem/src/context.rs:1`: CUDA primary context ownership.
- `rvllm/v3/crates/rvllm-mem/src/stream.rs:1`: non-blocking CUDA stream wrapper.
- `rvllm/v3/crates/rvllm-mem/src/hbm.rs:1`: `HbmArena`, one `cuMemAlloc` slab, bump allocation, stable graph-safe pointers.
- `rvllm/v3/crates/rvllm-mem/src/pinned.rs:1`: `PinnedBuf` and `PinnedPool`.
- `rvllm/v3/crates/rvllm-mem/src/unified.rs:1`: `UnifiedArena` for GB10/DGX Spark managed memory.
- `rvllm/v3/crates/rvllm-mem/src/capture.rs:1`: `CaptureScope`, `GraphSafe`, and bind-only graph handles.
- `rvllm/v3/crates/rvllm-mem/src/kv_layout.rs:1`: KV layout `[2, num_blocks, block_size, num_kv_heads, head_dim]`.

rvLLM has static arenas and stable graph-bound regions. It does not have NERVA `ResidentBlock`s: there is no explicit object that can migrate between VRAM, pinned DRAM, DRAM, and disk while retaining semantic identity and ledgered residency decisions.

rvLLM CUDA graph path:

- `rvllm/v3/crates/rvllm-graph/src/pool.rs:22`: `CapturedGraph`.
- `rvllm/v3/crates/rvllm-graph/src/pool.rs:39`: `CapturedGraph::capture`.
- `rvllm/v3/crates/rvllm-graph/src/pool.rs:110`: `CapturedGraph::replay`.
- `rvllm/v3/crates/rvllm-graph/src/pool.rs:156`: `GraphPool`.
- `rvllm/v3/crates/rvllm-metadata/src/layout.rs:1`: `MetadataLayout` and stable hash.

The graph crate states a strict pre-captured graph pool model. The production Gemma path captures per request/chunk after eager warm-up in `gemma4_bring_up.rs` (`CapturedGraph::capture` near lines 5207, 5368, 5396, and 5541). NERVA should keep the graph-hash and graph-owned-pointer invariants, but avoid per-request recapture in the production path.

rvLLM sampling/token state:

- `rvllm/v3/crates/rvllm-sampling/src/params.rs`: per-request sampling params.
- `rvllm/v3/crates/rvllm-sampling/src/sampler.rs:1`: device top-k plus deterministic host draw.
- `rvllm/v3/crates/rvllm-sampling/src/sampler.rs:86`: `SampleTopKLaunch`.
- `rvllm/v3/crates/rvllm-sampling/src/spec_accept.rs:1`: greedy speculative accept reference/launcher.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:5697`: `SampleTailState`.
- `rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs:5720`: `sample_tail_step`.

rvLLM's greedy graph path feeds the next token device-to-device, but harvests tokens to pinned host memory for EOS and output. The sampled path intentionally fences every step because the host finishes sampling over compact candidates. NERVA should move sampler and EOS state into a device-resident token transaction for the steady-state path.

rvLLM kernels and hardware:

- `rvllm/v3/crates/rvllm-core/src/arch.rs`: compile targets `Sm80`, `Sm89`, `Sm90`, `Sm121`; rejects `sm_120` RTX 5090 and `sm_122`; no `sm_75` or `sm_86`.
- `rvllm/kernels/build.sh`: build script comments list `sm_75`, `sm_86`, `sm_120`, etc., but runtime target enum does not support all of them.
- `rvllm/kernels/build_cutlass_sm120_so.sh`: useful Blackwell/RTX 5090 CUTLASS setup, not wired as a runtime target.
- `rvllm/v3/crates/rvllm-kernels/src/manifest.rs`: SHA-pinned kernel artifacts.
- `rvllm/v3/crates/rvllm-kernels/src/loader.rs`: only loads verified manifest entries.
- `rvllm/v3/crates/rvllm-fused/src/launch_raw.rs`: typed `cuLaunchKernel` wrapper.
- `rvllm/v3/crates/rvllm-attention/src/decode.rs`: paged decode launcher, FA3/FA2/Metal branches.
- `rvllm/v3/crates/rvllm-cutlass/src/policy.rs`: fail-closed autotune policy lookup.
- `rvllm/v3/crates/rvllm-cutlass/src/cublaslt.rs`: cuBLASLt FP8/F16 wrappers.

## Code Path Tables

### vLLM

| Area | Code path | Audit conclusion |
|---|---|---|
| Serving | `api_server.py -> AsyncLLM.from_vllm_config` | Keep as compatibility oracle, not runtime foundation. |
| Engine loop | `EngineCore.step` | Python schedules, waits, samples, and updates request state. |
| Worker | `Worker.execute_model -> GPUModelRunner.execute_model` | PyTorch-owned execution. |
| Token state | `InputBatch`, `CachedRequestState`, `_bookkeeping_sync` | Host-authoritative; async path only avoids some token round trips. |
| Host sync | `AsyncGPUModelRunnerOutput.get_output`, `AsyncOutput.get_output` | Explicit event sync before host-visible output. |
| CUDA graphs | `CUDAGraphWrapper`, `CUDAGraphDispatcher` | Forward acceleration inside Python engine loop. |
| KV | `KVCacheManager`, `BlockPool`, block tables | Strong logical paging, not multi-tier VM. |
| Kernels | `_custom_ops.py`, `csrc/`, attention backends | Rich concepts, Torch-bound implementation. |

### rvLLM

| Area | Code path | Audit conclusion |
|---|---|---|
| Runtime API | `rvllm-runtime/src/engine.rs` | Good type-state pattern, not the full production path. |
| Production path | `gemma4_bring_up.rs` | Fast Rust/CUDA path, too Gemma-specific. |
| CUDA context | `rvllm-mem/src/context.rs` | Good single-owner context model. |
| Streams | `rvllm-mem/src/stream.rs` | Good non-blocking stream wrapper. |
| Arenas | `rvllm-mem/src/hbm.rs`, `unified.rs` | Useful static slabs, not ResidentBlock residency. |
| Graphs | `rvllm-graph/src/pool.rs` | Useful graph handles and layout-hash replay checks. |
| Metadata | `rvllm-metadata/src/layout.rs` | Useful frozen per-bucket layout. |
| Kernels | `rvllm-kernels`, `rvllm-fused`, `rvllm-cutlass` | Useful manifest/loader/launch discipline. |
| Sampling | `rvllm-sampling` | Good deterministic sampling reference; host tail remains. |

## Hot Path Call Graphs

### vLLM Decode Step

```text
EngineCoreProc.run_busy_loop
-> _process_input_queue
-> _process_engine_step
-> EngineCore.step
   -> Scheduler.schedule
   -> ModelExecutor.execute_model(non_block=True)
      -> Worker.execute_model
         -> GPUModelRunner.execute_model
            -> _update_states
            -> _prepare_inputs
            -> _model_forward
            -> compute_logits
            -> save ExecuteModelState
   -> future.result()
   -> ModelExecutor.sample_tokens
      -> GPUModelRunner.sample_tokens
         -> _sample
         -> _bookkeeping_sync
         -> ModelRunnerOutput or AsyncGPUModelRunnerOutput
   -> Scheduler.update_from_output
-> output_queue / ZMQ
```

### rvLLM Gemma 4 Greedy Decode

```text
Gemma4Bringup::run_generate_sampled
-> run_generate_inner
   -> prefill / first token path
   -> seed token_ids_region
   -> prepare_decode_inputs
   -> decode_forward eager step
   -> fence + DtoH first token
   -> CapturedGraph::capture(|| decode_forward())
   -> loop:
      -> prepare_decode_inputs
      -> graph.replay(stream)
      -> async DtoH token harvest into PinnedBuf
      -> fence every HARVEST_EVERY steps
      -> host EOS/output update
-> stream.fence
-> arena.restore(checkpoint)
```

## Memory Ownership Comparison

vLLM physical memory is owned by PyTorch tensors and CUDA graph pools. `KVCacheManager` and `BlockPool` own logical block assignment, not physical residency. `allocate_uniform_kv_caches` and `_allocate_kv_cache_tensors` allocate Torch tensors for KV storage.

rvLLM owns CUDA memory directly through `HbmArena`, `UnifiedArena`, `PinnedBuf`, and typed device pointers. This is closer to NERVA, but still lacks semantic residency objects, multi-tier migration, ledgered eviction, disk backing, and warm DRAM compute.

NERVA decision: start with rvLLM-style direct allocation, but wrap every allocation in `ResidentBlock` identity and residency metadata from day one.

## Token State Comparison

vLLM's next token is host-visible and scheduler-visible. Async scheduling can reuse the sampled-token GPU tensor for the next input, but the authoritative request state is still host `InputBatch` and scheduler state.

rvLLM's greedy graph path has a stronger device feedback loop: graph decode writes token feedback device-to-device, while host token harvest is batched through pinned memory. However EOS, output assembly, and sampled-tail decisions remain host-visible.

NERVA decision: token ring, stop masks, sampler state, and accepted-token count must be device-resident with a host ledger, not host-owned with device acceleration.

## CUDA Graph Comparison

vLLM: graph capture/replay is a PyTorch wrapper for compiled forward regions, dispatched by Python context and batch descriptors.

rvLLM: graph capture/replay is a Rust CUDA handle with explicit layout hashes and graph exec ownership. Production Gemma code still captures after eager warm-up in request paths.

NERVA decision: use rvLLM-style graph handles and vLLM-style bucket dispatch concepts, but predeclare stable graph contracts around ResidentBlock addresses and token transaction layouts.

## KV Cache Comparison

vLLM: mature logical KV paging and prefix caching. Physical cache lives in GPU Torch tensors; block tables point attention kernels at pages.

rvLLM: fixed KV layout and arena-backed pages for its model path. No general preemption, prefix cache, remote KV connector, or multi-tier KV VM.

NERVA decision: borrow vLLM's logical block/prefix cache ideas and rvLLM's static pointer discipline, then implement KV cache as virtual memory over `ResidentBlock`s.

## Kernel Strategy Comparison

vLLM: broad backends and hardware coverage through Torch, Triton, CUDA C++, ROCm, CPU kernels, FlashAttention, FlashInfer, CUTLASS, and custom ops.

rvLLM: narrower but more reproducible kernel ownership: manifest-verified PTX/shared libraries, typed launchers, cuBLASLt wrappers, fail-closed policies, and model-specific fused kernels.

NERVA decision: use rvLLM's artifact discipline and vLLM's backend coverage as the target surface. Do not copy rvLLM's model-specific kernel sprawl as the main abstraction.

## Serving/Compatibility Comparison

vLLM is the compatibility oracle: OpenAI API, tokenizer/HF integration, model coverage, scheduling options, metrics, and benchmark ecosystem.

rvLLM has a serving story, but the audit did not find parity with vLLM's model/API breadth.

NERVA decision: early serving should be compatibility-shaped like vLLM, while runtime internals stay Rust/CUDA-first.

## Required Decision Table

| Area | vLLM | rvLLM | NERVA decision |
|---|---|---|---|
| runtime language | Python + Torch + C++/CUDA/Triton | Rust + CUDA, some JAX/TPU and Metal | Rust core with C/CUDA native boundary. |
| hot path owner | Python engine/scheduler | Rust production path for Gemma | Rust runtime owns hot path; Python only compatibility shim later. |
| request scheduler | EngineCore scheduler and Python request state | Narrow serving/runtime scheduler | Start with single-owner Rust scheduler; defer vLLM-compatible serving until runtime invariants pass. |
| GPU context ownership | Worker and GPUModelRunner through Torch/CUDA platform setup | Rust runtime/context wrappers own CUDA context and streams | CUDA-specific ownership stays in `nerva-cuda`; core runtime sees typed backend handles only. |
| GPU execution | Torch modules and custom ops | Direct driver/cuBLASLt/CUTLASS/PTX | Direct CUDA first. |
| CUDA graphs | PyTorch CUDA graph wrappers | Rust `CapturedGraph` handles | Rust graph executor with typed layout hashes. |
| graph capture/replay | Captured/replayed inside Python-owned model runner path | Rust graph pool keyed by layout/bucket | Predeclared NERVA transactions with stable ResidentBlock addresses and ledgered graph replay. |
| memory arenas | PyTorch allocator plus caches | HBM/unified bump arenas | Static arenas underneath `ResidentBlock` residency manager. |
| static arenas | Mostly framework allocator and preallocated tensors | HBM, unified, and pinned arenas | All hot-path arenas are preallocated; workspace reset is constant-time. |
| hot-path allocation | Python/Torch objects and selected tensor paths can still allocate | Stronger arena discipline, but model-specific | NERVA hot path rejects allocator events and ledgers violations. |
| KV cache | mature logical paged cache | fixed model KV layout | KV virtual memory over ResidentBlocks. |
| token state | host/scheduler authoritative | device feedback plus host harvest | device-resident token ring and sampler transaction. |
| token source of truth | Host scheduler and request state | Partly device feedback for greedy graph path | Device token ring is authoritative for next decode step; host output is a replica. |
| sampling | mature sampler, host-visible output | deterministic host tail plus device top-k | device sampler first, host audit ledger second. |
| host output handoff | Async output copy/event sync before Python output conversion | Pinned host harvest for output/EOS checks | SoftVisibilitySync is ledgered and not a dependency for device-fast-path decode. |
| model loading | broad HF ecosystem | generic loader plus Gemma-specialized path | start with HF metadata parser, keep model-specific lowering behind traits. |
| weight loading | Torch/HF loader materializes framework tensors | Loader plus model-specific packed paths | Canonical weight metadata first; packed replicas are backend contract outputs, not source of truth. |
| kernel strategy | broad backend registry | manifest-pinned owned kernels | manifest-pinned native kernels, backend traits, no silent fallback. |
| kernel contracts | CustomOp/backend dispatch via framework integration | Manifest-pinned kernels and launchers | Explicit kernel contracts declare dtype/layout/graph safety/fallback exactness. |
| silent fallback behavior | Framework fallback risk exists unless audited per path | Mostly fail-closed for owned paths | Missing optimized paths must select named exact fallback or planning failure. |
| CUDA portability | Broad NVIDIA path through Torch/native ops | CUDA-first, newer GPU focused | CUDA backend is capability-gated; no CUDA type leaks into core abstractions. |
| AMD/HIP portability | ROCm platform split exists in vLLM | No clean first-class HIP runtime found | Keep backend contracts HIP-capable; implement `nerva-hip` only after CUDA contracts stabilize. |
| serving API | mature OpenAI server | narrower serving crates | vLLM-compatible API surface after core smoke. |
| model coverage | very broad | strongest on Gemma 4 | do not begin Gemma-only; encode portability gates. |
| old hardware viability | broader via Torch backends | runtime lacks SM75/SM86 targets | support Linux x86_64/aarch64 first; GPU backends explicitly gated by compute target. |
| exact FP16/BF16 viability | strong via Torch/native kernels | present but FP8-optimized | FP16/BF16 exact path first, FP8 optional. |
| DRAM warm-tier compute | not a first-class tier | no ResidentBlock warm tier | implement warm-tier CPU compute as NERVA-specific work. |
| transport assumptions | Distributed abstractions lean on vLLM/PyTorch ecosystem | No general NERVA-Fabric transport contract | Transport moves named block versions and exposes direct GPU paths only when verified. |
| ResidentBlock compatibility | absent | absent | foundational abstraction. |

## Required Questions

1. Where does vLLM decide the next input token?

   In `GPUModelRunner.sample_tokens` (`vllm/vllm/v1/worker/gpu_model_runner.py:4482`) via `_sample` and `_bookkeeping_sync`, then host scheduler state is advanced in `EngineCore.step` through `scheduler.update_from_output` (`vllm/vllm/v1/engine/core.py:487`). Async scheduling can stage `prev_sampled_token_ids` on GPU for `_prepare_input_ids`, but request authority returns to the host.

2. Is vLLM's next token host-owned or device-owned?

   Host-owned. There are device tensors for sampled ids and async reuse, but `InputBatch`, `CachedRequestState.output_token_ids`, `ModelRunnerOutput`, and scheduler state are host/Python authoritative.

3. Where exactly does vLLM synchronize host output?

   In `AsyncGPUModelRunnerOutput.get_output` in `vllm/vllm/v1/worker/gpu_model_runner.py` where the async copy event is synchronized before CPU tensors are converted to Python lists. The V2 equivalent is `AsyncOutput.get_output` in `vllm/vllm/v1/worker/gpu/async_utils.py`, which synchronizes `copy_event`.

4. Can vLLM decode continue without host-visible token state?

   Not as the steady-state contract. Async scheduling can reuse GPU sampled tokens for common unchanged batches, but scheduler update, output handling, logprob paths, EOS/finish state, and many spec paths require host-visible token state.

5. Where does vLLM allocate during decode?

   Most large buffers are preallocated in `GPUModelRunner.__init__`, `InputBatch`, and KV initialization. Real decode-path allocations still occur in Python objects/lists/dicts for scheduler/model outputs and in `_prepare_input_ids` (`vllm/vllm/v1/worker/gpu_model_runner.py:1758`) when async scheduling reorder/spec paths create pinned `torch.tensor(...)` index tensors and transfer them to GPU.

6. Which vLLM allocations are real and which profiler/ATen artifacts?

   Real: `torch.tensor(..., pin_memory=PIN_MEMORY).to(self.device)` index tensors in async reorder/spec paths; `ModelRunnerOutput`/`AsyncGPUModelRunnerOutput`; CPU list/dict construction in `_bookkeeping_sync`; logprob/routed-expert CPU copies when enabled; KV cache Torch tensors during initialization. Likely profiler/ATen artifacts: dispatcher/temporary traces around preallocated buffer copies, CUDA graph replay bookkeeping, event/stream operations, and PyTorch allocator accounting that does not correspond to a new semantic decode buffer.

7. How does rvLLM keep Python out of hot path?

   The production generation path is Rust in `Gemma4Bringup::run_generate_sampled` and `run_generate_inner`; CUDA launches go through Rust wrappers, cuBLASLt, CUTLASS, loaded PTX, and `CapturedGraph`. No Python loop mediates token generation.

8. How does rvLLM own CUDA streams/graphs?

   `CudaContextHandle::init` retains and sets a CUDA primary context; `Stream::new` creates a non-blocking stream; `CapturedGraph::capture` calls CUDA driver graph capture/instantiate; `CapturedGraph::replay` calls `cuGraphLaunch`; `Drop` destroys graph exec handles.

9. Does rvLLM have static arena model?

   Yes. `HbmArena` is one stable `cuMemAlloc` slab with bump-allocated `Region`s; `UnifiedArena` mirrors that for GB10 managed memory; `PinnedBuf` owns pinned host memory. These are static arenas, not a residency VM.

10. Does rvLLM have anything equivalent to ResidentBlock?

   No. It has stable regions and tensors, but not semantic blocks with tier, residency policy, prefetch/evict state, ledger identity, and compute-near-data decisions.

11. Does rvLLM graph design generalize beyond Gemma/FP8/H100?

   The graph handle, layout hash, and graph pool concepts generalize. The production captured graph body is highly Gemma 4/FP8/kernel-set specific. Also, runtime compile targets are limited to SM80, SM89, SM90, and SM121.

12. Which rvLLM code useful for RTX 5090?

   Useful: `kernels/build_cutlass_sm120_so.sh`, Blackwell comments in `kernels/build.sh`, manifest/loader design, cuBLASLt wrappers, generic arena/graph infrastructure. Missing: `CompileTarget` rejects compute capability 12.0 (`sm_120`), so RTX 5090 needs target plumbing, kernels, attention backend validation, and policy files.

13. Which rvLLM code fail/underperform on 2080 Ti?

   Runtime compile target does not include SM75, so bring-up should fail. Even if PTX is built manually, FP8/BF16 tensor-core assumptions, FA3/Hopper kernels, cuBLASLt FP8 paths, and SM90/SM121-specific kernels will fail or underperform. NERVA needs explicit FP16 fallback kernels for Turing if 2080 Ti is supported later.

14. What should NERVA copy conceptually?

   Rust-owned CUDA context/stream/graph handles; static arena discipline; graph-safe pointer lifetime rules; manifest-verified kernel artifacts; fail-closed kernel/policy lookup; deterministic sampling references; vLLM's API compatibility, scheduler lessons, block-table/KV paging concepts, and benchmark discipline.

15. What should NERVA never inherit?

   vLLM's Python-owned token loop, host-authoritative decode state, Torch allocator as runtime memory model, and implicit host syncs. From rvLLM, never inherit Gemma-only architecture, FP8/H100 assumptions as defaults, per-request graph recapture as production steady state, or lack of a first-class residency abstraction.

## Final Recommendation

Build NERVA as a new Rust/CUDA Linux runtime. Start with a small Linux-only workspace and a CUDA smoke path, but make the architectural contracts explicit now:

- `ResidentBlock` is the unit of ownership and ledgering.
- GPU execution is backend-trait based, not model-specific global code.
- Token decode is a device-resident transaction.
- KV cache is virtual memory, not just a block table.
- CUDA graph replay is tied to stable metadata layout and ResidentBlock addresses.
- vLLM remains the behavior/performance oracle.
- rvLLM remains the Rust/CUDA implementation reference, not code to copy.
