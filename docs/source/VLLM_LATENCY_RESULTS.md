# vLLM Post-Load Latency Findings

Date: 2026-06-25
Host: RTX 5090, 32 GB VRAM
Repo measured: `$VLLM_DIR`
Model: `Qwen/Qwen3-0.6B`
Scope: post-load, post-init, post-warmup inference latency only.

Set `$VLLM_DIR` to the local vLLM checkout used for reproducing these commands.

This report intentionally ignores model loading, weight download, torch compile,
CUDA graph capture, KV-cache allocation, and engine startup. Those can be slow,
but they are not the problem being studied here.

## Bottom Line

After the model is loaded and warm, vLLM answer latency is dominated by the
decode loop: repeated one-token model forward passes. For this workload, prompt
prefill is small compared with generating many output tokens.

For a single prompt with 128 input tokens and 64 output tokens:

```text
first-token path:          6.61 ms
full 64-token answer:    108.27 ms
remaining 63 tokens:     101.65 ms
decode cost/token:         1.61 ms/token after first token
```

So roughly 94% of the single-prompt answer latency is repeated decode after the
first token.

## Measured Latency

All numbers are from `vllm bench latency` after warmup. Detokenization was
disabled unless noted.

| Workload | Avg latency | Median | P90 |
|---|---:|---:|---:|
| batch 1, input 128, output 1 | 6.61 ms | 6.53 ms | 6.96 ms |
| batch 1, input 128, output 64 | 108.27 ms | 108.18 ms | 108.74 ms |
| batch 8, input 128, output 1 | 19.00 ms | 19.66 ms | 19.85 ms |
| batch 8, input 128, output 64 | 151.80 ms | 151.65 ms | 152.83 ms |
| batch 8, input 1, output 64 | 134.04 ms | 134.26 ms | 134.53 ms |
| batch 8, input 128, output 64, detok enabled | 151.45 ms | 151.18 ms | 152.49 ms |

Detokenization had no material effect in this benchmark shape.

## Qwen3-8B Short Decode Baseline

Additional measurement on 2026-06-27 used the local Qwen3-8B BF16 checkpoint
and the uv-managed vLLM environment at `/root/vllm/.venv`.

```text
model:          Qwen3-8B local safetensors snapshot
vLLM:           0.23.1rc1.dev455+g27da2a2ac
torch:          2.11.0+cu130
GPU:            RTX 5090
input length:   1 token
output length:  2 tokens
batch size:     1
detokenization: disabled
```

Measured with `vllm bench latency` after warmup and graph capture:

| Workload | Avg request latency | Median | P90 | P99 |
|---|---:|---:|---:|---:|
| Qwen3-8B, batch 1, input 1, output 2 | 22.39 ms | 22.24 ms | 22.75 ms | 23.32 ms |

Derived output throughput is `2 / 0.0223879596 = 89.33` generated tokens/s.
The request P99 divided by two output tokens is `11.66 ms/token`. This is a
derived comparison value, not a per-token vLLM device ledger.

After the CUDA decode QKV-prep, per-head attention, cuBLASLt packed projection
dispatch, and first-token prepare/RMSNorm fuse changes, the current NERVA
Qwen3-8B path measured about `97.16` tokens/s, `10.36 ms` token P99, and `327`
graph nodes per token on the same short decode shape. That beats this recorded
vLLM comparison for the fully resident single-GPU sample. rvLLM is recorded
separately as unsupported for this Qwen3 workload at the audited commit, so
NERVA does not claim a measured speedup over rvLLM for this exact model.

The current rvLLM baseline status is tracked separately in
`docs/source/RVLLM_BASELINE_RESULTS.md`.

The machine-readable comparison evidence used by acceptance is stored in:

```text
docs/source/perf/qwen3_8b_nerva_cuda_generate.json
docs/source/perf/qwen3_8b_vllm_latency.json
```

## Profiler Result

The single-prompt profile for 128 input tokens and 64 output tokens shows the
dominant CUDA work as:

```text
execute_context_0(0)_generation_1(1): 126.713 ms over 63 decode steps
GEMV/GEMM kernel bucket:               64.616 ms
aten::mm:                              12.374 ms
FlashAttention split-KV kernel:        10.421 ms
FlashAttention split-KV combine:        6.551 ms
prompt prefill context step:            8.152 ms
KV cache write reshape_and_cache:       2.662 ms
RMSNorm/activation Triton kernels:      low single-digit ms
sampling gumbel kernel:                 0.500 ms
HtoD/DtoH transfer:                     tiny relative to GEMM/attention
```

Interpretation:

```text
The answer path is mostly dense linear algebra during decode.
Attention is second.
KV-cache update, sampling, host/device copies, and detokenization are not the
dominant latency sources for this model/workload.
```

## What The Trace Labels Mean

vLLM labels profiler iterations in `vllm/v1/worker/gpu_worker.py`:

```text
execute_context_<num_context_requests>(<context_tokens>)_generation_<num_generation_requests>(<generation_tokens>)
```

Examples from the profile:

```text
execute_context_1(128)_generation_0(0)
```

This is the prompt prefill/context step.

```text
execute_context_0(0)_generation_1(1)
```

This is a decode step for one active generation request. This repeated decode
step dominates single-prompt answer latency.

## Relevant vLLM Code Paths

Benchmark timing only wraps `llm.generate()`:

```text
vllm/benchmarks/latency.py
run_to_completion()
```

Profiler iteration labeling:

```text
vllm/v1/worker/gpu_worker.py
GPUWorker.annotate_profile()
```

CUDA graph replay/capture wrapper:

```text
vllm/compilation/cuda_graph.py
CUDAGraphWrapper.__call__()
```

Attention/KV update dispatch:

```text
vllm/model_executor/layers/attention/attention.py
Attention.forward()
```

FlashAttention backend KV-cache write:

```text
vllm/v1/attention/backends/flash_attn.py
FlashAttentionBackendImpl.write_to_kv_cache()
```

## Commands Used

Environment setup:

```bash
UV_CACHE_DIR=$VLLM_DIR/.uv-cache uv venv --python 3.12
UV_CACHE_DIR=$VLLM_DIR/.uv-cache \
  VLLM_TARGET_DEVICE=cuda \
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e8ec3d52ab4e163501c8c7bee8c03ca8359a7a/vllm-0.23.1rc1.dev454%2Bga2e8ec3d5-cp38-abi3-manylinux_2_28_x86_64.whl \
  uv pip install -e . --torch-backend=auto
```

Single-prompt first-token benchmark:

```bash
HF_HOME=$VLLM_DIR/.hf-cache \
UV_CACHE_DIR=$VLLM_DIR/.uv-cache \
VLLM_TARGET_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m vllm.entrypoints.cli.main bench latency \
  --model Qwen/Qwen3-0.6B \
  --input-len 128 \
  --output-len 1 \
  --batch-size 1 \
  --num-iters-warmup 3 \
  --num-iters 10 \
  --max-model-len 1024 \
  --disable-detokenize \
  --output-json $VLLM_DIR/.bench/single_bs1_i128_o1.json
```

Single-prompt 64-token benchmark:

```bash
HF_HOME=$VLLM_DIR/.hf-cache \
UV_CACHE_DIR=$VLLM_DIR/.uv-cache \
VLLM_TARGET_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m vllm.entrypoints.cli.main bench latency \
  --model Qwen/Qwen3-0.6B \
  --input-len 128 \
  --output-len 64 \
  --batch-size 1 \
  --num-iters-warmup 3 \
  --num-iters 10 \
  --max-model-len 1024 \
  --disable-detokenize \
  --output-json $VLLM_DIR/.bench/single_bs1_i128_o64.json
```

Single-prompt profiler:

```bash
HF_HOME=$VLLM_DIR/.hf-cache \
UV_CACHE_DIR=$VLLM_DIR/.uv-cache \
VLLM_TARGET_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m vllm.entrypoints.cli.main bench latency \
  --model Qwen/Qwen3-0.6B \
  --input-len 128 \
  --output-len 64 \
  --batch-size 1 \
  --num-iters-warmup 3 \
  --num-iters 1 \
  --max-model-len 1024 \
  --disable-detokenize \
  --profile \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir $VLLM_DIR/.bench/profile_single_bs1_i128_o64 \
  --profiler-config.torch_profiler_with_stack false \
  --profiler-config.torch_profiler_use_gzip false
```

Qwen3-8B short decode benchmark:

```bash
cd $VLLM_DIR
.venv/bin/python -m vllm.entrypoints.cli.main bench latency \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype bfloat16 \
  --max-model-len 3 \
  --input-len 1 \
  --output-len 2 \
  --batch-size 1 \
  --num-iters-warmup 3 \
  --num-iters 10 \
  --disable-detokenize \
  --output-json /tmp/vllm_qwen3_8b_latency_i1_o2_b1.json
```

## Local Artifacts

```text
$VLLM_DIR/.bench/single_bs1_i128_o1.json
$VLLM_DIR/.bench/single_bs1_i128_o64.json
$VLLM_DIR/.bench/decode_bs8_i128_o64.json
$VLLM_DIR/.bench/prefill_ttft_bs8_i128_o1.json
$VLLM_DIR/.bench/decode_bs8_i1_o64.json
$VLLM_DIR/.bench/decode_bs8_i128_o64_detok.json
$VLLM_DIR/.bench/profile_single_bs1_i128_o64/profiler_out_0.txt
$VLLM_DIR/.bench/profile_single_bs1_i128_o64/rank0.*.pt.trace.json
$VLLM_DIR/.bench/profile_decode_bs8_i128_o64/profiler_out_0.txt
$VLLM_DIR/.bench/profile_decode_bs8_i128_o64/rank0.*.pt.trace.json
```

## Practical Conclusion

For this model and GPU, optimizing post-load latency means attacking decode:

```text
1. reduce per-token GEMV/GEMM cost
2. reduce attention split-KV cost
3. reduce CUDA graph / launch / per-step overhead for batch 1
4. consider exact speculative decoding only if target verification remains cheap
5. keep KV writes and sampling in perspective; they are not the primary wall here
```

The current data does not support focusing first on model loading, PCIe
bandwidth, detokenization, or sampling for this measured latency problem.

