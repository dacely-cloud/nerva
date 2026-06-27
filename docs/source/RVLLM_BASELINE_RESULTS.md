# rvLLM Baseline Status

Date: 2026-06-27
Host: RTX 5090, 32 GB VRAM
rvLLM checkout: `/root/rvllm`
rvLLM commit: `17b1c85dff7cea3cc6259f19fce394d6cfea002e`
NERVA comparison workload: Qwen3-8B BF16, batch 1, input 1 token, output 2 tokens

## Status

The rvLLM baseline for this exact workload is not comparable at this commit.

The unmodified `rvllm-bench` build reached the local rvLLM source and CUDA
toolchain, then failed during Rust compilation before a benchmark binary was
available:

```text
crate: rvllm-loader
error: missing fields in Gemma4LayerWeights initializer
error: missing fields in Gemma4LoadedModel initializer
```

A temporary copy of the same rvLLM commit was then patched only to fill the
stale optional Gemma4 PLE initializer fields with `None` / zero defaults. That
patched copy built `rvllm-bench`, but the benchmark still exited before
inference on the Qwen3-8B checkpoint:

```text
rvllm-bench: config parse ... unsupported architecture: Qwen3ForCausalLM
```

This means the local rvLLM checkout is useful as a Rust/CUDA architecture
reference, but it does not provide a measured Qwen3-8B baseline for this
workload.

The failed command shape was:

```bash
CUDA_HOME=/usr/local/cuda-13.1 \
PATH=/usr/local/cuda-13.1/bin:$PATH \
CARGO_TARGET_DIR=/tmp/rvllm-target \
RVLLM_MODEL_DIR=/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
RVLLM_KERNELS_DIR=/root/rvllm/kernels/sm_121 \
RVLLM_BATCH=1 \
RVLLM_ITERS=2 \
RVLLM_WARMUP=1 \
RVLLM_ARENA_GB=28 \
cargo run --manifest-path /root/rvllm/v3/Cargo.toml \
  -p rvllm-bench --bin rvllm-bench --features cuda
```

## Claim Impact

NERVA currently beats the recorded vLLM short-decode baseline for the same
Qwen3-8B shape. The rvLLM side is recorded as `unsupported_workload` for this
specific Qwen3 comparison, so NERVA does not claim a measured speedup over
rvLLM on this workload.
