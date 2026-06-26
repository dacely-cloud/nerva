# Single-GPU Benchmarks

M0 benchmark entry point:

```bash
cargo run -p nerva-bench -- smoke
```

The smoke bench:

- loads the CUDA driver dynamically on Linux,
- initializes device 0 when available,
- reports GPU and CUDA versions,
- allocates device and pinned-host memory,
- launches a smoke kernel,
- emits summary JSON,
- reports `hot_path_allocations=0`.

Future single-GPU benches should compare against vLLM ledgers and use stable
JSON output so `tools/compare_ledgers` can diff token-level stalls, copies,
syncs, allocations, and kernel work.
