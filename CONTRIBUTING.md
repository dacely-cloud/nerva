# Contributing

NERVA is Linux-only in M0. Keep changes focused on Ubuntu x86_64/aarch64 unless a
roadmap item explicitly opens another platform.

Before implementation changes:

- Read `docs/source/ARCHITECTURE_START.md`.
- Read `docs/audit/AUDIT_VLLM_RVLLM_20260626.md`.
- Do not copy vLLM or rvLLM code into NERVA.
- Keep DPDK/RDMA as future transport work until the architecture gate opens it.

Local checks:

```bash
cargo fmt --all --check
cargo test --workspace
cargo run -p nerva-bench -- smoke
```

Commit policy:

- No generated co-author trailers.
- Keep audit/doc moves separate from unrelated refactors when possible.
- Do not commit local model weights, benchmark caches, or GPU traces.
