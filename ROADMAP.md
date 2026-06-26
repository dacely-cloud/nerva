# NERVA Roadmap

## M0: Linux Workspace and Smoke

- Linux-only Rust workspace.
- CUDA driver smoke path.
- vLLM/rvLLM audit.
- Root docs and design docs.
- Native CUDA scaffold.

## M1: Residency Core

- `ResidentBlock` allocator and registry.
- VRAM, pinned DRAM, and DRAM tier accounting.
- Token ledger schema.
- Hot-path allocation accounting.

## M2: CUDA Decode Skeleton

- Static metadata layout.
- CUDA graph executor.
- Device token ring.
- Smoke synthetic decode transaction.

## M3: KV Virtual Memory

- KV block table.
- Tiered KV residency.
- Prefetch/evict decisions.
- Ledgered stalls and copies.

## M4: Model Bring-Up

- FP16/BF16 exact block path.
- HF metadata parser.
- vLLM-style token identity parity harness.

## Future

- DPDK/RDMA transport.
- AMD/HIP backend.
- Non-Linux host support.
