# Memory Tiers

M0 names four tiers:

- VRAM,
- pinned DRAM,
- DRAM,
- disk.

The CUDA smoke path only proves device and pinned-host allocation. Later phases
must add promotion, demotion, prefetch, eviction, and compute-near-data policy.
