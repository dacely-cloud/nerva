# Future Transport

DPDK and RDMA are future phases.

The setup principle is strict:

- real DPDK discovered through `pkg-config libdpdk`,
- no silent compatibility fallback,
- same CFLAGS for C compilation and bindgen,
- libclang pinned through Cargo environment,
- transport isolated from inference runtime ownership.

NERVA should not add `nerva-transport-dpdk` or `nerva-transport-rdma` until the
architecture gate opens transport work.
