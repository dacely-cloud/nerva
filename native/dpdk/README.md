# DPDK Setup

NERVA does not build a DPDK or RDMA transport in M0. This directory carries the
in-tree DPDK shim sources for the future transport phase.

Source layout:

- `dpdk-shim/`: C/Rust shim sources, excluding generated build artifacts.
- `examples/dpdk_rx_demo.rs`: standalone RX demo kept outside the main
  workspace.

Build policy:

- Use an in-tree C shim and fail clearly when real DPDK is unavailable.
- Locate DPDK with `pkg-config libdpdk`.
- Feed the same `pkg-config --cflags libdpdk` to both `cc` and bindgen.
- Pin bindgen to LLVM/libclang through `.cargo/config.toml`.
- Do not provide a simulated DPDK data path.

NERVA will add transport crates only after the architecture gate opens DPDK/RDMA.
