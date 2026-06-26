# Copied DPDK Setup

NERVA does not build a DPDK or RDMA transport in M0. This directory carries the
actual DPDK shim sources copied from `toil-backend` for the future
transport phase.

Copied source:

- `dpdk-shim/`: copied from `toil-backend/dpdk-shim`, excluding `target`
  and its nested lockfile.
- `examples/dpdk_rx_demo.rs`: copied from `toil-backend/examples`.

The toil-backend pattern preserved here:

- Use an in-tree C shim instead of crates that silently stub DPDK.
- Locate DPDK with `pkg-config libdpdk`.
- Feed the same `pkg-config --cflags libdpdk` to both `cc` and bindgen.
- Pin bindgen to LLVM/libclang through `.cargo/config.toml`.
- Fail the build when real DPDK is unavailable; do not fall back to a fake shim.

NERVA will add transport crates only after the architecture gate opens DPDK/RDMA.
