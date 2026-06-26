# Backend Portability

M0 supports Linux hosts only. CUDA is the first backend.

Host targets:

- Ubuntu x86_64,
- Ubuntu aarch64.

GPU targets will be explicit and fail closed. Do not silently run a kernel built
for the wrong compute capability. AMD/HIP is a future backend, not an M0 crate.
