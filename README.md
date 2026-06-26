# NERVA

NERVA means Neural Execution & Residency Virtual Architecture.

NERVA is an inference operating system for AI models.

The model is not loaded. The model is scheduled.

Initial project direction is documented in:

- `ARCHITECTURE_START.md`
- `MAIN_INFO.md`
- `CONTEXT.md`
- `INTER_SYSTEM_GPU_NIC_COM.md`
- `PHASE_2_LATENCY_LEDGER.md`
- `VLLM_LATENCY_RESULTS.md`
- `CUSTOM_INTERNAL_GPU_NIC_FALLBACK.md`
- `REDESIGN.md`

NERVA is not a vLLM fork.
NERVA is not an rvLLM fork.

vLLM is used as a baseline and compatibility oracle.
rvLLM is used as a Rust/CUDA architecture reference.
NERVA is a new runtime.
