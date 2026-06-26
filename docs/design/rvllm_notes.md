# rvLLM Notes

rvLLM is the Rust/CUDA architecture reference:

- CUDA context and stream ownership,
- static arenas,
- graph handles,
- manifest-verified kernels,
- deterministic sampling references.

NERVA should not inherit Gemma-only assumptions, FP8/H100 as the default path, or
per-request graph recapture as a production steady state.
