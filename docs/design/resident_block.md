# ResidentBlock

`ResidentBlock` is the semantic unit of residency. It is not just a pointer or a
page. It identifies what the bytes mean: weight, KV cache, activation, token
state, sampler state, or ledger state.

Each block carries:

- stable id,
- kind,
- byte size,
- current tier,
- future policy metadata.

The runtime schedules blocks. Kernels consume stable device addresses derived
from block residency decisions.
