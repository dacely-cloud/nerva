# CUDA Graph Executor

The graph executor should combine:

- rvLLM-style explicit graph handles,
- metadata layout hashes,
- stable arena addresses,
- NERVA `ResidentBlock` identity.

Graph replay should fail closed when the layout or resident address contract
changes.
