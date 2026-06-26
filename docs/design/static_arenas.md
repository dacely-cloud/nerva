# Static Arenas

NERVA follows the rvLLM lesson that graph-captured addresses must be stable.
M0 uses small host-side and CUDA smoke allocations. Later phases should allocate
fixed arenas before graph capture and bind graph metadata to arena regions.

Hot decode must report zero runtime allocations unless a ledger explicitly marks
an allocation event.
