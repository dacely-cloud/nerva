# Token Ledger

Every token emits a ledger with:

- kernel launches,
- graph capture/replay,
- copies,
- syncs,
- stalls,
- allocations,
- evictions,
- prefetches,
- residency changes.

The ledger is part of the runtime contract, not a profiler side channel.
