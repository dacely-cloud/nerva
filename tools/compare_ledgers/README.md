# compare_ledgers

Compares two single-line JSON benchmark or ledger summaries and emits a JSON
comparison report. It is intentionally small and dependency-free so it can run in
the bootstrap environment.

```text
cargo run -p nerva-compare-ledgers -- [--tolerance-ns N] baseline.json candidate.json
```

The command exits with status `0` when all fields that are present in either file
match, and status `1` when a field differs or is present in only one file.
Fields missing from both inputs are reported as `missing_both` but do not fail
the comparison. This lets the tool compare today's compact summaries while still
making absent future ledger fields visible.

The first supported field set covers:

- total token or probe latency in nanoseconds;
- host synchronization event count;
- hot-path allocation count;
- H2D and D2H byte counters where present;
- KV residency decision count;
- kernel event count;
- graph replay event count;
- GPU idle nanoseconds derived from the device timeline.

Latency and GPU-idle fields use `--tolerance-ns`; counts and byte totals require
exact equality.
