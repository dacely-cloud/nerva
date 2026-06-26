# Token State Machine

NERVA's target decode state machine is device-resident:

```text
input token
-> decode transaction
-> sampler state
-> stop mask
-> output token ring
-> ledger event stream
```

The CPU may inspect and schedule. It should not be required to decide every next
token in the steady-state loop.
