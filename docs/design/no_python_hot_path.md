# No Python Hot Path

vLLM remains the compatibility oracle, but NERVA's steady-state decode loop must
not depend on Python.

Python bindings may exist later for model loading, service integration, or
compatibility tests. They must not own token state or schedule each decode step.
