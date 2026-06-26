# vLLM Baseline

vLLM is the behavioral and compatibility oracle:

- OpenAI API shape,
- tokenizer/HF integration,
- scheduling behavior,
- model coverage,
- PagedAttention and KV block concepts,
- benchmark comparison.

NERVA must not inherit the Python-owned token loop or Torch allocator as the
runtime memory model.
