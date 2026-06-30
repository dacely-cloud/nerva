# RT Core LLM Viability

This note records what the current RT experiments prove, what they do not prove,
and what has to change before `--rt-mode sparse` can be called semantic RT
retrieval for real Qwen decode.

## Current Correction

Current Qwen `--rt-mode sparse` is not semantic RT retrieval.

The default Qwen sparse path calls the OptiX selector with page/count metadata:
active pages, current page, local page count, sink page count, query count, and
candidate count. It produces sink pages, local pages, and synthetic far pages.
Those selected page ids are then passed back into the existing CUDA selected-page
attention path. Decode, KV cache, CUDA graphs, sampler, and token generation stay
on the normal Qwen path.

The semantic selectors in the current Qwen path are CUDA Q/K experiments:

| Policy | Uses live Q/K | Uses RT cores | Notes |
|---|---:|---:|---|
| `optix_synthetic_sink_local_far_page_pattern` | No | Yes | Current default sparse RT path. Search pattern is synthetic. |
| `cuda_qk_representative_page_selector` | Yes | No | Scores real KV keys with the live query before selected-page attention. |
| `cuda_qk_fused_attention_page_selector` | Yes | No | Fuses representative Q/K far-page choice into the shared-warp attention kernel. |
| `optix_rt_query_descriptor_candidate_selector` | Synthetic only | Yes | Microbench path; not integrated into Qwen decode. |

That is why JSON reports `rt_core_page_selector`, `semantic_page_selection`, and
`semantic_rt_retrieval` separately. The current default sparse Qwen path should
report `rt_core_page_selector:true`, `semantic_page_selection:false`, and
`semantic_rt_retrieval:false`.

## What Is Proven

The synthetic RT selector is real OptiX traversal on RT-capable hardware. On the
512k-token synthetic point, the query-derived OptiX selector measured about
9.135 us, and selector plus CUDA rerank measured about 13.279 us.

The real Qwen selected-page attention integration is wired into the existing
decode pipeline. It does not replace decode, attention projection, MLP, sampling,
or KV layout with a separate implementation.

On the 30,571-token Qwen3-8B prompt with 2,048 generated tokens and
`NERVA_EXPERIMENTAL_PREFILL_LOCAL_WINDOW_TOKENS=4096`, the measured decode
comparison was:

| Selector policy | Selected pages | Decode throughput | Decode wall | Attention per 256-token chunk |
|---|---:|---:|---:|---:|
| Dense no-RT | 512 | 78.24 tok/s | 26.25s | 890.92 ms |
| OptiX synthetic, far=1 | 67 | 87.57 tok/s | 23.41s | 516.51 ms |
| OptiX synthetic, far=14 | 80 | 86.63 tok/s | 23.66s | 536.07 ms |
| CUDA Q/K representative, far=14 | 80 | 80.74 tok/s | 25.39s | 777.78 ms |
| CUDA Q/K fused, far=14 | 80 | 81.65 tok/s | 25.11s | 723.46 ms |

This proves that selected-page attention can reduce long-context attention time
and can improve full decode throughput on this specific 32k run. It does not
prove that the sparse output is equivalent to dense output.

## What Is Not Proven

Sparse decode is not exact dense decode. On the measured 32k/2048 prompt, dense
versus synthetic RT and dense versus fused Q/K first diverged at generated token
index 7.

Current Qwen sparse RT does not reduce VRAM. The decode path still allocates the
full resident KV cache, so dense and sparse both used about 31.8 GiB on the RTX
5090 for the 32k run.

Current Qwen sparse RT is not semantic RT retrieval. Real semantic RT retrieval
requires a page-descriptor acceleration structure built from real Qwen KV/page
descriptors, a way to update it as decode writes new KV, and a query path that
traces against those descriptors. The current OptiX decode selector uses a fixed
page grid and page/count metadata.

RT cores do not run Transformer math. They can help candidate selection if that
selection can be expressed as traversal/search. CUDA still handles projections,
Q/K/V publication, selected-page attention, MLP, normalization, logits, and
sampling.

## KV Math

For local Qwen3-8B:

| Field | Value |
|---|---:|
| Layers | 36 |
| KV heads | 8 |
| Head dim | 128 |
| KV dtype | BF16 |
| K+V bytes per token across all layers | 147,456 bytes |
| K+V bytes per token across all layers | 144 KiB |
| Page size | 64 tokens |
| K+V bytes per page across all layers | 9,437,184 bytes |
| K+V bytes per page across all layers | 9.0 MiB |
| 32,768-token KV | 4.50 GiB |
| 1,000,000-token KV | 137.33 GiB |

The 67-page sparse setting has an active KV working set of about 603 MiB. The
80-page setting has an active KV working set of about 720 MiB. That is the
potential hot-KV target, but the current runtime still keeps all KV resident.

If cold pages are fetched from host memory, the transfer budget matters. One
64-token page is about 9 MiB across all layers. Fetching one cold far page per
token at 80 tok/s is roughly 720 MiB/s. Fetching fourteen cold far pages per
token at 80 tok/s is over 10 GiB/s before overhead. The current measurements do
not include that hot/cold transfer cost because cold KV staging is not
implemented in this path.

## Context Limits

RT selection does not make a model support arbitrary exact context length.
Qwen3-8B's local config reports `max_position_embeddings: 40960`. A 1M-token
exact context would require both model support for those positions and about
137 GiB of BF16 K/V for this model before weights and workspaces.

NERVA can still experiment with longer external memory, retrieval, summarization,
or approximate sparse memory, but that is not the same as exact 1M-token
Transformer context for a model whose configured window is about 40k.

## Required Work For Real Semantic RT

A real semantic RT path needs these pieces:

1. Build page descriptors from real Qwen KV, likely using representative keys or
   low-dimensional projections per page and KV head.
2. Build or update an OptiX acceleration structure over those descriptors instead
   of the current fixed page grid.
3. Feed live decode query descriptors into the OptiX raygen program.
4. Return candidate page ids into the existing selected-page attention path.
5. Keep `--rt-mode shadow` able to run dense decode as authoritative while
   measuring selected-page recall and divergence.
6. Add hot/cold KV residency only after semantic selection has useful recall,
   because moving KV without a good selector just moves the bottleneck.

Until those pieces exist, the honest label is:

`OptiX synthetic page selector + CUDA selected-page attention`, not semantic RT
retrieval.
