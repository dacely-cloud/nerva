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

At 1,048,576 synthetic tokens, `NERVA_EXPERIMENTAL_RT_SEMANTIC_OPTIX=1
target/debug/nerva-bench experimental-rt-sweep 1048576 8 1024 16 64 1 36`
completed on the RTX 5090 with 16,384 pages and zero candidate parity
mismatches. The summarized artifact is
`docs/source/perf/rt_1m_synthetic_sweep_summary.json`.

| Candidate pages/query | Selector | Selector + rerank | Estimated KV fraction | Attention-mass recall | RT attention stage | Dense full attention | Stage speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 11.722 us | 15.946 us | 1.5625% | 38.2955% | 0.401 ms | 12.276 ms | 30.59x |
| 256 | 10.684 us | 14.908 us | 2.3437% | 68.2687% | 0.674 ms | 12.273 ms | 18.22x |
| 512 | 10.866 us | 15.090 us | 3.9062% | 95.4528% | 1.215 ms | 12.231 ms | 10.07x |
| 1024 | 12.914 us | 19.150 us | 7.0312% | 99.9937% | 2.310 ms | 12.227 ms | 5.29x |

This is the strongest current evidence for the hot/cold direction: at 1M
synthetic tokens, the RT selector overhead stays in the low tens of
microseconds while the selected-page attention stage can run against roughly
4-7% of dense KV bytes. It is still synthetic descriptor work, so it proves the
shape of the systems opportunity, not real Qwen semantic quality.

A fixed 1024-candidate scale check was also run at 1M, 2M, 4M, and 8M synthetic
tokens. The summarized artifact is
`docs/source/perf/rt_context_scale_c1024_summary.json`.

| Context tokens | Pages | Selector + rerank | Estimated KV fraction | RT attention stage | Dense full attention | Stage speedup | Qwen3-8B dense KV | Qwen3-8B hot KV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 16,384 | 19.150 us | 7.0312% | 2.310 ms | 12.227 ms | 5.29x | 144 GiB | 10.125 GiB |
| 2,097,152 | 32,768 | 17.032 us | 3.5156% | 2.313 ms | 27.182 ms | 11.75x | 288 GiB | 10.125 GiB |
| 4,194,304 | 65,536 | 16.548 us | 1.7578% | 2.310 ms | 54.368 ms | 23.54x | 576 GiB | 10.125 GiB |
| 8,388,608 | 131,072 | 16.840 us | 0.8789% | 2.313 ms | 109.019 ms | 47.14x | 1152 GiB | 10.125 GiB |

This is the best current evidence that RT-style candidate selection can support
a fixed-size hot KV working set while total context grows. The Qwen3-8B byte
columns are estimates using 36 layers, 8 KV heads, 128 head dim, BF16 K/V, and
64-token pages. They are not a claim that Qwen3-8B can exact-attend to 8M
positions; they show the memory geometry a hot/cold sparse-memory system would
need if a semantic selector could pick useful far pages.

The real Qwen selected-page attention integration is wired into the existing
decode pipeline. It does not replace decode, attention projection, MLP, sampling,
or KV layout with a separate implementation.

The current integration boundary matters. The OptiX sparse selector launches
once per decode token before CUDA graph replay, so it can only use page/count
metadata available before the layer stack runs. The CUDA Q/K selectors launch
per layer after QKV prepare, because the live query for layer N does not exist
until the previous layer has produced its activation. That is the main reason
the current Qwen RT path is synthetic and the semantic Q/K path is CUDA-only.

The 32k selector overhead check in
`docs/source/perf/rt_semantic_integration_boundary_summary.json` shows that RT
traversal overhead itself is not the blocker:

| Shape | Selector + rerank | Interpretation |
|---|---:|---|
| 8 KV-head queries, 67 pages | 13.260 us | Current Qwen head count and fastest sparse page count. |
| 8 KV-head queries, 80 pages | 13.155 us | Current reproduced sparse RT page count. |
| 288 synthetic queries, 67 pages | 13.717 us | Lower bound if all 36 layers x 8 KV heads were known for one launch. |
| 288 synthetic queries, 80 pages | 13.670 us | Lower bound if all layer queries were known for one launch. |

If a real semantic RT selector had to launch once per layer, the measured 8-query
selector plus rerank cost estimates to about 0.47-0.49 ms/token for 36 layers
before descriptor-update cost. That cost is not fatal, but it has to buy more
than that in attention savings and it has to fit the graph/layer schedule.

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

The standalone cold-KV staging probe measures pinned host-to-device transfer for
one Qwen3-8B 64-token KV page, which is 9 MiB across all layers. The summarized
artifact is `docs/source/perf/rt_cold_kv_staging_summary.json`.

| Cold pages staged per step | Bytes per step | Transfer avg | Transfer avg/page | Effective bandwidth |
|---:|---:|---:|---:|---:|
| 1 | 9 MiB | 0.689 ms | 0.689 ms | 13.701 GB/s |
| 4 | 36 MiB | 2.743 ms | 0.686 ms | 13.762 GB/s |
| 8 | 72 MiB | 5.452 ms | 0.681 ms | 13.848 GB/s |
| 16 | 144 MiB | 20.559 ms | 1.285 ms | 7.344 GB/s |
| 32 | 288 MiB | 21.785 ms | 0.681 ms | 13.862 GB/s |
| 64 | 576 MiB | 65.663 ms | 1.026 ms | 9.198 GB/s |

This benchmark measures transfer only; it does not include attention, projection,
MLP, sampling, or graph replay. The result is still enough to set the design
constraint. Small cold deltas are viable. Fetching many pages from host every
token is not viable. A real RT/hot-cold path has to keep reusable selected pages
resident, prefetch cold deltas, and make cold misses rare enough that PCIe does
not become the decode bottleneck.

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
