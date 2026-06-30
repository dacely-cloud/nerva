#!/usr/bin/env python3
"""Run a same-prompt vLLM DeepSeek generation and emit JSON.

This is intentionally small and dependency-light beyond vLLM itself. It is used
by `nerva-bench deepseek-vllm-benchmark-plan` as the vLLM half of the
same-checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-log-stats", action="store_true", default=True)
    return parser.parse_args()


def resolve_prompt(prompt_spec: str) -> tuple[str, str]:
    if not prompt_spec.startswith("@"):
        return prompt_spec, "literal"
    path = Path(prompt_spec[1:])
    return path.read_text(), "file"


def token_ids_from_output(output: Any) -> list[int]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        return []
    return [int(token) for token in token_ids]


def main() -> None:
    args = parse_args()
    prompt, prompt_mode = resolve_prompt(args.prompt)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        disable_log_stats=args.disable_log_stats,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode(prompt)
    started = time.perf_counter_ns()
    outputs = llm.generate([prompt], sampling_params=sampling, use_tqdm=False)
    elapsed_ns = time.perf_counter_ns() - started
    candidate = outputs[0].outputs[0]
    generated_token_ids = token_ids_from_output(candidate)
    generated_tokens = len(generated_token_ids)
    tokens_per_second = (
        generated_tokens * 1_000_000_000.0 / elapsed_ns if elapsed_ns > 0 else 0.0
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "schema": "nerva-vllm-generate-v1",
                "engine": "vllm",
                "model": args.model,
                "prompt_mode": prompt_mode,
                "prompt": prompt,
                "prompt_tokens": len(prompt_token_ids),
                "prompt_token_ids": [int(token) for token in prompt_token_ids],
                "max_model_len": args.max_model_len,
                "max_tokens": args.max_tokens,
                "sampler": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                },
                "generated_tokens": generated_tokens,
                "tokens": generated_token_ids,
                "generated_text": candidate.text,
                "finish_reason": candidate.finish_reason,
                "elapsed_wall_ns": elapsed_ns,
                "tokens_per_second": tokens_per_second,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
