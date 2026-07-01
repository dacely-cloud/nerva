#!/usr/bin/env python3
"""Run a same-prompt vLLM DeepSeek generation and emit JSON.

This is intentionally small and dependency-light beyond vLLM itself. It is used
by `nerva-bench deepseek-vllm-benchmark-plan` as the vLLM half of the
same-checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
    parser.add_argument("--vllm-root", default="/root/vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--enable-prefix-caching", action="store_true")
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


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(1, int(quantile * len(sorted_values) + 0.999999))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def error_payload(args: argparse.Namespace | None, exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error",
        "schema": "nerva-vllm-generate-v1",
        "engine": "vllm",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -8192:
        ],
    }
    if args is not None:
        payload.update(
            {
                "vllm_root": str(Path(args.vllm_root).resolve()),
                "model": args.model,
                "prompt_mode": "file" if args.prompt.startswith("@") else "literal",
                "max_model_len": args.max_model_len,
                "max_tokens": args.max_tokens,
                "runs": args.runs,
                "warmup_runs": args.warmup_runs,
                "dtype": args.dtype,
                "enable_prefix_caching": args.enable_prefix_caching,
                "sampler": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                },
            }
        )
    return payload


def run_generation(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be zero or positive")
    prompt, prompt_mode = resolve_prompt(args.prompt)

    vllm_root = Path(args.vllm_root).resolve()
    if not vllm_root.is_dir():
        raise FileNotFoundError(f"vLLM root does not exist: {vllm_root}")
    sys.path.insert(0, str(vllm_root))

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
        enable_prefix_caching=args.enable_prefix_caching,
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
    outputs = None
    warmup_elapsed_ns: list[int] = []
    for _ in range(args.warmup_runs):
        started = time.perf_counter_ns()
        outputs = llm.generate([prompt], sampling_params=sampling, use_tqdm=False)
        warmup_elapsed_ns.append(time.perf_counter_ns() - started)

    latency_samples_ns: list[int] = []
    total_elapsed_ns = 0
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        outputs = llm.generate([prompt], sampling_params=sampling, use_tqdm=False)
        elapsed_ns = time.perf_counter_ns() - started
        latency_samples_ns.append(elapsed_ns)
        total_elapsed_ns += elapsed_ns

    if outputs is None:
        raise RuntimeError("vLLM generation did not run")
    candidate = outputs[0].outputs[0]
    generated_token_ids = token_ids_from_output(candidate)
    generated_tokens = len(generated_token_ids)
    best_elapsed_ns = min(latency_samples_ns)
    request_p50_ms = percentile([sample / 1_000_000.0 for sample in latency_samples_ns], 0.50)
    request_p95_ms = percentile([sample / 1_000_000.0 for sample in latency_samples_ns], 0.95)
    request_p99_ms = percentile([sample / 1_000_000.0 for sample in latency_samples_ns], 0.99)
    token_p99_ms = request_p99_ms / generated_tokens if generated_tokens > 0 else 0.0
    tokens_per_second = (
        generated_tokens * 1_000_000_000.0 / best_elapsed_ns if best_elapsed_ns > 0 else 0.0
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "schema": "nerva-vllm-generate-v1",
                "engine": "vllm",
                "vllm_root": str(vllm_root),
                "model": args.model,
                "prompt_mode": prompt_mode,
                "prompt": prompt,
                "prompt_tokens": len(prompt_token_ids),
                "prompt_token_ids": [int(token) for token in prompt_token_ids],
                "max_model_len": args.max_model_len,
                "max_tokens": args.max_tokens,
                "runs": args.runs,
                "measured_runs": args.runs,
                "warmup_runs": args.warmup_runs,
                "enable_prefix_caching": args.enable_prefix_caching,
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
                "elapsed_wall_ns": best_elapsed_ns,
                "total_elapsed_wall_ns": total_elapsed_ns,
                "warmup_elapsed_wall_ns": warmup_elapsed_ns,
                "tokens_per_second": tokens_per_second,
                "request_p50_ms": request_p50_ms,
                "request_p95_ms": request_p95_ms,
                "request_p99_ms": request_p99_ms,
                "p99_ms": token_p99_ms,
                "latency_samples_ms": [
                    sample / 1_000_000.0 for sample in latency_samples_ns
                ],
            },
            separators=(",", ":"),
        )
    )


def main() -> None:
    args: argparse.Namespace | None = None
    try:
        args = parse_args()
        run_generation(args)
    except Exception as exc:
        print(json.dumps(error_payload(args, exc), separators=(",", ":")))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
