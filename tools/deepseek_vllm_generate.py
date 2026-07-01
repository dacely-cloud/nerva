#!/usr/bin/env python3
"""Run a same-prompt vLLM DeepSeek generation and emit JSON.

This is intentionally small and dependency-light beyond vLLM itself. It is used
by `nerva-bench deepseek-vllm-benchmark-plan` as the vLLM half of the
same-checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-token-ids-json")
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
    parser.add_argument("--linear-backend", default="auto")
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    trust_remote_code = parser.add_mutually_exclusive_group()
    trust_remote_code.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=True,
    )
    trust_remote_code.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-flashinfer-autotune", action="store_true")
    parser.add_argument(
        "--deep-gemm-warmup",
        choices=["default", "skip", "relax", "full"],
        default="default",
    )
    parser.add_argument("--disable-log-stats", action="store_true", default=True)
    return parser.parse_args()


def resolve_prompt(prompt_spec: str) -> tuple[str, str]:
    if not prompt_spec.startswith("@"):
        return prompt_spec, "literal"
    path = Path(prompt_spec[1:])
    return path.read_text(), "file"


def parse_prompt_token_ids(source: str) -> list[int]:
    try:
        value = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("--prompt-token-ids-json must be a JSON integer array") from exc
    if not isinstance(value, list):
        raise ValueError("--prompt-token-ids-json must be a JSON integer array")
    token_ids: list[int] = []
    for index, token in enumerate(value):
        if not isinstance(token, int) or token < 0:
            raise ValueError(
                f"--prompt-token-ids-json entry {index} must be a non-negative integer"
            )
        token_ids.append(int(token))
    return token_ids


def token_ids_from_output(output: Any) -> list[int]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        return []
    return [int(token) for token in token_ids]


def top_logprobs_from_output(output: Any) -> list[list[dict[str, Any]]]:
    logprobs = getattr(output, "logprobs", None)
    if not logprobs:
        return []
    steps: list[list[dict[str, Any]]] = []
    for step in logprobs:
        if not step:
            steps.append([])
            continue
        entries: list[dict[str, Any]] = []
        for token, value in step.items():
            logprob = getattr(value, "logprob", value)
            rank = getattr(value, "rank", None)
            decoded = getattr(value, "decoded_token", None)
            try:
                logprob_value = float(logprob)
            except (TypeError, ValueError):
                continue
            entries.append(
                {
                    "token": int(token),
                    "logprob": logprob_value if math.isfinite(logprob_value) else None,
                    "rank": int(rank) if rank is not None else None,
                    "decoded": decoded,
                }
            )
        entries.sort(
            key=lambda item: (
                -(item["logprob"] if item["logprob"] is not None else float("-inf")),
                item["token"],
            )
        )
        steps.append(entries)
    return steps


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
        "python": sys.executable,
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
                "prompt_mode": (
                    "token_ids"
                    if args.prompt_token_ids_json is not None
                    else "file"
                    if args.prompt is not None and args.prompt.startswith("@")
                    else "literal"
                ),
                "max_model_len": args.max_model_len,
                "max_tokens": args.max_tokens,
                "runs": args.runs,
                "warmup_runs": args.warmup_runs,
                "dtype": args.dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "trust_remote_code": args.trust_remote_code,
                "enable_prefix_caching": args.enable_prefix_caching,
                "enable_flashinfer_autotune": not args.disable_flashinfer_autotune,
                "enforce_eager": args.enforce_eager,
                "kernel": {
                    "linear_backend": args.linear_backend,
                    "moe_backend": args.moe_backend,
                    "attention_backend": args.attention_backend,
                    "deep_gemm_warmup": args.deep_gemm_warmup,
                    "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM"),
                    "VLLM_MOE_USE_DEEP_GEMM": os.environ.get(
                        "VLLM_MOE_USE_DEEP_GEMM"
                    ),
                    "VLLM_USE_DEEP_GEMM_E8M0": os.environ.get(
                        "VLLM_USE_DEEP_GEMM_E8M0"
                    ),
                    "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": os.environ.get(
                        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"
                    ),
                },
                "sampler": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                },
                "logprobs_requested": args.logprobs,
            }
        )
    return payload


def run_generation(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be zero or positive")
    if args.prompt is None and args.prompt_token_ids_json is None:
        raise ValueError("--prompt or --prompt-token-ids-json is required")
    if args.prompt is not None and args.prompt_token_ids_json is not None:
        raise ValueError("--prompt and --prompt-token-ids-json are mutually exclusive")
    prompt: str | None = None
    prompt_mode = "token_ids"
    prompt_token_ids: list[int]
    if args.prompt_token_ids_json is not None:
        prompt_token_ids = parse_prompt_token_ids(args.prompt_token_ids_json)
    else:
        assert args.prompt is not None
        prompt, prompt_mode = resolve_prompt(args.prompt)

    vllm_root = Path(args.vllm_root).resolve()
    if not vllm_root.is_dir():
        raise FileNotFoundError(f"vLLM root does not exist: {vllm_root}")
    sys.path.insert(0, str(vllm_root))
    if args.deep_gemm_warmup != "default":
        os.environ["VLLM_DEEP_GEMM_WARMUP"] = args.deep_gemm_warmup

    from vllm import LLM, SamplingParams

    engine_kwargs: dict[str, Any] = {}
    if args.linear_backend != "auto":
        engine_kwargs["linear_backend"] = args.linear_backend
    if args.moe_backend != "auto":
        engine_kwargs["moe_backend"] = args.moe_backend
    if args.attention_backend != "auto":
        engine_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.disable_flashinfer_autotune:
        engine_kwargs["enable_flashinfer_autotune"] = False

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
        **engine_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        logprobs=args.logprobs if args.logprobs > 0 else None,
    )

    tokenizer = llm.get_tokenizer()
    if prompt is not None:
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    outputs = None
    warmup_elapsed_ns: list[int] = []
    for _ in range(args.warmup_runs):
        started = time.perf_counter_ns()
        outputs = llm.generate(
            [{"prompt_token_ids": list(prompt_token_ids)}],
            sampling_params=sampling,
            use_tqdm=False,
        )
        warmup_elapsed_ns.append(time.perf_counter_ns() - started)

    latency_samples_ns: list[int] = []
    total_elapsed_ns = 0
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        outputs = llm.generate(
            [{"prompt_token_ids": list(prompt_token_ids)}],
            sampling_params=sampling,
            use_tqdm=False,
        )
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
                "python": sys.executable,
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
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "trust_remote_code": args.trust_remote_code,
                "enable_prefix_caching": args.enable_prefix_caching,
                "enable_flashinfer_autotune": not args.disable_flashinfer_autotune,
                "enforce_eager": args.enforce_eager,
                "kernel": {
                    "linear_backend": args.linear_backend,
                    "moe_backend": args.moe_backend,
                    "attention_backend": args.attention_backend,
                    "deep_gemm_warmup": args.deep_gemm_warmup,
                    "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM"),
                    "VLLM_MOE_USE_DEEP_GEMM": os.environ.get(
                        "VLLM_MOE_USE_DEEP_GEMM"
                    ),
                    "VLLM_USE_DEEP_GEMM_E8M0": os.environ.get(
                        "VLLM_USE_DEEP_GEMM_E8M0"
                    ),
                    "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": os.environ.get(
                        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"
                    ),
                },
                "sampler": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                },
                "generated_tokens": generated_tokens,
                "tokens": generated_token_ids,
                "logprobs_requested": args.logprobs,
                "top_logprobs": (
                    top_logprobs_from_output(candidate) if args.logprobs > 0 else None
                ),
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
