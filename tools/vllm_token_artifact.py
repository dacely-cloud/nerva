#!/usr/bin/env python3
import argparse
import json
import time

from vllm import LLM, SamplingParams


def parse_prompt_ids(value: str) -> list[int]:
    if value.startswith("ids:"):
        value = value[4:]
    if not value:
        raise ValueError("prompt ids must not be empty")
    return [int(part) for part in value.split(",") if part]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit a compact vLLM greedy token artifact."
    )
    parser.add_argument("model")
    parser.add_argument("prompt_ids")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    prompt_token_ids = parse_prompt_ids(args.prompt_ids)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        detokenize=False,
    )
    started = time.perf_counter_ns()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    outputs = llm.generate([prompt_token_ids], params, use_tqdm=False)
    elapsed_ns = time.perf_counter_ns() - started
    token_ids = list(outputs[0].outputs[0].token_ids)

    print(
        json.dumps(
            {
                "schema": "nerva-vllm-token-artifact-v1",
                "engine": "vllm",
                "model": args.model,
                "prompt_token_ids": prompt_token_ids,
                "max_tokens": args.max_tokens,
                "token_ids": token_ids,
                "elapsed_ns": elapsed_ns,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
