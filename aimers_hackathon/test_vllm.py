#!/usr/bin/env python
import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM load test for quantized model")
    parser.add_argument(
        "--model_dir",
        default="/home/yjjang/aimers/aimers_hackathon/model",
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--prompt",
        default="안녕하세요. EXAONE 모델 테스트입니다. 한 줄로 자기소개해 주세요.",
        help="Prompt text",
    )
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm = LLM(model=args.model_dir, trust_remote_code=True)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    outputs = llm.generate([args.prompt], params)
    for out in outputs:
        print(out.outputs[0].text)


if __name__ == "__main__":
    main()
