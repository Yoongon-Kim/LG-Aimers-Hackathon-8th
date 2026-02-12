#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick quality/speed gate for candidate models")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def repeated_char_flag(text: str) -> bool:
    return re.search(r"(.)\\1{7,}", text) is not None


def run_one(model, tok, prompt: str, max_new_tokens: int) -> tuple[int, str, float]:
    text = tok.apply_chat_template([
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    inp = tok([text], return_tensors="pt")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.time() - t0

    new_toks = out.shape[1] - inp["input_ids"].shape[1]
    decoded = tok.decode(out[0], skip_special_tokens=True)
    return int(new_toks), decoded, dt


def main() -> None:
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)

    prompts = [
        ("대한민국의 수도는?", "서울"),
        ("What is the capital of France?", "Paris"),
        ("1+1은?", "2"),
        ("What is 7+5?", "12"),
    ]

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )
    model.eval()

    # warmup
    _ = run_one(model, tok, "안녕", 8)

    total_new = 0
    total_sec = 0.0
    bad_repeat = 0
    bad_empty = 0
    exact_hits = 0

    print("=" * 80)
    print(f"MODEL: {args.model_dir}")
    print("=" * 80)

    for q, kw in prompts:
        new_toks, dec, sec = run_one(model, tok, q, args.max_new_tokens)
        answer_tail = dec[-120:].replace("\n", " ")

        total_new += new_toks
        total_sec += sec

        if new_toks <= 1:
            bad_empty += 1
        if repeated_char_flag(answer_tail):
            bad_repeat += 1
        if kw.lower() in answer_tail.lower():
            exact_hits += 1

        print(f"Q: {q}")
        print(f"   new_tokens={new_toks} sec={sec:.2f}")
        print(f"   tail={answer_tail}")

    tps = (total_new / total_sec) if total_sec > 0 else 0.0
    print("-" * 80)
    print(f"total_new_tokens={total_new}")
    print(f"total_sec={total_sec:.2f}")
    print(f"tokens_per_sec={tps:.3f}")
    print(f"exact_keyword_hits={exact_hits}/{len(prompts)}")
    print(f"bad_empty={bad_empty}, bad_repeat={bad_repeat}")

    passed = (exact_hits >= 2) and (bad_empty <= 1) and (bad_repeat == 0)
    print(f"GATE_PASS={passed}")


if __name__ == "__main__":
    main()
