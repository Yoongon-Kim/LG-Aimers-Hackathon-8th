#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe FFN structured pruning for EXAONE")
    p.add_argument("--base-model", default="./base_model")
    p.add_argument("--out-model", required=True)
    p.add_argument("--prune-ratio", type=float, default=0.08)
    p.add_argument("--target-intermediate-size", type=int, default=None)
    p.add_argument("--round-to", type=int, default=256)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


@torch.no_grad()
def prune_layer_mlp(layer, keep_dim: int) -> None:
    up = layer.up_proj
    gate = layer.gate_proj
    down = layer.down_proj

    # Safer importance: combine three FFN projections.
    score = (
        torch.norm(up.weight, dim=1)
        + torch.norm(gate.weight, dim=1)
        + torch.norm(down.weight, dim=0)
    )

    keep_idx = torch.topk(score, keep_dim, largest=True).indices
    keep_idx, _ = torch.sort(keep_idx)

    dtype = up.weight.dtype
    device = up.weight.device

    new_up = torch.nn.Linear(up.in_features, keep_dim, bias=False, device=device, dtype=dtype)
    new_gate = torch.nn.Linear(gate.in_features, keep_dim, bias=False, device=device, dtype=dtype)
    new_down = torch.nn.Linear(keep_dim, down.out_features, bias=False, device=device, dtype=dtype)

    new_up.weight.copy_(up.weight[keep_idx, :])
    new_gate.weight.copy_(gate.weight[keep_idx, :])
    new_down.weight.copy_(down.weight[:, keep_idx])

    layer.up_proj = new_up
    layer.gate_proj = new_gate
    layer.down_proj = new_down


def get_keep_dim(orig_dim: int, prune_ratio: float, round_to: int, explicit_target: int | None) -> int:
    if explicit_target is not None:
        keep_dim = explicit_target
    else:
        raw = int(orig_dim * (1.0 - prune_ratio))
        keep_dim = max(round_to, (raw // round_to) * round_to)

    if keep_dim <= 0 or keep_dim >= orig_dim:
        raise ValueError(f"Invalid keep_dim={keep_dim}, orig_dim={orig_dim}")
    if keep_dim % round_to != 0:
        raise ValueError(f"keep_dim must be multiple of {round_to}: {keep_dim}")
    return keep_dim


def main() -> None:
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)

    if os.path.exists(args.out_model):
        if not args.force:
            raise RuntimeError(f"{args.out_model} exists. Use --force to overwrite.")
        shutil.rmtree(args.out_model)

    print("[1/4] loading tokenizer/model...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"      load_sec={time.time() - t0:.1f}")

    orig_dim = model.model.layers[0].mlp.up_proj.out_features
    keep_dim = get_keep_dim(orig_dim, args.prune_ratio, args.round_to, args.target_intermediate_size)
    print(f"[INFO] FFN dim: {orig_dim} -> {keep_dim} (prune {(1 - keep_dim / orig_dim) * 100:.2f}%)")

    print("[2/4] pruning all transformer layers...")
    t0 = time.time()
    for i, layer in enumerate(model.model.layers):
        prune_layer_mlp(layer.mlp, keep_dim)
        if i % 5 == 0 or i == len(model.model.layers) - 1:
            print(f"      layer {i:02d}/{len(model.model.layers) - 1}")
    print(f"      prune_sec={time.time() - t0:.1f}")

    if hasattr(model.config, "intermediate_size"):
        model.config.intermediate_size = keep_dim

    print("[3/4] saving...")
    t0 = time.time()
    os.makedirs(args.out_model, exist_ok=True)
    model.save_pretrained(args.out_model, safe_serialization=True)
    tok.save_pretrained(args.out_model)
    print(f"      save_sec={time.time() - t0:.1f}")

    print("[4/4] done")
    print(f"output={args.out_model}")


if __name__ == "__main__":
    main()
