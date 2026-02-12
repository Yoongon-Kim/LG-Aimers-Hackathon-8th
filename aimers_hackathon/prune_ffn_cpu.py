#!/usr/bin/env python3
"""
CPU 16-core Optimized FFN Structured Pruning
EXAONE-4.0-1.2B Phase2

- float32 (CPU stable)
- device_map="cpu" fixed
- no multiprocessing
- in-place weight replacement minimized
- config mismatch prevention

Usage:
    python prune_ffn_cpu.py --prune-ratio 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================
# CPU Environment
# ===============================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)
torch.set_num_interop_threads(16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-optimized FFN structured pruning for EXAONE models."
    )
    parser.add_argument(
        "--base-model",
        default="./base_model",
        help="Input model directory (will not be modified).",
    )
    parser.add_argument(
        "--out-model",
        default="./modelffn",
        help="Output directory for the pruned model.",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.25,
        help="Pruning ratio for FFN (recommended 0.2~0.3 for CPU).",
    )
    return parser.parse_args()


@torch.no_grad()
def prune_ffn(layer, ratio: float):
    """
    FFN structured pruning based on up_proj L2 norm.
    Creates new Linear layers to ensure clean shape consistency.
    """
    up = layer.up_proj
    gate = layer.gate_proj
    down = layer.down_proj

    # Calculate importance score based on up_proj
    score = torch.norm(up.weight, dim=1)  # [hidden_dim]

    hidden_dim = score.shape[0]
    keep_dim = int(hidden_dim * (1 - ratio))

    # Select top-k neurons
    keep_idx = torch.topk(score, keep_dim, largest=True).indices
    keep_idx, _ = torch.sort(keep_idx)

    # Create new Linear layers with correct dimensions
    new_up = torch.nn.Linear(
        up.in_features, keep_dim, bias=False, device='cpu', dtype=torch.float32
    )
    new_gate = torch.nn.Linear(
        gate.in_features, keep_dim, bias=False, device='cpu', dtype=torch.float32
    )
    new_down = torch.nn.Linear(
        keep_dim, down.out_features, bias=False, device='cpu', dtype=torch.float32
    )

    # Copy selected weights
    new_up.weight.copy_(up.weight[keep_idx, :])
    new_gate.weight.copy_(gate.weight[keep_idx, :])
    new_down.weight.copy_(down.weight[:, keep_idx])

    # Replace layers
    layer.up_proj = new_up
    layer.gate_proj = new_gate
    layer.down_proj = new_down

    return keep_dim


def main() -> None:
    args = parse_args()
    base_model_dir = args.base_model
    out_model_dir = args.out_model
    prune_ratio = args.prune_ratio

    print("=" * 80)
    print("CPU 16-Core FFN Structured Pruning")
    print("=" * 80)

    if not os.path.isdir(base_model_dir):
        print(
            f"[ERROR] base_model directory not found: {base_model_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.exists(out_model_dir):
        print(f"[WARN] {out_model_dir} already exists.")
        response = input("Delete and continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
        import shutil
        shutil.rmtree(out_model_dir)

    if not (0.0 < prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be between 0 and 1.")

    print(f"\n[INFO] Pruning ratio: {prune_ratio:.2%}")
    print(f"[INFO] Base model: {base_model_dir}")
    print(f"[INFO] Output: {out_model_dir}")

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Load model (CPU, float32)
    print("[2/4] Loading model (CPU, float32)...")
    print("      This may take 5-8 minutes on CPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"      Model loaded in {load_time:.1f}s")

    # FFN pruning
    print("\n[3/4] FFN structured pruning...")
    t0 = time.time()
    original_size = model.config.intermediate_size if hasattr(model.config, 'intermediate_size') else None
    
    pruned_sizes = []
    for i, layer in enumerate(model.model.layers):
        keep_dim = prune_ffn(layer.mlp, prune_ratio)
        pruned_sizes.append(keep_dim)
        if i % 5 == 0 or i == len(model.model.layers) - 1:
            print(f"      Layer {i:02d}/{len(model.model.layers)-1} done (keep_dim={keep_dim})")

    prune_time = time.time() - t0
    print(f"      Pruning completed in {prune_time:.1f}s")

    # Check consistency
    if len(set(pruned_sizes)) == 1:
        new_intermediate_size = pruned_sizes[0]
        print(f"\n[INFO] All layers pruned to same size: {new_intermediate_size}")
        if original_size:
            print(f"      Original: {original_size} → New: {new_intermediate_size}")
            print(f"      Reduction: {(1 - new_intermediate_size/original_size)*100:.1f}%")
        
        # Update config
        if hasattr(model.config, 'intermediate_size'):
            model.config.intermediate_size = new_intermediate_size
    else:
        print("[WARN] Inconsistent intermediate sizes detected!")
        print(f"      Sizes: {set(pruned_sizes)}")

    # Save
    print("\n[4/4] Saving pruned model...")
    print("      This may take 5-10 minutes...")
    t0 = time.time()
    os.makedirs(out_model_dir, exist_ok=True)
    model.save_pretrained(out_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_model_dir)
    save_time = time.time() - t0
    print(f"      Saved in {save_time:.1f}s")

    # Summary
    print("\n" + "=" * 80)
    print("✅ CPU FFN Structured Pruning Completed")
    print("=" * 80)
    print(f"Output: {out_model_dir}")
    print(f"Total time: {load_time + prune_time + save_time:.1f}s")
    print("\nNext steps:")
    print("  1. Copy LICENSE, README.md, assets/ from base_model")
    print("  2. Create submit.zip")
    print("  3. Run validation tests")


if __name__ == "__main__":
    main()
