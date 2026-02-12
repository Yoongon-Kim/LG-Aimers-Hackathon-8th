# ============================================================
# EXAONE-4.0-1.2B
# CPU-only Structured FFN Pruning
# base_model/  ->  model/
# vLLM compatible / submit-ready
# ============================================================

from __future__ import annotations

import argparse
import os
import shutil
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================
# 0. CPU 환경 고정
# ===============================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)
torch.set_num_interop_threads(1)

DTYPE = torch.bfloat16

# ===============================
# 3. FFN Structured Pruning
# ===============================
@torch.no_grad()
def structured_prune_mlp(layer, prune_ratio: float):
    """
    EXAONE MLP:
      up_proj   : [d_ff, d_model]
      gate_proj : [d_ff, d_model]
      down_proj : [d_model, d_ff]
    """

    up = layer.up_proj.weight
    gate = layer.gate_proj.weight
    down = layer.down_proj.weight

    # 중요도 (L2 norm)
    score = (
        up.norm(dim=1) +
        gate.norm(dim=1) +
        down.norm(dim=0)
    )

    d_ff = score.numel()
    keep = int(d_ff * (1.0 - prune_ratio))

    idx = torch.topk(score, k=keep, largest=True).indices
    idx, _ = torch.sort(idx)

    # 구조적 슬라이싱
    layer.up_proj.weight = torch.nn.Parameter(up[idx, :])
    layer.gate_proj.weight = torch.nn.Parameter(gate[idx, :])
    layer.down_proj.weight = torch.nn.Parameter(down[:, idx])

    if layer.up_proj.bias is not None:
        layer.up_proj.bias = torch.nn.Parameter(layer.up_proj.bias[idx])
    if layer.gate_proj.bias is not None:
        layer.gate_proj.bias = torch.nn.Parameter(layer.gate_proj.bias[idx])

    layer.intermediate_size = keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only structured FFN pruning for EXAONE models."
    )
    parser.add_argument(
        "--base-model",
        default="./base_model",
        help="Input model directory (will not be modified).",
    )
    parser.add_argument(
        "--out-model",
        default="./model",
        help="Output directory for the pruned model.",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.20,
        help="Pruning ratio for FFN (recommended 0.15~0.18).",
    )
    parser.add_argument(
        "--protect-from-layer",
        type=int,
        default=24,
        help="Do not prune layers >= this index.",
    )
    parser.add_argument(
        "--make-zip",
        action="store_true",
        help="Create submit.zip from the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model_dir = args.base_model
    out_model_dir = args.out_model
    prune_ratio = args.prune_ratio
    protect_from_layer = args.protect_from_layer

    if not os.path.isdir(base_model_dir):
        print(
            f"[ERROR] base_model directory not found: {base_model_dir}",
            file=sys.stderr,
        )
        print("[HINT] Run from the parent folder or pass --base-model.")
        sys.exit(1)

    if os.path.exists(out_model_dir):
        raise RuntimeError(
            f"{out_model_dir} already exists. "
            "Delete it first to avoid accidental overwrite."
        )

    if not (0.0 < prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be between 0 and 1.")

    print("[INFO] Loading tokenizer from base_model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("[INFO] Loading model from base_model (BF16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,
    )
    model.eval()

    print("[INFO] Start FFN structured pruning...")
    for i, layer in enumerate(model.model.layers):
        if i >= protect_from_layer:
            continue
        structured_prune_mlp(layer.mlp, prune_ratio)
        if i % 4 == 0:
            print(f"[PRUNE] layer {i} done")

    print("[INFO] Pruning completed.")

    # ===============================
    # 4. Update config with new intermediate_size
    # ===============================
    if hasattr(model.config, 'intermediate_size'):
        # Get actual intermediate_size from first pruned layer
        actual_size = model.model.layers[0].mlp.intermediate_size
        model.config.intermediate_size = actual_size
        print(f"[INFO] Updated config intermediate_size: {actual_size}")

    # ===============================
    # 5. Save to model/
    # ===============================
    os.makedirs(out_model_dir, exist_ok=True)
    model.save_pretrained(out_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_model_dir)

    print(f"\n[DONE] Pruned model saved to: {out_model_dir}")

    if args.make_zip:
        zip_base = "submit"
        zip_path = shutil.make_archive(zip_base, "zip", out_model_dir)
        print(f"[DONE] Created zip: {zip_path}")
    else:
        print("[NEXT] zip ./model -> submit.zip")


if __name__ == "__main__":
    main()
