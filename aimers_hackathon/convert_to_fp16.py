#!/usr/bin/env python3
"""
Convert pruned model from fp32 to fp16 for final submission.

modelffn (fp32) → modelffn_fp16 (fp16)
"""

from __future__ import annotations

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC_DIR = "./modelffn"
DST_DIR = "./modelffn_fp16"

def main():
    print("=" * 80)
    print("FP32 → FP16 Conversion")
    print("=" * 80)
    
    if not os.path.isdir(SRC_DIR):
        print(f"[ERROR] Source directory not found: {SRC_DIR}")
        sys.exit(1)
    
    if os.path.exists(DST_DIR):
        print(f"[WARN] {DST_DIR} already exists.")
        response = input("Delete and continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
        import shutil
        shutil.rmtree(DST_DIR)
    
    os.makedirs(DST_DIR, exist_ok=True)
    
    print(f"\n[1/3] Loading pruned model from {SRC_DIR} (fp32)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SRC_DIR,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )
    load_time = time.time() - t0
    print(f"      Loaded in {load_time:.1f}s")
    
    print("\n[2/3] Converting to fp16...")
    t0 = time.time()
    model = model.to(dtype=torch.float16)
    convert_time = time.time() - t0
    print(f"      Converted in {convert_time:.1f}s")
    
    tokenizer = AutoTokenizer.from_pretrained(
        SRC_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    print("\n[3/3] Saving fp16 model...")
    t0 = time.time()
    model.save_pretrained(DST_DIR, safe_serialization=True)
    tokenizer.save_pretrained(DST_DIR)
    save_time = time.time() - t0
    print(f"      Saved in {save_time:.1f}s")
    
    # Size comparison
    print("\n" + "=" * 80)
    print("Size Comparison")
    print("=" * 80)
    
    src_size = os.path.getsize(os.path.join(SRC_DIR, "model.safetensors"))
    dst_size = os.path.getsize(os.path.join(DST_DIR, "model.safetensors"))
    
    print(f"Source (fp32): {src_size / (1024**3):.2f} GB")
    print(f"Target (fp16): {dst_size / (1024**3):.2f} GB")
    print(f"Reduction: {(1 - dst_size/src_size)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ FP16 Conversion Completed")
    print("=" * 80)
    print(f"Output: {DST_DIR}")
    print(f"Total time: {load_time + convert_time + save_time:.1f}s")
    print("\nNext steps:")
    print("  1. Copy LICENSE, README.md, assets/ from base_model")
    print("  2. Use make_submit1.sh (update to use modelffn_fp16)")
    print("  3. Run validation tests")

if __name__ == "__main__":
    main()
