from __future__ import annotations

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import apply
from llmcompressor.recipe import Recipe


# ===============================
# 0. CPU 환경 고정 (중요)
# ===============================
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TORCH_NUM_THREADS"] = "8"

torch.set_num_threads(8)
torch.set_num_interop_threads(1)


# ===============================
# 1. Paths
# ===============================
MODEL_ID = "./base_model"  # base_model 경로
OUT_DIR = "./quant_cpu_w4w8bf16_exaone"


# ===============================
# 2. Calibration 설정
# ===============================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 512


# ===============================
# 3. Quantization 기본 설정
# ===============================
BASE_SCHEME = "W4A16"


def main() -> None:
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("[INFO] Loading model (BF16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,
    )
    model.eval()
    print("[INFO] Model loaded.")

    # ===============================
    # 4. 모듈 이름 sanity check
    # ===============================
    mods = dict(model.named_modules())
    for i in [17, 18, 21, 22, 26, 29]:
        for n in ["self_attn.o_proj", "mlp.gate_proj"]:
            name = f"model.layers.{i}.{n}"
            print(f"[CHECK] exists={name in mods} :: {name}")

    # ===============================
    # 5. IGNORE (BF16 보호 구간만)
    # ===============================
    IGNORE = [
        "embed_tokens",
        "lm_head",
    ]

    # o_proj: BF16 보호 (22~29)
    IGNORE += [
        f"model.layers.{i}.self_attn.o_proj"
        for i in range(22, 30)
    ]

    # gate_proj: BF16 보호 (27~29)
    IGNORE += [
        f"model.layers.{i}.mlp.gate_proj"
        for i in range(27, 30)
    ]

    print(f"[INFO] BF16 ignore count = {len(IGNORE)}")

    # ===============================
    # 6. Calibration dataset
    # ===============================
    print("[INFO] Loading calibration dataset...")
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    print("[INFO] Calibration dataset ready.")

    # ===============================
    # 7. GPTQ Recipe (Python dict 형식)
    # ===============================
    # o_proj과 gate_proj ignore 리스트 작성
    o_proj_ignore = [f"model.layers.{i}.self_attn.o_proj" for i in range(22, 30)]
    gate_proj_ignore = [f"model.layers.{i}.mlp.gate_proj" for i in range(27, 30)]
    
    ignore_list = IGNORE + o_proj_ignore + gate_proj_ignore
    
    o_proj_targets = [f"model.layers.{i}.self_attn.o_proj" for i in range(18, 22)]
    gate_proj_targets = [f"model.layers.{i}.mlp.gate_proj" for i in range(24, 27)]
    
    # YAML recipe 딕셔너리로 구성
    recipe_dict = {
        "version": 1,
        "stages": {
            "quantization_stage": {
                "run_type": "oneshot",
                "modifiers": {
                    "gptq_w4": {
                        "type": "GPTQModifier",
                        "scheme": "W4A16",
                        "targets": [
                            "model.layers.*.self_attn.q_proj",
                            "model.layers.*.self_attn.k_proj",
                            "model.layers.*.self_attn.v_proj",
                            "model.layers.*.mlp.up_proj",
                            "model.layers.*.mlp.down_proj",
                        ],
                        "ignore": ignore_list,
                        "block_size": 128,
                        "dampening_frac": 0.01,
                    },
                    "gptq_o_proj_w8": {
                        "type": "GPTQModifier",
                        "scheme": "W8A16",
                        "targets": o_proj_targets,
                        "block_size": 128,
                        "dampening_frac": 0.01,
                    },
                    "gptq_gate_proj_w8": {
                        "type": "GPTQModifier",
                        "scheme": "W8A16",
                        "targets": gate_proj_targets,
                        "block_size": 128,
                        "dampening_frac": 0.01,
                    },
                }
            }
        }
    }
    
    print(f"[INFO] Recipe dict created")
    
    # Recipe 객체 생성 (model_validate 사용)
    recipe = Recipe.model_validate(recipe_dict)
    print(f"[INFO] Recipe object validated")

    # ===============================
    # 8. GPTQ 실행
    # ===============================
    print("[INFO] Starting GPTQ (CPU, oneshot)...")
    apply(
        model=model,
        recipe=recipe,
        calib_data=ds,
    )
    print("[INFO] GPTQ finished.")

    # ===============================
    # 9. 결과 dtype 확인
    # ===============================
    mods = dict(model.named_modules())
    print("\n[POST-CHECK] dtype verification")
    for i in [17, 18, 21, 22, 26, 29]:
        for n in ["self_attn.o_proj", "mlp.gate_proj"]:
            name = f"model.layers.{i}.{n}"
            m = mods.get(name)
            if m is not None and hasattr(m, "weight"):
                print(
                    f"layer={i:<2} {n:<20} dtype={m.weight.dtype}"
                )

    # ===============================
    # 10. Save
    # ===============================
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"\n[INFO] Quantized model saved to: {OUT_DIR}")
    print("[INFO] Next step: zip model/ into submit.zip")


if __name__ == "__main__":
    main()
