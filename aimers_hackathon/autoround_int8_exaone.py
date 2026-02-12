import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

# ===============================
# 1. Paths
# ===============================
BASE_MODEL = "./base_model"          # 원본 모델 (절대 수정 X)
OUT_DIR = "./autoround_int8_exaone"  # 결과 저장 디렉토리

# ===============================
# 2. Calibration 설정
# ===============================
DATASET_ID = "pile-10k"          # AutoRound 지원 데이터셋
NUM_CALIBRATION_SAMPLES = 128    # CPU 기준 현실적인 값 (reduced for stability)
MAX_SEQUENCE_LENGTH = 512

# ===============================
# 3. CPU 환경 고정
# ===============================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)
torch.set_num_interop_threads(1)

def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("[INFO] Loading base model (FP32, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,   # CPU AutoRound는 FP32 권장
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,
    )
    model.eval()

    print("[INFO] Initializing AutoRound (INT8, CPU)...")
    quantizer = AutoRound(
        model,
        tokenizer,
        bits=8,                    # ⭐ CPU에서는 INT8이 정석
        group_size=128,
        sym=True,                  # Symmetric quantization
        iters=0,                   # RTN mode (no tuning, fast)
        seqlen=MAX_SEQUENCE_LENGTH,
        nsamples=NUM_CALIBRATION_SAMPLES,
        batch_size=1,              # 안정성 우선 (CPU)
        dataset=DATASET_ID,        # HuggingFace 데이터셋 ID
        device_map="cpu",          # CPU 강제
        amp=False,                 # CPU는 AMP 미지원
    )

    print("[INFO] Starting AutoRound quantization and saving (this takes time)...")
    os.makedirs(OUT_DIR, exist_ok=True)
    quantizer.quantize_and_save(output_dir=OUT_DIR, format="auto_round")
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[DONE] AutoRound INT8 model saved to: {OUT_DIR}")
    print("[INFO] base_model is untouched.")

if __name__ == "__main__":
    main()
