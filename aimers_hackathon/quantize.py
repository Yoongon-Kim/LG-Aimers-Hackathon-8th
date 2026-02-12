import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "/home/yjjang/aimers/aimers_hackathon/base_model"
OUT_DIR = "/home/yjjang/aimers/aimers_hackathon/model"

print("[INFO] 모델을 CPU로 명시적 로드 중 (BF16 정밀도 유지)...")
# 1. 모델을 CPU로 명시적 로드 (BF16 정밀도 유지)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="cpu", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("[INFO] FP8 레시피 설정 중...")
# 2. 레시피 설정 (FP8)
recipe = QuantizationModifier(targets="Linear", scheme="FP8")

print("[INFO] 데이터셋 준비 중...")
# 3. 데이터셋 준비
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train[:256]")

def tokenize_fn(examples):
    texts = []
    for conv in examples["conversations"]:
        text = tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        texts.append(text)
    # 숫자 데이터(input_ids 등)만 반환
    return tokenizer(texts, padding="max_length", max_length=512, truncation=True)

# ⚠️ 핵심: remove_columns 옵션을 추가하여 원본 데이터 컬럼을 삭제합니다.
ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

print("[INFO] 데이터셋 준비 완료 (불필요한 컬럼 제거됨)")

print("[INFO] FP8 양자화 실행 중 (이 과정이 수십 분 걸려야 정상입니다!)...")
# 4. 양자화 실행 (이때 1초 만에 끝나면 안 됩니다! 수십 분이 걸려야 정상입니다)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    output_dir=OUT_DIR,
    save_compressed=True
)

print("[INFO] 토크나이저 저장 중...")
tokenizer.save_pretrained(OUT_DIR)

print(f"✅ 체크: {OUT_DIR}/model.safetensors 용량이 줄었는지 확인하세요!")

# ========== 양자화 검증 ==========
import os
import json

print("\n" + "="*60)
print("🔍 양자화 검증 시작")
print("="*60)

# 1. 파일 크기 확인
model_size = os.path.getsize(f"{OUT_DIR}/model.safetensors") / (1024**3)
print(f"\n1️⃣ 파일 크기: {model_size:.2f} GB")
if model_size < 2.0:
    print("   ✅ 성공! (2.4GB → {:.2f}GB로 감소)".format(model_size))
else:
    print("   ❌ 실패! 파일이 줄어들지 않았습니다.")

# 2. config.json에서 양자화 설정 확인
config_path = f"{OUT_DIR}/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

print(f"\n2️⃣ 양자화 설정 확인:")
if "quantization_config" in config:
    print("   ✅ quantization_config 발견!")
    print(f"   설정: {json.dumps(config['quantization_config'], indent=2)}")
else:
    print("   ❌ quantization_config가 없습니다!")
    print("   현재 config.json 키:", list(config.keys()))

print("\n" + "="*60)
if model_size < 2.0 and "quantization_config" in config:
    print("🎉 양자화 완벽 성공!")
else:
    print("⚠️ 양자화에 문제가 있습니다.")
print("="*60)