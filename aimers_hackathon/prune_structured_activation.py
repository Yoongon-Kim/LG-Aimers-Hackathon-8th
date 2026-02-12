"""
Layer-wise Structured Pruning based on Activation Importance
실제 입력 데이터로 activation 측정 후 중요도 낮은 뉴런 제거
"""
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

# ========== 설정 ==========
MODEL_DIR = "./base_model"
OUT_DIR = "./structured_pruned_act30"
PRUNE_RATIO = 0.30  # 30% pruning

CALIB_SAMPLES = 512
MAX_LEN = 512
BATCH_SIZE = 1  # CPU이므로 1로 고정

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")

print("=" * 60)
print("🔥 Structured Pruning with Activation-based Importance")
print("=" * 60)
print(f"Model: {MODEL_DIR}")
print(f"Output: {OUT_DIR}")
print(f"Prune Ratio: {PRUNE_RATIO * 100:.1f}%")
print(f"Calibration Samples: {CALIB_SAMPLES}")
print("=" * 60)

# ========== 모델 로드 ==========
print("\n[1/5] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print(f"✓ Model loaded: {model.config.model_type}")

# ========== Activation 수집용 Hook 설정 ==========
print("\n[2/5] Setting up activation hooks...")
activation_scores = {}
target_layers = []

def hook_fn(name):
    def hook(module, inp, out):
        # out: (B, T, H) or (B, H)
        if isinstance(out, tuple):
            out = out[0]
        
        if out.dim() == 3:
            # (B, T, H) → H 차원 기준으로 L2 norm 계산
            score = out.detach().pow(2).sum(dim=(0, 1))  # (H,)
        elif out.dim() == 2:
            score = out.detach().pow(2).sum(dim=0)
        else:
            return
        
        if name not in activation_scores:
            activation_scores[name] = score.cpu()
        else:
            activation_scores[name] += score.cpu()
    
    return hook

hooks = []

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # embedding과 lm_head는 제외
        if any(x in name for x in ["embed_tokens", "lm_head", "embed"]):
            continue
        
        target_layers.append(name)
        hooks.append(module.register_forward_hook(hook_fn(name)))

print(f"✓ Registered hooks on {len(target_layers)} layers")
print(f"  Sample layers: {target_layers[:3]}")

# ========== Calibration 데이터 수집 ==========
print("\n[3/5] Running calibration on MANTA dataset...")
print(f"  This will take ~30-40 minutes on CPU...")

try:
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train", streaming=False)
    print(f"✓ Dataset loaded: {len(ds)} samples available")
except Exception as e:
    print(f"⚠ Failed to load MANTA-1M: {e}")
    print("  Falling back to wikitext...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Calibration 실행
processed = 0
for i in tqdm(range(min(CALIB_SAMPLES, len(ds))), desc="Calibration"):
    try:
        if "text" in ds[i]:
            text = ds[i]["text"]
        elif "content" in ds[i]:
            text = ds[i]["content"]
        else:
            text = str(ds[i])
        
        if len(text.strip()) < 10:
            continue
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        
        with torch.no_grad():
            model(**inputs)
        
        processed += 1
        
        # 메모리 정리 (중요!)
        if processed % 50 == 0:
            gc.collect()
    
    except Exception as e:
        if i < 5:  # 앞 몇개만 에러 표시
            print(f"\n⚠ Sample {i} failed: {e}")
        continue

print(f"✓ Processed {processed} samples")

# Hook 제거
print("\n[4/5] Computing importance scores...")
for h in hooks:
    h.remove()

# ========== Structured Pruning 적용 ==========
print(f"\n[5/5] Applying structured pruning (ratio={PRUNE_RATIO})...")

pruned_layers = 0
total_pruned_neurons = 0
total_neurons = 0

for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    if name not in activation_scores:
        continue
    
    scores = activation_scores[name]
    num_neurons = scores.numel()
    k = int(num_neurons * PRUNE_RATIO)
    
    if k == 0:
        continue
    
    # 중요도 낮은 뉴런 index (activation 작은 순)
    prune_idx = torch.argsort(scores)[:k]
    
    # Weight를 0으로 마스킹 (구조는 유지)
    with torch.no_grad():
        module.weight[prune_idx, :] = 0.0
        if module.bias is not None:
            module.bias[prune_idx] = 0.0
    
    pruned_layers += 1
    total_pruned_neurons += k
    total_neurons += num_neurons

print(f"✓ Pruned {pruned_layers} layers")
print(f"  Total neurons: {total_neurons:,}")
print(f"  Pruned neurons: {total_pruned_neurons:,}")
print(f"  Prune ratio: {total_pruned_neurons/total_neurons*100:.2f}%")

# ========== 모델 저장 ==========
print(f"\n[6/5] Saving pruned model to {OUT_DIR}...")
model.save_pretrained(OUT_DIR, max_shard_size="2GB")
tokenizer.save_pretrained(OUT_DIR)

print("\n" + "=" * 60)
print("✅ DONE! Structured Activation-based Pruning Complete")
print("=" * 60)
print(f"📁 Output: {OUT_DIR}")
print(f"🧠 Pruned {PRUNE_RATIO*100:.0f}% of neurons based on activation importance")
print("\n💡 Next steps:")
print(f"  1. Test: python test_submission.py {OUT_DIR}")
print(f"  2. Convert to fp16: python convert_to_fp16.py")
print(f"  3. Make submission: ./make_submit1.sh")
print("=" * 60)
