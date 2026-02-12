#!/usr/bin/env bash
# Safe submit zip creator
# - Copies selected model dir into temp model/ path
# - Zips synchronously (no race condition)
# - Verifies required files in zip

set -euo pipefail

WORK_DIR="/home/yjjang/aimers/aimers_hackathon"
cd "$WORK_DIR"

SRC_DIR="${1:-fix/model}"
ZIP_NAME="${2:-submit1.zip}"
TMP_DIR="_submit_tmp"

if [ ! -d "$SRC_DIR" ]; then
    echo "[ERROR] Source model directory not found: $SRC_DIR"
    echo "Usage: ./make_submit1.sh [source_model_dir] [zip_name]"
    exit 1
fi

echo "[INFO] Source model: $SRC_DIR"
echo "[INFO] Output zip  : $ZIP_NAME"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR/model"

echo "[INFO] Copying model to temp staging area..."
cp -a "$SRC_DIR"/. "$TMP_DIR/model/"

if [ -f "base_model/LICENSE" ] && [ ! -f "$TMP_DIR/model/LICENSE" ]; then
    cp -f "base_model/LICENSE" "$TMP_DIR/model/"
fi
if [ -f "base_model/README.md" ] && [ ! -f "$TMP_DIR/model/README.md" ]; then
    cp -f "base_model/README.md" "$TMP_DIR/model/"
fi
if [ -d "base_model/assets" ] && [ ! -d "$TMP_DIR/model/assets" ]; then
    cp -a "base_model/assets" "$TMP_DIR/model/"
fi

echo "[INFO] Creating zip synchronously..."
rm -f "$ZIP_NAME"
(
    cd "$TMP_DIR"
    zip -r0q "../$ZIP_NAME" model
)

echo "[INFO] Verifying zip content..."
unzip -l "$ZIP_NAME" | rg -q "model/config.json" || {
    echo "[ERROR] model/config.json not found in zip"
    exit 1
}
unzip -l "$ZIP_NAME" | rg -q "model/tokenizer.json" || {
    echo "[ERROR] model/tokenizer.json not found in zip"
    exit 1
}
unzip -l "$ZIP_NAME" | rg -q "model/model.safetensors|model/model.safetensors.index.json" || {
    echo "[ERROR] model weights not found in zip"
    exit 1
}

echo "[DONE] Zip created: $ZIP_NAME"
ls -lh "$ZIP_NAME"

echo "[INFO] Cleaning temp files..."
rm -rf "$TMP_DIR"
