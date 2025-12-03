#!/bin/bash

# ====================================================
# GeCo-Diff Model Weights Downloader
# ====================================================
set -e

# 프로젝트 루트 경로 설정 (스크립트 위치 기준 상위 폴더)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "⬇️  GeCo-Diff Weight Downloader"
echo "=========================================="

# 1. Hugging Face 토큰 입력 받기 (SV3D, VGGT용)
echo ""
echo "🔑 SV3D and VGGT require a Hugging Face Access Token."
read -p "👉 Please paste your Hugging Face Token (Read permissions): " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: Token is empty. Exiting."
    exit 1
fi

echo ""
echo "🚀 Starting downloads..."

# ----------------------------------------------------
# 1. SV3D (Stability AI)
# ----------------------------------------------------
TARGET_DIR="$PROJECT_ROOT/third_party/generative-models/checkpoints"
TARGET_FILE="$TARGET_DIR/sv3d_u.safetensors"

echo -e "\n📦 [1/3] Downloading SV3D..."
if [ -f "$TARGET_FILE" ]; then
    echo "   ⏩ File already exists. Skipping."
else
    mkdir -p "$TARGET_DIR"
    wget --header="Authorization: Bearer $HF_TOKEN" \
         "https://huggingface.co/stabilityai/sv3d/resolve/main/sv3d_u.safetensors" \
         -O "$TARGET_FILE"
    echo "   ✅ SV3D Downloaded."
fi

# ----------------------------------------------------
# 2. VGGT (Meta Research)
# ----------------------------------------------------
TARGET_DIR="$PROJECT_ROOT/third_party/vggt"
TARGET_FILE="$TARGET_DIR/model.pt"

echo -e "\n📦 [2/3] Downloading VGGT..."
if [ -f "$TARGET_FILE" ]; then
    echo "   ⏩ File already exists. Skipping."
else
    mkdir -p "$TARGET_DIR"
    wget --header="Authorization: Bearer $HF_TOKEN" \
         "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt" \
         -O "$TARGET_FILE"
    echo "   ✅ VGGT Downloaded."
fi

# ----------------------------------------------------
# 3. Real-ESRGAN (Open Source)
# ----------------------------------------------------
TARGET_DIR="$PROJECT_ROOT/third_party/Real-ESRGAN/weights"
TARGET_FILE="$TARGET_DIR/RealESRGAN_x4plus.pth"

echo -e "\n📦 [3/3] Downloading Real-ESRGAN..."
if [ -f "$TARGET_FILE" ]; then
    echo "   ⏩ File already exists. Skipping."
else
    mkdir -p "$TARGET_DIR"
    wget "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
         -O "$TARGET_FILE"
    echo "   ✅ Real-ESRGAN Downloaded."
fi

echo -e "\n=========================================="
echo "🎉 All weights downloaded successfully!"
echo "=========================================="