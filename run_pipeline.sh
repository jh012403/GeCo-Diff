#!/bin/bash

# ====================================================
# [설정] 에러 발생 시 즉시 중단
# ====================================================
set -e

# ====================================================
# [설정] 가상환경 이름
# ====================================================
ENV_SV3D="sv3d"
ENV_FILTER="or_filter"
ENV_SR="sr"
ENV_VGGT="vggt"
ENV_3DGS="gaussian_splatting"

if [ -z "$1" ]; then
  echo "❌ Usage: ./run_pipeline.sh <image_path>"
  exit 1
fi

ROOT=$(pwd)
INPUT_IMAGE=$(realpath "$1")
FILENAME=$(basename "$INPUT_IMAGE")
BASENAME="${FILENAME%.*}" # 타겟 이름 (예: shark)

# 출력 폴더
OUT_ROOT="$ROOT/outputs"
OUT_1="$OUT_ROOT/1_sv3d_video"
OUT_2="$OUT_ROOT/2_sv3d_frames"
OUT_3="$OUT_ROOT/3_orbit_filtered_image"
OUT_4="$OUT_ROOT/4_SR_image"
OUT_5="$OUT_ROOT/5_vggt_txt"
OUT_6="$OUT_ROOT/6_3dgs_train"

ITERATIONS=15000

# 폴더 생성
mkdir -p "$OUT_1" "$OUT_2" "$OUT_3" "$OUT_4" "$OUT_5" "$OUT_6"

echo "=========================================="
echo "🚀 GeCo-Diff Pipeline Start"
echo "📂 Target: $BASENAME"
echo "=========================================="

eval "$(conda shell.bash hook)"

# ----------------------------------------------------
# [Step 1] SV3D Video Generation
# ----------------------------------------------------
if [ -n "$(find "$OUT_1" -maxdepth 2 -name "${BASENAME}.mp4" -print -quit)" ]; then
    echo -e "\n⏩ [Step 1] Video found. Skipping..."
    
    # 영상은 있는데 프레임이 없으면 추출 (인자 전달 필수!)
    if [ -z "$(ls -A "$OUT_2/$BASENAME" 2>/dev/null)" ]; then
        echo "   ⚠️ Frames missing. Running extraction only..."
        conda activate $ENV_SV3D
        python modules/video_to_frame.py \
            --input_dir "$OUT_1" \
            --output_dir "$OUT_2" \
            --target_name "$BASENAME"
        conda deactivate
    fi
else
    echo -e "\n🔥 [Step 1] SV3D: Generating Video..."
    conda activate $ENV_SV3D
    
    # OOM 방지
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    cd "$ROOT/third_party/generative-models"
    
    # --- [Config 설정] ---
    REPO_CONFIG="configs/inference/sv3d_u.yaml"
    # (Config 수정 로직 제거됨 - 원본 사용)
    
    if [ ! -f "$REPO_CONFIG" ]; then
        echo "❌ Error: Base config not found at $REPO_CONFIG"
        exit 1
    fi

    # --- [Checkpoint 설정] ---
    mkdir -p checkpoints
    if [ -f "checkpoints/sv3d_u.safetensors" ]; then
        echo "   ✅ Checkpoint found."
    elif [ -f "sv3d_u_model/checkpoints/sv3d_u.safetensors" ]; then
        ln -sf "../sv3d_u_model/checkpoints/sv3d_u.safetensors" "checkpoints/sv3d_u.safetensors"
        echo "   ✅ Checkpoint linked from sv3d_u_model."
    else
        echo "❌ Error: 'sv3d_u.safetensors' missing!"
        exit 1
    fi
    
    # --- [실행] ---
    python scripts/sampling/simple_video_sample.py \
        --input_path "$INPUT_IMAGE" \
        --output_folder "$OUT_1" \
        --version sv3d_u
    
    cd "$ROOT"
    
    # 이름 정리
    LAST_VIDEO=$(ls -t "$OUT_1"/*.mp4 2>/dev/null | head -n 1)
    if [ -n "$LAST_VIDEO" ] && [ "$(basename "$LAST_VIDEO")" != "$BASENAME.mp4" ]; then
        mv "$LAST_VIDEO" "$OUT_1/$BASENAME.mp4"
    fi
    
    # 프레임 추출 실행 (인자 전달 필수!)
    python modules/video_to_frame.py \
        --input_dir "$OUT_1" \
        --output_dir "$OUT_2" \
        --target_name "$BASENAME"
    conda deactivate
fi

# ----------------------------------------------------
# [Step 2] Filtering
# ----------------------------------------------------
if [ -n "$(ls -A "$OUT_3/$BASENAME" 2>/dev/null)" ]; then
    echo -e "\n⏩ [Step 2] Filtered images found. Skipping..."
else
    echo -e "\n🔍 [Step 2] Filtering Unstable Frames..."
    conda activate $ENV_FILTER
    python modules/orbit_filter.py \
        --input_dir "$OUT_2" \
        --output_dir "$OUT_3" \
        --scene_name "$BASENAME"
    conda deactivate
fi

# ----------------------------------------------------
# [Step 3] Super Resolution
# ----------------------------------------------------
if [ -n "$(ls -A "$OUT_4/$BASENAME" 2>/dev/null)" ]; then
    echo -e "\n⏩ [Step 3] SR images found. Skipping..."
else
    echo -e "\n✨ [Step 3] Enhancing Resolution..."
    conda activate $ENV_SR
    SR_WEIGHT="$ROOT/third_party/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
    if [ ! -f "$SR_WEIGHT" ]; then
        mkdir -p "$(dirname "$SR_WEIGHT")"
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O "$SR_WEIGHT"
    fi
    python modules/super_resolution.py \
        --input_dir "$OUT_3" \
        --output_dir "$OUT_4" \
        --weights_dir "$(dirname "$SR_WEIGHT")" \
        --scene_name "$BASENAME"
    conda deactivate
fi

# ----------------------------------------------------
# [Step 4] Geometry (VGGT)
# ----------------------------------------------------
VGGT_DONE=false
if [ -f "$OUT_5/$BASENAME/sparse/0/cameras.txt" ]; then
    VGGT_DONE=true
fi

if [ "$VGGT_DONE" = true ]; then
    echo -e "\n⏩ [Step 4] VGGT output found. Skipping..."
else
    echo -e "\n📐 [Step 4] Estimating Geometry..."
    conda activate $ENV_VGGT
    python modules/vggt_to_colmap.py \
        --input_dir "$OUT_4" \
        --output_dir "$OUT_5" \
        --ckpt_path "$ROOT/third_party/vggt/model.pt" \
        --scene_name "$BASENAME"
    conda deactivate
fi

# ----------------------------------------------------
# [Step 5] 3DGS Training
# ----------------------------------------------------
SCENE_PATH="$OUT_5/$BASENAME"
OUTPUT_PATH="$OUT_6/$BASENAME"

if [ -f "$OUTPUT_PATH/point_cloud/iteration_7000/point_cloud.ply" ]; then
    echo -e "\n⏩ [Step 5] Trained model found. Skipping..."
else
    echo -e "\n🎨 [Step 5] Training 3DGS..."
    conda activate $ENV_3DGS

    if [ -f "$SCENE_PATH/sparse/0/cameras.txt" ]; then
        python third_party/gaussian-splatting/train.py -s "$SCENE_PATH" -m "$OUTPUT_PATH" --iterations $ITERATIONS
    else
        echo "❌ Error: COLMAP data missing for $BASENAME"
        exit 1
    fi
    conda deactivate
fi

# ----------------------------------------------------
# [Step 6] Rendering
# ----------------------------------------------------
echo -e "\n📸 [Step 6] Rendering Images..."
conda activate $ENV_3DGS

TRAIN_DIR="$OUT_6/$BASENAME"
SOURCE_PATH="$OUT_5/$BASENAME"

if [ -d "$TRAIN_DIR" ]; then
    echo "   -> Processing: $BASENAME"
    
    if [ -z "$(find "$TRAIN_DIR/train" -name "renders" -print -quit 2>/dev/null)" ]; then
        python third_party/gaussian-splatting/render.py -m "$TRAIN_DIR" -s "$SOURCE_PATH" --iteration $ITERATIONS --white_background
        echo "      ✅ Rendered."
    else
        echo "      ⏩ Renders exist."
    fi
else
    echo "⚠️ Training output not found for $BASENAME"
fi

conda deactivate

echo "=========================================="
echo "✅ All Pipeline Steps Completed for: $BASENAME"
echo "📊 Results: $OUT_6/$BASENAME"
echo "=========================================="