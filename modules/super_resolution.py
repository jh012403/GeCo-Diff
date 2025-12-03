import os
import sys
import shutil
import torch
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse

# --- RealESRGAN imports ---
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- diffusers imports ---
try:
    from diffusers import StableDiffusionUpscalePipeline, StableDiffusionImg2ImgPipeline
    from diffusers import DDIMScheduler
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    DIFFUSERS_AVAILABLE = True
except Exception as e:
    # print(f"⚠️ Diffusers import failed: {e}")
    DIFFUSERS_AVAILABLE = False

# =========================================================
# 🔧 경로 설정 (기본값 계산용 - 실제로는 args로 덮어씌워짐)
# =========================================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 기본값 (실행 시 인자로 대체됨)
INPUT_ROOT = os.path.join(project_root, "outputs", "3_orbit_filtered_image")
OUTPUT_ROOT = os.path.join(project_root, "outputs", "4_SR_image")
WEIGHTS_DIR = os.path.join(project_root, "third_party", "Real-ESRGAN", "weights")

# 임시 폴더 (Diffusion 사용 시)
TMP_DIFFUSION_DIR = os.path.join(project_root, "outputs", "_tmp_diffusion")

# =========================================================
# User configuration
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = True

# RealESRGAN params
rs_scale = 4
tile = 0
tile_pad = 10
half = False

# Diffusion params (Optional)
DIFFUSION_IMG2IMG_MODEL = "runwayml/stable-diffusion-v1-5"
enhancement_prompt = "photorealistic, ultra-detailed, high-resolution photograph, realistic textures, natural lighting, no artifacts"
negative_prompt = "cartoon, painting, low-res, artifacts, text, watermark"
strength = 0.45
guidance_scale = 7.5
num_inference_steps = 20

# -------------------------
# Helper Functions
# -------------------------
def build_diffusion_pipeline():
    if not DIFFUSERS_AVAILABLE:
        return None
    try:
        print(f"  -> Loading Diffusion Pipeline: {DIFFUSION_IMG2IMG_MODEL}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            DIFFUSION_IMG2IMG_MODEL,
            torch_dtype=torch.float16 if (use_fp16 and device=="cuda") else torch.float32
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    except Exception as e:
        print(f"  [Warn] Failed to load diffusion pipeline: {e}")
        return None

def build_realesrgan():
    # 전역 변수 WEIGHTS_DIR 사용 (main에서 업데이트됨)
    real_esrgan_weight = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
    if not os.path.isfile(real_esrgan_weight):
        print(f"❌ Error: Weight file not found at {real_esrgan_weight}")
        print("   Please download 'RealESRGAN_x4plus.pth' to the 'weights/' folder.")
        sys.exit(1)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=rs_scale)
    sr = RealESRGANer(
        scale=rs_scale,
        model_path=real_esrgan_weight,
        model=model,
        dni_weight=None,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=half,
        device=device
    )
    return sr

def process_scene(scene_dir, output_dir, sr_model, diff_pipe=None):
    frame_files = sorted([f for f in os.listdir(scene_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    if not frame_files:
        print(f"  [Skip] No images in {os.path.basename(scene_dir)}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if diff_pipe:
        os.makedirs(TMP_DIFFUSION_DIR, exist_ok=True)

    print(f"  -> Processing {len(frame_files)} frames...")

    for fname in tqdm(frame_files, leave=False):
        src_path = os.path.join(scene_dir, fname)
        tmp_path = src_path

        # 1. Diffusion Enhancement (Optional)
        if diff_pipe:
            try:
                pil_img = Image.open(src_path).convert("RGB")
                w, h = pil_img.size
                init_resized = pil_img.resize((w*2, h*2), resample=Image.Resampling.BICUBIC)
                
                out = diff_pipe(
                    prompt=enhancement_prompt,
                    negative_prompt=negative_prompt,
                    image=init_resized,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]

                tmp_path = os.path.join(TMP_DIFFUSION_DIR, fname)
                out.save(tmp_path)
            except Exception:
                tmp_path = src_path # 실패 시 원본 사용

        # 2. Real-ESRGAN Upscaling
        try:
            img_cv = cv2.imread(tmp_path, cv2.IMREAD_COLOR)  # BGR
            # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # RealESRGANer handles reading? usually expects BGR if using cv2
            
            sr_img, _ = sr_model.enhance(img_cv, outscale=rs_scale)
            
            save_path = os.path.join(output_dir, fname)
            cv2.imwrite(save_path, sr_img) # Save as BGR
        except Exception as e:
            print(f"  [Error] Failed on {fname}: {e}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    # 쉘 스크립트에서 받아올 인자들
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images (Step 3 output)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save SR images")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to Real-ESRGAN weights")
    parser.add_argument("--scene_name", type=str, required=False, help="Specific scene name to process")
    args = parser.parse_args()

    # [중요] 전역 변수 업데이트 (build_realesrgan 함수가 이 변수를 참조함)
    global WEIGHTS_DIR
    WEIGHTS_DIR = args.weights_dir

    input_root = args.input_dir
    output_root = args.output_dir
    target_scene = args.scene_name

    print(f"📂 Input Root: {input_root}")
    print(f"📂 Output Root: {output_root}")
    print(f"📂 Weights Dir: {WEIGHTS_DIR}")
    if target_scene:
        print(f"🎯 Target Scene: {target_scene}")

    if not os.path.exists(input_root):
        print("❌ Input directory not found. Please run Step 3 first.")
        return

    # 모델 로드 (한 번만)
    print("\n-> Loading Models...")
    try:
        sr_model = build_realesrgan()
    except SystemExit:
        return
    
    # Diffusion은 너무 무거우면 끄고 싶을 수 있음 (일단 로드 시도)
    diff_pipe = build_diffusion_pipeline()
    if diff_pipe:
        print("✅ Diffusion Pipeline Ready")
    else:
        print("ℹ️ Skipping Diffusion Enhancement (running Real-ESRGAN only)")

    # 폴더 순회
    scene_dirs = sorted([d for d in Path(input_root).iterdir() if d.is_dir()])
    
    for scene_dir in scene_dirs:
        current_scene_name = scene_dir.name
        
        # [검증된 수정] 타겟 이름이 주어졌다면, 이름이 일치하는 폴더만 처리
        if target_scene and current_scene_name != target_scene:
            continue

        print(f"\n🚀 Enhancing Scene: {current_scene_name}")
        
        output_dir = os.path.join(output_root, current_scene_name)
        process_scene(str(scene_dir), output_dir, sr_model, diff_pipe)
        
        print(f"  ✅ Saved to {output_dir}")

    # 청소
    if os.path.exists(TMP_DIFFUSION_DIR):
        try:
            shutil.rmtree(TMP_DIFFUSION_DIR)
        except:
            pass

if __name__ == "__main__":
    main()