import os
from pathlib import Path
import argparse
import numpy as np
import lpips
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torch
import shutil

# -----------------------------
# 기본 설정 (CPU 고정) - 원본 유지
# -----------------------------
DEVICE = "cpu"
LPIPS_DUP_THRESH = 0.40   # 중복 판단 기준
SHARP_THRESH = 20.0       # 흐림 판단 기준
MIN_KEEP = 18             # 최소 보존 이미지 수

# -----------------------------
# 유틸 함수 - 원본 유지
# -----------------------------
def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def pil_to_tensor(img):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def compute_lpips(model, a_pil, b_pil):
    a = pil_to_tensor(a_pil)
    b = pil_to_tensor(b_pil)
    with torch.no_grad():
        return float(model(a, b).cpu().numpy().squeeze())

def compute_sharpness(img_pil):
    arr = np.array(img_pil.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())

# -----------------------------
# Filtering Logic - 원본 유지
# -----------------------------
def simple_filtering(images, lpips_model):
    kept = []
    
    # 원본 파일명 보존을 위해 인덱스 대신 원본 리스트와 매핑하면 좋지만,
    # 여기서는 순서대로 필터링함
    for img in tqdm(images, desc="  Filtering", leave=False):
        # 1) 흐림 체크
        sharp = compute_sharpness(img)
        if sharp < SHARP_THRESH:
            continue

        # 2) 중복 체크
        is_dup = any(compute_lpips(lpips_model, img, k) < LPIPS_DUP_THRESH for k in kept)
        if is_dup:
            continue

        kept.append(img)

    # 최소 보존 개수 강제
    if len(kept) < MIN_KEEP:
        print(f"  [WARN] Too many filtered ({len(kept)}) -> Keeping first {MIN_KEEP} images")
        kept = images[:MIN_KEEP]

    return kept

# -----------------------------
# Main Process - 경로 및 타겟 설정 수정
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # 쉘 스크립트에서 받아올 인자들
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input frames")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save filtered images")
    parser.add_argument("--scene_name", type=str, required=False, help="Specific scene name to process (e.g., shark)")
    args = parser.parse_args()

    input_root = args.input_dir
    output_root = args.output_dir
    target_scene = args.scene_name

    print(f"📂 Input: {input_root}")
    print(f"📂 Output: {output_root}")
    if target_scene:
        print(f"🎯 Target Scene: {target_scene}")

    if not os.path.exists(input_root):
        print("❌ Input directory not found. Please run Step 2 first.")
        return

    # LPIPS 모델 로드 (루프 밖에서 한 번만 로드)
    print("-> Loading LPIPS model...")
    torch.backends.mkldnn.enabled = False
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)

    # 객체별 폴더 순회 (shark, dino ...)
    scene_dirs = sorted([d for d in Path(input_root).iterdir() if d.is_dir()])
    
    if not scene_dirs:
        print("❌ No scene directories found inside input root.")
        return

    for scene_dir in scene_dirs:
        current_scene_name = scene_dir.name
        
        # [검증된 수정] 타겟 이름이 주어졌다면, 이름이 일치하는 폴더만 처리
        if target_scene and current_scene_name != target_scene:
            continue

        print(f"\n🚀 Processing Scene: {current_scene_name}")

        output_dir = os.path.join(output_root, current_scene_name)
        os.makedirs(output_dir, exist_ok=True)

        # 이미지 로드
        image_paths = sorted([
            str(p) for p in scene_dir.glob("*") 
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if not image_paths:
            print(f"  [Skip] No images in {current_scene_name}")
            continue

        print(f"  -> Loaded {len(image_paths)} images")
        images = [load_image(p) for p in image_paths]

        # 필터링 수행
        kept = simple_filtering(images, lpips_model)

        # 저장 (파일명은 00000.png 포맷으로 재정렬)
        for i, img in enumerate(kept):
            save_path = os.path.join(output_dir, f"{i:05d}.png")
            save_image(img, save_path)

        print(f"  ✅ Saved {len(kept)} images to {output_dir}")

if __name__ == "__main__":
    main()