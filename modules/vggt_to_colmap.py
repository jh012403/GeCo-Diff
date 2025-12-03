#!/usr/bin/env python3
import os
import sys
import shutil
import torch
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
import types
from pathlib import Path
import glob
import argparse

# =========================================================
# 🔧 라이브러리 경로 설정 (VGGT 모듈 import용)
# =========================================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# VGGT 코드 폴더 (import를 위해 필요)
VGGT_DIR = os.path.join(project_root, "third_party", "vggt")
sys.path.append(VGGT_DIR)

try:
    import demo_colmap as demo
    import pycolmap # 필수
except ImportError as e:
    print(f"❌ Error: {e}")
    print("   Please install pycolmap: pip install pycolmap")
    sys.exit(1)

# =========================================================
# Core Logic (원본 유지)
# =========================================================

# OOM 방지용 배치 추론
def run_VGGT_batched(model, images, dtype, resolution=518, batch_size=16):
    assert len(images.shape) == 4
    assert images.shape[1] == 3
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    all_extrinsics, all_intrinsics, all_depth_maps, all_depth_confs = [], [], [], []
    num_batches = math.ceil(images.shape[0] / batch_size)
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="    VGGT Inference", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, images.shape[0])
            image_batch = images[start_idx:end_idx]
            with torch.cuda.amp.autocast(dtype=dtype):
                image_batch_5d = image_batch[None]
                aggregated_tokens_list_batch, ps_idx_batch = model.aggregator(image_batch_5d)
            pose_enc_batch = model.camera_head(aggregated_tokens_list_batch)[-1]
            extrinsic_batch, intrinsic_batch = demo.pose_encoding_to_extri_intri(pose_enc_batch, image_batch_5d.shape[-2:])
            depth_map_batch, depth_conf_batch = model.depth_head(aggregated_tokens_list_batch, image_batch_5d, ps_idx_batch)
            all_extrinsics.append(extrinsic_batch.squeeze(0).cpu().numpy())
            all_intrinsics.append(intrinsic_batch.squeeze(0).cpu().numpy())
            all_depth_maps.append(depth_map_batch.squeeze(0).cpu().numpy())
            all_depth_confs.append(depth_conf_batch.squeeze(0).cpu().numpy())
    extrinsic = np.concatenate(all_extrinsics, axis=0)
    intrinsic = np.concatenate(all_intrinsics, axis=0)
    depth_map = np.concatenate(all_depth_maps, axis=0)
    depth_conf = np.concatenate(all_depth_confs, axis=0)
    return extrinsic, intrinsic, depth_map, depth_conf

class _Args(types.SimpleNamespace):
    pass

def process_scene(scene_name, src_dir, dst_dir, model_loader_func):
    # 이미지 폴더 준비
    images_dir = os.path.join(dst_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for img in glob.glob(os.path.join(src_dir, "*")):
        shutil.copy(img, images_dir)
        
    args = _Args(
        scene_dir=dst_dir, 
        seed=42, 
        use_ba=False, 
        max_reproj_error=3.0,
        shared_camera=True, 
        camera_type="SIMPLE_PINHOLE", 
        vis_thresh=0.2,
        query_frame_num=12, 
        max_query_pts=8192,
        fine_tracking=True, 
        conf_thres_value=1.5,
    )

    demo.VGGT = model_loader_func
    demo.run_VGGT = run_VGGT_batched

    # 1. VGGT 추론 실행
    with torch.no_grad():
        demo.demo_fn(args)

    # ------------------------------------------------------------------
    # 🔧 폴더 구조 강제 정리 (sparse/ -> sparse/0/)
    # ------------------------------------------------------------------
    sparse_root = os.path.join(dst_dir, "sparse")
    sparse_0 = os.path.join(sparse_root, "0")
    
    if os.path.exists(sparse_root):
        # 만약 sparse 폴더 바로 밑에 .bin 파일들이 흩어져 있다면?
        bin_files = glob.glob(os.path.join(sparse_root, "*.bin"))
        
        if len(bin_files) > 0:
            print(f"   🔧 Organizing files: Moving {len(bin_files)} .bin files to sparse/0/")
            os.makedirs(sparse_0, exist_ok=True)
            for f in bin_files:
                shutil.move(f, sparse_0)
        
        # 0 폴더가 잘 만들어졌는지 확인 후 변환
        if os.path.exists(sparse_0):
            print("   🔄 Converting BIN to TXT...")
            try:
                rec = pycolmap.Reconstruction(sparse_0)
                rec.write_text(sparse_0)
                print(f"   ✅ Saved .txt files to: {sparse_0}")
            except Exception as e:
                print(f"   ⚠️ Conversion failed: {e}")
        else:
            print(f"   ❌ Error: 'sparse/0' folder creation failed. Check {sparse_root}")
    else:
        print(f"   ❌ Error: VGGT output (sparse folder) not found in {dst_dir}")

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    # 쉘 스크립트에서 받아올 인자들
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input SR images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save VGGT outputs")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to VGGT model.pt")
    parser.add_argument("--scene_name", type=str, required=False, help="Specific scene name to process")
    args = parser.parse_args()

    input_root = args.input_dir
    output_root = args.output_dir
    ckpt_path = args.ckpt_path
    target_scene = args.scene_name

    print(f"📂 Input Root: {input_root}")
    print(f"📂 Output Root: {output_root}")
    print(f"📂 Weights: {ckpt_path}")
    if target_scene:
        print(f"🎯 Target Scene: {target_scene}")

    if not os.path.exists(ckpt_path):
        print(f"❌ Error: VGGT model not found at {ckpt_path}")
        return

    # 모델 로드 함수 정의 (클로저로 경로 바인딩)
    def _load_model_local():
        from vggt.models.vggt import VGGT
        m = VGGT()
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state, strict=False)
        m.eval().to("cuda")
        return m

    scene_dirs = sorted([d for d in Path(input_root).iterdir() if d.is_dir()])
    
    if not scene_dirs:
        print("⚠️ No input scenes found. Check Step 3 output.")
        return

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        
        # [검증된 수정] 타겟 이름이 주어졌다면, 이름이 일치하는 폴더만 처리
        if target_scene and scene_name != target_scene:
            continue

        print(f"\n🚀 Geometry: {scene_name}")
        output_dir = os.path.join(output_root, scene_name)
        process_scene(scene_name, str(scene_dir), output_dir, _load_model_local)

if __name__ == "__main__":
    main()