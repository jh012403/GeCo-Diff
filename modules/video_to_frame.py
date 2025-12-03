import cv2
import os
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save frames")
    # [추가] 특정 비디오 이름만 처리하기 위한 인자
    parser.add_argument("--target_name", type=str, required=False, help="Specific video name to process (without extension)")
    args = parser.parse_args()

    # mp4 파일들 불러오기
    all_videos = sorted(glob(os.path.join(args.input_dir, "*.mp4")))
    
    # [수정] 타겟이 지정되어 있으면 리스트 필터링
    if args.target_name:
        target_file = os.path.join(args.input_dir, f"{args.target_name}.mp4")
        if target_file in all_videos:
            video_files = [target_file]
        else:
            print(f"⚠️ Warning: Target video '{args.target_name}.mp4' not found in input dir.")
            video_files = []
    else:
        video_files = all_videos

    if not video_files:
        print(f"❌ 처리할 mp4 파일을 찾을 수 없습니다: {args.input_dir}")
        return

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 출력 경로: output_dir/video_name
        save_dir = os.path.join(args.output_dir, video_name)
        
        # 이미 처리된 것 같으면 스킵 (선택사항)
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
             print(f"⏩ {video_name} 이미 프레임이 존재합니다. 스킵.")
             continue

        os.makedirs(save_dir, exist_ok=True)
        frame_count = 0
        success = True
        print(f"🎞️ Extracting: {video_name} -> {save_dir}")

        while success:
            success, frame = cap.read()
            if not success:
                break
            
            frame_filename = os.path.join(save_dir, f"{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"   ✅ Done: {frame_count} frames")

if __name__ == "__main__":
    main()