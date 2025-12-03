

SV3D : https://github.com/Stability-AI/generative-models
SR : https://github.com/xinntao/Real-ESRGAN (huggingface에서 Weight만 사용)
VGGT : https://github.com/facebookresearch/vggt
3DGS : https://github.com/graphdeco-inria/gaussian-splatting

input image -> ~.py -> ~.py -> ~.py -> output

### 입력 이미지 디렉토리
input/

### 이미지 -> SV3D
SV3D [conda sv3d]
> python scripts/sampling/simple_video_sample.py --input_path <path/to/image.png> --version sv3d_u --elevations_deg 10.0

### SV3D 결과 및 프레임 추출
SV3D video
+
videotoframe.py

### 프레임 이미지 디렉토리
=> frame image/

### 필터링
Filter [conda or_filter]
> orbit_filter.py
> Environment : orbit.yml

### 필터링된 이미지 디렉토리
=> filtered image

### SR 과정
SR2 [conda sr]
> frame_SR2.py
> Environment : sr.yml

### SR된 이미지
=> SR image

### VGGT로 txt추출
VGGT [conda vggt]
> 

### images.txt, cameras.txt, point3D.txt등 생성

### 3DGS 학습
3DGS [conda gaussian_splatting]

### 결과 디렉토리
output/