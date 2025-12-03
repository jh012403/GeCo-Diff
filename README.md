# GeCo-Diff: Geometric Consistent 3D Reconstruction

[![Project Page](https://img.shields.io/badge/Project-Website-87CEEB)](https://jh012403.github.io/GeCo-Diff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Arxiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-B31B1B.svg)](https://arxiv.org/) -->

> **Official implementation of "GeCo-Diff: Geometric Consistent 3D Reconstruction from a Single Image".**

**GeCo-Diff** is an automated end-to-end pipeline that reconstructs high-fidelity 3D Gaussian Splatting scenes from a single input image. It integrates video generation, geometric estimation, and 3D training into a unified workflow without manual intervention.

For more visual results and detailed methodology, please visit our **[Project Page](https://YOUR_USERNAME.github.io/GeCo-Diff/)**.

## 🚀 Quick Start

### 1. Git Clone
We recommend using Anaconda to manage environments.

```bash
git clone [https://github.com/jh012403/GeCo-Diff.git](https://github.com/jh012403/GeCo-Diff.git)
cd GeCo-Diff
```

### 2. Environment Setup
Run the following script to create all necessary Conda environments (sv3d, or_filter, sr, vggt, gaussian_splatting) automatically.

```bash
chmod +x scripts/install_envs.sh # First Time Setup
./scripts/install_envs.sh
```

### 3. Download Pretrained Weights
You can download all necessary model weights (SV3D, VGGT, Real-ESRGAN) using the helper script.
**Note:** You will be prompted to enter your [Hugging Face Access Token](https://huggingface.co/settings/tokens).

```bash
chmod +x scripts/install_weights.sh # First Time Setup
./scripts/install_weights.sh
```

### 4. Usage
We provide a unified shell script run_pipeline.sh that handles environment switching and data flow automatically.

#### Run End-to-End Pipeline
Simply provide the path to your input image.

```bash
chmod +x run_pipeline.sh # First Time Setup
./run_pipeline.sh examples/shark.jpg
```
This script will automatically:
1. Generate an orbital video using SV3D.
2. Filter & Upscale frames using Orbit-Filter and Real-ESRGAN.
3. Estimate Poses using VGGT foundation model.
4. Train 3DGS with background regularization.
5. Render the final 360° video.

### 📂 Output Structure
Results are organized in the outputs/ directory:

- 1_sv3d_video/: Generated orbital videos
- 4_SR_image/: Super-resolved frames (High-frequency details)
- 5_vggt_txt/: Estimated camera poses (COLMAP format)
- 6_3dgs_train/: Final 3DGS models (point_cloud.ply) and renders

```
@misc{geco-diff2025,
  title={GeCo-Diff: Geometric Consistent 3D Reconstruction from a Single Image},
  author={Jihwan Bae, Jonghyun Lee, Jounghoon Jo},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository}
}
```

## 📢 Acknowledgements & Code Modifications

This project is built upon several excellent open-source projects. We express our gratitude to the authors for their contributions.
Some modules have been **modified** to fit our end-to-end automated pipeline:

* **[VGGT](https://github.com/facebookresearch/vggt) (Pose Estimation):**
    * We adapted the inference logic to support **batch processing** for consistent memory usage.
    * Modified camera parameters (e.g., `shared_camera=True`) to optimize for orbital video inputs.
    * Integrated automatic conversion from `.bin` to COLMAP `.txt` format.

* **[SV3D](https://github.com/Stability-AI/generative-models) (Video Generation):**
    * Modified `simple_video_sample.py` to support **sequential frame decoding** to prevent OOM errors on consumer GPUs.
    * Added CLI arguments for seamless integration with shell scripts.

* **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (Super Resolution):**
    * Integrated as a pre-processing module for texture enhancement.

* **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Reconstruction):**
    * Used as the final reconstruction engine with custom training iterations and background regularization options.

Please follow the license of each repository for usage.