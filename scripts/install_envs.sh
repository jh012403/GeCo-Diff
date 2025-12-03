#!/bin/bash

# ====================================================
# GeCo-Diff Environment Installer
# ====================================================

# 스크립트 실행 중 에러가 나도 멈추지 않고 다음 환경 설치 시도 (이미 설치된 경우 등 대비)
set +e

# 현재 스크립트 위치를 기준으로 프로젝트 루트 및 환경 설정 폴더 경로 찾기
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_ROOT/environment"

# Conda 초기화
eval "$(conda shell.bash hook)"

echo "=========================================="
echo "🚀 GeCo-Diff Environment Setup Started"
echo "📂 Environment Dir: $ENV_DIR"
echo "=========================================="

# 설치할 환경 목록 (파일명과 동일해야 함)
declare -a envs=("sv3d" "or_filter" "sr" "vggt" "gaussian_splatting")

for env_name in "${envs[@]}"; do
    yaml_file="$ENV_DIR/$env_name.yaml"
    
    echo -e "\n------------------------------------------"
    echo "🛠️  Installing environment: [$env_name]"
    echo "📄 Source: $yaml_file"
    echo "------------------------------------------"

    if [ -f "$yaml_file" ]; then
        # 환경 생성 실행
        conda env create -f "$yaml_file"
        
        if [ $? -eq 0 ]; then
            echo "✅ [$env_name] created successfully."
        else
            echo "⚠️  [$env_name] creation failed or already exists."
            echo "   (If it exists, you can ignore this warning.)"
        fi
    else
        echo "❌ Error: YAML file not found at $yaml_file"
    fi
done

echo -e "\n=========================================="
echo "🎉 All setup steps finished!"
echo "   Please check above for any errors."
echo "=========================================="