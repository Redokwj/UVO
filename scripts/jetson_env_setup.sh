#!/usr/bin/env bash
# ==============================================================================
# UVO (Unnamed Visual Odometry) - NVIDIA Jetson Orin Environment Setup Blueprint
# Target Platforms: Jetson Orin Nano (8GB), Orin NX (16GB), AGX Orin (32/64GB)
# ==============================================================================

set -e

echo "=== [1/5] Setting Jetson High-Performance Clocks (MAXN) ==="
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0 || true
    sudo jetson_clocks || true
    echo "Power profile set to MAXN and clocks locked to maximum."
else
    echo "nvpmodel not found, skipping hardware clock lock."
fi

echo "=== [2/5] Checking PyTorch CUDA & GPU Capabilities ==="
python3 -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
"

echo "=== [3/5] Setting Environment Variables for Low-Latency UVO ==="
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "=== [4/5] Testing UVO Pipeline Components ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from uvo.pipeline import UVOPipeline, UVOConfig
from uvo.backend.factors import NonHolonomicTrackedFactor, MarginalizationPriorFactor
print('UVO modules successfully imported on Jetson environment!')
"

echo "=== [5/5] Ready for UVO Mission Execution ==="
echo "To run live camera tracking on Jetson:"
echo "  python3 UVO/scripts/run_uvo_live_camera.py --jetson --source csi:0"
echo "To run offline dataset benchmark:"
echo "  python3 UVO/scripts/run_uvo_offline.py --video front_190610.mp4"
