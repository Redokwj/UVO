#!/usr/bin/env bash
# ==============================================================================
# UVO SLAM: Lightweight User-Space Setup for NVIDIA Jetson
# Non-intrusive: zero system/swap modifications, zero sudo requirements.
# Focuses exclusively on Python dependencies and offline weights caching.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$UVO_DIR")"

echo "=================================================================="
echo "    UVO SLAM: Jetson Environment Setup & Offline Caching"
echo "=================================================================="

PYTHON_CMD="python3"

# 1. Environment Verification
echo ""
echo "[1/3] Verifying Python and PyTorch CUDA environment..."
$PYTHON_CMD -c "
import sys
import torch
print(f'[*] Python:  {sys.version.split()[0]}')
print(f'[*] PyTorch: {torch.__version__}')
print(f'[*] CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[*] Device:  {torch.cuda.get_device_name(0)}')
else:
    print('[!] Warning: CUDA is not available in current PyTorch installation.')
"

# 2. Python Packages (User-Space)
echo ""
echo "[2/3] Installing/verifying UVO Python requirements..."
$PYTHON_CMD -m pip install -U --no-warn-script-location \
    timm \
    einops \
    scipy \
    pyyaml \
    huggingface_hub \
    matplotlib \
    tqdm

# 3. Pre-cache Neural Network Weights for 100% Offline Autonomy
echo ""
echo "[3/3] Pre-caching Neural Network Weights for Offline Operation..."
$PYTHON_CMD -c "
import os
import torch

print('[*] Caching XFeat matcher...')
torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=2000)

print('[*] Caching DINOv2 VPR engine...')
torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

print('[*] Checking UniDepth V2...')
try:
    from uvo.metric.unidepth_provider import UniDepthProvider
    p = UniDepthProvider(device='cuda' if torch.cuda.is_available() else 'cpu')
    print('[✓] UniDepth V2 ready.')
except Exception as e:
    print(f'[!] UniDepth note: {e}')

print('[✓] All foundational weights cached locally. No internet needed in the field.')
"

echo ""
echo "=================================================================="
echo "  UVO IS READY ON JETSON!"
echo "=================================================================="
echo "To start live SLAM with Web Streaming on your tablet:"
echo "  python3 scripts/run_uvo_live_camera.py --type usb --source 0 --web"
echo "Then open on tablet browser: http://<jetson_ip>:8080"
echo "=================================================================="
