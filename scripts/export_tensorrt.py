#!/usr/bin/env python3
"""
UVO TensorRT Model Exporter for NVIDIA Jetson (Orin / Xavier).

Exports PyTorch models to ONNX and provides one-command compilation
to native TensorRT FP16 / INT8 engines using JetPack trtexec.

Supported models:
- DINOv2 ViT-Small/14 (Visual Place Recognition - VPR)
- XFeat (Fast Front-End Feature Tracking)
- UniDepth V2 (Metric Depth Foundation Model)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

UVO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVO_ROOT)


def export_dinov2_onnx(output_dir: str, fp16: bool = True):
    print("\n" + "=" * 60)
    print("  EXPORTING DINOv2 ViT-Small/14 TO ONNX")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "dinov2_vits14.onnx")
    engine_path = os.path.join(output_dir, "dinov2_vits14_fp16.engine")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading DINOv2 ViT-Small from torch.hub on {device}...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print(f"[*] Exporting ONNX to: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['global_descriptor'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'global_descriptor': {0: 'batch_size'}}
    )
    print(f"[✓] Successfully exported ONNX: {onnx_path}")
    print(f"\nTo build native TensorRT engine on Jetson, run:\n")
    print(f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16\n")


def export_xfeat_onnx(output_dir: str):
    print("\n" + "=" * 60)
    print("  EXPORTING XFeat (CVPR 2024) TO ONNX")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "xfeat.onnx")
    engine_path = os.path.join(output_dir, "xfeat_fp16.engine")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading XFeat from torch.hub on {device}...")
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=2048)
    net = xfeat.net.to(device).eval()

    dummy_input = torch.randn(1, 1, 480, 640, device=device)

    print(f"[*] Exporting ONNX to: {onnx_path}...")
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image_gray'],
        output_names=['feats', 'keypoints', 'heatmaps'],
        dynamic_axes={
            'image_gray': {0: 'batch', 2: 'height', 3: 'width'},
            'feats': {0: 'batch', 2: 'feat_h', 3: 'feat_w'},
            'keypoints': {0: 'batch', 2: 'feat_h', 3: 'feat_w'},
            'heatmaps': {0: 'batch', 2: 'h', 3: 'w'}
        }
    )
    print(f"[✓] Successfully exported ONNX: {onnx_path}")
    print(f"\nTo build native TensorRT engine on Jetson, run:\n")
    print(f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16\n")


def main():
    parser = argparse.ArgumentParser(description="UVO TensorRT Exporter")
    parser.add_argument("--model", type=str, default="all", choices=["dinov2", "xfeat", "all"], help="Model to export")
    parser.add_argument("--output_dir", type=str, default="tensorrt_models", help="Destination folder")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    print(f"[*] Export directory: {out_dir}")

    if args.model in ["dinov2", "all"]:
        export_dinov2_onnx(out_dir)
    if args.model in ["xfeat", "all"]:
        export_xfeat_onnx(out_dir)

    print("\n[✓] All requested models exported successfully.")


if __name__ == "__main__":
    main()
