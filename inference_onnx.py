#!/usr/bin/env python3
"""
inference_onnx.py — Single & Batch Image Inference for BiRefNet-Turbo via ONNX Runtime

Usage:
    python inference_onnx.py --image input.jpg --model birefnet_general_swinL_1024x576_int8.onnx
    python inference_onnx.py --input-dir ./images --model birefnet_general_swinL_1024x576_int8.onnx --out-dir ./masks
"""

import os
import sys
import argparse
import time
from glob import glob
import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not found. Please install: pip install onnxruntime (or onnxruntime-gpu)")
    sys.exit(1)


def run_inference(image_path: str, session, input_name: str, in_shape: tuple, out_dir: str):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    mask_path = os.path.join(out_dir, f"{base_name}_mask.png")
    cutout_path = os.path.join(out_dir, f"{base_name}_cutout.png")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[WARN] Failed to read {image_path}")
        return

    orig_h, orig_w = img_bgr.shape[:2]
    in_h, in_w = in_shape

    # Preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = ((resized.astype(np.float32) / 255.0) - mean) / std
    tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    # Run
    t0 = time.time()
    logits = session.run(None, {input_name: tensor})[0]
    elapsed_ms = (time.time() - t0) * 1000.0

    # Sigmoid & resize back
    prob = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    mask = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Save Mask
    cv2.imwrite(mask_path, mask_u8)

    # Save RGBA Cutout
    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask_u8])
    cv2.imwrite(cutout_path, rgba)

    print(f"[{elapsed_ms:.1f}ms] {image_path} -> {mask_path}, {cutout_path}")


def main():
    parser = argparse.ArgumentParser(description="BiRefNet-Turbo Fast ONNX Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"], help="Device: cpu or cuda")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.intra_op_num_threads = args.threads

    providers = ["CPUExecutionProvider"]
    if args.device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"Loading ONNX Model: {args.model} on {args.device.upper()}...")
    session = ort.InferenceSession(args.model, opts, providers=providers)
    input_name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    in_h = shape[2] if isinstance(shape[2], int) else 1024
    in_w = shape[3] if isinstance(shape[3], int) else 576

    images = []
    if args.image:
        images.append(args.image)
    if args.input_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            images.extend(glob(os.path.join(args.input_dir, ext)))
        images = sorted(list(set(images)))

    if not images:
        print("[ERROR] Please provide --image <file> or --input-dir <dir>")
        sys.exit(1)

    print(f"Processing {len(images)} image(s)...")
    for img_p in images:
        run_inference(img_p, session, input_name, (in_h, in_w), args.out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
