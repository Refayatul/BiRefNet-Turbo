#!/usr/bin/env python3
"""
video_matting.py — High-Performance Streaming Video Matting for BiRefNet-Turbo

Supports:
- ONNX Runtime INT8 and FP32 models (CPU and CUDA/GPU)
- PyTorch models (ZhengPeng7/BiRefNet, local .pth)
- Streaming frame pipeline: No dumping thousands of temporary PNGs to disk
- Generates both binary mask video and extracted subject cutout video
"""

import os
import sys
import argparse
import time
from glob import glob
import cv2
import numpy as np
from PIL import Image

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Try importing PyTorch & Torchvision
try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None
    transforms = None


def get_onnx_session(model_path: str, device: str = "cpu", num_threads: int = 4):
    if ort is None:
        raise ImportError("onnxruntime is not installed. Install with: pip install onnxruntime (or onnxruntime-gpu)")
    
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False  # Prevents excessive RAM pre-allocation on CPU runners
    opts.intra_op_num_threads = num_threads
    
    providers = ["CPUExecutionProvider"]
    if device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    session = ort.InferenceSession(model_path, opts, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 1024, 576]
    
    # Infer target (H, W)
    h = input_shape[2] if isinstance(input_shape[2], int) else 1024
    w = input_shape[3] if isinstance(input_shape[3], int) else 576
    return session, input_name, (h, w)


def get_pytorch_model(model_name_or_path: str, device: str = "cuda"):
    if torch is None:
        raise ImportError("torch is not installed. Install with: pip install torch torchvision")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.birefnet import BiRefNet
    from utils import check_state_dict

    if os.path.isfile(model_name_or_path):
        model = BiRefNet(bb_pretrained=False)
        state_dict = torch.load(model_name_or_path, map_location="cpu", weights_only=True)
        model.load_state_dict(check_state_dict(state_dict))
    else:
        model = BiRefNet.from_pretrained(model_name_or_path)

    dev = "cuda" if device.lower() in ("cuda", "gpu") and torch.cuda.is_available() else "cpu"
    model.to(dev)
    model.eval()
    return model, dev


def process_video(
    video_path: str,
    onnx_session_tuple=None,
    pt_model_tuple=None,
    output_dir: str = None,
    batch_size: int = 1,
    bg_color: tuple = (0, 0, 0),  # BGR for subject cutout background
):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[-1]
    out_dir = output_dir or os.path.dirname(video_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    dst_mask_path = os.path.join(out_dir, f"{base_name}-preds_mask{ext}")
    dst_subject_path = os.path.join(out_dir, f"{base_name}-preds_subject{ext}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n🎬 Processing: {video_path}")
    print(f"   Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_mask = cv2.VideoWriter(dst_mask_path, fourcc, fps, (width, height), isColor=False)
    writer_subj = cv2.VideoWriter(dst_subject_path, fourcc, fps, (width, height), isColor=True)

    # Normalization constants (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frame_idx = 0
    t_start = time.time()

    if onnx_session_tuple is not None:
        session, input_name, (in_h, in_w) = onnx_session_tuple

        while True:
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break

            # Preprocess batch
            tensors = []
            for frame in batch_frames:
                # Resize directly in OpenCV (RGB)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                norm = ((resized.astype(np.float32) / 255.0) - mean) / std
                chw = np.transpose(norm, (2, 0, 1))
                tensors.append(chw)

            batch_tensor = np.stack(tensors, axis=0).astype(np.float32)

            # Inference
            logits = session.run(None, {input_name: batch_tensor})[0]
            # Probabilities via Sigmoid
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Postprocess & write frames
            for i, frame in enumerate(batch_frames):
                p = probs[i, 0]  # shape (in_h, in_w)
                # Resize mask back to original video dimensions
                mask = cv2.resize(p, (width, height), interpolation=cv2.INTER_LINEAR)
                mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)

                # Composite subject
                alpha = mask[..., np.newaxis]
                bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
                subject = (frame.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)

                writer_mask.write(mask_u8)
                writer_subj.write(subject)

                frame_idx += 1
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    elapsed = time.time() - t_start
                    fps_proc = frame_idx / max(elapsed, 1e-5)
                    print(f"   [{frame_idx}/{total_frames}] {fps_proc:.2f} fps ({elapsed:.1f}s elapsed)")

    elif pt_model_tuple is not None:
        model, dev = pt_model_tuple
        in_h, in_w = 1024, 1024

        while True:
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break

            tensors = []
            for frame in batch_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                norm = ((resized.astype(np.float32) / 255.0) - mean) / std
                tensors.append(torch.from_numpy(np.transpose(norm, (2, 0, 1))))

            batch_tensor = torch.stack(tensors, dim=0).to(dev)

            with torch.no_grad():
                preds = model(batch_tensor)
                scaled_preds = preds[-1].sigmoid().cpu().numpy()

            for i, frame in enumerate(batch_frames):
                p = scaled_preds[i, 0]
                mask = cv2.resize(p, (width, height), interpolation=cv2.INTER_LINEAR)
                mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)

                alpha = mask[..., np.newaxis]
                bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
                subject = (frame.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)

                writer_mask.write(mask_u8)
                writer_subj.write(subject)

                frame_idx += 1
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    elapsed = time.time() - t_start
                    fps_proc = frame_idx / max(elapsed, 1e-5)
                    print(f"   [{frame_idx}/{total_frames}] {fps_proc:.2f} fps ({elapsed:.1f}s elapsed)")

    cap.release()
    writer_mask.release()
    writer_subj.release()

    total_time = time.time() - t_start
    print(f"✅ Finished in {total_time:.2f}s ({frame_idx / max(total_time, 1e-5):.2f} avg fps)")
    print(f"   Mask saved to:    {dst_mask_path}")
    print(f"   Subject saved to: {dst_subject_path}\n")


def main():
    parser = argparse.ArgumentParser(description="BiRefNet-Turbo High-Performance Video Matting")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory of videos to process (e.g. ./videos_todo)")
    parser.add_argument("--model", type=str, default="birefnet_general_swinL_1024x576_int8.onnx",
                        help="Path to .onnx file or PyTorch model (default: birefnet_general_swinL_1024x576_int8.onnx)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"], help="Device to run inference on")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for video frame processing")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads for ONNX Runtime")
    args = parser.parse_args()

    # Collect videos
    video_paths = []
    if args.video:
        video_paths.append(args.video)
    if args.video_dir:
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_paths.extend(glob(os.path.join(args.video_dir, ext)))
        video_paths = sorted(list(set(video_paths)))

    if not video_paths:
        print("[ERROR] Please specify --video <path> or --video-dir <dir>")
        sys.exit(1)

    print(f"Found {len(video_paths)} video(s) to process.")

    # Load Model
    onnx_tuple = None
    pt_tuple = None
    if args.model.endswith(".onnx"):
        print(f"Loading ONNX Model: {args.model} on {args.device.upper()}...")
        onnx_tuple = get_onnx_session(args.model, device=args.device, num_threads=args.threads)
    else:
        print(f"Loading PyTorch Model: {args.model} on {args.device.upper()}...")
        pt_tuple = get_pytorch_model(args.model, device=args.device)

    # Process all videos
    for v_path in video_paths:
        process_video(
            video_path=v_path,
            onnx_session_tuple=onnx_tuple,
            pt_model_tuple=pt_tuple,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
