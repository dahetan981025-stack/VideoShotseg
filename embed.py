"""shotseg — DINOv2 视频帧 Embedding 提取 (GPU-ONLY).

需要一个 CUDA GPU 和 PyTorch. 纯 ffmpeg 操作请用 shotseg.ffmpeg.
"""
from __future__ import annotations
import subprocess, logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from shotseg.ffmpeg import get_video_info, _cuda_decoder

logger = logging.getLogger("shotseg.embed")

FFMPEG_BIN = "ffmpeg"
DEFAULT_FPS = 2.0
BATCH_SIZE = 32


def extract_embeddings(
    video_path: str,
    fps: float = DEFAULT_FPS,
    cache: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract DINOv2 ViT-L/14 embeddings from video (GPU-ONLY).

    Args:
        video_path: Path to video file.
        fps: Target sample rate.
        cache: Save/load .embeddings.npz cache file.

    Returns:
        (embeddings, timestamps) as float32 arrays.
        Returns (None, None) if PyTorch/CUDA unavailable.

    Dependencies: torch, torchvision, opencv-python, DINOv2 via torch.hub.
    """
    cache_path = Path(video_path).with_suffix(".embeddings.npz")
    if cache and cache_path.exists():
        logger.info(f"Loading cached: {cache_path}")
        data = np.load(cache_path)
        return data["embeddings"].astype(np.float32), data["timestamps"].astype(np.float32)

    try:
        import torch
        from torchvision import transforms
        import cv2
    except ImportError:
        logger.error("PyTorch / torchvision / opencv-python not installed. "
                     "Run: pip install torch torchvision opencv-python")
        return None, None

    if not torch.cuda.is_available():
        logger.error("GPU_ONLY: CUDA not available.")
        return None, None

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda")
    logger.info("Loading DINOv2 ViT-L/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(device).eval()
    logger.info(f"Model on {next(model.parameters()).device}")

    info = get_video_info(video_path)
    native_fps = info["fps"]
    frame_interval = max(1, round(native_fps / fps))

    cuvid_dec = _cuda_decoder(info["codec"])
    if not cuvid_dec:
        logger.error(f'GPU_ONLY: no CUDA decoder for codec "{info["codec"]}"')
        return None, None

    logger.info(f"Video: {info['total_frames']} frames @ {info['fps']:.1f}fps "
                f"{info['width']}x{info['height']} [{info['codec']}]")

    cmd = [FFMPEG_BIN, "-c:v", cuvid_dec, "-hwaccel", "cuda",
           "-i", video_path,
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
           "-loglevel", "error"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=8 * 1024 * 1024)

    frame_bytes = info["width"] * info["height"] * 3 // 2
    frame_batch, ts_batch = [], []
    embs_list, ts_list = [], []
    idx, sampled = 0, 0
    vid_w, vid_h = info["width"], info["height"]

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break

        if idx % frame_interval == 0:
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(vid_h * 3 // 2, vid_w)
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            frame_batch.append(frame)
            ts_batch.append(idx / native_fps)
            sampled += 1

            if len(frame_batch) >= BATCH_SIZE:
                _infer_batch(preprocess, model, device, frame_batch, ts_batch,
                             embs_list, ts_list, sampled)
        idx += 1

    proc.stdout.close()
    proc.stderr.close()
    proc.wait()

    if frame_batch:
        _infer_batch(preprocess, model, device, frame_batch, ts_batch,
                     embs_list, ts_list, sampled)

    if len(ts_list) < 2:
        logger.warning(f"Too few frames: {len(ts_list)}")
        return None, None

    embeddings = np.concatenate(embs_list).astype(np.float32)
    timestamps = np.array(ts_list, dtype=np.float32)

    if cache:
        np.savez(cache_path, embeddings=embeddings, timestamps=timestamps)
        logger.info(f"Cached: {cache_path}")

    logger.info(f"Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    return embeddings, timestamps


def _infer_batch(preprocess, model, device, frame_batch, ts_batch,
                 embs_list, ts_list, sampled):
    """Run DINOv2 inference on one batch."""
    import torch
    tensors = [preprocess(f) for f in frame_batch]
    with torch.no_grad():
        inp = torch.stack(tensors).to(device)
        e = model(inp)
    embs_list.append(e.cpu().numpy())
    ts_list.extend(ts_batch)
    frame_batch.clear()
    ts_batch.clear()
    if sampled % 320 == 0:
        logger.info(f"  ... {sampled} frames")
