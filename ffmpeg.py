"""shotseg — FFmpeg 工具集: 视频信息 / 抽帧 / 场景帧 / 视频切割.

GPU 版本: 默认 CUDA 硬件加速解码 (NVDEC), 无 GPU 时自动降级 CPU。
"""
from __future__ import annotations
import json, subprocess, os, glob, logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger("shotseg.ffmpeg")
FFMPEG_BIN = "ffmpeg"


# ── Video info ──────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    """Return {fps, total_frames, codec, width, height, duration}."""
    result = subprocess.run(
        [FFMPEG_BIN, "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,nb_frames,duration,codec_name,width,height",
         "-of", "json", video_path],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]

    fps_str = stream.get("r_frame_rate", "25/1")
    fps = float(fps_str.split("/")[0]) / float(fps_str.split("/")[1]) if "/" in fps_str else float(fps_str)

    nb = stream.get("nb_frames", "0")
    total_frames = int(nb) if nb and nb not in ("N/A", "0") else 0
    if total_frames == 0:
        dur = float(stream.get("duration", 0))
        total_frames = int(dur * fps) if dur > 0 else 0

    return {
        "fps": fps,
        "total_frames": total_frames,
        "codec": stream.get("codec_name", "unknown"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0)),
    }


def _cuda_decoder(codec_name: str) -> Optional[str]:
    """返回对应编码的 CUVID decoder 名, None 表示不支持."""
    return {
        "h264": "h264_cuvid", "hevc": "hevc_cuvid", "av1": "av1_cuvid",
        "vp9": "vp9_cuvid", "vp8": "vp8_cuvid", "mpeg4": "mpeg4_cuvid",
        "mjpeg": "mjpeg_cuvid",
    }.get(codec_name.lower())


def _hwaccel_args(video_path: str) -> list[str]:
    """返回 CUDA hwaccel args (自动检测 decoder), 无 GPU 则空列表."""
    try:
        info = get_video_info(video_path)
        dec = _cuda_decoder(info["codec"])
        if dec:
            return ["-c:v", dec, "-hwaccel", "cuda"]
    except Exception:
        pass
    logger.warning("CUDA decoder not available, falling back to CPU")
    return []


# ── Frame extraction (PNG) ──────────────────────────────────────────

def extract_frames(
    video_path: str,
    out_dir: str = "/tmp/shotseg_frames",
    fps: float = 2.0,
    scale: float = 0.5,
    scene_start: Optional[float] = None,
    scene_end: Optional[float] = None,
    use_gpu: bool = True,
) -> list[str]:
    """Extract frames from video at given FPS (默认 CUDA GPU 加速).

    Args:
        video_path: 视频路径。
        out_dir: 输出 PNG 目录。
        fps: 目标帧率。
        scale: 缩放比例 (0.5 = 半分辨率)。
        scene_start: 起始时间 (秒)。
        scene_end: 结束时间 (秒)。
        use_gpu: 启用 CUDA GPU 解码。

    Returns:
        PNG 文件路径列表。
    """
    os.makedirs(out_dir, exist_ok=True)
    info = get_video_info(video_path)
    native_fps = info["fps"]
    frame_interval = max(1, round(native_fps / fps))

    cmd = [FFMPEG_BIN]
    if use_gpu:
        cmd += _hwaccel_args(video_path)
    cmd += ["-i", video_path]

    if scene_start is not None and scene_start > 0:
        cmd += ["-ss", str(scene_start)]

    select_expr = f"not(mod(n\\,{frame_interval}))"
    vf_parts = [f"select='{select_expr}'", f"scale=iw*{scale}:ih*{scale}"]
    cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-vsync", "0"]

    if scene_end is not None:
        cmd += ["-to", str(scene_end)]

    cmd += ["-frame_pts", "1", f"{out_dir}/%010d.png", "-loglevel", "error"]

    logger.info(f"Extracting frames @ {fps}fps from {Path(video_path).name}")
    subprocess.run(cmd, check=True, timeout=600)

    frames = sorted(glob.glob(f"{out_dir}/*.png"),
                    key=lambda p: int(os.path.basename(p).split(".")[0]))
    logger.info(f"  {len(frames)} frames -> {out_dir}")
    return frames


# ── Scene-based keyframe extraction ─────────────────────────────────

def _fmt_ts(s: float) -> str:
    m, s_ = divmod(int(s), 60)
    return f"{m:02d}:{s_:02d}"


def extract_scene_frames(
    video_path: str,
    scenes: list,
    out_dir: str = "/tmp/scene_frames",
    frames_per_scene: int = 5,
    fps: float = 24.0,
    use_gpu: bool = True,
) -> list[dict]:
    """Extract N keyframes per scene (默认 CUDA GPU 加速).

    Args:
        video_path: 视频路径。
        scenes: [{start, end}, ...] 场景列表。
        out_dir: 输出目录。
        frames_per_scene: 每场景帧数。
        fps: 用于帧号计算的帧率。
        use_gpu: 启用 CUDA GPU 解码。

    Returns:
        [{scene_id, start, end, frames: [path, ...]}, ...]
    """
    os.makedirs(out_dir, exist_ok=True)

    all_entries = []
    for sid, sc in enumerate(scenes):
        dur = sc["end"] - sc["start"]
        if dur < 1.5:
            continue
        if dur < 2.0:
            ts_list = [(sc["start"] + sc["end"]) / 2]
        else:
            ts_list = [
                sc["start"] + dur * 0.05,
                sc["start"] + dur * 0.25,
                (sc["start"] + sc["end"]) / 2,
                sc["end"] - dur * 0.25,
                sc["end"] - dur * 0.05,
            ]
        for fi, t in enumerate(ts_list[:frames_per_scene], 1):
            all_entries.append((sid, fi, t))

    if not all_entries:
        return []

    frame_queries = []
    seen_frames = {}
    for sid, fi, t in all_entries:
        fn = round(t * fps)
        if fn not in seen_frames:
            seen_frames[fn] = len(frame_queries)
            frame_queries.append((sid, fi, fn))

    logger.info(f"Extracting {len(frame_queries)} frames for {len(scenes)} scenes")

    select_parts = ["eq(n\\,%d)" % fn for _, _, fn in frame_queries]
    select_expr = "+".join(select_parts)
    out_pat = out_dir + "/%d.png"

    hwaccel_args = _hwaccel_args(video_path) if use_gpu else []
    hw_str = " ".join(hwaccel_args) + " " if hwaccel_args else ""
    shell_cmd = (
        f"ffmpeg {hw_str}-i {video_path} "
        f"-vf 'select={select_expr},scale=iw/2:ih/2' -vsync 0 {out_pat} -loglevel error"
    )
    r = subprocess.run(shell_cmd, shell=True, capture_output=True, timeout=300)
    if r.returncode != 0:
        logger.error(f"ffmpeg error: {r.stderr.decode()[:300]}")
        return []

    raw_files = sorted(glob.glob(f"{out_dir}/*.png"),
                       key=lambda p: int(os.path.basename(p).split(".")[0]))

    fm_file_idx = {}
    for i, (_, _, fn) in enumerate(frame_queries):
        if i < len(raw_files):
            fm_file_idx[fn] = raw_files[i]

    for sid, fi, t in all_entries:
        fn = round(t * fps)
        if fn not in fm_file_idx:
            continue
        src = fm_file_idx[fn]
        sc = scenes[sid]
        prefix = f"[{_fmt_ts(sc['start'])}-{_fmt_ts(sc['end'])}]"
        dst = os.path.join(out_dir, f"{prefix}_{fi:02d}.png")
        if not os.path.exists(dst):
            os.rename(src, dst)

    for f in glob.glob(f"{out_dir}/*.png"):
        if "[" not in os.path.basename(f):
            os.remove(f)

    result = []
    for sid, sc in enumerate(scenes):
        prefix = f"[{_fmt_ts(sc['start'])}-{_fmt_ts(sc['end'])}]"
        scene_frames = sorted(glob.glob(f"{out_dir}/{prefix}_*.png"))
        result.append({
            "scene_id": sid,
            "start": sc["start"],
            "end": sc["end"],
            "frames": scene_frames,
        })
    return result


# ── Video cutting (stream copy, GPU 仅加速 seek 阶段) ────────────────

def cut_segments(
    video_path: str,
    boundaries: list[float],
    out_dir: str = "segments",
    prefix: str = "scene",
    scene_ids: list[int] | None = None,
    use_gpu: bool = True,
) -> list[dict]:
    """按边界切分视频为独立片段 (ffmpeg -c copy, 无损).

    流复制模式不重新编码, GPU 仅用于加速 seek。
    输出命名: scene_001_00m00s-00m12s.mp4

    Args:
        video_path: 源视频。
        boundaries: 切分时间戳列表 (秒)。
        out_dir: 输出目录。
        prefix: 文件名前缀。
        scene_ids: 自定义场景 ID。
        use_gpu: 启用 CUDA GPU 加速 seek。

    Returns:
        [{scene_id, start, end, duration, path}, ...]
    """
    import subprocess as _sp

    if scene_ids is not None and len(scene_ids) != len(boundaries) - 1:
        raise ValueError("scene_ids length must equal len(boundaries) - 1")

    os.makedirs(out_dir, exist_ok=True)

    info = get_video_info(video_path)
    total_dur = info["duration"]

    b = list(boundaries)
    if b[0] > 0.5:
        b = [0.0] + b
    if b[-1] < total_dur - 0.5:
        b = b + [total_dur]

    hwaccel = _hwaccel_args(video_path) if use_gpu else []

    segments = []
    for i in range(len(b) - 1):
        start = b[i]
        end = b[i + 1]
        dur = end - start
        if dur < 0.5:
            continue

        sid = scene_ids[i] if scene_ids else i + 1

        def _fmt(s):
            m, s_ = divmod(int(s), 60)
            return f"{m:02d}m{s_:02d}s"

        fname = f"{prefix}_{sid:03d}_{_fmt(start)}-{_fmt(end)}.mp4"
        out_path = os.path.join(out_dir, fname)

        cmd = [FFMPEG_BIN]
        if hwaccel:
            cmd += hwaccel
        cmd += [
            "-ss", str(start),
            "-i", video_path,
            "-to", str(end),
            "-c", "copy",
            "-copyts",
            "-avoid_negative_ts", "make_zero",
            "-y",
            out_path,
            "-loglevel", "error",
        ]
        _sp.run(cmd, check=True, timeout=600)

        segments.append({
            "scene_id": sid,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(dur, 2),
            "path": out_path,
        })
        logger.info(
            f"  [{sid:03d}] {_fmt(start)}~{_fmt(end)} ({dur:.1f}s) -> {fname}"
        )

    logger.info(f"Cut {len(segments)} segments to {out_dir}/")
    return segments
