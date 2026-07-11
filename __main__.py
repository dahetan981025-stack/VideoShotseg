"""shotseg — CLI: 视频 → 镜头分割 → 视频切割.

标准输出: scene_boundaries (时间戳列表), 可对接任意切割工具。

Usage:
    python -m shotseg video.mp4 --show
    python -m shotseg video.mp4 --cut segments/
    python -m shotseg embeddings.npz --show
    python -m shotseg video.mp4 --extract-frames /tmp/frames
    python -m shotseg video.mp4 --method valley --show
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mts", ".m2ts", ".ts"}


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def main():
    p = argparse.ArgumentParser(
        description="VideoShotseg — 基于聚类分析的视频镜头分割与切割"
    )
    p.add_argument("input", help="视频文件或 .embeddings.npz")
    p.add_argument("--output", "-o", default=None, help="输出结果 JSON")
    p.add_argument("--show", action="store_true", help="打印场景摘要")
    p.add_argument("--method", choices=("hdbscan", "valley"), default="hdbscan",
                   help="分割方法: hdbscan (推荐) / valley (附带)")

    # Video cutting
    p.add_argument("--cut", default=None, dest="cut_dir", metavar="DIR",
                   help="按场景切分视频到目录")
    p.add_argument("--cut-prefix", default="scene", help="切割文件名前缀")

    # Frame extraction only
    p.add_argument("--extract-frames", default=None, metavar="DIR",
                   help="仅抽帧到目录")
    p.add_argument("--fps", type=float, default=2.0, help="抽帧帧率")

    # HDBSCAN params
    p.add_argument("--min-cluster-size", type=int, default=4)
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--time-weight", type=float, default=0.3)
    p.add_argument("--max-gap", type=float, default=2.5)
    p.add_argument("--min-duration", type=float, default=0.8)
    p.add_argument("--spread-factor", type=float, default=1.2)
    p.add_argument("--merge-max-gap", type=float, default=3.0)

    # Valley params
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--k", type=float, default=1.8)

    args = p.parse_args()

    # ── 输入加载 ──
    if is_video(args.input):
        if not Path(args.input).exists():
            print(f"ERROR: 视频不存在: {args.input}", file=sys.stderr)
            sys.exit(1)

        if args.extract_frames:
            from shotseg.ffmpeg import extract_frames
            frames = extract_frames(args.input, args.extract_frames, fps=args.fps)
            print(f"已提取 {len(frames)} 帧到 {args.extract_frames}")
            return

        print(f"正在处理视频: {args.input}")
        from shotseg.embed import extract_embeddings
        embeddings, timestamps = extract_embeddings(args.input, fps=args.fps)
        if embeddings is None:
            print("ERROR: 需 PyTorch + CUDA + DINOv2 提取 embedding", file=sys.stderr)
            sys.exit(1)
        print(f"已提取 {len(embeddings)} 帧, dim={embeddings.shape[1]}")

    else:
        if not Path(args.input).exists():
            print(f"ERROR: 文件不存在: {args.input}", file=sys.stderr)
            sys.exit(1)
        data = np.load(args.input)
        embeddings = data["embeddings"].astype(np.float64)
        timestamps = data["timestamps"]
        print(f"已加载 {len(embeddings)} 帧, dim={embeddings.shape[1]}")

    # ── 切点检测 (标准输出: scene_boundaries) ──
    from shotseg import ShotSeg
    seg = ShotSeg(method=args.method)
    result = seg.segment(
        embeddings, timestamps,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        time_weight=args.time_weight,
        max_gap_sec=args.max_gap,
        min_duration_sec=args.min_duration,
        spread_factor=args.spread_factor,
        merge_max_gap=args.merge_max_gap,
        window=args.window,
        k=args.k,
    )
    print(f"\n{result.summary()}")

    # ── 打印场景表 ──
    if args.show or args.cut_dir:
        print(f"\n  {'Scn':>3s} {'Time Range':>18s} {'Dur':>7s} {'Frames':>6s}")
        print(f"  {'-'*45}")
        for sc in result.scenes:
            st = f"{int(sc.start_time//60):02d}:{int(sc.start_time%60):02d}"
            et = f"{int(sc.end_time//60):02d}:{int(sc.end_time%60):02d}"
            print(f"  {sc.scene_id:>3d} [{st}~{et}] {sc.duration:>7.1f}s {sc.n_frames:>6d}")

    # ── 视频切割 ──
    if args.cut_dir and is_video(args.input):
        from shotseg.ffmpeg import cut_segments
        segs = cut_segments(args.input, result.scene_boundaries,
                            out_dir=args.cut_dir, prefix=args.cut_prefix)
        print(f"\n已切割 {len(segs)} 个片段到 {args.cut_dir}/")

    # ── 保存结果 ──
    if args.output:
        out = {
            "n_frames": result.n_frames, "n_shots": result.n_shots,
            "n_scenes": result.n_scenes,
            "total_duration": round(result.total_duration, 2),
            "scenes": [s.to_dict() for s in result.scenes],
            "scene_boundaries": [round(b, 2) for b in result.scene_boundaries],
            "diagnostics": result.diagnostics,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
