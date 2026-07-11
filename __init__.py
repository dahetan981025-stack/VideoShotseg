"""shotseg — Video Shot Segmentation Tool.

基于聚类分析的视频镜头分割工具。
默认 HDBSCAN 聚类分割 (推荐), 可选谷值检测。

用法:
    from shotseg import ShotSeg
    result = ShotSeg().segment(embeddings, timestamps)
"""
from shotseg.pipeline import ShotSeg
from shotseg.types import Scene, SegResult
from shotseg.detection import detect_cuts, compute_centroid_curve
from shotseg.ffmpeg import (
    get_video_info, extract_frames, extract_scene_frames, cut_segments,
)
from shotseg.embed import extract_embeddings

__version__ = "0.2.0"
__all__ = [
    "ShotSeg", "detect_cuts", "compute_centroid_curve",
    "Scene", "SegResult",
    "get_video_info", "extract_frames", "extract_scene_frames",
    "cut_segments", "extract_embeddings",
]
