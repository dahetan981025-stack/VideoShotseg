"""shotseg — Primary Shot Segmentation Module.

Core: centroid similarity curve → adaptive threshold → cut points.

Usage:
    from shotseg import ShotSeg, detect_cuts
    result = ShotSeg().segment(embeddings, timestamps)
    cuts, times, info = detect_cuts(embeddings, timestamps)
"""
from shotseg.pipeline import ShotSeg
from shotseg.types import Scene, SegResult
from shotseg.detection import detect_cuts, compute_centroid_curve

__version__ = "0.1.0"
__all__ = [
    "ShotSeg", "detect_cuts", "compute_centroid_curve",
    "Scene", "SegResult",
]
