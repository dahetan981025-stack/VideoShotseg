"""shotseg — Data types for shot segmentation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Shot:
    """A contiguous group of frames between two cuts."""
    shot_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    frame_indices: list = None
    centroid_embedding: "np.ndarray | None" = None
    duration: float = 0.0

    def __post_init__(self):
        self.duration = self.end_time - self.start_time
        if self.frame_indices is None:
            self.frame_indices = []


@dataclass
class Scene:
    """A meaningful scene — one or more visually similar shots."""
    scene_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    shot_indices: List[int] = field(default_factory=list)
    centroid_embedding: Optional[np.ndarray] = None
    duration: float = 0.0
    n_frames: int = 0

    def __post_init__(self):
        self.duration = self.end_time - self.start_time

    def time_range(self) -> str:
        st = f"{int(self.start_time//60):02d}:{int(self.start_time%60):02d}"
        et = f"{int(self.end_time//60):02d}:{int(self.end_time%60):02d}"
        return f"[{st}~{et}]"

    def __repr__(self) -> str:
        return f"Scene({self.scene_id:03d} {self.time_range()} {self.duration:.0f}s)"

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "start_time": round(float(self.start_time), 2),
            "end_time": round(float(self.end_time), 2),
            "duration": round(float(self.duration), 2),
            "time_range": self.time_range(),
            "n_shots": len(self.shot_indices),
            "n_frames": self.n_frames,
        }


@dataclass
class SegResult:
    """Complete segmentation result."""
    n_frames: int = 0
    n_shots: int = 0
    n_scenes: int = 0
    total_duration: float = 0.0
    strategy: str = ""
    shots: list = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
    boundaries: List[float] = field(default_factory=list)
    scene_boundaries: List[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def print_scenes(self, max_lines: int = 20) -> None:
        """Print scene list with time ranges."""
        print(f"  {'Scn':>3s} {'Time Range':>18s} {'Dur':>7s} {'Frames':>6s}")
        print(f"  {'-'*45}")
        for sc in self.scenes[:max_lines]:
            print(f"  {sc.scene_id:>3d} {sc.time_range():>18s} {sc.duration:>7.1f}s {sc.n_frames:>6d}")
        if len(self.scenes) > max_lines:
            print(f"  ... ({len(self.scenes) - max_lines} more)")

    def summary(self) -> str:
        return (
            f"SegResult(strategy={self.strategy}, "
            f"{self.n_frames} frames -> {self.n_shots} shots -> {self.n_scenes} scenes, "
            f"duration={self.total_duration:.1f}s)"
        )
