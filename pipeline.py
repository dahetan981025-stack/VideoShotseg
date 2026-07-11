"""shotseg — Shot segmentation pipeline.

Primary pipeline (V10 optimal):
  embeddings + timestamps
    → temporal encoding (+ normalized timestamp feature)
    → HDBSCAN clustering (min_cluster_size=4, auto cluster count)
    → density merge (centroid distance vs within-cluster spread × 1.2)
    → scenes

Fallback: valley detection on centroid similarity curve (no HDBSCAN dependency).

Usage:
    from shotseg import ShotSeg
    result = ShotSeg().segment(embeddings, timestamps)
    print(result.scene_boundaries)
"""
from __future__ import annotations
import numpy as np
from shotseg.types import Scene, SegResult
from shotseg.clustering import run_clustering as _hdbscan
from shotseg.merge import merge_shots, shots_from_labels
from shotseg.detection import detect_cuts as _valley_cuts
from shotseg.merge import shots_from_cut_indices


class ShotSeg:
    """Shot segmentation — HDBSCAN + density merge (primary path).

    Falls back to centroid valley detection when hdbscan/umap unavailable.
    """

    def __init__(self, method: str = "hdbscan"):
        assert method in ("hdbscan", "valley"), f"Unknown method: {method}"
        self.method = method

    def segment(
        self,
        embeddings: np.ndarray,
        timestamps: np.ndarray,
        # HDBSCAN params
        min_cluster_size: int = 4,
        min_samples: int = 3,
        time_weight: float = 0.3,
        max_gap_sec: float = 2.5,
        min_duration_sec: float = 0.8,
        # Density merge params
        spread_factor: float = 1.2,
        merge_max_gap: float = 3.0,
        # Valley fallback params
        window: int = 40,
        k: float = 1.8,
    ) -> SegResult:
        """Run segmentation.

        Args:
            embeddings: (N, D) frame embeddings (DINOv2 ViT-L/14, 1024D).
            timestamps: (N,) frame timestamps (seconds) @ 2fps.

            min_cluster_size: HDBSCAN min cluster (frames). Default 4.
            min_samples: HDBSCAN min samples.
            time_weight: temporal feature weight relative to embedding std.
            max_gap_sec: split cluster if frames gap > this.
            min_duration_sec: merge clusters shorter than this.

            spread_factor: density merge threshold multiplier. 1.2 = optimal.
            merge_max_gap: max gap to consider merge (seconds).

            window: centroid curve window (frames, valley fallback only).
            k: adaptive threshold multiplier (valley fallback only).

        Returns:
            SegResult.
        """
        total_dur = float(timestamps[-1] - timestamps[0])

        if self.method == "hdbscan":
            return self._hdbscan_path(
                embeddings, timestamps, total_dur,
                min_cluster_size, min_samples, time_weight,
                max_gap_sec, min_duration_sec,
                spread_factor, merge_max_gap,
            )
        else:
            return self._valley_path(
                embeddings, timestamps, total_dur,
                window, k, merge_max_gap, spread_factor,
            )

    def _hdbscan_path(self, emb, ts, total_dur,
                      mcs, ms, tw, mgs, mds, sf, mmg):
        labels = _hdbscan(
            emb, ts,
            min_cluster_size=mcs,
            min_samples=ms,
            time_weight=tw,
            max_gap_sec=mgs,
            min_duration_sec=mds,
        )
        n_clusters = len(set(int(l) for l in labels if l >= 0))
        shots = shots_from_labels(labels, ts, emb)
        scenes = merge_shots(
            shots,
            spread_factor=sf,
            max_gap=mmg,
            min_duration=mds,
        )

        return SegResult(
            n_frames=len(emb), n_shots=len(shots), n_scenes=len(scenes),
            total_duration=total_dur, strategy=f"hdbscan_mcs={mcs}",
            shots=shots, scenes=scenes,
            boundaries=[s.start_time for s in shots if s.start_time > 0.5],
            scene_boundaries=[s.start_time for s in scenes if s.start_time > 0.5],
            diagnostics={"n_clusters": n_clusters, "method": "hdbscan"},
        )

    def _valley_path(self, emb, ts, total_dur, window, k, mmg, sf):
        cut_frames, cut_times, info = _valley_cuts(
            emb, ts, window=window, k=k,
        )
        shots = shots_from_cut_indices(cut_frames, ts, emb)
        scenes = merge_shots(
            shots,
            spread_factor=sf,
            max_gap=mmg,
            min_duration=0.8,
        )

        return SegResult(
            n_frames=len(emb), n_shots=len(shots), n_scenes=len(scenes),
            total_duration=total_dur, strategy="valley",
            shots=shots, scenes=scenes,
            boundaries=[s.start_time for s in shots if s.start_time > 0.5],
            scene_boundaries=[s.start_time for s in scenes if s.start_time > 0.5],
            diagnostics=info,
        )
