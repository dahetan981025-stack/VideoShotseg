"""shotseg — Shot merge into scenes (V10 density merge)."""
from __future__ import annotations
import numpy as np
from typing import List, Optional
from shotseg.types import Shot, Scene


def merge_shots(
    shots: List[Shot],
    spread_factor: float = 1.2,
    max_gap: float = 3.0,
    min_duration: float = 0.8,
) -> List[Scene]:
    """V10 density merge: iterative until convergence.

    Two adjacent shots merged if:
      centroid_distance(a, b) < avg_spread(a, b) * spread_factor
    AND temporal gap <= max_gap.

    Uses each shot's _spread attribute (computed as mean L2 from centroid
    of actual frame embeddings in shots_from_labels).
    """
    if not shots:
        return []

    groups = [[s] for s in shots]  # each group = list of Shots

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups) - 1:
            a_shots, b_shots = groups[i], groups[i + 1]

            gap = b_shots[0].start_time - a_shots[-1].end_time
            if gap > max_gap:
                i += 1
                continue

            a_emb = _group_centroid(a_shots)
            b_emb = _group_centroid(b_shots)

            if a_emb is not None and b_emb is not None:
                dist = float(np.linalg.norm(a_emb - b_emb))
                a_spread = getattr(a_shots[-1], "_spread", 0.1)
                b_spread = getattr(b_shots[0], "_spread", 0.1)
                avg_spread = (a_spread + b_spread) / 2 + 1e-8

                if dist < avg_spread * spread_factor:
                    groups[i] = a_shots + b_shots
                    groups.pop(i + 1)
                    changed = True
                    continue

            i += 1

    # Convert groups to scenes
    scenes = []
    for g in groups:
        st = g[0].start_time
        et = g[-1].end_time
        dur = et - st

        if dur < min_duration and scenes:
            prev = scenes[-1]
            prev.shot_indices.extend([s.shot_id for s in g])
            prev.end_time = et
            prev.n_frames += sum(len(s.frame_indices) for s in g)
            prev.duration = prev.end_time - prev.start_time
            continue

        valid = [s.centroid_embedding for s in g if s.centroid_embedding is not None]
        centroid = np.mean(valid, axis=0) if valid else None
        n_frames = sum(len(s.frame_indices) for s in g)

        scenes.append(Scene(
            scene_id=len(scenes),
            start_time=st, end_time=et,
            shot_indices=[s.shot_id for s in g],
            centroid_embedding=centroid,
            n_frames=n_frames,
        ))

    return scenes


def _group_centroid(shots: List[Shot]) -> Optional[np.ndarray]:
    """Weighted centroid of shot group by n_frames."""
    embs, weights = [], []
    for s in shots:
        if s.centroid_embedding is not None:
            embs.append(s.centroid_embedding)
            weights.append(len(s.frame_indices))
    if not embs:
        return None
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.average(embs, axis=0, weights=weights)


def shots_from_labels(
    labels: np.ndarray, timestamps: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> List[Shot]:
    """Convert HDBSCAN cluster labels to Shot objects with computed spread.

    For each cluster: compute centroid embedding and mean-L2 spread.
    Splits non-contiguous frames within same label into separate shots.
    """
    unique = sorted(set(int(l) for l in labels))
    shots = []
    for lid in unique:
        idx = np.where(labels == lid)[0]
        if len(idx) == 0:
            continue

        # Split non-contiguous frame groups within same label
        gaps = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, gaps)

        for g in groups:
            if len(g) == 0:
                continue
            st = float(timestamps[g[0]])
            et = float(timestamps[g[-1]])

            if embeddings is not None:
                frame_embs = embeddings[g]
                centroid = frame_embs.mean(axis=0)
                spreads = np.linalg.norm(frame_embs - centroid, axis=1)
                spread = float(spreads.mean()) if len(spreads) > 0 else 0.1
            else:
                centroid = None
                spread = 0.1

            shot = Shot(
                shot_id=len(shots),
                start_time=st, end_time=et,
                frame_indices=g.tolist(),
                centroid_embedding=centroid,
            )
            shot._spread = spread
            shots.append(shot)
    shots.sort(key=lambda s: s.start_time)
    return shots


def shots_from_cut_indices(
    cut_indices: np.ndarray, timestamps: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> List[Shot]:
    """Convert cut-point frame indices to Shot objects (valley path)."""
    n = len(timestamps)
    shots = []
    for i in range(len(cut_indices)):
        si = int(cut_indices[i])
        ei = int(cut_indices[i + 1]) if i + 1 < len(cut_indices) else n
        if si >= ei:
            continue
        st, et = float(timestamps[si]), float(timestamps[ei - 1])

        if embeddings is not None:
            frame_embs = embeddings[si:ei]
            centroid = frame_embs.mean(axis=0)
            spreads = np.linalg.norm(frame_embs - centroid, axis=1)
            spread = float(spreads.mean()) if len(spreads) > 0 else 0.1
        else:
            centroid = None
            spread = 0.1

        shot = Shot(
            shot_id=len(shots),
            start_time=st, end_time=et,
            frame_indices=list(range(si, ei)),
            centroid_embedding=centroid,
        )
        shot._spread = spread
        shots.append(shot)
    return shots
