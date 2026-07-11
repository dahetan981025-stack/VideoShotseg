"""
shotseg — Valley Detection
==========================
Find natural cut points (valleys/troughs) in centroid similarity curve.

Core: centroid[i] = cos(emb[i], mean(emb[i-window:i]))
Cut when: centroid[i] < mean(curve) - k * std(curve)
→ adaptive threshold, no fixed cut values needed.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def compute_centroid_curve(embeddings: np.ndarray, window: int = 40) -> np.ndarray:
    """
    Centroid similarity curve (ONLY centroid, no adjacent).

    For each frame i, compute:
      centroid[i] = cos(emb[i], mean(emb[i-window:i]))

    Args:
        window: number of previous frames for centroid (default 40 ≈ 20s @ 2fps).
               Larger window = smoother curve = fewer cuts.

    Returns:
        centroid: (N,) array. centroid[0] = 1.0.
    """
    n = len(embeddings)
    normed = l2_normalize(embeddings)

    centroid = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        start = max(0, i - window)
        local_center = normed[start:i].mean(axis=0)
        local_center /= (np.linalg.norm(local_center) + 1e-8)
        centroid[i] = float(np.dot(normed[i], local_center))

    return centroid


# ---------------------------------------------------------------------------
# Adaptive threshold cut
# ---------------------------------------------------------------------------

def detect_valleys_adaptive(
    curve: np.ndarray,
    timestamps: np.ndarray,
    k: float = 1.8,
    min_gap_frames: int = 5,
    min_gap_sec: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find cut points using adaptive threshold: cut where curve < mean - k*std.

    Args:
        curve: centroid similarity curve (N,).
        timestamps: frame timestamps (N,).
        k: multiplier on std. Higher = fewer/stronger cuts.
        min_gap_frames: consecutive low-sim frames → single cut (frames).
        min_gap_sec: force cut if temporal gap > this (seconds).

    Returns:
        cut_frame_indices: sorted array of frame indices where cuts happen.
        cut_timestamps: sorted array of cut times (seconds).
    """
    n = len(curve)
    threshold = float(np.mean(curve) - k * float(np.std(curve)))

    # All frames below threshold
    low_mask = curve < threshold

    # Group consecutive low frames
    cuts = []
    i = 0
    while i < n:
        if low_mask[i]:
            start = i
            while i < n and low_mask[i]:
                i += 1
            # Use the first frame of each low region as cut
            cuts.append(start)
        else:
            i += 1

    # Also cut at temporal gaps > min_gap_sec
    for i in range(1, n):
        if timestamps[i] - timestamps[i - 1] > min_gap_sec:
            cuts.append(i)

    cut_frames = np.array(sorted(set(cuts)))
    cut_times = timestamps[cut_frames]

    return cut_frames, cut_times


def detect_valleys_peaks(
    curve: np.ndarray,
    timestamps: np.ndarray,
    prominence: float = 0.03,
    distance: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find cut points using scipy find_peaks on negated curve.
    Alternative to adaptive threshold; kept for comparison.

    Returns (cut_frames, cut_times).
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-curve, prominence=prominence, distance=distance)
    return peaks, timestamps[peaks]


# ---------------------------------------------------------------------------
# Shot boundary detection — main entry
# ---------------------------------------------------------------------------

def detect_cuts(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    window: int = 40,
    k: float = 1.8,
    min_gap_frames: int = 5,
    min_gap_sec: float = 3.0,
    method: str = "adaptive",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Detect shot cut points from DINOv2/CLIP frame embeddings.

    Only uses centroid similarity curve (as instructed).

    Args:
        embeddings: (N, D) frame embeddings.
        timestamps: (N,) frame timestamps (seconds).
        window: centroid window (frames). Default 40 ≈ 20s @ 2fps.
        k: adaptive threshold multiplier. Higher = fewer cuts. Default 1.8.
        min_gap_frames: group consecutive low-sim frames (frames).
        min_gap_sec: force cut at temporal gaps longer than this (seconds).
        method: "adaptive" (threshold) or "peaks" (find_peaks).

    Returns:
        cut_frames: (M,) frame indices of cuts.
        cut_times: (M,) timestamps of cuts.
        info: dict with diagnostics (threshold, curve stats).
    """
    curve = compute_centroid_curve(embeddings, window=window)

    if method == "peaks":
        cut_frames, cut_times = detect_valleys_peaks(
            curve, timestamps,
            prominence=k * 0.01,
            distance=min_gap_frames,
        )
    else:
        cut_frames, cut_times = detect_valleys_adaptive(
            curve, timestamps,
            k=k, min_gap_frames=min_gap_frames,
            min_gap_sec=min_gap_sec,
        )

    info = {
        "method": method,
        "window": window,
        "k": k,
        "threshold": float(np.mean(curve) - k * float(np.std(curve))),
        "curve_mean": float(np.mean(curve)),
        "curve_std": float(np.std(curve)),
    }

    return cut_frames, cut_times, info


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compat)
# ---------------------------------------------------------------------------

def compute_similarity_curves(embeddings, window=5):
    adj = np.ones(len(embeddings))
    cent = compute_centroid_curve(embeddings, window)
    return adj, cent


def detect_cuts_valley(embeddings, timestamps, window=40, prominence=0.01,
                       distance=5, max_gap=3.0, combine_curves="centroid"):
    """Legacy — delegates to detect_cuts with adaptive threshold."""
    cut_frames, cut_times, info = detect_cuts(
        embeddings, timestamps, window=window,
        k=prominence * 180,  # Map prominence values from tests
        min_gap_frames=distance,
        min_gap_sec=max_gap,
    )
    return cut_frames, np.array([]), []


def detect_cuts_threshold(embeddings, timestamps, sim_threshold=0.88,
                          window=5, use_centroid=False, max_gap=3.0):
    """Legacy fixed-threshold cut."""
    curve = compute_centroid_curve(embeddings, window)
    if not use_centroid:
        adj = compute_centroid_curve(embeddings, 1)  # adjacent = window=1
        curve = adj
    n = len(embeddings)
    starts = [0]
    for i in range(1, n):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > max_gap:
            starts.append(i)
        elif curve[i] < sim_threshold:
            starts.append(i)
    return np.array(starts)
