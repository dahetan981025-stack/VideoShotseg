"""shotseg — HDBSCAN clustering (optional, for Path B)."""
from __future__ import annotations
import numpy as np

def add_temporal_encoding(embeddings, timestamps, weight=0.3):
    t_norm = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0] + 1e-8)
    t_feat = t_norm[:, None] * weight * float(np.std(embeddings))
    return np.concatenate([embeddings, t_feat], axis=1)

def reduce_umap(features, n_components=16, random_state=42):
    import umap
    return umap.UMAP(n_components=n_components, n_neighbors=15,
                     min_dist=0.1, metric="cosine",
                     random_state=random_state, verbose=False).fit_transform(features)

def cluster_hdbscan(reduced, min_cluster_size=4, min_samples=3, epsilon=0.5):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           metric="euclidean", cluster_selection_epsilon=epsilon).fit_predict(reduced)

def enforce_temporal_continuity(labels, timestamps, max_gap_sec=2.5):
    new_labels = labels.copy()
    next_new_id = int(labels.max()) + 1 if labels.max() >= 0 else 0
    for old_id in sorted(set(int(l) for l in labels)):
        if old_id == -1: continue
        indices = np.where(labels == old_id)[0]
        if len(indices) == 0: continue
        indices.sort()
        gaps = np.diff(timestamps[indices])
        split_points = np.where(gaps > max_gap_sec)[0] + 1
        if len(split_points) > 0:
            segments = np.split(indices, split_points)
            for seg in segments[1:]:
                if len(seg) > 0: new_labels[seg] = next_new_id; next_new_id += 1
    return new_labels

def merge_tiny_neighbors(labels, timestamps, min_duration_sec=0.8):
    new_labels = labels.copy()
    unique_ids = sorted(set(int(l) for l in labels) - {-1})
    for cid in unique_ids:
        indices = np.where(new_labels == cid)[0]
        if len(indices) == 0: continue
        duration = timestamps[indices[-1]] - timestamps[indices[0]]
        if duration >= min_duration_sec: continue
        first_idx = int(indices[0])
        if first_idx > 0:
            prev_idx = first_idx - 1
            while prev_idx >= 0 and new_labels[prev_idx] == -1: prev_idx -= 1
            if prev_idx >= 0 and new_labels[prev_idx] != cid and new_labels[prev_idx] != -1:
                new_labels[indices] = new_labels[prev_idx]; continue
        last_idx = int(indices[-1])
        if last_idx < len(new_labels) - 1:
            next_idx = last_idx + 1
            while next_idx < len(new_labels) and new_labels[next_idx] == -1: next_idx += 1
            if next_idx < len(new_labels) and new_labels[next_idx] != cid and new_labels[next_idx] != -1:
                new_labels[indices] = new_labels[next_idx]
    return relabel_sequential(new_labels)

def relabel_sequential(labels):
    unique = sorted(set(int(l) for l in labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique)}; mapping[-1] = -1
    return np.array([mapping[int(l)] for l in labels])

def run_clustering(embeddings, timestamps, time_weight=0.3, min_cluster_size=4,
                   min_samples=3, max_gap_sec=2.5, min_duration_sec=0.8, epsilon=0.5):
    features = add_temporal_encoding(embeddings, timestamps, weight=time_weight)
    reduced = reduce_umap(features)
    labels = cluster_hdbscan(reduced, min_cluster_size, min_samples, epsilon)
    labels = enforce_temporal_continuity(labels, timestamps, max_gap_sec)
    labels = merge_tiny_neighbors(labels, timestamps, min_duration_sec)
    return relabel_sequential(labels)
