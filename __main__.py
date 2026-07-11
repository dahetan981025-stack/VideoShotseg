"""shotseg — CLI: python -m shotseg <embeddings.npz> [options]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from shotseg import ShotSeg


def main():
    p = argparse.ArgumentParser(description="shotseg — Shot Segmentation")
    p.add_argument("embeddings", help="Path to .embeddings.npz")
    p.add_argument("--output", "-o", default=None, help="Output JSON")
    p.add_argument("--show", action="store_true", help="Print scene summary")
    p.add_argument("--window", type=int, default=40, help="Centroid window (frames)")
    p.add_argument("--k", type=float, default=1.8, help="Adaptive threshold multiplier")
    p.add_argument("--min-gap", type=float, default=3.0, help="Min gap for forced cut (s)")
    p.add_argument("--scene-sim", type=float, default=0.85, help="Scene merge sim threshold")
    p.add_argument("--scene-dur", type=float, default=0.8, help="Min scene duration (s)")
    args = p.parse_args()

    if not Path(args.embeddings).exists():
        print(f"ERROR: {args.embeddings} not found", file=sys.stderr); sys.exit(1)

    data = np.load(args.embeddings)
    embeddings = data["embeddings"].astype(np.float64)
    timestamps = data["timestamps"]
    print(f"Loaded {len(embeddings)} frames, dim={embeddings.shape[1]}", flush=True)

    seg = ShotSeg()
    result = seg.segment(
        embeddings, timestamps,
        window=args.window, k=args.k, min_gap_sec=args.min_gap,
        scene_sim_threshold=args.scene_sim, scene_min_duration=args.scene_dur,
    )
    print(f"\n{result.summary()}", flush=True)

    if args.show:
        print(f"\n  {'Scn':>3s} {'Time Range':>18s} {'Dur':>7s} {'Frames':>6s}")
        print(f"  {'-'*45}")
        for sc in result.scenes:
            st = f"{int(sc.start_time//60):02d}:{int(sc.start_time%60):02d}"
            et = f"{int(sc.end_time//60):02d}:{int(sc.end_time%60):02d}"
            print(f"  {sc.scene_id:>3d} [{st}~{et}] {sc.duration:>7.1f}s {sc.n_frames:>6d}")

    if args.output:
        out = {
            "n_frames": result.n_frames, "n_shots": result.n_shots,
            "n_scenes": result.n_scenes, "total_duration": round(result.total_duration, 2),
            "scenes": [s.to_dict() for s in result.scenes],
            "boundaries": [round(b, 2) for b in result.boundaries],
            "scene_boundaries": [round(b, 2) for b in result.scene_boundaries],
            "diagnostics": result.diagnostics,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
