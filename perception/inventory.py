"""Inventory of the 4 test clips: file counts, decoded video frames, duration and frame timing.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.inventory

Writes results/m2_perception/inventory.json. Stops with an error if a clip folder is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from perception import stray

RAW = Path("data/raw")
OUT = Path("results/m2_perception")
CLIPS = {"26d4118b0c": "top-down", "4e103843db": "90deg", "49aff04f7a": "45deg", "e8825dea53": "45deg"}


def inventory(clip: str) -> dict:
    path = RAW / clip
    rec = stray.load(path)   # also checks: odometry frame column = 0..N-1, one depth + one confidence PNG per row
    n_video = sum(1 for _ in rec.rgb_frames(size=(64, 48)))   # tiny frames: we only count (raises if count != N - 1)
    dt = np.diff(rec.t)
    return {
        "view_label": CLIPS[clip],
        "n_odometry_rows": len(rec),
        "n_depth_png": len(list((path / "depth").glob("*.png"))),
        "n_confidence_png": len(list((path / "confidence").glob("*.png"))),
        "n_video_frames_decoded": n_video,
        "duration_s": round(float(rec.t[-1] - rec.t[0]), 4),
        "median_dt_s": round(float(np.median(dt)), 5),
        "max_dt_s": round(float(dt.max()), 5),
        "n_dt_above_1p5_median": int(np.sum(dt > 1.5 * np.median(dt))),
    }


def main():
    missing = [c for c in CLIPS if not (RAW / c / "odometry.csv").exists()]
    if missing:
        raise SystemExit(f"missing clip folders: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    inv = {c: inventory(c) for c in CLIPS}
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1))
    for c, r in inv.items():
        print(c, r)


if __name__ == "__main__":
    main()
