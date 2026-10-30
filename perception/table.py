"""S4 -- the table plane in the ARKit world frame.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.table --clip <id>

Writes data/work/<clip>/table/table.json: the plane [nx, ny, nz, d] (n.p + d = 0, n.(0, 1, 0) > 0), its windows and fit
statistics, and the end-window plane for the drift check.

Initial window: i = 1 .. (first hand_valid frame - 15). The hand is not in view yet, so only the mug and the coaster hide
the table; their SAM 2 masks are cut out, dilated 3 px at depth resolution because the mask edge and the LiDAR's blurred
depth edges do not line up exactly. Points: high-confidence (== 2) LiDAR depth, back-projected with K_depth, moved to
world with T_world_cam and pooled over the window (the world frame is fixed, so pooling averages the LiDAR noise).
Fit: RANSAC plane (1 cm, 1000 iterations, seed 0), least-squares refit on the RANSAC inliers, normal turned to world +y
(ARKit world is y up). This initial-window plane is the one used downstream.

Drift check: the same fit on the end window, (last hand_valid frame + 15) .. N-1. If ARKit's world frame moved during the
clip, the table moves in world coordinates and the two planes differ. The offset at the initial mug position is computed
in S7 (table_frame.py), which knows that position. An end window shorter than 10 frames -> N/A (null).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from perception import stray
from perception.common import RAW, WORK, stage_cli, stage_dir
from perception.geometry import transform
from perception.masks_sam2 import BODY, COASTER, HANDLE, load_masks

HAND_MARGIN = 15      # frames kept between a window and the first / last hand detection
MIN_FRAMES = 10       # shorter initial window -> flagged; shorter end window -> drift N/A
MASK_DILATE_PX = 3
RANSAC_TOL = 0.01     # m
RANSAC_ITERS = 1000
SEED = 0


def window_points(rec, masks, K_depth, T_world_cam, frames) -> np.ndarray:
    """(M, 3) world points pooled over `frames`: confidence 2, depth > 0, outside the dilated object masks."""
    kernel = np.ones((2 * MASK_DILATE_PX + 1, 2 * MASK_DILATE_PX + 1), np.uint8)
    pts = []
    for i in frames:
        depth, conf = rec.depth(i), rec.confidence(i)
        objects = masks.get(BODY, i, "depth") | masks.get(HANDLE, i, "depth") | masks.get(COASTER, i, "depth")
        keep = (conf == 2) & (depth > 0) & (cv2.dilate(objects.astype(np.uint8), kernel) == 0)
        pts.append(transform(T_world_cam[i], stray.backproject(depth, K_depth[i])[keep]))
    return np.concatenate(pts)


def fit_plane(pts: np.ndarray, rng) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane, then a least-squares refit on its inliers. Returns (n, d, inlier mask), n.(0, 1, 0) > 0."""
    best_count, best_n, best_d = -1, None, None
    for _ in range(RANSAC_ITERS):
        a, b, c = pts[rng.integers(len(pts), size=3)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:            # repeated or collinear sample
            continue
        n /= np.linalg.norm(n)
        count = np.count_nonzero(np.abs(pts @ n - n @ a) < RANSAC_TOL)
        if count > best_count:
            best_count, best_n, best_d = count, n, -n @ a
    inliers = np.abs(pts @ best_n + best_d) < RANSAC_TOL
    centre = pts[inliers].mean(0)
    n = np.linalg.eigh(np.cov(pts[inliers].T))[1][:, 0]     # smallest-variance direction = normal
    if n[1] < 0:
        n = -n
    return n, float(-n @ centre), inliers


def fit_window(rec, masks, meta, frames) -> tuple[np.ndarray, float, dict]:
    pts = window_points(rec, masks, meta["K_depth"], meta["T_world_cam"], frames)
    n, d, inliers = fit_plane(pts, np.random.default_rng(SEED))
    dist = pts[inliers] @ n + d
    return n, d, {"n_points": len(pts), "inlier_frac": float(inliers.mean()), "rms_m": float(np.sqrt(np.mean(dist ** 2))),
                  "normal_vs_y_deg": float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))}


def run(clip: str) -> dict:
    out_dir = stage_dir(clip, "table")
    meta = np.load(WORK / clip / "frames" / "meta.npz")
    hand_valid = np.load(WORK / clip / "hand" / "hand.npz")["hand_valid"]
    rec = stray.load(RAW / clip)
    masks = load_masks(clip)
    n_frames = len(meta["t"])

    detected = np.flatnonzero(hand_valid)
    if len(detected) == 0:
        raise RuntimeError(f"{clip}: no hand_valid frame, so the S4 windows are undefined")
    first, last = int(detected[0]), int(detected[-1])
    init = range(1, first - HAND_MARGIN + 1)
    end = range(last + HAND_MARGIN, n_frames)
    if len(init) == 0:
        raise RuntimeError(f"{clip}: first hand_valid frame is {first}, the initial window 1..{first - HAND_MARGIN} is empty")

    n, d, stats = fit_window(rec, masks, meta, init)
    result = {"plane_world": [*n.tolist(), d], "init_window": [init[0], init[-1]], "init_n_frames": len(init),
              "init_short_flag": len(init) < MIN_FRAMES, "first_hand_valid": first, "last_hand_valid": last, **stats,
              "end_window": [end[0], end[-1]] if len(end) else None, "end_n_frames": len(end),
              "end_plane_world": None, "end_n_points": None, "end_inlier_frac": None, "end_rms_m": None,
              "end_normal_vs_y_deg": None, "normal_diff_deg": None,
              "ransac": {"threshold_m": RANSAC_TOL, "seed": SEED, "iterations": RANSAC_ITERS}}
    if len(end) >= MIN_FRAMES:
        n_end, d_end, end_stats = fit_window(rec, masks, meta, end)
        result.update({f"end_{k}": v for k, v in end_stats.items()})
        result["end_plane_world"] = [*n_end.tolist(), d_end]
        result["normal_diff_deg"] = float(np.degrees(np.arccos(np.clip(n @ n_end, -1, 1))))

    (out_dir / "table.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    stage_cli("table", run)
