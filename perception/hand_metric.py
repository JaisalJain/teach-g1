"""S5 -- metric hand: move the WiLoR hand along its viewing ray so that its depth agrees with the iPhone LiDAR.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.hand_metric --clip <id>

WiLoR only sees a crop around the hand, so its distance from the camera is the weakest part of its 3D hand. This stage
corrects that distance with the LiDAR and nothing else: the direction to the hand (its centroid's viewing ray) stays as
S2 placed it (see the plumbing check below for how good that placement is). Per hand_valid frame i:
1. Render the WiLoR mesh into the 256x192 depth grid (ORIGINAL orientation, K_depth[i]) with a z-buffer triangle
   rasteriser: every pixel centre gets the depth of the nearest triangle covering it. The depth is perspective-correct:
   1/z (not z) is linear in screen coordinates across a flat triangle, so we interpolate 1/z with the screen-space
   barycentrics. Rendering at the depth map's own resolution compares the mesh and the LiDAR pixel for pixel.
2. Hand pixels = rendered silhouette eroded 1 px (edge pixels mix hand and background in the coarse LiDAR map)
   & confidence >= 1 & depth > 0 & NOT the mug body / handle masks dilated 2 px (where the hand holds the mug, the LiDAR
   may see the mug instead of the hand).
3. Two-pass median of (lidar - mesh): pass 1 over all hand pixels gives D0; pass 2 keeps the pixels within 3 cm of
   mesh + D0 (drops background seen through gaps between the fingers, the mug, LiDAR smearing) and takes the median D.
4. Translate verts, kp3d and cam_t by (D / z_c) c, c = mesh centroid in the camera frame. The vector is parallel to c,
   so the centroid stays on its pixel, and its z part is exactly D, so every vertex moves D deeper. The hand is NOT
   rescaled: its size is MANO's with WiLoR's shape (betas).

metric_valid[i] = hand_valid[i] and >= 30 pass-2 pixels. D, the residual (median |lidar - (mesh + D)| over the pass-2
pixels) and the pixel counts are logged for every frame where D exists, also below 30 pixels (for the plots). The
corrected hand (verts, kp3d, cam_t) and its plumbing error exist only on metric_valid frames; NaN everywhere else.

Plumbing check: the corrected kp3d projected with the RGB intrinsics K[i] vs WiLoR's own 2D keypoints kp2d (median over
the 21 keypoints). Both come from the same network output, so this tests OUR transforms (crop camera -> full image,
upright -> original, the shift), NOT WiLoR's accuracy. The number is large, and almost all of it is S2's placement of
the hand, not S5: WiLoR makes kp2d with its near-orthographic crop camera (focal 5000 px on a 256 px crop), where a few
cm of depth inside the hand do not matter. S2's cam_crop_to_full_K (as specified) uses the real fx (~1400 px) and gives
WiLoR's weak-perspective depth tz to the MANO origin, but the joints sit ~5 cm deeper than tz. Full perspective then
draws the whole hand toward the principal point by about (joint depth - tz) / joint depth, ~10 % of its pixel distance
from it: a sideways error of the metric hand of ~1 cm, plus a smaller shape term from the hand's own depth span.
Projecting S2's kp3d with weak perspective about tz instead matches kp2d to ~1 px, so the transforms are consistent with
each other. S5 keeps the centroid's ray, so it keeps the sideways error (the shift itself adds only about
|D| / (z_c + D) x each keypoint's pixel distance from the centroid). Per-clip numbers go to docs/log.md; whether S2
should place the hand under full perspective is an open question for the user.

Writes data/work/<clip>/hand_metric/hand_metric.npz (all arrays over canonical index i, NaN / False when undefined).
"""

from __future__ import annotations

import cv2
import numpy as np

from perception import stray
from perception.common import RAW, stage_cli, stage_dir
from perception.geometry import project
from perception.masks_sam2 import BODY, HANDLE, load_masks

PASS2_TOL_M = 0.03     # pass 2 keeps |lidar - (mesh + D0)| < 3 cm
MIN_PIXELS = 30        # pass-2 pixels needed for metric_valid
CONF_MIN = 1           # ARKit confidence: 0 low, 1 medium, 2 high
MUG_DILATE_PX = 2


def render_depth(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, size_wh) -> np.ndarray:
    """Z-buffer render of a triangle mesh. verts (V, 3) in the camera frame [m], faces (F, 3) vertex indices, K and
    size_wh = (W, H) of the target image. -> (H, W) depth along the optical axis [m], inf where nothing is rendered.

    Vectorised over triangles: each triangle tests the pixel centres (integer coordinates) inside its own bounding box,
    all triangles at once (a ragged list built with np.repeat)."""
    w, h = size_wh
    tri = verts[faces]                                       # (F, 3 corners, xyz)
    tri = tri[np.all(tri[..., 2] > 0, axis=1)]               # triangles behind the camera cannot be projected
    uv = project(K, tri)                                     # (F, 3, 2)
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    e1, e2 = b - a, c - a
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]         # twice the signed screen area
    u0 = np.maximum(np.ceil(uv[..., 0].min(1)), 0)
    u1 = np.minimum(np.floor(uv[..., 0].max(1)), w - 1)
    v0 = np.maximum(np.ceil(uv[..., 1].min(1)), 0)
    v1 = np.minimum(np.floor(uv[..., 1].max(1)), h - 1)
    keep = (np.abs(area) > 1e-12) & (u1 >= u0) & (v1 >= v0)  # edge-on triangles and triangles between pixel centres
    zbuf = np.full(h * w, np.inf)
    if not keep.any():
        return zbuf.reshape(h, w)
    a, e1, e2, area, z = a[keep], e1[keep], e2[keep], area[keep], tri[keep][..., 2]
    u0, v0 = u0[keep].astype(int), v0[keep].astype(int)
    bw, bh = u1[keep].astype(int) - u0 + 1, v1[keep].astype(int) - v0 + 1

    # one entry per (triangle, pixel of its bounding box)
    n = bw * bh
    f = np.repeat(np.arange(len(n)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)  # position inside the triangle's box
    pu, pv = u0[f] + k % bw[f], v0[f] + k // bw[f]

    # screen-space barycentrics: p - a = l1 (b - a) + l2 (c - a)
    px, py = pu - a[f, 0], pv - a[f, 1]
    l1 = (px * e2[f, 1] - py * e2[f, 0]) / area[f]
    l2 = (e1[f, 0] * py - e1[f, 1] * px) / area[f]
    l0 = 1 - l1 - l2
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    f, pu, pv, l0, l1, l2 = f[inside], pu[inside], pv[inside], l0[inside], l1[inside], l2[inside]
    depth = 1 / (l0 / z[f, 0] + l1 / z[f, 1] + l2 / z[f, 2])
    np.minimum.at(zbuf, pv * w + pu, depth)                  # nearest surface wins
    return zbuf.reshape(h, w)


def hand_pixels(zbuf: np.ndarray, depth: np.ndarray, conf: np.ndarray, mug: np.ndarray) -> np.ndarray:
    """(H, W) bool: eroded rendered silhouette & confidence >= 1 & depth > 0 & outside the dilated mug mask."""
    sil = cv2.erode(np.isfinite(zbuf).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    k = 2 * MUG_DILATE_PX + 1
    mug = cv2.dilate(mug.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return sil & (conf >= CONF_MIN) & (depth > 0) & ~mug


def depth_shift(lidar: np.ndarray, mesh: np.ndarray):
    """Two-pass median of lidar - mesh over the hand pixels (1D arrays, metres).
    -> (D, residual, n_pass1, n_pass2); D and residual are NaN when a pass has no pixels."""
    diff = np.asarray(lidar, dtype=np.float64) - mesh
    if diff.size == 0:
        return np.nan, np.nan, 0, 0
    d0 = np.median(diff)
    keep = np.abs(diff - d0) < PASS2_TOL_M                   # |lidar - (mesh + D0)| < 3 cm
    if not keep.any():                                       # possible for an even count with two far-apart middles
        return np.nan, np.nan, diff.size, 0
    d = np.median(diff[keep])
    return d, np.median(np.abs(diff[keep] - d)), diff.size, int(keep.sum())


def ray_shift(verts: np.ndarray, d: float) -> np.ndarray:
    """(3,) translation moving the mesh centroid c along its own viewing ray from depth z_c to z_c + d."""
    c = verts.mean(0)
    return d / c[2] * c


def run(clip: str) -> dict:
    out = stage_dir(clip, "hand_metric")
    meta = np.load(stage_dir(clip, "frames") / "meta.npz")
    hand = dict(np.load(stage_dir(clip, "hand") / "hand.npz"))   # dict: an NpzFile re-reads an array on every access
    masks = load_masks(clip)
    rec = stray.load(RAW / clip)
    K, K_depth, depth_size = meta["K"], meta["K_depth"], tuple(int(s) for s in meta["depth_size"])
    faces = hand["faces"]
    hand_valid = hand["hand_valid"]
    n = len(meta["t"])
    assert len(hand_valid) == n == len(rec), (len(hand_valid), n, len(rec))

    shift_m = np.full(n, np.nan)
    resid_m = np.full(n, np.nan)
    n_pass1 = np.zeros(n, np.int32)
    n_pass2 = np.zeros(n, np.int32)
    kp3d = np.full((n, 21, 3), np.nan, np.float32)
    verts = np.full((n, 778, 3), np.nan, np.float16)
    cam_t = np.full((n, 3), np.nan, np.float32)
    reproj = np.full(n, np.nan)

    for i in np.flatnonzero(hand_valid):
        v = hand["verts"][i].astype(np.float64)
        zbuf = render_depth(v, faces, K_depth[i], depth_size)
        depth, conf = rec.depth(i), rec.confidence(i)
        mug = masks.get(BODY, i, "depth") | masks.get(HANDLE, i, "depth")
        pix = hand_pixels(zbuf, depth, conf, mug)
        shift_m[i], resid_m[i], n_pass1[i], n_pass2[i] = depth_shift(depth[pix], zbuf[pix])
        if n_pass2[i] < MIN_PIXELS:                           # not metric_valid: D stays in the log, no corrected hand
            continue
        s = ray_shift(v, shift_m[i])
        verts[i] = v + s
        kp3d[i] = hand["kp3d"][i] + s
        cam_t[i] = hand["cam_t"][i] + s
        # plumbing check (our transforms), not WiLoR's accuracy: corrected 3D keypoints vs WiLoR's own 2D keypoints
        reproj[i] = np.median(np.linalg.norm(project(K[i], kp3d[i].astype(np.float64)) - hand["kp2d"][i], axis=1))

    metric_valid = hand_valid & (n_pass2 >= MIN_PIXELS)
    np.savez_compressed(out / "hand_metric.npz", depth_shift_m=shift_m, lidar_residual_m=resid_m,
                        n_pixels_pass1=n_pass1, n_pixels=n_pass2, metric_valid=metric_valid, kp3d=kp3d, verts=verts,
                        cam_t=cam_t, reproj_err_px=reproj)

    stats = {"n_frames": n, "n_hand_valid": int(hand_valid.sum()), "n_shift_defined": int(np.isfinite(shift_m).sum()),
             "n_metric_valid": int(metric_valid.sum()),
             "plumbing_note": "reproj = corrected kp3d vs WiLoR's own kp2d: tests our transforms, not WiLoR's "
                              "accuracy. Dominated by S2's full-perspective placement vs WiLoR's near-orthographic "
                              "crop camera (see the hand_metric docstring)"}
    if metric_valid.any():                                   # summaries over metric_valid frames
        mv = metric_valid
        stats.update({"depth_shift_m_median": float(np.median(shift_m[mv])),
                      "depth_shift_m_range": [float(shift_m[mv].min()), float(shift_m[mv].max())],
                      "lidar_residual_m_median": float(np.median(resid_m[mv])),
                      "n_pixels_median": float(np.median(n_pass2[mv])),
                      "plumbing_reproj_err_px_median": float(np.median(reproj[mv]))})
    return stats


if __name__ == "__main__":
    stage_cli("hand_metric", run)
