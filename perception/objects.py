"""S6 -- mug and coaster poses: fit the known geometry (config/objects.yaml) to the SAM 2 masks.

    # A4 mask test (both mask sources, every clip), then S6 itself (needs the test on all 4 clips):
    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.objects --clip <id> --stage test
    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.objects --clip <id>

The mug is glossy, so its LiDAR depth is unreliable: depth only gives the starting guess. The pose comes from the
silhouette instead. A cylinder with the label radius and height, standing on the S4 table plane, has only its base
centre unknown: 2 numbers in the plane.

Windows (hand out of view): init = the S4 initial window; final = (last hand_valid frame + 15) .. N-1, or the last
0.5 s, flagged, if that is empty (the hand never leaves). Each fit uses the 11 frames centred in its window: a pixelwise
majority mask (set in >= 6 of the 11 frames, so a frame where SAM 2 slips does not count) and the middle frame's camera
pose and intrinsics. v1 fits ran on the half-res video masks (720x960 HxW, ORIGINAL orientation, K scaled by 0.5); A4
(below) adds full-res masks, fitted with the unscaled K.

- Mug body, 2-DOF: silhouette = filled convex hull of 72 projected points on each end circle (a cylinder's outline is
  the hull of its two end circles); a pixel is in the silhouette if its CENTRE is inside the (unrounded) hull -- the same
  rule that defines the SAM 2 masks (logits sampled at pixel centres, > 0). (cv2.fillConvexPoly, the first version, also
  fills every pixel the hull edge touches and rounds the vertices: ~0.65 px too large per side, a known bias.)
  Score = IoU with the body majority mask.
  Start (the spec's rule): the median of the body's depth points (confidence >= 1), dropped onto the plane and moved r
  away from the camera. The rule assumes the depth points lie on the near side of the mug wall. When the camera sees
  into the open mug, many lie on the inner far wall, and the start can end up about r beyond the axis. Grid +-6 cm at
  2 mm steps around the start, then Nelder-Mead from the best cell. Each window records whether the best grid cell is
  on the grid edge, so a start that is too far off shows up in the output instead of passing silently.
- Validation, 4-DOF: the same fit with r in [3, 7] cm and h in [5, 13] cm free. If the label r and h explain the masks,
  the free fit lands on them. h counts as observable only if the optical axis is >= 25 deg away from straight down.
  Seen from straight down, the outline is essentially the top rim, whose image radius is ~ f r / (Z_cam - h) (Z_cam =
  camera height above the table). So r and h trade off along a ridge of near-equal IoU, and the free DIAMETER is not
  observable either. Example, top-down clip 26d4118b0c (camera ~0.40 m above the table), A4 init window: the video and
  the full-res masks give free fits (r, h) = (52.6, 88.3) mm and (57.1, 63.5) mm, IoU 0.977 and 0.975, and the same
  r / (Z_cam - h) = 0.171 -- Nelder-Mead stopped at different points of the ridge, the masks barely differ. So
  h_observable = False means "neither r nor h is measured"; the flag keeps its v1 definition (it names h only).
- Handle point P: the ray through the centroid of the handle majority mask, cut with the plane h/2 above the table
  (h = label height). The yaw (cup axis -> P) is computed later, in the table frame. Invalid if the handle mask has
  less than 0.2 % of the body mask's area.
- Coaster (init window only): outline pixels of the coaster majority mask -> rays -> the plane `thickness` above the
  table (the spec's choice) -> RANSAC circle (4 mm) -> least-squares circle on the inliers. Its diameter is reported
  next to the config estimate. That plane is exact on the far side, where the outline is the top rim. On the near side
  the outline is the bottom edge of the side wall, so those points land up to thickness * cot(elevation) too close to
  the camera, and the centre shifts toward the camera by roughly half that (2.5-4 mm in the synthetic test). Known bias.

Known limitation (spec): in the final window the mug stands on the 5 mm coaster but is modelled standing on the table.

A4 (M2 v2). With the video masks the free fit is 6-7 mm wider than the label on every clip, and a ruler confirms the
label, so the masks are tested:
- `--stage test` -> data/work/<clip>/mask_test/mask_test.json: the 2-DOF fit and the 4-DOF validation fit of both windows
  on two mask sources. "video" = exactly v1. "full_res_image" = the majority (>= 6 of 11) of the SAM 2 image-predictor
  masks (masks_sam2 --stage full) at full res, with the middle frame's full-res K; the start guess resamples that mask
  to depth resolution with the same nearest-pixel-centre rule. Same functions, grid, Nelder-Mead, bounds and start rule.
  SAM 2 works at 1024x1024 in both predictors (see masks_sam2), so this tests the image predictor on full-res input,
  not a higher segmentation resolution.
- default stage = S6 v2: needs the test on all 4 clips. The user's pre-set rule: if the full-res masks give an init
  free-fit diameter within 5 mm of the label on at least 3 clips, every clip uses them; otherwise every clip keeps the
  video masks (one source for all clips). The rule is applied literally, the top-down clip included (its diameter is
  not observable, see the validation fit above). The chosen source's fits are copied from mask_test.json; the handle
  and the coaster always come from the video masks (as v1), since A4 only re-segments the mug body.
  The decision reads all four mask_test.json files: each one's init frames are checked against that clip's current S4
  window, but a masks_full rerun with the same frames is not detected. run_clip reruns masks_full and mask_test on
  every clip (pass A) before any S6 (pass B), so the four tests are current there.

Writes data/work/<clip>/objects/: objects.json (the v1 layout + free_fit.diameter_m (= 2 r_m, as in mask_test.json) +
mug_mask_source + mask_decision), and init_body_majority.npz (packbits of the VIDEO init body majority mask, ORIGINAL
half-res; S8 compares every frame's video body mask with it, so it stays on the video masks whatever the fit source).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from matplotlib.path import Path as Polygon
from scipy.optimize import minimize

from perception import geometry as G
from perception import stray
from perception.common import CLIPS, RAW, WORK, cfg, is_done, load_config, stage_cli, stage_dir
from perception.masks_sam2 import BODY, COASTER, HANDLE, load_full_masks, load_masks, resample

N_FIT = 11                               # frames per fit (majority mask)
N_RIM = 72                               # projected points per end circle of the cylinder
GRID = np.linspace(-0.06, 0.06, 61)      # m, 2 mm steps around the start guess (both in-plane directions)
STEP = 0.002                             # m, initial Nelder-Mead simplex step
R_BOUNDS, H_BOUNDS = (0.03, 0.07), (0.05, 0.13)   # m, free r and h of the validation fit
MIN_SIDE_VIEW_DEG = 25.0                 # h is observable if the optical axis is at least this far from straight down
HANDLE_MIN_AREA = 0.002                  # handle / body mask area below which the handle (yaw) is invalid
CIRCLE_TOL = 0.004                       # m, RANSAC circle inlier distance
FINAL_GAP = 15                           # frames between the last hand_valid frame and the final window
FINAL_FALLBACK_S = 0.5                   # s, final window if the hand never leaves
DECISION_TOL_M = 0.005                   # A4 rule: full-res init free-fit diameter within 5 mm of the label (= G6's tolerance)
DECISION_MIN_CLIPS = 3                   # A4 rule: ... on at least 3 of the 4 clips


def final_window(t: np.ndarray, last_hand_valid: int) -> tuple[list, str | None]:
    """[a, b] (inclusive) of the final window, and a flag that is None unless the fallback was used."""
    n = len(t)
    if last_hand_valid + FINAL_GAP <= n - 1:
        return [last_hand_valid + FINAL_GAP, n - 1], None
    a = int(np.flatnonzero(t >= t[-1] - FINAL_FALLBACK_S)[0])
    return [a, n - 1], (f"no frame at or after last hand_valid + {FINAL_GAP} (hand never leaves): "
                        f"used the last {FINAL_FALLBACK_S} s")


def fit_frames(a: int, b: int) -> list:
    """The 11 frames centred in the inclusive window [a, b] (all of them if the window is shorter)."""
    start = a + max(b - a + 1 - N_FIT, 0) // 2
    return list(range(start, min(start + N_FIT, b + 1)))


def majority(masks) -> np.ndarray:
    """Pixels set in more than half of the masks (>= 6 of 11); masks = a list or (n, H, W) stack of bool masks."""
    return 2 * np.sum(masks, axis=0) > len(masks)


def onto_plane(p: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return p - (p @ n + d) * n


def start_guess(rec, i: int, K_depth: np.ndarray, T_wc: np.ndarray, body: np.ndarray, n, d, r) -> tuple:
    """Median of the body's depth points in frame i, dropped onto the plane and moved r away from the camera.
    `body` may be half or full res: resample takes it to depth resolution (nearest pixel centre) either way.
    Returns (guess_world, number of depth points used)."""
    depth, conf = rec.depth(i), rec.confidence(i)
    sel = resample(body, "depth") & (conf >= 1) & (depth > 0)
    if not sel.any():
        raise ValueError(f"frame {i}: no depth points under the body majority mask")
    v, u = np.nonzero(sel)
    pts = G.transform(T_wc, G.backproject_pixels(np.column_stack([u, v]), depth[sel], K_depth))
    foot = onto_plane(np.median(pts, axis=0), n, d)
    away = foot - onto_plane(T_wc[:3, 3], n, d)          # horizontal direction camera -> mug
    return foot + r * away / np.linalg.norm(away), int(sel.sum())


def silhouette(base, r, h, n, T_cw, K, shape) -> np.ndarray:
    """Filled convex hull of the projected bottom and top circles of a cylinder standing on the plane at `base`."""
    e1, e2 = G.plane_basis(n)
    a = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False)
    rim = r * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
    uv = G.project(K, G.transform(T_cw, np.concatenate([base + rim, base + h * n + rim])))
    hull = cv2.convexHull(uv.astype(np.float32))[:, 0].astype(np.float64)
    img = np.zeros(shape, bool)
    (u0, v0), (u1, v1) = np.clip(np.floor(hull.min(0)), 0, None).astype(int), np.ceil(hull.max(0)).astype(int)
    u1, v1 = min(u1, shape[1] - 1), min(v1, shape[0] - 1)
    if u1 < u0 or v1 < v0:                                # hull entirely outside the image
        return img
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]               # pixel centres of the hull's bounding box
    inside = Polygon(hull).contains_points(np.column_stack([uu.ravel(), vv.ravel()]))
    img[v0:v1 + 1, u0:u1 + 1] = inside.reshape(uu.shape)
    return img


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1)


def fit_body(mask, guess, r, h, n, T_cw, K) -> tuple:
    """2-DOF fit: the base centre in the plane that maximises the IoU. Returns (base_world, iou, [iu, iv] of the best
    grid cell)."""
    e1, e2 = G.plane_basis(n)

    def score(x):
        return iou(silhouette(guess + x[0] * e1 + x[1] * e2, r, h, n, T_cw, K, mask.shape), mask)

    grid = np.array([[score((u, v)) for v in GRID] for u in GRID])
    iu, iv = np.unravel_index(np.argmax(grid), grid.shape)
    x0 = np.array([GRID[iu], GRID[iv]])
    res = minimize(lambda x: -score(x), x0, method="Nelder-Mead",
                   options={"initial_simplex": [x0, x0 + [STEP, 0], x0 + [0, STEP]]})
    return guess + res.x[0] * e1 + res.x[1] * e2, -res.fun, [int(iu), int(iv)]


def fit_body_free(mask, base, r, h, n, T_cw, K) -> tuple:
    """4-DOF validation fit, starting from the 2-DOF base and the label r, h: x = (in-plane offsets from base, r, h).
    Returns (x, iou)."""
    e1, e2 = G.plane_basis(n)

    def score(x):
        return iou(silhouette(base + x[0] * e1 + x[1] * e2, x[2], x[3], n, T_cw, K, mask.shape), mask)

    x0 = np.array([0.0, 0.0, r, h])
    res = minimize(lambda x: -score(x), x0, method="Nelder-Mead", bounds=[(None, None), (None, None), R_BOUNDS, H_BOUNDS],
                   options={"initial_simplex": np.vstack([x0, x0 + STEP * np.eye(4)])})
    return res.x, -res.fun


def handle_point(mask, T_wc, K, n, d, h) -> np.ndarray:
    """P (world): the ray through the centroid of the handle mask, cut with the plane h/2 above the table."""
    v, u = np.nonzero(mask)
    ray = T_wc[:3, :3] @ G.pixel_rays(np.array([u.mean(), v.mean()]), K)
    return G.ray_plane(T_wc[:3, 3], ray, n, d - h / 2)


def circle_3pts(p: np.ndarray):
    """(centre, radius) of the circle through 3 points in 2D; None if they are collinear."""
    A = 2 * (p[1:] - p[0])           # |x - p_k|^2 = |x - p_0|^2  ->  2 (p_k - p_0) . x = |p_k|^2 - |p_0|^2
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    c = np.linalg.solve(A, (p[1:] ** 2).sum(1) - (p[0] ** 2).sum())
    return c, np.linalg.norm(p[0] - c)


def circle_lsq(p: np.ndarray) -> tuple:
    """Algebraic least-squares circle: x^2 + y^2 = 2a x + 2b y + k is linear in (a, b, k); radius^2 = k + a^2 + b^2."""
    (a, b, k), *_ = np.linalg.lstsq(np.column_stack([2 * p, np.ones(len(p))]), (p ** 2).sum(1), rcond=None)
    return np.array([a, b]), float(np.sqrt(k + a * a + b * b))


def ransac_circle(p: np.ndarray, tol: float, iters: int = 1000) -> tuple:
    """RANSAC over 3-point circles (inlier: |distance to centre - radius| < tol), then circle_lsq on the inliers.
    Returns (centre, radius, inlier mask)."""
    rng = np.random.default_rng(0)
    best = np.zeros(len(p), bool)
    for _ in range(iters):
        fit = circle_3pts(p[rng.choice(len(p), 3, replace=False)])
        if fit is None:
            continue
        inl = np.abs(np.linalg.norm(p - fit[0], axis=1) - fit[1]) < tol
        if inl.sum() > best.sum():
            best = inl
    centre, radius = circle_lsq(p[best])
    return centre, radius, best


def fit_coaster(mask, T_wc, K, n, d, thickness) -> dict:
    """Circle through the outline of the coaster mask, cast onto the plane of its top rim. Centre reported on the table."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty coaster majority mask")
    uv = max(contours, key=cv2.contourArea)[:, 0].astype(np.float64)       # (M, 2) outline pixels (u, v)
    rim = G.ray_plane(T_wc[:3, 3], G.pixel_rays(uv, K) @ T_wc[:3, :3].T, n, d - thickness)
    e1, e2 = G.plane_basis(n)
    o = rim.mean(0)
    centre, radius, inl = ransac_circle(np.column_stack([(rim - o) @ e1, (rim - o) @ e2]), CIRCLE_TOL)
    c = onto_plane(o + centre[0] * e1 + centre[1] * e2, n, d)
    return {"centre_world": c.tolist(), "diameter_m": 2 * radius, "n_contour_px": len(uv), "inlier_frac": float(inl.mean())}


def load_clip(clip: str) -> tuple:
    """What both S6 stages read: config, frames/meta.npz, the S4 plane (unit n, d), the two windows and the final flag."""
    c = load_config()
    meta = np.load(WORK / clip / "frames" / "meta.npz")
    table = json.loads((WORK / clip / "table" / "table.json").read_text())
    plane = np.array(table["plane_world"], dtype=np.float64)
    n, d = plane[:3] / np.linalg.norm(plane[:3]), plane[3] / np.linalg.norm(plane[:3])
    final, final_flag = final_window(meta["t"], table["last_hand_valid"])
    return c, meta, n, d, {"init": list(table["init_window"]), "final": final}, final_flag


def fit_mug(body, K_fit, rec, m, K_depth, T_wc, n, d, r, h) -> dict:
    """The v1 mug fit of one window on a majority body mask of any resolution (K_fit = the intrinsics of that
    resolution; m, K_depth, T_wc = the middle frame): start guess -> 2-DOF fit -> 4-DOF validation fit."""
    T_cw = G.invert(T_wc)
    guess, n_depth = start_guess(rec, m, K_depth, T_wc, body, n, d, r)
    base, body_iou, cell = fit_body(body, guess, r, h, n, T_cw, K_fit)
    free, free_iou = fit_body_free(body, base, r, h, n, T_cw, K_fit)
    side_view_deg = float(np.degrees(np.arccos(np.clip(T_wc[:3, 2] @ -n, -1, 1))))   # optical axis vs straight down
    return {
        "base_centre_world": base.tolist(), "iou": body_iou, "init_guess_world": guess.tolist(), "n_depth_points": n_depth,
        # diagnostics of the start guess: a best cell on the grid edge means the optimum may lie outside the grid
        "fit_minus_guess_m": float(np.linalg.norm(base - guess)),
        "grid_best_cell": cell, "grid_best_on_edge": bool({0, len(GRID) - 1} & set(cell)),
        "free_fit": {"r_m": free[2], "h_m": free[3], "diameter_m": 2 * free[2], "iou": free_iou,
                     "optical_axis_vs_normal_deg": side_view_deg, "h_observable": side_view_deg >= MIN_SIDE_VIEW_DEG}}


def run_test(clip: str) -> dict:
    """A4 test: the mug fits of both windows on both mask sources -> mask_test/mask_test.json."""
    out = stage_dir(clip, "mask_test")
    c, meta, n, d, windows, _ = load_clip(clip)
    r, h = cfg(c, "mug", "body_outer_diameter_m") / 2, cfg(c, "mug", "height_m")
    video, full = load_masks(clip), load_full_masks(clip)
    rec = stray.load(RAW / clip)
    result = {"video": {}, "full_res_image": {}}
    for name, (a, b) in windows.items():
        frames = fit_frames(a, b)
        m = frames[(len(frames) - 1) // 2]                # = (a + b) // 2, the same middle frame S7 uses
        if full[name][0] != frames:
            raise ValueError(f"masks_full has frames {full[name][0]} for the {name} window but S6 uses {frames}: "
                             "rerun masks_sam2 --stage full")
        sources = {"video": (majority([video.get(BODY, i, "half") for i in frames]), G.scale_K(meta["K"][m], 0.5)),
                   "full_res_image": (majority(full[name][1]), meta["K"][m])}
        for source, (body, K_fit) in sources.items():
            result[source][name] = {"frames_used": frames, "mid_frame": m, **fit_mug(
                body, K_fit, rec, m, meta["K_depth"][m], meta["T_world_cam"][m], n, d, r, h)}
    (out / "mask_test.json").write_text(json.dumps(result, indent=1))
    stats = {}
    for w in windows:
        stats[f"{w}_iou"] = {s: result[s][w]["iou"] for s in result}
        stats[f"{w}_free_diameter_m"] = {s: result[s][w]["free_fit"]["diameter_m"] for s in result}
        stats[f"{w}_free_h_m"] = {s: result[s][w]["free_fit"]["h_m"] for s in result}
    return stats


def mask_decision() -> dict:
    """The user's pre-set A4 rule, one answer for all clips: the full-res image masks if their init-window free-fit
    diameter is within 5 mm of the label on at least 3 clips, otherwise the video masks."""
    missing = [cl for cl in CLIPS if not is_done(cl, "mask_test")]
    if missing:
        raise FileNotFoundError(f"S6 needs objects --stage test on all {len(CLIPS)} clips first; missing: {', '.join(missing)}")
    label = cfg(load_config(), "mug", "body_outer_diameter_m")
    diam, stale = {}, []
    for cl in CLIPS:
        test = json.loads((WORK / cl / "mask_test" / "mask_test.json").read_text())["full_res_image"]["init"]
        init_window = json.loads((WORK / cl / "table" / "table.json").read_text())["init_window"]
        if test["frames_used"] != fit_frames(*init_window):    # S4 rerun on another clip since its mask test
            stale.append(cl)
        diam[cl] = test["free_fit"]["diameter_m"]
    if stale:
        raise ValueError(f"mask_test of {', '.join(stale)} used other init frames than the current S4 window: "
                         "rerun objects --stage test there")
    within = [cl for cl, D in diam.items() if abs(D - label) <= DECISION_TOL_M]
    chosen = "full_res_image" if len(within) >= DECISION_MIN_CLIPS else "video"
    return {"rule": f"full_res_image for all clips if |full-res init free-fit diameter - label| <= "
                    f"{DECISION_TOL_M * 1e3:.0f} mm on >= {DECISION_MIN_CLIPS} of {len(CLIPS)} clips, else video",
            "label_diameter_m": label, "full_res_init_diameter_m": diam, "clips_within_5mm": within,
            "n_within_5mm": len(within), "chosen": chosen,
            "reason": f"{len(within)} of {len(CLIPS)} clips within {DECISION_TOL_M * 1e3:.0f} mm of the "
                      f"{label * 1e3:.1f} mm label ({', '.join(within) or 'none'}); the rule needs {DECISION_MIN_CLIPS} -> "
                      + ("use the full-res image masks" if chosen == "full_res_image" else "keep the video masks")}


def run(clip: str) -> dict:
    """S6 v2: the mug fits of the source the A4 rule chose (copied from mask_test.json), handle and coaster as v1."""
    out = stage_dir(clip, "objects")
    decision = mask_decision()
    source = decision["chosen"]
    test = json.loads((WORK / clip / "mask_test" / "mask_test.json").read_text())[source]
    c, meta, n, d, windows, final_flag = load_clip(clip)
    h = cfg(c, "mug", "height_m")
    K, T_wc = meta["K"], meta["T_world_cam"]
    masks = load_masks(clip)

    result = {"windows": {**windows, "final_flag": final_flag}}
    for name, (a, b) in windows.items():
        frames = fit_frames(a, b)
        if test[name]["frames_used"] != frames:
            raise ValueError(f"mask_test used frames {test[name]['frames_used']} for the {name} window but S6 now uses "
                             f"{frames}: rerun objects --stage test")
        m = test[name]["mid_frame"]
        K_half = G.scale_K(K[m], 0.5)
        # handle and coaster: always the video masks, as v1 (A4 only re-segments the mug body)
        body = majority([masks.get(BODY, i, "half") for i in frames])
        handle = majority([masks.get(HANDLE, i, "half") for i in frames])
        area_ratio = float(handle.sum() / body.sum())
        result[name] = {
            **test[name],
            "handle": {"P_world": handle_point(handle, T_wc[m], K_half, n, d, h).tolist() if handle.any() else None,
                       "area_ratio": area_ratio, "valid": area_ratio >= HANDLE_MIN_AREA}}
        if name == "init":
            np.savez_compressed(out / "init_body_majority.npz", bits=np.packbits(body.reshape(-1)),
                                shape=np.array(body.shape))
            coaster = majority([masks.get(COASTER, i, "half") for i in frames])
            result["coaster"] = {**fit_coaster(coaster, T_wc[m], K_half, n, d, cfg(c, "coaster", "thickness_m")),
                                 "config_diameter_m": cfg(c, "coaster", "diameter_m")}
    result["mug_mask_source"], result["mask_decision"] = source, decision

    (out / "objects.json").write_text(json.dumps(result, indent=1))
    return {"mug_mask_source": source, "n_within_5mm": decision["n_within_5mm"],
            "init_iou": result["init"]["iou"], "final_iou": result["final"]["iou"], "final_flag": final_flag,
            "grid_best_on_edge": [result[w]["grid_best_on_edge"] for w in ("init", "final")],
            "fit_minus_guess_m": [result[w]["fit_minus_guess_m"] for w in ("init", "final")],
            "init_free_r_m": result["init"]["free_fit"]["r_m"], "init_free_h_m": result["init"]["free_fit"]["h_m"],
            "h_observable": result["init"]["free_fit"]["h_observable"],
            "handle_valid": [result["init"]["handle"]["valid"], result["final"]["handle"]["valid"]],
            "coaster_diameter_m": result["coaster"]["diameter_m"],
            "coaster_config_diameter_m": cfg(c, "coaster", "diameter_m")}


if __name__ == "__main__":
    stage_cli({"objects": ("objects", run, False), "test": ("mask_test", run_test, False)})
