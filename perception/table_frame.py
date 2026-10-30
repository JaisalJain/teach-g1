"""S7 -- the table frame: z = table normal, origin = initial mug base centre, x = my forward (from the desk edges).

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.table_frame --clip <id>

Writes data/work/<clip>/table_frame/table_frame.json and lines.png (the desk-edge line fits, for the report).

Why the desk edges: ARKit's world heading is wherever the session started, so it says nothing about where I sat. The desk
is a rectangle (config/objects.yaml, desk length_m x width_m) and I sit at its near long edge, so my forward is
perpendicular to the long edges.

1. Desk boundary, in the middle frame m of the S4 initial window (no hand yet). Plane-inlier pixels (confidence 2,
   depth > 0, within 1 cm of the S4 plane) -> largest 4-connected component -> its EXTERNAL contour only
   (cv2.findContours, RETR_EXTERNAL, every contour pixel), without the pixels within 3 px of the image border (there
   the component is cut by the field of view, not by a desk edge). The external contour ignores everything the desk
   encloses (holes: low-confidence specks, a mug standing inside the desk outline). Its pixels are the pixels of the
   hole-filled component that have a 4-neighbour outside it, i.e. in the region outside the outline, not in a hole
   (checked in the self-test). v1 took the hole-filled component minus its 3x3 erosion: the same outline, plus every
   pixel that touches the outside only diagonally (the inner corner of each step of a jagged depth edge). So the
   same edge now gives fewer boundary pixels, hence fewer line inliers, against the unchanged LINE_MIN_INLIERS.
   The boundary pixel rays are intersected with the plane, so every boundary point lies exactly on it. Then the
   points on object outlines are dropped: in-plane distance to the initial mug axis < r + 3 cm (r = config mug
   radius), or to the coaster centre < its S6 fitted radius + 3 cm. Where the mug's region touches the off-desk region
   (through the desk's slot or the image edge), its outline is part of the external contour, and this drop is what
   removes it. It only removes points on the plane near the footprints: for a mug seen from the side, the desk
   boundary just above its silhouette lies on the plane BEHIND the mug (its "shadow") and may stay. As a diagnostic,
   the kept boundary pixels within 3 px of an S3 object mask are counted (not used).
2. Sequential RANSAC lines in 2D plane coordinates: up to 4 lines, 1 cm, >= 40 inliers each, inliers removed after each
   line. The lines are infinite, so one line can collect separate collinear clusters (v1 bug: a cup holder + a desk
   corner made one 0.58 m "long edge" with a 0.44 m gap). Extent = the longest contiguous run of its inliers: sorted
   along the line, consecutive inliers at most 2 cm apart; extent = last - first of that run. The largest gap between
   consecutive inliers is recorded as information. Lines within 10 deg of a group's first line join it.
3. Long-edge direction: the spec bounds the short edges at <= 0.39 m (config desk width_m and its tolerance), so a line
   with an extent over 0.42 m can only be a long edge; the group of the longest such line gives the direction. If no
   line is that long, the group with the largest total extent (sum of the members' extents) is used and the rule is
   recorded as not applied. (The near long edge is slightly curved and may break into short or slanted lines; the far
   one is straight.)
4. x = the in-plane perpendicular to the long edges, signed so that my first metric_valid palm position has x < 0
   (I sit at the near edge, so my hand starts on the -x side). y = z x x.
5. Cross-check at frame m against a reference direction: the optical axis projected on the plane (90 deg and 45 deg
   clips, expected angle 0 and |angle| 45; left / right is not confirmed for the 45 deg clips) or the upright image's
   "up" projected on the plane (top-down clip, expected 0). If no line was found: x = the reference for the 90 deg and
   top-down clips; a 45 deg clip is marked "failed" (no x, no T_table_world) instead of guessing.
6. Drift (the S4 check): signed distance from the origin to the S4 end-window plane, and the normal angle from S4.
   Flagged if > 1 cm or > 2 deg; null (N/A) if S4 had no end-window plane. It sees only a change of the plane (offset
   along the normal, tilt), not a drift of the world frame within the plane.

T_table_world maps world points into the table frame: p_table = R^T (p_world - origin), R = [x y z] (columns).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from scipy import ndimage

from perception import stray
from perception.common import CLIPS, RAW, WORK, cfg, load_config, stage_cli, stage_dir
from perception.geometry import R_up_from_orig, pix_orig_to_upright, pixel_rays, plane_basis, ray_plane, transform
from perception.masks_sam2 import BODY, COASTER, HANDLE, load_masks

PLANE_TOL = 0.01          # m, plane-inlier pixels (the S4 RANSAC threshold)
BORDER_PX = 3
OBJECT_MARGIN_M = 0.03    # drop boundary points within this (in-plane) of the mug footprint / the coaster disc
MAX_LINES = 4
LINE_TOL = 0.01           # m
LINE_MIN_INLIERS = 40
RANSAC_ITERS = 1000
SEED = 0
RUN_GAP_M = 0.02          # extent: consecutive inliers along a line at most this far apart
GROUP_DEG = 10.0
LONG_EDGE_MIN_M = 0.42    # spec: longer than any short edge (config desk width_m)
NEAR_OBJECT_PX = 3        # diagnostic only: boundary pixels this close to an S3 object mask (the S4 dilation)
DRIFT_OFFSET_M = 0.01
DRIFT_NORMAL_DEG = 2.0
GROUP_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DROPPED_COLOUR = "#b8329a"


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def desk_boundary(depth, conf, K_depth, T_wc, n, d) -> tuple[np.ndarray, np.ndarray]:
    """Step 1 on one depth frame, in pixels. Returns the external-contour pixels (M, 2) as (u, v), without the image
    border, and the desk component (H, W)."""
    p = transform(T_wc, stray.backproject(depth, K_depth))
    inlier = (conf == 2) & (depth > 0) & (np.abs(p @ n + d) < PLANE_TOL)
    labels, count = ndimage.label(inlier)                 # default structure = 4-connectivity
    if count == 0:
        return np.zeros((0, 2), int), inlier
    desk = labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])
    contours, _ = cv2.findContours(desk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # one contour (a 4-connected component is one 8-connected outline); it walks a 1-px-wide spur twice -> unique pixels
    uv = np.unique(np.concatenate(contours).reshape(-1, 2), axis=0)
    h, w = desk.shape
    inside = ((uv[:, 0] >= BORDER_PX) & (uv[:, 0] < w - BORDER_PX)
              & (uv[:, 1] >= BORDER_PX) & (uv[:, 1] < h - BORDER_PX))
    return uv[inside], desk


def in_plane_distance(pts: np.ndarray, centre: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Distance of points (M, 3) to the line through centre along the plane normal n (a vertical axis)."""
    v = pts - centre
    return np.linalg.norm(v - np.outer(v @ n, n), axis=1)


def longest_run(s: np.ndarray) -> tuple[float, float]:
    """Step 2 extent: positions s along a line -> (first, last) of the longest run whose sorted neighbours are at most
    RUN_GAP_M apart. extent = last - first."""
    s = np.sort(s)
    breaks = np.flatnonzero(np.diff(s) > RUN_GAP_M)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [len(s) - 1]])
    k = int(np.argmax(s[stops] - s[starts]))            # ties -> the first run along the line
    return float(s[starts[k]]), float(s[stops[k]])


def fit_lines(q: np.ndarray, rng) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Step 2: sequential RANSAC lines on 2D points q (M, 2). Returns [(centre, direction, inlier indices into q)]."""
    lines = []
    left = np.arange(len(q))
    while len(lines) < MAX_LINES and len(left) >= LINE_MIN_INLIERS:
        p = q[left]
        best_count, best = -1, None
        for _ in range(RANSAC_ITERS):
            a, b = p[rng.integers(len(p), size=2)]
            if np.linalg.norm(b - a) < 1e-9:          # repeated sample
                continue
            normal = unit(np.array([a[1] - b[1], b[0] - a[0]]))
            inl = np.abs((p - a) @ normal) < LINE_TOL
            if inl.sum() > best_count:
                best_count, best = inl.sum(), inl
        if best_count < LINE_MIN_INLIERS:
            break
        centre = p[best].mean(0)
        direction = np.linalg.eigh(np.cov(p[best].T))[1][:, 1]     # largest-variance direction (least-squares line)
        lines.append((centre, direction, left[best]))
        left = left[~best]
    return lines


def group_lines(dirs: list[np.ndarray], extents: list[float]) -> list[dict]:
    """Greedy grouping by undirected angle to each group's first line; direction = extent-weighted mean."""
    members = []
    for k, u in enumerate(dirs):
        for g in members:
            if np.degrees(np.arccos(min(1.0, abs(u @ dirs[g[0]])))) < GROUP_DEG:
                g.append(k)
                break
        else:
            members.append([k])
    groups = []
    for g in members:
        mean = sum(extents[k] * np.sign(dirs[k] @ dirs[g[0]]) * dirs[k] for k in g)
        groups.append({"line_ids": g, "dir_world": unit(mean).tolist(), "total_extent_m": float(sum(extents[k] for k in g))})
    return groups


def reference_direction(view: str, R_wc: np.ndarray, rot_k: int, z: np.ndarray) -> tuple[str, np.ndarray]:
    """Step 5: the view's reference direction, projected on the plane."""
    if view == "top-down":
        kind, a = "upright_image_up", R_wc @ R_up_from_orig(rot_k).T @ np.array([0.0, -1.0, 0.0])
    else:
        kind, a = "optical_axis", R_wc[:, 2]
    return kind, unit(a - (a @ z) * z)


def run(clip: str) -> dict:
    out_dir = stage_dir(clip, "table_frame")
    view = CLIPS[clip]
    meta = np.load(WORK / clip / "frames" / "meta.npz")
    table = json.loads((WORK / clip / "table" / "table.json").read_text())
    objects = json.loads((WORK / clip / "objects" / "objects.json").read_text())
    hf = np.load(WORK / clip / "hand_fit" / "hand_fit.npz")        # the hand after S5b (scale + lateral fit)
    rec = stray.load(RAW / clip)
    T_wc, K_depth, rot_k = meta["T_world_cam"], meta["K_depth"], int(meta["rot_k"])

    n, d = np.array(table["plane_world"][:3]), table["plane_world"][3]
    z = n
    origin = np.array(objects["init"]["base_centre_world"])

    # step 1: desk boundary of the middle frame, on the plane, without the mug and coaster outlines
    m = (table["init_window"][0] + table["init_window"][1]) // 2
    uv, desk = desk_boundary(rec.depth(m), rec.confidence(m), K_depth[m], T_wc[m], n, d)
    rays = pixel_rays(uv.astype(float), K_depth[m]) @ T_wc[m, :3, :3].T
    pts = ray_plane(np.broadcast_to(T_wc[m, :3, 3], rays.shape), rays, n, d)
    r_mug = cfg(load_config(), "mug", "body_outer_diameter_m") / 2
    coaster = objects["coaster"]
    near_mug = in_plane_distance(pts, origin, n) < r_mug + OBJECT_MARGIN_M
    r_coaster = coaster["diameter_m"] / 2                 # S6's fitted diameter, not the config value
    near_coaster = in_plane_distance(pts, np.array(coaster["centre_world"]), n) < r_coaster + OBJECT_MARGIN_M
    dropped = near_mug | near_coaster
    uv_dropped, pts_dropped = uv[dropped], pts[dropped]
    uv, pts = uv[~dropped], pts[~dropped]

    # step 2: lines in 2D plane coordinates (e1, e2) around the points' mean
    e = np.stack(plane_basis(n))                          # (2, 3)
    anchor = pts.mean(0) if len(pts) else origin
    found = fit_lines((pts - anchor) @ e.T, np.random.default_rng(SEED))

    # diagnostic: which kept boundary pixels still lie on an object outline rather than on a desk edge
    masks = load_masks(clip)
    objects_px = masks.get(BODY, m, "depth") | masks.get(HANDLE, m, "depth") | masks.get(COASTER, m, "depth")
    kernel = np.ones((2 * NEAR_OBJECT_PX + 1, 2 * NEAR_OBJECT_PX + 1), np.uint8)
    near_object = cv2.dilate(objects_px.astype(np.uint8), kernel)[uv[:, 1], uv[:, 0]] > 0

    lines, line_px = [], []
    for centre, direction, idx in found:
        s = ((pts[idx] - anchor) @ e.T - centre) @ direction
        point, dir_w = anchor + centre @ e, direction @ e
        first, last = longest_run(s)
        lines.append({"point_world": point.tolist(), "dir_world": dir_w.tolist(), "n_inliers": len(idx),
                      "extent_m": last - first, "n_inliers_in_run": int(np.count_nonzero((s >= first) & (s <= last))),
                      "span_m": float(s.max() - s.min()), "largest_gap_m": float(np.diff(np.sort(s)).max()),
                      "n_inliers_near_objects": int(near_object[idx].sum()),
                      "ends_world": [(point + first * dir_w).tolist(), (point + last * dir_w).tolist()]})
        line_px.append(uv[idx])
    extents = [ln["extent_m"] for ln in lines]
    groups = group_lines([np.array(ln["dir_world"]) for ln in lines], extents)

    # step 3: long-edge group
    long_lines = [k for k, ext in enumerate(extents) if ext > LONG_EDGE_MIN_M]
    if long_lines:
        k_long = max(long_lines, key=lambda k: extents[k])
        long_group = next(g for g, grp in enumerate(groups) if k_long in grp["line_ids"])
        rule = "extent>0.42"
    elif groups:
        long_group = int(np.argmax([grp["total_extent_m"] for grp in groups]))
        rule = "largest_total_extent"                    # the > 0.42 m rule did not apply
    else:
        long_group, rule = None, None

    # steps 4-5: x, its sign, the reference cross-check
    ref_kind, ref = reference_direction(view, T_wc[m, :3, :3], rot_k, z)
    if long_group is not None:
        x = unit(np.cross(z, np.array(groups[long_group]["dir_world"])))
        method = "long_edge_extent" if rule == "extent>0.42" else "largest_total_extent"
    elif view in ("90deg", "top-down"):
        x, method = ref, "reference_fallback"
    else:
        x, method = None, "failed"

    valid = np.flatnonzero(hf["metric_valid"])
    palm_first = None
    if len(valid):
        kp = hf["kp3d"][valid[0]].astype(np.float64)
        palm_first = transform(T_wc[valid[0]], 0.5 * (kp[0] + kp[9]))
    flipped = None                                       # null = sign undetermined (no metric_valid frame)
    if x is not None and palm_first is not None:
        flipped = bool(x @ (palm_first - origin) > 0)
        x = -x if flipped else x

    reference = {"kind": ref_kind, "dir_world": ref.tolist(), "angle_deg": None,
                 "expected_deg": 45.0 if view == "45deg" else 0.0, "abs_err_deg": None}
    y = T = None
    if x is not None:
        y = np.cross(z, x)
        angle = float(np.degrees(np.arctan2(z @ np.cross(ref, x), ref @ x)))   # signed, from the reference to x about z
        reference["angle_deg"] = angle
        reference["abs_err_deg"] = abs(abs(angle) - 45.0) if view == "45deg" else abs(angle)
        R = np.column_stack([x, y, z])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, -R.T @ origin

    # step 6: drift of the S4 plane, measured at the initial mug position
    drift = {"offset_m": None, "normal_diff_deg": None, "flag": None}
    if table["end_plane_world"] is not None:
        n_end, d_end = np.array(table["end_plane_world"][:3]), table["end_plane_world"][3]
        offset = float(n_end @ origin + d_end)
        drift = {"offset_m": offset, "normal_diff_deg": table["normal_diff_deg"],
                 "flag": bool(abs(offset) > DRIFT_OFFSET_M or table["normal_diff_deg"] > DRIFT_NORMAL_DEG)}

    as_list = lambda a: None if a is None else a.tolist()
    # contour px = external contour without the border; boundary px = the ones kept for the lines; a dropped point can
    # be near both objects (the two drop discs can overlap), so the two per-object counts may add up to more than dropped
    result = {"mid_frame": m, "n_desk_px": int(desk.sum()), "n_contour_px": len(uv) + len(uv_dropped),
              "n_dropped_px": len(uv_dropped), "n_dropped_px_near_mug": int(near_mug.sum()),
              "n_dropped_px_near_coaster": int(near_coaster.sum()), "n_boundary_px": len(uv),
              "n_boundary_px_near_objects": int(near_object.sum()),
              "origin_world": origin.tolist(), "x_world": as_list(x), "y_world": as_list(y), "z_world": z.tolist(),
              "T_table_world": as_list(T), "lines": lines, "groups": groups, "long_edge_group": long_group,
              "rule_used": rule, "x_axis_method": method, "sign_flipped": flipped,
              "palm_first_index": int(valid[0]) if len(valid) else None,
              "palm_first_x_m": float(x @ (palm_first - origin)) if x is not None and palm_first is not None else None,
              "reference": reference, "drift": drift}
    (out_dir / "table_frame.json").write_text(json.dumps(result, indent=1))
    plot_lines(out_dir / "lines.png", clip, m, rot_k, meta, uv, pts, uv_dropped, pts_dropped, line_px, result)
    return {"x_axis_method": method, "rule_used": rule, "line_extents_m": extents,
            "line_spans_m": [ln["span_m"] for ln in lines], "line_largest_gaps_m": [ln["largest_gap_m"] for ln in lines],
            **{k: result[k] for k in ("n_contour_px", "n_dropped_px", "n_dropped_px_near_mug", "n_dropped_px_near_coaster",
                                      "n_boundary_px")},
            "line_inliers_near_objects": [ln["n_inliers_near_objects"] for ln in lines], "long_edge_group": long_group,
            "sign_flipped": flipped, "reference_angle_deg": reference["angle_deg"],
            "reference_abs_err_deg": reference["abs_err_deg"], "drift": drift}


def plot_lines(path, clip, m, rot_k, meta, uv, pts, uv_dropped, pts_dropped, line_px, r) -> None:
    """Left: frame m (upright) with the boundary pixels and each line's inliers. Right: the same, top-down on the table.
    The dropped boundary points (near the mug / coaster) are drawn in their own colour; each line is drawn over its
    extent (the longest gap-free run)."""
    fig = Figure(figsize=(13, 6.5))
    ax_img, ax_top = fig.subplots(1, 2)
    groups_of = {k: g for g, grp in enumerate(r["groups"]) for k in grp["line_ids"]}
    label = lambda k: (f"L{k} (group {groups_of[k]}) extent {r['lines'][k]['extent_m']:.2f} m "
                       f"(span {r['lines'][k]['span_m']:.2f} m, max gap {r['lines'][k]['largest_gap_m']:.2f} m), "
                       f"{r['lines'][k]['n_inliers_near_objects']} px near objects")
    dropped_label = f"dropped: within {OBJECT_MARGIN_M * 100:.0f} cm of the mug footprint / coaster disc"
    width = lambda k: 3.0 if groups_of[k] == r["long_edge_group"] else 1.2

    # image panel, in upright depth-pixel units (the RGB frame covers the same field of view as the depth map)
    depth_wh = tuple(int(s) for s in meta["depth_size"])
    up_w, up_h = depth_wh if rot_k % 2 == 0 else depth_wh[::-1]
    img = WORK / clip / "frames" / "full" / f"{m:06d}.jpg"
    if img.exists():
        ax_img.imshow(Image.open(img), extent=(-0.5, up_w - 0.5, up_h - 0.5, -0.5))
    ax_img.set_xlim(-0.5, up_w - 0.5)
    ax_img.set_ylim(up_h - 0.5, -0.5)
    b = pix_orig_to_upright(uv, depth_wh, rot_k)
    ax_img.plot(b[:, 0], b[:, 1], ".", ms=2, color="#d0cfc8", label="desk boundary")
    b = pix_orig_to_upright(uv_dropped, depth_wh, rot_k)
    ax_img.plot(b[:, 0], b[:, 1], "x", ms=3, color=DROPPED_COLOUR, label=dropped_label)
    for k, px in enumerate(line_px):
        q = pix_orig_to_upright(px, depth_wh, rot_k)
        ax_img.plot(q[:, 0], q[:, 1], ".", ms=2 + width(k), color=GROUP_COLOURS[groups_of[k]], label=label(k))
    ax_img.set_title(f"frame {m} (upright, depth pixels)")
    ax_img.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8)   # below: the near edge is at the bottom

    # top-down panel: table frame if x exists, else plane coordinates around the origin
    origin = np.array(r["origin_world"])
    if r["x_world"] is not None:
        a, c, names = np.array(r["x_world"]), np.array(r["y_world"]), ("table x (my forward) [m]", "table y [m]")
    else:
        a, c = plane_basis(np.array(r["z_world"]))
        names = ("plane e1 [m] (no x axis)", "plane e2 [m]")
    to2d = lambda p: np.stack([(np.asarray(p) - origin) @ a, (np.asarray(p) - origin) @ c], -1)
    q = to2d(pts)
    ax_top.plot(q[:, 0], q[:, 1], ".", ms=2, color="#b8b7b0", label="desk boundary on the plane")
    q = to2d(pts_dropped)
    ax_top.plot(q[:, 0], q[:, 1], "x", ms=3, color=DROPPED_COLOUR, label=dropped_label)
    for k, ln in enumerate(r["lines"]):
        ends = to2d(ln["ends_world"])
        ax_top.plot(ends[:, 0], ends[:, 1], "-", lw=width(k), color=GROUP_COLOURS[groups_of[k]], label=label(k))
        ax_top.annotate(f"L{k}", ends.mean(0), fontsize=9, color="#0b0b0b")
    ax_top.plot(0, 0, "o", color="#0b0b0b", ms=6, label="origin (initial mug base)")
    if r["x_world"] is not None:
        for v, name in ((np.array([0.1, 0.0]), "x"), (np.array([0.0, 0.1]), "y")):
            ax_top.annotate("", v, (0, 0), arrowprops={"arrowstyle": "->", "color": "#0b0b0b", "lw": 1.5})
            ax_top.annotate(name, v * 1.15, fontsize=10, color="#0b0b0b")
    ref = to2d(origin + 0.15 * np.array(r["reference"]["dir_world"]))
    ax_top.annotate("", ref, (0, 0), arrowprops={"arrowstyle": "->", "color": "#52514e", "lw": 1, "ls": "--"})
    ax_top.annotate(r["reference"]["kind"], ref, fontsize=8, color="#52514e")
    ax_top.set_aspect("equal")
    ax_top.set_xlabel(names[0])
    ax_top.set_ylabel(names[1])
    ax_top.grid(color="#e8e7e2", lw=0.5)
    ang = r["reference"]["angle_deg"]
    ax_top.set_title(f"rule {r['rule_used']}, x method {r['x_axis_method']}\n"
                     f"x vs {r['reference']['kind']} = {'n/a' if ang is None else f'{ang:.1f}'} deg")
    ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle(f"{clip} ({CLIPS[clip]}): desk-edge lines (thick = long-edge group)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    stage_cli("table_frame", run)
