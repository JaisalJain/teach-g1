"""S11 -- per-clip report and the M2 v2 gates; --summary builds the cross-clip tables.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.report --clip <id>
    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.report --summary

Per clip, into results/m2_perception_v2/<clip>/ (v1 in results/m2_perception/ is only read, for the v1 -> v2 table).
Inputs: data/demos/<clip>.npz (S10, schema 3) plus the stage files that hold numbers the demo file does not (event
indices, S8 d_axis / z, object-fit windows and free fit, S7 drift and reference, the fitted mesh of S5b, the MANO faces
of S2).
- overlay.mp4 (gitignored): the upright half-res frames of S1 (720x960) with, drawn by OpenCV: the S5b MANO mesh
  (depth-corrected, scaled by s_hand, shifted; translucent, flat-shaded, painter's order) and the 21-keypoint skeleton;
  the fitted mug cylinder (config r, h at the 2-DOF base centre, wireframe) during the initial and the final window;
  the coaster circle; the table axes at the origin (x red, y green, z blue, 10 cm); the smoothed palm frame (x_f red,
  y green, n blue, 5 cm); frame index, time and phase. Every 3D point goes through the true K of its frame into the
  ORIGINAL half-res image and is then mapped to the upright view with geometry.pix_orig_to_upright. ffmpeg encodes at
  the capture rate (1 / median dt); frames are not resampled, so a dropped frame just shortens the video by one frame.
- overlay.gif (the 90deg clip only, committed): the same drawings, 10 fps (first frame of every 0.1 s), 360 px wide,
  with the largest palette of 256 / 128 / 64 colours that keeps it under 5 MB (gif_colors, gif_under_5mb in done.json;
  a WARNING is printed if even 64 colours are over the limit).
- topdown.png, palm_height.png, depth_shift.png, and copies of the stage plots: mask_areas.png (S3), lines.png (S7),
  events.png (S8).
- gates.json: G1-G9 (the v2 table), each {"value", "text", "result": "pass" | "fail" | "N/A", "reason"}. A gate whose
  inputs are undefined (an event not detected, a stage output missing, no smoothed palm path for G4) fails and says
  why; N/A is used only where the spec allows it (G4 without a static hold, G6 height unobservable, G8 drift with a
  short end window), always with a reason. G1 reports every sub-check whose events exist, also when another one is
  undefined. G2 / G3 take the median over the metric_valid frames where the value exists: S5b's re-measured LiDAR
  residual is NaN on a frame where no hand pixel is left within 3 cm of the fitted mesh; those frames are counted in
  the text and listed by index in gates.json (n_undefined, undefined_frames). G5 also records g, contact and lift and
  says so in its text when g comes before contact (information). G7 and the cross-view coaster distance are horizontal
  distances between S6's world points (normal component along the S4 plane normal removed), so they do not need table
  x and stay defined when S7 found none; only dx, dy do. Also palm_normal_flip = S9's flip of the palm normal
  {flipped (null = not decided), decided, normal_dot_at_grasp}.
- crossview.json: this clip's row of the cross-view table.
- resources.json: seconds / peak RSS / peak VRAM of every earlier stage, from their done.json (the report's own numbers
  are in data/work/<clip>/report/done.json, written after run() returns).
- stages/: every stage's done.json and <stage>.json (e.g. mask_test.json, objects.json), so that every reported number
  is under results/.
--summary reads every clip's gates.json, crossview.json and stages/ (plus the v1 gates.json) and writes
results/m2_perception_v2/summary.json and summary.md: the v2 gate table; the hand scale (s_hand, each clip's median s
and IQR); a v1 -> v2 table of every gate per clip; the A4 mask test (both mask sources in both static windows, and the
decision S6 recorded, which uses the initial window only); the cross-view table (v1 columns + D_min, median s, mug mask
source) with the spread (min, max, max - min) of each column, and notes on S8's closest approach (D_min flagged above
the mug top, D_min after the lift, g before contact); the S9 palm-normal flip per clip with an explicit ERROR line when
the decided clips disagree (the spec requires the same flip in all clips); and the confound statement.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from perception.build_demo import arr, get, load_stage  # noqa: E402
from perception.common import CLIPS, DEMOS, RESULTS, RESULTS_V1, WORK, cfg, load_config, stage_cli  # noqa: E402
from perception.geometry import invert, pix_orig_to_upright, project, scale_K, transform  # noqa: E402
from perception.objects import H_BOUNDS, R_BOUNDS  # noqa: E402  (free-fit bounds, for the A4 table)

# M2 v2 gate thresholds (the spec's A6 table)
G1_COVERAGE, G1_MAX_RUN_S, G1_MARGIN_S = 0.90, 0.10, 0.5
G2_RESIDUAL_M = 0.015
G3_REPROJ_PX = 8.0
G4_SPEED_MPS, G4_HOLD_S, G4_STD_M = 0.02, 0.3, 0.01
G5_Z_FRAC = (0.25, 0.85)
G6_IOU, G6_TOL_M = 0.80, 0.005
G7_DIST_M = 0.025
G8_RMS_M, G8_OFFSET_M, G8_X_DEG = 0.006, 0.01, 20.0
G9_SCALE_TOL = 0.10
GATES = {
    "G1": "hand coverage: metric_valid >= 90 % from first hand_valid to release, no invalid run > 0.1 s in "
          "[contact - 0.5 s, place + 0.5 s], g and lift valid",
    "G2": "LiDAR residual, re-measured on the S5b hand: median over metric_valid < 1.5 cm",
    "G3": "palm-keypoint reprojection after S5b pass 2: median over metric_valid < 8 px (all-21 median = information)",
    "G4": "static jitter: in every hold (smoothed speed < 2 cm/s for >= 0.3 s) raw palm std < 1 cm (N/A if no hold)",
    "G5": "grasp height: z_grasp in [0.25 h, 0.85 h] (palm-to-axis distance at g and distance - r = information)",
    "G6": "mug fit on the A4-chosen mask source: initial IoU >= 0.80, free-fit diameter and height within 5 mm of the "
          "label (height N/A if unobservable)",
    "G7": "placement: horizontal distance final mug base centre to coaster centre < 2.5 cm",
    "G8": "table frame: plane RMS < 6 mm, drift plane offset at the initial mug position < 1 cm (N/A if the end window "
          "< 10 frames; normal-angle drift = information), x within 20 deg of the expected reference angle",
    "G9": "hand-scale consistency: this clip's median s within +-10 % of s_hand",
}
# what changed from v1, per gate (for the v1 -> v2 table); "inputs" = upstream stages that changed in v2
GATE_CHANGES = {
    "G1": "same rule; inputs changed (S5b hand, contact = D_min + 1 cm)",
    "G2": "residual re-measured on the S5b hand (v1: the S5 hand)",
    "G3": "5 palm keypoints after S5b pass 2 (v1: all 21 keypoints of the S5 hand, plumbing check)",
    "G4": "same rule; input = the palm path of the S5b hand",
    "G5": "z_grasp only (v1 also required the palm-to-axis distance in [r + 1, r + 5] cm); its input g moves with the "
          "relative contact rule (lift is searched after contact; g = last valid frame <= lift - 3)",
    "G6": "same thresholds, on the A4-chosen mask source (v1: video masks)",
    "G7": "same rule; S6 input follows the A4 mask source",
    "G8": "drift = plane offset at the initial mug < 1 cm (v1: failed if offset > 1 cm OR normal angle > 2 deg); "
          "desk-edge fix A3",
    "G9": "new in v2",
}
# cross-view columns: key, label, display scale, format
CROSSVIEW = [
    ("coaster_dx_m", "coaster offset from initial cup, dx [cm]", 100, ".1f"),
    ("coaster_dy_m", "coaster offset from initial cup, dy [cm]", 100, ".1f"),
    ("coaster_dist_m", "coaster offset from initial cup, distance [cm]", 100, ".1f"),
    ("approach_yaw_deg", "approach yaw at g, from the handle direction [deg]", 1, ".0f"),
    ("grasp_height_frac", "grasp height fraction z_grasp / h", 1, ".2f"),
    ("finger_pitch_deg", "finger-direction pitch at g [deg]", 1, ".0f"),
    ("palm_normal_elev_deg", "palm-normal elevation at g [deg]", 1, ".0f"),
    ("lift_height_m", "lift height (max z in [lift, place] - z_grasp) [cm]", 100, ".1f"),
    ("reach_duration_s", "reach duration (contact - first hand_valid) [s]", 1, ".2f"),
    ("transport_duration_s", "transport duration (place - lift) [s]", 1, ".2f"),
    ("init_handle_yaw_deg", "initial handle yaw from table x [deg]", 1, ".0f"),
    ("coaster_diameter_m", "fitted coaster diameter [mm]", 1000, ".1f"),
    ("long_edge_rule", "desk long-edge rule (S7 x_axis_method)", None, None),
    ("d_axis_min_m", "D_min: closest horizontal palm-to-initial-cup-axis distance [mm]", 1000, ".1f"),
    ("median_s", "hand scale: median pass-1 s over metric_valid frames", 1, ".3f"),
    ("mug_mask_source", "mug mask source (S6, A4 decision)", None, None),
]
CONFOUND = "Viewpoint and execution vary together across these clips (views: {views}); n = {n}."

STAGES = ["frames", "hand", "masks", "table", "hand_metric", "hand_scale", "masks_full", "mask_test", "hand_fit",
          "objects", "table_frame", "events", "palm", "demo"]
COPIED_PLOTS = {"masks": "mask_areas.png", "table_frame": "lines.png", "events": "events.png"}
EVENTS = ("contact", "grasp", "lift", "place", "release")
PHASES = {-1: "undefined", 0: "approach", 1: "grasp", 2: "transport", 3: "place", 4: "release/retreat"}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_BGR = [(60, 60, 255), (0, 200, 255), (60, 220, 60), (255, 160, 0), (255, 60, 200)]   # thumb .. pinky
RED, GREEN, BLUE = (0, 0, 255), (0, 200, 0), (255, 80, 0)
GIF_WIDTH, GIF_FPS, GIF_MAX_MB, GIF_COLORS = 360, 10, 5.0, (256, 128, 64)


def num(x) -> float:
    return np.nan if x is None else float(x)


def clean(x):
    """numpy -> plain Python for JSON; NaN -> null."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) index of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def rms_dev(p: np.ndarray) -> float:
    """3D RMS deviation from the mean = sqrt(trace(cov)) of (M, 3) points."""
    return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1)))) if len(p) else np.nan


def gate(value: dict, text: str, ok, reason: str | None = None) -> dict:
    """ok: True / False, or None for N/A (then reason says why)."""
    return {"value": value, "text": text, "result": {True: "pass", False: "fail", None: "N/A"}[ok],
            "reason": None if ok else reason}


def undefined(**inputs) -> str | None:
    """'undefined: a, b' for the inputs that are NaN, or None when all are defined."""
    bad = [name for name, v in inputs.items() if np.isnan(v)]
    return "undefined: " + ", ".join(bad) if bad else None


def median_defined(x: np.ndarray, mv: np.ndarray) -> tuple[float, list[int]]:
    """Median of x over the metric_valid frames where x is finite, and the metric_valid frames that have no value."""
    ok = mv & np.isfinite(x)
    return (float(np.median(x[ok])) if ok.any() else np.nan), np.flatnonzero(mv & ~ok).tolist()


def clip_scale(demo) -> dict:
    """This clip's pass-1 hand scale s (S5b) over its metric_valid frames: median, IQR [q25, q75], frame count."""
    s = demo["hand_scale_per_frame"][demo["metric_valid"]]
    s = s[np.isfinite(s)]
    if not len(s):
        return {"median_s": np.nan, "s_iqr": [np.nan, np.nan], "n_frames": 0}
    return {"median_s": float(np.median(s)), "s_iqr": np.percentile(s, [25, 75]).tolist(), "n_frames": len(s)}


def horizontal_dist(a_world, b_world, plane) -> float:
    """Distance between two world points (S6) with the component along the table normal (S4) removed. It needs no table
    x, so it stays defined when S7 found none (then every table-frame position in the demo file is NaN)."""
    n = plane[:3] / np.linalg.norm(plane[:3])
    d = arr(a_world, 3) - arr(b_world, 3)
    return float(np.linalg.norm(d - (d @ n) * n))


# ---------------------------------------------------------------------------------------------------------- gates
def gates(demo, idx, evz, ev, objs, tf, c) -> dict:
    t, hv, mv = demo["t"], demo["hand_valid"], demo["metric_valid"]
    dt = float(np.median(np.diff(t)))
    r, h = cfg(c, "mug", "body_outer_diameter_m") / 2, cfg(c, "mug", "height_m")
    out = {}

    # G1 hand coverage: each of the four checks is computed when its events exist; the gate needs all four
    not_found = [e for e in EVENTS if idx[e] < 0] + ([] if hv.any() else ["hand_valid (no frame)"])
    cov = float(mv[np.argmax(hv): idx["release"] + 1].mean()) if hv.any() and idx["release"] >= 0 else np.nan
    longest, longest_n = np.nan, None                  # duration [s] and frame count of the longest invalid run
    if idx["contact"] >= 0 and idx["place"] >= 0:
        window = (t >= t[idx["contact"]] - G1_MARGIN_S) & (t <= t[idx["place"]] + G1_MARGIN_S)
        longest, longest_n = max([(t[b] - t[a] + dt, int(b - a + 1)) for a, b in runs(window & ~mv)], default=(0.0, 0))
    g_ok = bool(mv[idx["grasp"]]) if idx["grasp"] >= 0 else None
    lift_ok = bool(mv[idx["lift"]]) if idx["lift"] >= 0 else None
    show = lambda v: "undefined" if v is None else v
    out["G1"] = gate({"coverage": cov, "longest_invalid_run_s": longest, "longest_invalid_run_frames": longest_n,
                      "grasp_valid": g_ok, "lift_valid": lift_ok},
                     (f"coverage {cov:.1%}" if np.isfinite(cov) else "coverage undefined") + ", " +
                     (f"longest invalid run {longest:.3f} s ({longest_n} frames)" if np.isfinite(longest)
                      else "longest invalid run undefined") + f", g valid {show(g_ok)}, lift valid {show(lift_ok)}",
                     bool(not not_found and cov >= G1_COVERAGE and longest <= G1_MAX_RUN_S and g_ok and lift_ok),
                     "undefined: " + ", ".join(not_found) if not_found else None)

    # G2 / G3: medians over the metric_valid frames that have a value (module doc)
    no_frames = "no metric_valid frame" if not mv.any() else None
    res, res_nan = median_defined(demo["lidar_residual_m"], mv)
    out["G2"] = gate({"median": res, "n_frames": int(mv.sum()), "n_undefined": len(res_nan),
                      "undefined_frames": res_nan},
                     f"median {res * 100:.2f} cm over {mv.sum() - len(res_nan)} metric_valid frames"
                     + (f" ({len(res_nan)} more without a residual: frames {res_nan})" if res_nan else ""),
                     bool(res < G2_RESIDUAL_M), no_frames or undefined(median=res))
    palm_px, palm_nan = median_defined(demo["reproj_err_palm_px"], mv)
    all21_px, _ = median_defined(demo["reproj_err_all_px"], mv)
    out["G3"] = gate({"median": palm_px, "median_all21_px": all21_px, "n_frames": int(mv.sum()),
                      "n_undefined": len(palm_nan), "undefined_frames": palm_nan},
                     f"palm median {palm_px:.2f} px over {mv.sum() - len(palm_nan)} metric_valid frames"
                     + (f" ({len(palm_nan)} more without a value: frames {palm_nan})" if palm_nan else "")
                     + f"; all 21 keypoints {all21_px:.2f} px (information)",
                     bool(palm_px < G3_REPROJ_PX), no_frames or undefined(median=palm_px))

    # G4 static jitter: holds in the smoothed palm path (adjacent valid frames only)
    p, p_raw = demo["palm_pos_table"], demo["palm_pos_table_raw"]
    valid = np.isfinite(p).all(1)
    step = valid[:-1] & valid[1:]
    speed = np.full(len(t) - 1, np.inf)
    speed[step] = np.linalg.norm(np.diff(p, axis=0)[step], axis=1) / np.diff(t)[step]
    holds = []
    for a, b in runs(speed < G4_SPEED_MPS):            # slow steps a..b = frames a..b+1
        if t[b + 1] - t[a] >= G4_HOLD_S:
            raw = p_raw[a: b + 2][np.isfinite(p_raw[a: b + 2]).all(1)]
            holds.append({"frames": [int(a), int(b + 1)], "start_s": float(t[a] - t[0]),
                          "duration_s": float(t[b + 1] - t[a]), "std_raw_m": rms_dev(raw),
                          "std_smoothed_m": rms_dev(p[a: b + 2]), "n_raw": len(raw)})
    if not valid.any():                                # no path at all: an undefined input, not "no hold"
        out["G4"] = gate({"holds": []}, "no smoothed palm path", False,
                         "no smoothed palm path (S9 missing, or S7 found no table x)")
    elif not holds:
        out["G4"] = gate({"holds": []}, "no static hold", None,
                         "N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s")
    else:
        worst_raw, worst_sm = max(x["std_raw_m"] for x in holds), max(x["std_smoothed_m"] for x in holds)
        out["G4"] = gate({"holds": holds, "max_std_raw_m": worst_raw, "max_std_smoothed_m": worst_sm},
                         f"{len(holds)} hold(s), max raw std {worst_raw * 100:.2f} cm (smoothed {worst_sm * 100:.2f} cm)",
                         all(x["std_raw_m"] < G4_STD_M for x in holds))

    # G5 grasp height at g (S8's raw-palm z_grasp; h from the config). The palm-to-axis distance d at g (S8's d_axis)
    # and d - r are reported, not gated. Also information: g (<= lift - 3) is not forced after contact (S8), so when
    # contact comes late (D_min after the lift) g can precede it; the text then says so, since z_grasp is not taken
    # in the grasp phase.
    g = idx["grasp"]
    d = float(evz["d_axis"][g]) if g >= 0 and evz is not None else np.nan
    z = num(get(ev, "z_grasp_m"))
    z_lo, z_hi = G5_Z_FRAC[0] * h, G5_Z_FRAC[1] * h
    early = g >= 0 and idx["contact"] >= 0 and g < idx["contact"]
    out["G5"] = gate({"z_grasp_m": z, "z_range_m": [z_lo, z_hi], "d_axis_m": d, "d_minus_r_m": d - r,
                      "grasp_index": g, "contact_index": idx["contact"], "lift_index": idx["lift"],
                      "grasp_before_contact": early},
                     f"z_grasp {z * 1000:.1f} mm (pass {z_lo * 1000:.1f}-{z_hi * 1000:.1f}); information: palm-to-axis "
                     f"distance {d * 1000:.1f} mm, distance - r {(d - r) * 1000:.1f} mm"
                     + (f"; NOTE: g {g} comes before contact {idx['contact']} (lift {idx['lift']})" if early else ""),
                     bool(z_lo <= z <= z_hi), "grasp frame g undefined" if g < 0 else undefined(z_grasp=z))

    # G6 mug fit (initial window, on the mask source S6 used; label values from the config)
    src = get(objs, "mug_mask_source") or "missing"
    iou = num(get(objs, "init", "iou"))
    diam = 2 * num(get(objs, "init", "free_fit", "r_m"))
    diam_label, h_label = cfg(c, "mug", "body_outer_diameter_m"), cfg(c, "mug", "height_m")
    if get(objs, "init", "free_fit", "h_observable"):
        h_free = num(get(objs, "init", "free_fit", "h_m"))
        h_ok, h_text = bool(abs(h_free - h_label) <= G6_TOL_M), f"{h_free * 1000:.1f} mm (label {h_label * 1000:.0f})"
    else:
        h_free, h_ok, h_text = np.nan, None, "N/A (unobservable from this view)"
    iou_ok, diam_ok = bool(iou >= G6_IOU), bool(abs(diam - diam_label) <= G6_TOL_M)
    out["G6"] = gate({"mask_source": src, "init_iou": iou, "free_fit_diameter_m": diam, "free_fit_height_m": h_free,
                      "height_check": h_ok},
                     f"{src} masks: IoU {iou:.3f}, diameter {diam * 1000:.1f} mm (label {diam_label * 1000:.0f}), "
                     f"height {h_text}",
                     iou_ok and diam_ok and h_ok is not False, undefined(init_iou=iou, free_fit_diameter=diam))

    # G7 placement (world points, horizontal part: needs no table x)
    dist = horizontal_dist(get(objs, "final", "base_centre_world"), get(objs, "coaster", "centre_world"),
                           demo["table_plane_world"])
    out["G7"] = gate({"distance_m": dist}, f"{dist * 100:.2f} cm", bool(dist < G7_DIST_M), undefined(distance=dist))

    # G8 table frame. Drift: S7's signed offset of the end-window plane at the initial mug position, null when S4 had
    # no end-window plane (end window < 10 frames) -> N/A. The normal angle between the two planes is information.
    rms = float(demo["table_rms_m"])
    offset, ndiff = num(get(tf, "drift", "offset_m")), num(get(tf, "drift", "normal_diff_deg"))
    drift_ok = None if np.isnan(offset) else bool(abs(offset) < G8_OFFSET_M)
    if tf is None:
        drift_text = "not available (no S7 output)"
    elif drift_ok is None:
        drift_text = "N/A (end window < 10 frames)"
    else:
        drift_text = (f"offset {offset * 1000:.1f} mm ({'pass' if drift_ok else 'fail'}), "
                      f"normal {ndiff:.2f} deg (information)")
    method, err = str(demo["x_axis_method"]), num(get(tf, "reference", "abs_err_deg"))
    x_ok = method not in ("failed", "missing") and bool(err <= G8_X_DEG)
    out["G8"] = gate({"plane_rms_m": rms, "drift_offset_m": offset, "drift_offset_check": drift_ok,
                      "drift_normal_diff_deg": ndiff, "x_axis_method": method,
                      "x_vs_reference_deg": float(demo["x_axis_vs_reference_deg"]),
                      "expected_deg": num(get(tf, "reference", "expected_deg")), "x_abs_err_deg": err},
                     f"plane RMS {rms * 1000:.2f} mm, drift {drift_text}, x ({method}) error "
                     + (f"{err:.1f} deg" if np.isfinite(err) else "undefined"),
                     bool(rms < G8_RMS_M) and drift_ok is not False and x_ok,
                     f"x_axis_method = {method}" if method in ("failed", "missing") else
                     undefined(plane_rms=rms, x_abs_err_deg=err))

    # G9 hand-scale consistency: this clip's median pass-1 s against s_hand (pooled over the 4 clips)
    sc, s_hand = clip_scale(demo), float(demo["s_hand"])
    ratio = sc["median_s"] / s_hand
    q25, q75 = sc["s_iqr"]
    out["G9"] = gate({**sc, "s_hand": s_hand, "ratio_to_s_hand": ratio},
                     f"median s {sc['median_s']:.3f} (IQR {q25:.3f}-{q75:.3f}, {sc['n_frames']} frames) vs s_hand "
                     f"{s_hand:.3f}: {(ratio - 1) * 100:+.1f} %",
                     bool(abs(ratio - 1) <= G9_SCALE_TOL), undefined(median_s=sc["median_s"], s_hand=s_hand))
    return out


def crossview(demo, idx, evz, ev, objs, c) -> dict:
    t = demo["t"]
    h = cfg(c, "mug", "height_m")
    at = lambda e: t[idx[e]] if idx[e] >= 0 else np.nan
    off = demo["coaster_pos_table"][:2] - demo["cup_init_pos_table"][:2]       # dx, dy need table x; the distance not
    T = demo["T_palm_in_cup_at_grasp"]            # cup frame: z up, x = initial handle direction (if the yaw is valid)
    yaw_ok = bool(demo["cup_yaw_valid"][0])
    z_grasp = num(get(ev, "z_grasp_m"))
    lift_z = np.nan
    if idx["lift"] >= 0 and idx["place"] >= 0:
        z = evz["z"][idx["lift"]: idx["place"] + 1]          # S8 raw palm z, NaN where not metric_valid
        lift_z = np.nanmax(z) if np.isfinite(z).any() else np.nan
    first_hand = t[np.argmax(demo["hand_valid"])] if demo["hand_valid"].any() else np.nan
    scale = clip_scale(demo)
    return {
        "coaster_dx_m": off[0], "coaster_dy_m": off[1],
        "coaster_dist_m": horizontal_dist(get(objs, "coaster", "centre_world"), get(objs, "init", "base_centre_world"),
                                          demo["table_plane_world"]),
        "approach_yaw_deg": np.degrees(np.arctan2(T[1, 3], T[0, 3])) if yaw_ok else np.nan,
        "grasp_height_frac": z_grasp / h,
        "finger_pitch_deg": np.degrees(np.arcsin(np.clip(T[2, 0], -1, 1))),     # x_f above horizontal
        "palm_normal_elev_deg": np.degrees(np.arcsin(np.clip(T[2, 2], -1, 1))),  # n above horizontal
        "lift_height_m": lift_z - z_grasp,
        "reach_duration_s": at("contact") - first_hand,
        "transport_duration_s": at("place") - at("lift"),
        "init_handle_yaw_deg": np.degrees(demo["cup_init_yaw"]) if yaw_ok else np.nan,
        "coaster_diameter_m": float(demo["coaster_diameter_fit_m"]),
        "long_edge_rule": str(demo["x_axis_method"]),
        "d_axis_min_m": float(demo["d_axis_min_m"]),
        "median_s": scale["median_s"],
        "mug_mask_source": str(demo["mug_mask_source"]),
        # not columns: kept with the row for the notes under the table
        "handle_yaw_valid": yaw_ok,
        "d_axis_min_index": int(demo["d_axis_min_index"]),
        "d_min_z_flag": get(ev, "d_min_z_flag"),
        "event_index": {e: idx[e] for e in ("contact", "grasp", "lift")},
        "s_iqr": scale["s_iqr"],
    }


# ---------------------------------------------------------------------------------------------------------- overlay
def to_upright(X_cam: np.ndarray, K_half: np.ndarray, k: int, half_wh) -> np.ndarray:
    """(..., 3) ORIGINAL camera-frame points -> (..., 2) pixels of the upright half-res frame (NaN behind the camera)."""
    X = np.asarray(X_cam, dtype=np.float64)
    uv = project(K_half, X)
    uv[X[..., 2] < 0.05] = np.nan
    return pix_orig_to_upright(uv, half_wh, k)


def polyline(img, uv, color, thick=2, closed=False):
    if np.isfinite(uv).all():
        cv2.polylines(img, [np.round(uv).astype(np.int32)], closed, color, thick, cv2.LINE_AA)


def draw_mesh(img, verts_cam, faces, uv, color=(255, 190, 90), alpha=0.5):
    """Translucent flat-shaded mesh: far triangles first (painter's order), blended over the frame."""
    tri, tri_uv = verts_cam[faces], uv[faces]
    if not np.isfinite(tri_uv).all():
        return
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    view = tri.mean(1)
    cos = np.abs(np.sum(normal * view, 1)) / (np.linalg.norm(normal, axis=1) * np.linalg.norm(view, axis=1) + 1e-12)
    pts = np.round(tri_uv).astype(np.int32)
    layer, mask = img.copy(), np.zeros(img.shape[:2], np.uint8)
    for f in np.argsort(-view[:, 2]):
        cv2.fillConvexPoly(layer, pts[f], tuple(float(v) * (0.35 + 0.65 * cos[f]) for v in color))
        # the mask face by face too: one cv2.fillPoly over all faces fills even-odd and drops the pixels where front
        # and back faces overlap
        cv2.fillConvexPoly(mask, pts[f], 255)
    sel = mask > 0
    img[sel] = ((1 - alpha) * img[sel] + alpha * layer[sel]).astype(np.uint8)


def ring(centre, radius, z, n=48) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), np.full(n, centre[2] + z)])


def draw_frame(img, i, demo, verts, faces, windows, dims, k, half_wh):
    """Draw everything for canonical frame i onto img (upright half-res BGR, in place)."""
    K_half = scale_K(demo["K"][i], 0.5)
    T_ct = invert(demo["T_world_cam"][i]) @ invert(demo["T_table_world"])      # table -> camera
    cam = lambda X: to_upright(X, K_half, k, half_wh)
    table = lambda X: cam(transform(T_ct, X))

    if verts is not None and faces is not None and np.isfinite(verts[i]).all():
        draw_mesh(img, verts[i].astype(np.float64), faces, cam(verts[i]))
    kp = cam(demo["hand_kp_cam"][i])
    for b, (u, v) in enumerate(BONES):
        polyline(img, kp[[u, v]], FINGER_BGR[b // 4], 2)

    r, h, coaster_z = dims
    for w, key in (("init", "cup_init_pos_table"), ("final", "cup_final_pos_table")):
        if windows.get(w) and windows[w][0] <= i <= windows[w][1]:
            bottom, top = ring(demo[key], r, 0.0), ring(demo[key], r, h)
            polyline(img, table(bottom), (0, 255, 255), 2, True)
            polyline(img, table(top), (0, 255, 255), 2, True)
            for j in range(0, len(bottom), len(bottom) // 8):
                polyline(img, table(np.stack([bottom[j], top[j]])), (0, 255, 255), 1)
    polyline(img, table(ring(demo["coaster_pos_table"], float(demo["coaster_diameter_fit_m"]) / 2, coaster_z)),
             (255, 0, 255), 2, True)

    for axis, col in zip(np.eye(3), (RED, GREEN, BLUE)):
        polyline(img, table(np.stack([np.zeros(3), 0.10 * axis])), col, 3)
    p, R = demo["palm_pos_table"][i], demo["palm_rot_table"][i]
    for j, col in enumerate((RED, GREEN, BLUE)):
        polyline(img, table(np.stack([p, p + 0.05 * R[:, j]])), col, 2)

    mv = "metric" if demo["metric_valid"][i] else ("hand" if demo["hand_valid"][i] else "no hand")
    text = f"i {i}  t {demo['t'][i] - demo['t'][0]:.2f} s  {PHASES[int(demo['phase'][i])]}  [{mv}]"
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_video(clip, demo, verts, faces, windows, dims, k, out) -> dict:
    t = demo["t"]
    rate = round(1 / float(np.median(np.diff(t))))
    first = cv2.imread(str(WORK / clip / "frames" / "sam2" / "00000.jpg"))
    hh, ww = first.shape[:2]
    half_wh = (hh, ww) if k % 2 else (ww, hh)          # ORIGINAL half-res (W, H)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ww}x{hh}",
                            "-r", str(rate), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "23", str(out / "overlay.mp4")],
                           stdin=subprocess.PIPE)
    gif, last_bin = [], -1
    make_gif = CLIPS[clip] == "90deg"
    for i in range(1, len(t)):
        img = cv2.imread(str(WORK / clip / "frames" / "sam2" / f"{i - 1:05d}.jpg"))
        draw_frame(img, i, demo, verts, faces, windows, dims, k, half_wh)
        enc.stdin.write(img.tobytes())
        b = int((t[i] - t[1]) * GIF_FPS)
        if make_gif and b != last_bin:                 # first frame of every 0.1 s
            small = cv2.resize(img, (GIF_WIDTH, round(hh * GIF_WIDTH / ww)), interpolation=cv2.INTER_AREA)
            gif.append(Image.fromarray(small[..., ::-1]))
            last_bin = b
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode overlay.mp4")
    stats = {"overlay_frames": len(t) - 1, "overlay_fps": rate, "overlay_size_wh": [ww, hh]}
    if make_gif:
        # A real 9 s clip at 360 px / 10 fps is ~9 MB with 256 colours (camera texture compresses badly in a GIF), so take
        # the largest palette that stays under 5 MB. Octree quantisation keeps the green mug green at 64 colours.
        for colors in GIF_COLORS:
            pal = [f.quantize(colors, method=Image.Quantize.FASTOCTREE) for f in gif]
            pal[0].save(out / "overlay.gif", save_all=True, append_images=pal[1:], duration=1000 // GIF_FPS, loop=0,
                        optimize=True)
            mb = (out / "overlay.gif").stat().st_size / 1e6
            if mb < GIF_MAX_MB:
                break
        else:
            print(f"WARNING: overlay.gif is {mb:.1f} MB even with {colors} colours (limit {GIF_MAX_MB} MB)",
                  file=sys.stderr)
        stats.update({"gif_frames": len(gif), "gif_colors": colors, "gif_mb": round(mb, 2),
                      "gif_under_5mb": mb < GIF_MAX_MB})
    return stats


# ---------------------------------------------------------------------------------------------------------- plots
def plots(clip, demo, idx, c, out):
    t = demo["t"] - demo["t"][0]
    p, p_raw = demo["palm_pos_table"], demo["palm_pos_table_raw"]
    r = cfg(c, "mug", "body_outer_diameter_m") / 2
    marks = [(e, idx[e]) for e in EVENTS if idx[e] >= 0]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(p_raw[:, 0], p_raw[:, 1], ".", ms=2, color="0.6", label="palm raw")
    ax.plot(p[:, 0], p[:, 1], "-", color="C0", lw=1.5, label="palm smoothed")
    for n, (e, i) in enumerate(marks):
        ax.plot(p[i, 0], p[i, 1], "o", color=f"C{n + 3}", label=e)
    handle = r + cfg(c, "mug", "handle_protrusion_m")
    # Arrow ends go into the data limits below: autoscale ignores annotations, and matplotlib does not draw an arrow
    # whose tip lies outside the axes.
    ends = [(0.0, 0.0)]
    for key, yaw_key, valid, col, name in (("cup_init_pos_table", "cup_init_yaw", 0, "C1", "cup initial"),
                                           ("cup_final_pos_table", "cup_final_yaw", 1, "C2", "cup final")):
        xy = demo[key][:2]
        ax.add_patch(plt.Circle(xy, r, fill=False, color=col, lw=2, label=name))
        if demo["cup_yaw_valid"][valid]:
            yaw = float(demo[yaw_key])
            tip = xy + handle * np.array([np.cos(yaw), np.sin(yaw)])
            ax.annotate("", tip, xy, arrowprops={"arrowstyle": "->", "color": col, "lw": 2})
            ax.plot([], [], color=col, lw=2, label=f"{name}: handle direction (arrow)")
            ends.append(tip)
        else:
            ax.plot([], [], " ", label=f"{name}: handle yaw invalid, not drawn")
    ax.add_patch(plt.Circle(demo["coaster_pos_table"][:2], float(demo["coaster_diameter_fit_m"]) / 2, fill=False,
                            color="m", ls="--", label="coaster (fit)"))
    for xy, name, col in (((0.08, 0), "x", "r"), ((0, 0.08), "y", "g")):    # table axes at the origin
        ax.annotate("", xy, (0, 0), arrowprops={"arrowstyle": "->", "color": col})
        ax.text(xy[0] * 1.15, xy[1] * 1.15, name, color=col)
        ends.append((xy[0] * 1.15, xy[1] * 1.15))
    ends = np.array(ends, dtype=np.float64)
    ax.update_datalim(ends[np.isfinite(ends).all(1)])
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("table x [m] (forward)")
    ax.set_ylabel("table y [m]")
    ax.set_title(f"{clip} ({CLIPS[clip]}): top-down, table frame")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "topdown.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, p_raw[:, 2] * 100, ".", ms=2, color="0.6", label="raw")
    ax.plot(t, p[:, 2] * 100, "-", color="C0", label="smoothed")
    for n, (e, i) in enumerate(marks):
        ax.axvline(t[i], color=f"C{n + 3}", ls="--", lw=1, label=e)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("palm height above table [cm]")
    ax.set_title(f"{clip}: palm height")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "palm_height.png", dpi=110)
    plt.close(fig)

    mv = demo["metric_valid"]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, key, name in zip(axs, ("depth_shift_m", "lidar_residual_m"),
                             ("S5 depth shift D", "LiDAR residual\n(S5b hand)")):
        y = demo[key] * 100
        ax.plot(t[~mv], y[~mv], ".", ms=3, color="0.6", label="not metric_valid")
        ax.plot(t[mv], y[mv], ".", ms=3, color="C0", label="metric_valid")
        ax.set_ylabel(f"{name} [cm]")
        ax.legend(fontsize=7)
    axs[1].set_xlabel("time [s]")
    axs[0].set_title(f"{clip}: S5 LiDAR depth correction; residual re-measured on the S5b hand")
    fig.tight_layout()
    fig.savefig(out / "depth_shift.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- stage
def run(clip: str) -> dict:
    out = RESULTS / clip
    out.mkdir(parents=True, exist_ok=True)
    c = load_config()
    with np.load(DEMOS / f"{clip}.npz") as f:
        demo = {key: f[key] for key in f.files}
    k = int(load_stage(clip, "frames", "meta.npz")["rot_k"])
    ev, evz = load_stage(clip, "events", "events.json"), load_stage(clip, "events", "events.npz")
    objs, tf = load_stage(clip, "objects", "objects.json"), load_stage(clip, "table_frame", "table_frame.json")
    hf, hand = load_stage(clip, "hand_fit", "hand_fit.npz"), load_stage(clip, "hand", "hand.npz")   # S5b mesh
    palm = load_stage(clip, "palm", "palm.npz")
    idx = {e: -1 if get(ev, e, "index") is None else int(get(ev, e, "index")) for e in EVENTS}

    g = gates(demo, idx, evz, ev, objs, tf, c)
    # S9's palm-normal flip, for the across-clip check in --summary (the demo file's palm_normal_flipped is also False
    # when the flip was not decided, so flip_decided comes from palm.npz)
    decided = bool(get(palm, "flip_decided"))
    flip = {"flipped": bool(get(palm, "normal_flipped")) if decided else None, "decided": decided,
            "normal_dot_at_grasp": num(get(palm, "normal_dot_at_grasp"))}
    (out / "gates.json").write_text(json.dumps(clean({"clip": clip, "view": CLIPS[clip], "gates": g,
                                                      "palm_normal_flip": flip}), indent=1))
    (out / "crossview.json").write_text(json.dumps(clean(crossview(demo, idx, evz, ev, objs, c)), indent=1))

    resources = {}
    for s in STAGES:
        done = load_stage(clip, s, "done.json")
        resources[s] = None if done is None else {key: done.get(key) for key in ("seconds", "peak_rss_gb", "peak_vram_gb")}
    (out / "resources.json").write_text(json.dumps(resources, indent=1))

    # the stage outputs behind every reported number, copied next to them (data/work/ is not committed)
    (out / "stages").mkdir(exist_ok=True)
    for s in STAGES:
        if (WORK / clip / s / "done.json").exists():
            shutil.copy(WORK / clip / s / "done.json", out / "stages" / f"{s}_done.json")
        if (WORK / clip / s / f"{s}.json").exists():                    # table.json, objects.json, events.json, ...
            shutil.copy(WORK / clip / s / f"{s}.json", out / "stages" / f"{s}.json")

    copied = []
    for stage, name in COPIED_PLOTS.items():
        if (WORK / clip / stage / name).exists():
            shutil.copy(WORK / clip / stage / name, out / name)
            copied.append(name)
    plots(clip, demo, idx, c, out)
    windows = get(objs, "windows") or {}
    dims = (cfg(c, "mug", "body_outer_diameter_m") / 2, cfg(c, "mug", "height_m"), cfg(c, "coaster", "thickness_m"))
    video = overlay_video(clip, demo, get(hf, "verts"), get(hand, "faces"), windows, dims, k, out)
    return {"out_dir": str(out), "gates": {name: x["result"] for name, x in g.items()},
            "copied_plots": copied, "missing_plots": [n for n in COPIED_PLOTS.values() if n not in copied], **video}


# ---------------------------------------------------------------------------------------------------------- summary
def fmt(v, scale, spec) -> str:
    if v is None:
        return "n/a"
    return str(v) if spec is None else format(v * scale, spec)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def at_bound(v, bounds) -> str:
    """' (fit bound)' when a free-fit value sits on a bound of objects.py's validation fit (within 0.1 mm), else ''."""
    return " (fit bound)" if v is not None and min(abs(v - b) for b in bounds) < 1e-4 else ""


def gate_cell(x) -> str:
    """One gate of one clip as a markdown cell: result, text and (if any) the reason."""
    if x is None:
        return "-"
    return f"**{x['result']}** {x['text']}" + (f" ({x['reason']})" if x["reason"] else "")


def summary():
    rows = {}
    for clip, view in CLIPS.items():
        d = RESULTS / clip
        if (d / "gates.json").exists() and (d / "crossview.json").exists():
            gj = json.loads((d / "gates.json").read_text())
            rows[clip] = {"view": view, "gates": gj["gates"], "palm_normal_flip": gj.get("palm_normal_flip"),
                          "crossview": json.loads((d / "crossview.json").read_text())}
    missing = [clip for clip in CLIPS if clip not in rows]
    confound = CONFOUND.format(views=", ".join(r["view"] for r in rows.values()), n=len(rows))
    c = load_config()
    diam_label, h_label = cfg(c, "mug", "body_outer_diameter_m"), cfg(c, "mug", "height_m")

    # S9: the palm-normal flip must be the same in every clip (spec); compare the clips where it was decided
    flips = {clip: (r["palm_normal_flip"] or {}).get("flipped") for clip, r in rows.items()}
    undecided = [clip for clip, f in flips.items() if f is None]
    flip_error = None
    if len({f for f in flips.values() if f is not None}) > 1:
        flip_error = ("ERROR: the palm-normal flip differs between clips (flipped: "
                      + ", ".join(cl for cl, f in flips.items() if f is True) + "; not flipped: "
                      + ", ".join(cl for cl, f in flips.items() if f is False) + ")")
        print(flip_error, file=sys.stderr)
    flip_line = (flip_error or "The palm-normal flip is the same in every clip where it was decided") + \
        f". Not decided (grasp frame undefined or no S9 output): {', '.join(undecided) or 'none'}."

    # hand scale (S5b): s_hand is one number for all clips (pooled in hand_fit); each clip's median s is G9's value
    scale = {clip: {key: r["gates"]["G9"]["value"].get(key) for key in ("median_s", "s_iqr", "n_frames", "s_hand",
                                                                       "ratio_to_s_hand")} for clip, r in rows.items()}
    s_hands = sorted({x["s_hand"] for x in scale.values() if x["s_hand"] is not None})
    if len(s_hands) == 1:
        s_hand_line = f"s_hand = {s_hands[0]:.4f}"
    elif s_hands:
        s_hand_line = "**ERROR: the demo files disagree on s_hand (" + ", ".join(f"{v:.4f}" for v in s_hands) + ")**"
    else:
        s_hand_line = "s_hand undefined (no hand_fit output)"

    # v1 -> v2: every gate per clip (v1 from results/m2_perception/<clip>/gates.json; G9 is new in v2)
    keep = lambda x: None if x is None else {k: x[k] for k in ("result", "text", "value", "reason")}
    v1_to_v2 = {}
    for clip, r in rows.items():
        v1 = (read_json(RESULTS_V1 / clip / "gates.json") or {}).get("gates", {})
        v1_to_v2[clip] = {name: {"v1": keep(v1.get(name)), "v2": keep(r["gates"][name])} for name in GATES}

    # A4: both mask sources in both static windows (stages/mask_test.json) and the decision S6 recorded (objects.json),
    # which uses the initial window only
    mask_test, decision = {}, None
    for clip in rows:
        mt = read_json(RESULTS / clip / "stages" / "mask_test.json")
        mask_test[clip] = None if mt is None else {
            src: {w: {"iou": get(mt, src, w, "iou"), "diameter_m": get(mt, src, w, "free_fit", "diameter_m"),
                      "height_m": get(mt, src, w, "free_fit", "h_m"),
                      "h_observable": get(mt, src, w, "free_fit", "h_observable")} for w in ("init", "final")}
            for src in mt}
        decision = decision or get(read_json(RESULTS / clip / "stages" / "objects.json"), "mask_decision")
    sources = {clip: r["crossview"]["mug_mask_source"] for clip, r in rows.items()}
    source_error = None
    if len(set(sources.values())) > 1:          # the rule picks ONE source for all clips
        source_error = "ERROR: the mug mask source differs between clips (" + \
            ", ".join(f"{cl}: {v}" for cl, v in sources.items()) + ")"
        print(source_error, file=sys.stderr)

    spreads = {}
    for key, _, _, spec in CROSSVIEW:
        vals = [r["crossview"][key] for r in rows.values()]
        if spec is None:
            spreads[key] = {"counts": {v: vals.count(v) for v in sorted(set(map(str, vals)))}}
        else:
            x = [v for v in vals if v is not None]
            spreads[key] = {"min": min(x), "max": max(x), "range": max(x) - min(x), "n": len(x)} if x else {"n": 0}
    # S8's closest approach and event order, as information for the notes under the cross-view table
    d_min_flagged = [clip for clip, r in rows.items() if r["crossview"].get("d_min_z_flag") is True]
    ev_idx = {clip: r["crossview"].get("event_index") or {} for clip, r in rows.items()}
    d_min_after_lift = {clip: {"d_axis_min_index": r["crossview"]["d_axis_min_index"], "lift": ev_idx[clip]["lift"]}
                        for clip, r in rows.items()
                        if ev_idx[clip].get("lift", -1) >= 0 and r["crossview"]["d_axis_min_index"] > ev_idx[clip]["lift"]}
    grasp_early = {clip: e for clip, e in ev_idx.items()
                   if e.get("grasp", -1) >= 0 and e.get("contact", -1) >= 0 and e["grasp"] < e["contact"]}
    (RESULTS / "summary.json").write_text(json.dumps(clean(
        {"clips": rows, "clips_missing": missing, "hand_scale": {"s_hand": s_hands[0] if len(s_hands) == 1 else s_hands, "per_clip": scale},   # a list = the demos disagree
         "v1_to_v2": v1_to_v2,
         "mask_test": {"by_window": mask_test, "decision": decision, "source_per_clip": sources,
                       "error": source_error},
         "spreads": spreads, "d_min_z_flagged": d_min_flagged, "d_min_after_lift": d_min_after_lift,
         "grasp_before_contact": grasp_early,
         "palm_normal_flip": {"flipped": flips, "undecided": undecided, "error": flip_error}, "confound": confound}),
        indent=1))

    head = [f"{clip} ({r['view']})" for clip, r in rows.items()]
    md = ["# M2 perception v2: gates, hand scale, v1 -> v2, mug mask test, cross-view table", "",
          "Made by `python -m perception.report --summary` from results/m2_perception_v2/<clip>/gates.json, "
          "crossview.json and stages/; v1 values from results/m2_perception/<clip>/gates.json.",
          f"Clips without a report: {', '.join(missing) or 'none'}.", "",
          "## Gates (v2)", "", "| gate | pass if | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(rows)]
    for name, rule in GATES.items():
        md.append(f"| {name} | {rule} | " + " | ".join(gate_cell(r["gates"][name]) for r in rows.values()) + " |")

    md += ["", "## Hand scale (S5b)", "", s_hand_line + ".",
           f"s_hand is the median of the pass-1 scale s over the metric_valid frames of all {len(CLIPS)} clips pooled; "
           "the hand is scaled by it in every frame of every clip. Per clip (pass-1 s over its metric_valid frames):",
           "",
           "| | " + " | ".join(head) + " |", "|---|" + "---|" * len(rows),
           "| median s | " + " | ".join(fmt(x["median_s"], 1, ".3f") for x in scale.values()) + " |",
           "| IQR (25 - 75 %) | " + " | ".join(f"{fmt(x['s_iqr'][0], 1, '.3f')} - {fmt(x['s_iqr'][1], 1, '.3f')}"
                                               for x in scale.values()) + " |",
           "| metric_valid frames | " + " | ".join(str(x["n_frames"]) for x in scale.values()) + " |",
           "| median s / s_hand - 1 [%] (G9: within +-10 %) | "
           + " | ".join(fmt(None if x["ratio_to_s_hand"] is None else x["ratio_to_s_hand"] - 1, 100, "+.1f")
                        for x in scale.values()) + " |"]

    md += ["", "## v1 -> v2: every gate per clip", "",
           "v1 = results/m2_perception (its own gate definitions); v2 = this run. Gates whose rule did not change can "
           "still move because their inputs changed (S5b hand, relative contact rule, desk-edge fix, mask source).", "",
           "| gate | what changed in v2 | clip | v1 | v2 |", "|---|---|---|---|---|"]
    for name in GATES:
        for j, (clip, r) in enumerate(rows.items()):
            x = v1_to_v2[clip][name]
            md.append(f"| {name if j == 0 else ''} | {GATE_CHANGES[name] if j == 0 else ''} | {clip} ({r['view']}) | "
                      f"{gate_cell(x['v1'])} | {gate_cell(x['v2'])} |")

    md += ["", "## Mug mask test (A4, both static windows)", "",
           f"2-DOF fit IoU and the 4-DOF free fit, on the SAM 2 video masks (half res, as v1) and on the SAM 2 image "
           f"masks of the full-res frames, in the initial and the final static window. Label: diameter "
           f"{diam_label * 1000:.0f} mm, height {h_label * 1000:.0f} mm. The decision rule below uses the initial "
           f"window only; \"(fit bound)\" = the free fit stopped at its bound (r {R_BOUNDS[0] * 1000:.0f}-"
           f"{R_BOUNDS[1] * 1000:.0f} mm, h {H_BOUNDS[0] * 1000:.0f}-{H_BOUNDS[1] * 1000:.0f} mm).",
           "", "| clip | window | mask source | IoU (2-DOF) | free-fit diameter [mm] | free-fit height [mm] |",
           "|---|---|---|---|---|---|"]
    for clip, r in rows.items():
        if mask_test[clip] is None:
            md.append(f"| {clip} ({r['view']}) | | (no mask_test.json) | n/a | n/a | n/a |")
            continue
        for w in ("init", "final"):
            for src, by_window in mask_test[clip].items():
                x = by_window[w]
                d_m = x["diameter_m"]
                height = (fmt(x["height_m"], 1000, ".1f") + at_bound(x["height_m"], H_BOUNDS) if x["h_observable"]
                          else "N/A (unobservable)")
                md.append(f"| {clip} ({r['view']}) | {w} | {src} | {fmt(x['iou'], 1, '.3f')} | "
                          f"{fmt(d_m, 1000, '.1f')}{at_bound(None if d_m is None else d_m / 2, R_BOUNDS)} | {height} |")
    md += ["", "Decision (pre-set rule, recorded by S6 in objects.json): " + (
        f"{decision['rule']}. {decision['reason']}. Chosen: **{decision['chosen']}**." if decision else
        "not available (no objects.json mask_decision).")]
    md += [f"**{source_error}**" if source_error else
           f"Mask source used by S6: {', '.join(sorted(set(sources.values())))} in every clip."]

    md += ["", "## Cross-view table", "", "| | " + " | ".join(head) + " | min | max | max - min |",
           "|---|" + "---|" * (len(rows) + 3)]
    for key, label, scale_, spec in CROSSVIEW:
        cells = [fmt(r["crossview"][key], scale_, spec) for r in rows.values()]
        s = spreads[key]
        if spec is None:
            extra = ["; ".join(f"{v}: {n}" for v, n in s["counts"].items()), "", ""]
        else:
            extra = [fmt(s.get("min"), scale_, spec), fmt(s.get("max"), scale_, spec),
                     fmt(s.get("range"), scale_, spec)]
        md.append(f"| {label} | " + " | ".join(cells + extra) + " |")
    md += ["", "Angles at g come from T_palm_in_cup_at_grasp (smoothed palm); approach yaw and handle yaw are n/a when "
           "the handle yaw is invalid. Grasp height and lift height use S8's raw palm z. The two yaw columns lie in "
           "(-180, 180] deg; their min, max and max - min are taken linearly, without the wrap at +-180 deg. "
           "D_min is S8's closest approach (contact = first frame within D_min + 1 cm); clips where the palm was at or "
           f"above the mug top at that frame (S8 d_min_z_flag): {', '.join(d_min_flagged) or 'none'}. "
           "Clips where the D_min frame comes after the lift (the closest approach happens while the hand carries the "
           "mug): " + (", ".join(f"{cl} (D_min frame {x['d_axis_min_index']}, lift {x['lift']})"
                                 for cl, x in d_min_after_lift.items()) or "none") + ". "
           "Clips where the grasp frame g (last valid frame <= lift - 3) comes before contact, so z_grasp, the angles "
           "at g and phase[g] "
           "are not taken in the grasp phase (information; no rule changed): "
           + (", ".join(f"{cl} (g {e['grasp']}, contact {e['contact']}, lift {e['lift']})"
                        for cl, e in grasp_early.items()) or "none") + ".", "",
           "## Palm-normal flip (S9)", "",
           "| | " + " | ".join(head) + " |", "|---|" + "---|" * len(rows),
           "| n flipped | " + " | ".join("undecided" if f is None else str(f) for f in flips.values()) + " |", "",
           f"**{flip_line}**" if flip_error else flip_line, "",
           f"**Confound.** {confound}", ""]
    (RESULTS / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        stage_cli("report", run)
