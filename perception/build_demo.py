"""S10 -- one demo file per clip: data/demos/<clip>.npz (schema_version 3), the file the retargeter (hoi-sim) reads.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.build_demo --clip <id>

This stage measures nothing new. It collects the outputs of S1-S9 (with S5b), converts the camera-frame hand into the
table frame and writes every field of the schema below. The file is ALWAYS written, also for a clip that fails its
gates and also when a stage output is missing: what could not be produced stays NaN (False for flags, -1 for indices /
phase, "missing" for the method / rule / source strings), and the missing stage files are listed in
data/work/<clip>/demo/done.json ("missing").

Schema 3 = schema 2 + the M2 v2 fields (hand scale and its fit errors, the relative contact rule, the mug mask source).
Existing fields keep their names and shapes, but their values come from the v2 stages: the S5b hand (hand_fit.npz: S5's
depth-corrected hand, scaled by s_hand and shifted sideways), the relative contact rule (-> event times, phase,
grasp_index, T_palm_in_cup_at_grasp) and the desk-edge fix A3 (-> table frame); lidar_residual_m is now defined on
metric_valid frames only. Every hand keypoint, palm and residual field below comes from the S5b hand, except
reproj_err_px (S5, before S5b); mano_* stay WiLoR's values, depth_shift_m is S5's and hand_valid is S2's.

Frames: "table" = the S7 table frame (z = table normal, up; origin = initial mug base centre on the table; x = my
forward, away from the near desk edge; y = z x x). "cam" = ORIGINAL (sideways) camera frame, OpenCV axes (x right, y down,
z forward). "world" = ARKit world, y up. Units: metres, radians, seconds. T = number of canonical frames i (odometry
rows); i = 0 has no image, so every per-image field is NaN / False there.

Metadata
  schema_version          () int     3
  clip_id, view_label     () str     view_label: "top-down" | "90deg" | "45deg"
  t                       (T,)       odometry timestamps [s], ARKit clock (absolute; t - t[0] = clip time)
  K                       (T,3,3)    RGB intrinsics [px] of the ORIGINAL 1920x1440 image
  T_world_cam             (T,4,4)    camera-to-world
  T_table_world           (4,4)      p_table = T_table_world @ p_world; all NaN if S7 found no table x, and then so is
                                     every table-frame field below
  table_plane_world       (4,)       [nx, ny, nz, d] with n.p + d = 0 (world), n.(0,1,0) > 0
  table_rms_m             ()         table-plane inlier RMS (S4)
  x_axis_method           () str     how S7 found table x: long_edge_extent | largest_total_extent | reference_fallback |
                                     failed ("missing" if S7 has no output)
  x_axis_vs_reference_deg ()         signed angle from the S7 reference direction to table x, about z [deg]
Hand (WiLoR MANO right hand; S5 moves it along its viewing ray to the LiDAR depth, S5b scales it by s_hand about its
     mesh centroid and slides it parallel to the image plane, fitted by robust least squares (soft_l1, f_scale 5 px) to
     WiLoR's 2D palm keypoints; the fit leaves a palm reprojection error, see reproj_err_palm_px)
  hand_valid              (T,) bool  a right hand was detected and reconstructed (S2)
  metric_valid            (T,) bool  hand_valid and >= 30 LiDAR pixels in the S5 correction; use only these frames
                                     (S5b keeps S5's metric_valid)
  hand_kp_cam             (T,21,3)   21 keypoints of the S5b hand (S5 depth shift, then scale s_hand + sideways shift),
                                     cam frame; NaN where not metric_valid (S5 and S5b correct only those frames)
  hand_kp_table           (T,21,3)   the same keypoints in the table frame (NaN where not metric_valid)
                                     order (OpenPose, verified in WiLoR's mano_wrapper.py): wrist 0, thumb 1-4,
                                     index 5-8, middle 9-12, ring 13-16, pinky 17-20 (4, 8, 12, 16, 20 = fingertips)
  palm_pos_table_raw      (T,3)      0.5 (kp0 + kp9), metric_valid frames (S9)
  palm_pos_table          (T,3)      gap-filled (gaps <= 0.3 s) and Savitzky-Golay smoothed (S9); finite on the
                                     metric_valid frames AND the filled gap frames
  palm_rot_table_raw      (T,3,3)    columns [x_f, y, n]: x_f = wrist -> middle-finger base, n = palm normal (points
                                     toward the cup at grasp), y = n x x_f
  palm_rot_table          (T,3,3)    gap-filled (SLERP), smoothed, re-orthonormalised (S9)
  palm_normal_flipped     () bool    S9 flipped n in every frame. Also False when the flip was NOT decided (grasp frame
                                     undefined) or S9 has no output: palm/palm.npz flip_decided tells them apart
  mano_global_orient      (T,3,3)    MANO root rotation in the table frame (R_table_cam @ WiLoR's cam-frame global_orient)
  mano_hand_pose          (T,45)     15 MANO joint rotations, axis-angle [rad], each relative to its parent joint
  mano_betas              (T,10)     MANO shape coefficients
                                     (mano_*: WiLoR's own values on every hand_valid frame, also where the hand is not
                                     metric_valid. S5 and S5b only move and scale the hand, so the metric hand is the
                                     MANO hand of these values scaled by s_hand, plus a per-frame translation: align
                                     its keypoints to hand_kp_cam / hand_kp_table)
  hand_scale_per_frame    (T,)       S5b pass 1: the scale s fitted per frame to the 5 palm keypoints (with its own
                                     sideways shift); NaN where not metric_valid. Not applied: a per-frame consistency
                                     record (gate G9 compares each clip's median with s_hand)
  s_hand                  ()         the scale applied to the hand in every frame and every clip: median of the pass-1 s
                                     over the metric_valid frames of all 4 clips pooled (one hand)
  depth_shift_m           (T,)       S5 depth shift D = median(lidar - mesh) (+ = WiLoR's hand was too close); finite
                                     wherever S5 could compute D, i.e. also on hand_valid frames that are not metric_valid
  lidar_residual_m        (T,)       re-measured on the S5b hand (no new depth shift): median |lidar - mesh| over the hand
                                     pixels within 3 cm of the mesh; metric_valid frames only, NaN where no pixel is left
  reproj_err_palm_px      (T,)       S5b pass 2: median over the 5 palm keypoints (wrist, index / middle / ring / pinky
                                     bases) of |project(K, kp) - WiLoR's 2D keypoint| [px]; metric_valid frames (gate G3)
  reproj_err_all_px       (T,)       the same over all 21 keypoints (the fingers included)
  reproj_err_px           (T,)       S5's plumbing check from BEFORE S5b, kept from schema 2: the S5 hand's 21 keypoints
                                     projected with K vs WiLoR's own 2D keypoints (tests our transforms, NOT WiLoR's
                                     accuracy)
Objects (table frame)
  cup_init_pos_table, cup_final_pos_table (3,)  mug base centre on the table plane, initial / final window (S6)
  cup_init_yaw, cup_final_yaw  ()    handle direction: angle of (handle point P - cup axis) from table x, about z [rad]
  cup_yaw_valid           (2,) bool  [init, final]: handle mask >= 0.2 % of the body mask (else do not trust the yaw)
  cup_init_iou, cup_final_iou ()     silhouette IoU of the 2-DOF cylinder fit
  cup_free_fit_r_h        (2,)       [r, h] of the 4-DOF validation fit, initial window; h NaN = unobservable
  coaster_pos_table       (3,)       coaster centre on the table plane (initial window)
  coaster_diameter_fit_m  ()         diameter of the RANSAC circle fit
  mug_mask_source         () str     the mug body masks behind the cup fits above (S6, chosen by the A4 rule for all
                                     clips): "video" (SAM 2 video predictor, half res) | "full_res_image" (SAM 2 image
                                     predictor on the full-res frames of the two static windows)
Events (S8, from the raw metric palm)
  phase                   (T,) int8  0 approach, 1 grasp, 2 transport, 3 place, 4 release/retreat, -1 undefined
  t_contact, t_lift, t_place, t_release ()   [s], same clock as t (NaN = not detected)
  contact_rule            () str     "relative_min_plus_1cm": t_contact = first metric_valid frame with
                                     d_axis <= D_min + 1 cm (d_axis = horizontal palm distance to the initial cup axis)
  d_axis_min_m            ()         D_min = min d_axis over the metric_valid frames
  d_axis_min_index        () int     the frame of D_min (index into t), -1 = undefined
  grasp_index             () int     grasp frame g (index into t), -1 = undefined. g = last valid frame <= lift - 3, so
                                     it can come BEFORE t_contact when D_min falls after the lift (S8); phase[g] is then
                                     0 (approach)
  T_palm_in_cup_at_grasp  (4,4)      smoothed palm pose at g in the cup frame (origin = initial base centre, z up,
                                     x toward the handle at the initial yaw; table x if that yaw is invalid). Raw pose
                                     if g's valid run is shorter than the smoothing window. NaN if g is undefined, or
                                     if S7 found no table x and the handle yaw is invalid
Dimensions
  mug_dims, coaster_dims  () str     JSON of the config/objects.yaml blocks, copied as they are (value + source flag per
                                     entry; the mug's capacity_ml is a bare number in the config)
"""

from __future__ import annotations

import json

import numpy as np

from perception.common import CLIPS, DEMOS, WORK, load_config, stage_cli
from perception.geometry import transform

SCHEMA_VERSION = 3


def load_stage(clip: str, stage: str, name: str):
    """A stage output file as a dict (.npz arrays or .json), or None when the stage has not written it."""
    path = WORK / clip / stage / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def get(d, *keys):
    """d[k1][k2]..., or None when the file, a key or the value (JSON null) is missing."""
    try:
        for k in keys:
            d = d[k]
    except (KeyError, IndexError, TypeError):
        return None
    return d


def arr(x, shape) -> np.ndarray:
    """x as a float64 array of the given shape, all NaN when x is None."""
    return np.full(shape, np.nan) if x is None else np.asarray(x, dtype=np.float64).reshape(shape)


def flags(x, n: int) -> np.ndarray:
    return np.zeros(n, bool) if x is None else np.asarray(x, dtype=bool).reshape(n)


def run(clip: str) -> dict:
    files = {"frames": "meta.npz", "hand": "hand.npz", "hand_metric": "hand_metric.npz", "hand_fit": "hand_fit.npz",
             "table": "table.json", "objects": "objects.json", "table_frame": "table_frame.json",
             "events": "events.json", "palm": "palm.npz"}
    s = {stage: load_stage(clip, stage, name) for stage, name in files.items()}
    s["events_npz"] = load_stage(clip, "events", "events.npz")
    missing = [stage for stage, v in s.items() if v is None]
    if s["frames"] is None:
        raise FileNotFoundError(f"{WORK / clip / 'frames' / 'meta.npz'} missing: S1 defines the frame index")
    # hf = the S5b hand (every hand field); hm = S5, read only for its pre-S5b plumbing number reproj_err_px
    meta, hand, hm, hf, objs, tf, ev, palm = (s[k] for k in ("frames", "hand", "hand_metric", "hand_fit", "objects",
                                                              "table_frame", "events", "palm"))
    t = meta["t"]
    n = len(t)
    c = load_config()

    # table frame <- camera frame, per frame (all NaN if S7 has no table frame)
    T_tw = arr(get(tf, "T_table_world"), (4, 4))
    T_tc = T_tw @ meta["T_world_cam"]
    kp_cam = arr(get(hf, "kp3d"), (n, 21, 3))
    kp_table = np.einsum("nij,nkj->nki", T_tc[:, :3, :3], kp_cam) + T_tc[:, None, :3, 3]

    def cup(window):
        base = transform(T_tw, arr(get(objs, window, "base_centre_world"), 3))
        P = transform(T_tw, arr(get(objs, window, "handle", "P_world"), 3))
        return base, np.arctan2(P[1] - base[1], P[0] - base[0]), bool(get(objs, window, "handle", "valid"))

    cup_init, yaw_init, yaw_init_ok = cup("init")
    cup_final, yaw_final, yaw_final_ok = cup("final")
    h_free = get(objs, "init", "free_fit", "h_m") if get(objs, "init", "free_fit", "h_observable") else None

    def event_time(name):
        i = get(ev, name, "index")
        return t[i] if i is not None and i >= 0 else np.nan

    phase = get(s["events_npz"], "phase")
    grasp = get(ev, "grasp", "index")
    d_min_index = get(ev, "d_axis_min_index")
    d = {
        "schema_version": SCHEMA_VERSION, "clip_id": clip, "view_label": CLIPS[clip],
        "t": t, "K": meta["K"], "T_world_cam": meta["T_world_cam"], "T_table_world": T_tw,
        "table_plane_world": arr(get(s["table"], "plane_world"), 4), "table_rms_m": arr(get(s["table"], "rms_m"), ()),
        "x_axis_method": get(tf, "x_axis_method") or "missing",
        "x_axis_vs_reference_deg": arr(get(tf, "reference", "angle_deg"), ()),

        "hand_valid": flags(get(hand, "hand_valid"), n), "metric_valid": flags(get(hf, "metric_valid"), n),
        "hand_kp_cam": kp_cam, "hand_kp_table": kp_table,
        "palm_pos_table_raw": arr(get(palm, "palm_pos_raw"), (n, 3)), "palm_pos_table": arr(get(palm, "palm_pos"), (n, 3)),
        "palm_rot_table_raw": arr(get(palm, "palm_rot_raw"), (n, 3, 3)),
        "palm_rot_table": arr(get(palm, "palm_rot"), (n, 3, 3)),
        "palm_normal_flipped": bool(get(palm, "normal_flipped")),
        "mano_global_orient": T_tc[:, :3, :3] @ arr(get(hand, "global_orient"), (n, 3, 3)),
        "mano_hand_pose": arr(get(hand, "hand_pose_aa"), (n, 45)), "mano_betas": arr(get(hand, "betas"), (n, 10)),
        "hand_scale_per_frame": arr(get(hf, "hand_scale_per_frame"), n), "s_hand": arr(get(hf, "s_hand"), ()),
        "depth_shift_m": arr(get(hf, "depth_shift_m"), n), "lidar_residual_m": arr(get(hf, "lidar_residual_m"), n),
        "reproj_err_palm_px": arr(get(hf, "reproj_err_palm_px"), n),
        "reproj_err_all_px": arr(get(hf, "reproj_err_all_px"), n),
        "reproj_err_px": arr(get(hm, "reproj_err_px"), n),

        "cup_init_pos_table": cup_init, "cup_final_pos_table": cup_final,
        "cup_init_yaw": yaw_init, "cup_final_yaw": yaw_final, "cup_yaw_valid": np.array([yaw_init_ok, yaw_final_ok]),
        "cup_init_iou": arr(get(objs, "init", "iou"), ()), "cup_final_iou": arr(get(objs, "final", "iou"), ()),
        "cup_free_fit_r_h": np.array([arr(get(objs, "init", "free_fit", "r_m"), ()), arr(h_free, ())]),
        "coaster_pos_table": transform(T_tw, arr(get(objs, "coaster", "centre_world"), 3)),
        "coaster_diameter_fit_m": arr(get(objs, "coaster", "diameter_m"), ()),
        "mug_mask_source": get(objs, "mug_mask_source") or "missing",

        "phase": np.full(n, -1, np.int8) if phase is None else phase.astype(np.int8),
        "t_contact": event_time("contact"), "t_lift": event_time("lift"), "t_place": event_time("place"),
        "t_release": event_time("release"), "grasp_index": -1 if grasp is None else int(grasp),
        "contact_rule": get(ev, "contact_rule") or "missing",
        "d_axis_min_m": arr(get(ev, "d_axis_min_m"), ()),
        "d_axis_min_index": -1 if d_min_index is None else int(d_min_index),
        "T_palm_in_cup_at_grasp": arr(get(palm, "T_palm_in_cup_at_grasp"), (4, 4)),

        "mug_dims": json.dumps(c["mug"]), "coaster_dims": json.dumps(c["coaster"]),
    }
    DEMOS.mkdir(parents=True, exist_ok=True)
    path = DEMOS / f"{clip}.npz"
    np.savez_compressed(path, **d)
    return {"path": str(path), "schema_version": SCHEMA_VERSION, "n_fields": len(d), "n_frames": n, "missing": missing,
            "n_hand_valid": int(d["hand_valid"].sum()), "n_metric_valid": int(d["metric_valid"].sum()),
            "grasp_index": d["grasp_index"], "x_axis_method": d["x_axis_method"], "s_hand": float(d["s_hand"]),
            "contact_rule": d["contact_rule"], "d_axis_min_m": float(d["d_axis_min_m"]),
            "mug_mask_source": d["mug_mask_source"]}


if __name__ == "__main__":
    stage_cli("demo", run)
