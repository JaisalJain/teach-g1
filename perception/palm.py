"""S9 -- palm trajectory in the table frame (position + orientation, raw and smoothed) and the grasp pose in the cup frame.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.palm --clip <id>

Inputs: the placed hand keypoints (A1v3, hand_place.npz, read by events.hand_kp_table) in the table frame (S7), the
events (S8) and the initial cup (S6).

Position   palm_pos = 0.5 (kp[0] + kp[9]), wrist + middle-finger base (the same point the events use).
Axes       x_f = normalize(kp[9] - kp[0])                            finger direction
           n   = normalize(cross(kp[5] - kp[0], kp[17] - kp[0]))    palm normal (index base x pinky base)
           The sign of n depends on the keypoint convention, so it is fixed from the data: at the grasp frame g the palm
           faces the cup, i.e. n . (horizontal direction palm -> initial cup axis) > 0. If not, n is flipped on ALL
           frames and the flip is recorded. The flip must be the same in every clip; one clip cannot check that, so
           palm.npz records normal_flipped, flip_decided (False when g is undefined) and normal_dot_at_grasp for a
           comparison across clips (not done in this stage).
           Then n <- normalize(n - (n.x_f) x_f), y = n x x_f, palm_rot = [x_f, y, n] (columns, det +1).
Gaps       a gap = the invalid frames between two valid frames a < b. Filled only if t[b] - t[a] <= 0.3 s and none of
           contact, g, lift lies inside: position linear in time, rotation SLERP. Longer gaps and the clip ends stay NaN.
           (With S8's definitions the event rule cannot fire: all three events are metric_valid frames, and a gap
           holds only invalid frames. It is kept as the spec states it.)
Smoothing  Savitzky-Golay, order 2, odd window w = 2 floor(0.2 / median dt / 2) + 1, on each contiguous valid run
           (after filling) that is at least w long; shorter runs keep their raw values. Rotations: smooth the 9 matrix
           entries, then take the nearest rotation (SVD), because an average of rotation matrices is not a rotation.
Grasp pose T_palm_in_cup_at_grasp = inv(T_table_cup) T_table_palm at g, smoothed pose. Cup frame: origin = initial
           base centre, z = table z, x = toward the handle (initial yaw, S6); table x if the yaw is invalid (flagged).
           This is the object-centric grasp the retargeter replays on a cup anywhere on the table.

If S7 found no table x (events.table_from_world returns a stand-in x), no table-frame pose exists: the palm poses are
NaN and invalid. The flip and the grasp pose in a handle-based cup frame do not depend on table x, so they are kept;
the grasp pose is NaN if the cup frame would need table x (invalid handle yaw).

Writes data/work/<clip>/palm/palm.npz (table frame, NaN where invalid).
"""

from __future__ import annotations

import json

import numpy as np
from scipy.signal import savgol_filter

from perception.common import WORK, stage_cli, stage_dir
from perception.events import hand_kp_table, table_from_world
from perception.geometry import invert, orthonormalize, slerp_rotmat, transform

MAX_GAP_S = 0.3        # fill gaps up to this long (t[b] - t[a])
SG_WINDOW_S = 0.2      # Savitzky-Golay window length ...
SG_ORDER = 2           # ... and polynomial order


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def palm_axes(kp: np.ndarray):
    """(N, 21, 3) keypoints -> palm position, finger direction x_f, palm normal n (not yet orthogonal to x_f)."""
    pos = 0.5 * (kp[:, 0] + kp[:, 9])
    x_f = normalize(kp[:, 9] - kp[:, 0])
    n = normalize(np.cross(kp[:, 5] - kp[:, 0], kp[:, 17] - kp[:, 0]))
    return pos, x_f, n


def palm_rotation(x_f: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Columns [x_f, y, n] with n made orthogonal to x_f and y = n x x_f."""
    n = normalize(n - np.sum(n * x_f, -1, keepdims=True) * x_f)
    return np.stack([x_f, np.cross(n, x_f), n], -1)


def fill_gaps(t, pos, rot, valid, keep_out):
    """Fill the short gaps (module doc); keep_out = frame indices a filled gap must not contain.
    Returns filled copies of pos and rot, the new valid mask and one log entry per gap."""
    pos, rot, valid_filled = pos.copy(), rot.copy(), valid.copy()
    vi = np.flatnonzero(valid)
    log = []
    for a, b in zip(vi[:-1], vi[1:]):
        if b - a == 1:
            continue
        span = t[b] - t[a]
        inside = [int(e) for e in keep_out if a < e < b]       # always empty in the pipeline: S8 events are valid frames
        fill = bool(span <= MAX_GAP_S and not inside)
        log.append({"a": int(a), "b": int(b), "span_s": round(float(span), 4), "filled": fill, "events_inside": inside})
        if fill:
            s = (t[a + 1:b] - t[a]) / span
            pos[a + 1:b] = pos[a] + s[:, None] * (pos[b] - pos[a])
            rot[a + 1:b] = [slerp_rotmat(rot[a], rot[b], si) for si in s]
            valid_filled[a + 1:b] = True
    return pos, rot, valid_filled, log


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every run of consecutive valid frames."""
    d = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def sg_window(t: np.ndarray) -> int:
    """Odd window length (frames) nearest SG_WINDOW_S: 2 floor(0.2 / median dt / 2) + 1 (the contract's formula).
    An exact 60 (30) Hz clock gives 13 (7). The real ARKit median dt is slightly longer (0.016674 s, 0.033334 s), so
    0.2 / dt is just under 12 (6) and this gives 11 (5); the window is logged in done.json."""
    return 2 * int(np.floor(SG_WINDOW_S / np.median(np.diff(t)) / 2)) + 1


def smooth_runs(pos, rot, valid, w):
    """Savitzky-Golay on every valid run of >= w frames; returns smoothed copies and the number of runs smoothed."""
    pos, rot = pos.copy(), rot.copy()
    n_smoothed = 0
    for a, b in valid_runs(valid):
        if b - a >= w:
            pos[a:b] = savgol_filter(pos[a:b], w, SG_ORDER, axis=0)
            rot[a:b] = orthonormalize(savgol_filter(rot[a:b], w, SG_ORDER, axis=0))
            n_smoothed += 1
    return pos, rot, n_smoothed


def run(clip: str) -> dict:
    out = stage_dir(clip, "palm")
    t = np.load(WORK / clip / "frames" / "meta.npz")["t"]
    T_tw, x_known = table_from_world(clip)   # x_known False: stand-in x (module doc)
    kp, valid = hand_kp_table(clip, T_tw)
    valid &= np.isfinite(kp).all(axis=(1, 2))
    ev = json.loads((WORK / clip / "events" / "events.json").read_text())
    contact, g, lift = (ev[e]["index"] for e in ("contact", "grasp", "lift"))
    obj = json.loads((WORK / clip / "objects" / "objects.json").read_text())
    cup = transform(T_tw, np.array(obj["init"]["base_centre_world"]))

    pos_raw, x_f, n = palm_axes(kp)
    normal_dot = None                     # n . (horizontal direction palm -> cup axis) at g, before any flip
    if g >= 0:
        normal_dot = float(n[g] @ normalize(np.append(cup[:2] - pos_raw[g, :2], 0.0)))
    flipped = normal_dot is not None and normal_dot < 0
    if flipped:
        n = -n
    rot_raw = palm_rotation(x_f, n)

    pos_filled, rot_filled, valid_filled, gaps = fill_gaps(t, pos_raw, rot_raw, valid,
                                                           [e for e in (contact, g, lift) if e >= 0])
    w = sg_window(t)
    pos, rot, n_smoothed = smooth_runs(pos_filled, rot_filled, valid_filled, w)

    handle = obj["init"]["handle"]
    from_yaw = bool(handle["valid"])
    yaw = 0.0                             # table x when the handle yaw is invalid
    if from_yaw:
        P = transform(T_tw, np.array(handle["P_world"]))
        yaw = float(np.arctan2(P[1] - cup[1], P[0] - cup[0]))
    T_table_cup = np.eye(4)
    T_table_cup[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    T_table_cup[:3, 3] = cup
    T_grasp = np.full((4, 4), np.nan)
    if g >= 0 and (x_known or from_yaw):   # with a stand-in x, only a handle-based cup frame is meaningful
        T_table_palm = np.eye(4)
        T_table_palm[:3, :3], T_table_palm[:3, 3] = rot[g], pos[g]
        T_grasp = invert(T_table_cup) @ T_table_palm

    if not x_known:                       # no table-frame pose without table x (module doc)
        for a in (pos_raw, rot_raw, pos, rot):
            a[:] = np.nan
        valid[:] = valid_filled[:] = False

    np.savez(out / "palm.npz", palm_pos_raw=pos_raw, palm_rot_raw=rot_raw, palm_pos=pos, palm_rot=rot,
             valid_raw=valid, valid=valid_filled, normal_flipped=flipped, flip_decided=g >= 0,
             normal_dot_at_grasp=np.nan if normal_dot is None else normal_dot, sg_window=w,
             T_palm_in_cup_at_grasp=T_grasp, cup_frame_x_from_yaw=from_yaw)

    if g < 0:
        print(f"{clip}: grasp frame undefined -- normal flip not decided (left unflipped), no grasp pose")
    if not x_known:
        print(f"{clip}: no table x (S7 failed) -- palm poses NaN; flip and handle-based grasp pose kept")
    runs = valid_runs(valid_filled)
    return {"grasp_index": g, "normal_flipped": flipped, "flip_decided": g >= 0, "normal_dot_to_cup_at_grasp": normal_dot,
            "table_x_known": x_known, "n_valid_raw": int(valid.sum()), "n_valid": int(valid_filled.sum()),
            "n_gaps_filled": sum(gp["filled"] for gp in gaps), "n_gaps_unfilled": sum(not gp["filled"] for gp in gaps),
            "gaps": gaps, "median_dt_s": float(np.median(np.diff(t))), "sg_window": w, "n_runs": len(runs),
            "n_runs_smoothed": n_smoothed, "cup_frame_x_from_yaw": from_yaw,
            "cup_yaw_init_deg": round(float(np.degrees(yaw)), 2) if from_yaw and x_known else None,
            "palm_in_cup_at_grasp_m": T_grasp[:3, 3].tolist() if np.isfinite(T_grasp).all() else None}


if __name__ == "__main__":
    stage_cli("palm", run)
