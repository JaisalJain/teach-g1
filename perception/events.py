"""S8 -- grasp / lift / place / release events from the raw metric palm path.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.events --clip <id>

Palm = 0.5 (kp[0] + kp[9]) (wrist + middle-finger base) of the placed hand keypoints (hand_place.npz: S5's LiDAR depth
shift, then the A1v3 sideways shift that puts the mean of the 5 palm keypoints on the mean of WiLoR's 2D palm
keypoints; hand scale 1), in the table frame (S7), on metric_valid frames only. The events use the RAW palm, not the
smoothed one: smoothing would round off the lift and shift it in time.

If S7 found no table x axis (x_axis_method "failed", a 45 deg clip without desk edges), the events are computed in a
stand-in frame with the same origin and z and an arbitrary in-plane x: heights and horizontal distances do not depend
on x, so nothing is guessed. events.json records this as table_x_known = false.

    d_axis(i)  horizontal palm distance to the INITIAL cup axis (vertical line through the initial base centre, S6)
    d_final(i) the same to the FINAL cup axis
    z(i)       palm height above the table (the table frame origin is on the plane)

Contact (v3, contact_rule "pre_rise_min_plus_1cm"):
    near       frames with d_axis <= r + 0.08 m (r = config mug radius): the palm is within 8 cm of the mug surface.
               This still includes the last few cm of the approach (the palm stops 94-107 mm from the axis on the 4
               clips; the near boundary is 129.5 mm).
    t_rise     scan the metric_valid frames in time order, keeping z_min_near = the running minimum of z over the near
               frames so far (undefined before the first near frame). t_rise = the first near frame with
               z >= z_min_near + 0.03 m that stays so on the 4 next valid frames (5 consecutive valid frames, each
               compared with the running z_min_near at that frame). This is the hand going up at the mug. Only near
               frames enter the minimum, so a hand that comes in low MORE than r + 8 cm from the axis and rises on its
               way does not set it. A low approach inside r + 8 cm does: on e8825dea53 z_min_near is set at frame 90,
               the first near frame (d_axis 129.3 mm, 0.2 mm inside the boundary), while the hand is still reaching in.
    D_min      min of d_axis over the metric_valid frames BEFORE t_rise. This is not always before the lift: t_rise
               comes 2-4 frames after the lift on all 4 clips and d_axis can keep shrinking while the hand lifts, so
               D_min can fall in [lift, t_rise) (4e103843db and e8825dea53: 1 frame after the lift).
    contact    first metric_valid frame with d_axis <= D_min + 0.01 m. It always exists and comes before t_rise (the
               D_min frame itself qualifies).
    Why: v1's fixed threshold r + 5 cm sat on the data's edge (the palm point stopped 96-102 mm from the axis), so the
    top-down clip missed contact by a few mm and lost every event. v2 took D_min over the whole clip, but on 3 of 4
    clips the palm point came closest to the initial axis 8-19 frames AFTER the lift, while the hand carried the mug,
    and on one clip that moved contact after g. Taking D_min only before t_rise brings it back to at most 1 frame
    after the lift on the 4 clips; it does not rule out an overlap with the lift (see D_min). If t_rise is undefined,
    every event is undefined (the reason is in events.json).
The other events keep the v1 rules:
    lift       first valid k > contact with z >= m_k + 0.015 m on k and on the 4 next valid frames, m_k = min z over the
               valid frames in [contact, k]. The running minimum lets the hand settle lower after contact; 5 frames
               ignore a short bump. (t_rise and lift both see the hand going up, with different references and
               thresholds; neither is forced to come first.)
    grasp g    last valid frame with index <= lift - 3: the hand pose just before the cup starts to move.
    place      first valid frame after the transport peak (max z over valid frames from lift on) with z <= z(g) + 0.01 m
    release    first valid frame after place with d_final >= d_final(place) + 0.03 m: the hand moves away from the
               placed cup (the FINAL axis, since the cup is no longer at its initial position).

Lift confirmation (recorded, never used to move the lift): the IoU of the SAM 2 body mask with the initial-window
majority mask (S6) drops below 0.7 within 0.3 s of the lift -- the cup leaves its initial footprint.

phase: 0 approach [0, contact), 1 grasp [contact, lift), 2 transport [lift, place), 3 place [place, release),
4 release/retreat [release, N). A span whose bounding event is undefined is -1. An undefined event has index -1 and
t = null, with the reason in events.json.

Writes data/work/<clip>/events/: events.json (the events, t_rise, contact_rule, d_axis_min_m, d_axis_min_index,
z_at_d_axis_min_m, z_min_near_at_rise_m), events.npz (phase, d_axis, d_final, z, body_iou), events.png.
"""

from __future__ import annotations

import json

import numpy as np
from matplotlib.figure import Figure

from perception.common import WORK, cfg, load_config, stage_cli, stage_dir
from perception.geometry import plane_basis, transform
from perception.masks_sam2 import BODY, load_masks

EVENTS = ("contact", "lift", "grasp", "place", "release")
NEAR_M = 0.08               # near the mug: d_axis <= r + this
RISE_M = 0.03               # t_rise: z above its running minimum over near frames by this ...
RISE_FRAMES = 5             # ... on this many consecutive valid frames
CONTACT_ABOVE_MIN_M = 0.01  # contact: d_axis <= D_min + this
CONTACT_RULE = "pre_rise_min_plus_1cm"
LIFT_RISE_M = 0.015         # lift: z above its running minimum by this ...
LIFT_FRAMES = 5             # ... on this many consecutive valid frames
GRASP_BEFORE_LIFT = 3       # g = last valid frame at least this many frames before the lift
PLACE_MARGIN_M = 0.01       # place: z <= z_grasp + this
RELEASE_MOVE_M = 0.03       # release: d_final grows by this after place
IOU_DROP = 0.7              # lift confirmation: body-mask IoU below this ...
IOU_WINDOW_S = 0.3          # ... within this many seconds of the lift

BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"


def table_from_world(clip: str) -> tuple[np.ndarray, bool]:
    """T_table_world (4x4) from S7, p_table = R^T (p_world - origin), and whether S7 found the table x axis.
    Without x ("failed"): the stand-in frame of the module doc, R = [e1, e2, z] with (e1, e2) = plane_basis(z). Only
    quantities that do not depend on x (heights, horizontal distances, angles between vectors) may be read in it."""
    tf = json.loads((WORK / clip / "table_frame" / "table_frame.json").read_text())
    if tf["T_table_world"] is not None:
        return np.array(tf["T_table_world"]), True
    z, origin = np.array(tf["z_world"]), np.array(tf["origin_world"])
    R = np.column_stack([*plane_basis(z), z / np.linalg.norm(z)])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ origin
    return T, False


def hand_kp_table(clip: str, T_tw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Placed keypoints (A1v3) in the table frame: (N, 21, 3), NaN where not metric_valid; and metric_valid (N,)."""
    T_wc = np.load(WORK / clip / "frames" / "meta.npz")["T_world_cam"]
    hp = np.load(WORK / clip / "hand_place" / "hand_place.npz")
    valid = hp["metric_valid"].astype(bool)
    T_tc = T_tw @ T_wc                                         # camera -> table, per frame
    kp = np.einsum("nij,nkj->nki", T_tc[:, :3, :3], hp["kp3d"].astype(np.float64)) + T_tc[:, None, :3, 3]
    kp[~valid] = np.nan
    return kp, valid


def palm_rise(valid, z, d_axis, near_m) -> tuple[int, float, str | None]:
    """t_rise (module doc), the running near-minimum of z at t_rise, and None; or (-1, NaN, reason) if the palm never
    rises at the mug. near_m = r + NEAR_M."""
    vi = np.flatnonzero(valid)
    near = d_axis[vi] <= near_m
    # running minimum of z over the near frames so far; +inf before the first near frame, so no frame qualifies there
    z_min_near = np.minimum.accumulate(np.where(near, z[vi], np.inf))
    up = z[vi] >= z_min_near + RISE_M                          # each frame against the running value at that frame
    ok = [j for j in range(len(vi) - RISE_FRAMES + 1) if near[j] and up[j:j + RISE_FRAMES].all()]
    if ok:
        return int(vi[ok[0]]), float(z_min_near[ok[0]]), None
    if len(vi) == 0:
        return -1, np.nan, "no metric_valid frames"
    if not near.any():
        return -1, np.nan, (f"palm never within r + {NEAR_M} m = {near_m:.4f} m of the initial cup axis "
                            f"(closest {d_axis[vi].min():.4f} m)")
    return -1, np.nan, (f"palm z never {RISE_M} m above its running minimum over near frames, starting on a near "
                        f"frame and on the {RISE_FRAMES - 1} next valid frames")


def closest_approach(valid, d_axis, t_rise) -> tuple[float, int]:
    """D_min = min of d_axis over the valid frames BEFORE t_rise, and its frame; (NaN, -1) if t_rise is undefined.
    With t_rise defined there is always such a frame: the near frame that set the minimum t_rise rose from."""
    if t_rise < 0:
        return np.nan, -1
    vi = np.flatnonzero(valid[:t_rise])
    i_min = int(vi[np.argmin(d_axis[vi])])
    return float(d_axis[i_min]), i_min


def detect_events(valid, z, d_axis, d_final, d_min):
    """Event indices (-1 = undefined) and the reason for each undefined event. Only valid frames are looked at.
    d_min = D_min from closest_approach (NaN if t_rise is undefined)."""
    ev, why = {}, {}
    vi = np.flatnonzero(valid)

    if np.isnan(d_min):
        why["contact"] = "t_rise undefined, so D_min is undefined"
    else:                                                      # never empty: the D_min frame itself qualifies
        ev["contact"] = int(vi[d_axis[vi] <= d_min + CONTACT_ABOVE_MIN_M][0])

    if "contact" in ev:
        k = vi[vi >= ev["contact"]]
        zk = z[k]
        run_min = np.minimum.accumulate(zk)
        ok = [j for j in range(1, len(k) - LIFT_FRAMES + 1) if np.all(zk[j:j + LIFT_FRAMES] >= run_min[j] + LIFT_RISE_M)]
        if ok:
            ev["lift"] = int(k[ok[0]])
        else:
            why["lift"] = f"z never stays {LIFT_RISE_M} m above its running minimum for {LIFT_FRAMES} valid frames"

    if "lift" in ev:
        before = vi[vi <= ev["lift"] - GRASP_BEFORE_LIFT]
        if len(before):
            ev["grasp"] = int(before[-1])
        else:
            why["grasp"] = f"no metric_valid frame at or before lift - {GRASP_BEFORE_LIFT}"

    if "grasp" in ev:
        z_grasp = z[ev["grasp"]]
        after = vi[vi >= ev["lift"]]
        peak = after[np.argmax(z[after])]
        down = vi[(vi > peak) & (z[vi] <= z_grasp + PLACE_MARGIN_M)]
        ev["peak"] = int(peak)
        if len(down):
            ev["place"] = int(down[0])
        else:
            why["place"] = f"z never back to z_grasp + {PLACE_MARGIN_M} m = {z_grasp + PLACE_MARGIN_M:.3f} m " \
                           f"after the peak (frame {peak})"

    if "place" in ev:
        p = ev["place"]
        away = vi[(vi > p) & (d_final[vi] >= d_final[p] + RELEASE_MOVE_M)]
        if len(away):
            ev["release"] = int(away[0])
        else:
            why["release"] = f"palm never {RELEASE_MOVE_M} m further from the final cup axis than at place " \
                             f"({d_final[p]:.3f} m)"

    # every later event depends on the earlier ones
    for prev, name in zip(("",) + EVENTS[:-1], EVENTS):
        if name not in ev and name not in why:
            why[name] = f"{prev} undefined"
        ev.setdefault(name, -1)
    ev.setdefault("peak", -1)
    return ev, why


def phase_labels(n: int, ev: dict) -> np.ndarray:
    """0..4 between consecutive events (see module doc), -1 on spans with an undefined bounding event."""
    bounds = [0, ev["contact"], ev["lift"], ev["place"], ev["release"], n]
    phase = np.full(n, -1, np.int8)
    for p in range(5):
        if bounds[p] >= 0 and bounds[p + 1] >= 0:
            phase[bounds[p]:bounds[p + 1]] = p
    return phase


def body_iou(clip: str, n: int) -> np.ndarray:
    """IoU of each frame's body mask (half res) with the initial-window majority mask from S6; NaN at i = 0."""
    f = np.load(WORK / clip / "objects" / "init_body_majority.npz")
    h, w = (int(s) for s in f["shape"])
    ref = np.unpackbits(f["bits"], count=h * w).reshape(h, w).astype(bool)
    masks = load_masks(clip)
    iou = np.full(n, np.nan)
    for i in range(1, n):
        m = masks.get(BODY, i, "half")
        union = np.count_nonzero(m | ref)
        if union:
            iou[i] = np.count_nonzero(m & ref) / union
    return iou


def plot(path, clip, t, valid, d_axis, d_final, z, iou, ev, near_m, t_rise, z_min_near, d_min, i_min, z_grasp):
    s = t - t[0]
    fig = Figure(figsize=(11, 8.5), layout="constrained")     # constrained: no clipped axis labels
    ax = fig.subplots(4, 1, sharex=True, height_ratios=[3, 3, 2, 1])
    ax[0].plot(s, d_axis, color=BLUE, lw=1.5, label="to initial cup axis")
    ax[0].plot(s, d_final, color=ORANGE, lw=1.5, label="to final cup axis")
    ax[0].axhline(near_m, color=GREY, ls="-.", lw=1, label=f"r + {NEAR_M * 100:.0f} cm (near, for t_rise)")
    if i_min >= 0:
        ax[0].axhline(d_min + CONTACT_ABOVE_MIN_M, color=GREY, ls=":", lw=1, label="D_min + 1 cm (contact)")
        ax[0].plot(s[i_min], d_min, "o", color=INK, ms=5, label=f"D_min = {d_min * 100:.1f} cm (before t_rise)")
    ax[0].set_ylabel("horizontal palm\ndistance [m]")
    ax[1].plot(s, z, color=BLUE, lw=1.5, label="palm z")
    if t_rise >= 0:
        ax[1].axhline(z_min_near + RISE_M, color=ORANGE, ls=":", lw=1, label="near-minimum at t_rise + 3 cm")
    if np.isfinite(z_grasp):
        ax[1].axhline(z_grasp + PLACE_MARGIN_M, color=GREY, ls=":", lw=1, label="z_grasp + 1 cm (place)")
    if ev["peak"] >= 0:
        ax[1].plot(s[ev["peak"]], z[ev["peak"]], "o", color=INK, ms=5, label="transport peak")
    ax[1].set_ylabel("palm height\nabove table [m]")
    ax[2].plot(s, iou, color=BLUE, lw=1.5, label="body mask vs initial")
    ax[2].axhline(IOU_DROP, color=GREY, ls=":", lw=1, label="0.7 (lift confirmation)")
    ax[2].set_ylabel("body IoU")
    ax[2].set_ylim(-0.05, 1.05)
    ax[3].fill_between(s, valid.astype(float), step="mid", color=BLUE, alpha=0.5, lw=0)
    ax[3].set_ylabel("metric\nvalid")
    ax[3].set_yticks([0, 1])
    ax[3].set_xlabel("time since first frame [s]")
    # events and t_rise in time order; labels cycle through four heights: contact, grasp, lift and t_rise can all fall
    # within a few frames of each other (contact and grasp in either order)
    marks = [(ev[e], e, INK, "-") for e in EVENTS if ev[e] >= 0]
    if t_rise >= 0:
        marks.append((t_rise, "t_rise", ORANGE, "--"))
    for j, (i, name, colour, style) in enumerate(sorted(marks)):
        for a in ax:
            a.axvline(s[i], color=colour, ls=style, lw=0.8, alpha=0.7)
        ax[0].text(s[i], 1.03 + 0.07 * (j % 4), name, transform=ax[0].get_xaxis_transform(), ha="center", fontsize=9,
                   color=colour)
    for a in ax[:3]:
        a.legend(loc="best", fontsize=8, frameon=False)     # "best": the curves differ per clip
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"{clip}: events from the raw metric palm (table frame)")
    fig.savefig(path, dpi=120)


def run(clip: str) -> dict:
    out = stage_dir(clip, "events")
    t = np.load(WORK / clip / "frames" / "meta.npz")["t"]
    n = len(t)
    near_m = cfg(load_config(), "mug", "body_outer_diameter_m") / 2 + NEAR_M
    T_tw, x_known = table_from_world(clip)          # x_known False: stand-in x, fine for d_axis, d_final and z
    kp, valid = hand_kp_table(clip, T_tw)
    obj = json.loads((WORK / clip / "objects" / "objects.json").read_text())
    cup_init = transform(T_tw, np.array(obj["init"]["base_centre_world"]))
    cup_final = transform(T_tw, np.array(obj["final"]["base_centre_world"]))
    palm = 0.5 * (kp[:, 0] + kp[:, 9])
    d_axis = np.linalg.norm(palm[:, :2] - cup_init[:2], axis=1)
    d_final = np.linalg.norm(palm[:, :2] - cup_final[:2], axis=1)
    z = palm[:, 2]
    if not np.isfinite(kp[valid]).all():            # a NaN would silently become D_min (np.argmin returns it)
        raise ValueError(f"{clip}: hand_place.npz has non-finite keypoints on metric_valid frames")
    t_rise, z_min_near, why_rise = palm_rise(valid, z, d_axis, near_m)
    d_min, i_min = closest_approach(valid, d_axis, t_rise)
    z_at_min = float(z[i_min]) if i_min >= 0 else None
    ev, why = detect_events(valid, z, d_axis, d_final, d_min)
    if why_rise is not None:
        why = {"t_rise": why_rise, **why}
    z_grasp = z[ev["grasp"]] if ev["grasp"] >= 0 else np.nan
    phase = phase_labels(n, ev)
    iou = body_iou(clip, n)

    confirm, min_iou = None, None
    if ev["lift"] >= 0:
        win = (np.abs(t - t[ev["lift"]]) <= IOU_WINDOW_S) & np.isfinite(iou)
        if win.any():
            min_iou = float(iou[win].min())
            confirm = min_iou < IOU_DROP

    events = {e: {"index": ev[e], "t": float(t[ev[e]]) if ev[e] >= 0 else None} for e in EVENTS + ("peak",)}
    result = {**events,
              "t_rise": {"index": t_rise, "t": float(t[t_rise]) if t_rise >= 0 else None},
              "lift_confirm": {"iou_below_0p7_within_0p3s": confirm, "min_iou_in_window": min_iou},
              "z_grasp_m": float(z_grasp) if np.isfinite(z_grasp) else None,
              "contact_rule": CONTACT_RULE, "d_axis_min_m": d_min if i_min >= 0 else None, "d_axis_min_index": i_min,
              "z_at_d_axis_min_m": z_at_min, "z_min_near_at_rise_m": z_min_near if t_rise >= 0 else None,
              "table_x_known": x_known, "reasons": why}
    (out / "events.json").write_text(json.dumps(result, indent=1))
    np.savez(out / "events.npz", phase=phase, d_axis=d_axis, d_final=d_final, z=z, body_iou=iou)
    plot(out / "events.png", clip, t, valid, d_axis, d_final, z, iou, ev, near_m, t_rise, z_min_near, d_min, i_min,
         z_grasp)

    for e, reason in why.items():
        print(f"{clip}: {e} undefined -- {reason}")
    t_rel = {f"t_{e}_rel_s": round(float(t[ev[e]] - t[0]), 3) if ev[e] >= 0 else None for e in EVENTS}
    return {"t_rise": t_rise, "t_rise_rel_s": round(float(t[t_rise] - t[0]), 3) if t_rise >= 0 else None,
            **{e: ev[e] for e in EVENTS}, **t_rel, "z_grasp_m": result["z_grasp_m"],
            **{k: result[k] for k in ("contact_rule", "d_axis_min_m", "d_axis_min_index", "z_at_d_axis_min_m",
                                      "z_min_near_at_rise_m")},
            "lift_confirm": confirm,
            "min_iou_in_window": min_iou, "table_x_known": x_known, "n_metric_valid": int(valid.sum()),
            "phase_counts": {int(p): int(np.count_nonzero(phase == p)) for p in range(-1, 5)}, "reasons": why}


if __name__ == "__main__":
    stage_cli("events", run)
