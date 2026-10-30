"""S8 -- grasp / lift / place / release events from the raw metric palm path.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.events --clip <id>

Palm = 0.5 (kp[0] + kp[9]) (wrist + middle-finger base) of the LiDAR-corrected keypoints (S5), in the table frame (S7),
on metric_valid frames only. The events use the RAW palm, not the smoothed one: smoothing would round off the lift and
shift it in time.

If S7 found no table x axis (x_axis_method "failed", a 45 deg clip without desk edges), the events are computed in a
stand-in frame with the same origin and z and an arbitrary in-plane x: heights and horizontal distances do not depend
on x, so nothing is guessed. events.json records this as table_x_known = false.

    d_axis(i)  horizontal palm distance to the INITIAL cup axis (vertical line through the initial base centre, S6)
    d_final(i) the same to the FINAL cup axis
    z(i)       palm height above the table (the table frame origin is on the plane)

    contact  first valid frame with d_axis <= r + 0.05 m (r = mug radius from config)
    lift     first valid k > contact with z >= m_k + 0.015 m on k and on the 4 next valid frames, m_k = min z over the
             valid frames in [contact, k]. The running minimum lets the hand settle lower after contact; 5 frames
             ignore a short bump.
    grasp g  last valid frame with index <= lift - 3: the hand pose just before the cup starts to move.
    place    first valid frame after the transport peak (max z over valid frames from lift on) with z <= z(g) + 0.01 m
    release  first valid frame after place with d_final >= d_final(place) + 0.03 m: the hand moves away from the
             placed cup (the FINAL axis, since the cup is no longer at its initial position).

Lift confirmation (recorded, never used to move the lift): the IoU of the SAM 2 body mask with the initial-window
majority mask (S6) drops below 0.7 within 0.3 s of the lift -- the cup leaves its initial footprint.

phase: 0 approach [0, contact), 1 grasp [contact, lift), 2 transport [lift, place), 3 place [place, release),
4 release/retreat [release, N). A span whose bounding event is undefined is -1. An undefined event has index -1 and
t = null, with the reason in events.json.

Writes data/work/<clip>/events/: events.json, events.npz (phase, d_axis, d_final, z, body_iou), events.png.
"""

from __future__ import annotations

import json

import numpy as np
from matplotlib.figure import Figure

from perception.common import WORK, cfg, load_config, stage_cli, stage_dir
from perception.geometry import plane_basis, transform
from perception.masks_sam2 import BODY, load_masks

EVENTS = ("contact", "lift", "grasp", "place", "release")
CONTACT_MARGIN_M = 0.05     # contact: d_axis <= r + this
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
    """Corrected keypoints (S5) in the table frame: (N, 21, 3), NaN where not metric_valid; and metric_valid (N,)."""
    T_wc = np.load(WORK / clip / "frames" / "meta.npz")["T_world_cam"]
    hm = np.load(WORK / clip / "hand_metric" / "hand_metric.npz")
    valid = hm["metric_valid"].astype(bool)
    T_tc = T_tw @ T_wc                                         # camera -> table, per frame
    kp = np.einsum("nij,nkj->nki", T_tc[:, :3, :3], hm["kp3d"].astype(np.float64)) + T_tc[:, None, :3, 3]
    kp[~valid] = np.nan
    return kp, valid


def detect_events(valid, z, d_axis, d_final, r):
    """Event indices (-1 = undefined) and the reason for each undefined event. Only valid frames are looked at."""
    ev, why = {}, {}
    vi = np.flatnonzero(valid)

    near = vi[d_axis[vi] <= r + CONTACT_MARGIN_M]
    if len(vi) == 0:
        why["contact"] = "no metric_valid frames"
    elif len(near) == 0:
        why["contact"] = f"d_axis never <= r + {CONTACT_MARGIN_M} m = {r + CONTACT_MARGIN_M:.3f} m " \
                         f"(min {d_axis[vi].min():.3f} m)"
    else:
        ev["contact"] = int(near[0])

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


def plot(path, clip, t, valid, d_axis, d_final, z, iou, ev, r, z_grasp):
    s = t - t[0]
    fig = Figure(figsize=(11, 8.5))
    ax = fig.subplots(4, 1, sharex=True, height_ratios=[3, 3, 2, 1])
    ax[0].plot(s, d_axis, color=BLUE, lw=1.5, label="to initial cup axis")
    ax[0].plot(s, d_final, color=ORANGE, lw=1.5, label="to final cup axis")
    ax[0].axhline(r + CONTACT_MARGIN_M, color=GREY, ls=":", lw=1, label="r + 5 cm (contact)")
    ax[0].set_ylabel("horizontal palm\ndistance [m]")
    ax[1].plot(s, z, color=BLUE, lw=1.5, label="palm z")
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
    # events in time order; labels alternate between two heights because grasp and lift are 3 frames apart
    timed = sorted((ev[e], e) for e in EVENTS if ev[e] >= 0)
    for j, (i, name) in enumerate(timed):
        for a in ax:
            a.axvline(s[i], color=INK, lw=0.8, alpha=0.7)
        ax[0].text(s[i], 1.03 + 0.09 * (j % 2), name, transform=ax[0].get_xaxis_transform(), ha="center", fontsize=9)
    for a in ax[:3]:
        a.legend(loc="upper right", fontsize=8, frameon=False)
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"{clip}: events from the raw metric palm (table frame)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def run(clip: str) -> dict:
    out = stage_dir(clip, "events")
    t = np.load(WORK / clip / "frames" / "meta.npz")["t"]
    n = len(t)
    r = cfg(load_config(), "mug", "body_outer_diameter_m") / 2
    T_tw, x_known = table_from_world(clip)          # x_known False: stand-in x, fine for d_axis, d_final and z
    kp, valid = hand_kp_table(clip, T_tw)
    obj = json.loads((WORK / clip / "objects" / "objects.json").read_text())
    cup_init = transform(T_tw, np.array(obj["init"]["base_centre_world"]))
    cup_final = transform(T_tw, np.array(obj["final"]["base_centre_world"]))
    palm = 0.5 * (kp[:, 0] + kp[:, 9])
    d_axis = np.linalg.norm(palm[:, :2] - cup_init[:2], axis=1)
    d_final = np.linalg.norm(palm[:, :2] - cup_final[:2], axis=1)
    z = palm[:, 2]
    ev, why = detect_events(valid, z, d_axis, d_final, r)
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
              "lift_confirm": {"iou_below_0p7_within_0p3s": confirm, "min_iou_in_window": min_iou},
              "z_grasp_m": float(z_grasp) if np.isfinite(z_grasp) else None,
              "table_x_known": x_known, "reasons": why}
    (out / "events.json").write_text(json.dumps(result, indent=1))
    np.savez(out / "events.npz", phase=phase, d_axis=d_axis, d_final=d_final, z=z, body_iou=iou)
    plot(out / "events.png", clip, t, valid, d_axis, d_final, z, iou, ev, r, z_grasp)

    for e, reason in why.items():
        print(f"{clip}: {e} undefined -- {reason}")
    t_rel = {f"t_{e}_rel_s": round(float(t[ev[e]] - t[0]), 3) if ev[e] >= 0 else None for e in EVENTS}
    return {**{e: ev[e] for e in EVENTS}, **t_rel, "z_grasp_m": result["z_grasp_m"], "lift_confirm": confirm,
            "min_iou_in_window": min_iou, "table_x_known": x_known, "n_metric_valid": int(valid.sum()),
            "phase_counts": {int(p): int(np.count_nonzero(phase == p)) for p in range(-1, 5)}, "reasons": why}


if __name__ == "__main__":
    stage_cli("events", run)
