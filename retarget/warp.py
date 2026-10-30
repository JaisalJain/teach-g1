"""M3 A3 (b): the object-aware dense palm path -- my demo's palm motion, re-anchored on the sim mug and coaster, at 50 Hz.

The path, frame by frame (frame 0 = the standing palm p_s after PickPlace's 2 s hold):
  prelude    p_s -> (p_s.x, p_s.y, z_clear) in 1.2 s, turning to R_G: the oracle's first pregrasp waypoint. The robot starts
             with its arms at its sides, below the table top; I start with my hand already near the mug, so this part is robot-
             specific. z_clear = max(HERO's table-clearing height for this turn, G_z). Its end point is p_r0.
  approach   my approach (frames a0..g), time-scaled by k, warped so that it starts at p_r0 and ends at the grasp pose G:
             p(s) = p~(s) + (1 - w) (p_r0 - p~(0)) + w (G - p~(1)), w(s) = 10 s^3 - 15 s^4 + 6 s^5 (min-jerk blend), s = normalised
             time, p~ = my palm mapped into the world with the translation that puts my mug's base on the sim mug's base. Then
             z >= table top + 3 cm (clamped frames counted). Orientation is held at R_G for the whole approach (design choice:
             the human wrist rotation during the approach is not transferred; only the path shape is). A 0.5 s hold is inserted
             at the first sample whose remaining arc length to G is <= 8 cm (i_pause): M4 replaces it with the oracle's closed-
             loop settle, so the open-loop path stays as recorded and M4 only corrects the palm there.
  close      hold at G for 1.75 s (the oracle's closure ramp, 1 / 0.8 s, plus its 0.5 s hold); closure 0 -> 1 over 1.25 s.
  transport  my transport (frames t_lift..t_place), time-scaled by k: p(s) = p~(s) + (1 - s) (G - p~(0)) + s (P - p~(1)),
             P = G + (coaster - cup, COASTER_THICKNESS + PLACE_CLEARANCE) (the cup then sits 1 cm above the coaster, as in the
             oracle). If the path rises more than LIFT_HEIGHT (10 cm) above the straight G -> P line, the bump is scaled down
             uniformly to 10 cm (bump_scale). Orientation held at R_G.
  release    as the oracle: closure 1 -> 0 in 0.6 s, then a 0.6 s hold.
  retreat    as the oracle: by -0.10 u + (0, 0, 0.05) in 1.5 s (min-jerk), u = horizontal unit vector of q - p (palm -> cup
             axis) for this variant; then a 1.0 s hold.

Time scale k (per clip, at the recorded layout): v_h = the 95th percentile of my smoothed palm speed over the approach and the
transport segments; v_o = the larger min-jerk peak speed (1.875 |dp| / T) of the oracle's pregrasp leg (z_clear point ->
pre-grasp point, T = 1.8 s) and its transport leg (T = max(1.5, |dp| / 0.08)) with mug_v1; k = max(1, v_h / v_o). My times
are multiplied by k: the robot never moves faster than the oracle's own peak (95th percentile), and keeps my timing otherwise.
A segment of human duration D becomes n = round(k D 50) intervals of 20 ms; the positions are linearly interpolated at the
human times t0 + (j / n) D (so its effective scale is n / (50 D), within half a frame of k D).

Approach start a0: the first frame of the last contiguous run ending at g in which my palm is finite and z >= 2 cm. (The first
tracked frames lie below the table, an entry artefact; this finds a0 at z 0.022-0.068 m in all 4 clips.)

Frame indices: i_G = arrival at G (closure 0 there; the close ramp starts at i_G), i_P = arrival at P (= i_release: the
release ramp starts there, closure 1 at i_release and 0 thirty frames later), i_pause = the approach sample where the 0.5 s
hold starts (the palm stays there for the next 25 frames, phase "pause").

Run (results/m3_retarget/human_values.json: per clip phi_h, phi_ora, azimuth, h_frac_h, h_frac_ora, k, a0 time):
  env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.warp
"""

from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from data_tools import hero_reach_generator as gen
from data_tools import reach_specs as rs
from sim.pick_place import (CLOSE_RATE, LIFT_HEIGHT, PALM_BOX, PLACE_CLEARANCE, PREGRASP_BACKOFF, RETREAT_BACK, RETREAT_UP,
                            TRANSPORT_SPEED, Grasp)
from sim.reference_builder import min_jerk_path
from sim.scene import COASTER_THICKNESS, mug_config

from retarget.frames import TABLE_TOP, base_to_centre, human_to_sim_offset, load_demo, nearest_frame, table_to_world
from retarget.grasp import grasp_pose, human_at_grasp, oracle_reference

FPS = 50
A0_MIN_Z = 0.02                     # a0 rule: my palm at least 2 cm above the table plane
APPROACH_MIN_Z = TABLE_TOP + 0.03   # approach clamp
PAUSE_ARC_M = 0.08                  # pause where 8 cm of path are left (the oracle's pre-grasp backoff is also 8 cm)
PAUSE_S = 0.5
PRELUDE_S = 1.2                     # the oracle's first pregrasp waypoint
CLOSE_RAMP_S = 1.0 / CLOSE_RATE     # 1.25 s
CLOSE_S = CLOSE_RAMP_S + 0.5        # + the oracle's 0.5 s hold after full closure
RELEASE_S, RELEASE_HOLD_S = 0.6, 0.6
RETREAT_S, RETREAT_HOLD_S = 1.5, 1.0
ORACLE_PREGRASP_S = 1.8             # the oracle's z_clear point -> pre-grasp leg
ORACLE_TRANSPORT_MIN_S = 1.5
MIN_JERK_PEAK = 1.875               # peak speed of a min-jerk move = 1.875 |dp| / T
V_H_PERCENTILE = 95


def n_frames(seconds: float) -> int:
    """Frames for a duration at 50 Hz. 1.75 s -> 88 (round half to even), which is what the oracle does: 63 ramp ticks of
    0.8 / 50 (the 63rd reaches 1.0) + 25 hold ticks."""
    return int(round(seconds * FPS))


def min_jerk_w(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def quat_of(R: np.ndarray) -> np.ndarray:
    q = rs.mat_to_quat_wxyz(R)
    return q / np.linalg.norm(q)


def continuous(quats: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions have a positive dot product (same rotations; frame 0 keeps its sign)."""
    q = np.array(quats, dtype=float)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


# ---------------------------------------------------------------------------------------------------------------- my demo
def event_frames(demo: dict) -> dict:
    """g and the event frames (nearest frame to each event time) of a demo."""
    t = demo["t"]
    ev = dict(g=int(demo["grasp_index"]), **{k: nearest_frame(t, demo[f"t_{k}"]) for k in ("contact", "lift", "place", "release")})
    assert ev["g"] >= 0 and ev["g"] < ev["lift"] < ev["place"] < ev["release"], ev
    return ev


def approach_start(demo: dict, g: int) -> int:
    """a0 = the first frame of the last contiguous run ending at g with a finite palm at z >= 2 cm."""
    p = demo["palm_pos_table"]
    ok = np.isfinite(p).all(axis=1) & (np.nan_to_num(p[:, 2], nan=-1.0) >= A0_MIN_Z)
    if not ok[g]:
        raise ValueError(f"{demo['clip_id']}: palm not finite / above 2 cm at g")
    a0 = g
    while a0 > 0 and ok[a0 - 1]:
        a0 -= 1
    return a0


def resample(t: np.ndarray, values: np.ndarray, i0: int, i1: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``values`` (rows = frames) at n + 1 evenly spaced human times from t[i0] to t[i1], linearly interpolated per column.
    Returns (samples (n + 1, d), the human times)."""
    th = t[i0] + (t[i1] - t[i0]) * np.arange(n + 1) / n
    seg = np.asarray(values[i0:i1 + 1], dtype=float)
    assert np.isfinite(seg).all(), "non-finite palm sample in a segment"
    return np.column_stack([np.interp(th, t[i0:i1 + 1], seg[:, j]) for j in range(seg.shape[1])]), th


def intervals(k: float, t: np.ndarray, i0: int, i1: int) -> int:
    """Number of 50 Hz intervals for my segment i0..i1 slowed down by k."""
    return max(1, int(round(k * (t[i1] - t[i0]) * FPS)))


def recorded_layout(demo: dict, oracle: dict | None = None) -> dict:
    """Spec A4's recorded layout: cup_xy = PALM_BOX centre - the legacy palm offset (sample_layouts' mapping), coaster = cup + my
    (coaster - cup) offset, handle yaw = cup_init_yaw if valid, else phi_h. The place-palm flag is computed as the spec writes it,
    coaster - legacy palm offset (sample_layouts would give coaster + offset; see the report's open questions)."""
    h = (oracle or {}).get("h", mug_config()["height_m"])
    legacy = Grasp.hero_side_grasp(h)
    cup = np.array([np.mean(PALM_BOX["x"]), np.mean(PALM_BOX["y"])]) - legacy.palm_offset[:2]
    coaster = cup + (np.asarray(demo["coaster_pos_table"])[:2] - np.asarray(demo["cup_init_pos_table"])[:2])
    valid = bool(demo["cup_yaw_valid"][0]) and math.isfinite(float(demo["cup_init_yaw"]))
    yaw = float(demo["cup_init_yaw"]) if valid else human_at_grasp(demo, h)["phi_h"]
    place_palm = coaster - legacy.palm_offset[:2]
    inside = bool(PALM_BOX["x"][0] <= place_palm[0] <= PALM_BOX["x"][1] and PALM_BOX["y"][0] <= place_palm[1] <= PALM_BOX["y"][1])
    return dict(name="recorded", cup_xy=cup, coaster_xy=coaster, handle_yaw=yaw, cup_height=h,
                handle_yaw_source="cup_init_yaw" if valid else "phi_h", coaster_place_palm_xy=place_palm, coaster_place_palm_in_palm_box=inside)


def time_scale(demo: dict, p_s, q_now_wxyz, recorded: dict, oracle: dict) -> dict:
    """k = max(1, v_h / v_o) (module docstring). The oracle legs are its nominal ones at the recorded layout: cup at rest, the
    in-hand offset kept through the lift (the oracle itself measures them in the sim)."""
    t, p = demo["t"], demo["palm_pos_table"]
    ev = event_frames(demo)
    a0 = approach_start(demo, ev["g"])
    speeds = []
    for i0, i1 in ((a0, ev["g"]), (ev["lift"], ev["place"])):      # forward differences inside each segment
        seg = np.asarray(p[i0:i1 + 1], dtype=float)
        speeds.append(np.linalg.norm(np.diff(seg, axis=0), axis=1) / np.diff(t[i0:i1 + 1]))
    v = np.concatenate(speeds)
    assert np.isfinite(v).all()
    v_h = float(np.percentile(v, V_H_PERCENTILE))

    # the oracle's pregrasp leg: (p_s.x, p_s.y, z_clear) -> pre-grasp point, 1.8 s
    h = oracle["h"]
    c = base_to_centre(np.array([*recorded["cup_xy"], TABLE_TOP]), h)
    grasp_p = c + oracle["palm_offset"]
    pre_p = grasp_p - PREGRASP_BACKOFF * oracle["approach_dir"]
    z_clear = max(gen._clearance_height(np.asarray(q_now_wxyz, float), oracle["quat_ora_wxyz"], "right", TABLE_TOP), pre_p[2])
    d_pre = float(np.linalg.norm(pre_p - np.array([p_s[0], p_s[1], z_clear])))
    # its transport leg: lifted palm -> over the place palm (the oracle's above_p), T = max(1.5, |dp| / 0.08)
    lifted = grasp_p + np.array([0.0, 0.0, LIFT_HEIGHT])
    cup_in_hand = c - grasp_p
    cup_place = np.array([*recorded["coaster_xy"], TABLE_TOP + COASTER_THICKNESS]) + np.array([0.0, 0.0, h / 2 + PLACE_CLEARANCE])
    above_p = cup_place - cup_in_hand
    above_p[2] = lifted[2]
    d_tr = float(np.linalg.norm(above_p - lifted))
    T_tr = max(ORACLE_TRANSPORT_MIN_S, d_tr / TRANSPORT_SPEED)
    peak_pre, peak_tr = MIN_JERK_PEAK * d_pre / ORACLE_PREGRASP_S, MIN_JERK_PEAK * d_tr / T_tr
    v_o = max(peak_pre, peak_tr)
    return dict(k=max(1.0, v_h / v_o), v_h=v_h, v_o=v_o, n_speed_samples=int(len(v)), a0=a0,
                oracle_pregrasp_leg=dict(dist_m=d_pre, T_s=ORACLE_PREGRASP_S, peak_m_s=peak_pre, z_clear=float(z_clear)),
                oracle_transport_leg=dict(dist_m=d_tr, T_s=T_tr, peak_m_s=peak_tr),
                recorded_cup_xy=np.asarray(recorded["cup_xy"], float).tolist(), recorded_coaster_xy=np.asarray(recorded["coaster_xy"], float).tolist())


def prelude(p_s, q_now_wxyz, G: np.ndarray, quat_G: np.ndarray):
    """p_s -> (p_s.x, p_s.y, z_clear) in 1.2 s, turning to quat_G (the oracle's first pregrasp waypoint, min-jerk).
    Returns (pos (61, 3), quat (61, 4), p_r0, z_clear); sample 0 = (p_s, q_now)."""
    z_clear = max(gen._clearance_height(np.asarray(q_now_wxyz, float), quat_G, "right", TABLE_TOP), float(G[2]))
    p_r0 = np.array([p_s[0], p_s[1], z_clear])
    pos, quat = min_jerk_path(np.asarray(p_s, float), np.asarray(q_now_wxyz, float), [(p_r0, quat_G, PRELUDE_S)])
    return pos, quat, p_r0, float(z_clear)


class PathBuilder:
    """Collects the dense path frame by frame (position, quaternion, closure, phase)."""

    def __init__(self):
        self.pos, self.quat, self.closure, self.phase = [], [], [], []

    def add(self, pos, quat, closure, phase: str) -> int:
        """Append frames (pos (n, 3) or (3,); quat (n, 4) or (4,); closure scalar or (n,)). Returns the index of the last one."""
        pos = np.atleast_2d(np.asarray(pos, dtype=float))
        n = len(pos)
        self.pos += list(pos)
        self.quat += list(np.broadcast_to(np.asarray(quat, dtype=float), (n, 4)))
        self.closure += list(np.broadcast_to(np.asarray(closure, dtype=float), (n,)))
        self.phase += [phase] * n
        return len(self.pos) - 1

    def last(self) -> int:
        return len(self.pos) - 1

    def arrays(self) -> dict:
        return dict(palm_pos=np.array(self.pos), palm_quat_wxyz=continuous(np.array(self.quat)), closure=np.array(self.closure),
                    phase=np.array(self.phase))


# ---------------------------------------------------------------------------------------------------------------- (b) path
def object_aware_path(demo: dict, variant: str, layout: dict, p_s, q_now_wxyz, *, recorded: dict | None = None,
                      oracle: dict | None = None) -> dict:
    """Variant (b): the dense 50 Hz path for ``variant`` ("b_level" | "b_tilt") on ``layout`` (dict: name, cup_xy = sim cup
    centre xy, coaster_xy, handle_yaw, cup_height). p_s / q_now_wxyz = the standing palm after PickPlace's 2 s hold.
    k is computed at the recorded layout (``recorded``; default ``recorded_layout(demo)``). Returns the spec's npz fields
    (palm_pos, palm_quat_wxyz, closure, phase, i_pause, i_G, i_P, i_release) and ``info``."""
    h = float(layout["cup_height"])
    oracle = oracle or oracle_reference(dict(r=mug_config()["body_outer_diameter_m"] / 2, h=h))
    assert abs(oracle["h"] - h) < 1e-12, "layout and oracle disagree on the mug height"
    recorded = recorded or recorded_layout(demo, oracle)
    human = human_at_grasp(demo, h)
    t = demo["t"]
    ev = event_frames(demo)
    g, a0 = ev["g"], approach_start(demo, ev["g"])
    ts = time_scale(demo, p_s, q_now_wxyz, recorded, oracle)
    k = ts["k"]

    cup_xy, coaster_xy = np.asarray(layout["cup_xy"], float), np.asarray(layout["coaster_xy"], float)
    G, R_G, q = grasp_pose(variant, cup_xy, human, oracle)
    quat_G = quat_of(R_G)
    P = G + np.array([*(coaster_xy - cup_xy), COASTER_THICKNESS + PLACE_CLEARANCE])
    offset_xy = human_to_sim_offset(demo, cup_xy, h)
    p_tilde = table_to_world(demo["palm_pos_table"], offset_xy)      # my palm, my mug's base on the sim mug's base

    path = PathBuilder()
    pre_pos, pre_quat, p_r0, z_clear = prelude(p_s, q_now_wxyz, G, quat_G)
    path.add(pre_pos, pre_quat, 0.0, "prelude")

    # approach: warp my a0..g onto p_r0 -> G, clamp above the table, then find the pause sample
    n_a = intervals(k, t, a0, g)
    pt, _ = resample(t, p_tilde, a0, g, n_a)
    w = min_jerk_w(np.arange(n_a + 1) / n_a)[:, None]
    app = pt + (1 - w) * (p_r0 - pt[0]) + w * (G - pt[-1])
    low = app[:, 2] < APPROACH_MIN_Z
    app[low, 2] = APPROACH_MIN_Z
    remaining = np.concatenate([np.cumsum(np.linalg.norm(np.diff(app, axis=0), axis=1)[::-1])[::-1], [0.0]])
    j_pause = int(np.argmax(remaining <= PAUSE_ARC_M))         # remaining[-1] = 0, so a sample always qualifies
    i_pause = path.last() if j_pause == 0 else None            # sample 0 is p_r0 = the prelude's last frame
    if j_pause == 0:
        path.add(np.repeat(app[:1], n_frames(PAUSE_S), axis=0), quat_G, 0.0, "pause")
    for j in range(1, n_a + 1):                                # sample 0 = p_r0 is already there
        idx = path.add(app[j], quat_G, 0.0, "approach")
        if j == j_pause:
            i_pause = idx
            path.add(np.repeat(app[j:j + 1], n_frames(PAUSE_S), axis=0), quat_G, 0.0, "pause")
    i_G = path.last() if j_pause != n_a else i_pause           # arrival at G (a pause AT G comes after the arrival)

    # close: hold at G, closure ramps at the oracle's rate 0.8 / s from i_G
    n_c = n_frames(CLOSE_S)
    path.add(np.repeat(G[None], n_c, axis=0), quat_G, np.minimum(1.0, np.arange(1, n_c + 1) / (CLOSE_RAMP_S * FPS)), "close")

    # transport: warp my lift..place onto G -> P, then cap the height above the straight G -> P line at LIFT_HEIGHT
    n_t = intervals(k, t, ev["lift"], ev["place"])
    pt, _ = resample(t, p_tilde, ev["lift"], ev["place"], n_t)
    s = (np.arange(n_t + 1) / n_t)[:, None]
    tr = pt + (1 - s) * (G - pt[0]) + s * (P - pt[-1])
    line_z = (1 - s[:, 0]) * G[2] + s[:, 0] * P[2]
    bump = tr[:, 2] - line_z
    bump_max = float(bump.max())
    bump_scale = LIFT_HEIGHT / bump_max if bump_max > LIFT_HEIGHT else 1.0
    tr[:, 2] = line_z + bump_scale * bump
    i_P = path.add(tr[1:], quat_G, 1.0, "transport")          # sample 0 = G is the close hold's last frame

    # release and retreat, as the oracle
    n_r = n_frames(RELEASE_S)
    path.add(np.repeat(P[None], n_r, axis=0), quat_G, 1.0 - np.arange(1, n_r + 1) / n_r, "release")
    path.add(np.repeat(P[None], n_frames(RELEASE_HOLD_S), axis=0), quat_G, 0.0, "release")
    u = (q - G) * np.array([1.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    back = P - RETREAT_BACK * u + np.array([0.0, 0.0, RETREAT_UP])
    rp, rq = min_jerk_path(P, quat_G, [(back, quat_G, RETREAT_S)])
    path.add(rp[1:], rq[1:], 0.0, "retreat")
    path.add(np.repeat(back[None], n_frames(RETREAT_HOLD_S), axis=0), quat_G, 0.0, "retreat")

    out = path.arrays()
    out.update(i_pause=int(i_pause), i_G=int(i_G), i_P=int(i_P), i_release=int(i_P))
    out["info"] = dict(
        variant=variant, layout=layout.get("name"), clip=str(demo["clip_id"]),
        k=k, time_scale=ts, a0=a0, a0_time_s=float(t[a0] - t[0]), a0_z_m=float(demo["palm_pos_table"][a0, 2]), events=ev,
        n_clamped=int(low.sum()), G_clamped=bool(low[-1]), bump_max_m=bump_max, bump_scale=bump_scale,
        pause_remaining_arc_m=float(remaining[j_pause]),
        G=G.tolist(), R_G=R_G.tolist(), quat_G_wxyz=quat_G.tolist(), P=P.tolist(), q=q.tolist(), u=u.tolist(), retreat_to=back.tolist(),
        p_r0=p_r0.tolist(), z_clear=z_clear, human_to_sim_offset_xy=offset_xy.tolist(),
        frames=dict(prelude=len(pre_pos), approach_intervals=n_a, pause=n_frames(PAUSE_S), close=n_c, transport_intervals=n_t,
                    release=n_r + n_frames(RELEASE_HOLD_S), retreat=len(rp) - 1 + n_frames(RETREAT_HOLD_S), total=len(out["palm_pos"])),
        effective_k=dict(approach=n_a / (FPS * float(t[g] - t[a0])), transport=n_t / (FPS * float(t[ev["place"]] - t[ev["lift"]]))))
    return out


# ---------------------------------------------------------------------------------------------------------------- setup
def standing_palm(task, layout: dict):
    """PickPlace.run up to its first move (scene.reset, standing reference, policy reset, reset step, 2 s hold), then the
    right palm. Returns (p_s, q_now_wxyz, state)."""
    from sim2sim.mathutil import xyzw_to_wxyz
    from sim2sim.state import read_state
    from sim.m1_reach_test import standing_reference
    from sim.pick_place import Log

    s = task.scene
    s.reset(layout["cup_xy"], layout["coaster_xy"], layout["handle_yaw"])
    task.log = Log(s, None)
    task.plans, task.settles, task.k, task.t, task.corrector_active = [], [], 0, 0, False
    task.ref = standing_reference(s, 200)
    task.policy.reset()
    s.step(task.policy.reset_target_for(task.ref))
    task.hold("stand", 2.0)
    st = read_state(s.plant)
    return st.palm_pos_w[1].copy(), xyzw_to_wxyz(st.palm_quat_w[1]), st


def main():
    """Per clip: my values at g, the oracle's, k and a0 (p_s / q_now from a mug_v1 PickPlace setup at the recorded layout)."""
    from hero_isaacsim.paths import CHECKPOINTS_ROOT
    from sim2sim.policy_hero_export import HeroExportPolicy
    from sim.pick_place import GRASP_PROFILES, PickPlace

    rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    t0 = time.perf_counter()
    task = PickPlace(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), **GRASP_PROFILES["mug_v1"])
    oracle = oracle_reference(task.scene)
    out = dict(oracle=dict(r=oracle["r"], h=oracle["h"], o=oracle["o"].tolist(), phi_ora_rad=oracle["phi_ora"], phi_ora_deg=math.degrees(oracle["phi_ora"]),
                           h_frac_ora=oracle["h_frac_ora"], azimuth_ora_deg=math.degrees(oracle["azimuth_ora"])),
               clips={}, resources=dict(build_task_s=round(time.perf_counter() - t0, 2), build_task_peak_rss_mb=rss()))
    for clip in ("26d4118b0c", "4e103843db", "49aff04f7a", "e8825dea53"):
        t1 = time.perf_counter()
        demo = load_demo(clip)
        human = human_at_grasp(demo, oracle["h"])
        rec = recorded_layout(demo, oracle)
        p_s, q_now, _ = standing_palm(task, rec)
        ts = time_scale(demo, p_s, q_now, rec, oracle)
        t = demo["t"]
        out["clips"][clip] = dict(
            view=str(demo["view_label"]), g=human["g"], phi_h_rad=human["phi_h"], phi_h_deg=math.degrees(human["phi_h"]),
            phi_ora_rad=oracle["phi_ora"], phi_ora_deg=math.degrees(oracle["phi_ora"]),
            phi_h_minus_phi_ora_deg=math.degrees(math.atan2(math.sin(human["phi_h"] - oracle["phi_ora"]), math.cos(human["phi_h"] - oracle["phi_ora"]))),
            azimuth_deg=human["azimuth_info"]["azimuth_deg"], h_frac_h=human["h_frac_h"], h_frac_ora=oracle["h_frac_ora"],
            k=ts["k"], v_h_m_s=ts["v_h"], v_o_m_s=ts["v_o"], a0=ts["a0"], a0_time_s=float(t[ts["a0"]] - t[0]),
            a0_z_m=float(demo["palm_pos_table"][ts["a0"], 2]), time_scale=ts, events=event_frames(demo),
            recorded_layout=dict(cup_xy=rec["cup_xy"].tolist(), coaster_xy=rec["coaster_xy"].tolist(), handle_yaw_rad=rec["handle_yaw"],
                                 handle_yaw_source=rec["handle_yaw_source"]),
            standing_palm=dict(p_s=p_s.tolist(), q_now_wxyz=q_now.tolist()),
            resources=dict(wall_s=round(time.perf_counter() - t1, 2), peak_rss_mb_so_far=rss()))
        c = out["clips"][clip]
        print(f"{clip}: phi_h {c['phi_h_deg']:7.2f} deg (phi_ora {c['phi_ora_deg']:.2f}), azimuth {c['azimuth_deg']:7.2f} deg, "
              f"h_frac_h {c['h_frac_h']:.3f} (ora {c['h_frac_ora']:.3f}), k {c['k']:.3f} (v_h {c['v_h_m_s']:.3f}, v_o {c['v_o_m_s']:.3f} m/s), "
              f"a0 {c['a0']} at {c['a0_time_s']:.3f} s (z {c['a0_z_m'] * 100:.1f} cm)")
    out["resources"].update(total_wall_s=round(time.perf_counter() - t0, 2), peak_rss_mb=rss())
    path = Path("results/m3_retarget/human_values.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
