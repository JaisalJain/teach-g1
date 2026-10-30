"""M3 A3 (a): the naive kinematic replay -- my palm path and wrist orientation played back as recorded (the ablation baseline).

The translation puts my mug's base onto the sim mug's base of the RECORDED layout and is then fixed: the path ignores where
the sim mug and coaster actually are, so on any other layout it grasps at the recorded spot.
  prelude   as in (b): p_s -> (p_s.x, p_s.y, z_clear) in 1.2 s, turning to R_G. Here G / R_G = my mapped palm pose at the grasp
            frame g (the pose this replay grasps with), so z_clear = max(HERO's table-clearing height, G_z).
  to_a0     a 1.0 s min-jerk move to my mapped pose at a0 (position and R_h(a0) M).
  replay    my mapped palm position and R_h(t) M from a0 to t_release, time-scaled by k (the same k as (b): warp.time_scale
            at the recorded layout), resampled at 50 Hz: positions linearly, orientations by slerp between the two
            neighbouring frames. No warp, clamp or pause.
  release   closure 1 -> 0 over 0.6 s starting at the mapped t_release, the palm holding my last pose (the replay ends at
            t_release, so these 30 frames are appended).
Closure: 0 -> 1 over 1.25 s (the oracle's rate) starting at the mapped t_contact.
Indices: i_G / i_P / i_release = the frames of the mapped g / t_place / t_release; i_pause = -1 (no pause).
"""

from __future__ import annotations

import numpy as np

from sim2sim.bench.replan import slerp_wxyz
from sim.reference_builder import min_jerk_path
from sim.scene import mug_config

from retarget.frames import human_to_sim_offset, robot_rot, table_to_world
from retarget.grasp import oracle_reference
from retarget.warp import (CLOSE_RAMP_S, FPS, RELEASE_S, PathBuilder, approach_start, continuous, event_frames, intervals, n_frames,
                           prelude, quat_of, resample, time_scale)

TO_A0_S = 1.0


def resample_quats(t: np.ndarray, quats: np.ndarray, i0: int, th: np.ndarray) -> np.ndarray:
    """Orientations at human times ``th`` (inside t[i0 .. i0 + len(quats) - 1]): slerp between the two neighbouring frames."""
    ts = t[i0:i0 + len(quats)]
    out = []
    for x in th:
        i = int(np.clip(np.searchsorted(ts, x, side="right") - 1, 0, len(ts) - 2))
        w = float(np.clip((x - ts[i]) / (ts[i + 1] - ts[i]), 0.0, 1.0))
        out.append(slerp_wxyz(quats[i], quats[i + 1], w))
    return np.array(out)


def naive_path(demo: dict, recorded_layout: dict, p_s, q_now_wxyz, *, oracle: dict | None = None) -> dict:
    """Variant (a) for the recorded layout (dict: name, cup_xy = sim cup centre xy, coaster_xy, handle_yaw, cup_height).
    Returns the spec's npz fields (palm_pos, palm_quat_wxyz, closure, phase, i_pause = -1, i_G, i_P, i_release) and ``info``."""
    h = float(recorded_layout["cup_height"])
    oracle = oracle or oracle_reference(dict(r=mug_config()["body_outer_diameter_m"] / 2, h=h))
    t = demo["t"]
    ev = event_frames(demo)
    g, a0, ir = ev["g"], approach_start(demo, ev["g"]), ev["release"]
    ts = time_scale(demo, p_s, q_now_wxyz, recorded_layout, oracle)
    k = ts["k"]

    offset_xy = human_to_sim_offset(demo, np.asarray(recorded_layout["cup_xy"], float), h)
    p_tilde = table_to_world(demo["palm_pos_table"], offset_xy)
    R_rob = robot_rot(demo["palm_rot_table"][a0:ir + 1])            # my wrist, mapped to the robot palm axes
    assert np.isfinite(R_rob).all(), "palm rotation undefined between a0 and t_release"
    q_rob = continuous(np.array([quat_of(R) for R in R_rob]))
    G, R_G = p_tilde[g], R_rob[g - a0]
    quat_G = q_rob[g - a0]

    path = PathBuilder()
    pre_pos, pre_quat, p_r0, z_clear = prelude(p_s, q_now_wxyz, G, quat_G)
    path.add(pre_pos, pre_quat, 0.0, "prelude")
    mv_pos, mv_quat = min_jerk_path(p_r0, quat_G, [(p_tilde[a0], q_rob[0], TO_A0_S)])
    base = path.add(mv_pos[1:], mv_quat[1:], 0.0, "to_a0")          # = the frame of replay sample 0 (my pose at a0)

    n = intervals(k, t, a0, ir)
    rp, th = resample(t, p_tilde, a0, ir, n)
    rq = resample_quats(t, q_rob, a0, th)
    path.add(rp[1:], rq[1:], 0.0, "replay")
    frame_of = lambda i: base + int(round((t[i] - t[a0]) / (t[ir] - t[a0]) * n))     # my frame i -> path frame
    i_G, i_P, i_release = frame_of(g), frame_of(ev["place"]), base + n
    i_contact = base + max(0, int(round((t[ev["contact"]] - t[a0]) / (t[ir] - t[a0]) * n)))
    path.add(np.repeat(rp[-1:], n_frames(RELEASE_S), axis=0), rq[-1], 0.0, "release")

    out = path.arrays()
    i = np.arange(len(out["closure"]))
    up = np.clip((i - i_contact) / (CLOSE_RAMP_S * FPS), 0.0, 1.0)
    down = np.clip(1.0 - (i - i_release) / n_frames(RELEASE_S), 0.0, 1.0)
    out["closure"] = np.minimum(up, down)
    out.update(i_pause=-1, i_G=int(i_G), i_P=int(i_P), i_release=int(i_release))
    out["info"] = dict(
        variant="a_naive", layout=recorded_layout.get("name"), clip=str(demo["clip_id"]),
        k=k, time_scale=ts, a0=a0, a0_time_s=float(t[a0] - t[0]), a0_z_m=float(demo["palm_pos_table"][a0, 2]), events=ev,
        n_clamped=0, bump_scale=None, contact_before_a0=bool(t[ev["contact"]] < t[a0]), i_contact=int(i_contact),
        G=G.tolist(), R_G=R_G.tolist(), quat_G_wxyz=quat_G.tolist(), P=out["palm_pos"][i_P].tolist(),
        p_r0=p_r0.tolist(), z_clear=z_clear, human_to_sim_offset_xy=offset_xy.tolist(),
        frames=dict(prelude=len(pre_pos), to_a0=len(mv_pos) - 1, replay_intervals=n, release=n_frames(RELEASE_S), total=len(out["palm_pos"])),
        effective_k=n / (FPS * float(t[ir] - t[a0])))
    return out
