"""M3 A1: frames and axes -- how my table-frame demo maps onto the sim world and onto the robot's palm.

Human table frame -> robot world (design choice): identity rotation plus a per-layout translation. Both frames have z up and
the same "forward": table x = my forward (away from the near desk edge), y = my left (perception S7); the robot stands at the
world origin facing +x with the table in front of it, so the robot stands where I sat. The table plane z = 0 maps to the sim
table top, TABLE_TOP = 0.74 m. The translation is chosen per layout (warp.py, naive.py): it puts my mug's base centre onto
the sim mug's base centre (``human_to_sim_offset``).

Cup positions: the demo stores the mug BASE centre (on the table plane); the sim cup position (``TableScene.cup_pose``) is the
body CENTRE, cup_height / 2 higher. ``base_to_centre`` / ``centre_to_base`` convert explicitly, so the two are never mixed up.

Palm frames:
- robot (HERO): palm pose = right ``wrist_yaw_link`` pose + PALM_OFFSET (``sim2sim.plant.MujocoPlant.palm_pose``). The oracle
  grasp puts the cup axis at o = (0.077, r + 0.015, 0) in this frame, so palm x should point toward the finger bases and palm
  y (the palm normal) toward the object. ``verify_palm_axes`` checks both on the model; M3 stops if either fails.
- human (perception S9, ``palm_rot_table``): columns [x_f, y_h, n]; x_f = wrist -> middle-finger base, n = palm normal
  (toward the cup at grasp; ``palm_normal_flipped`` is False in every clip), y_h = n x x_f.
- mapping: R_robot = R_human @ M with M's columns (e1, e3, -e2): robot x = x_f, robot y = n, robot z = -y_h = x_f x n.
Heading = the horizontal angle of the palm normal: the y column of a robot rotation, the n column of a human one.

Run (writes results/m3_retarget/palm_axes.json; exit code 1 and "PALM_AXES_FAIL" if a condition fails):
  env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.frames
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

from sim.pick_place import TABLE_HEIGHT

TABLE_TOP = TABLE_HEIGHT                      # the human table plane z = 0 lands here
DEMOS = Path("data/demos")

# Columns (e1, e3, -e2): robot x = human x_f, robot y = human n, robot z = -human y_h (= x_f x n, so M is a rotation)
M = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])
assert np.array_equal(M[:, 0], [1, 0, 0]) and np.array_equal(M[:, 1], [0, 0, 1]) and np.array_equal(M[:, 2], [0, -1, 0])
assert np.linalg.det(M) == 1.0


def load_demo(clip: str) -> dict:
    """Spec A0: data/demos/<clip>.npz with allow_pickle=False (every field is a plain array or string), as a plain dict."""
    with np.load(DEMOS / f"{clip}.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def nearest_frame(t: np.ndarray, time_s: float) -> int:
    """Spec A0: an event time maps to the nearest frame index."""
    if not math.isfinite(float(time_s)):
        raise ValueError("event time is NaN (perception did not detect it)")
    return int(np.argmin(np.abs(np.asarray(t) - float(time_s))))


def robot_rot(R_h: np.ndarray) -> np.ndarray:
    """Human palm rotation(s) [x_f, y_h, n] (..., 3, 3) -> robot palm rotation(s) [x, y, z] = R_h @ M."""
    return np.asarray(R_h, dtype=float) @ M


def heading_robot(R: np.ndarray) -> float:
    """Horizontal angle of the robot palm normal (the y column): atan2(R[1,1], R[0,1])."""
    return math.atan2(R[1, 1], R[0, 1])


def heading_human(R_h: np.ndarray) -> float:
    """Horizontal angle of the human palm normal n (column 2): atan2(R_h[1,2], R_h[0,2]). Equals heading_robot(R_h @ M)."""
    return math.atan2(R_h[1, 2], R_h[0, 2])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_diff(a: float, b: float) -> float:
    """a - b wrapped to (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def table_to_world(p: np.ndarray, offset_xy) -> np.ndarray:
    """Table-frame point(s) (..., 3) -> sim world: identity rotation, + (offset_x, offset_y, TABLE_TOP)."""
    return np.asarray(p, dtype=float) + np.array([offset_xy[0], offset_xy[1], TABLE_TOP])


def base_to_centre(p_base: np.ndarray, cup_height: float) -> np.ndarray:
    """Mug base centre (on the table) -> body centre (what TableScene.cup_pose reports): cup_height / 2 up."""
    return np.asarray(p_base, dtype=float) + np.array([0.0, 0.0, cup_height / 2])


def centre_to_base(p_centre: np.ndarray, cup_height: float) -> np.ndarray:
    return np.asarray(p_centre, dtype=float) - np.array([0.0, 0.0, cup_height / 2])


def human_to_sim_offset(demo: dict, cup_xy, cup_height: float) -> np.ndarray:
    """The translation (x, y) for ``table_to_world`` that puts my mug's base centre onto the sim mug's base centre.

    ``cup_xy`` = the sim cup CENTRE's xy (a layout's ``cup_xy``); the sim base is that centre cup_height / 2 lower, i.e. on the
    table top. The demo's mug base lies on the table plane (z = 0: the table frame's origin is the initial mug base centre).
    """
    human_base = np.asarray(demo["cup_init_pos_table"], dtype=float)
    assert abs(human_base[2]) < 1e-9, f"mug base not on the table plane: z = {human_base[2]}"
    sim_base = centre_to_base(np.array([cup_xy[0], cup_xy[1], TABLE_TOP + cup_height / 2]), cup_height)
    offset_xy = sim_base[:2] - human_base[:2]
    assert np.allclose(table_to_world(human_base, offset_xy), sim_base, atol=1e-9)
    return offset_xy


# ---------------------------------------------------------------------------------------------------------------- palm axes
class PalmAxesError(RuntimeError):
    """verify_palm_axes found a condition false; ``.result`` holds the full measurement (saved before stopping)."""

    def __init__(self, result: dict):
        super().__init__("PALM_AXES_FAIL: " + ", ".join(k for k, v in result["conditions"].items() if not v["pass"]))
        self.result = result


def verify_palm_axes(task=None) -> dict:
    """Spec A1: check on the model that robot palm x points to the fingers and palm y toward the cup at the oracle grasp pose.

    Kinematic only (no physics step). Scene: PickPlace with the mug_v1 profile, mug at the recorded-layout cup position (the
    palm-box centre minus the legacy palm offset, as in A4) with its handle along the oracle's approach direction (away from
    the palm; the centre of m1_pick_mug's handle spread). Pose: the oracle's own waypoints (z_clear point 1.2 s, pre-grasp
    1.8 s, grasp 2.0 s) through ONE ``PathPlanner.plan_path`` call (base "plan", base goal = the grasp), started from the spawn
    pose; the robot is then set to the path's last frame (root pose + 29 joints, HC.DOF_NAMES order at plant.dof_qadr, as
    HERO's sim2sim rollout maps a clip) with HERO's open right hand, and mj_forward. On that model state:
      (1) the mean of the right fingertip BODY positions (thumb_2, index_1, middle_1 link origins) has palm-x > 0;
      (2) the cup axis (the vertical line through the mug body centre, taken at the palm's height) has palm-y > 0.
    The palm frame is read from the plant (wrist_yaw_link + PALM_OFFSET), not from the target, so a convention mismatch
    between HERO's IK quaternion and the body frame would show. Returns the measurement; raises PalmAxesError if (1) or (2)
    fails.
    """
    import mujoco
    from data_tools import hero_reach_generator as gen
    from data_tools import reach_specs as rs
    from hero_isaacsim.paths import CHECKPOINTS_ROOT
    from sim2sim.interactive_scene import HAND_OPEN
    from sim2sim.mathutil import xyzw_to_wxyz
    from sim2sim.policy_hero_export import HeroExportPolicy
    from sim2sim.state import read_state
    from sim.pick_place import GRASP_PROFILES, PALM_BOX, PREGRASP_BACKOFF, Grasp, PickPlace
    from sim.reference_builder import min_jerk_path, quat_angle_deg
    from sim.scene import FINGERTIP_LINKS, SIDES

    t0 = time.perf_counter()
    task = task or PickPlace(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), **GRASP_PROFILES["mug_v1"])
    s, g = task.scene, task.grasp
    h = s.cup_height
    cup_xy = np.array([np.mean(PALM_BOX["x"]), np.mean(PALM_BOX["y"])]) - Grasp.hero_side_grasp(h).palm_offset[:2]
    yaw = math.atan2(g.approach_dir[1], g.approach_dir[0])
    s.reset(cup_xy, None, yaw)
    c = base_to_centre(np.array([*cup_xy, TABLE_TOP]), h)        # nominal resting centre (the spawn adds 1 mm)
    grasp_p, quat = g.palm_for_cup(c), g.quat_wxyz

    # the oracle's pregrasp + approach waypoints, planned in one call
    st = read_state(s.plant)
    task.planner.goal = task._goal(grasp_p, quat)
    p0, q0 = task.planner.palm_start(st)
    pre_p = grasp_p - PREGRASP_BACKOFF * g.approach_dir
    q_now = xyzw_to_wxyz(st.palm_quat_w[1])
    z_clear = max(gen._clearance_height(q_now, quat, "right", TABLE_TOP), pre_p[2])
    path_pos, path_quat = min_jerk_path(p0, q0, [(np.array([p0[0], p0[1], z_clear]), quat, 1.2), (pre_p, quat, 1.8), (grasp_p, quat, 2.0)])
    ref, info = task.planner.plan_path(st, s.plant, path_pos, path_quat, base="plan")
    i = len(path_pos) - 1

    # robot at that frame: plant.reset would also reset the mug and the fingers (mj_resetData), so restore them after it
    cup_qpos = s.data.qpos[s.cup_qadr:s.cup_qadr + 7].copy()
    s.plant.reset(ref.root_pos_from_joint_pos[i], ref.root_quat_from_joint_pos[i], ref.joint_pos[i])
    s.data.qpos[s.finger_qadr] = np.concatenate([HAND_OPEN[side] for side in SIDES])
    s.data.qpos[s.cup_qadr:s.cup_qadr + 7] = cup_qpos
    mujoco.mj_forward(s.model, s.data)

    palm_p, palm_q = s.plant.palm_pose()
    p_palm, q_palm = palm_p[1], xyzw_to_wxyz(palm_q[1])
    R_palm = rs.quat_wxyz_to_mat(q_palm)
    in_palm = lambda x: R_palm.T @ (np.asarray(x, dtype=float) - p_palm)
    tips = {f: in_palm(s.data.xpos[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, f"right_hand_{f}_link")]) for f in FINGERTIP_LINKS}
    tip_mean = np.mean(list(tips.values()), axis=0)
    cup_c = s.cup_pose()[0]
    axis_pt = in_palm([cup_c[0], cup_c[1], p_palm[2]])
    fk_p, fk_q = ref.palm_pose_w(i)

    conditions = {
        "fingertip_mean_palm_x_positive": {"value_m": float(tip_mean[0]), "pass": bool(tip_mean[0] > 0)},
        "cup_axis_palm_y_positive": {"value_m": float(axis_pt[1]), "pass": bool(axis_pt[1] > 0)},
    }
    result = dict(
        passed=all(v["pass"] for v in conditions.values()),
        conditions=conditions,
        fingertip_bodies_in_palm_m={f: v.round(5).tolist() for f, v in tips.items()},
        fingertip_mean_in_palm_m=tip_mean.round(5).tolist(),
        cup_axis_at_palm_height_in_palm_m=axis_pt.round(5).tolist(),
        oracle_cup_axis_in_palm_o_m=task.cup_in_palm.round(5).tolist(),
        pose=dict(layout_cup_xy=cup_xy.round(5).tolist(), handle_yaw_rad=yaw, oracle_grasp_palm=grasp_p.round(5).tolist(),
                  oracle_grasp_quat_wxyz=quat.round(6).tolist(), plant_palm=p_palm.round(5).tolist(), plant_palm_quat_wxyz=q_palm.round(6).tolist(),
                  plant_palm_minus_target_mm=float(np.linalg.norm(p_palm - grasp_p) * 1e3),
                  plant_palm_vs_target_deg=quat_angle_deg(q_palm, quat),
                  plant_palm_minus_clip_fk_palm_mm=float(np.linalg.norm(p_palm - fk_p[1]) * 1e3),
                  plant_palm_vs_clip_fk_deg=quat_angle_deg(q_palm, xyzw_to_wxyz(fk_q[1])),
                  path_frames=len(path_pos), plan_info=info),
        method=("PickPlace(mug_v1), spawn pose (no physics), oracle waypoints (z_clear 1.2 s, pre-grasp 1.8 s, grasp 2.0 s) in one "
                "plan_path(base='plan'); robot set to the last path frame (plant.reset with the clip's root pose + 29 joints), right and "
                "left hands at HERO's open pose, mug at its layout pose, mj_forward; palm frame = plant.palm_pose() (wrist_yaw_link + "
                "PALM_OFFSET); fingertips = body origins of right_hand_{thumb_2,index_1,middle_1}_link"),
        resources=dict(wall_s=round(time.perf_counter() - t0, 2), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)),
    )
    if not result["passed"]:
        raise PalmAxesError(result)
    return result


def main():
    out = Path("results/m3_retarget/palm_axes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = verify_palm_axes()
    except PalmAxesError as e:
        out.write_text(json.dumps(e.result, indent=2))
        print(f"PALM_AXES_FAIL -- {e}; wrote {out}. STOP (spec A1).")
        sys.exit(1)
    out.write_text(json.dumps(result, indent=2))
    c = result["conditions"]
    print(f"palm axes pass: fingertip mean palm-x {c['fingertip_mean_palm_x_positive']['value_m'] * 100:.2f} cm, cup axis palm-y "
          f"{c['cup_axis_palm_y_positive']['value_m'] * 100:.2f} cm; plant palm {result['pose']['plant_palm_minus_target_mm']:.2f} mm / "
          f"{result['pose']['plant_palm_vs_target_deg']:.2f} deg from the oracle grasp; wrote {out}")


if __name__ == "__main__":
    main()
