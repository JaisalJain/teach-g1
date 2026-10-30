"""M4 Step 3: the hand's clearance from the mug along the approach. Kinematic, information only: never a gate.

Question: does the planned approach itself sweep the open hand through the mug before the grasp? M1's approach knocks (the
cup pushed 2-5 cm before any finger closed, Track B final trials 34, 38, 44, 45, 49) were never explained; this measures, per
run, how close the reference the robot is asked to track brings the hand to the mug on the way in.

For one run (``collision_min``):
1. The single-plan IK clip of M3's harness (retarget/run_m3.py run_one, steps 1-3): PickPlace.run's setup up to its first move
   (warp.standing_palm, which M3's verify showed bit-identical to run_m3's SetupOnly), then ONE
   plan_path(base="plan", base_goal=task._goal(G, R_G), return_fk=True) over the whole path, its first sample checked against
   the IK's palm_start (< 1 mm, M3's assert).
2. The frame window: from the prelude's last frame to the last frame whose remaining path arc length to G (along the path's
   palm targets, to frame i_G) is still > 1 cm. Why stop 1 cm short of G: at G the open hand is meant to touch the mug (the
   oracle's own exact grasp pose is at -2.4 mm, results/m1_grasp_fix/v3/l1_check.json), so the last centimetre would report
   the intended grasp, not a knock on the way in.
3. For each frame in the window, as M3's step 6 does at i_G: scene.reset (both hands at HERO's open pose, the mug at its layout
   pose), the robot set to that clip frame (run_m3.set_robot_to_clip_frame, HERO's clip -> plant mapping), mj_forward, the
   plant palm checked against the clip's FK palm (< 1 mm), and the minimum signed distance (mj_geomDistance; negative =
   overlap; meshes as convex hulls) from the 8 right-hand collision geoms to the mug body and to its 3 handle capsules
   (run_m3.hand_mug_distances).
Reported: the minimum over the window (and where), plus the body and handle minima separately.

``collision_min`` resets the task's scene and sets its planner goal; a physics run after it must start with its own full setup
(PickPlace.run / Execute.run do: scene.reset, standing reference, policy reset).
"""

from __future__ import annotations

import mujoco
import numpy as np

from retarget.run_m3 import MAPPING_TOL_M, START_TOL_M, hand_mug_distances, set_robot_to_clip_frame
from retarget.warp import standing_palm

ARC_STOP_M = 0.01               # the window ends where the approach is within 1 cm of arc length of G (spec Step 3)
METHOD = ("M3 harness single-plan IK clip (standing_palm setup, one plan_path(base='plan', base_goal=_goal(G, R_G), return_fk=True)); "
          "frames from the prelude's last frame to the last frame with remaining path arc length to G > 1 cm; per frame: scene.reset "
          "(open hands, mug at its layout pose), set_robot_to_clip_frame, mj_forward, min mj_geomDistance from the 8 right-hand geoms "
          "to cup_geom and the cup_handle_* capsules; information only, never a gate")


def approach_window(path: dict) -> tuple[int, int, np.ndarray]:
    """(first frame, last frame, remaining arc length to G per frame up to i_G). First = the prelude's last frame; last = the
    last frame whose remaining arc length (sum of |p[k+1] - p[k]| from that frame to i_G) is > ARC_STOP_M."""
    pos, phase = np.asarray(path["palm_pos"], float), np.asarray(path["phase"])
    i_G = int(path["i_G"])
    i0 = int(np.flatnonzero(phase == "prelude")[-1])
    seg = np.linalg.norm(np.diff(pos[:i_G + 1], axis=0), axis=1)
    remaining = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])          # remaining[i] = arc from frame i to i_G
    inside = np.flatnonzero(remaining[i0:] > ARC_STOP_M)
    if not len(inside):
        raise ValueError("the path is within 1 cm of arc length of G already at the prelude's end")
    return i0, i0 + int(inside[-1]), remaining


def collision_min_from_clip(scene, ref, path: dict, layout: dict) -> dict:
    """Step 3 on an already planned clip ``ref`` (frames = path frames), e.g. the clip run_m3.run_one planned. See module docstring."""
    i0, i1, remaining = approach_window(path)
    phase = np.asarray(path["phase"])
    m, d = scene.model, scene.data
    scene.reset(layout["cup_xy"], layout["coaster_xy"], layout["handle_yaw"])
    best = dict(min_m=np.inf)
    body, handle = dict(min_m=np.inf), dict(min_m=np.inf)
    map_max = 0.0
    for i in range(i0, i1 + 1):
        set_robot_to_clip_frame(scene, ref, i)
        mujoco.mj_forward(m, d)
        map_max = max(map_max, float(np.linalg.norm(scene.plant.palm_pose()[0][1] - ref.palm_pose_w(i)[0][1])))
        dist = hand_mug_distances(scene)
        for key, rec in (("hand_mug", body), ("hand_handle", handle)):
            if dist[key]["min_m"] < rec["min_m"]:
                rec.update(min_m=dist[key]["min_m"], frame=i, hand_body=dist[key]["hand_body"], object_geom=dist[key]["object_geom"])
    for rec in (body, handle):
        if rec["min_m"] < best["min_m"]:
            best = dict(rec)
    passed = map_max < MAPPING_TOL_M
    return dict(
        min_m=float(best["min_m"]) if passed else None, min_mm=float(best["min_m"] * 1e3) if passed else None,
        frame=best["frame"], phase=str(phase[best["frame"]]), remaining_arc_to_G_m=float(remaining[best["frame"]]),
        hand_body=best["hand_body"], object_geom=best["object_geom"],
        body=dict(min_mm=float(body["min_m"] * 1e3), frame=body["frame"], hand_body=body["hand_body"]),
        handle=dict(min_mm=float(handle["min_m"] * 1e3), frame=handle["frame"], hand_body=handle["hand_body"], object_geom=handle["object_geom"]),
        window=[int(i0), int(i1)], n_frames=int(i1 - i0 + 1), i_G=int(path["i_G"]), arc_stop_m=ARC_STOP_M,
        mapping_max_mm=map_max * 1e3, mapping_passed=passed, method=METHOD)


def collision_min(task, layout: dict, path: dict) -> dict:
    """Step 3 for one run. ``task`` = a PickPlace (any subclass) with the mug_v1 profile; ``layout`` = dict(cup_xy, coaster_xy,
    handle_yaw, ...); ``path`` = the run's path dict (v2_path / oracle_path output: palm_pos, palm_quat_wxyz, phase, i_G,
    info["G"], info["quat_G_wxyz"]), built from the standing palm of this same setup. Returns collision_min_from_clip's dict
    plus the plan's info and the start checks."""
    p_s, _, st = standing_palm(task, layout)
    s = task.scene
    pos, quat = np.asarray(path["palm_pos"], float), np.asarray(path["palm_quat_wxyz"], float)
    G, quat_G = np.asarray(path["info"]["G"], float), np.asarray(path["info"]["quat_G_wxyz"], float)
    task.planner.goal = task._goal(G, quat_G)
    p0, _ = task.planner.palm_start(st)
    start_err = float(np.linalg.norm(pos[0] - p0))
    assert start_err < START_TOL_M, f"path sample 0 is {start_err * 1e3:.3f} mm from palm_start"
    ref, info, _ = task.planner.plan_path(st, s.plant, pos, quat, base="plan", return_fk=True)
    out = collision_min_from_clip(s, ref, path, layout)
    out.update(plan=info, path_start_minus_palm_start_mm=start_err * 1e3, path_start_minus_p_s_m=float(np.abs(pos[0] - p_s).max()))
    return out
