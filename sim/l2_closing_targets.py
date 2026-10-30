"""Track B v3, fix-ladder rung L2: Dex3 closing targets for my 99 mm mug, by fingertip IK. Kinematics only, nothing is simulated.

Why: the oracle closes the hand toward HERO's closed demo pose, one pose for every object, so how hard and where each digit
ends up pressing depends on where that pose happens to collide with the object. L2 gives each digit a target computed for
this mug instead: its fingertip 5 mm inside the mug surface, at the place where that digit reaches farthest around it.

Per digit (thumb = thumb_0/1/2, index = index_0/1, middle = middle_0/1):
- start: the digit's max-wrap configuration at 99 mm from B2v3 (results/m1_grasp_fix/v3/b2v3_capacity.json, row D 99 mm,
  digits.<digit>.argmax = the touching configuration that reaches farthest around the cylinder). Same scene and placement as
  B2v3 (sim/b2_capacity.py, ``Capacity(anchor="radius_consistent").place_cylinder(0.099)``): the cylinder axis at palm-frame
  (0.077, r + 0.015), which is where L1's grasp puts the mug body relative to the palm, so the targets fit the L1 grasp.
- fixed point: the point of the fingertip's collision geom closest to the cylinder in that configuration (mj_geomDistance
  witness point), carried in the fingertip link's frame.
- target: the closest point on the cylinder in that configuration (asserted equal to B2v3's saved point), moved 5 mm radially
  into the cylinder (horizontally toward the axis).
- IK: scipy least_squares over that digit's joints only, bounds = the joint ranges (so the angles stay in range), start = the
  max-wrap configuration; residual = FK position of the fixed point (mj_kinematics) - target, 3 components, palm frame. The
  thumb has 3 joints and can reach a 3-D point; index and middle flex in one plane (2 joints), so a residual may remain: it is
  reported, never hidden. Jacobian: scipy's finite differences (simple; the FK is exact to machine precision).
- the 7 angles (FINGER_SUFFIXES order) replace HERO's closed targets of the right hand: PickPlace(closing_targets="ik_mug")
  reads ``closed_targets_right`` from the JSON. Open pose, ramp, gains, torque limits and the squeeze rule are unchanged.
A digit without a touching configuration in B2v3 at 99 mm would be "not computable" (PickPlace then refuses the targets).

Information only (nothing is tuned on it): the target's depth below the cylinder surface (5 mm on the side, less if the point
was at a rim), each digit geom's signed distance to the cylinder at the IK solution, the joint change from the max-wrap
configuration, and HERO's closed pose for comparison.

Run:  env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.l2_closing_targets
      -> results/m1_grasp_fix/v3/l2_targets.json
"""

from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path

import mujoco
import numpy as np
import scipy
from scipy.optimize import least_squares
from sim2sim.interactive_scene import FINGER_SUFFIXES, HAND_CLOSED, HAND_OPEN

from sim.b2_capacity import DIGITS, MUG_D, SIDE, Capacity
from sim.pick_place import L2_TARGETS
from sim.scene import FLEXION_JOINT, GRIP_BLOCKED_ERROR_RAD

B2V3 = Path("results/m1_grasp_fix/v3/b2v3_capacity.json")
TARGET_DEPTH_M = 0.005                  # spec: each fingertip's target 5 mm inside the mug surface


def depth_inside(c: Capacity, p: np.ndarray) -> float:
    """How far the palm-frame point ``p`` is inside the cylinder (> 0 inside): the smallest of its distances to the side
    and to the two caps."""
    r, half = float(c.model.geom_size[c.cyl, 0]), c.height / 2
    rho = math.hypot(p[0] - c.axis[0], p[1] - c.axis[1])
    return min(r - rho, c.axis[2] + half - p[2], p[2] - (c.axis[2] - half))


def solve_digit(c: Capacity, digit: str, b2: dict) -> dict:
    """Rebuild the digit's B2v3 max-wrap configuration (``b2`` = its entry in B2v3's 99 mm row), place its target 5 mm inside
    the cylinder, solve the IK."""
    m, d, argmax = c.model, c.data, b2["argmax"]
    names = list(argmax["joints"])
    q0 = np.array(list(argmax["joints"].values()))
    d.qpos[c.all_finger_qadr] = 0.0                     # as in B2v3: the digits not being solved stay at 0
    d.qpos[c.qadr[digit]] = q0
    mujoco.mj_kinematics(m, d)
    fromto = np.zeros(6)
    gap = c.dist(c.tip[digit], fromto)                  # fromto = (point on the fingertip, point on the cylinder), world
    on_cyl = c.to_palm(fromto[3:])
    # the configuration is B2v3's: same fingertip gap and closest point (as B2v3 rounds them)
    assert round(gap * 1e3, 3) == argmax["tip_gap_mm"], (digit, gap, argmax["tip_gap_mm"])
    assert on_cyl.round(5).tolist() == argmax["closest_point_on_cylinder_palm_frame_m"], (digit, on_cyl)

    tip_body = int(m.geom_bodyid[c.tip[digit]])
    R_tip = d.xmat[tip_body].reshape(3, 3)
    fixed_local = R_tip.T @ (fromto[:3] - d.xpos[tip_body])     # the fixed point, in the fingertip link frame
    radial = np.array([on_cyl[0] - c.axis[0], on_cyl[1] - c.axis[1], 0.0])
    radial /= np.linalg.norm(radial)
    target = on_cyl - TARGET_DEPTH_M * radial

    def residual(q):
        d.qpos[c.qadr[digit]] = q
        mujoco.mj_kinematics(m, d)
        return c.to_palm(d.xpos[tip_body] + d.xmat[tip_body].reshape(3, 3) @ fixed_local) - target

    lo, hi = c.ranges[digit][:, 0], c.ranges[digit][:, 1]
    sol = least_squares(residual, q0, bounds=(lo, hi))
    q = sol.x
    res = residual(q)                                   # also leaves the digit posed at the solution for the distances below
    link = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])).removeprefix(f"{SIDE}_hand_")
    out = dict(
        computable=True, joints=names,
        start=dict(source=f"{B2V3} rows[D 99 mm].digits.{digit}.argmax", q_rad=q0.tolist(), phi_deg=b2["phi_max_deg"],
                   tip_gap_mm=argmax["tip_gap_mm"], closest_point_on_cylinder_palm_frame_m=on_cyl.tolist(),
                   closest_point_surface=argmax["closest_point_surface"]),
        fixed_point_tip_link_frame_m=fixed_local.tolist(), fixed_point_link=link(c.tip[digit]),
        target_palm_frame_m=target.tolist(), radial_direction_palm_frame=radial.tolist(),
        target_depth_inside_mm=round(depth_inside(c, target) * 1e3, 3),
        ik=dict(q_rad=q.tolist(), residual_palm_frame_mm=(res * 1e3).round(6).tolist(),
                residual_norm_mm=round(float(np.linalg.norm(res)) * 1e3, 6),
                at_range_limit=[bool(np.isclose(v, lo_j, atol=1e-9) or np.isclose(v, hi_j, atol=1e-9)) for v, lo_j, hi_j in zip(q, lo, hi)],
                status=int(sol.status), message=sol.message, nfev=int(sol.nfev), cost=float(sol.cost)),
        joint_change_from_start_rad=(q - q0).round(6).tolist(),
        geom_distance_at_ik_mm={link(g): round(c.dist(g) * 1e3, 3) for g in c.digit_geoms[digit]})
    d.qpos[c.all_finger_qadr] = 0.0
    return out


def main():
    t0 = time.perf_counter()
    b2 = json.loads(B2V3.read_text())
    row = next(r for r in b2["rows"] if r["is_mug"])
    assert row["D_mm"] == round(MUG_D * 1e3, 3)
    c = Capacity(anchor="radius_consistent")
    placement = c.place_cylinder(MUG_D)
    assert placement == row["cylinder"], "cylinder placement differs from B2v3's 99 mm row"
    digits = {}
    for digit in DIGITS:
        if row["digits"][digit]["argmax"] is None:
            digits[digit] = dict(computable=False, reason="no touching configuration in B2v3 at 99 mm")
            continue
        digits[digit] = solve_digit(c, digit, row["digits"][digit])
        v = digits[digit]
        print(f"{digit}: q {np.round(v['start']['q_rad'], 4).tolist()} -> {np.round(v['ik']['q_rad'], 4).tolist()} rad, residual "
              f"{v['ik']['residual_norm_mm']} mm, target {v['target_depth_inside_mm']} mm inside, geom distances "
              f"{v['geom_distance_at_ik_mm']} mm", flush=True)

    # the right hand's closed pose: each digit's IK angles in FINGER_SUFFIXES order (HERO's for a digit not computable)
    closed = HAND_CLOSED[SIDE].copy()
    for digit, v in digits.items():
        if v["computable"]:
            for name, q in zip(v["joints"], v["ik"]["q_rad"]):
                closed[FINGER_SUFFIXES.index(name.removeprefix(f"{SIDE}_hand_").removesuffix("_joint"))] = q
    names = [f"{SIDE}_hand_{f}_joint" for f in FINGER_SUFFIXES]
    dir_hero, dir_l2 = np.sign(HAND_CLOSED[SIDE] - HAND_OPEN[SIDE]), np.sign(closed - HAND_OPEN[SIDE])
    flex = FLEXION_JOINT[7:]                            # the right hand's half of the mask
    out = dict(
        question="L2: per-digit Dex3 closing targets for the 99 mm mug: fingertip 5 mm inside the mug surface, by IK from each "
                 "digit's B2v3 max-wrap configuration",
        method=dict(
            scene="sim/b2_capacity.py Capacity(anchor='radius_consistent') = B2v3's scene and placement; cylinder D 99 mm",
            start="per digit, B2v3's 99 mm argmax (max-wrap touching configuration); other digits at 0, as in B2v3",
            fixed_point="the fingertip collision geom's closest point to the cylinder in the start configuration (mj_geomDistance "
                        "witness point), fixed in the fingertip link frame",
            target=f"the start configuration's closest point on the cylinder, moved {TARGET_DEPTH_M * 1e3:g} mm radially (horizontally, "
                   "toward the axis) into the cylinder",
            ik="scipy.optimize.least_squares (trf, default tolerances, 2-point finite-difference Jacobian) over that digit's joints "
               "only, bounds = joint ranges, x0 = start; residual = FK(fixed point) - target in the palm frame",
            use="PickPlace(closing_targets='ik_mug') sets these 7 angles as the right hand's closed pose (TableScene.set_closed_pose); "
                "the closure ramp blends HERO's open pose into them; gains, torque limits, touch logic and squeeze rule unchanged",
            target_depth_inside="information: smallest distance from the target to the cylinder's side or caps (> 0 inside)",
            geom_distance_at_ik="information: mj_geomDistance of each geom of the digit to the 99 mm cylinder at the IK solution "
                                "(negative = inside; convex hulls)"),
        source=dict(file=str(B2V3), row="D_mm 99.0", verdict_99mm=row["verdict"]),
        cylinder=placement, target_depth_m=TARGET_DEPTH_M,
        digits=digits,
        closed_targets_right=dict(zip(names, closed.tolist())),
        hero_closed_right=dict(zip(names, HAND_CLOSED[SIDE].tolist())),
        hero_open_right=dict(zip(names, HAND_OPEN[SIDE].tolist())),
        closing_direction=dict(
            note="sign of closed - open per joint; the squeeze rule (sim/scene.py, flexion joints only) pushes blocked joints this "
                 f"way when they are > {GRIP_BLOCKED_ERROR_RAD} rad short of their target at full closure",
            hero=dict(zip(names, dir_hero.tolist())), l2=dict(zip(names, dir_l2.tolist())),
            flexion_joints=[n for n, f in zip(names, flex) if f],
            same_on_flexion_joints=bool(np.array_equal(dir_hero[flex], dir_l2[flex]))),
        mujoco=mujoco.__version__, scipy=scipy.__version__, runtime_s=round(time.perf_counter() - t0, 2),
        peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), gpu="none (CPU only)")
    L2_TARGETS.parent.mkdir(parents=True, exist_ok=True)
    L2_TARGETS.write_text(json.dumps(out, indent=2))
    print(f"closed targets (right, {', '.join(FINGER_SUFFIXES)}): {np.round(closed, 4).tolist()}; HERO's: {HAND_CLOSED[SIDE].tolist()}")
    print(f"closing direction same as HERO's on the flexion joints: {out['closing_direction']['same_on_flexion_joints']}")
    print(f"wrote {L2_TARGETS} ({out['runtime_s']} s, peak RSS {out['peak_rss_mb']} MB)")


if __name__ == "__main__":
    main()
