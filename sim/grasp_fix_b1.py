"""Track B, B1: what the compiled sim model really uses for the grasp. Read only: simulated time never advances.

Builds the oracle's scene (``TableScene`` at 0.74 m, as ``sim/pick_place.py`` does) once with the M1 cylinder and once with
the mug, and writes what MuJoCo compiled to results/m1_grasp_fix/b1_model.json:
- global solver options (cone, impratio, noslip_iterations, solver, the global solref / solimp override);
- the right Dex3: joint names, ranges, armature / damping, HERO's open / closed targets (after the scene clips them to the
  ranges), torque limits and PD gains;
- contact parameters of the right-hand collision geoms, the cup geoms and their explicit pairs;
- which parameters MuJoCo applies to cup-hand contacts. The cup sits at the oracle's nominal grasp pose with the hand closed,
  ``mj_forward`` runs collision detection, and every cup contact is listed with its friction / condim / solref (does the
  explicit pair win over MuJoCo's per-geom mixing?);
- the nominal grasp geometry: the cup placed where the oracle's grasp puts it relative to the right palm, then the signed
  distance (``mj_geomDistance``; meshes collide as their convex hulls) from the cup body to every right-hand collision geom,
  hand open and hand closed, plus each hand geom's height range above the table.
"Nominal" = the oracle's target pose. In a run the palm settles to within 1.5 cm of it (``settle_or_fail``), so these
distances describe the plan, not a run. b1_facts.md quotes these numbers next to the file:line they come from.

Run:  env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.grasp_fix_b1
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.geometry import _geom_points_local
from sim2sim.interactive_scene import FINGER_EFFORT, FINGER_SUFFIXES, HAND, HAND_CLOSED, HAND_OPEN
from sim2sim.policy_hero_export import HeroExportPolicy

from sim.pick_place import GRASP_OFFSET, GRASP_Z_ABOVE_TABLE, PREGRASP_BACKOFF, TABLE_HEIGHT
from sim.scene import FLEXION_JOINT, GRIP_KP_SCALE, GRIP_SQUEEZE_FRACTION, TableScene

OUT = Path("results/m1_grasp_fix/b1_model.json")
PALM_BODY = "right_hand_palm_link"
CUP_MODELS = ("cylinder", "mug")       # the M1 cup and the mug at its config mass


def name(m, kind, i) -> str:
    return mujoco.mj_id2name(m, kind, int(i)) or ""


def rounded(a, n: int = 6) -> list:
    return np.round(np.asarray(a, dtype=float), n).tolist()


def options(m) -> dict:
    o = m.opt
    return dict(timestep=o.timestep, integrator=mujoco.mjtIntegrator(o.integrator).name, solver=mujoco.mjtSolver(o.solver).name,
                iterations=int(o.iterations), ls_iterations=int(o.ls_iterations), tolerance=o.tolerance,
                cone=mujoco.mjtCone(o.cone).name, impratio=o.impratio, noslip_iterations=int(o.noslip_iterations),
                noslip_tolerance=o.noslip_tolerance, o_solref=rounded(o.o_solref), o_solimp=rounded(o.o_solimp),
                override_enabled=bool(o.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE),
                enableflags=int(o.enableflags), disableflags=int(o.disableflags))


def right_hand_joints(scene: TableScene) -> list[dict]:
    """Right Dex3 joints in HERO's finger order, with HERO's open / closed targets and the scene's controller constants."""
    m = scene.model
    scene.set_hand_closure("right", 1.0)
    closed_used = scene.finger_goal[7:].copy()          # the closed targets after the scene's clip to the joint ranges
    scene.set_hand_closure("right", 0.0)
    rows = []
    for i, finger in enumerate(FINGER_SUFFIXES):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"right_hand_{finger}_joint")
        v = m.jnt_dofadr[j]
        rows.append(dict(joint=f"right_hand_{finger}_joint", range_rad=rounded(m.jnt_range[j]), axis=rounded(m.jnt_axis[j], 3),
                         open_rad=float(HAND_OPEN["right"][i]), closed_rad=float(HAND_CLOSED["right"][i]),
                         closed_used_rad=float(closed_used[i]), closing_direction=int(scene.closing_direction[7 + i]),
                         squeezed_when_blocked=bool(FLEXION_JOINT[7 + i]), effort_limit_nm=float(FINGER_EFFORT[7 + i]),
                         armature=float(m.dof_armature[v]), damping=float(m.dof_damping[v]), frictionloss=float(m.dof_frictionloss[v])))
    return rows


def geom_params(m, g: int) -> dict:
    return dict(geom=name(m, mujoco.mjtObj.mjOBJ_GEOM, g), body=name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
                type=mujoco.mjtGeom(m.geom_type[g]).name, friction=rounded(m.geom_friction[g]), condim=int(m.geom_condim[g]),
                solref=rounded(m.geom_solref[g]), solimp=rounded(m.geom_solimp[g]), solmix=float(m.geom_solmix[g]),
                priority=int(m.geom_priority[g]), margin=float(m.geom_margin[g]), gap=float(m.geom_gap[g]))


def pair_params(m, p: int) -> dict:
    return dict(friction=rounded(m.pair_friction[p]), condim=int(m.pair_dim[p]), solref=rounded(m.pair_solref[p]),
                solreffriction=rounded(m.pair_solreffriction[p]), solimp=rounded(m.pair_solimp[p]),
                margin=float(m.pair_margin[p]), gap=float(m.pair_gap[p]))


def distinct_pairs(m, cup: list[int], surfaces: list[int]) -> dict:
    """Explicit pairs between any cup geom and any of ``surfaces``: how many, and their distinct parameter sets."""
    found = [p for p in range(m.npair) if {int(m.pair_geom1[p]), int(m.pair_geom2[p])} & set(cup)
             and {int(m.pair_geom1[p]), int(m.pair_geom2[p])} & set(surfaces)]
    distinct = []
    for p in found:
        if pair_params(m, p) not in distinct:
            distinct.append(pair_params(m, p))
    return dict(n_pairs=len(found), distinct_params=distinct)


def place_cup_at_nominal_grasp(scene: TableScene, closure: float) -> None:
    """Robot at its spawn pose, right fingers at ``closure``, cup where the oracle's grasp has it relative to the right palm.

    pick_place.Grasp: palm = cup centre - R (0.077, 0.055, 0) + (0, 0, 0.065 - h/2), R a pure yaw. So in the palm frame the
    cup centre is at (0.077, 0.055, h/2 - 0.065) and the cup axis is the palm's z. The mug's handle is turned along the
    approach direction (palm -> cup), the centre of the M1 protocol's handle-yaw spread.
    """
    m, d = scene.model, scene.data
    scene.reset()
    scene.set_hand_closure("right", closure)
    d.qpos[scene.finger_qadr] = scene.finger_goal
    mujoco.mj_kinematics(m, d)
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    R = d.xmat[palm].reshape(3, 3)
    centre = GRASP_OFFSET + np.array([0.0, 0.0, scene.cup_height / 2 - GRASP_Z_ABOVE_TABLE])
    a = math.atan2(GRASP_OFFSET[1], GRASP_OFFSET[0])
    yaw = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (R @ yaw).flatten())
    d.qpos[scene.cup_qadr:scene.cup_qadr + 7] = [*(d.xpos[palm] + R @ centre), *quat]
    mujoco.mj_forward(m, d)         # positions + collision detection; time is not advanced


def clearances_mm(scene: TableScene, geoms: list[int]) -> dict[str, float]:
    """Signed distance [mm] from the cup's body cylinder to each geom (negative = overlap), keyed by body."""
    m, d = scene.model, scene.data
    cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
    fromto = np.zeros(6)
    return {name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]): round(1000 * mujoco.mj_geomDistance(m, d, cup, g, 0.2, fromto), 2)
            for g in geoms}


def cup_contacts(scene: TableScene, geoms: list[int], pairs: dict) -> dict:
    """Every contact between a cup geom and ``geoms``, and whether each carries its explicit pair's parameters."""
    m, d = scene.model, scene.data
    cup, rows = {int(g) for g in np.flatnonzero(m.geom_bodyid == scene.cup_body)}, []
    for k in range(d.ncon):
        c = d.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ({g1, g2} & cup and {g1, g2} & set(geoms)):
            continue
        p = pairs.get(frozenset((g1, g2)))
        same = p is not None and np.allclose(c.friction, m.pair_friction[p]) and c.dim == m.pair_dim[p] and \
            np.allclose(c.solref, m.pair_solref[p]) and np.allclose(c.solimp, m.pair_solimp[p])
        rows.append(dict(body=name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2 if g1 in cup else g1]), dist_mm=round(1000 * c.dist, 2),
                         friction=rounded(c.friction), condim=int(c.dim), solref=rounded(c.solref), solimp=rounded(c.solimp),
                         matches_explicit_pair=bool(same)))
    return dict(n_contacts=len(rows), all_match_explicit_pair=all(r["matches_explicit_pair"] for r in rows), contacts=rows)


def z_above_table(scene: TableScene, geoms: list[int]) -> dict[str, list[float]]:
    """Height range [m] above the table top of each geom at the nominal grasp (target palm level: palm z = world z)."""
    m, d = scene.model, scene.data
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    R_palm, out = d.xmat[palm].reshape(3, 3), {}
    for g in geoms:
        b = int(m.geom_bodyid[g])
        pts = _geom_points_local(m, g) @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
        z = (pts - d.xpos[palm]) @ R_palm[:, 2] + GRASP_Z_ABOVE_TABLE
        out[name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = [round(float(z.min()), 4), round(float(z.max()), 4)]
    return out


def main():
    policy = HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
    result = dict(mujoco=mujoco.__version__, table_height_m=TABLE_HEIGHT,
                  grasp=dict(cup_centre_in_palm_frame_xy_m=rounded(GRASP_OFFSET[:2]), palm_above_table_m=GRASP_Z_ABOVE_TABLE,
                             palm_point_to_cup_axis_m=round(float(np.linalg.norm(GRASP_OFFSET[:2])), 5),
                             approach_dir_in_palm_frame=rounded(GRASP_OFFSET[:2] / np.linalg.norm(GRASP_OFFSET[:2]), 4),
                             pregrasp_backoff_m=PREGRASP_BACKOFF),
                  finger_control=dict(kp=float(HAND["finger_kp"]), kd=float(HAND["finger_kd"]), grip_kp_scale=GRIP_KP_SCALE,
                                      kp_while_closing=float(HAND["finger_kp"]) * GRIP_KP_SCALE, squeeze_fraction=GRIP_SQUEEZE_FRACTION),
                  cups={})
    for cup_model in CUP_MODELS:
        s = TableScene(TABLE_HEIGHT, policy.kp, policy.kd, policy.effort_limit, policy.default_dof_pos, cup_model=cup_model)
        m = s.model
        body = lambda g: name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        hand = sorted(g for g in s.hand_geoms if body(g).startswith("right_hand_"))
        wrist = sorted(g for g in s.arm_geoms if body(g).startswith("right_wrist_"))
        cup = [int(g) for g in np.flatnonzero(m.geom_bodyid == s.cup_body)]
        pairs = {frozenset((int(m.pair_geom1[p]), int(m.pair_geom2[p]))): p for p in range(m.npair)}
        geom_id = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        if cup_model == CUP_MODELS[0]:          # robot-side values do not depend on the cup
            result.update(options=options(m), right_hand_joints=right_hand_joints(s),
                          right_hand_geoms=[geom_params(m, g) for g in hand], right_wrist_geoms=[geom_params(m, g) for g in wrist],
                          right_wrist_geoms_with_cup_pair=sorted({body(g) for g in wrist if any(frozenset((c, g)) in pairs for c in cup)}))
        else:
            assert options(m) == result["options"], "solver options differ between the cup models"
        place_cup_at_nominal_grasp(s, 1.0)
        contacts = cup_contacts(s, hand + wrist, pairs)
        closed = clearances_mm(s, hand)
        place_cup_at_nominal_grasp(s, 0.0)
        result["cups"][cup_model] = dict(
            mass_kg=round(float(m.body_mass[s.cup_body]), 6), inertia_diag_kgm2=rounded(m.body_inertia[s.cup_body], 8),
            com_offset_m=rounded(m.body_ipos[s.cup_body], 5), height_m=s.cup_height, body_radius_m=float(m.geom_size[cup[0], 0]),
            geoms=[geom_params(m, g) for g in cup],
            pairs=dict(right_hand=distinct_pairs(m, cup, hand), table_top=distinct_pairs(m, cup, [geom_id("table_top")]),
                       coaster=distinct_pairs(m, cup, [geom_id("coaster")])),
            nominal_grasp=dict(clearance_open_mm=clearances_mm(s, hand + wrist), clearance_closed_mm=closed,
                               contacts_hand_closed=contacts, hand_geom_z_above_table_m=z_above_table(s, hand)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    for cup_model, c in result["cups"].items():
        print(f"{cup_model}: r {c['body_radius_m']} m, h {c['height_m']} m, {c['mass_kg']} kg; nominal grasp, hand open, signed distance "
              f"[mm]: " + ", ".join(f"{k.removeprefix('right_hand_').removesuffix('_link')} {v}" for k, v in c["nominal_grasp"]["clearance_open_mm"].items()))
    print(f"options: {result['options']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
