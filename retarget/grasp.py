"""M3 A2: the robot grasp pose for my mug, from the oracle's grasp and my hand at the grasp frame g.

Oracle reference (mug_v1, Track B final's 44/50 grasp): R_ora = the rotation of ``Grasp.hero_side_grasp`` (a pure -45 deg
yaw; the same for the legacy and the mug_v1 offsets), o = ``radius_consistent_offset(r)`` = the cup axis in the palm frame.
For a cup centre c the oracle palm is p_ora = c + the grasp's palm_offset (exactly ``Grasp.palm_for_cup``), the axis point
opposite the palm is q_ora = p_ora + R_ora o, and h_frac_ora = (q_ora_z - table top) / h is how high up the mug that point is.

My values at g (perception's grasp frame): heading phi_h of my palm normal, h_frac_h = palm height / mug height, R_h(g); and,
as information only, the palm-point azimuth around the cup axis.

Two grasp variants (design choice). Both keep the oracle's palm-to-axis geometry o (fingers wrapped the way that grasps
the mug 44/50 in M1) and put the axis point at my grasp height: q = (cup axis x, y, table top + h_frac_h h), p = q - R o.
- b_level: R = R_z(phi_h - phi_ora) R_ora -- my palm heading, the oracle's level wrist.
- b_tilt:  R = R_h(g) M -- my full wrist orientation (heading and tilt).
Both have heading phi_h by construction (asserted to 1e-9); they differ only in the wrist tilt. So my data sets where around
the mug the hand comes from and how high it grasps; the robot keeps a palm-to-mug offset that is known to hold the mug.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from data_tools import reach_specs as rs
from sim.pick_place import Grasp, radius_consistent_offset
from sim.scene import mug_config

from retarget.frames import TABLE_TOP, angle_diff, base_to_centre, heading_human, heading_robot, robot_rot, rot_z

VARIANTS = ("b_level", "b_tilt")
HEADING_TOL = 1e-9


def _cup_dims(scene_or_dims) -> tuple[float, float]:
    """(r, h) of the mug body: from a TableScene / PickPlace (compiled ``cup_geom`` radius, as PickPlace reads it, and the
    scene's cup_height), from a dict(r=..., h=...), or (None) from config/objects.yaml."""
    if scene_or_dims is None:
        mug = mug_config()
        return mug["body_outer_diameter_m"] / 2, mug["height_m"]
    if isinstance(scene_or_dims, dict):
        return float(scene_or_dims["r"]), float(scene_or_dims["h"])
    scene = getattr(scene_or_dims, "scene", scene_or_dims)          # PickPlace -> its TableScene
    m = scene.model
    r = float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom"), 0])
    return r, float(scene.cup_height)


def oracle_reference(scene_or_dims=None, cup_centre=None) -> dict:
    """The mug_v1 oracle grasp for a cup body centre ``cup_centre`` (default: resting at the world xy origin; h_frac_ora and
    phi_ora do not depend on the xy). Keys: r, h, o, R_ora, quat_ora_wxyz, palm_offset, approach_dir, cup_centre, p_ora,
    q_ora, h_frac_ora, phi_ora (rad), azimuth_ora (information: the palm point's angle around the cup axis, rad)."""
    r, h = _cup_dims(scene_or_dims)
    o = radius_consistent_offset(r)
    grasp = Grasp.hero_side_grasp(h, o)
    assert np.array_equal(grasp.quat_wxyz, Grasp.hero_side_grasp(h).quat_wxyz), "legacy and mug_v1 rotations differ"
    R_ora = rs.quat_wxyz_to_mat(grasp.quat_wxyz)
    c = base_to_centre(np.array([0.0, 0.0, TABLE_TOP]), h) if cup_centre is None else np.asarray(cup_centre, dtype=float)
    p_ora = grasp.palm_for_cup(c)
    q_ora = p_ora + R_ora @ o
    d = p_ora - c
    return dict(r=r, h=h, o=o, R_ora=R_ora, quat_ora_wxyz=grasp.quat_wxyz, palm_offset=grasp.palm_offset, approach_dir=grasp.approach_dir,
                cup_centre=c, p_ora=p_ora, q_ora=q_ora, h_frac_ora=float((q_ora[2] - TABLE_TOP) / h), phi_ora=heading_robot(R_ora),
                azimuth_ora=math.atan2(d[1], d[0]))


def human_at_grasp(demo: dict, h: float | None = None) -> dict:
    """My hand at the grasp frame g (smoothed palm, table frame). h = mug height (default: config height_m)."""
    h = mug_config()["height_m"] if h is None else float(h)
    g = int(demo["grasp_index"])
    if g < 0:
        raise ValueError(f"{demo['clip_id']}: grasp frame undefined")
    R_hg = np.asarray(demo["palm_rot_table"][g], dtype=float)
    p_g = np.asarray(demo["palm_pos_table"][g], dtype=float)
    assert np.isfinite(R_hg).all() and np.isfinite(p_g).all(), "palm undefined at g"
    d = p_g - np.asarray(demo["cup_init_pos_table"], dtype=float)
    azimuth = math.atan2(d[1], d[0])
    return dict(g=g, phi_h=heading_human(R_hg), h_frac_h=float(p_g[2] / h), R_hg=R_hg, palm_normal_flipped=bool(demo["palm_normal_flipped"]),
                azimuth_info=dict(azimuth_rad=azimuth, azimuth_deg=math.degrees(azimuth),
                                  note="information only: angle of (palm point at g - initial mug base centre) about the cup axis, "
                                       "from table x (= world x)"))


def grasp_pose(variant: str, cup_axis_xy, human: dict, oracle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p_G (3,), R_G (3, 3), q (3,)) for the mug's axis at ``cup_axis_xy`` (world; = the layout's cup centre xy)."""
    q = np.array([cup_axis_xy[0], cup_axis_xy[1], TABLE_TOP + human["h_frac_h"] * oracle["h"]])
    if variant == "b_level":
        R = rot_z(human["phi_h"] - oracle["phi_ora"]) @ oracle["R_ora"]
    elif variant == "b_tilt":
        R = robot_rot(human["R_hg"])
    else:
        raise ValueError(f"variant must be one of {VARIANTS}")
    p = q - R @ oracle["o"]
    assert abs(angle_diff(heading_robot(R), human["phi_h"])) < HEADING_TOL, (variant, heading_robot(R), human["phi_h"])
    return p, R, q
