"""MuJoCo scene for teach-g1: HERO's G1 (movable Dex3 fingers, capsule feet) + table + cup + coaster.

Built on HERO's own plant (``sim2sim.plant.MujocoPlant``), so the 29 body joints get exactly the physics HERO's example
policy was benchmarked on (URDF compiled at runtime, joint armature tables, Python PD written to ``qfrc_applied``,
1 kHz physics x 20 substeps = 50 Hz control). We add our scene through the plant's ``spec_callback``, copying HERO's
interactive tabletop demo (``sim2sim/interactive_scene.py``) for the table and the contact setup, and we drive the
14 Dex3 finger joints with our own PD loop (HERO's plant only drives the 29 body joints).

Choices (see docs/log.md, 2026-10-28):
- robot: ``g1_29dof_with_hand_rev_1_0.urdf`` = the training URDF with the 14 finger joints movable (same masses/geometry);
- feet: ``sonic_train`` capsules = HERO's training feet (``sonic_box`` only if standing turns out unstable);
- table: HERO benchmark layout -- robot stance centre at the world origin, near table edge 0.18 m in front of the toes
  (toes are 0.12 m ahead of the ankles), 0.60 m deep;
- contact: explicit cup pairs with HERO's "glazed ceramic mug" friction priors (not measured values);
- touch sensors: one MuJoCo ``touch`` sensor per distal finger link (thumb_2, index_1, middle_1), sized to the link's
  collision geometry.

Units: metres, radians, seconds. World frame: z up, x forward from the robot. Public quaternions are xyzw (HERO's sim2sim
convention); MuJoCo's own qpos / xquat are wxyz.
"""

from __future__ import annotations

import mujoco
import numpy as np

from hero_isaacsim.paths import G1_ASSET_ROOT
from sim2sim.geometry import _geom_points_local
from sim2sim.interactive_scene import (CONTACT_ROLLING_M, CONTACT_TORSIONAL_ARM_M, DEX3_SLIDING_FRICTION, FINGER_EFFORT,
                                       FINGER_SUFFIXES, HAND, HAND_CLOSED, HAND_OPEN, OBJECT_CONTACT_MATERIALS)
from sim2sim.mathutil import wxyz_to_xyzw
from sim2sim.plant import MujocoPlant, body_subtree_ids

ROBOT_URDF = G1_ASSET_ROOT / "g1_29dof_with_hand_rev_1_0.urdf"
FOOT_COLLISION = "sonic_train"
TABLE_HEIGHTS = (0.50, 0.74, 0.88)

# Table: HERO benchmark geometry (data_tools/reach_specs.py: toe 0.12 m ahead of the ankles, edge gap 0.16-0.20 m, depth 0.60 m).
TABLE_EDGE_X = 0.12 + 0.18
TABLE_DEPTH = 0.60
TABLE_WIDTH = 1.00
TABLE_THICKNESS = 0.035

# Cup: solid cylinder, 8 cm diameter, 10 cm tall, 150 g. Coaster: thin static disc on the table.
CUP_RADIUS, CUP_HEIGHT, CUP_MASS = 0.04, 0.10, 0.15
COASTER_RADIUS, COASTER_THICKNESS = 0.05, 0.006
CUP_XY = (0.45, 0.18)        # left half of the table, out of the way of the right hand in M1
COASTER_XY = (0.45, 0.32)
# Pair friction priors (sliding). Cup = HERO's glazed-ceramic mug profile; cork coaster = our nominal guess. Not measured.
CUP_FRICTION = dict(hand=OBJECT_CONTACT_MATERIALS["mug"]["hand"], tabletop=OBJECT_CONTACT_MATERIALS["mug"]["tabletop"], coaster=0.5)

FINGERTIP_LINKS = ("thumb_2", "index_1", "middle_1")
SIDES = ("left", "right")


class TableScene:
    """A standing G1 in front of a table with a cup and a coaster. Use ``step(q_body)`` so the fingers are controlled."""

    def __init__(self, table_height: float, kp, kd, effort_limit, default_dof_pos):
        if not any(abs(float(table_height) - h) < 1e-6 for h in TABLE_HEIGHTS):
            raise ValueError(f"table_height must be one of {TABLE_HEIGHTS}")
        self.table_height = float(table_height)
        self._tip_boxes = _fingertip_boxes(ROBOT_URDF)
        self.plant = MujocoPlant(ROBOT_URDF, foot_collision=FOOT_COLLISION, spec_callback=self._add_scene,
                                 extra_free_joint_names=("cup_free",), kp=kp, kd=kd, effort_limit=effort_limit)
        self.model, self.data = self.plant.model, self.plant.data
        m = self.model

        names = [f"{s}_hand_{f}_joint" for s in SIDES for f in FINGER_SUFFIXES]
        jids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
        self.finger_qadr, self.finger_vadr = m.jnt_qposadr[jids].copy(), m.jnt_dofadr[jids].copy()
        self.finger_limits = m.jnt_range[jids].copy()
        self.finger_kp, self.finger_kd = float(HAND["finger_kp"]), float(HAND["finger_kd"])
        self.finger_goal = np.concatenate([HAND_OPEN[s] for s in SIDES])
        self.finger_target = self.finger_goal.copy()

        cup_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
        self.cup_qadr = int(m.jnt_qposadr[cup_joint])
        self.cup_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.touch_names = tuple(f"touch_{s}_{f}" for s in SIDES for f in FINGERTIP_LINKS)

        # Geom groups for contact bookkeeping: environment = static world geoms (floor, table, coaster); robot geoms split into
        # feet, hands and the rest of the body (a body part touching the floor or table means the robot slumped or fell).
        robot_bodies = set(body_subtree_ids(m, self.plant.root_body_id))
        body_name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        self.env_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        robot = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.hand_geoms = {g for g in robot if "_hand_" in body_name(g)}
        self.body_geoms = {g for g in robot if "_hand_" not in body_name(g) and "ankle_roll" not in body_name(g)}

        # Spawn like HERO's demo: policy default pose (arms folded back for the 0.88 m table, whose top the
        # straight fingers would otherwise intersect), then lower the root until the lowest sole point touches z = 0.
        self.initial_dof_pos = np.asarray(default_dof_pos, dtype=float).reshape(29).copy()
        if self.table_height >= 0.86:
            self.initial_dof_pos[15:] = [0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0]
        self.initial_root_pos = np.array([0.0, 0.0, 0.80])
        self.plant.reset(self.initial_root_pos, np.array([0, 0, 0, 1.0]), self.initial_dof_pos)
        self.initial_root_pos[2] -= self._sole_bottom()
        self.reset()

    # ------------------------------------------------------------------ model construction
    def _add_scene(self, spec):
        # Finger joints: HERO demo's rotor inertia / damping (regularises the tiny finger inertias at 1 kHz).
        for side in SIDES:
            for finger in FINGER_SUFFIXES:
                j = spec.joint(f"{side}_hand_{finger}_joint")
                j.armature = float(HAND["finger_armature"])
                j.damping = np.full(3, float(HAND["finger_damping"])) if np.ndim(j.damping) else float(HAND["finger_damping"])

        # Visual only (no effect on physics): checker floor, sky gradient and a key light; the URDF brings no lights.
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.82, 0.82, 0.80], rgb2=[0.70, 0.70, 0.68], width=512, height=512)
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.55, 0.62, 0.70], rgb2=[0.90, 0.90, 0.88], width=128, height=768)
        floor_mat = spec.add_material(name="floor_mat", texrepeat=[4, 4], texuniform=True)
        floor_mat.textures = ["grid" if i == int(mujoco.mjtTextureRole.mjTEXROLE_RGB) else t for i, t in enumerate(floor_mat.textures)]
        next(g for g in spec.geoms if g.name == "floor").material = "floor_mat"
        spec.worldbody.add_light(name="key", pos=[1.5, -1.0, 3.0], dir=[-0.4, 0.3, -0.85], diffuse=[0.7, 0.7, 0.7], castshadow=True)

        # Table top + four legs (static, collides with robot and cup), like HERO's "workbench".
        top = self.table_height
        cx, hx, hy = TABLE_EDGE_X + TABLE_DEPTH / 2, TABLE_DEPTH / 2, TABLE_WIDTH / 2
        spec.worldbody.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, TABLE_THICKNESS / 2],
                                pos=[cx, 0, top - TABLE_THICKNESS / 2], rgba=[0.67, 0.57, 0.43, 1], contype=1, conaffinity=1,
                                friction=[0.8, 0.01, 0.001])
        leg_half_z = (top - TABLE_THICKNESS) / 2
        for dx in (-hx + 0.045, hx - 0.045):
            for dy in (-hy + 0.045, hy - 0.045):
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, leg_half_z], pos=[cx + dx, dy, leg_half_z],
                                        rgba=[0.17, 0.20, 0.21, 1], contype=1, conaffinity=1, friction=[0.8, 0.01, 0.001])

        # Coaster: a static disc lying on the table (move it later through model.geom_pos).
        spec.worldbody.add_geom(name="coaster", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[COASTER_RADIUS, COASTER_THICKNESS / 2, 0],
                                pos=[*COASTER_XY, top + COASTER_THICKNESS / 2], rgba=[0.55, 0.38, 0.22, 1], contype=1, conaffinity=1)

        # Cup: free body, condim 4 (torsional friction) and the demo's stiffer-than-default contact (solref 0.008 s) for small props.
        cup = spec.worldbody.add_body(name="cup", pos=[*CUP_XY, top + CUP_HEIGHT / 2 + 0.001])
        cup.add_freejoint(name="cup_free")
        cup_geom = cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS, CUP_HEIGHT / 2, 0], mass=CUP_MASS,
                                rgba=[0.85, 0.85, 0.80, 1], contype=1, conaffinity=1, condim=4, solref=[0.008, 1.0],
                                friction=[CUP_FRICTION["tabletop"], CUP_FRICTION["tabletop"] * CONTACT_TORSIONAL_ARM_M["tabletop"], CONTACT_ROLLING_M])

        # Hand geoms get the Dex3 sliding friction; the cup gets explicit pairs with hand, table top and coaster (HERO's approach:
        # explicit pairs stop MuJoCo's max-per-geom rule from overriding the pair estimates; normal-contact softness is the
        # solmix-weighted mix of the two geoms, as in HERO's _add_object_contact_pairs).
        hands = [g for g in spec.geoms if (g.contype or g.conaffinity) and g.parent.name.startswith(("left_hand_", "right_hand_"))]
        for i, g in enumerate(hands):
            if not g.name:
                g.name = f"hand_contact_{g.parent.name}_{i}"
            friction = np.asarray(g.friction).copy()
            friction[0] = DEX3_SLIDING_FRICTION
            g.friction = friction
        by_name = {g.name: g for g in spec.geoms if g.name}
        surfaces = [(g, "hand") for g in hands] + [(by_name["table_top"], "tabletop"), (by_name["coaster"], "coaster")]
        for surface, role in surfaces:
            w = cup_geom.solmix / (cup_geom.solmix + surface.solmix)
            solref = w * np.asarray(cup_geom.solref) + (1 - w) * np.asarray(surface.solref)
            solimp = w * np.asarray(cup_geom.solimp) + (1 - w) * np.asarray(surface.solimp)
            mu = CUP_FRICTION[role]
            arm = CONTACT_TORSIONAL_ARM_M["hand" if role == "hand" else "tabletop"]
            spec.add_pair(name=f"cup__{surface.name}", geomname1="cup_geom", geomname2=surface.name, condim=4,
                          friction=[mu, mu, mu * arm, CONTACT_ROLLING_M, CONTACT_ROLLING_M], solref=solref, solimp=solimp,
                          margin=max(cup_geom.margin, surface.margin), gap=max(cup_geom.gap, surface.gap))

        # Fingertip touch sensors: a box site covering each distal link's collision geometry (+2 mm).
        for side in SIDES:
            for finger in FINGERTIP_LINKS:
                pos, quat, half = self._tip_boxes[f"{side}_hand_{finger}_link"]
                site = spec.body(f"{side}_hand_{finger}_link").add_site(
                    name=f"tip_{side}_{finger}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, quat=quat, size=half + 0.002,
                    rgba=[0.9, 0.2, 0.2, 0.0])
                spec.add_sensor(name=f"touch_{side}_{finger}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)

    def _sole_bottom(self) -> float:
        """Lowest world z of the feet's collision geometry in the current pose (same rule as HERO's demo)."""
        m, d = self.model, self.data
        heights = []
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if "ankle_roll_link" in body and (m.geom_contype[g] or m.geom_conaffinity[g]):
                b = int(m.geom_bodyid[g])
                pts = _geom_points_local(m, g)
                heights.append(float(np.min(pts @ d.xmat[b].reshape(3, 3)[2] + d.xpos[b, 2])))
        return min(heights)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.plant.reset(self.initial_root_pos, np.array([0, 0, 0, 1.0]), self.initial_dof_pos)
        self.finger_goal = np.concatenate([HAND_OPEN[s] for s in SIDES])
        self.finger_target = self.finger_goal.copy()
        self.data.qpos[self.finger_qadr] = self.finger_goal
        self.data.qpos[self.cup_qadr:self.cup_qadr + 7] = [*CUP_XY, self.table_height + CUP_HEIGHT / 2 + 0.001, 1, 0, 0, 0]
        self.plant.forward()

    def set_hand_closure(self, side: str, fraction: float) -> None:
        """Finger goal = blend of HERO's open and closed Dex3 poses (0 = open, 1 = closed)."""
        ids = slice(0, 7) if side == "left" else slice(7, 14)
        goal = HAND_OPEN[side] * (1 - fraction) + HAND_CLOSED[side] * fraction
        self.finger_goal[ids] = np.clip(goal, self.finger_limits[ids, 0], self.finger_limits[ids, 1])

    def step(self, body_qtarget) -> None:
        """Advance one 20 ms control step: HERO's body PD + our finger PD, both recomputed every 1 ms substep."""
        p, m, d = self.plant, self.model, self.data
        dt = p.physics_dt
        for _ in range(p.substeps):
            # finger targets slew toward the goal at <= 2 rad/s (as in HERO's demo), torque-limited PD
            self.finger_target += np.clip(self.finger_goal - self.finger_target, -2.0 * dt, 2.0 * dt)
            finger_tau = self.finger_kp * (self.finger_target - d.qpos[self.finger_qadr]) - self.finger_kd * d.qvel[self.finger_vadr]
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[p.dof_vadr] = p.pd_torque(body_qtarget)
            d.qfrc_applied[self.finger_vadr] = np.clip(finger_tau, -FINGER_EFFORT, FINGER_EFFORT)
            mujoco.mj_step(m, d)

    def cup_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.cup_body].copy(), wxyz_to_xyzw(self.data.xquat[self.cup_body]).copy()

    def touch(self) -> dict[str, float]:
        """Fingertip normal force [N] per distal link."""
        return {n: float(self.data.sensor(n).data[0]) for n in self.touch_names}

    def env_contacts(self) -> dict[str, bool]:
        """Is a hand / a non-foot body part currently touching the floor, table or coaster?"""
        hand = body = False
        for c in self.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            other = b if a in self.env_geoms else a if b in self.env_geoms else None
            hand |= other in self.hand_geoms
            body |= other in self.body_geoms
        return dict(hand=hand, body=body)

    def ik_table(self) -> dict:
        """The table top as HERO's IK ``TableBox`` dict (world frame), so the planner keeps the hand above the real slab."""
        return {"center": [TABLE_EDGE_X + TABLE_DEPTH / 2, 0.0, self.table_height - TABLE_THICKNESS / 2],
                "half_size": [TABLE_DEPTH / 2, TABLE_WIDTH / 2, TABLE_THICKNESS / 2], "surface_z": self.table_height}


def _fingertip_boxes(urdf) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Box (pos, quat wxyz, half size) in the link frame covering each distal finger link's collision geoms.

    Compiles the bare URDF once to read MuJoCo's geom bounding boxes (mesh sizes are only known after compilation).
    """
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.meshdir = "."
    m = spec.compile()
    boxes = {}
    for side in SIDES:
        for finger in FINGERTIP_LINKS:
            name = f"{side}_hand_{finger}_link"
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
            corners = []
            for g in geoms:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                corners.append(local @ R.reshape(3, 3).T + m.geom_pos[g])
            corners = np.vstack(corners)
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            boxes[name] = ((lo + hi) / 2, np.array([1.0, 0, 0, 0]), (hi - lo) / 2)
    return boxes
