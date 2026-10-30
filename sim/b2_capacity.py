"""B2 (Track B v2): can the Dex3 reach around a cylinder of diameter D? Kinematics only, nothing is simulated.

Question: with the right palm against an upright cylinder of diameter D, can the thumb and the fingers reach far enough around
it to oppose the palm's push? Per D: put the cylinder against the palm, sweep each digit's joints on a grid, and record how far
around the cylinder (angle phi from the palm contact point) that digit's fingertip can touch it.

Model: the sim's own compiled scene, ``TableScene(0.74, cup_model="mug")`` (same URDF, collision geoms and contact pairs as the
pick-and-place runs). Its body cylinder ``cup_geom`` is the test cylinder, resized in place for each D; the mug's handle geoms
are never queried. The robot stays in its spawn pose: only finger joints are set, ``mj_kinematics`` places the links and
``mujoco.mj_geomDistance`` gives signed geom distances (negative = penetration) on the convex hulls MuJoCo collides meshes with.
The cylinder is placed relative to the palm instead of moving the hand -- the same relative geometry, so the hand is
effectively free-floating and the arm plays no role.

Palm frame (all positions below and in the JSON): ``right_hand_palm_link``, whose origin is HERO's tracked palm point
(wrist_yaw_link + (0.0415, -0.003, 0)). x runs along the fingers, z is the finger hinge axis (index at +z, middle at -z) and +y
is the palm side: flexing a finger moves its tip toward +y, and the thin principal axis of the palm collision mesh is ~3 deg
from y. In the oracle's side grasp (sim/pick_place.py) the palm is only yawed, so palm z = world up.

Setup per D (``place_cylinder``):
- palm contact surface = the palm link's collision mesh (geom ``hand_contact_right_hand_palm_link_*``); palm normal = +y.
- cylinder axis parallel to palm z (upright in the oracle's grasp), crossing the palm normal drawn through the palm geom's
  centre (MuJoCo puts a mesh geom's frame at the mesh centroid: 4.2 cm along the palm, z = 0) -- i.e. the middle of the palm.
- the axis slides along the normal until mj_geomDistance(palm, cylinder) = 0 (bisection to < 1e-7 m).
- height: the base sits on the table, GRASP_Z_ABOVE_TABLE (6.5 cm, the oracle's grasp height) below the palm origin; the
  cylinder is config height_m (0.091 m) tall for every D.

Sweep per digit (``sweep_digit``): thumb = thumb_0/1/2 (21^3 = 9261 configurations), index = index_0/1 and middle = middle_0/1
(21^2 = 441 each); 21 values per joint, evenly over the joint's range, ends included. The digits not being swept stay at 0
(they are not checked). A configuration touches if the fingertip collision geom (the distal link's, the one carrying the touch
sensor: thumb_2, index_1, middle_1) is 0-3 mm from the cylinder AND no collision geom of that digit penetrates it by more than
1 mm. phi = angle about the cylinder axis of the fingertip's closest point on the cylinder, from the palm contact point (0 deg)
toward the digit's own side (positive). That side comes from the hand, not from where the cylinder sits (``WRAP_SIGN``): the
fingers point along +x and flex toward +y (counter-clockwise about palm +z), the thumb stands out along +y and curls toward
+x (clockwise).

Result per D: phi_thumb_max, phi_finger_max (best of index and middle). Verdict: comfortable if both >= 90 deg, marginal if both
>= 90 - atan(mu), infeasible otherwise. mu = sliding friction of the explicit cup x fingertip pairs in the compiled model
(``model.pair_friction``: explicit pairs replace MuJoCo's per-geom max rule), cross-checked against the friction of the contacts
``mj_forward`` actually generates for those pairs. For L2, the JSON keeps each digit's argmax-phi configuration (joint values
and grid indices) and its closest point on the cylinder, per D. Diagnostic per digit and D: the largest distance each of its
links reaches over the grid (below -1 mm = that link is inside the cylinder in every configuration, so the digit can never
count as touching).

Open question (not decided here; ``open_questions`` in the JSON): the spec fixes the palm normal's direction and the grasp
height, not WHERE along the palm (palm-frame x) the normal is drawn. The verdict uses the palm centre. There the cylinder
covers the thumb's mount: the thumb's base link (< 2 cm behind the normal) stands 8 mm above the palm face
(``palm_side_max_y_m``), and ``largest_distance_over_grid_mm`` shows which thumb links stay inside the cylinder in every
configuration. Because this choice can decide the verdict, ``anchor_sensitivity`` repeats the check for the M1 cylinder and
the mug with the normal drawn every 5 mm along the palm and at the oracle's own grasp position (``GRASP_OFFSET`` x). It is
information for that question only; it does not change the verdict.

Run:  env -u PYTHONPATH conda run -n hoi-sim python -m sim.b2_capacity   -> results/m1_grasp_fix/b2_capacity.json + .png
"""

from __future__ import annotations

import itertools
import json
import math
import resource
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.policy_hero_export import HeroExportPolicy

from sim.pick_place import GRASP_OFFSET, GRASP_Z_ABOVE_TABLE, TABLE_HEIGHT
from sim.scene import CUP_RADIUS, FINGERTIP_LINKS, TableScene, mug_config

OUT_DIR = Path("results/m1_grasp_fix")
OLD_CYLINDER_D = 2 * CUP_RADIUS                 # the M1 cylinder: 0.080 m
MUG_D = 0.099                                   # the spec's 99 mm (= the label, config body_outer_diameter_m)
DIAMETERS = sorted({round(0.060 + 0.005 * k, 3) for k in range(11)} | {OLD_CYLINDER_D, MUG_D})
GRID_N = 21
TOUCH_MAX_M = 0.003                             # fingertip within 0-3 mm of the cylinder
PENETRATION_MAX_M = 0.001                       # no digit geom deeper than 1 mm inside it
DISTMAX = 1.0                                   # mj_geomDistance search range; larger than anything here, so every distance is exact
DIGITS = {"thumb": ("thumb_0", "thumb_1", "thumb_2"), "index": ("index_0", "index_1"), "middle": ("middle_0", "middle_1")}
TIP = {digit: next(link for link in FINGERTIP_LINKS if link.startswith(digit)) for digit in DIGITS}   # thumb_2, index_1, middle_1
# Which way each digit wraps (sign of phi), fixed by the hand: fingers flex from +x toward +y = counter-clockwise about palm
# +z, the thumb curls from +y toward +x = clockwise. Not "digit base vs axis x": that flips once the axis passes the finger bases.
WRAP_SIGN = {"thumb": -1, "index": +1, "middle": +1}
SIDE = "right"
# Open question only (anchor_sensitivity): the normal drawn every 5 mm along the palm (its hull spans x 0-0.087 m) and at the
# oracle's grasp position (GRASP_OFFSET x = where sim/pick_place.py puts the cup axis along the palm), for these two diameters.
ORACLE_AXIS_X = float(GRASP_OFFSET[0])
SENSITIVITY_ANCHORS_X = sorted({round(0.005 * k, 3) for k in range(18)} | {ORACLE_AXIS_X})
SENSITIVITY_DIAMETERS = (OLD_CYLINDER_D, MUG_D)


def collision_geom(m, body: str) -> int:
    """The single collision geom (contype or conaffinity set) of ``body``; the visual copies have both at 0."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert len(geoms) == 1, (body, geoms)
    return geoms[0]


class Capacity:
    """The scene, the ids we need, and the palm frame (fixed: only finger joints move)."""

    def __init__(self):
        policy = HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
        self.scene = TableScene(TABLE_HEIGHT, policy.kp, policy.kd, policy.effort_limit, policy.default_dof_pos, cup_model="mug")
        m, d = self.model, self.data = self.scene.model, self.scene.data
        self.height = mug_config()["height_m"]
        self.cyl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
        self.palm_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{SIDE}_hand_palm_link")
        self.palm = collision_geom(m, f"{SIDE}_hand_palm_link")
        self.digit_geoms = {dg: [collision_geom(m, f"{SIDE}_hand_{j}_link") for j in joints] for dg, joints in DIGITS.items()}
        self.tip = {dg: collision_geom(m, f"{SIDE}_hand_{TIP[dg]}_link") for dg in DIGITS}
        jid = lambda j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{SIDE}_hand_{j}_joint")
        self.qadr = {dg: np.array([m.jnt_qposadr[jid(j)] for j in joints]) for dg, joints in DIGITS.items()}
        self.ranges = {dg: m.jnt_range[[jid(j) for j in joints]].copy() for dg, joints in DIGITS.items()}
        self.base_body = {dg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{SIDE}_hand_{joints[0]}_link") for dg, joints in DIGITS.items()}
        self.all_finger_qadr = np.concatenate(list(self.qadr.values()))
        d.qpos[self.all_finger_qadr] = 0.0
        mujoco.mj_kinematics(m, d)
        self.p_palm = d.xpos[self.palm_body].copy()
        self.R_palm = d.xmat[self.palm_body].reshape(3, 3).copy()
        self.q_palm = d.xquat[self.palm_body].copy()
        # palm normal line: through the palm collision geom's centre (mesh centroid), along palm +y
        self.palm_centre = self.to_palm(d.geom_xpos[self.palm])
        R_geom = self.R_palm.T @ d.geom_xmat[self.palm].reshape(3, 3)
        thin = R_geom[:, int(np.argmin(m.geom_aabb[self.palm, 3:]))]    # mesh's thinnest principal axis, in the palm frame
        self.palm_thin_axis_angle_deg = math.degrees(math.acos(abs(thin[1])))
        self.digit_base_x = {dg: float(self.to_palm(d.xpos[b])[0]) for dg, b in self.base_body.items()}
        # how far the palm and the thumb's base link reach out on the palm side (thumb at 0; thumb_0 only spins about palm y)
        self.palm_side_max_y = {"palm": self.max_y(self.palm), "thumb_0_link": self.max_y(self.digit_geoms["thumb"][0])}

    def to_palm(self, p_world) -> np.ndarray:
        return self.R_palm.T @ (np.asarray(p_world) - self.p_palm)

    def max_y(self, g) -> float:
        """Largest palm-frame y over a mesh geom's vertices (= over its convex hull) in the current pose."""
        m, d = self.model, self.data
        mesh = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        world = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        return float(np.max((world - self.p_palm) @ self.R_palm[:, 1]))

    def dist(self, g, fromto=None) -> float:
        return mujoco.mj_geomDistance(self.model, self.data, g, self.cyl, DISTMAX, fromto)

    # ------------------------------------------------------------------ cylinder
    def resize(self, radius: float):
        """Cylinder size in place; rbound / aabb too, so the broad phase of mj_forward (contact check) sees the new size."""
        m, half = self.model, self.height / 2
        m.geom_size[self.cyl] = [radius, half, 0.0]
        m.geom_rbound[self.cyl] = math.hypot(radius, half)
        m.geom_aabb[self.cyl] = [0, 0, 0, radius, radius, half]

    def set_cylinder(self, centre_palm: np.ndarray):
        """Cup body (= cylinder centre, axes = palm axes) at ``centre_palm`` in the palm frame."""
        a = self.scene.cup_qadr
        self.data.qpos[a:a + 3] = self.p_palm + self.R_palm @ centre_palm
        self.data.qpos[a + 3:a + 7] = self.q_palm
        mujoco.mj_kinematics(self.model, self.data)

    def place_cylinder(self, diameter: float, anchor_x: float | None = None) -> dict:
        """Resize to D and slide the axis along the palm normal until the palm geom just touches the cylinder (distance 0).
        The normal is drawn through the palm centre, or at palm-frame x = ``anchor_x`` (anchor_sensitivity only)."""
        r = diameter / 2
        self.resize(r)
        x0 = self.palm_centre[0] if anchor_x is None else anchor_x
        z_centre = -GRASP_Z_ABOVE_TABLE + self.height / 2       # base on the table, 6.5 cm below the palm origin
        centre = lambda s: np.array([x0, self.palm_centre[1] + s, z_centre])
        # bisection between the axis at the palm centre (deep inside) and well clear. "<= 0" goes low: below ~1e-7 m MuJoCo
        # reports exactly 0 without the witness points (fromto) we need, so the final position keeps a strictly positive gap.
        lo, hi = 0.0, r + 0.10
        while hi - lo > 1e-7:
            mid = (lo + hi) / 2
            self.set_cylinder(centre(mid))
            lo, hi = (mid, hi) if self.dist(self.palm) <= 0 else (lo, mid)
        self.axis = centre(hi)
        self.set_cylinder(self.axis)
        fromto = np.zeros(6)
        palm_gap = self.dist(self.palm, fromto)
        contact = self.to_palm(fromto[3:])
        self.a0 = math.atan2(contact[1] - self.axis[1], contact[0] - self.axis[0])   # direction of the palm contact point = phi 0
        radial = math.hypot(contact[0] - self.axis[0], contact[1] - self.axis[1])
        assert abs(radial - r) < 1e-5, f"palm contact not on the cylinder side (radius {radial:.5f} vs {r})"   # also proves the resize took
        wrist = collision_geom(self.model, f"{SIDE}_wrist_yaw_link")
        return dict(radius_m=r, height_m=self.height, axis_palm_frame_m=self.axis.round(5).tolist(),
                    palm_gap_mm=round(palm_gap * 1e3, 6), palm_contact_point_palm_frame_m=contact.round(5).tolist(),
                    palm_contact_off_normal_deg=round(math.degrees(self.a0) + 90.0, 3),
                    table_plane_z_palm_frame_m=-GRASP_Z_ABOVE_TABLE,
                    wrist_yaw_geom_distance_mm=round(self.dist(wrist) * 1e3, 3))

    # ------------------------------------------------------------------ sweep
    def phi_deg(self, p_palm: np.ndarray, sign: int) -> float:
        a = math.atan2(p_palm[1] - self.axis[1], p_palm[0] - self.axis[0])
        return math.degrees(math.remainder(sign * (a - self.a0), 2 * math.pi))     # wrapped to [-180, 180]

    def sweep_digit(self, digit: str) -> dict:
        m, d = self.model, self.data
        r = float(m.geom_size[self.cyl, 0])
        sign = WRAP_SIGN[digit]
        grids = [np.linspace(lo, hi, GRID_N) for lo, hi in self.ranges[digit]]
        geoms, tip = self.digit_geoms[digit], self.tip[digit]
        assert geoms[-1] == tip                              # links in joint order: the fingertip is the last one
        d.qpos[self.all_finger_qadr] = 0.0
        fromto = np.zeros(6)
        best, n_touch, n_band = None, 0, 0
        largest_gap = np.full(len(geoms), -np.inf)          # diagnostic: < -1 mm = that link is inside the cylinder in EVERY configuration
        for idx in itertools.product(range(GRID_N), repeat=len(grids)):
            d.qpos[self.qadr[digit]] = [g[i] for g, i in zip(grids, idx)]
            mujoco.mj_kinematics(m, d)
            gaps = [self.dist(g, fromto if g == tip else None) for g in geoms]   # fromto = (point on the tip, point on the cylinder)
            largest_gap = np.maximum(largest_gap, gaps)
            if not 0.0 <= gaps[-1] <= TOUCH_MAX_M:
                continue
            n_band += 1
            if min(gaps) < -PENETRATION_MAX_M:
                continue
            assert fromto.any(), f"{digit} {idx}: gap {gaps[-1]} reported without witness points"   # see place_cylinder
            tip_gap, point = gaps[-1], self.to_palm(fromto[3:])
            n_touch += 1
            phi = self.phi_deg(point, sign)
            if best is None or phi > best["phi_deg"]:
                best = dict(phi_deg=phi, idx=idx, tip_gap=tip_gap, point=point)
        link = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])).removeprefix(f"{SIDE}_hand_")
        out = dict(n_configs=GRID_N ** len(grids), n_tip_within_0_3mm=n_band, n_touching=n_touch, phi_max_deg=None, argmax=None,
                   largest_distance_over_grid_mm={link(g): round(float(v) * 1e3, 3) for g, v in zip(geoms, largest_gap)})
        if best:
            q = [float(g[i]) for g, i in zip(grids, best["idx"])]
            d.qpos[self.qadr[digit]] = q
            mujoco.mj_kinematics(m, d)
            p, half = best["point"], self.height / 2
            on_side = math.hypot(p[0] - self.axis[0], p[1] - self.axis[1]) >= r - 1e-4
            cap = "top" if p[2] >= self.axis[2] + half - 1e-4 else "bottom" if p[2] <= self.axis[2] - half + 1e-4 else None
            assert on_side or cap, p                         # a closest point on the cylinder is on its side or a cap
            # at radius r AND at a cap height = the rim edge (e.g. the index tip riding over the top rim)
            surface = f"{cap} rim edge" if on_side and cap else "side" if on_side else f"{cap} cap"
            out.update(phi_max_deg=round(best["phi_deg"], 3), argmax=dict(
                joints={f"{SIDE}_hand_{j}_joint": q_j for j, q_j in zip(DIGITS[digit], q)}, grid_index=list(best["idx"]),
                tip_gap_mm=round(best["tip_gap"] * 1e3, 3), closest_point_on_cylinder_palm_frame_m=p.round(5).tolist(),
                closest_point_surface=surface, closest_point_height_above_table_m=round(p[2] + GRASP_Z_ABOVE_TABLE, 5),
                min_digit_geom_distance_mm=round(min(self.dist(g) for g in self.digit_geoms[digit]) * 1e3, 3)))
        d.qpos[self.all_finger_qadr] = 0.0
        return out

    # ------------------------------------------------------------------ friction
    def pair_mu(self) -> dict:
        """Sliding friction of the explicit cup_geom x fingertip pairs, as compiled."""
        m, out = self.model, {}
        for dg in DIGITS:
            p = [i for i in range(m.npair) if {int(m.pair_geom1[i]), int(m.pair_geom2[i])} == {self.cyl, self.tip[dg]}]
            assert len(p) == 1, (dg, p)
            out[dg] = dict(pair=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_PAIR, p[0]), friction=m.pair_friction[p[0]].round(6).tolist(),
                           condim=int(m.pair_dim[p[0]]))
        return out

    def contact_check(self, diameter: float, argmax: dict) -> dict:
        """Push each fingertip 0.5 mm into the cylinder (its argmax configuration, radius grown by the gap + 0.5 mm), run
        mj_forward, and report every contact MuJoCo generates between that fingertip and the cylinder: their friction shows
        which mu the solver really uses (the pair's, or also a dynamic contact with the geoms' mixed friction)."""
        m, d, out = self.model, self.data, {}
        for dg, a in argmax.items():
            self.place_cylinder(diameter)
            d.qpos[self.all_finger_qadr] = 0.0
            d.qpos[self.qadr[dg]] = list(a["joints"].values())
            self.resize(diameter / 2 + a["tip_gap_mm"] / 1e3 + 0.0005)
            mujoco.mj_forward(m, d)
            cons = [c for c in d.contact[:d.ncon] if {int(c.geom1), int(c.geom2)} == {self.cyl, self.tip[dg]}]
            out[dg] = dict(n_contacts=len(cons), friction=sorted({round(float(c.friction[0]), 6) for c in cons}),
                           dim=sorted({int(c.dim) for c in cons}), depth_mm=[round(-float(c.dist) * 1e3, 3) for c in cons])
        d.qpos[self.all_finger_qadr] = 0.0
        self.place_cylinder(diameter)
        return out


def verdict(phi_thumb, phi_finger, marginal_deg: float) -> str:
    if phi_thumb is None or phi_finger is None:
        return "infeasible"
    if min(phi_thumb, phi_finger) >= 90.0:
        return "comfortable"
    return "marginal" if min(phi_thumb, phi_finger) >= marginal_deg else "infeasible"


def plot(rows: list[dict], mu: float, marginal_deg: float, path: Path):
    """The two maxima against D with the two thresholds (validated reference palette: series 1 blue, series 2 orange)."""
    D = [r["D_mm"] for r in rows]
    nan = lambda v: np.nan if v is None else v
    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    no_thumb = all(r["phi_thumb_max_deg"] is None for r in rows)
    ax.plot(D, [nan(r["phi_thumb_max_deg"]) for r in rows], "-o", color="#2a78d6", lw=2, ms=6,
            label="thumb" + (": no touching configuration at any D (not plotted)" if no_thumb else ""))
    ax.plot(D, [nan(r["phi_finger_max_deg"]) for r in rows], "-s", color="#eb6834", lw=2, ms=6, label="best finger (index or middle)")
    ax.axhline(90.0, color=muted, lw=1.2)
    ax.axhline(marginal_deg, color=muted, lw=1.2, ls="--")
    ax.text(D[-1], 90.0 + 1.5, "comfortable: 90°", ha="right", va="bottom", color=muted, fontsize=8)
    ax.text(D[-1], marginal_deg + 1.5, f"marginal: 90° − atan(μ = {mu:g}) = {marginal_deg:.1f}°", ha="right", va="bottom", color=muted, fontsize=8)
    ax.set_ylim(0, 125)                     # room under the marginal line for the D labels
    for x, name in ((OLD_CYLINDER_D * 1e3, "M1 cylinder"), (MUG_D * 1e3, "mug")):
        ax.axvline(x, color="#c3c2b7", lw=1, ls=":")
        ax.text(x, 2, f"{name} ", rotation=90, va="bottom", ha="right", color=muted, fontsize=8)
    ax.set_xlabel("cylinder diameter D (mm)", color=ink)
    ax.set_ylabel("max wrap angle φ from the palm contact (deg)", color=ink)
    ax.set_title("Dex3 right hand: how far around a cylinder each side reaches\n(B2, kinematics; palm normal through the palm centre)",
                 color=ink, fontsize=10)
    ax.grid(color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def check_diameter(c: Capacity, D: float, marginal_deg: float, anchor_x: float | None = None) -> dict:
    """Place the cylinder of diameter D against the palm, sweep every digit and rate it."""
    cyl = c.place_cylinder(D, anchor_x)
    digits = {dg: c.sweep_digit(dg) for dg in DIGITS}
    fingers = {dg: digits[dg]["phi_max_deg"] for dg in ("index", "middle") if digits[dg]["phi_max_deg"] is not None}
    best_finger = max(fingers, key=fingers.get) if fingers else None
    phi_t, phi_f = digits["thumb"]["phi_max_deg"], fingers.get(best_finger)
    return dict(D_mm=round(D * 1e3, 3), is_old_cylinder=abs(D - OLD_CYLINDER_D) < 1e-9, is_mug=abs(D - MUG_D) < 1e-9,
                phi_thumb_max_deg=phi_t, phi_finger_max_deg=phi_f, best_finger=best_finger,
                verdict=verdict(phi_t, phi_f, marginal_deg), cylinder=cyl, digits=digits)


def anchor_sensitivity(c: Capacity, marginal_deg: float) -> dict:
    """Open question 1, information only: the same check with the palm normal drawn at other points along the palm (palm-frame
    x), for the M1 cylinder and the mug. Per D also the runs of anchors that share a verdict, e.g. "x 0.000-0.055: infeasible"."""
    centre_x = float(c.palm_centre[0])          # exact, so this row repeats the main result
    anchors = sorted(set(SENSITIVITY_ANCHORS_X) | {centre_x})
    name = {centre_x: "palm centre (the verdict's anchor)", ORACLE_AXIS_X: "oracle grasp position"}
    rows, runs = [], {}
    for D in SENSITIVITY_DIAMETERS:
        for x in anchors:
            r = check_diameter(c, D, marginal_deg, anchor_x=x)
            rows.append(dict(D_mm=r["D_mm"], anchor_x_m=round(x, 5), anchor=name.get(x, ""),
                             palm_contact_off_normal_deg=r["cylinder"]["palm_contact_off_normal_deg"],
                             wrist_yaw_geom_distance_mm=r["cylinder"]["wrist_yaw_geom_distance_mm"],
                             phi_thumb_max_deg=r["phi_thumb_max_deg"], phi_finger_max_deg=r["phi_finger_max_deg"],
                             best_finger=r["best_finger"], n_touching={dg: v["n_touching"] for dg, v in r["digits"].items()},
                             verdict=r["verdict"]))
        mine = [r for r in rows if r["D_mm"] == round(D * 1e3, 3)]
        spans = []                                      # consecutive anchors with the same verdict
        for r in mine:
            if spans and spans[-1][2] == r["verdict"]:
                spans[-1][1] = r["anchor_x_m"]
            else:
                spans.append([r["anchor_x_m"], r["anchor_x_m"], r["verdict"]])
        runs[f"{D * 1e3:g}mm"] = "; ".join(f"x {a:.4f}-{b:.4f}: {v}" for a, b, v in spans)
    return dict(note="information for open question 1 only; the verdict above uses the palm centre. Same placement, grid, "
                     "touch test and thresholds; only the point on the palm where the normal is drawn moves. "
                     f"{ORACLE_AXIS_X} = GRASP_OFFSET x in sim/pick_place.py (its palm frame = wrist_yaw_link axes = "
                     "right_hand_palm_link axes, URDF right_hand_palm_joint rpy 0 0 0). The M1 cylinder is included because the "
                     "oracle grasps it reliably (B3 R0): a check on which anchor matches the dynamics.",
                anchors_x_m=[round(x, 5) for x in anchors], verdict_runs=runs, rows=rows)


def main():
    t0 = time.perf_counter()
    c = Capacity()
    pairs = c.pair_mu()
    mus = {p["friction"][0] for p in pairs.values()}
    assert len(mus) == 1, pairs
    mu = mus.pop()
    marginal_deg = 90.0 - math.degrees(math.atan(mu))
    rows = []
    for D in DIAMETERS:
        rows.append(check_diameter(c, D, marginal_deg))
        r = rows[-1]
        print(f"D {D * 1e3:5.1f} mm  thumb {r['phi_thumb_max_deg']}  finger {r['phi_finger_max_deg']} ({r['best_finger']})  -> "
              f"{r['verdict']}   touching: " + ", ".join(f"{k} {v['n_touching']}/{v['n_configs']}" for k, v in r["digits"].items()), flush=True)
    mug_row = next(r for r in rows if r["is_mug"])
    have = {dg: v["argmax"] for dg, v in mug_row["digits"].items() if v["argmax"]}
    check = c.contact_check(MUG_D, have)
    sens = anchor_sensitivity(c, marginal_deg)
    for r in sens["rows"]:          # the palm-centre rows must repeat the main result
        if r["anchor"].startswith("palm centre"):
            main_row = next(m for m in rows if m["D_mm"] == r["D_mm"])
            assert (r["verdict"], r["phi_thumb_max_deg"], r["phi_finger_max_deg"]) == \
                   (main_row["verdict"], main_row["phi_thumb_max_deg"], main_row["phi_finger_max_deg"]), r
    print("anchor sensitivity:", json.dumps(sens["verdict_runs"]), flush=True)
    at = lambda D, anchor: next(r["verdict"] for r in sens["rows"] if r["D_mm"] == round(D * 1e3, 3) and r["anchor"].startswith(anchor))
    m = c.model
    result = dict(
        question="With the palm against an upright cylinder of diameter D, can the thumb and the fingers reach far enough around it "
                 "to oppose the palm's push? (kinematics only)",
        method=dict(
            model="TableScene(0.74, cup_model='mug') from sim/scene.py: the sim's compiled model (URDF g1_29dof_with_hand_rev_1_0, "
                  "same collision geoms and pairs); its body cylinder 'cup_geom' is resized per D, the handle geoms are never queried; "
                  "only finger joints are set; mj_kinematics + mj_geomDistance (signed, convex hulls); no time stepping",
            hand=SIDE, palm_frame=f"{SIDE}_hand_palm_link (origin = HERO palm point = wrist_yaw_link + (0.0415, -0.003, 0)); x along the "
                                  "fingers, z = finger hinge axis (index +z, middle -z), +y = palm side",
            palm_contact_geom=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.palm),
            palm_normal="+y of the palm frame: the side the fingers flex toward; the palm mesh's thinnest principal axis is "
                        f"{c.palm_thin_axis_angle_deg:.2f} deg from it",
            palm_normal_anchor_palm_frame_m=c.palm_centre.round(5).tolist(),
            palm_normal_anchor="centre of the palm collision geom (MuJoCo puts a mesh geom's frame at the mesh centroid); "
                               "not fixed by the spec: see open_questions[0] and anchor_sensitivity",
            palm_side_max_y_m={k: round(v, 5) for k, v in c.palm_side_max_y.items()},
            palm_side_max_y_note="largest palm-frame y of the palm geom and of the thumb's base link geom (all joints 0): the thumb "
                                 "base stands out above the palm face, next to the palm normal",
            cylinder_axis="parallel to palm z (= world up in the oracle's side grasp), crossing the palm normal line; slid along the "
                          "normal until mj_geomDistance(palm, cylinder) = 0 (bisection, < 1e-7 m)",
            grasp_height=f"palm origin {GRASP_Z_ABOVE_TABLE} m above the table (sim/pick_place.py GRASP_Z_ABOVE_TABLE); cylinder base on "
                         f"the table; cylinder height = config height_m = {c.height} m for every D",
            diameters_mm=[round(D * 1e3, 3) for D in DIAMETERS],
            diameters_note=f"60-110 mm in 5 mm steps + the M1 cylinder's {OLD_CYLINDER_D * 1e3:g} mm (sim/scene.py CUP_RADIUS x 2; "
                           f"already on the grid; height {c.height} m like every D, not the M1 cylinder's 0.10 m: this row "
                           f"varies the diameter only) + 99 mm (mug label)",
            grid=f"{GRID_N} values per joint, np.linspace over the joint range (ends included); other digits at 0. phi_max is a "
                 "lower bound of the continuous maximum (a finer grid containing this one can only raise it); joint steps "
                 "are range / 20 (4.5-6 deg)",
            digits={dg: dict(joints=[f"{SIDE}_hand_{j}_joint" for j in DIGITS[dg]], ranges_rad=c.ranges[dg].round(6).tolist(),
                             fingertip_geom=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.tip[dg]),
                             digit_geoms=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in c.digit_geoms[dg]],
                             base_x_palm_frame_m=round(c.digit_base_x[dg], 5),
                             positive_phi="counter-clockwise about palm +z (fingers flex from +x toward +y)" if WRAP_SIGN[dg] > 0
                             else "clockwise about palm +z (thumb curls from +y toward +x)") for dg in DIGITS},
            touch=f"fingertip geom distance in [0, {TOUCH_MAX_M * 1e3:g}] mm AND every collision geom of the digit >= -{PENETRATION_MAX_M * 1e3:g} mm",
            phi="angle about the cylinder axis of the fingertip's closest point on the cylinder (mj_geomDistance fromto), from the palm "
                "contact point (0 deg), positive toward the digit's own side (fixed per digit by the hand, WRAP_SIGN), wrapped to [-180, 180]",
            closest_point_surface="side / top cap / bottom cap, or 'top rim edge' / 'bottom rim edge' when the point is at radius r "
                                  "AND at a cap height (within 0.1 mm)",
            phi_finger_max="max over index and middle",
            verdict="comfortable if both maxima >= 90 deg; marginal if both >= 90 - atan(mu); infeasible otherwise "
                    "(a digit with no touching configuration counts as infeasible)"),
        mu=dict(value=mu, source="model.pair_friction[0] of the explicit pairs cup_geom x fingertip geom in the compiled scene "
                                 "(sim/scene.py: CUP_FRICTION['hand'] = HERO's glazed-ceramic prior); explicit pairs replace the "
                                 "geoms' max-friction mixing (hand geoms 1.5, cup 0.35)",
                pairs=pairs, contact_check_at_99mm=check,
                contact_check="each fingertip at its 99 mm argmax configuration pushed 0.5 mm into the cylinder, mj_forward, all "
                              "contacts between that fingertip geom and the cylinder (digits without a touching configuration: none)",
                pyramidal_cone_info=dict(mu_over_sqrt2=round(mu / math.sqrt(2), 4),
                                         marginal_threshold_deg=round(90.0 - math.degrees(math.atan(mu / math.sqrt(2))), 4),
                                         note="information only, not used: MuJoCo's pyramidal cone is inscribed in the friction "
                                              "cone, so between its edges the tangential limit drops to mu/sqrt(2)"),
                solver=dict(cone="pyramidal" if m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL else "elliptic",
                            impratio=float(m.opt.impratio), noslip_iterations=int(m.opt.noslip_iterations))),
        thresholds_deg=dict(comfortable=90.0, marginal=round(marginal_deg, 4)),
        rows=rows,
        verdict_99mm=mug_row["verdict"],
        verdict_99mm_condition=f"with the palm normal drawn through the palm centre (x {c.palm_centre[0]:.4f} m); depends on that "
                               "choice, see open_questions[0]",
        open_questions=[dict(
            question="Where along the palm (palm-frame x) should the palm normal be drawn? The spec fixes its direction, the "
                     "contact (distance 0) and the grasp height, not its point on the palm.",
            used=f"the palm centre (palm collision geom centroid, x {c.palm_centre[0]:.4f} m)",
            alternative=f"the oracle's grasp position (GRASP_OFFSET x = {ORACLE_AXIS_X} m, sim/pick_place.py), where the oracle "
                        "puts the cup axis",
            effect=("the verdict depends on it" if len({r["verdict"] for r in sens["rows"]}) > 1 else
                    "the verdict is the same at every anchor tried") + f". 99 mm: {at(MUG_D, 'palm centre')} at the palm centre, "
                   f"{at(MUG_D, 'oracle')} at the oracle position; by anchor: {sens['verdict_runs'][f'{MUG_D * 1e3:g}mm']}. "
                   f"M1 cylinder ({OLD_CYLINDER_D * 1e3:g} mm): {at(OLD_CYLINDER_D, 'palm centre')} at the palm centre, "
                   f"{at(OLD_CYLINDER_D, 'oracle')} at the oracle position; by anchor: "
                   f"{sens['verdict_runs'][f'{OLD_CYLINDER_D * 1e3:g}mm']}. Numbers: anchor_sensitivity.",
            b4_gate="B4 (STOP Track B if 99 mm is infeasible) follows from verdict_99mm, so it inherits this choice: confirm the "
                    "anchor with the user before treating B5 as ruled out")],
        anchor_sensitivity=sens,
        mujoco=mujoco.__version__, runtime_s=round(time.perf_counter() - t0, 1),
        peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),       # ru_maxrss is in kB on Linux
        gpu="none (CPU only)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "b2_capacity.json").write_text(json.dumps(result, indent=2))
    plot(rows, mu, marginal_deg, OUT_DIR / "b2_capacity.png")
    print(f"mu {mu}  marginal threshold {marginal_deg:.2f} deg  -> 99 mm: {mug_row['verdict']} (palm-centre anchor)   contact check: {check}")
    print(f"wrote {OUT_DIR / 'b2_capacity.json'} and {OUT_DIR / 'b2_capacity.png'} ({result['runtime_s']} s, peak RSS {result['peak_rss_mb']} MB)")


if __name__ == "__main__":
    main()
