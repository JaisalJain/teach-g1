"""Scripted right-hand pick-and-place of the cup onto the coaster (0.74 m table), with HERO doing whole-body control.

Every phase plans a fresh HERO reference from the live robot state with our reference builder (``sim/reference_builder.py``):
  stand 2 s
  -> pregrasp: lift the palm over the table, then 8 cm before the grasp pose (side approach)       [base: HERO's plan for the grasp]
  -> settle:   HERO's goal correction (replan + g <- g - 0.6 e) until the palm error < 1.5 cm
  -> approach: slow straight move to the grasp pose (2 s), settle again
  -> close:    ramp the fingers closed until the thumb and index or middle fingertip touch the cup, then finish
               closing to HERO's demo grip (full closure; blocked joints squeeze); both touches must still be on
  -> lift 10 cm -> move over the coaster -> lower until the cup is 1 cm above it -> settle -> open -> retreat 10 cm back, 5 cm up

Grasp = HERO's Dex3 side grasp (sim2sim/interactive_client/controller.js computeGraspPreview), written in the CUP frame
(origin at the cup centre, z up; the cup is round, so its x axis is taken along world x): palm yaw -45 deg (HERO's default
outward wrist yaw), palm = cup centre - R (0.077, 0.055, 0), at 6.5 cm above the table like HERO's mug / can grasps.
For the mug the cup centre is the centre of its body cylinder and the grasp ignores the handle (same palm pose for any yaw);
the cup's height comes from the scene, so "6.5 cm above the table", the resting height and the place clearance keep their meaning.

``grasp_offset`` (Track B v3 fix ladder, rung L1; the default is M1's behaviour):
- "legacy": the offset (0.077, 0.055, 0) above for every object. It leaves 15 mm along the palm normal between the palm point and
  the surface of the 8 cm M1 cylinder it was validated on, but wider objects come closer: 5.5 mm for the 99 mm mug, whose
  planned pose then puts the palm hull 11.65 mm inside it (results/m1_grasp_fix/b1_facts.md).
- "radius": ``radius_consistent_offset(r)``, r = the compiled radius of the cup's body cylinder: only the palm-normal component
  changes, to r + 0.015, so every object keeps the M1 cylinder's 15 mm. For r = 0.040 this is the legacy offset exactly.
In both, the pre-grasp is 8 cm back along palm -> cup centre, the approach direction of whichever offset is used.

``closing_targets`` (rung L2; the default is M1's behaviour):
- "hero": the closure ramp blends HERO's open Dex3 pose into HERO's closed one (one pose for every object).
- "ik_mug": the right hand's closed pose becomes the mug-specific targets in results/m1_grasp_fix/v3/l2_targets.json
  (sim/l2_closing_targets.py: per digit, fingertip IK to 5 mm inside the 99 mm mug from its B2v3 max-wrap configuration).
  Open pose, ramp, touch logic, gains, torque limits and the squeeze rule are unchanged (TableScene.set_closed_pose).

``contact_model`` (rung L3; the default is M1's behaviour): passed to ``TableScene`` -- "legacy" = MuJoCo's default contact
solver settings, "grasp" = elliptic cones, impratio 10, 3 NoSlip iterations and condim 4 on the right fingertips and palm
(sim/scene.py GRASP_OPTIONS). Global options: they also change the feet's contacts.
Note: ``sample_layouts`` derives the cup positions from the grasp's palm offset, so layouts that must stay fixed across grasps are
built with the legacy grasp (sim/m1_grasp_ladder.py does).

Success = cup upright (< 10 deg), centred on the coaster (centre within the coaster radius) and resting on it, and the robot
never fell. A failed run reports the phase it failed in.

Run:  env -u PYTHONPATH conda run -n hoi-sim python -m sim.pick_place [--cup-model mug|cylinder] [--trials 10] [--seed 0]
Writes <out>/pick_results.json and one video per run under <out>/videos/; <out> defaults to results/m1_pick for the cylinder
(the M1 results: add --cup-model cylinder to reproduce them) and results/m1_pick_mug_yaw0 for the mug, which this script
always places with its handle along world +x (the mug protocol with varied handle yaws is sim/m1_pick_mug.py). It refuses to
overwrite a pick_results.json written for the other cup model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from data_tools import hero_reach_generator as gen
from data_tools import reach_specs as rs
from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.bench.replan import BenchGoal, BenchReplanner, ReplanConfig
from sim2sim.interactive_scene import FINGER_SUFFIXES
from sim2sim.mathutil import xyzw_to_wxyz
from sim2sim.policy_hero_export import HeroExportPolicy
from sim2sim.state import read_state

from sim.m1_reach_test import FALL_PELVIS_Z, FALL_TILT_DEG, VideoWriter, standing_reference
from sim.reference_builder import PathPlanner, min_jerk_path
from sim.scene import CONTACT_MODELS, COASTER_RADIUS, CUP_MODELS, TableScene

FPS = 50
TABLE_HEIGHT = 0.74
# Grasp, in the cup frame (HERO demo values for the Dex3 side grasp)
GRASP_OFFSET = np.array([0.077, 0.055, 0.0])       # cup centre in the palm frame (right hand)
WRIST_YAW = math.radians(-45.0)                     # HERO demo: 45 deg outward wrist yaw (right hand -> negative yaw)
GRASP_Z_ABOVE_TABLE = 0.065
PREGRASP_BACKOFF = 0.08
# Radius-consistent grasp pose (M2 v3; used by B2v3 in sim/b2_capacity.py and by L1). GRASP_OFFSET ignores the object's size:
# its palm-normal component 0.055 = the M1 cylinder's radius 0.040 + a 15 mm gap between the palm and the cylinder's surface.
# Keeping that 15 mm gap for any radius r (and the other two components as they are) keeps the palm where it was relative to
# the object's surface, instead of e.g. inside a 99 mm mug.
PALM_SURFACE_GAP = 0.015


def radius_consistent_offset(r: float) -> np.ndarray:
    """Object axis in the palm frame (same frame and meaning as GRASP_OFFSET) for an object of radius ``r`` [m]."""
    return np.array([GRASP_OFFSET[0], r + PALM_SURFACE_GAP, GRASP_OFFSET[2]])


def l2_closed_targets() -> np.ndarray:
    """L2's right-hand closed pose (FINGER_SUFFIXES order) as saved by sim/l2_closing_targets.py. Read from the results file,
    so a run uses exactly the targets that file reports; it refuses targets with a digit marked not computable."""
    data = json.loads(L2_TARGETS.read_text())
    missing = [digit for digit, v in data["digits"].items() if not v["computable"]]
    if missing:
        raise ValueError(f"{L2_TARGETS}: no L2 target for {missing} (no touching configuration in B2v3 at 99 mm)")
    return np.array([data["closed_targets_right"][f"right_hand_{f}_joint"] for f in FINGER_SUFFIXES])


# At the M1 cylinder's radius the radius-consistent pose IS the oracle's (exact in float64)
assert radius_consistent_offset(0.040)[1] == 0.055 and np.array_equal(radius_consistent_offset(0.040), GRASP_OFFSET)
# Motion
LIFT_HEIGHT = 0.10
PLACE_CLEARANCE = 0.01          # cup bottom above the coaster top when the fingers open
RETREAT_BACK, RETREAT_UP = 0.10, 0.05
TRANSPORT_SPEED = 0.08          # m/s (palm), min 1.5 s
SETTLE_STAY_M = 0.015
SETTLE_MAX_S = 5.0
REACH_FAIL_M = 0.03              # a settle that ends farther than this from its target = target not reached
LIFT_SLIP_FAIL_M = 0.02          # cup moving this much relative to the palm during the lift = slip
CLOSE_RATE = 0.8                # closure fraction per second (fingers slew <= 2 rad/s anyway)
TOUCH_N = 0.3                   # fingertip contact threshold
TOUCH_WINDOW_S = 0.2            # contacts chatter at 1 ms: a fingertip counts as touching if it read > TOUCH_N within this window
# Measured reachable palm workspace at 0.74 m (M1 reach test: all 6 goals within 2 cm): x 0.35-0.45, y -0.32..-0.12
PALM_BOX = dict(x=(0.35, 0.45), y=(-0.32, -0.12))
MIN_CUP_COASTER_DIST = 0.13
OUT_DIR = Path("results/m1_pick")
MUG_OUT_DIR = Path("results/m1_pick_mug_yaw0")
GRASP_OFFSETS = ("legacy", "radius")        # PickPlace(grasp_offset=...): GRASP_OFFSET, or radius_consistent_offset(cup radius)
CLOSING_TARGETS = ("hero", "ik_mug")        # PickPlace(closing_targets=...): HERO's closed Dex3 pose, or L2's IK targets
L2_TARGETS = Path(__file__).resolve().parents[1] / "results/m1_grasp_fix/v3/l2_targets.json"   # sim/l2_closing_targets.py

SETTLE_CFG = ReplanConfig(first_event="start", period_s=0.6, base_mode="current", goal_adjust=True, handover_blend_s=0.3,
                          stay_threshold_m=SETTLE_STAY_M, skip_below_m=SETTLE_STAY_M, hold_s=6.0)


@dataclasses.dataclass
class Grasp:
    """Palm pose relative to the cup (cup frame = cup centre, world-aligned axes) and the approach direction."""
    palm_offset: np.ndarray       # palm position - cup centre
    quat_wxyz: np.ndarray         # palm orientation
    approach_dir: np.ndarray      # horizontal unit vector, palm -> cup

    @classmethod
    def hero_side_grasp(cls, cup_height: float, cup_in_palm: np.ndarray = GRASP_OFFSET) -> "Grasp":
        """HERO's side grasp with the cup centre at ``cup_in_palm`` in the palm frame (default: HERO's GRASP_OFFSET)."""
        R = rs.palm_rotation(WRIST_YAW, 0.0, 0.0)
        offset = -R @ cup_in_palm + np.array([0.0, 0.0, GRASP_Z_ABOVE_TABLE - cup_height / 2])   # cup centre is cup_height / 2 up
        approach = -offset * np.array([1, 1, 0])
        return cls(offset, rs.mat_to_quat_wxyz(R), approach / np.linalg.norm(approach))

    def palm_for_cup(self, cup_center: np.ndarray) -> np.ndarray:
        return np.asarray(cup_center, dtype=float) + self.palm_offset


# Failure phase -> the report's categories (the last two are extra: target out of reach, cup released but not on the coaster)
CATEGORY = {"approach": "missed/knocked cup", "close": "missed/knocked cup", "lift": "slip during lift", "transport": "dropped while moving",
            "lower": "dropped while moving", "fell": "fell", "reach": "palm could not reach the target", "place": "released but not on the coaster"}


class Failure(Exception):
    def __init__(self, phase: str, reason: str):
        super().__init__(f"{phase}: {reason}")
        self.phase, self.reason = phase, reason


class PickPlace:
    """One scene + HERO policy + planners; ``run(cup_xy, coaster_xy)`` performs one pick-and-place attempt.

    ``cup_model`` / ``cup_mass`` go to ``TableScene`` ("mug" or "cylinder"; None = the model's own mass).
    Fix-ladder parameters (Track B v3), legacy = M1 by default: ``grasp_offset`` "legacy" | "radius", ``closing_targets``
    "hero" | "ik_mug", ``contact_model`` "legacy" | "grasp" (module docstring).
    """

    def __init__(self, policy: HeroExportPolicy, cup_model: str = "mug", cup_mass: float | None = None, grasp_offset: str = "legacy",
                 closing_targets: str = "hero", contact_model: str = "legacy"):
        if grasp_offset not in GRASP_OFFSETS:
            raise ValueError(f"grasp_offset must be one of {GRASP_OFFSETS}")
        if closing_targets not in CLOSING_TARGETS:
            raise ValueError(f"closing_targets must be one of {CLOSING_TARGETS}")
        if contact_model not in CONTACT_MODELS:
            raise ValueError(f"contact_model must be one of {CONTACT_MODELS}")
        self.policy = policy
        self.scene = TableScene(TABLE_HEIGHT, policy.kp, policy.kd, policy.effort_limit, policy.default_dof_pos,
                                cup_model=cup_model, cup_mass=cup_mass, contact_model=contact_model)
        self.closing_targets = closing_targets
        if closing_targets == "ik_mug":
            self.scene.set_closed_pose("right", l2_closed_targets())
        # r from the object itself: every cup model's body cylinder is the geom "cup_geom" (the mug's handle is ignored)
        self.cup_radius = float(self.scene.model.geom_size[mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom"), 0])
        self.grasp_offset = grasp_offset
        self.cup_in_palm = GRASP_OFFSET if grasp_offset == "legacy" else radius_consistent_offset(self.cup_radius)
        self.grasp = Grasp.hero_side_grasp(self.scene.cup_height, self.cup_in_palm)
        self.goal_template = BenchGoal(clip_name="pick", hand="right", pos_w=np.zeros(3), quat_wxyz=self.grasp.quat_wxyz,
                                       table=self.scene.ik_table(), reach_end_frame=1, pelvis_drop=0.0, pelvis_pitch=0.0,
                                       waist_pitch=0.0, base_family="stand", height_label="h074")
        stand_ref = standing_reference(self.scene, 2)
        self.planner = PathPlanner(self.goal_template, ReplanConfig(), arm_posture=stand_ref.joint_pos[0])
        self.corrector = BenchReplanner(self.goal_template, SETTLE_CFG, reference=stand_ref)

    # ------------------------------------------------------------------ control primitives
    def _goal(self, pos, quat_wxyz) -> BenchGoal:
        """A goal at ``pos`` with HERO's benchmark base plan (pelvis drop / lean) for reaching it at this table height."""
        cfg = rs.PROFILES["hero_bench_v1"]
        target = rs.HandTarget(hand="right", pos=tuple(pos), yaw=WRIST_YAW, pitch=0.0, roll=0.0, canonical_grasp=True, yaw_clipped=False,
                               approach="ingress", hover=rs.BENCH_HOVER_M, ingress=rs.BENCH_INGRESS_M, lift=rs.BENCH_LIFT_M)
        base, _ = rs._bench_base(TABLE_HEIGHT, float(np.mean(cfg["bench_stand_drop_range"])), target, cfg)
        return dataclasses.replace(self.goal_template, pos_w=np.asarray(pos, float), quat_wxyz=np.asarray(quat_wxyz, float),
                                   pelvis_drop=base.pelvis_drop, pelvis_pitch=base.pelvis_pitch, waist_pitch=base.waist_pitch)

    def _tick(self, phase: str):
        """One 50 Hz step on the live reference, with logging and fall detection."""
        st = read_state(self.scene.plant)
        if self.corrector_active:
            new_ref = self.corrector.step(self.k, st, self.scene.plant)
            if new_ref is not None:
                self.ref, self.t = new_ref, 0
        self.scene.step(self.policy.control(self.ref, self.t, st))
        self.t = self.ref.clamp(self.t + 1)
        self.k += 1
        self.log.add(phase)
        if self.log.fell_now():
            raise Failure(phase, "robot fell")

    def move(self, phase: str, waypoints, base: str = "current", base_goal=None):
        """Plan our min-jerk palm path through ``waypoints`` [(pos, quat_wxyz, seconds)] and play it to the end."""
        st = read_state(self.scene.plant)
        self.planner.goal = base_goal or self._goal(waypoints[-1][0], waypoints[-1][1])
        p0, q0 = self.planner.palm_start(st)
        path_pos, path_quat = min_jerk_path(p0, q0, waypoints)
        self.ref, info = self.planner.plan_path(st, self.scene.plant, path_pos, path_quat, base=base)
        self.t, self.corrector_active = 0, False
        for _ in range(len(path_pos)):
            self._tick(phase)
        self.plans.append(dict(phase=phase, **{k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()}))

    def settle(self, phase: str, pos, quat_wxyz, max_s: float = SETTLE_MAX_S) -> dict:
        """HERO's closed-loop goal correction toward (pos, quat) until the palm error < 1.5 cm or ``max_s`` passes."""
        c = self.corrector
        c.goal = dataclasses.replace(self.goal_template, pos_w=np.asarray(pos, float), quat_wxyz=np.asarray(quat_wxyz, float))
        c.reference = self.ref
        c.arm_posture = np.asarray(self.ref.joint_pos[-1], dtype=float).copy()
        c.reset()
        c.live_k0 = -self.t          # replanner's live frame at local step k is k - live_k0 = the frame we are showing
        self.k, self.corrector_active = 0, True
        while not c.adjuster.stayed and self.k < max_s * FPS:
            self._tick(phase)
        self.corrector_active = False
        err = float(np.linalg.norm(read_state(self.scene.plant).palm_pos_w[1] - pos))
        return dict(phase=phase, converged=c.adjuster.stayed, seconds=round(self.k / FPS, 2), palm_err_cm=round(err * 100, 2),
                    replans=c.adjuster.n_replans)

    def settle_or_fail(self, phase: str, pos, quat_wxyz, max_s: float = SETTLE_MAX_S):
        """Settle, and stop the run (before touching the cup or opening the hand) if the palm ended far from the target."""
        r = self.settle(phase, pos, quat_wxyz, max_s)
        self.settles.append(r)
        if r["palm_err_cm"] > REACH_FAIL_M * 100:
            raise Failure("reach", f"{phase}: palm {r['palm_err_cm']:.1f} cm from the target after {r['seconds']} s (not reachable)")

    def hold(self, phase: str, seconds: float):
        for _ in range(int(seconds * FPS)):
            self._tick(phase)

    # ------------------------------------------------------------------ the task
    def run(self, cup_xy, coaster_xy, video_path: Path | None = None, cup_yaw: float = 0.0) -> dict:
        """One attempt; ``cup_yaw`` [rad] turns the cup about z (for the mug: the handle points along (cos, sin) of it)."""
        s, g = self.scene, self.grasp
        s.reset(cup_xy, coaster_xy, cup_yaw)
        self.log = Log(s, video_path)
        self.plans, self.settles, self.k, self.t, self.corrector_active = [], [], 0, 0, False
        self.ref = standing_reference(s, 200)
        self.policy.reset()
        s.step(self.policy.reset_target_for(self.ref))
        cup0 = s.cup_pose()[0]
        result = dict(cup_xy=[round(float(v), 4) for v in cup_xy], coaster_xy=[round(float(v), 4) for v in coaster_xy])
        try:
            self.hold("stand", 2.0)
            grasp_p = g.palm_for_cup(cup0)
            pre_p = grasp_p - PREGRASP_BACKOFF * g.approach_dir
            # pregrasp: first up to the table-clearing height (HERO's first approach waypoint), then over to the pregrasp
            palm0 = read_state(s.plant).palm_pos_w[1]
            q_now = xyzw_to_wxyz(read_state(s.plant).palm_quat_w[1])
            z_clear = max(gen._clearance_height(q_now, g.quat_wxyz, "right", TABLE_HEIGHT), pre_p[2])
            self.move("pregrasp", [(np.array([palm0[0], palm0[1], z_clear]), g.quat_wxyz, 1.2), (pre_p, g.quat_wxyz, 1.8)],
                      base="plan", base_goal=self._goal(grasp_p, g.quat_wxyz))
            self.settle_or_fail("settle_pregrasp", pre_p, g.quat_wxyz)
            self.move("approach", [(grasp_p, g.quat_wxyz, 2.0)])
            self.settle_or_fail("settle_grasp", grasp_p, g.quat_wxyz, max_s=3.0)
            self.check_cup_undisturbed(cup0, "approach")

            result["contact_closure"] = self.close_hand()
            self.check_grasp("close")

            in_hand_closed = s.cup_pose()[0] - read_state(s.plant).palm_pos_w[1]
            lifted = s.cup_pose()[0]
            self.move("lift", [(grasp_p + [0, 0, LIFT_HEIGHT], g.quat_wxyz, 1.5)])
            rise = s.cup_pose()[0][2] - lifted[2]
            slip = float(np.linalg.norm(s.cup_pose()[0] - read_state(s.plant).palm_pos_w[1] - in_hand_closed))
            result.update(lift_cup_rise_cm=round(rise * 100, 2), lift_slip_in_hand_cm=round(slip * 100, 2))
            if slip > LIFT_SLIP_FAIL_M or rise < 0.5 * LIFT_HEIGHT:
                raise Failure("lift", f"cup slipped {slip * 100:.1f} cm in the hand, rose {rise * 100:.1f} of {LIFT_HEIGHT * 100:.0f} cm")
            self.check_grasp("lift")

            # place: keep the palm orientation, so the measured palm->cup offset after the lift stays valid
            palm = read_state(s.plant).palm_pos_w[1]
            cup_in_hand = s.cup_pose()[0] - palm
            cup_place = s.coaster_top_center() + [0, 0, s.cup_height / 2 + PLACE_CLEARANCE]
            place_p = cup_place - cup_in_hand
            above_p = place_p + [0, 0, palm[2] - place_p[2]]
            dist = float(np.linalg.norm(above_p - palm))
            self.move("transport", [(above_p, g.quat_wxyz, max(1.5, dist / TRANSPORT_SPEED))], base="plan")
            self.check_in_hand(cup_in_hand, "transport")
            self.move("lower", [(place_p, g.quat_wxyz, 1.5)])
            self.settle_or_fail("settle_place", place_p, g.quat_wxyz, max_s=3.0)
            self.check_in_hand(cup_in_hand, "lower")

            for i in range(1, 31):          # open over 0.6 s
                s.set_hand_closure("right", 1.0 - i / 30)
                self._tick("open")
            self.hold("open", 0.6)
            back = read_state(s.plant).palm_pos_w[1] - RETREAT_BACK * g.approach_dir + [0, 0, RETREAT_UP]
            self.move("retreat", [(back, g.quat_wxyz, 1.5)])
            self.hold("retreat", 1.0)
            result.update(self.evaluate())
        except Failure as f:
            result.update(success=False, failed_phase=f.phase, reason=f.reason)
            result.update(self.evaluate(final=False))
        finally:
            self.log.close()
        if not result["success"]:
            result["category"] = "fell" if self.log.fell else CATEGORY[result["failed_phase"]]
        result.update(plans=self.plans, settles=self.settles, sim_seconds=round(float(s.data.time), 2))
        return result

    # ------------------------------------------------------------------ grasp + checks
    def close_hand(self) -> float:
        """Ramp the closure until thumb and index-or-middle touch the cup, then finish to full closure (HERO demo grip)."""
        s, closure, contact_at = self.scene, 0.0, None
        while closure < 1.0:
            closure = min(1.0, closure + CLOSE_RATE / FPS)
            s.set_hand_closure("right", closure)
            self._tick("close")
            if contact_at is None and self._grasp_touch():
                contact_at = closure
        if contact_at is None:
            self.hold("close", 0.5)
            if not self._grasp_touch():
                raise Failure("close", "no thumb + finger contact at full closure (missed the cup)")
            contact_at = 1.0
        self.hold("close", 0.5)
        return round(contact_at, 3)

    def _grasp_touch(self) -> bool:
        t = self.log.recent_touch(int(TOUCH_WINDOW_S * FPS))
        return t["touch_right_thumb_2"] > TOUCH_N and max(t["touch_right_index_1"], t["touch_right_middle_1"]) > TOUCH_N

    def check_grasp(self, phase: str):
        if not self._grasp_touch():
            recent = self.log.recent_touch(int(TOUCH_WINDOW_S * FPS))
            raise Failure(phase, "fingertip contact lost (max over 0.2 s: " + ", ".join(f"{k[12:]} {v:.2f} N" for k, v in recent.items() if "right" in k) + ")")

    def check_cup_undisturbed(self, cup0, phase: str):
        moved = float(np.linalg.norm(self.scene.cup_pose()[0][:2] - cup0[:2]))
        if moved > 0.02 or self.scene.cup_tilt_deg() > 15:
            raise Failure(phase, f"knocked the cup before grasping (moved {moved * 100:.1f} cm, tilt {self.scene.cup_tilt_deg():.0f} deg)")

    def check_in_hand(self, cup_in_hand, phase: str):
        now = self.scene.cup_pose()[0] - read_state(self.scene.plant).palm_pos_w[1]
        drift = float(np.linalg.norm(now - cup_in_hand))
        if drift > 0.03 or not self._grasp_touch():
            raise Failure(phase, f"dropped / slipped while moving (cup moved {drift * 100:.1f} cm in the hand)")

    def evaluate(self, final: bool = True) -> dict:
        s = self.scene
        cup = s.cup_pose()[0]
        off = float(np.linalg.norm(cup[:2] - s.coaster_xy))
        dz = float(cup[2] - (s.coaster_top_center()[2] + s.cup_height / 2))
        tilt = s.cup_tilt_deg()
        out = dict(cup_final=[round(float(v), 4) for v in cup], cup_off_coaster_center_cm=round(off * 100, 2),
                   cup_height_above_resting_cm=round(dz * 100, 2), cup_tilt_deg=round(tilt, 1), fell=self.log.fell)
        if final:
            upright, centred, resting = tilt < 10.0, off <= COASTER_RADIUS, abs(dz) < 0.004
            out["success"] = bool(upright and centred and resting and not self.log.fell)
            if not out["success"]:
                out["failed_phase"] = "place"
                out["reason"] = ", ".join(r for ok, r in ((upright, "cup not upright"), (centred, "cup not on the coaster"),
                                                          (resting, "cup not resting on the coaster")) if not ok)
        return out


class Log:
    """Per-step robot / cup state for fall detection and the summary; optional video."""

    def __init__(self, scene: TableScene, video_path: Path | None):
        self.scene, self.fell, self.phases, self.touches = scene, False, [], []
        self.video = VideoWriter(scene, video_path) if video_path else None

    def add(self, phase: str):
        self.phases.append(phase)
        self.touches.append(self.scene.touch())
        if self.video:
            self.video.frame(None)

    def recent_touch(self, n_steps: int) -> dict[str, float]:
        """Per fingertip, the max touch force over the last ``n_steps`` control steps."""
        recent = self.touches[-n_steps:]
        return {k: max(t[k] for t in recent) for k in recent[0]}

    def fell_now(self) -> bool:
        p = self.scene.plant
        R = self.scene.data.xmat[p.root_body_id].reshape(3, 3)
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        self.fell |= bool(p.root_pos[2] < FALL_PELVIS_Z or tilt > FALL_TILT_DEG or self.scene.env_contacts()["body"])
        return self.fell

    def close(self):
        if self.video:
            self.video.close()


def sample_layouts(grasp: Grasp, n: int, seed: int, box: dict = PALM_BOX) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cup / coaster positions whose grasp and place palm poses fall inside ``box`` (default: the measured reachable palm box)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        palms = rng.uniform([box["x"][0], box["y"][0]], [box["x"][1], box["y"][1]], size=(2, 2))
        cup, coaster = palms - grasp.palm_offset[:2]   # coaster: where the cup centre will be with the nominal in-hand offset
        if np.linalg.norm(cup - coaster) >= MIN_CUP_COASTER_DIST:
            out.append((cup, coaster))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cup-model", choices=CUP_MODELS, default="mug")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help=f"default: {OUT_DIR} (cylinder), {MUG_OUT_DIR} (mug)")
    ap.add_argument("--palm-box", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), default=None,
                    help="sample palm positions from this box instead of the measured workspace (stress test)")
    args = ap.parse_args()
    box = dict(x=tuple(args.palm_box[:2]), y=tuple(args.palm_box[2:])) if args.palm_box else PALM_BOX
    args.out = args.out or (OUT_DIR if args.cup_model == "cylinder" else MUG_OUT_DIR)
    # M1's command lines predate --cup-model (now default "mug"): never let one model's run overwrite the other's results
    old = args.out / "pick_results.json"
    if old.exists():
        old_model = json.loads(old.read_text())["protocol"].get("cup_model", "cylinder")    # M1's files have no cup_model: cylinder
        if old_model != args.cup_model:
            raise SystemExit(f"{old} holds {old_model} results; pass --cup-model {old_model} to re-run them, or another --out")
    (args.out / "videos").mkdir(parents=True, exist_ok=True)
    policy = HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
    task = PickPlace(policy, cup_model=args.cup_model)
    g = task.grasp
    fixed = (np.array([0.38, -0.15]) - g.palm_offset[:2], np.array([0.42, -0.29]) - g.palm_offset[:2])
    layouts = [("fixed", *fixed)] + [(f"random_{i:02d}", *l) for i, l in enumerate(sample_layouts(g, args.trials, args.seed, box))]
    runs = []
    for name, cup_xy, coaster_xy in layouts:
        t0 = time.perf_counter()
        r = task.run(cup_xy, coaster_xy, args.out / "videos" / f"{name}.mp4")
        r.update(name=name, wall_s=round(time.perf_counter() - t0, 1))
        runs.append(r)
        print(f"{name:<10} cup {r['cup_xy']} coaster {r['coaster_xy']} -> {'SUCCESS' if r['success'] else 'FAIL ' + r['failed_phase'] + ': ' + r['reason']}"
              f"  (cup {r['cup_off_coaster_center_cm']} cm off centre, tilt {r['cup_tilt_deg']} deg, {r['wall_s']} s)")
    rand = [r for r in runs if r["name"].startswith("random")]
    results = dict(
        protocol=dict(table_height_m=TABLE_HEIGHT, cup_model=args.cup_model, cup_mass_kg=task.scene.cup_mass, cup_height_m=task.scene.cup_height,
                      cup_yaw_deg=0.0, cup_yaw="cup frame = world axes (mug: handle along world +x) in every run",
                      grasp=dict(frame="cup centre, world-aligned axes", palm_offset_m=g.palm_offset.round(4).tolist(),
                      palm_quat_wxyz=g.quat_wxyz.round(4).tolist(), approach_dir=g.approach_dir.round(4).tolist(),
                      source="HERO Dex3 side grasp: offset (0.077, 0.055, 0), wrist yaw 45 deg outward, 6.5 cm above table"),
                      pregrasp_backoff_m=PREGRASP_BACKOFF, lift_m=LIFT_HEIGHT, place_clearance_m=PLACE_CLEARANCE, settle_stay_m=SETTLE_STAY_M,
                      touch_threshold_N=TOUCH_N, touch_window_s=TOUCH_WINDOW_S, palm_box=box, min_cup_coaster_dist_m=MIN_CUP_COASTER_DIST, seed=args.seed,
                      settle_cfg=SETTLE_CFG.as_dict(), odom="truth", reach_fail_m=REACH_FAIL_M, lift_slip_fail_m=LIFT_SLIP_FAIL_M,
                      palm_box_constrains="grasp palm xy and nominal place palm xy (pregrasp / lift / transport waypoints are not constrained)",
                      close_mode="ramp closure until thumb + index/middle touch, then HERO-demo full closure with squeeze; touch also gates the lift",
                      place_target="cup's true in-hand offset measured after the lift (privileged sim state), palm orientation kept",
                      cup_and_robot_state="exact simulator state (no perception, no odometry noise)",
                      success="cup tilt < 10 deg, cup centre within coaster radius (5 cm), resting on the coaster (|dz| < 4 mm), robot never fell",
                      fall_rule=f"pelvis z < {FALL_PELVIS_Z} m, pelvis tilt > {FALL_TILT_DEG} deg, or a non-foot/hand body part on floor/table"),
        fixed_layout_success=runs[0]["success"],
        random_success_rate=dict(successes=sum(r["success"] for r in rand), trials=len(rand)),
        failures_by_phase={p: sum(1 for r in rand if not r["success"] and r["failed_phase"] == p) for p in sorted({r.get("failed_phase") for r in rand if not r["success"]})},
        failures_by_category={c: sum(1 for r in rand if r.get("category") == c) for c in sorted({r.get("category") for r in rand if not r["success"]})},
        runs=runs)
    (args.out / "pick_results.json").write_text(json.dumps(results, indent=2))
    print(f"random layouts: {results['random_success_rate']}  failures by category: {results['failures_by_category']}")
    print(f"wrote {args.out / 'pick_results.json'}")


if __name__ == "__main__":
    main()
