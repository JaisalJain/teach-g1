"""M4: physical execution of my retargeted palm paths -- HERO tracks them in MuJoCo with the mug_v1 grasp, fingers close on touch.

``Execute`` is the oracle (sim/pick_place.py ``PickPlace``, profile mug_v1, my mug from config at 0.40 kg) with one new motion
primitive, ``follow``, and two variants built from it. Everything that decides success or failure is the oracle's own code,
inherited unchanged: ``settle_or_fail``, ``check_cup_undisturbed``, ``close_hand``, ``check_grasp``, ``check_in_hand``,
``evaluate``, the fall rule and the failure categories (M1's, plus "reach").

Setup = exactly PickPlace.run's lines before its first move (scene.reset with the layout's cup, coaster and handle yaw,
standing reference, policy reset, reset step, the 2 s stand). The path is built AFTER the stand, because it starts at the
standing palm (p_s, q_now), like M3's.

follow(phase, pos, quat, base, base_goal) plays a dense 50 Hz path slice the way ``move`` plays its min-jerk path:
planner.plan_path, then one _tick per path frame. The IK's frame 0 is always the robot's live palm (planner.palm_start), so
the slice is bent onto it: pos_j + (1 - w_j)(p0 - pos_0) and slerp(q0, quat_j, w_j), w = 10 s^3 - 15 s^4 + 6 s^5, s = j / (N - 1).
The correction fades out with zero velocity (same profile as the min-jerk blends of M3's warps), and the end frame is
unchanged. follow never touches the fingers, except in the naive variant, which passes the path's closure per frame.

(b) b_level / b_tilt -- the path is cut into four slices (``segments``): prelude + approach 1 (to the old pause point
i_pause), approach 2 (to G), transport 1 (from the close hold's last frame to the transport's peak), transport 2 (to P).
  1. follow "pregrasp" = prelude + approach 1 (base "plan", base goal = the grasp pose); settle_or_fail at the pause pose (5 s)
  2. follow "approach" = approach 2 (base "current"); settle_or_fail at G (3 s); check_cup_undisturbed
  3. close_hand(); check_grasp("close")
  4. follow "lift" = transport 1 (base "plan", base goal = its last pose). At the peak: rise = cup z rise since the close,
     slip = |cup - palm - in-hand offset at the close|; Failure "lift" if slip > LIFT_SLIP_FAIL_M or rise < 0.5 x the PLANNED
     lift (path peak z - G z: the oracle's rule with its 10 cm replaced by this path's own lift); check_grasp("lift")
  5. place correction at the peak, as the oracle does it: cup_in_hand = cup - palm, cup_place = coaster top + h / 2 + 1 cm,
     P' = cup_place - cup_in_hand. Transport 2's end is moved to P' with x'(s) = x(s) + s (P' - x_end), s = the slice's
     normalised re-timed progress sigma(t) / D0 (``slice_progress``, Step 1's own sigma), so the correction also starts and
     ends at rest. follow "transport" (base "current"); check_in_hand("transport");
     settle_or_fail at P' (3 s); check_in_hand("lower")
  6. open (0.6 s ramp + 0.6 s hold), retreat by -0.10 u + (0, 0, 0.05) in 1.5 s with u = M3's (path info), hold 1.0 s,
     evaluate() -- the oracle's lines, with u and the path's grasp orientation in place of the oracle's.
  The place correction reads privileged simulator state (the true cup pose in the hand), exactly as the oracle's does;
  every row says so (detail.place_correction.uses).
(a) naive -- one follow "replay" of the whole naive path (base "plan", base goal = the path's G), closure set from the path
  every tick; no settle, no grasp check, no place correction. Then retreat by -0.10 u + (0, 0, 0.05) in 1.5 s, u = the
  horizontal unit vector palm -> cup centre at that moment, hold 1.0 s, evaluate(). A naive failure is evaluate()'s reason
  ("place"), or "fell".

Paths (``build_path``): "v2" = retarget/paths_v2.py v2_path (Step 1: M3's paths, pause removed, every human segment re-timed
rest to rest; its ``follow`` dict gives the four slices); "m3" = M3's paths (retarget/warp.py object_aware_path, retarget/naive.py
naive_path), for smoke tests only. k is always M3's, computed at M3's recorded layout.
Diagnostics: after each physics run, by default the Step 3 collision minimum of b runs (``collision_diagnostic`` =
retarget/collision.py collision_min: its own setup, M3's single-plan IK clip of this run's path, hand-mug clearance along the
approach). Kinematic, information only, never a gate; it runs after the physics, so it cannot change the run.
Layouts (``m4_layouts``): trial00-trial49 = trials 0-49 of sim.m1_pick_mug.trials() built with the legacy grasp, handle yaw =
the trial's + (phi_h - phi_ora) as in M3 (trial00-09 asserted equal to M3's); recorded_v2 = cup at PALM_BOX centre - legacy
palm offset - delta / 2, coaster = cup + delta (delta = my coaster - cup offset), handle yaw = M3's recorded layout's; so both
nominal palms are centred on the box.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.m4_execute ...):
  run          runs (clip, variant, layout) items on N worker processes (<= 8); one JSON row per run in <out>/runs/, written
               as soon as the run ends; <out>/run_resources_<stamp>.json at the end. Items = --items FILE (JSON list of
               {clip, variant, layout}) or the product of --clips, --variants, --layouts. Refuses to overwrite a row unless
               --overwrite (or skips existing rows with --skip-existing).
  rerun        re-runs one saved row (its clip, variant, layout, path source and diagnostics) and compares every field but the
               wall time and RSS; exit code 1 if anything differs; --report FILE saves the comparison.
  check-oracle the determinism check through the UNCHANGED oracle: Track B final's mug_v1 trials 0-4 (ForcePickPlace via the
               ladder's worker) against results/m1_grasp_fix/v3_final/runs/fifty.json, every field; plus Execute's setup vs
               PickPlace.run's own (run_m3.SetupOnly), byte for byte -> results/m4_execute/determinism_checks.json
Rendering: none (no video is written by any of these commands).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.bench.replan import slerp_wxyz
from sim2sim.mathutil import xyzw_to_wxyz
from sim2sim.policy_hero_export import HeroExportPolicy
from sim2sim.state import read_state

from retarget import paths_v2, run_m3
from retarget.collision import collision_min
from retarget.frames import load_demo
from retarget.grasp import human_at_grasp, oracle_reference
from sim.m1_grasp_ladder import peak_rss_mb
from sim.m1_pick_mug import RESULT_KEYS, trials
from sim.m1_reach_test import standing_reference
from sim.pick_place import (CATEGORY, GRASP_PROFILES, LIFT_SLIP_FAIL_M, PALM_BOX, PLACE_CLEARANCE, RETREAT_BACK, RETREAT_UP,
                            SETTLE_MAX_S, Failure, Grasp, Log, PickPlace)
from sim.reference_builder import quat_angle_deg
from sim.scene import mug_config

PROFILE = "mug_v1"
MUG_MASS_KG = 0.40                 # spec: the mug from config at 0.40 kg (asserted against config/objects.yaml)
CLIPS = run_m3.CLIPS
VARIANTS = ("b_level", "b_tilt", "naive")
M3_VARIANT = {"b_level": "b_level", "b_tilt": "b_tilt", "naive": "a_naive"}     # M3's name for each variant
PATH_SOURCES = ("v2", "m3")
N_TRIALS = 50                      # layouts trial00-trial49
FPS = 50
RETREAT_S, RETREAT_HOLD_S = 1.5, 1.0          # the oracle's retreat move and hold
OPEN_TICKS, OPEN_HOLD_S = 30, 0.6             # the oracle's 0.6 s opening ramp (30 ticks) and hold
SETTLE_GRASP_S = SETTLE_PLACE_S = 3.0         # the oracle's max_s for settle_grasp / settle_place (settle_pregrasp: SETTLE_MAX_S)
MAX_WORKERS = 8
RSS_CAP_MB = 16 * 1024             # spec: 16 GB RSS per process
INDEX_KEYS = ("i_prelude_end", "i_pause", "i_G", "i_close_end", "i_peak", "i_P", "i_release")
VOLATILE = ("wall_s", "worker_peak_rss_mb")   # row fields that measure the computer, not the simulation (left out of comparisons)
OUT_DIR = Path("results/m4_execute")
FIFTY = Path("results/m1_grasp_fix/v3_final/runs/fifty.json")        # Track B final's 50 mug_v1 trials (read only)
COLLISION_DIAGNOSTIC = "sim.m4_execute:collision_diagnostic"
PRIVILEGED = ("privileged simulator state: the cup's true pose in the hand at the transport peak sets the place pose, exactly "
              "as the oracle's place target (sim/pick_place.py) does")


# ---------------------------------------------------------------------------------------------------------------- the executor
class Execute(PickPlace):
    """The mug_v1 oracle plus ``follow`` and the M4 variants (module docstring). One per worker process; every run starts
    from scene.reset + policy.reset, like the oracle's."""

    def __init__(self, policy: HeroExportPolicy | None = None):
        assert mug_config()["mass_kg"] == MUG_MASS_KG, "config/objects.yaml mug mass is no longer 0.40 kg"
        policy = policy or HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
        super().__init__(policy, cup_model="mug", cup_mass=MUG_MASS_KG, **GRASP_PROFILES[PROFILE])

    # ------------------------------------------------------------------ setup (PickPlace.run's own lines)
    def setup(self, layout: dict, video_path: Path | None = None) -> np.ndarray:
        """PickPlace.run's lines before its try block: reset with the layout's cup, coaster and handle yaw, log, standing
        reference, policy reset, reset step. Returns cup0 (the cup centre the approach check compares with). The 2 s stand
        follows in ``run_variant`` (inside the try, as in PickPlace.run). A diagnostic that needs the setup state again calls
        ``setup(layout)`` then ``hold("stand", 2.0)``: the state is bit-identical (determinism_checks.json)."""
        s = self.scene
        s.reset(layout["cup_xy"], layout["coaster_xy"], layout["handle_yaw"])
        self.log = Log(s, video_path)
        self.plans, self.settles, self.k, self.t, self.corrector_active = [], [], 0, 0, False
        self.ref = standing_reference(s, 200)
        self.policy.reset()
        s.step(self.policy.reset_target_for(self.ref))
        return s.cup_pose()[0]

    # ------------------------------------------------------------------ the new primitive
    def follow(self, phase: str, pos, quat, base: str = "current", base_goal=None, closure=None):
        """Play a dense 50 Hz path slice (pos (N, 3), quat (N, 4) wxyz) like ``move``: one plan, one _tick per frame.

        Frame 0 must be the live IK palm (p0, q0), so the slice is bent onto it with a min-jerk weight that is 0 at the start
        and 1 at the end: pos_j + (1 - w_j)(p0 - pos_0), slerp(q0, quat_j, w_j). The end frame stays exactly as given
        (w = 1: no position change, and slerp returns quat_j itself). ``closure`` (N,) = set the right hand's closure before
        each tick (naive only); None = leave the fingers alone. ``base_goal`` defaults to the slice's last pose, as in move.
        """
        pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
        n = len(pos)
        if n < 2 or quat.shape != (n, 4):
            raise ValueError(f"follow {phase}: need >= 2 frames with matching quaternions, got {pos.shape} / {quat.shape}")
        st = read_state(self.scene.plant)
        self.planner.goal = base_goal or self._goal(pos[-1], quat[-1])
        p0, q0 = self.planner.palm_start(st)
        s = np.arange(n) / (n - 1)
        w = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        path_pos = pos + (1 - w)[:, None] * (np.asarray(p0) - pos[0])
        path_quat = np.array([slerp_wxyz(q0, quat[j], w[j]) for j in range(n)])
        self.ref, info = self.planner.plan_path(st, self.scene.plant, path_pos, path_quat, base=base)
        self.t, self.corrector_active = 0, False
        for j in range(n):
            if closure is not None:
                self.scene.set_hand_closure("right", float(closure[j]))
            self._tick(phase)
        self.plans.append(dict(phase=phase, **{k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()},
                               base=base, start_offset_mm=round(float(np.linalg.norm(np.asarray(p0) - pos[0])) * 1e3, 3),
                               start_offset_deg=round(quat_angle_deg(np.asarray(q0), quat[0]), 3)))

    # ------------------------------------------------------------------ one run
    def run_variant(self, variant: str, layout: dict, make_path, video_path: Path | None = None) -> dict:
        """One attempt of ``variant`` on ``layout`` (dict: cup_xy, coaster_xy, handle_yaw). ``make_path(p_s, q_now)`` builds
        the path dict after the stand (it starts at the standing palm). Returns PickPlace.run's result keys (plus the
        variant's own records: segments, lift_planned_cm, in_hand_closed_m, place_correction, naive_retreat_u)."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        s = self.scene
        cup0 = self.setup(layout, video_path)
        result = dict(cup_xy=[round(float(v), 4) for v in layout["cup_xy"]], coaster_xy=[round(float(v), 4) for v in layout["coaster_xy"]])
        self.exec_path = None
        try:
            self.hold("stand", 2.0)
            st = read_state(s.plant)
            self.exec_path = make_path(st.palm_pos_w[1].copy(), xyzw_to_wxyz(st.palm_quat_w[1]))
            if variant == "naive":
                self._naive(result)
            else:
                self._object_aware(cup0, result)
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

    def _object_aware(self, cup0: np.ndarray, result: dict):
        """(b): the four slices with the oracle's settles and checks between them (module docstring, steps 1-6)."""
        s, p = self.scene, self.exec_path
        pos, quat = np.asarray(p["palm_pos"], float), np.asarray(p["palm_quat_wxyz"], float)
        info = p["info"]
        G, quat_G, u = (np.asarray(info[k], float) for k in ("G", "quat_G_wxyz", "u"))
        seg = segments(p)
        cut = lambda name: slice(seg[name][0], seg[name][1] + 1)
        result["segments"] = seg

        # 1-2. approach in two slices, each followed by the oracle's settle (the pause point plays the oracle's pre-grasp)
        i_pause = seg["approach1"][1]
        self.follow("pregrasp", pos[cut("approach1")], quat[cut("approach1")], base="plan", base_goal=self._goal(G, quat_G))
        self.settle_or_fail("settle_pregrasp", pos[i_pause], quat[i_pause], max_s=SETTLE_MAX_S)
        self.follow("approach", pos[cut("approach2")], quat[cut("approach2")])
        self.settle_or_fail("settle_grasp", G, quat_G, max_s=SETTLE_GRASP_S)
        self.check_cup_undisturbed(cup0, "approach")

        # 3. the oracle's grasp
        result["contact_closure"] = self.close_hand()
        self.check_grasp("close")

        # 4. transport 1 = the lift to the path's highest point, then the oracle's lift test against this path's own lift
        palm = lambda: read_state(s.plant).palm_pos_w[1]
        in_hand_closed = s.cup_pose()[0] - palm()
        lifted = s.cup_pose()[0]
        self.follow("lift", pos[cut("transport1")], quat[cut("transport1")], base="plan")
        i_peak = seg["transport1"][1]
        planned_lift = float(pos[i_peak][2] - G[2])
        rise = s.cup_pose()[0][2] - lifted[2]
        slip = float(np.linalg.norm(s.cup_pose()[0] - palm() - in_hand_closed))
        result.update(lift_cup_rise_cm=round(rise * 100, 2), lift_slip_in_hand_cm=round(slip * 100, 2),
                      lift_planned_cm=round(planned_lift * 100, 2), in_hand_closed_m=[round(float(v), 4) for v in in_hand_closed])
        if slip > LIFT_SLIP_FAIL_M or rise < 0.5 * planned_lift:
            raise Failure("lift", f"cup slipped {slip * 100:.1f} cm in the hand, rose {rise * 100:.1f} of {planned_lift * 100:.1f} cm")
        self.check_grasp("lift")

        # 5. place correction at the peak (the oracle's place target), carried by transport 2 with its own timing
        cup_in_hand = s.cup_pose()[0] - palm()
        cup_place = s.coaster_top_center() + [0, 0, s.cup_height / 2 + PLACE_CLEARANCE]
        place_p = cup_place - cup_in_hand
        x = pos[cut("transport2")]
        prog, prog_source = slice_progress(p, "transport2", seg["transport2"])
        x_new = x + prog[:, None] * (place_p - x[-1])
        result["place_correction"] = dict(uses=PRIVILEGED, cup_in_hand_m=[round(float(v), 4) for v in cup_in_hand],
                                          P_path=[round(float(v), 4) for v in x[-1]], P_corrected=[round(float(v), 4) for v in place_p],
                                          shift_mm=round(float(np.linalg.norm(place_p - x[-1])) * 1e3, 2), progress=prog_source)
        self.follow("transport", x_new, quat[cut("transport2")])
        self.check_in_hand(cup_in_hand, "transport")
        self.settle_or_fail("settle_place", place_p, quat[seg["transport2"][1]], max_s=SETTLE_PLACE_S)
        self.check_in_hand(cup_in_hand, "lower")

        # 6. the oracle's release and retreat, with M3's u and the path's grasp orientation
        for i in range(1, OPEN_TICKS + 1):
            s.set_hand_closure("right", 1.0 - i / OPEN_TICKS)
            self._tick("open")
        self.hold("open", OPEN_HOLD_S)
        back = palm() - RETREAT_BACK * u + [0, 0, RETREAT_UP]
        self.move("retreat", [(back, quat_G, RETREAT_S)])
        self.hold("retreat", RETREAT_HOLD_S)

    def _naive(self, result: dict):
        """(a): the whole path in one follow with its own closure, then the retreat away from wherever the cup is."""
        s, p = self.scene, self.exec_path
        pos, quat = np.asarray(p["palm_pos"], float), np.asarray(p["palm_quat_wxyz"], float)
        info = p["info"]
        G, quat_G = np.asarray(info["G"], float), np.asarray(info["quat_G_wxyz"], float)
        self.follow("replay", pos, quat, base="plan", base_goal=self._goal(G, quat_G), closure=np.asarray(p["closure"], float))
        palm = read_state(s.plant).palm_pos_w[1]
        d = (s.cup_pose()[0] - palm) * np.array([1.0, 1.0, 0.0])
        if np.linalg.norm(d) < 1e-9:          # cup centre straight below / above the palm: no horizontal direction exists
            raise RuntimeError("naive retreat: palm and cup centre coincide horizontally (u undefined)")
        u = d / np.linalg.norm(d)
        result["naive_retreat_u"] = [round(float(v), 4) for v in u]
        back = palm - RETREAT_BACK * u + [0, 0, RETREAT_UP]
        self.move("retreat", [(back, quat[-1], RETREAT_S)])     # the replay's last orientation (the naive path has no fixed grasp quat)
        self.hold("retreat", RETREAT_HOLD_S)


# ---------------------------------------------------------------------------------------------------------------- path slices
def segments(path: dict) -> dict[str, tuple[int, int]]:
    """The four followed slices of a (b) path as (first, last) frame indices, both inclusive (neighbours share a frame).

    approach1 = 0 .. i_pause (prelude + approach up to the split point), approach2 = i_pause .. i_G, transport1 = the close
    hold's last frame (transport sample 0, at G) .. i_peak, transport2 = i_peak .. i_P. i_peak = path["i_peak"] when the path
    carries it (v2); M3's paths do not, so it is then the highest palm sample of the transport (Step 1's definition). A v2
    path also lists the slices itself (path["follow"]); they must be these."""
    pos = np.asarray(path["palm_pos"], float)
    phase = np.asarray([str(x) for x in path["phase"]])
    i_pause, i_G, i_P = int(path["i_pause"]), int(path["i_G"]), int(path["i_P"])
    close = np.flatnonzero(phase == "close")
    if not len(close):
        raise ValueError("path has no 'close' phase")
    i_t0 = int(close[-1])
    i_peak = int(path["i_peak"]) if "i_peak" in path else i_t0 + int(np.argmax(pos[i_t0:i_P + 1, 2]))
    if not 0 < i_pause < i_G <= i_t0 < i_peak < i_P:
        raise ValueError(f"slice indices out of order: i_pause {i_pause}, i_G {i_G}, transport start {i_t0}, i_peak {i_peak}, i_P {i_P}")
    seg = dict(approach1=(0, i_pause), approach2=(i_pause, i_G), transport1=(i_t0, i_peak), transport2=(i_peak, i_P))
    if "follow" in path:
        assert {k: tuple(v) for k, v in path["follow"].items()} == seg, f"path['follow'] {path['follow']} != {seg}"
    return seg


def slice_progress(path: dict, name: str, frames: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Normalised progress s_j in [0, 1] along slice ``name`` (frames first..last), for the place correction x' = x + s (P' - x_end).

    v2 slices: Step 1 samples the M3-timed segment x(tau) at tau = sigma(j / 50), with sigma(D) = D0 (its re-timing), so the
    progress along it is s = sigma / D0 -- from paths_v2.sigma itself, with that segment's D0 from the path's info. s starts
    and ends at rest like the slice, so the correction adds no velocity jump. M3 slices (smoke tests only; not re-timed):
    s = j / (n - 1). Returns (s, how it was computed)."""
    n = frames[1] - frames[0] + 1
    segs = {s["name"]: s for s in path.get("info", {}).get("v2", {}).get("segments", [])}
    if name not in segs:
        return np.arange(n) / (n - 1), "uniform j / (n - 1) (M3 timing: the path carries no v2 re-timing)"
    seg = segs[name]
    assert seg["frames"] == n and list(seg["v2_frames"]) == list(frames), f"{name}: path info {seg['v2_frames']} vs slice {frames}"
    D0 = float(seg["D0_s"])
    s = paths_v2.sigma(np.arange(n) / FPS, D0) / D0
    s[0], s[-1] = 0.0, 1.0       # as Step 1 sets its last sample exactly to x(D0)
    return s, f"paths_v2.sigma(j / {FPS}, D0) / D0, D0 = {D0:.2f} s (Step 1's re-timing of this segment)"


# ---------------------------------------------------------------------------------------------------------------- inputs
def m4_layouts(demo: dict, human: dict, oracle: dict, cup_height: float) -> dict[str, dict]:
    """M3's recorded layout (k is computed there; not run in M4), trial00-trial49 and recorded_v2 (module docstring)."""
    m3 = run_m3.layouts_for(demo, human, oracle, cup_height)            # recorded + trial00-09, M3's own code
    out = {"recorded": m3["recorded"]}
    legacy = Grasp.hero_side_grasp(cup_height)
    turn = float(human["phi_h"]) - float(oracle["phi_ora"])            # the handle turns with the grasp heading (M3)
    for t in trials(legacy)[:N_TRIALS]:
        name = f"trial{t['trial']:02d}"
        out[name] = dict(name=name, cup_xy=np.asarray(t["cup_xy"], float), coaster_xy=np.asarray(t["coaster_xy"], float),
                         handle_yaw=math.radians(t["handle_yaw_deg"]) + turn, handle_yaw_source="trials() handle_yaw_deg + (phi_h - phi_ora)",
                         trial_handle_yaw_deg=t["handle_yaw_deg"], cup_height=cup_height,
                         trial=t["trial"], layout_seed=t["layout_seed"], layout_index=t["layout_index"])
        if name in m3:
            assert all(np.array_equal(out[name][k], m3[name][k]) for k in ("cup_xy", "coaster_xy", "handle_yaw")), f"{name} differs from M3's"
    delta = np.asarray(demo["coaster_pos_table"], float)[:2] - np.asarray(demo["cup_init_pos_table"], float)[:2]
    off = legacy.palm_offset[:2]
    cup = np.array([np.mean(PALM_BOX["x"]), np.mean(PALM_BOX["y"])]) - off - delta / 2
    rec = m3["recorded"]
    out["recorded_v2"] = dict(name="recorded_v2", cup_xy=cup, coaster_xy=cup + delta, handle_yaw=rec["handle_yaw"],
                              handle_yaw_source=f"M3 recorded layout ({rec['handle_yaw_source']})", cup_height=cup_height,
                              nominal_grasp_palm_xy=cup + off, nominal_place_palm_xy=cup + delta + off)
    return out


def layout_names(spec: list[str]) -> list[str]:
    """Layout names; "trialAA-BB" expands to trialAA .. trialBB (inclusive)."""
    out = []
    for s in spec:
        m = re.fullmatch(r"trial(\d+)-(\d+)", s)
        out += [f"trial{k:02d}" for k in range(int(m[1]), int(m[2]) + 1)] if m else [s]
    for s in out:
        m = re.fullmatch(r"trial(\d+)", s)
        if not (s in ("recorded", "recorded_v2") or (m and int(m[1]) < N_TRIALS)):
            raise ValueError(f"unknown layout {s!r}")
    return out


def build_path(source: str, clip: str, variant: str, ctx: dict, layout: dict, p_s, q_now) -> dict:
    """The path dict for one run (palm_pos (T, 3), palm_quat_wxyz (T, 4), closure (T,), phase (T,), i_pause, i_G, i_P,
    i_release, info; v2 also i_prelude_end, i_close_end, i_peak and ``follow``). k is always computed at M3's recorded layout:
    run_m3.build_path gets it explicitly, paths_v2.v2_path recomputes it with warp.recorded_layout (the same call). Both get
    the oracle reference built from this task's compiled mug."""
    demo, oracle, layouts = ctx["demo"], ctx["oracle"], ctx["layouts"]
    if source == "m3":
        return run_m3.build_path(M3_VARIANT[variant], demo, layouts, layout, oracle, p_s, q_now)
    if source == "v2":
        return paths_v2.v2_path(clip, variant, layout, p_s, q_now, oracle=oracle)
    raise ValueError(f"path source must be one of {PATH_SOURCES}")


def path_sha256(path: dict) -> str:
    """Fingerprint of the played path (positions, quaternions, closure as float64 bytes, plus the phase labels)."""
    h = hashlib.sha256()
    for k in ("palm_pos", "palm_quat_wxyz", "closure"):
        h.update(np.ascontiguousarray(np.asarray(path[k], dtype=np.float64)).tobytes())
    h.update("|".join(str(x) for x in path["phase"]).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- run_one
_WORKER: dict = {}          # per process: the Execute task and each clip's context, built on first use


def worker_task() -> Execute:
    if "task" not in _WORKER:
        _WORKER["task"] = Execute()
    return _WORKER["task"]


def clip_context(clip: str) -> dict:
    """Demo, human values at g, the oracle reference (from the task's compiled mug) and the M4 layouts of one clip."""
    if clip not in _WORKER:
        task = worker_task()
        demo = load_demo(clip)
        oracle = oracle_reference(task.scene)
        human = human_at_grasp(demo, oracle["h"])
        _WORKER[clip] = dict(demo=demo, oracle=oracle, human=human, layouts=m4_layouts(demo, human, oracle, task.scene.cup_height))
    return _WORKER[clip]


def run_name(clip: str, variant: str, layout: str) -> str:
    return f"{clip}__{variant}__{layout}"


def resolve(spec: str):
    """"package.module:function" -> the function (diagnostics are passed by name so worker processes can import them)."""
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def collision_diagnostic(task: Execute, ctx: dict) -> dict:
    """Step 3 for b runs, after the physics run: retarget/collision.py collision_min on this run's layout and path (its own
    PickPlace.run setup, which is bit-identical to the run's, then M3's single plan). naive: None (Step 3 is for b runs)."""
    if ctx["variant"] == "naive" or ctx["path"] is None:
        return dict(collision_min=None)
    return dict(collision_min=collision_min(task, ctx["layout"], ctx["path"]))


def run_one(clip: str, variant: str, layout: str, *, paths: str = "v2", diagnostics: tuple[str, ...] = (COLLISION_DIAGNOSTIC,)) -> dict:
    """One run: (clip, variant, layout name) -> its row. Row = sim/m1_pick_mug.py's row fields (trial, layout_seed,
    layout_index, cup_xy, coaster_xy, handle_yaw_deg, mass_kg, RESULT_KEYS, wall_s, detail) + run, clip, variant, layout,
    paths, settles (top level) and the diagnostics' fields. ``diagnostics`` = "module:function" names (default: the Step 3
    collision minimum, as the CLI); each is called AFTER
    the physics run as f(task, ctx) with ctx = dict(clip, variant, layout (dict), path, p_s, q_now, demo, human, oracle,
    layouts) and returns a dict merged into the row (it may change the simulator state: the run is over)."""
    t0 = time.perf_counter()
    task, ctx = worker_task(), clip_context(clip)
    lay = ctx["layouts"][layout]
    made = {}

    def make_path(p_s, q_now):
        made.update(p_s=p_s, q_now=q_now, path=build_path(paths, clip, variant, ctx, lay, p_s, q_now))
        return made["path"]

    r = task.run_variant(variant, lay, make_path)
    path = made.get("path")
    row = dict(run=run_name(clip, variant, layout), clip=clip, variant=variant, layout=layout, paths=paths, profile=PROFILE,
               trial=lay.get("trial"), layout_seed=lay.get("layout_seed"), layout_index=lay.get("layout_index"),
               cup_xy=r["cup_xy"], coaster_xy=r["coaster_xy"], handle_yaw_deg=round(math.degrees(lay["handle_yaw"]), 2),
               trial_handle_yaw_deg=None if lay.get("trial_handle_yaw_deg") is None else round(lay["trial_handle_yaw_deg"], 2),
               mass_kg=task.scene.cup_mass, **{k: r.get(k) for k in RESULT_KEYS}, settles=r["settles"],
               diagnostics_run=list(diagnostics))
    detail = {k: v for k, v in r.items() if k not in row}
    if path is not None:
        detail["path"] = dict(source=paths, T=len(path["palm_pos"]), sha256=path_sha256(path),
                              **{k: int(path[k]) for k in INDEX_KEYS if k in path}, info=run_m3.jsonable(path.get("info", {})))
    row["detail"] = run_m3.jsonable(detail)
    for spec in diagnostics:
        extra = resolve(spec)(task, dict(clip=clip, variant=variant, layout=lay, path=path, p_s=made.get("p_s"), q_now=made.get("q_now"), **ctx))
        clash = set(extra) & set(row)
        assert not clash, f"diagnostic {spec} would overwrite {clash}"
        row.update(run_m3.jsonable(extra))
    row.update(wall_s=round(time.perf_counter() - t0, 1), worker_peak_rss_mb=peak_rss_mb())
    return run_m3.jsonable(row)


# ---------------------------------------------------------------------------------------------------------------- batch
def run_item(item: dict, paths: str, runs_dir: str, diagnostics: tuple[str, ...]) -> dict:
    """Worker: one item -> its row file, written as soon as the run ends. An exception becomes an error row (the batch goes
    on; ``run`` reports it and exits non-zero)."""
    name = run_name(item["clip"], item["variant"], item["layout"])
    try:
        row = run_one(item["clip"], item["variant"], item["layout"], paths=paths, diagnostics=diagnostics)
    except Exception:
        row = dict(run=name, **item, paths=paths, success=None, error=traceback.format_exc(), worker_peak_rss_mb=peak_rss_mb())
    (Path(runs_dir) / f"{name}.json").write_text(json.dumps(row, indent=2))
    if row["worker_peak_rss_mb"] > RSS_CAP_MB:
        raise MemoryError(f"worker {os.getpid()} peak RSS {row['worker_peak_rss_mb']} MB > {RSS_CAP_MB} MB (spec cap)")
    return dict(run=name, pid=os.getpid(), success=row.get("success"), category=row.get("category"), reason=row.get("reason"),
                error=row.get("error"), wall_s=row.get("wall_s"), peak_rss_mb=row["worker_peak_rss_mb"])


def cmd_run(args):
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    if args.items:
        items = json.loads(args.items.read_text())
    else:
        items = [dict(clip=c, variant=v, layout=l) for c in args.clips for v in args.variants for l in layout_names(args.layouts)]
    for it in items:                    # validate before any worker starts
        assert it["clip"] in CLIPS and it["variant"] in VARIANTS, it
        layout_names([it["layout"]])
    runs_dir = args.out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    exists = [it for it in items if (runs_dir / f"{run_name(it['clip'], it['variant'], it['layout'])}.json").exists()]
    if exists and args.skip_existing:
        items = [it for it in items if it not in exists]
    elif exists and not args.overwrite:
        raise SystemExit(f"{len(exists)} rows exist in {runs_dir}; pass --overwrite or --skip-existing")
    t0 = time.perf_counter()
    done, workers = [], {}
    diag = tuple(args.diagnostics)
    # fork: the parent has built nothing (no scene, no ONNX session), so every worker starts clean (as sim/m1_grasp_ladder.py);
    # HERO's policy runs one onnxruntime thread, so N workers keep N cores busy.
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(run_item, it, args.paths, str(runs_dir), diag) for it in items]
        for f in as_completed(futures):
            try:
                r = f.result()
            except BaseException:           # a worker broke the RSS cap or died: drop the queued runs instead of waiting for them
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            done.append(r)
            workers[r["pid"]] = max(workers.get(r["pid"], 0), r["peak_rss_mb"])
            status = ("ERROR " + r["error"].strip().splitlines()[-1]) if r["error"] else \
                     "SUCCESS" if r["success"] else f"FAIL {r['category']}: {r['reason']}"
            print(f"[{len(done):3d}/{len(items)}] {r['run']:<40} {status}  ({r['wall_s']} s)", flush=True)
    res = dict(command="run", items=len(items), workers=args.workers, paths=args.paths, diagnostics=list(diag),
               wall_s=round(time.perf_counter() - t0, 1), parent_peak_rss_mb=peak_rss_mb(),
               per_process_peak_rss_mb=sorted(workers.values()), errors=[r["run"] for r in done if r["error"]],
               successes=sum(bool(r["success"]) for r in done), gpu="none (CPU only, onnxruntime CPU)")
    (args.out / f"run_resources_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(res, indent=2))
    print(f"{res['successes']}/{len(items)} successes, {len(res['errors'])} errors, {res['wall_s']} s, peak RSS per process "
          f"{res['per_process_peak_rss_mb']} MB; rows in {runs_dir}")
    if res["errors"]:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------- rerun
def row_differences(a, b, prefix: str = "") -> list[str]:
    """Dotted names of the fields in which two rows differ (recursing into dicts; lists compared whole), VOLATILE left out."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if not prefix and k in VOLATILE:
                continue
            if k not in a or k not in b:
                out.append(prefix + k)
            else:
                out += row_differences(a[k], b[k], f"{prefix}{k}.")
        return out
    return [] if a == b else [prefix.rstrip(".")]


def cmd_rerun(args):
    saved = json.loads(args.row.read_text())
    t0 = time.perf_counter()
    again = run_one(saved["clip"], saved["variant"], saved["layout"], paths=saved["paths"], diagnostics=tuple(saved.get("diagnostics_run", ())))
    differing = row_differences(json.loads(json.dumps(again)), saved)
    report = dict(row=str(args.row), run=saved["run"], compared="every field, recursively, except " + ", ".join(VOLATILE),
                  identical=not differing, fields_differing=differing, saved_success=saved.get("success"), rerun_success=again["success"],
                  wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(f"{saved['run']}: rerun identical {report['identical']}" + (f"; differs in {differing}" if differing else ""))
    if differing:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------- determinism
def setup_identity(layout: dict) -> dict:
    """Execute's setup + 2 s stand vs PickPlace.run's own (run_m3.SetupOnly stops it at its first move), on one layout:
    qpos, qvel and sim time byte for byte."""
    ex, so = Execute(), run_m3.make_task()
    ex.setup(layout)
    ex.hold("stand", 2.0)
    run_m3.setup(so, layout)
    snap = lambda t: (t.scene.data.qpos.tobytes(), t.scene.data.qvel.tobytes(), float(t.scene.data.time))
    a, b = snap(ex), snap(so)
    return dict(layout="trial 0 (legacy-grasp cup / coaster, its handle yaw)", qpos_identical=a[0] == b[0], qvel_identical=a[1] == b[1],
                sim_time_identical=a[2] == b[2], identical=a == b)


def cmd_check_oracle(args):
    """Track B final's mug_v1 trials 0..n-1 again through the UNCHANGED oracle, compared with fifty.json, every field."""
    from sim.m1_grasp_final import PROFILE as FINAL_PROFILE, ForcePickPlace, reference_rows
    from sim.m1_grasp_ladder import run_trials, same_rows

    t0 = time.perf_counter()
    assert FINAL_PROFILE == GRASP_PROFILES[PROFILE]
    fifty = json.loads(FIFTY.read_text())
    part = run_trials("R3", FINAL_PROFILE, list(range(args.trials)), "oracle_recheck", ForcePickPlace, reference_rows("fifty"))
    rows = part["rows"]
    for r in rows:            # as sim/m1_grasp_final.py stores them: the force record at the top level
        r["grasp_forces"] = r["detail"].pop("grasp_forces")
    oracle = same_rows(rows, fifty["runs"][:args.trials], FIFTY, ref_name="track_b_fifty")
    t = trials(Grasp.hero_side_grasp(mug_config()["height_m"]))[0]
    setup = setup_identity(dict(cup_xy=np.asarray(t["cup_xy"]), coaster_xy=np.asarray(t["coaster_xy"]), handle_yaw=math.radians(t["handle_yaw_deg"])))
    out = dict(
        oracle_trials=dict(what=f"Track B final mug_v1 trials 0-{args.trials - 1} re-run through the unchanged oracle "
                                "(sim/m1_grasp_final.py ForcePickPlace via sim/m1_grasp_ladder.py run_trials, layouts asserted)",
                           **oracle, passed=oracle["trials_identical"] == args.trials),
        execute_setup_vs_pickplace=dict(**setup, passed=setup["identical"]),
        wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), gpu="none (CPU only)")
    out["passed"] = out["oracle_trials"]["passed"] and out["execute_setup_vs_pickplace"]["passed"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "determinism_checks.json").write_text(json.dumps(out, indent=2))
    print(f"oracle trials 0-{args.trials - 1}: {oracle['trials_identical']}/{args.trials} identical to fifty.json "
          f"(differences {oracle['differences']}); Execute setup = PickPlace setup: {setup['identical']}; "
          f"wrote {args.out / 'determinism_checks.json'} ({out['wall_s']} s)")
    if not out["passed"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run (clip, variant, layout) items on worker processes, one JSON row per run")
    run.add_argument("--items", type=Path, default=None, help="JSON list of {clip, variant, layout}; else the product below")
    run.add_argument("--clips", nargs="+", default=list(CLIPS), choices=CLIPS)
    run.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    run.add_argument("--layouts", nargs="+", default=[f"trial00-{N_TRIALS - 1:02d}"], help="names; trialAA-BB = a range; recorded_v2")
    run.add_argument("--paths", choices=PATH_SOURCES, default="v2")
    run.add_argument("--workers", type=int, default=MAX_WORKERS)
    run.add_argument("--diagnostics", nargs="*", default=[COLLISION_DIAGNOSTIC],
                     help=f"module:function names called after each run (default {COLLISION_DIAGNOSTIC}; give none to skip)")
    run.add_argument("--out", type=Path, default=OUT_DIR)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--skip-existing", action="store_true")
    rerun = sub.add_parser("rerun", help="re-run one saved row and compare it field by field")
    rerun.add_argument("--row", type=Path, required=True)
    rerun.add_argument("--report", type=Path, default=None)
    chk = sub.add_parser("check-oracle", help="Track B mug_v1 trials through the unchanged oracle + Execute setup identity")
    chk.add_argument("--trials", type=int, default=5)
    chk.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    dict(run=cmd_run, rerun=cmd_rerun, **{"check-oracle": cmd_check_oracle})[args.cmd](args)


if __name__ == "__main__":
    main()
