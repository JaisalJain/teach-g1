"""M1 check: HERO keeps our G1 standing, then reaches the right palm to 6 goals just above the table.

Protocol per table height (0.50 / 0.74 / 0.88 m):
1. Stand still for 10 s on a static reference (the robot's own spawn pose).
2. For each of 6 palm goals, one after the other (goal 0 starts from the stand, goals 1-5 from the previous goal's hold):
   - lift (unscored, only if the palm is below the table-clearing height): HERO's first approach waypoint -- straight up to
     ``_clearance_height`` (hand envelope >= 3 cm above the table) -- so the reach does not sweep the fingers through the slab;
   - initial reach: HERO's planner (``BenchReplanner.plan_from_state``) from the live robot state, with the pelvis drop / lean
     HERO's benchmark uses for this height (``reach_specs._bench_base``: squat below 0.60 m tables, stand above);
   - closed-loop correction = HERO's published protocol (``scripts/hero_bench.py`` REPLAN_ARGS): first replan at the end of
     the reach, then every 3 s, base held where it is, goal adjustment ``g <- g - 0.6 e`` (<= 1 cm per update), 0.3 s blend,
     inactive-arm posture prior = the initial plan's joints at reach end (as in ``sim2sim/bench/run.py``).
   Each goal gets 10 s from the start of the initial reach.
Goals follow HERO's benchmark ranges: x 0.35/0.45 m, y -0.12/-0.22/-0.32 m (right hand), palm 10 cm above the table
(raised if the hand envelope needs it), orientation = HERO's canonical side grasp (fingers forward, palm facing the midline).
Deviation from HERO's published runs: the policy's root-feedback inputs use the exact simulator pose (``odom_source="truth"``);
HERO's numbers use a LiDAR-inertial odometry noise model (``--odom so``).

Metrics per goal (right palm, world frame, vs the ORIGINAL goal): mean / max over the hold window (end of the initial reach
-> end of the 10 s segment) and final = mean over the last 1 s (HERO's "final-1 s"). Fell = HERO's rule (pelvis z < 0.30 m),
or pelvis tilt > 60 deg, or a body part other than feet / hands touching the floor or table.

Run:  env -u PYTHONPATH conda run -n hoi-sim python -m sim.m1_reach_test
Writes results/m1_reach/reach_results.json, results/m1_reach/series_h*.npz and results/m1_reach/reach_h074.mp4.
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

import imageio.v2 as imageio
import mujoco
import numpy as np

from data_tools import hero_reach_generator as gen
from data_tools import reach_specs as rs
from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.bench.replan import BenchGoal, BenchReplanner, ReplanConfig
from sim2sim.mathutil import quat_apply, xyzw_to_wxyz
from sim2sim.policy_hero_export import HeroExportPolicy
from sim2sim.reference import ClipReference
from sim2sim.state import read_state

from sim.scene import TABLE_HEIGHTS, TableScene

FPS = 50
STAND_S = 10.0
SEGMENT_S = 10.0
FINAL_WINDOW_S = 1.0
LIFT_SETTLE_STEPS = 25          # extra steps on the lift waypoint after its reach ends
GOAL_XS = (0.35, 0.45)
GOAL_YS = (-0.12, -0.22, -0.32)
GOAL_Z_ABOVE_TABLE = 0.10
FALL_PELVIS_Z = 0.30            # HERO's "fall" termination
FALL_TILT_DEG = 60.0
OUT_DIR = Path("results/m1_reach")

# HERO's closed-loop benchmark protocol (scripts/hero_bench.py REPLAN_ARGS), first event = end of the initial reach.
CORRECTION_CFG = ReplanConfig(first_event="reach_end", period_s=3.0, base_mode="current", goal_adjust=True, handover_blend_s=0.3)
# The initial reach (and the lift) move the base to HERO's benchmark pelvis plan for this height ("clip" = the goal's pelvis_drop / pitch).
REACH_CFG = ReplanConfig(first_event="start", base_mode="clip")


def standing_reference(scene: TableScene, n_frames: int) -> ClipReference:
    """A static clip of the robot's current pose (joints + FK of the 32 canonical bodies), as in HERO's make_fixture."""
    p = scene.plant
    p.forward()
    body_pos, body_quat = p.canonical_body_poses()
    joint_pos = np.concatenate([p.data.qpos[0:7], p.dof_pos])   # root quat stays wxyz, as from_arrays expects
    return ClipReference.from_arrays(fps=FPS, joint_pos=np.repeat(joint_pos[None], n_frames, 0),
                                     body_pos_w=np.repeat(body_pos[None], n_frames, 0),
                                     body_quat_w=np.repeat(xyzw_to_wxyz(body_quat)[None], n_frames, 0), name="stand")


def make_goals(scene: TableScene) -> list[BenchGoal]:
    """6 right-hand goals in HERO's benchmark ranges, with HERO's base strategy for this table height."""
    cfg = rs.PROFILES["hero_bench_v1"]
    squat = scene.table_height < cfg["bench_squat_below_m"]
    drop = float(np.mean(cfg["bench_squat_drop_range" if squat else "bench_stand_drop_range"]))
    R = rs.palm_rotation(0.0, 0.0, 0.0)                     # canonical inward-facing side grasp
    z = max(scene.table_height + GOAL_Z_ABOVE_TABLE, scene.table_height + rs.ee_min_height_above_surface(R, "right"))
    goals = []
    for x in GOAL_XS:
        for y in GOAL_YS:
            target = rs.HandTarget(hand="right", pos=(x, y, z), yaw=0.0, pitch=0.0, roll=0.0, canonical_grasp=True, yaw_clipped=False,
                                   approach="ingress", hover=rs.BENCH_HOVER_M, ingress=rs.BENCH_INGRESS_M, lift=rs.BENCH_LIFT_M)
            base, reachable = rs._bench_base(scene.table_height, drop, target, cfg)
            goals.append(BenchGoal(clip_name=f"h{round(scene.table_height * 100):03d}_x{round(x * 100)}_y{round(-y * 100)}", hand="right",
                                   pos_w=np.array([x, y, z]), quat_wxyz=rs.mat_to_quat_wxyz(R), table=scene.ik_table(), reach_end_frame=1,
                                   pelvis_drop=base.pelvis_drop, pelvis_pitch=base.pelvis_pitch, waist_pitch=base.waist_pitch,
                                   base_family=base.family + ("" if reachable else "_prefilter_failed"), height_label=f"h{round(scene.table_height * 100):03d}"))
    return goals


class Recorder:
    """Per-control-step log of what the report needs."""

    def __init__(self, scene: TableScene, video_path: Path | None):
        self.scene, self.rows = scene, []
        self.video = VideoWriter(scene, video_path) if video_path else None

    def add(self, phase: str, goal_index: int, goal: BenchGoal | None):
        scene = self.scene
        st = read_state(scene.plant)
        palm_p, palm_q = st.palm_pos_w[1], xyzw_to_wxyz(st.palm_quat_w[1])
        gravity_b = quat_apply(st.root_quat * np.array([-1, -1, -1, 1.0]), np.array([0, 0, -1.0]))   # world gravity in the pelvis frame
        contacts = scene.env_contacts()
        row = dict(t=float(scene.data.time), phase=phase, goal=goal_index, palm=palm_p, pelvis=st.root_pos.copy(),
                   tilt_deg=math.degrees(math.acos(np.clip(-gravity_b[2], -1, 1))), cup=scene.cup_pose()[0], touch=scene.touch(),
                   hand_env=contacts["hand"], body_env=contacts["body"], err_cm=np.nan, rot_err_deg=np.nan)
        if goal is not None:
            row["err_cm"] = float(np.linalg.norm(palm_p - goal.pos_w)) * 100
            row["rot_err_deg"] = math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(palm_q, goal.quat_wxyz))))))
        self.rows.append(row)
        if self.video:
            self.video.frame(None if goal is None else goal.pos_w)

    def close(self):
        if self.video:
            self.video.close()


class VideoWriter:
    """Offscreen MuJoCo render every 2nd control step (25 fps), with the current goal drawn as a green sphere."""

    def __init__(self, scene: TableScene, path: Path):
        self.renderer = mujoco.Renderer(scene.model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.25, -0.15, 0.75]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.2, 140.0, -25.0   # in front of the robot, on its right side
        self.writer = imageio.get_writer(path, fps=25, quality=7, macro_block_size=8)
        self.scene, self.n = scene, 0

    def frame(self, goal: np.ndarray | None):
        self.n += 1
        if self.n % 2:
            return
        self.renderer.update_scene(self.scene.data, camera=self.cam)
        scn = self.renderer.scene
        if goal is not None and scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]), goal,
                                np.eye(3).flatten(), np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32))
            scn.ngeom += 1
        self.writer.append_data(self.renderer.render())

    def close(self):
        self.writer.close()
        self.renderer.close()


def run_height(height: float, policy: HeroExportPolicy, video_path: Path | None) -> tuple[dict, dict]:
    # the M1 cylinder cup (not the mug), so the documented M1 numbers reproduce
    scene = TableScene(height, policy.kp, policy.kd, policy.effort_limit, policy.default_dof_pos, cup_model="cylinder")
    goals = make_goals(scene)
    stand_ref = standing_reference(scene, int(STAND_S * FPS) + 30)
    # inactive (left) arm prior: the spawn pose for new reaches; the initial plan's reach-end joints for corrections (HERO run.py)
    reach_planner = BenchReplanner(goals[0], REACH_CFG, arm_posture=stand_ref.joint_pos[0])
    correction_planner = BenchReplanner(goals[0], CORRECTION_CFG, reference=stand_ref)
    rec = Recorder(scene, video_path)

    def play(ref: ClipReference, n_steps: int, phase: str, gi: int, goal: BenchGoal | None):
        """Show frames 0..n_steps-1 of a freshly planned reference (frame j after j control steps, HERO's replan timing)."""
        for t in range(n_steps):
            scene.step(policy.control(ref, ref.clamp(t), read_state(scene.plant)))
            rec.add(phase, gi, goal)

    # ---- 1. stand: reset, HERO's zero-action hold step, then the policy on the static reference ----
    policy.reset()
    scene.step(policy.reset_target_for(stand_ref))
    for k in range(1, int(STAND_S * FPS) + 1):
        scene.step(policy.control(stand_ref, stand_ref.clamp(k), read_state(scene.plant)))
        rec.add("stand", -1, None)

    # ---- 2. reaches ----
    plans, n_seg = [], int(SEGMENT_S * FPS)
    for gi, goal in enumerate(goals):
        reach_planner.goal = goal
        reach_planner.reset()
        st = read_state(scene.plant)
        palm0, q0 = st.palm_pos_w[1], xyzw_to_wxyz(st.palm_quat_w[1])
        z_clear = gen._clearance_height(q0, goal.quat_wxyz, "right", scene.table_height)
        lift = None
        if palm0[2] < z_clear:  # HERO's first approach waypoint: straight up to the table-clearing height
            lift_goal = np.array([palm0[0], palm0[1], max(palm0[2] + 0.5 * rs.BENCH_LIFT_M, z_clear)])
            ref, info = reach_planner.plan_from_state(st, scene.plant, lift_goal)
            play(ref, int(info["n_reach"]) + LIFT_SETTLE_STEPS, "lift", gi, None)
            lift = dict(to_z=round(float(lift_goal[2]), 3), ik_terminal_err_cm=round(info["terminal_palm_err_cm"], 2))
            reach_planner.reset()

        t0 = time.perf_counter()
        ref, info = reach_planner.plan_from_state(read_state(scene.plant), scene.plant, goal.pos_w)
        info["ik_s"] = time.perf_counter() - t0
        n_reach = int(info["n_reach"])
        # the initial reach plays the role of the benchmark's original clip; frame j is shown at segment step j+1, so the
        # replanner's live frame is k - 1 (live_k0 = 1) and its first event fires when frame n_reach (reach end) is shown
        correction_planner.goal = dataclasses.replace(goal, reach_end_frame=n_reach + 1)
        correction_planner.reference = ref
        correction_planner.arm_posture = np.asarray(ref.joint_pos[ref.clamp(n_reach)], dtype=np.float64).copy()
        correction_planner.reset()
        correction_planner.live_k0 = 1
        ref_live, t_live = ref, 0
        for k in range(1, n_seg + 1):
            st = read_state(scene.plant)
            new_ref = correction_planner.step(k, st, scene.plant)
            if new_ref is not None:
                ref_live, t_live = new_ref, 0
            scene.step(policy.control(ref_live, t_live, st))
            rec.add("reach" if k <= n_reach else "hold", gi, goal)
            t_live = ref_live.clamp(t_live + 1)
        plans.append(dict(lift=lift, initial=info, correction=correction_planner.summary()))
    rec.close()
    return summarize(scene, goals, plans, rec.rows), series(rec.rows)


def summarize(scene: TableScene, goals: list[BenchGoal], plans: list[dict], rows: list[dict]) -> dict:
    pelvis_z = np.array([r["pelvis"][2] for r in rows])
    tilt = np.array([r["tilt_deg"] for r in rows])
    root_xy = np.array([r["pelvis"][:2] for r in rows])
    body_env = np.array([r["body_env"] for r in rows])
    cup = np.array([r["cup"] for r in rows])
    stand = [r for r in rows if r["phase"] == "stand"]
    n_final = int(FINAL_WINDOW_S * FPS)
    per_goal = []
    for gi, (goal, plan) in enumerate(zip(goals, plans)):
        seg = [r for r in rows if r["goal"] == gi and r["phase"] in ("reach", "hold")]
        hold = [r for r in seg if r["phase"] == "hold"]
        err = np.array([r["err_cm"] for r in hold])
        final = np.array([r["err_cm"] for r in seg[-n_final:]])
        final_rot = np.array([r["rot_err_deg"] for r in seg[-n_final:]])
        c, ini = plan["correction"], plan["initial"]
        per_goal.append(dict(
            goal=goal.clip_name, goal_pos_w=[round(float(v), 4) for v in goal.pos_w],
            base=dict(family=goal.base_family, pelvis_drop_m=round(goal.pelvis_drop, 3),
                      pelvis_pitch_deg=round(math.degrees(goal.pelvis_pitch), 1), waist_pitch_deg=round(math.degrees(goal.waist_pitch), 1)),
            palm_err_cm=dict(mean=round(float(err.mean()), 2), max=round(float(err.max()), 2), final=round(float(final.mean()), 2)),
            palm_rot_err_deg_final=round(float(final_rot.mean()), 1),
            lift=plan["lift"],
            initial_plan=dict(dist_cm=round(ini["dist_cm"], 1), reach_s=ini["reach_s"], ik_terminal_palm_err_cm=round(ini["terminal_palm_err_cm"], 2),
                              ik_max_path_err_cm=round(ini["max_palm_path_err_cm"], 2), ik_s=round(ini["ik_s"], 2)),
            correction=dict(n_replans=c["n_replans"], stayed_at_step=c["stayed_at_step"], goal_offset_cm=round(c["goal_offset_cm"], 2)),
            pelvis_z_min_m=round(min(r["pelvis"][2] for r in seg), 3), pelvis_tilt_max_deg=round(max(r["tilt_deg"] for r in seg), 1),
            hand_touched_env=any(r["hand_env"] for r in seg),
        ))
    errs = {k: [g["palm_err_cm"][k] for g in per_goal] for k in ("mean", "max", "final")}
    return dict(
        table_height_m=scene.table_height,
        fell=bool(pelvis_z.min() < FALL_PELVIS_Z or tilt.max() > FALL_TILT_DEG or body_env.any()),
        pelvis_z_min_m=round(float(pelvis_z.min()), 3), pelvis_tilt_max_deg=round(float(tilt.max()), 1),
        body_part_touched_floor_or_table=bool(body_env.any()),
        root_xy_drift_max_m=round(float(np.linalg.norm(root_xy - root_xy[0], axis=1).max()), 3),
        stand_phase=dict(seconds=STAND_S, pelvis_z_range_m=[round(min(r["pelvis"][2] for r in stand), 3), round(max(r["pelvis"][2] for r in stand), 3)],
                         root_xy_drift_m=round(float(np.linalg.norm(stand[-1]["pelvis"][:2] - stand[0]["pelvis"][:2])), 4),
                         pelvis_tilt_max_deg=round(max(r["tilt_deg"] for r in stand), 2)),
        goals=per_goal,
        palm_err_cm_over_goals={k: dict(mean=round(float(np.mean(v)), 2), max=round(float(np.max(v)), 2)) for k, v in errs.items()},
        cup_displacement_max_cm=round(float(np.linalg.norm(cup - cup[0], axis=1).max()) * 100, 2),
        fingertip_touch_max_N={n: round(max(r["touch"][n] for r in rows), 3) for n in rows[0]["touch"]},
    )


def series(rows: list[dict]) -> dict:
    return dict(t=np.array([r["t"] for r in rows], np.float32), goal=np.array([r["goal"] for r in rows], np.int16),
                phase=np.array([r["phase"] for r in rows]), err_cm=np.array([r["err_cm"] for r in rows], np.float32),
                palm=np.array([r["palm"] for r in rows], np.float32), pelvis=np.array([r["pelvis"] for r in rows], np.float32),
                tilt_deg=np.array([r["tilt_deg"] for r in rows], np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heights", type=float, nargs="+", default=list(TABLE_HEIGHTS))
    ap.add_argument("--video-height", type=float, default=0.74)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    policy = HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
    results = dict(protocol=dict(stand_s=STAND_S, segment_s=SEGMENT_S, final_window_s=FINAL_WINDOW_S, goal_xs=GOAL_XS, goal_ys=GOAL_YS,
                                 goal_z_above_table_m=GOAL_Z_ABOVE_TABLE, goal_order="goal 0 from the stand, goals 1-5 from the previous goal's hold",
                                 reach_cfg=REACH_CFG.as_dict(), correction_cfg=CORRECTION_CFG.as_dict(), odom="truth (HERO's published runs: so)",
                                 fall_rule=f"pelvis z < {FALL_PELVIS_Z} m, or pelvis tilt > {FALL_TILT_DEG} deg, or a non-foot/hand body part touching floor/table",
                                 policy=str(policy.onnx_path), mujoco=mujoco.__version__),
                   heights=[])
    for h in args.heights:
        t0 = time.perf_counter()
        video = args.out / f"reach_h{round(h * 100):03d}.mp4" if abs(h - args.video_height) < 1e-6 else None
        summary, ser = run_height(h, policy, video)
        summary["wall_time_s"] = round(time.perf_counter() - t0, 1)
        results["heights"].append(summary)
        np.savez_compressed(args.out / f"series_h{round(h * 100):03d}.npz", **ser)
        print(f"h={h:.2f}  fell={summary['fell']}  palm err over goals: {summary['palm_err_cm_over_goals']}  ({summary['wall_time_s']} s)")
    (args.out / "reach_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out / 'reach_results.json'}")


if __name__ == "__main__":
    main()
