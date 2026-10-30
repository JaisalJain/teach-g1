"""M3 harness: can HERO's whole-body IK follow the retargeted palm paths? Kinematic checks only.

One RUN = one clip x one variant x one layout (spec A4):
  a_naive          the mapped human palm replayed as recorded (retarget/naive.py). Recorded layout only: the path ignores where
                   the sim cup and coaster are, so its reachability is the same for every layout.
  b_level, b_tilt  the object-aware path (retarget/warp.py) with the two grasp poses of retarget/grasp.py, on the recorded
                   layout and on 10 random layouts.
4 clips x (1 a_naive + 2 b variants x 11 layouts) = 92 runs.

Per run, nothing is simulated beyond the setup:
1. Setup = PickPlace.run's own code up to its first move, mug_v1 profile (scene.reset with the layout's cup, coaster and
   handle yaw, standing reference, policy reset, the 2 s physics hold). ``SetupOnly`` runs PickPlace.run itself and stops it
   at its first ``move``, so this setup cannot drift from the oracle's.
2. The path starts at the standing palm p_s with orientation q_now (HERO's tracked palm after the hold, ``read_state``).
3. ONE ``PathPlanner.plan_path`` call for the whole dense path, base "plan" with base goal ``task._goal(G, R_G)`` (HERO's
   benchmark pelvis drop / lean for reaching the grasp pose; G, R_G from the path's info), ``return_fk=True``. Its first sample must be the IK model's
   ``palm_start`` within 1 mm (the IK's frame 0 is the robot's own palm, so a path starting elsewhere would jump).
4. FK errors: path target vs the clip's FK palm (training geometry, what HERO is asked to track), position and orientation,
   max over the path, at i_G and at i_P; ``qp_failures`` from plan_path. Reachable = at G and at P < 1.0 cm and < 10 deg,
   AND the path's max position error < 2 cm.
5. Joint-limit hits: path frames in which any of the 14 arm joints of the clip is within 0.01 rad of its URDF limit.
   (Information: HERO's IK keeps every joint >= 0.031 rad inside the URDF range on the arms -- its QP limits use 98 % of the
   range -- so a hit means the QP left its own limits.)
6. Hand-mug distance at G: the robot set to the clip's frame i_G with HERO's clip -> plant mapping (``set_robot_to_clip_frame``),
   the right fingers at HERO's open pose, the mug at its layout pose, mj_forward. Checked first: the plant palm equals the
   clip's FK palm within 1 mm. Then the minimum signed distance (mj_geomDistance; negative = overlap) from the 8 right-hand
   collision geoms to ``cup_geom`` (the body) and to the ``cup_handle_*`` capsules.

Layouts (spec A4):
- recorded: cup = PALM_BOX centre - the LEGACY grasp's palm offset xy (the mapping sample_layouts uses), coaster = cup + the
  clip's (coaster - cup) offset in xy, handle yaw = cup_init_yaw if valid, else phi_h. Flagged if the coaster's nominal place
  palm (coaster - legacy palm offset xy) lies outside PALM_BOX.
- trial00-09: trials 0-9 of sim.m1_pick_mug.trials() built with the legacy grasp (their cup and coaster xy), handle yaw =
  that trial's handle_yaw_deg + (phi_h - phi_ora), so the handle turns with the grasp heading.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.run_m3 ...):
  verify     -> results/m3_retarget/verify.json: palm axes (frames.verify_palm_axes; STOP if it fails), the plan_path
                return_fk byte-identity check, the setup's repeatability, and SetupOnly = warp.standing_palm (the setup the
                core's k comes from)
  run        -> runs/<clip>__<variant>__<layout>.json + paths/<same>.npz for all 92 runs, 4 worker processes;
                run_resources.json. Refuses to start without a passed verify.json, or over existing run files without --overwrite
  summarize  -> summary.json + summary.md (every table of spec A5), read back from the saved files
  videos     -> clip 4e103843db, recorded layout: re-plans a_naive, b_level and b_tilt (checking the numbers equal the saved
                runs), videos/<run>.mp4 (gitignored) + <clip>__b_level__recorded.gif (< 5 MB, committed) + videos.json.
                Renders like sim/m1_pick_mug.py videos: MUJOCO_GL=egl with __EGL_VENDOR_LIBRARY_FILENAMES (Mesa) so
                rendering stays off the NVIDIA GPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")       # before mujoco's first import, which picks the GL backend (only `videos` renders)

import mujoco
import numpy as np

from data_tools import hero_reach_generator as gen
from hero_isaacsim import constants as HC
from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.mathutil import xyzw_to_wxyz
from sim2sim.policy_hero_export import HeroExportPolicy
from sim2sim.state import read_state

from retarget.frames import PalmAxesError, load_demo, verify_palm_axes
from retarget.grasp import human_at_grasp, oracle_reference
from retarget.naive import naive_path
from retarget.warp import object_aware_path, recorded_layout, standing_palm
from sim.m1_grasp_causes import hand_parts
from sim.m1_grasp_ladder import DISTMAX, peak_rss_mb
from sim.m1_pick_mug import trials
from sim.m1_reach_test import VideoWriter
from sim.pick_place import GRASP_PROFILES, PALM_BOX, PREGRASP_BACKOFF, TABLE_HEIGHT, Grasp, PickPlace
from sim.reference_builder import min_jerk_path, quat_angle_deg

OUT = Path("results/m3_retarget")
CLIPS = ("26d4118b0c", "4e103843db", "49aff04f7a", "e8825dea53")
VARIANTS = ("a_naive", "b_level", "b_tilt")
N_RANDOM = 10                     # random layouts = trials 0-9 of sim.m1_pick_mug.trials()
N_PROCESSES = 4
PROFILE = "mug_v1"                # Track B final's grasp profile (radius offset, HERO closing targets, grasp contact model)
# Spec A4 thresholds (fixed by the spec, never tuned)
START_TOL_M = 0.001               # path sample 0 vs the IK's palm_start
REACH_POS_M, REACH_ORI_DEG = 0.010, 10.0      # at G and at P
REACH_PATH_M = 0.020              # max position error over the whole path
LIMIT_MARGIN_RAD = 0.01           # "within 0.01 rad of its limit"
MAPPING_TOL_M = 0.001             # plant palm vs clip FK palm at i_G
VIDEO_CLIP = "4e103843db"
GIF_MAX_BYTES = 5 * 1024 * 1024
GIF_STRIDES, GIF_SCALE = (2, 3, 4), 2     # GIF: every 2nd (else 3rd, 4th) video frame, half the video's 640 x 480 (make_gif)
CODE = ("retarget/frames.py", "retarget/grasp.py", "retarget/warp.py", "retarget/naive.py", "retarget/run_m3.py",
        "sim/reference_builder.py", "sim/pick_place.py", "sim/scene.py")
PATH_FIELDS = ("palm_pos", "palm_quat_wxyz", "closure", "phase", "i_pause", "i_G", "i_P", "i_release")   # spec A3's npz


# ---------------------------------------------------------------------------------------------------------------- inputs
def run_name(clip: str, variant: str, layout: str) -> str:
    return f"{clip}__{variant}__{layout}"


def run_names() -> list[str]:
    """The 92 runs, clip by clip: a_naive on the recorded layout, then each b variant on the recorded + 10 random layouts."""
    layouts = ["recorded"] + [f"trial{k:02d}" for k in range(N_RANDOM)]
    return [run_name(c, v, l) for c in CLIPS for v in VARIANTS for l in (layouts[:1] if v == "a_naive" else layouts)]


def layouts_for(demo: dict, human: dict, oracle: dict, cup_height: float) -> dict[str, dict]:
    """The recorded layout and the 10 random ones for one clip (module docstring). Cup / coaster xy = sim cup centres.
    The recorded layout is warp.recorded_layout's -- the same dict the time scale k is computed at."""
    out = {"recorded": recorded_layout(demo, oracle)}
    assert out["recorded"]["cup_height"] == cup_height
    legacy = Grasp.hero_side_grasp(cup_height)          # the random layouts are built with the LEGACY grasp (spec A4)
    turn = float(human["phi_h"]) - float(oracle["phi_ora"])     # the handle turns with the grasp heading
    for t in trials(legacy)[:N_RANDOM]:
        name = f"trial{t['trial']:02d}"
        out[name] = dict(name=name, cup_xy=np.asarray(t["cup_xy"], float), coaster_xy=np.asarray(t["coaster_xy"], float),
                         handle_yaw=math.radians(t["handle_yaw_deg"]) + turn, handle_yaw_source="trials() handle_yaw_deg + (phi_h - phi_ora)",
                         trial_handle_yaw_deg=t["handle_yaw_deg"], cup_height=cup_height)
    return out


# ---------------------------------------------------------------------------------------------------------------- setup
class _FirstMove(Exception):
    pass


class SetupOnly(PickPlace):
    """PickPlace whose ``run`` stops at its first ``move``: scene, policy and robot are then exactly where PickPlace.run has
    them before it plans the pregrasp (reset, standing reference, policy reset, reset step, 2 s hold). Nothing else changes."""

    def move(self, *args, **kwargs):
        raise _FirstMove


def make_task() -> SetupOnly:
    """The mug (config, 0.40 kg) with Track B final's mug_v1 profile; one per worker process, reused for all its runs
    (every run starts from scene.reset + policy.reset; verify checks that a setup does not depend on the previous one)."""
    return SetupOnly(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), cup_model="mug", **GRASP_PROFILES[PROFILE])


def setup(task: SetupOnly, layout: dict) -> None:
    try:
        task.run(layout["cup_xy"], layout["coaster_xy"], cup_yaw=layout["handle_yaw"])
    except _FirstMove:
        return
    raise RuntimeError("PickPlace.run finished without reaching its first move")      # e.g. the robot fell during the hold


def set_robot_to_clip_frame(scene, ref, t: int) -> None:
    """The robot at clip frame ``t``, mapped as HERO's sim2sim does it (sim2sim/bench/rollout.py ``rollout_clip``:
    ``plant.reset(ref.root_pos_w[t], ref.root_quat_w[t], ref.joint_pos[t])``): root position, root quaternion (xyzw -> MuJoCo's
    wxyz) and the 29 joints in HC.DOF_NAMES order at ``plant.dof_qadr``. Written into qpos directly, because plant.reset
    calls mj_resetData, which would also reset the fingers and the mug. Velocities stay as they are; call mj_forward after."""
    d, p = scene.data, scene.plant
    q = np.asarray(ref.root_quat_w[t], dtype=float)
    d.qpos[0:3] = ref.root_pos_w[t]
    d.qpos[3:7] = xyzw_to_wxyz(q / np.linalg.norm(q))
    d.qpos[p.dof_qadr] = ref.joint_pos[t]


def hand_mug_distances(scene) -> dict:
    """Minimum signed distance (mj_geomDistance, m; negative = overlap; meshes as their convex hulls) from the right-hand
    collision geoms (palm, thumb 0-2, index 0-1, middle 0-1) to the mug body ``cup_geom`` and to the ``cup_handle_*`` capsules."""
    m, d = scene.model, scene.data
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
    hand = sorted(hand_parts(scene))
    cup = [g for g in range(m.ngeom) if m.geom_bodyid[g] == scene.cup_body]
    mug, handles = [g for g in cup if name(g) == "cup_geom"], [g for g in cup if name(g).startswith("cup_handle_")]
    assert len(mug) == 1 and len(handles) == 3, [name(g) for g in cup]

    def closest(objs):
        dist, h, o = min((mujoco.mj_geomDistance(m, d, h, o, DISTMAX, None), h, o) for h in hand for o in objs)
        return dict(min_m=float(dist), hand_body=body(h), object_geom=name(o))
    return dict(hand_mug=closest(mug), hand_handle=closest(handles), n_hand_geoms=len(hand))


# ---------------------------------------------------------------------------------------------------------------- one run
def build_path(variant: str, demo: dict, layouts: dict, layout: dict, oracle: dict, p_s, q_now) -> dict:
    """The core's path for this run; k is always computed at the clip's recorded layout (passed explicitly)."""
    if variant == "a_naive":
        return naive_path(demo, layouts["recorded"], p_s, q_now, oracle=oracle)      # fixed translation: the recorded layout's
    return object_aware_path(demo, variant, layout, p_s, q_now, recorded=layouts["recorded"], oracle=oracle)


def run_one(task: SetupOnly, clip: str, variant: str, layout_name: str, demo: dict, human: dict, oracle: dict, layouts: dict):
    """One run (module docstring, steps 1-6). Returns (row for the JSON, the planned ClipReference, the path dict)."""
    t0 = time.perf_counter()
    layout = layouts[layout_name]
    setup(task, layout)
    s = task.scene
    st = read_state(s.plant)
    p_s, q_now = st.palm_pos_w[1].copy(), xyzw_to_wxyz(st.palm_quat_w[1])
    path = build_path(variant, demo, layouts, layout, oracle, p_s, q_now)
    pos, quat = np.asarray(path["palm_pos"], float), np.asarray(path["palm_quat_wxyz"], float)
    closure, phase = np.asarray(path["closure"], float), np.asarray(path["phase"])
    i_G, i_P = int(path["i_G"]), int(path["i_P"])
    T = len(pos)
    assert pos.shape == (T, 3) and quat.shape == (T, 4) and closure.shape == (T,) and phase.shape == (T,), "path arrays disagree in length"
    assert closure.min() >= 0.0 and closure.max() <= 1.0, "closure outside [0, 1]"
    assert 0 <= i_G < T and 0 <= i_P < T

    # 3. one plan for the whole path; base goal = the grasp pose (G, R_G) as the core defines it. For b it is the path sample
    # at i_G; for a_naive it is my mapped pose at g, and the resampled path sample at i_G lies within half a frame of it.
    G, quat_G = np.asarray(path["info"]["G"], float), np.asarray(path["info"]["quat_G_wxyz"], float)
    task.planner.goal = task._goal(G, quat_G)
    p0, q0 = task.planner.palm_start(st)
    start_err = float(np.linalg.norm(pos[0] - p0))
    assert start_err < START_TOL_M, f"{clip} {variant} {layout_name}: path sample 0 is {start_err * 1e3:.3f} mm from palm_start"
    ref, info, (fk_pos, fk_quat) = task.planner.plan_path(st, s.plant, pos, quat, base="plan", return_fk=True)

    # 4. FK errors and reachability
    e_pos = np.linalg.norm(fk_pos - pos, axis=1)
    e_ori = np.array([quat_angle_deg(a, b) for a, b in zip(fk_quat, quat)])
    reach = dict(at_G=bool(e_pos[i_G] < REACH_POS_M and e_ori[i_G] < REACH_ORI_DEG),
                 at_P=bool(e_pos[i_P] < REACH_POS_M and e_ori[i_P] < REACH_ORI_DEG),
                 path=bool(e_pos.max() < REACH_PATH_M))
    reachable = all(reach.values())
    # information: where along the path the errors are (the prelude is robot-specific, the rest comes from my demo)
    by_phase = {str(p): dict(pos_cm=float(e_pos[phase == p].max() * 100), ori_deg=float(e_ori[phase == p].max()))
                for p in dict.fromkeys(phase.tolist())}

    # 5. joint-limit hits over the path frames (clip joints, HC.DOF_NAMES order; URDF limits as the plant compiled them)
    arms = list(HC.ARM_DOF_IDX)
    q_arm = ref.joint_pos[:T, arms]
    lim = s.plant.dof_pos_limits[arms]
    margin = np.minimum(q_arm - lim[:, 0], lim[:, 1] - q_arm)            # (T, 14) distance to the nearer limit
    near = margin < LIMIT_MARGIN_RAD
    joint_hits = {HC.DOF_NAMES[j]: int(n) for j, n in zip(arms, near.sum(axis=0)) if n}

    # 6. mapping check and hand-mug distances at G (kinematics only: scene.reset puts the mug at its layout pose, both hands open)
    s.reset(layout["cup_xy"], layout["coaster_xy"], layout["handle_yaw"])
    set_robot_to_clip_frame(s, ref, i_G)
    mujoco.mj_forward(s.model, s.data)
    plant_palm_p, plant_palm_q = s.plant.palm_pose()
    map_err = float(np.linalg.norm(plant_palm_p[1] - fk_pos[i_G]))
    mapping = dict(plant_palm_minus_clip_fk_palm_mm=map_err * 1e3, tolerance_mm=MAPPING_TOL_M * 1e3, passed=map_err < MAPPING_TOL_M,
                   orientation_diff_deg=quat_angle_deg(xyzw_to_wxyz(plant_palm_q[1]), fk_quat[i_G]),
                   mapping="sim2sim/bench/rollout.py rollout_clip: plant.reset(ref.root_pos_w[t], ref.root_quat_w[t], ref.joint_pos[t])")
    distances = hand_mug_distances(s) if mapping["passed"] else None      # a wrong mapping would give meaningless distances

    row = dict(
        run=run_name(clip, variant, layout_name), clip=clip, variant=variant, layout=jsonable(layout), profile=PROFILE,
        human=jsonable(human), oracle=jsonable(oracle),
        path_info=jsonable(path.get("info", {})),
        path=dict(T=T, duration_s=T / 50.0, i_pause=int(path["i_pause"]), i_G=i_G, i_P=i_P, i_release=int(path["i_release"]),
                  sample_at_i_G=pos[i_G].tolist(), sample_at_i_P=pos[i_P].tolist(),
                  sample_at_i_G_minus_G_mm=float(np.linalg.norm(pos[i_G] - G) * 1e3),
                  sample_at_i_G_vs_R_G_deg=quat_angle_deg(quat[i_G], quat_G),
                  phases=[str(p) for p in dict.fromkeys(phase.tolist())]),
        setup=dict(p_s=p_s.tolist(), q_now_wxyz=q_now.tolist(), palm_start=np.asarray(p0).tolist(), palm_start_quat_wxyz=np.asarray(q0).tolist(),
                   path_start_minus_palm_start_mm=start_err * 1e3, path_start_vs_palm_start_ori_deg=quat_angle_deg(quat[0], q0)),
        plan=dict(base="plan", base_goal=dict(pos=task.planner.goal.pos_w.tolist(), quat_wxyz=task.planner.goal.quat_wxyz.tolist(),
                                              pelvis_drop=float(task.planner.goal.pelvis_drop), pelvis_pitch=float(task.planner.goal.pelvis_pitch),
                                              waist_pitch=float(task.planner.goal.waist_pitch)), **jsonable(info)),
        fk_error=dict(pos_cm=dict(max=float(e_pos.max() * 100), argmax_frame=int(e_pos.argmax()), argmax_phase=str(phase[e_pos.argmax()]),
                                  at_G=float(e_pos[i_G] * 100), at_P=float(e_pos[i_P] * 100)),
                      ori_deg=dict(max=float(e_ori.max()), argmax_frame=int(e_ori.argmax()), argmax_phase=str(phase[e_ori.argmax()]),
                                   at_G=float(e_ori[i_G]), at_P=float(e_ori[i_P])),
                      max_by_phase=by_phase),
        qp_failures=int(info["qp_failures"]),
        reachable=reachable, reachable_parts=reach,
        joint_limits=dict(hit_frames=int(near.any(axis=1).sum()), of_frames=T, frames_per_joint=joint_hits,
                          min_margin_rad=float(margin.min()), min_margin_joint=HC.DOF_NAMES[arms[int(margin.min(axis=0).argmin())]]),
        mapping_check=mapping, distances_at_G=distances,
        wall_s=round(time.perf_counter() - t0, 2), worker_peak_rss_mb=peak_rss_mb())
    return row, ref, path


def save_path(path: dict, file: Path) -> None:
    """The spec's npz (A3): palm_pos, palm_quat_wxyz, closure, phase (fixed-width unicode, so allow_pickle=False loads it), indices."""
    arrays = {k: np.asarray(path[k]) for k in PATH_FIELDS}
    arrays["phase"] = np.asarray([str(p) for p in path["phase"]], dtype=str)
    np.savez(file, **arrays)


def code_hashes() -> dict:
    """SHA-256 of the code a step ran with, so a result can be matched to the code that made it (summarize checks they agree)."""
    root = Path(__file__).resolve().parents[1]
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in CODE}


def jsonable(v):
    """numpy scalars / arrays -> plain Python, recursively (JSON)."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def clip_context(task: SetupOnly, clip: str) -> tuple[dict, dict, dict, dict]:
    """Demo, human values at g, the oracle reference and the layouts of one clip."""
    demo = load_demo(clip)
    oracle = oracle_reference(task.scene)                       # r = the compiled cup_geom radius, h = the scene's cup height
    human = human_at_grasp(demo, oracle["h"])
    return demo, human, oracle, layouts_for(demo, human, oracle, task.scene.cup_height)


# ---------------------------------------------------------------------------------------------------------------- run
def run_chunk(names: list[str]) -> dict:
    """One worker process: build the task once, run its runs in order, write each run's JSON and path as soon as it is done."""
    t0 = time.perf_counter()
    task, ctx = make_task(), {}
    for name in names:
        clip, variant, layout = name.split("__")
        if clip not in ctx:
            ctx[clip] = clip_context(task, clip)
        demo, human, oracle, layouts = ctx[clip]
        row, _, path = run_one(task, clip, variant, layout, demo, human, oracle, layouts)
        save_path(path, OUT / "paths" / f"{name}.npz")
        (OUT / "runs" / f"{name}.json").write_text(json.dumps(row, indent=2))
        fe = row["fk_error"]
        print(f"{name:<36} reachable {str(row['reachable']):<5} pos cm max {fe['pos_cm']['max']:6.2f} G {fe['pos_cm']['at_G']:5.2f} "
              f"P {fe['pos_cm']['at_P']:5.2f}  ori deg G {fe['ori_deg']['at_G']:5.1f} P {fe['ori_deg']['at_P']:5.1f}  "
              f"qp fail {row['qp_failures']}  limit hits {row['joint_limits']['hit_frames']}  ({row['wall_s']} s)", flush=True)
    return dict(runs=names, wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb())


def cmd_run(args):
    verify = OUT / "verify.json"
    if not verify.exists() or not json.loads(verify.read_text())["passed"]:
        raise SystemExit(f"{verify}: missing or a check failed -- run `verify` first (spec A1: STOP if the palm axes fail)")
    names = run_names()
    existing = [n for n in names if (OUT / "runs" / f"{n}.json").exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} run files exist in {OUT / 'runs'}; pass --overwrite to replace them")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "paths").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    chunks = [c.tolist() for c in np.array_split(np.array(names), N_PROCESSES)]
    # fork: the parent has built nothing (no scene, no ONNX session), so every worker starts clean (as sim/m1_grasp_ladder.py)
    with ProcessPoolExecutor(N_PROCESSES, mp_context=multiprocessing.get_context("fork")) as pool:
        parts = list(pool.map(run_chunk, chunks))
    res = dict(step="run", runs=len(names), processes=N_PROCESSES, chunks=chunks, wall_s=round(time.perf_counter() - t0, 1),
               parent_peak_rss_mb=peak_rss_mb(), per_process=[dict(runs=len(p["runs"]), wall_s=p["wall_s"], peak_rss_mb=p["peak_rss_mb"]) for p in parts],
               peak_rss_mb_sum_of_processes=sum(p["peak_rss_mb"] for p in parts), gpu="none (CPU only, onnxruntime CPU)",
               code_sha256=code_hashes())
    (OUT / "run_resources.json").write_text(json.dumps(res, indent=2))
    print(f"{len(names)} runs in {res['wall_s']} s; peak RSS per process {[p['peak_rss_mb'] for p in parts]} MB; wrote {OUT / 'runs'}")


# ---------------------------------------------------------------------------------------------------------------- verify
CLIP_ARRAYS = ("joint_pos", "joint_vel", "root_pos_from_joint_pos", "root_quat_from_joint_pos", "body_pos_w", "body_quat_w",
               "body_lin_vel_w", "body_ang_vel_w", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")


def same_clip(a, b) -> list[str]:
    """Fields in which two ClipReferences differ, byte for byte (empty = identical)."""
    diff = [k for k in CLIP_ARRAYS if getattr(a, k).dtype != getattr(b, k).dtype or getattr(a, k).shape != getattr(b, k).shape
            or getattr(a, k).tobytes() != getattr(b, k).tobytes()]
    return diff + [k for k in ("T", "fps", "name") if getattr(a, k) != getattr(b, k)]


def return_fk_check(task: SetupOnly) -> dict:
    """plan_path with and without return_fk on one real plan: the oracle's own pregrasp plan (PickPlace.run's first move) on
    trial 0's layout with mug_v1. Plans A (default), B (return_fk=True), C (default again) from the same state; A must equal B
    and C byte for byte, info included. Also: the returned FK palm equals the clip's own palm (ClipReference.palm_pose_w)."""
    t = trials(Grasp.hero_side_grasp(task.scene.cup_height))[0]
    layout = dict(cup_xy=np.asarray(t["cup_xy"]), coaster_xy=np.asarray(t["coaster_xy"]), handle_yaw=math.radians(t["handle_yaw_deg"]))
    setup(task, layout)
    s, g = task.scene, task.grasp
    st = read_state(s.plant)
    # PickPlace.run's pregrasp move, rebuilt from its own lines (sim/pick_place.py run + move)
    grasp_p = g.palm_for_cup(s.cup_pose()[0])
    pre_p = grasp_p - PREGRASP_BACKOFF * g.approach_dir
    palm0, q_now = st.palm_pos_w[1], xyzw_to_wxyz(st.palm_quat_w[1])
    z_clear = max(gen._clearance_height(q_now, g.quat_wxyz, "right", TABLE_HEIGHT), pre_p[2])
    task.planner.goal = task._goal(grasp_p, g.quat_wxyz)
    p0, q0 = task.planner.palm_start(st)
    pp, pq = min_jerk_path(p0, q0, [(np.array([palm0[0], palm0[1], z_clear]), g.quat_wxyz, 1.2), (pre_p, g.quat_wxyz, 1.8)])
    ref_a, info_a = task.planner.plan_path(st, s.plant, pp, pq, base="plan")
    ref_b, info_b, (fk_pos, fk_quat) = task.planner.plan_path(st, s.plant, pp, pq, base="plan", return_fk=True)
    ref_c, info_c = task.planner.plan_path(st, s.plant, pp, pq, base="plan")
    clip_palm = np.array([ref_b.palm_pose_w(i)[0][1] for i in range(len(pp))])
    clip_quat = np.array([xyzw_to_wxyz(ref_b.palm_pose_w(i)[1][1]) for i in range(len(pp))])
    out = dict(plan="PickPlace.run's pregrasp move (oracle waypoints, base 'plan'), trial 0 layout, mug_v1",
               path_frames=len(pp), clip_frames=ref_a.T, fields_compared=list(CLIP_ARRAYS) + ["T", "fps", "name", "info"],
               a_vs_b_differing=same_clip(ref_a, ref_b) + (["info"] if info_a != info_b else []),
               a_vs_c_differing=same_clip(ref_a, ref_c) + (["info"] if info_a != info_c else []), info=info_a,
               fk_shapes=[list(fk_pos.shape), list(fk_quat.shape)],
               fk_pos_vs_clip_palm_max_abs_m=float(np.abs(fk_pos - clip_palm).max()),
               fk_quat_vs_clip_palm_max_deg=max(quat_angle_deg(a, b) for a, b in zip(fk_quat, clip_quat)))
    out["identical"] = not out["a_vs_b_differing"] and not out["a_vs_c_differing"]
    return out


def setup_repeat_check(task: SetupOnly) -> dict:
    """A setup must not depend on the run before it (a worker reuses one task): set up layout X, then Y, then X again and
    compare the full MuJoCo state (qpos, qvel, time) and the HERO policy's state-derived inputs after the 2 s hold."""
    ts = trials(Grasp.hero_side_grasp(task.scene.cup_height))
    lay = [dict(cup_xy=np.asarray(t["cup_xy"]), coaster_xy=np.asarray(t["coaster_xy"]), handle_yaw=math.radians(t["handle_yaw_deg"])) for t in ts[:2]]
    snap = lambda: (task.scene.data.qpos.tobytes(), task.scene.data.qvel.tobytes(), float(task.scene.data.time))
    setup(task, lay[0])
    first = snap()
    setup(task, lay[1])
    setup(task, lay[0])
    again = snap()
    return dict(layouts="trial 0, then trial 1, then trial 0 again", qpos_identical=first[0] == again[0], qvel_identical=first[1] == again[1],
                sim_time_s=first[2], sim_time_identical=first[2] == again[2], plans_before_first_move=len(task.plans),
                identical=first == again)


def setup_vs_standing_palm(task: SetupOnly) -> dict:
    """The core computes k from ``warp.standing_palm`` (PickPlace.run's setup lines, copied); the runs use ``SetupOnly``
    (PickPlace.run itself). Both at one clip's recorded layout: the palm (p_s, q_now) must be bit-identical, so k is the same."""
    demo, human, oracle, layouts = clip_context(task, CLIPS[0])
    setup(task, layouts["recorded"])
    st = read_state(task.scene.plant)
    mine = (st.palm_pos_w[1].tobytes(), xyzw_to_wxyz(st.palm_quat_w[1]).tobytes())
    p_s, q_now, _ = standing_palm(task, layouts["recorded"])
    return dict(clip=CLIPS[0], layout="recorded", identical=mine == (p_s.tobytes(), q_now.tobytes()), p_s=p_s.tolist(), q_now_wxyz=q_now.tolist())


def cmd_verify(args):
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(step="verify")
    task = make_task()
    try:
        out["palm_axes"] = jsonable(verify_palm_axes(task))            # spec A1; its own dict has "passed"
    except PalmAxesError as e:      # spec A1: if either condition fails, STOP and report (the measurement is kept)
        out["palm_axes"] = jsonable(e.result) | dict(error=str(e))
    out["return_fk"] = jsonable(return_fk_check(task))
    out["setup_repeatable"] = setup_repeat_check(task)
    out["setup_vs_warp_standing_palm"] = setup_vs_standing_palm(task)
    out["passed"] = bool(out["palm_axes"]["passed"] and out["return_fk"]["identical"] and out["setup_repeatable"]["identical"]
                         and out["setup_vs_warp_standing_palm"]["identical"])
    out.update(runtime_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), gpu="none (CPU only)", mujoco=mujoco.__version__,
               code_sha256=code_hashes())
    (OUT / "verify.json").write_text(json.dumps(out, indent=2))
    print(f"palm axes passed: {out['palm_axes']['passed']} {out['palm_axes']['conditions']}")
    print(f"return_fk byte identity: {out['return_fk']['identical']} (A vs B differing {out['return_fk']['a_vs_b_differing']}, "
          f"A vs C {out['return_fk']['a_vs_c_differing']}); setup repeatable: {out['setup_repeatable']['identical']}; "
          f"setup = warp.standing_palm: {out['setup_vs_warp_standing_palm']['identical']}")
    print(f"wrote {OUT / 'verify.json'} ({out['runtime_s']} s, peak RSS {out['peak_rss_mb']} MB)")
    if not out["palm_axes"]["passed"]:
        raise SystemExit("STOP: the palm-axes check failed (spec A1)")


# ---------------------------------------------------------------------------------------------------------------- videos
def attach_frame(path: dict) -> tuple[int, str]:
    """Where the mug starts to move with the palm: the end of the close hold = the last frame of the "close" phase. A path
    without a "close" phase (a_naive: closure ramps at the mapped t_contact, with no hold) uses the first frame at which
    closure reaches 1."""
    phase, closure = [str(p) for p in path["phase"]], np.asarray(path["closure"], float)
    if "close" in phase:
        return len(phase) - 1 - phase[::-1].index("close"), "last frame of the 'close' phase (end of the close hold)"
    full = np.flatnonzero(closure >= 1.0)
    if not len(full):
        raise ValueError("no close phase and closure never reaches 1")
    return int(full[0]), "first frame with closure = 1 (no 'close' phase in this path)"


def render(task: SetupOnly, ref, path: dict, layout: dict, mp4: Path) -> dict:
    """Kinematic replay (spec A5): the robot from the clip's joints, the right fingers blended open -> closed by the path's
    closure (TableScene.set_hand_closure: HERO's open pose to the hand's closed pose, here HERO's), the mug at its layout
    pose until the attach frame, then rigidly with the palm (offset palm -> mug captured at that frame) up to the start of the
    release (i_release: closure is still 1 there, the opening ramp starts on the next frame), then left where it is. One video
    frame per path frame; VideoWriter keeps every 2nd (25 fps)."""
    s, m, d = task.scene, task.scene.model, task.scene.data
    s.reset(layout["cup_xy"], layout["coaster_xy"], layout["handle_yaw"])
    i_attach, rule = attach_frame(path)
    i_release = int(path["i_release"])
    closure = np.asarray(path["closure"], float)
    a = s.cup_qadr
    mug_in_palm = None
    writer = VideoWriter(s, mp4)
    for i in range(len(closure)):
        set_robot_to_clip_frame(s, ref, i)
        s.set_hand_closure("right", closure[i])
        d.qpos[s.finger_qadr[7:]] = s.finger_goal[7:]          # right hand = the last 7 finger joints
        mujoco.mj_kinematics(m, d)
        palm_p, palm_q = s.plant.palm_pose()
        R_palm = quat_to_mat(xyzw_to_wxyz(palm_q[1]))
        if i == i_attach:           # mug pose in the palm frame, captured once
            R_mug = quat_to_mat(d.qpos[a + 3:a + 7])
            mug_in_palm = (R_palm.T @ (d.qpos[a:a + 3] - palm_p[1]), R_palm.T @ R_mug)
        if mug_in_palm is not None and i_attach <= i <= i_release:
            d.qpos[a:a + 3] = palm_p[1] + R_palm @ mug_in_palm[0]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (R_palm @ mug_in_palm[1]).flatten())
            d.qpos[a + 3:a + 7] = q
        mujoco.mj_forward(m, d)
        writer.frame(None)
    writer.close()
    return dict(attach_frame=i_attach, attach_rule=rule, release_frame=i_release, frames=len(closure),
                mug_carried_frames=max(0, i_release - i_attach + 1))


def quat_to_mat(q_wxyz) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=float))
    return R.reshape(3, 3)


def make_gif(mp4: Path, gif: Path) -> dict:
    """GIF from the mp4 at 1/GIF_SCALE size, playing in real time. Frame stride: the first of GIF_STRIDES (every 2nd video
    frame = 12.5 fps, then every 3rd, every 4th) that keeps the file under 5 MB -- an encoding rule fixed before the first
    real render (the path length depends on the clip's time scale k); every attempt is reported."""
    import imageio.v2 as imageio
    from PIL import Image
    reader = imageio.get_reader(mp4)
    video = [Image.fromarray(f) for f in reader]
    reader.close()
    small = [img.resize((img.width // GIF_SCALE, img.height // GIF_SCALE), Image.LANCZOS) for img in video]
    attempts = []
    for stride in GIF_STRIDES:
        frames, ms = small[::stride], round(1000 * stride / 25)          # the mp4 runs at 25 fps
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
        attempts.append(dict(stride=stride, frames=len(frames), frame_ms=ms, bytes=gif.stat().st_size))
        if attempts[-1]["bytes"] < GIF_MAX_BYTES:
            break
    return dict(gif=str(gif), size=[small[0].width, small[0].height], attempts=attempts, used=attempts[-1],
                bytes=attempts[-1]["bytes"], under_5MB=attempts[-1]["bytes"] < GIF_MAX_BYTES)


def cmd_videos(args):
    # This laptop's NVIDIA GPU is reserved for other jobs: EGL must load Mesa's driver (the Intel iGPU) instead of NVIDIA's.
    if not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        raise SystemExit("videos: set __EGL_VENDOR_LIBRARY_FILENAMES (here: /usr/share/glvnd/egl_vendor.d/50_mesa.json) so "
                         "rendering does not land on the NVIDIA GPU")
    t0 = time.perf_counter()
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    task = make_task()
    demo, human, oracle, layouts = clip_context(task, VIDEO_CLIP)
    report = []
    for variant in VARIANTS:
        name = run_name(VIDEO_CLIP, variant, "recorded")
        saved = json.loads((OUT / "runs" / f"{name}.json").read_text())
        row, ref, path = run_one(task, VIDEO_CLIP, variant, "recorded", demo, human, oracle, layouts)
        drop = lambda r: {k: v for k, v in r.items() if k not in ("wall_s", "worker_peak_rss_mb")}
        differing = [k for k in drop(row) if json.loads(json.dumps(drop(row)))[k] != drop(saved).get(k)]
        saved_path = np.load(OUT / "paths" / f"{name}.npz", allow_pickle=False)
        path_identical = all(np.array_equal(saved_path[k], np.asarray(path[k]) if k != "phase" else np.asarray([str(p) for p in path["phase"]]))
                             for k in PATH_FIELDS)
        mp4 = OUT / "videos" / f"{name}.mp4"
        entry = dict(run=name, video=str(mp4), rerun_fields_differing_from_saved_run=differing, path_identical_to_saved=path_identical,
                     **render(task, ref, path, layouts["recorded"], mp4))
        if variant == "b_level":
            entry["gif"] = make_gif(mp4, OUT / f"{name}.gif")
        report.append(entry)
        print(f"{name}: rerun differs in {differing or 'nothing'}, path identical {path_identical}; mug carried frames "
              f"{entry['attach_frame']}-{entry['release_frame']} ({entry['attach_rule']}); wrote {mp4}"
              + (f"; GIF {entry['gif']['bytes'] / 1e6:.2f} MB" if "gif" in entry else ""))
    out = dict(step="videos", clip=VIDEO_CLIP, layout="recorded", renderer="sim/m1_reach_test.py VideoWriter (640 x 480, every 2nd "
               "frame -> 25 fps), MUJOCO_GL=egl on Mesa (__EGL_VENDOR_LIBRARY_FILENAMES)", videos=report,
               runtime_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), code_sha256=code_hashes())
    (OUT / "videos.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / 'videos.json'} ({out['runtime_s']} s, peak RSS {out['peak_rss_mb']} MB)")


# ---------------------------------------------------------------------------------------------------------------- summarize
R0_FILE = OUT / "r0_after_m3" / "runs" / "R0_after_m3.json"      # python -m sim.m1_grasp_ladder run --rung R0 --tag after_m3 --out ...
L1_CHECK = Path("results/m1_grasp_fix/v3/l1_check.json")         # the oracle's own planned grasp pose, open hand (read only)


def place_palm_flags(layout: dict) -> dict:
    """The recorded layout's place-palm flag as the spec writes it (coaster - legacy palm offset, from warp.recorded_layout) and,
    as information, with the sign sample_layouts implies (palm = cup + palm offset, so the place palm = coaster + offset)."""
    off = Grasp.hero_side_grasp(layout["cup_height"]).palm_offset[:2]
    plus = np.asarray(layout["coaster_xy"]) + off
    inside = lambda p: bool(PALM_BOX["x"][0] <= p[0] <= PALM_BOX["x"][1] and PALM_BOX["y"][0] <= p[1] <= PALM_BOX["y"][1])
    return dict(spec_formula="coaster - legacy palm_offset[:2]", spec_xy=layout["coaster_place_palm_xy"],
                spec_in_box=layout["coaster_place_palm_in_palm_box"], coaster_plus_offset_xy=plus.tolist(), coaster_plus_offset_in_box=inside(plus),
                palm_box=PALM_BOX)


def stats(v: list) -> dict | None:
    v = [x for x in v if x is not None]
    return dict(n=len(v), min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v))) if v else None


def deg(rad) -> float | None:
    return None if rad is None else math.degrees(rad)


def table_row(r: dict) -> dict:
    """One run as a flat row of the run table (cm, deg, mm)."""
    fe, dist, info = r["fk_error"], r["distances_at_G"], r["path_info"]
    return dict(run=r["run"], clip=r["clip"], variant=r["variant"], layout=r["layout"]["name"], reachable=r["reachable"],
                reachable_parts=r["reachable_parts"],
                pos_err_cm_max=fe["pos_cm"]["max"], pos_err_cm_G=fe["pos_cm"]["at_G"], pos_err_cm_P=fe["pos_cm"]["at_P"],
                pos_err_cm_max_phase=fe["pos_cm"]["argmax_phase"],
                pos_err_cm_max_after_prelude=max(v["pos_cm"] for p, v in fe["max_by_phase"].items() if p != "prelude"),
                pos_err_cm_max_by_phase={p: v["pos_cm"] for p, v in fe["max_by_phase"].items()},
                ori_err_deg_max=fe["ori_deg"]["max"], ori_err_deg_G=fe["ori_deg"]["at_G"], ori_err_deg_P=fe["ori_deg"]["at_P"],
                qp_failures=r["qp_failures"], joint_limit_hit_frames=r["joint_limits"]["hit_frames"], frames=r["path"]["T"],
                n_clamped=info.get("n_clamped"), bump_scale=info.get("bump_scale"),
                hand_mug_mm=dist["hand_mug"]["min_m"] * 1e3 if dist else None,
                hand_handle_mm=dist["hand_handle"]["min_m"] * 1e3 if dist else None,
                mapping_mm=r["mapping_check"]["plant_palm_minus_clip_fk_palm_mm"],
                start_mm=r["setup"]["path_start_minus_palm_start_mm"])


def cmd_summarize(args):
    t0 = time.perf_counter()
    names = run_names()
    missing = [n for n in names if not (OUT / "runs" / f"{n}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} runs missing (first: {missing[0]}); run `run` first")
    rows = [json.loads((OUT / "runs" / f"{n}.json").read_text()) for n in names]
    verify = json.loads((OUT / "verify.json").read_text())
    run_res = json.loads((OUT / "run_resources.json").read_text())
    r0 = json.loads(R0_FILE.read_text())
    videos = json.loads((OUT / "videos.json").read_text()) if (OUT / "videos.json").exists() else None
    table = [table_row(r) for r in rows]

    per_clip, reach, dists = {}, {}, {}
    for clip in CLIPS:
        mine = [r for r in rows if r["clip"] == clip]
        rec = {r["variant"]: r for r in mine if r["layout"]["name"] == "recorded"}
        h, o = rec["b_level"]["human"], rec["b_level"]["oracle"]
        # human and oracle values are per clip: the same in every run of the clip
        assert all(r["human"] == h and r["oracle"] == o for r in mine), f"{clip}: human / oracle values differ between runs"
        info = rec["b_level"]["path_info"]
        b_rows = [t for t in table if t["clip"] == clip and t["variant"] != "a_naive"]
        per_clip[clip] = dict(
            phi_h_deg=deg(h["phi_h"]), phi_ora_deg=deg(o["phi_ora"]), palm_point_azimuth=h.get("azimuth_info"),
            h_frac_h=h["h_frac_h"], h_frac_ora=o["h_frac_ora"], k=info.get("k"), a0=info.get("a0"), a0_time_s=info.get("a0_time_s"),
            a0_z_m=info.get("a0_z_m"), v_h_m_s=info["time_scale"]["v_h"], v_o_m_s=info["time_scale"]["v_o"],
            n_clamped_recorded={v: rec[v]["path_info"].get("n_clamped") for v in ("b_level", "b_tilt")},
            n_clamped_over_layouts={v: stats([t["n_clamped"] for t in b_rows if t["variant"] == v]) for v in ("b_level", "b_tilt")},
            k_same_in_every_run=len({json.dumps(r["path_info"].get("k")) for r in mine}) == 1,
            recorded_layout=rec["b_level"]["layout"], coaster_place_palm=place_palm_flags(rec["b_level"]["layout"]))
        reach[clip], dists[clip] = {}, {}
        for v in VARIANTS:
            vr = [t for t in table if t["clip"] == clip and t["variant"] == v]
            reach[clip][v] = dict(reachable=sum(t["reachable"] for t in vr), of=len(vr), fraction=sum(t["reachable"] for t in vr) / len(vr),
                                  parts_passed={k: sum(t["reachable_parts"][k] for t in vr) for k in ("at_G", "at_P", "path")},
                                  layouts="recorded only; the path ignores the layout, so this holds for every layout" if v == "a_naive"
                                  else "recorded + trials 0-9")
            dists[clip][v] = dict(hand_mug_mm=stats([t["hand_mug_mm"] for t in vr]), hand_handle_mm=stats([t["hand_handle_mm"] for t in vr]),
                                  recorded_hand_mug_mm=next(t["hand_mug_mm"] for t in vr if t["layout"] == "recorded"),
                                  recorded_hand_handle_mm=next(t["hand_handle_mm"] for t in vr if t["layout"] == "recorded"))
    l1 = json.loads(L1_CHECK.read_text())["b_mug_L1"]
    oracle_ref = dict(file=str(L1_CHECK), field="b_mug_L1", min_signed_distance_mm=l1["min_signed_distance_mm"], min_pair=l1["min_pair"],
                      handle_min_distance_mm=l1["handle_min_distance_mm"],
                      note="information: the mug with the mug_v1 (radius) offset placed exactly where the oracle plans it relative to "
                           "the palm (no IK error), open hand, at the 20 trials' handle yaws (minimum reported)")
    reach_all = {v: dict(reachable=sum(reach[c][v]["reachable"] for c in CLIPS), of=sum(reach[c][v]["of"] for c in CLIPS)) for v in VARIANTS}
    chk = r0["legacy_check"]
    checks = dict(
        palm_axes=verify["palm_axes"], return_fk_byte_identity=verify["return_fk"]["identical"],
        setup_repeatable=verify["setup_repeatable"]["identical"],
        r0_after_m3=dict(file=str(R0_FILE), successes=r0["summary"]["successes"], trials=r0["summary"]["trials"],
                         trials_identical_to_m1=chk["vs_m1"]["trials_identical"], trials_identical_to_b3_r0=chk["vs_b3_r0"]["trials_identical"],
                         passed=r0["summary"]["successes"] == 20 and chk["vs_m1"]["trials_identical"] == 20),
        path_start_vs_palm_start_mm_max=max(t["start_mm"] for t in table),
        mapping_check_passed=sum(r["mapping_check"]["passed"] for r in rows), mapping_check_of=len(rows),
        mapping_mm_max=max(t["mapping_mm"] for t in table),
        same_code=dict(now=code_hashes(), verify_equal=verify.get("code_sha256") == code_hashes(), run_equal=run_res.get("code_sha256") == code_hashes(),
                       videos_equal=(videos.get("code_sha256") == code_hashes()) if videos else None),
        coaster_place_palm_outside_palm_box=[c for c in CLIPS if not per_clip[c]["recorded_layout"]["coaster_place_palm_in_palm_box"]],
        coaster_plus_offset_outside_palm_box_information=[c for c in CLIPS if not per_clip[c]["coaster_place_palm"]["coaster_plus_offset_in_box"]])
    resources = dict(verify=dict(runtime_s=verify["runtime_s"], peak_rss_mb=verify["peak_rss_mb"]),
                     run=dict(wall_s=run_res["wall_s"], parent_peak_rss_mb=run_res["parent_peak_rss_mb"], per_process=run_res["per_process"],
                              peak_rss_mb_sum_of_processes=run_res["peak_rss_mb_sum_of_processes"]),
                     r0_after_m3=dict(wall_s=r0["resources"]["wall_s"], peak_rss_mb_sum_of_processes=r0["resources"]["peak_rss_mb_sum_of_processes"],
                                      peak_rss_mb_max_process=max(p["peak_rss_mb"] for p in r0["resources"]["per_process"])),
                     videos=dict(runtime_s=videos["runtime_s"], peak_rss_mb=videos["peak_rss_mb"]) if videos else None,
                     gpu="none (CPU only; videos render on Mesa EGL)")
    summary = dict(
        protocol=dict(
            runs=f"{len(rows)} = 4 clips x (a_naive on the recorded layout + b_level and b_tilt on the recorded layout and trials 0-9)",
            profile=f"PickPlace(cup_model='mug', **GRASP_PROFILES['{PROFILE}'])", setup="PickPlace.run up to its first move (SetupOnly), incl. the 2 s hold",
            plan="one PathPlanner.plan_path(base='plan', base_goal=task._goal(G, R_G), return_fk=True) per run",
            reachable=f"FK error at G and at P < {REACH_POS_M * 100:g} cm and < {REACH_ORI_DEG:g} deg, AND path max position error < {REACH_PATH_M * 100:g} cm",
            joint_limit_hit=f"a path frame in which any of the 14 arm joints is within {LIMIT_MARGIN_RAD} rad of its URDF limit",
            distances="mj_geomDistance from the 8 right-hand collision geoms (open hand) to cup_geom / cup_handle_* at the clip's frame i_G; "
                      "negative = overlap", angles="deg; phi = heading of the palm normal, atan2(y, x)"),
        checks=checks, per_clip=per_clip, reachability=reach, reachability_all_clips=reach_all, distances_at_G=dists,
        oracle_planned_pose_distance_reference=oracle_ref,
        recorded_layout=[t for t in table if t["layout"] == "recorded"], runs=table, videos=videos, resources=resources)
    summary["resources"]["summarize"] = dict(runtime_s=round(time.perf_counter() - t0, 2), peak_rss_mb=peak_rss_mb())
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    (OUT / "summary.md").write_text(summary_md(summary))
    print(summary_md(summary))
    print(f"wrote {OUT / 'summary.json'} and {OUT / 'summary.md'}")


PHASE_ORDER = ("prelude", "to_a0", "replay", "approach", "pause", "close", "transport", "release", "retreat")    # (b) and (a) phases


def num(v, spec: str = ".2f") -> str:
    """A number for the markdown tables ("-" when missing)."""
    return "-" if v is None else format(v, spec)


def summary_md(s: dict) -> str:
    c = s["checks"]
    L = ["# M3 kinematic retargeting: reachability on HERO's whole-body IK", "",
         "Made by `python -m retarget.run_m3 summarize` from results/m3_retarget/runs/*.json, verify.json, run_resources.json, "
         "r0_after_m3/runs/R0_after_m3.json and videos.json. Kinematic only: no physics beyond PickPlace.run's 2 s setup hold.", "",
         f"Reachable = {s['protocol']['reachable']}. Joint-limit hit = {s['protocol']['joint_limit_hit']}. "
         f"Distances: {s['protocol']['distances']}.", "",
         "## Checks", "",
         f"- Palm axes (frames.verify_palm_axes): {'pass' if c['palm_axes']['passed'] else 'FAIL'}.",
         f"- plan_path(return_fk) default output byte-identical (A vs B vs C): {c['return_fk_byte_identity']}; setup repeatable: {c['setup_repeatable']}.",
         f"- R0 after the plan_path change: {c['r0_after_m3']['successes']}/{c['r0_after_m3']['trials']}, identical to M1 in "
         f"{c['r0_after_m3']['trials_identical_to_m1']}/20 trials, to B3 R0 in {c['r0_after_m3']['trials_identical_to_b3_r0']}/20.",
         f"- verify, run and videos ran on the current code (SHA-256 of {len(CODE)} files): verify {c['same_code']['verify_equal']}, "
         f"run {c['same_code']['run_equal']}, videos {c['same_code']['videos_equal']}.",
         f"- Path sample 0 vs palm_start: max {c['path_start_vs_palm_start_mm_max']:.3f} mm (limit 1 mm, asserted per run).",
         f"- Clip -> plant mapping at i_G: {c['mapping_check_passed']}/{c['mapping_check_of']} within 1 mm (max {c['mapping_mm_max']:.4f} mm).",
         f"- Recorded layouts whose coaster place palm (spec formula: coaster - legacy palm offset) lies outside PALM_BOX: "
         f"{', '.join(c['coaster_place_palm_outside_palm_box']) or 'none'}. Information, with coaster + offset (palm = cup + offset, as "
         f"sample_layouts implies): {', '.join(c['coaster_plus_offset_outside_palm_box_information']) or 'none'}.", "",
         "## Per clip", "",
         "| clip | phi_h (deg) | phi_ora (deg) | palm-point azimuth | h_frac_h | h_frac_ora | k | a0 time (s) | clamped frames, recorded (b_level / b_tilt) | clamped, max over layouts (b_level / b_tilt) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for clip, p in s["per_clip"].items():
        az = num(p["palm_point_azimuth"]["azimuth_deg"], ".1f") + " deg"
        ov = p["n_clamped_over_layouts"]
        L.append(f"| {clip} | {num(p['phi_h_deg'], '.1f')} | {num(p['phi_ora_deg'], '.1f')} | {az} | {num(p['h_frac_h'], '.3f')} | {num(p['h_frac_ora'], '.3f')} | "
                 f"{num(p['k'], '.3f')} | {num(p['a0_time_s'], '.3f')} | {p['n_clamped_recorded']['b_level']} / {p['n_clamped_recorded']['b_tilt']} | "
                 f"{num(ov['b_level'] and ov['b_level']['max'], 'g')} / {num(ov['b_tilt'] and ov['b_tilt']['max'], 'g')} |")
    L += ["", "## Reachability (runs reachable / runs)", "", "| clip | a_naive | b_level | b_tilt |", "|---|---|---|---|"]
    for clip, r in s["reachability"].items():
        L.append(f"| {clip} | " + " | ".join(f"{r[v]['reachable']}/{r[v]['of']}" for v in VARIANTS) + " |")
    ra = s["reachability_all_clips"]
    L.append("| all clips | " + " | ".join(f"{ra[v]['reachable']}/{ra[v]['of']}" for v in VARIANTS) + " |")
    L += ["", "a_naive runs only on the recorded layout: its path ignores the layout, so its reachability holds for every layout.", "",
          "Which part of the rule passed (runs passing / runs): at G (< 1 cm and < 10 deg), at P (same), path (max position error < 2 cm).", "",
          "| clip | variant | at G | at P | path max < 2 cm | reachable |", "|---|---|---|---|---|---|"]
    for clip, r in s["reachability"].items():
        for v in VARIANTS:
            pp = r[v]["parts_passed"]
            L.append(f"| {clip} | {v} | {pp['at_G']}/{r[v]['of']} | {pp['at_P']}/{r[v]['of']} | {pp['path']}/{r[v]['of']} | {r[v]['reachable']}/{r[v]['of']} |")
    L += ["", "Information: the largest FK position error per phase (cm), recorded layout. The prelude (arms at the sides -> over the "
          "table) is robot-specific; the other phases carry my demo.", "",
          "| run | where the path max is | " + " | ".join(PHASE_ORDER) + " |", "|---|---|" + "---|" * len(PHASE_ORDER)]
    for t in s["recorded_layout"]:
        L.append(f"| {t['run']} | {t['pos_err_cm_max_phase']} | " + " | ".join(num(t["pos_err_cm_max_by_phase"].get(p)) for p in PHASE_ORDER) + " |")
    L += ["",
          "## Hand-mug and hand-handle distances at G (mm; open hand; negative = overlap)", "",
          "| clip | variant | hand-mug, recorded | hand-mug min / median / max | hand-handle, recorded | hand-handle min / median / max |",
          "|---|---|---|---|---|---|"]
    o = s["oracle_planned_pose_distance_reference"]
    L.insert(len(L) - 2, f"Reference (information): at the oracle's own planned grasp pose (mug_v1 offset, no IK error, open hand) the "
                         f"hand-mug distance is {o['min_signed_distance_mm']} mm ({o['min_pair']['hand_body']}), hand-handle "
                         f"{o['handle_min_distance_mm']} mm ({o['file']}, {o['field']}).")
    L.insert(len(L) - 2, "")
    for clip, dv in s["distances_at_G"].items():
        for v, d in dv.items():
            hm, hh = d["hand_mug_mm"], d["hand_handle_mm"]
            L.append(f"| {clip} | {v} | {num(d['recorded_hand_mug_mm'], '.1f')} | " + (f"{hm['min']:.1f} / {hm['median']:.1f} / {hm['max']:.1f}" if hm else "-")
                     + f" | {num(d['recorded_hand_handle_mm'], '.1f')} | " + (f"{hh['min']:.1f} / {hh['median']:.1f} / {hh['max']:.1f}" if hh else "-") + " |")
    head = ("| run | reachable | pos err max / G / P (cm) | ori err max / G / P (deg) | qp failures | limit-hit frames / frames | clamped | "
            "bump scale | hand-mug (mm) | hand-handle (mm) |")
    line = lambda t: (f"| {t['run']} | {'yes' if t['reachable'] else 'no'} | {t['pos_err_cm_max']:.2f} / {t['pos_err_cm_G']:.2f} / {t['pos_err_cm_P']:.2f} | "
                      f"{t['ori_err_deg_max']:.1f} / {t['ori_err_deg_G']:.1f} / {t['ori_err_deg_P']:.1f} | {t['qp_failures']} | "
                      f"{t['joint_limit_hit_frames']} / {t['frames']} | {num(t['n_clamped'], 'd') if isinstance(t['n_clamped'], int) else num(t['n_clamped'])} | "
                      f"{num(t['bump_scale'], '.3f')} | {num(t['hand_mug_mm'], '.1f')} | {num(t['hand_handle_mm'], '.1f')} |")
    L += ["", "## Recorded layout", "", head, "|---|---|---|---|---|---|---|---|---|---|"] + [line(t) for t in s["recorded_layout"]]
    L += ["", "## All runs", "", head, "|---|---|---|---|---|---|---|---|---|---|"] + [line(t) for t in s["runs"]]
    if s["videos"]:
        L += ["", "## Videos (clip 4e103843db, recorded layout)", ""]
        for v in s["videos"]["videos"]:
            L.append(f"- {v['run']}: {v['video']} (gitignored); mug carried from frame {v['attach_frame']} ({v['attach_rule']}) to "
                     f"{v['release_frame']}; re-planned run differs from the saved one in: {v['rerun_fields_differing_from_saved_run'] or 'nothing'}"
                     + (f"; GIF {v['gif']['gif']} {v['gif']['bytes'] / 1e6:.2f} MB (< 5 MB: {v['gif']['under_5MB']})" if "gif" in v else ""))
    r = s["resources"]
    L += ["", "## Runtime and peak RSS (CPU only)", "", "| step | wall (s) | peak RSS (MB) |", "|---|---|---|",
          f"| verify | {r['verify']['runtime_s']} | {r['verify']['peak_rss_mb']} |",
          f"| run (4 processes) | {r['run']['wall_s']} | max process {max(p['peak_rss_mb'] for p in r['run']['per_process'])}, "
          f"sum {r['run']['peak_rss_mb_sum_of_processes']}, parent {r['run']['parent_peak_rss_mb']} |",
          f"| R0 after_m3 (4 processes) | {r['r0_after_m3']['wall_s']} | max process {r['r0_after_m3']['peak_rss_mb_max_process']}, "
          f"sum {r['r0_after_m3']['peak_rss_mb_sum_of_processes']} |"]
    if r["videos"]:
        L.append(f"| videos | {r['videos']['runtime_s']} | {r['videos']['peak_rss_mb']} |")
    L.append(f"| summarize | {r['summarize']['runtime_s']} | {r['summarize']['peak_rss_mb']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="palm axes, plan_path return_fk byte identity, setup repeatability")
    run = sub.add_parser("run", help="all 92 runs on 4 worker processes")
    run.add_argument("--overwrite", action="store_true", help="replace existing run files")
    sub.add_parser("summarize", help="summary.json + summary.md from the saved files")
    sub.add_parser("videos", help="clip 4e103843db recorded layout: mp4s + the b_level GIF")
    args = ap.parse_args()
    dict(verify=cmd_verify, run=cmd_run, summarize=cmd_summarize, videos=cmd_videos)[args.cmd](args)


if __name__ == "__main__":
    main()
