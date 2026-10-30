"""Track B v3: the grasp fix ladder on R3 (my mug from config, 0.40 kg, B3's 20 trials), and the L1 geometry check.

A rung = one of B3's object runs (sim/m1_grasp_causes.py RUNS) + a dict of named PickPlace parameters. Every parameter's
default is its legacy (M1) value, so an empty dict is the M1 oracle. Ladder rungs are cumulative: each adds its own change to
the previous rung's dict.
  R0  B3's R0 (M1 cylinder, 0.15 kg), {}          legacy check: must give 20/20, every trial identical to B3's R0 (and so to M1)
  L1  B3's R3, {grasp_offset: "radius"}           radius-consistent palm target (sim/pick_place.py radius_consistent_offset)
  L2  L1 + {closing_targets: "ik_mug"}            mug-specific finger closing targets (sim/l2_closing_targets.py ->
                                                  results/m1_grasp_fix/v3/l2_targets.json, which the run reads)
  L3  L2 + {contact_model: "grasp"}               elliptic cones, impratio 10, 3 NoSlip iterations, condim 4 on the right
                                                  fingertips and palm (sim/scene.py GRASP_OPTIONS; doc quotes: L3_SOURCES)

L3's guards (spec): cone, impratio and NoSlip are global options, so they also act on the feet. L3 is accepted only if, with
L3's contact model, (1) the M1 reach test at 0.74 m shows no fall (sim/m1_reach_test.py --contact-model grasp, written to
reach_l3/) and (2) R0 stays 20/20 (rung "L3_R0" = R0's cylinder and trials with {contact_model: "grasp"} only: L1 is a no-op on
the 80 mm cylinder, radius_consistent_offset(0.040) == GRASP_OFFSET, and L2's targets are for the 99 mm mug). A rejected L3
still has its R3 numbers recorded. As a check of the reach test's new option, the reach test is also run with its legacy
default (reach_legacy/) and compared with M1's results/m1_reach at 0.74 m.

Trials = B3's, held fixed for every rung: trials 0-19 of sim.m1_pick_mug.trials() built with the LEGACY grasp (cup and coaster
xy; R3's handle yaws, R0 at yaw 0). sample_layouts derives the cup positions from the grasp's palm offset (and trials() the
yaws from its approach direction), so a rung's own grasp would re-sample them. Each worker asserts its trials equal B3's rows
before running them.

A run = 20 trials as 4 parallel CPU processes of 5 consecutive trials (B3 ran them one after another in one process). Each
process builds its own scene, policy and planners, and every trial starts from scene.reset() + policy.reset(), so a trial
should not depend on which trials ran before it in its process; R0 checks this against B3, field by field.
Rows = B3's (m1_grasp_causes.trial_row). Summary = B3's run_summary: success with Wilson 95 % CI and M1's failure categories,
lift slip mean and median, cup rise, thumb + finger + palm contact at the lift frame (of 20).

ladder.json = one line per rung, in ladder order, read back from the saved runs files (so every number in it comes from a
run): its change, its named parameters, success /20 with Wilson CI, lift slip mean and median, cup rise, contact fraction at
lift and failure categories; plus R0's legacy check and, for reference, B3's R3 (the mug before any rung). It is rebuilt after
every full 20-trial run, which appends the new rung. Stop rule (spec): the ladder stops at the first rung with >= 18/20, so a
rung after that one is not run (``run`` refuses). A rung with guards (L3) carries them and ``accepted`` (null = no guard in the
spec for that rung, or a guard not run yet); the final evaluation needs a rung that meets the stop rule and was not rejected.

L1 check (``check``): at the planned grasp pose with the hand open, the minimum signed distance (mj_geomDistance; meshes collide
as their convex hulls) between the right-hand collision geoms and the object, for (a) the M1 cylinder with the legacy offset
and (b) the mug with the radius-consistent offset. L1 keeps the palm-to-surface gap along the palm normal, so (a) and (b) are
expected to agree within 0.5 mm. If they do not, the per-geom table shows which hand geom differs; the offset is never adjusted.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.m1_grasp_ladder ...):
  check                -> results/m1_grasp_fix/v3/l1_check.json
  run --rung R0 | L1 | L2 | L3 | L3_R0 -> results/m1_grasp_fix/v3/runs/<rung>.json (refuses to replace it unless --overwrite),
                             then ladder.json
  (L3's reach guard: python -m sim.m1_reach_test --contact-model grasp --heights 0.74 --out results/m1_grasp_fix/v3/reach_l3;
   its legacy check: --heights 0.74 --video-height 0 --out results/m1_grasp_fix/v3/reach_legacy)
  run --rung R0 --tag T   -> runs/R0_T.json: the legacy check again after a change to shared code (sim/scene.py,
                             sim/pick_place.py), without replacing the earlier R0.json; ladder.json lists it under legacy_r0_rechecks
  ladder               -> results/m1_grasp_fix/v3/ladder.json from the runs files already saved (nothing is simulated)
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.interactive_scene import FINGER_SUFFIXES
from sim2sim.mathutil import xyzw_to_wxyz
from sim2sim.policy_hero_export import HeroExportPolicy
from sim2sim.state import read_state

from sim.m1_grasp_causes import CONTACT_RULE, LIFT_FRAME, RUNS, ContactPickPlace, hand_parts, r0_vs_m1, run_summary, trial_row
from sim.m1_pick_mug import mass_properties, trials
from sim.pick_place import GRASP_OFFSET, L2_TARGETS, LIFT_SLIP_FAIL_M, Grasp, PickPlace, radius_consistent_offset
from sim.scene import CUP_RADIUS

OUT_DIR = Path("results/m1_grasp_fix/v3")
REPO = Path(__file__).resolve().parents[1]
B3_RUNS = Path("results/m1_grasp_fix/b3_runs")      # B3's raw runs (v2, read only): the trials, and R0's reference rows
B3_SUMMARY = Path("results/m1_grasp_fix/b3_causes.json")     # B3's per-run summaries (v2, read only): R3 = the mug before any rung
N_TRIALS, N_PROCESSES = 20, 4
# (rung, its own change as named PickPlace parameters, what it changes) in order; cumulative
LADDER = [("L1", dict(grasp_offset="radius"),
           "radius-consistent palm target: object axis at palm-frame (0.077, r + 0.015, 0) with r from the object (mug body "
           "cylinder, 0.0495 m) instead of the fixed (0.077, 0.055, 0); 8 cm pre-grasp backoff unchanged"),
          ("L2", dict(closing_targets="ik_mug"),
           "finger closing targets for this radius: the right hand's closed pose = per-digit fingertip IK (5 mm radially inside "
           "the 99 mm mug, from each digit's B2v3 max-wrap configuration; results/m1_grasp_fix/v3/l2_targets.json) instead of "
           "HERO's closed pose; open pose, ramp, gains, torque limits and squeeze rule unchanged"),
          ("L3", dict(contact_model="grasp"),
           "contact model: elliptic friction cones, impratio 10 and 3 NoSlip iterations (global solver options, MuJoCo docs' advice "
           "against slip), condim 4 on the right fingertip (thumb_2, index_1, middle_1) and palm geoms (the mug geoms and the "
           "cup pairs are condim 4 already); friction coefficients, solref / solimp, gains, torque limits, mug geometry and "
           "mass unchanged (sim/scene.py GRASP_OPTIONS)")]
# Guards a rung must pass to be accepted (spec: only L3, whose options are global and so also act on the feet). Both run with
# the rung's own change only, on the legacy tasks: the M1 reach test at 0.74 m (no fall) and R0's cylinder (20/20).
GUARDS = {"L3": dict(r0_rung="L3_R0", reach_dir=Path("results/m1_grasp_fix/v3/reach_l3"), reach_height=0.74,
                     legacy_reach_dir=Path("results/m1_grasp_fix/v3/reach_legacy"), contact_model="grasp")}
M1_REACH = Path("results/m1_reach")                  # M1's reach test (read only): the legacy reference
SPEC_RUNGS = ("L1", "L2", "L3")                      # the spec's ladder, in order (LADDER holds the ones implemented so far)
STOP_SUCCESSES = 18                                  # spec: stop at the first rung with >= 18/20
CHECK_TOLERANCE_M = 0.0005                           # L1 check: (a) and (b) are expected to agree within 0.5 mm
DISTMAX = 1.0                                        # mj_geomDistance search range: larger than anything here, so distances are exact
LAYOUT_KEYS = ("trial", "layout_seed", "layout_index", "cup_xy", "coaster_xy", "cup_yaw_deg")

# Where L3's values come from (ladder.json). Quotes are verbatim from the MuJoCo docs of the installed version (3.13.0: doc/*.rst
# at tag 3.13.0, RST cross-reference markup removed). Spot check: the rendered "stable" Modeling page reads the same in the
# Slow slippage, Solver settings and Performance tuning passages, including "A small number (1, 2 or 3) is usually sufficient."
DOCS_SRC = "https://github.com/google-deepmind/mujoco/blob/3.13.0/doc/"
DOCS = "https://mujoco.readthedocs.io/en/stable/"
L3_SOURCES = [
    dict(url=DOCS + "modeling.html#cslowslippage", source=DOCS_SRC + "modeling.rst",
         section="Modeling > Tips and tricks > Preventing slip > Slow slippage", used_for="cone, impratio, noslip_iterations",
         quote="After ruling out the faster failure modes above, slow slippage can be reduced in two ways. a. Use elliptic cones and "
               "increase impratio. This makes the friction dimensions harder relative to the normal dimension without increasing the "
               "friction coefficient. It reduces slow slippage but does not guarantee exact sticking. The slip rate also depends on "
               "contact parameters, loading, geometry and dynamics, so impratio alone does not define a universal rate. b. For "
               "stronger suppression, enable the NoSlip solver by setting noslip_iterations to a positive integer. A small number "
               "(1, 2 or 3) is usually sufficient. [...] Exact zero slip is not guaranteed. NoSlip also adds computational cost, "
               "makes inverse dynamics ill-defined and can occasionally cause instabilities in complex multi-contact systems."),
    dict(url=DOCS + "modeling.html#calgorithms", source=DOCS_SRC + "modeling.rst", section="Modeling > Introduction > Solver settings",
         used_for="cone, impratio, noslip_iterations",
         quote="When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the Newton "
               "algorithm with very small tolerance. If that is not sufficient, enable the Noslip solver."),
    dict(url=DOCS + "overview.html#soft", source=DOCS_SRC + "overview.rst", section="Overview > Softness and slip",
         used_for="cone, impratio",
         quote="Thus using the Newton solver with elliptic friction cones and large value of impratio is the recommended way of "
               "reducing slip."),
    dict(url=DOCS + "XMLreference.html#option-impratio", source=DOCS_SRC + "XMLreference.rst", section="XML Reference > option > impratio",
         used_for="impratio (only with elliptic cones)",
         quote="Settings larger than 1 cause friction forces to be \"harder\" than normal forces, having the general effect of "
               "reducing slow slippage without increasing the actual friction coefficient or guaranteeing exact sticking. For "
               "pyramidal friction cones the situation is more complex because the pyramidal approximation mixes normal and "
               "frictional dimensions within each basis vector; it is not recommended to use high impratio values with pyramidal cones."),
    dict(url=DOCS + "XMLreference.html#option-noslip-iterations", source=DOCS_SRC + "XMLreference.rst",
         section="XML Reference > option > noslip_iterations", used_for="noslip_iterations",
         quote="Maximum number of iterations of the NoSlip solver. This is a post-processing step executed after the main solver. It "
               "uses a modified PGS method to suppress slip/drift in friction dimensions resulting from the soft-constraint model. "
               "The default setting 0 disables this post-processing step."),
    dict(url=DOCS + "computation/index.html#sonoslip", source=DOCS_SRC + "computation/index.rst", section="Computation > Constraint solver > Algorithms > NoSlip",
         used_for="noslip_iterations (its caveat)",
         quote="This suppresses the contact slip that is inherent to soft-constraint models. However, this cascade of optimization "
               "steps no longer solves a single well-defined optimization problem; it is an ad-hoc correction that can occasionally "
               "cause instabilities in models with complex multi-contact interactions."),
    dict(url=DOCS + "XMLreference.html#option-cone", source=DOCS_SRC + "XMLreference.rst", section="XML Reference > option > cone",
         used_for="cone", quote="Elliptic cones are a better model of the physical reality, but pyramidal cones sometimes make the "
                                "solver faster and more robust."),
    dict(url=DOCS + "modeling.html#cslippage", source=DOCS_SRC + "modeling.rst", section="Modeling > Tips and tricks > Preventing slip",
         used_for="condim 4", quote="Torsional friction is insufficient to apply the required torques. Increase condim to 4 or 6 and "
                                    "choose appropriate friction coefficients. condim 4 enables torsional friction, preventing rotation "
                                    "around the normal."),
]
L3_VALUE_CHOICE = dict(
    impratio=10.0, noslip_iterations=3,
    note="The docs say 'increase' / 'large' impratio but give no number anywhere (searched every doc page at tag 3.13.0). 10 is the "
         "value DeepMind's MuJoCo Menagerie pairs with cone=\"elliptic\" in its grippers and hands (main branch, read 2026-10-30: "
         "robotiq_2f85/2f85.xml, aloha/aloha.xml, wonik_allegro/right_hand.xml, shadow_hand/right_hand.xml: <option "
         "cone=\"elliptic\" impratio=\"10\"/>; leap_hand/right_hand.xml uses 100; unitree_g1/g1_with_hands.xml sets neither). "
         "noslip_iterations 3 = the top of the docs' '1, 2 or 3'. Both fixed before any L3 run; no other value was tried.",
    not_changed="The docs' solver advice also says 'Newton algorithm with very small tolerance' (Solver settings: 'say 1e-10'); the "
                "solver is already Newton, and its tolerance stays 1e-8 because the spec names only cone, condim, impratio and "
                "noslip_iterations.")


def rungs() -> dict[str, dict]:
    """R0 (the legacy check) and every ladder rung: its B3 object run, its own change and its cumulative PickPlace parameters."""
    out, params = {"R0": dict(base="R0", params={}, own_params={}, change="none: legacy settings (the M1 oracle)")}, {}
    for name, change, what in LADDER:
        params = params | change
        out[name] = dict(base="R3", params=dict(params), own_params=dict(change), change=what)
    for name, guard in GUARDS.items():          # guard run: R0's cylinder and trials with only that rung's own change
        own = out[name]["own_params"]
        out[guard["r0_rung"]] = dict(base="R0", params=dict(own), own_params=dict(own),
                                     change=f"{name} guard: R0 (M1 cylinder, 0.15 kg, B3's R0 trials) with {name}'s own change {own} only")
    return out


RUNGS = rungs()


def legacy_defaults() -> dict:
    """PickPlace's fix-ladder parameters and their legacy defaults, read from its signature (so they are never restated here)."""
    return {k: p.default for k, p in inspect.signature(PickPlace).parameters.items() if k not in ("policy", "cup_model", "cup_mass")}


def make_task(base: str, params: dict) -> ContactPickPlace:
    cfg = RUNS[base]
    return ContactPickPlace(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), cfg["cup_model"], cfg["mass_kg"], **params)


def fixed_trials(task: PickPlace) -> list[dict]:
    """B3's trials 0-19, built with the LEGACY grasp whatever grasp ``task`` uses (see the module docstring)."""
    return trials(Grasp.hero_side_grasp(task.scene.cup_height))[:N_TRIALS]


def layout_of(t: dict, use_handle_yaw: bool) -> dict:
    """A trial's layout as B3's rows store it (xy rounded to 0.1 mm, yaw to 0.01 deg)."""
    return dict(trial=t["trial"], layout_seed=t["layout_seed"], layout_index=t["layout_index"],
                cup_xy=[round(float(v), 4) for v in t["cup_xy"]], coaster_xy=[round(float(v), 4) for v in t["coaster_xy"]],
                cup_yaw_deg=round(t["handle_yaw_deg"], 2) if use_handle_yaw else 0.0)


def b3_rows(run: str) -> list[dict]:
    return json.loads((B3_RUNS / f"{run}.json").read_text())["runs"]


def grasp_info(task: PickPlace) -> dict:
    g, c = task.grasp, task.cup_in_palm
    return dict(grasp_offset=task.grasp_offset, cup_radius_m=task.cup_radius, cup_in_palm_frame_m=[float(v) for v in c],
                palm_point_to_surface_along_normal_m=round(float(c[1]) - task.cup_radius, 6),
                approach_dir_in_palm_frame_deg=round(math.degrees(math.atan2(c[1], c[0])), 2),
                palm_offset_m=g.palm_offset.round(4).tolist(), approach_dir_world=g.approach_dir.round(4).tolist(),
                palm_quat_wxyz=g.quat_wxyz.round(4).tolist(), closing_targets=task.closing_targets,
                closing_targets_source=str(L2_TARGETS.relative_to(REPO)) if task.closing_targets == "ik_mug" else
                "HERO's closed Dex3 pose (sim2sim.interactive_scene HAND_CLOSED)",
                right_hand_closed_pose_rad={f: float(v) for f, v in zip(FINGER_SUFFIXES, task.scene.hand_closed["right"])})


def contact_info(scene) -> dict:
    """The contact settings as MuJoCo compiled them, so a run file records what the physics actually used."""
    m = scene.model
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
    return dict(contact_model=scene.contact_model, cone=mujoco.mjtCone(int(m.opt.cone)).name, impratio=float(m.opt.impratio),
                noslip_iterations=int(m.opt.noslip_iterations), solver=mujoco.mjtSolver(int(m.opt.solver)).name,
                solver_tolerance=float(m.opt.tolerance),
                condim_right_hand_geoms={body(g): int(m.geom_condim[g]) for g in sorted(hand_parts(scene))},
                condim_cup_geoms=sorted({int(m.geom_condim[g]) for g in range(m.ngeom) if m.geom_bodyid[g] == scene.cup_body}),
                condim_explicit_pairs=sorted(set(m.pair_dim.tolist())))


def peak_rss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)        # ru_maxrss is in kB on Linux


# ---------------------------------------------------------------------------------------------------------------- run
def run_trials(rung: str, trial_ids: list[int]) -> dict:
    """One worker process: build the rung's task, check its trials are B3's, run them. Also returns its wall time and peak RSS."""
    t0 = time.perf_counter()
    spec = RUNGS[rung]
    use_yaw = RUNS[spec["base"]]["use_handle_yaw"]
    task = make_task(spec["base"], spec["params"])
    all_trials, b3 = fixed_trials(task), b3_rows(spec["base"])
    rows = []
    for k in trial_ids:
        mine, theirs = layout_of(all_trials[k], use_yaw), {f: b3[k][f] for f in LAYOUT_KEYS}
        assert mine == theirs, f"trial {k}: layout {mine} is not B3's {theirs}"
        rows.append(trial_row(task, all_trials[k], use_yaw))
        r = rows[-1]
        print(f"{rung} trial {k:2d} -> " + ("SUCCESS" if r["success"] else f"FAIL {r['failed_phase']}: {r['reason']}")
              + f"  lift contacts {r['lift_frame_contacts']}  ({r['wall_s']} s)", flush=True)
    return dict(trials=trial_ids, rows=rows, wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(),
                grasp=grasp_info(task), contact=contact_info(task.scene), mass_properties=mass_properties(task.scene),
                cup_height_m=task.scene.cup_height)


def same_as_b3(rows: list[dict], b3: list[dict]) -> dict:
    """Every field of every row against B3's, including everything under "detail" (plans, settles, final cup pose, sim time);
    only the wall time is left out."""
    diffs = []
    for r, o in zip(rows, b3):
        mine = json.loads(json.dumps(r))          # as stored (tuples become lists)
        fields = [k for k in sorted(mine.keys() | o.keys()) if k not in ("wall_s", "detail") and mine.get(k) != o.get(k)]
        fields += [f"detail.{k}" for k in sorted(mine["detail"].keys() | o["detail"].keys()) if mine["detail"].get(k) != o["detail"].get(k)]
        if fields:
            diffs.append(dict(trial=r["trial"], fields=fields))
    return dict(reference=str(B3_RUNS / "R0.json"), fields_compared="every row field except wall_s, including all of detail",
                trials=len(rows), trials_identical=len(rows) - len(diffs), successes=sum(r["success"] for r in rows),
                b3_successes=sum(o["success"] for o in b3[:len(rows)]), differences=diffs)


def cmd_run(args):
    spec = RUNGS[args.rung]
    cfg = RUNS[spec["base"]]
    if args.tag and args.rung != "R0":
        raise SystemExit("--tag is only for re-running the legacy check R0")
    path = args.out / "runs" / (f"{args.rung}_{args.tag}.json" if args.tag else f"{args.rung}.json")
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    order = [name for name, *_ in LADDER]
    stopped = [e["rung"] for e in ladder_entries(args.out) if e["meets_stop_rule"]]
    if args.rung in order and stopped and order.index(args.rung) > order.index(stopped[0]):
        raise SystemExit(f"the ladder stopped at {stopped[0]} (>= {STOP_SUCCESSES}/{N_TRIALS}); later rungs are not run (spec)")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    chunks = [c.tolist() for c in np.array_split(np.arange(args.trials), N_PROCESSES)]
    # fork: the parent has built nothing yet (no scene, no ONNX session), so every worker starts clean. HERO's policy runs one
    # onnxruntime thread (intra_op_num_threads 1), so 4 workers keep 4 cores busy.
    with ProcessPoolExecutor(N_PROCESSES, mp_context=multiprocessing.get_context("fork")) as pool:
        parts = list(pool.map(run_trials, [args.rung] * len(chunks), chunks))
    rows = [r for p in parts for r in p["rows"]]
    assert [r["trial"] for r in rows] == list(range(args.trials))
    assert all(p["grasp"] == parts[0]["grasp"] and p["contact"] == parts[0]["contact"] for p in parts)
    results = dict(
        rung=args.rung, tag=args.tag, base_run=spec["base"], object=cfg["object"], cup_model=cfg["cup_model"], mass_kg=cfg["mass_kg"],
        cup_yaw="each trial's handle yaw (= B3 R3)" if cfg["use_handle_yaw"] else "0 deg (round object, as in M1 and B3 R0)",
        params=spec["params"], effective_params=legacy_defaults() | spec["params"], legacy_defaults=legacy_defaults(),
        change=spec["change"], ladder_order=[name for name, *_ in LADDER],
        trials=f"trials 0-{args.trials - 1} of sim.m1_pick_mug.trials() built with the legacy grasp (B3's layouts and handle yaws, "
               f"asserted per trial against {B3_RUNS / (spec['base'] + '.json')}), whatever grasp this rung uses",
        processes=f"{N_PROCESSES} parallel CPU processes of consecutive trials: {chunks}",
        grasp=parts[0]["grasp"], contact=parts[0]["contact"], mass_properties=parts[0]["mass_properties"],
        cup_height_m=parts[0]["cup_height_m"],
        summary=run_summary(rows))
    if args.rung == "R0":
        results["legacy_check"] = dict(vs_b3_r0=same_as_b3(rows, b3_rows("R0")), vs_m1=r0_vs_m1(rows))
    results["resources"] = dict(wall_s=round(time.perf_counter() - t0, 1), parent_peak_rss_mb=peak_rss_mb(),
                                per_process=[dict(trials=p["trials"], wall_s=p["wall_s"], peak_rss_mb=p["peak_rss_mb"]) for p in parts],
                                peak_rss_mb_sum_of_processes=sum(p["peak_rss_mb"] for p in parts), gpu="none (CPU only, onnxruntime CPU)")
    results["runs"] = rows
    path.write_text(json.dumps(results, indent=2))
    s = results["summary"]
    print(f"{args.rung}: {s['successes']}/{s['trials']}, Wilson 95 % CI {s['wilson95']}, failures {s['failures_by_category']}; "
          f"lift slip mm {s['lift_slip_in_hand_mm']}; rise cm {s['lift_cup_rise_cm']}; grasp contact at lift "
          f"{s['lift_frame']['grasp_contact']}/{s['trials']}")
    if "legacy_check" in results:
        print(f"R0 vs B3 R0: {results['legacy_check']['vs_b3_r0']['trials_identical']}/{len(rows)} identical; "
              f"vs M1: {results['legacy_check']['vs_m1']['trials_identical']}/{len(rows)}")
    print(f"{results['resources']}; wrote {path}")
    if args.trials == N_TRIALS:                    # a shorter (smoke) run is not a ladder result
        write_ladder(args.out)


# ---------------------------------------------------------------------------------------------------------------- ladder.json
def metrics(s: dict) -> dict:
    """The spec's per-rung numbers from a run_summary (sim/m1_grasp_causes.py) dict."""
    slip, rise, lf = s["lift_slip_in_hand_mm"] or {}, s["lift_cup_rise_cm"] or {}, s["lift_frame"]
    return dict(success=f"{s['successes']}/{s['trials']}", successes=s["successes"], trials=s["trials"], wilson95=s["wilson95"],
                failures_by_category=s["failures_by_category"],
                lift_slip_in_hand_mm=dict(mean=slip.get("mean"), median=slip.get("median"), n=slip.get("n", 0),
                                          over_lift_slip_limit=slip.get("over_lift_slip_limit", 0)),
                lift_cup_rise_cm=dict(mean=rise.get("mean"), median=rise.get("median"), n=rise.get("n", 0)),
                contact_fraction_at_lift=dict(grasp_contact=lf["grasp_contact"], of=s["trials"], fraction=lf["fraction_of_all"],
                                              part_touching_counts=lf["part_touching_counts"], reached_lift=lf["reached"]))


def ladder_entries(out: Path) -> list[dict]:
    """The rungs run so far, in ladder order, each read from its runs file. Ends at the first rung without a full 20-trial
    run (rungs are cumulative, so a later rung only means something after the earlier ones) and after the first rung that
    meets the stop rule."""
    entries = []
    for name, *_ in LADDER:
        path = out / "runs" / f"{name}.json"
        if not path.exists():
            break
        res = json.loads(path.read_text())
        assert res["rung"] == name and res["params"] == RUNGS[name]["params"], f"{path} is not rung {name} as LADDER defines it"
        if res["summary"]["trials"] != N_TRIALS:
            break
        guards = guard_results(out, name) if name in GUARDS else None
        entries.append(dict(rung=name, change=RUNGS[name]["change"], own_params=RUNGS[name]["own_params"], params=res["params"],
                            effective_params=res["effective_params"], base_run=res["base_run"], object=res["object"],
                            mass_kg=res["mass_kg"], **metrics(res["summary"]),
                            meets_stop_rule=res["summary"]["successes"] >= STOP_SUCCESSES,
                            guards=guards, accepted=guards["accepted"] if guards else None,
                            grasp=res["grasp"], contact=res.get("contact"),       # L1 / L2 runs predate the contact record
                            **({"contact_model_sources": L3_SOURCES, "contact_model_values": L3_VALUE_CHOICE} if name == "L3" else {}),
                            run_file=str(path), resources=run_resources(res)))
        if entries[-1]["meets_stop_rule"]:
            break
    return entries


def run_resources(res: dict) -> dict:
    r = res["resources"]
    return dict(wall_s=r["wall_s"], peak_rss_mb_sum_of_processes=r["peak_rss_mb_sum_of_processes"],
                peak_rss_mb_max_process=max(p["peak_rss_mb"] for p in r["per_process"]), gpu=r["gpu"])


def guard_results(out: Path, rung: str) -> dict:
    """A rung's guards (GUARDS), read from their saved results. accepted = True when both passed, False as soon as one failed,
    None while one has not been run."""
    g = GUARDS[rung]
    path = out / "runs" / f"{g['r0_rung']}.json"
    r0 = None
    if path.exists():
        res = json.loads(path.read_text())
        assert res["rung"] == g["r0_rung"] and res["params"] == RUNGS[g["r0_rung"]]["params"], f"{path} is not {g['r0_rung']}"
        s = res["summary"]
        r0 = dict(run=g["r0_rung"], change=RUNGS[g["r0_rung"]]["change"], run_file=str(path), params=res["params"],
                  effective_params=res["effective_params"], contact=res["contact"], **metrics(s),
                  required=f"{N_TRIALS}/{N_TRIALS}", passed=s["trials"] == N_TRIALS and s["successes"] == N_TRIALS,
                  resources=run_resources(res))
    reach = reach_guard(g)
    done = [x["passed"] for x in (reach, r0) if x is not None]
    accepted = False if not all(done) else True if len(done) == 2 else None
    return dict(rule=f"spec: with {rung}'s contact model, m1_reach_test at {g['reach_height']} m must show no fall and R0 must stay "
                     f"{N_TRIALS}/{N_TRIALS}; otherwise {rung} is rejected (its R3 numbers are still recorded)",
                reach_test=reach, r0=r0, accepted=accepted)


def height_entry(res: dict, height: float) -> dict:
    return next(e for e in res["heights"] if abs(e["table_height_m"] - height) < 1e-9)


REACH_KEYS = ("fell", "pelvis_z_min_m", "pelvis_tilt_max_deg", "body_part_touched_floor_or_table", "root_xy_drift_max_m",
              "palm_err_cm_over_goals", "cup_displacement_max_cm")


def reach_guard(g: dict) -> dict | None:
    """The reach test at the guard's height with the rung's contact model (passed = no fall), with M1's numbers beside it, and
    the check that the same test at its legacy default still gives M1's (None while the guard run is missing)."""
    path = g["reach_dir"] / "reach_results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    assert res["protocol"]["contact_model"] == g["contact_model"], f"{path}: not run with contact_model {g['contact_model']}"
    h = height_entry(res, g["reach_height"])
    m1 = height_entry(json.loads((M1_REACH / "reach_results.json").read_text()), g["reach_height"])
    return dict(results_file=str(path), table_height_m=g["reach_height"], contact_model=res["protocol"]["contact_model"],
                fall_rule=res["protocol"]["fall_rule"], required="no fall", passed=not h["fell"], **{k: h[k] for k in REACH_KEYS},
                m1_same_height={k: m1[k] for k in REACH_KEYS}, wall_time_s=h["wall_time_s"],
                peak_rss_mb=res["resources"]["peak_rss_mb"], legacy_check=legacy_reach_check(g))


def without_clock(entry: dict) -> dict:
    """A reach-test height summary without its wall-clock times (they vary from run to run; the simulation does not)."""
    goals = [g | dict(initial_plan={k: v for k, v in g["initial_plan"].items() if k != "ik_s"}) for g in entry["goals"]]
    return {k: v for k, v in entry.items() if k != "wall_time_s"} | dict(goals=goals)


def legacy_reach_check(g: dict) -> dict | None:
    """m1_reach_test at its legacy default vs M1's results/m1_reach at the same height: every summary field except the two
    wall-clock times (wall_time_s, and each goal's initial_plan.ik_s = how long the IK took), and every per-step array of the
    series file."""
    path = g["legacy_reach_dir"] / "reach_results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    assert res["protocol"]["contact_model"] == "legacy", path
    mine, m1 = (without_clock(height_entry(r, g["reach_height"])) for r in (res, json.loads((M1_REACH / "reach_results.json").read_text())))
    fields = [k for k in sorted(mine.keys() | m1.keys()) if mine.get(k) != m1.get(k)]
    series = f"series_h{round(g['reach_height'] * 100):03d}.npz"
    a, b = np.load(g["legacy_reach_dir"] / series), np.load(M1_REACH / series)
    same = lambda x, y: x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")    # err_cm is NaN while standing
    arrays = [k for k in sorted(set(a.files) | set(b.files)) if k not in a.files or k not in b.files or not same(a[k], b[k])]
    return dict(results_file=str(path), reference=str(M1_REACH / "reach_results.json"), compared=f"every field of the "
                f"{g['reach_height']} m summary except the wall-clock times (wall_time_s, goals[*].initial_plan.ik_s), and every "
                f"array of {series}", fields_differing=fields,
                series_arrays_differing=arrays, identical=not fields and not arrays, peak_rss_mb=res["resources"]["peak_rss_mb"])


def legacy_entry(path: Path) -> dict:
    """A legacy-check run (R0 at legacy settings) for ladder.json: its numbers and whether every trial is still M1's."""
    r0 = json.loads(path.read_text())
    chk = r0["legacy_check"]
    return dict(run_file=str(path), tag=r0.get("tag"), params=r0["params"], effective_params=r0["effective_params"],
                **metrics(r0["summary"]),
                trials_identical_to_b3_r0=f"{chk['vs_b3_r0']['trials_identical']}/{chk['vs_b3_r0']['trials']}",
                trials_identical_to_m1=f"{chk['vs_m1']['trials_identical']}/{len(r0['runs'])}",
                required=f"{N_TRIALS}/{N_TRIALS}, every trial identical to M1",
                passed=r0["summary"]["successes"] == N_TRIALS and chk["vs_m1"]["trials_identical"] == N_TRIALS
                and chk["vs_b3_r0"]["trials_identical"] == N_TRIALS)


def write_ladder(out: Path):
    """ladder.json from the saved runs files (see the module docstring)."""
    t0 = time.perf_counter()
    entries = ladder_entries(out)
    stopped_at = next((e["rung"] for e in entries if e["meets_stop_rule"] and e["accepted"] is not False), None)
    rejected = [e["rung"] for e in entries if e["accepted"] is False]
    remaining = [n for n in SPEC_RUNGS if n not in {e["rung"] for e in entries}]
    r0_path = out / "runs" / "R0.json"
    legacy = legacy_entry(r0_path) if r0_path.exists() else None
    rechecks = [legacy_entry(p) for p in sorted((out / "runs").glob("R0_*.json"))]
    b3_r3 = json.loads(B3_SUMMARY.read_text())["summary"]["R3"]
    out_json = dict(
        question=f"Track B v3 fix ladder: which cumulative change first gets my mug (B3's R3: full mug from config, 0.40 kg) to "
                 f">= {STOP_SUCCESSES}/{N_TRIALS} on B3's {N_TRIALS} trials?",
        protocol=dict(
            object=f"B3's R3: {RUNS['R3']['object']}, {RUNS['R3']['mass_kg']} kg; each trial's handle yaw (= B3 R3)",
            trials=f"trials 0-{N_TRIALS - 1} of sim.m1_pick_mug.trials() built with the legacy grasp (B3's layouts and handle yaws, "
                   "asserted per trial), the same for every rung",
            cumulative="each rung = the previous rung's named PickPlace parameters + its own change; every parameter defaults to its "
                       "legacy (M1) value",
            spec_order=list(SPEC_RUNGS), stop_rule=f"stop at the first rung with >= {STOP_SUCCESSES}/{N_TRIALS}",
            constraints="no friction coefficient, PD gain, torque limit, mug geometry or mass is changed: a rung only sets named "
                        "PickPlace parameters (effective_params)",
            success="cup tilt < 10 deg, cup centre within coaster radius (5 cm), resting on the coaster (|dz| < 4 mm), robot never "
                    "fell (sim/pick_place.py, unchanged); Wilson score interval, z = 1.96",
            lift_slip=f"PickPlace lift_slip_in_hand_cm x 10, over the trials that reached the end of the lift (n); "
                      f"over_lift_slip_limit = trials above {LIFT_SLIP_FAIL_M * 1e3:g} mm",
            lift_cup_rise="PickPlace lift_cup_rise_cm (cup rise over the 10 cm palm lift), over the trials that reached the end of the lift",
            contact_fraction_at_lift=f"B3's definition: lift frame = {LIFT_FRAME}; {CONTACT_RULE}; denominator {N_TRIALS} (a trial "
                                     "that never reached the lift counts as no contact)",
            accepted="only rungs with guards in the spec (L3) can be accepted or rejected; null = no guard defined for that rung, or "
                     "a guard not run yet; the final evaluation needs a rung that meets the stop rule and was not rejected"),
        legacy_r0=legacy,
        legacy_r0_rechecks=rechecks,
        before_ladder=dict(rung="B3 R3 (mug, no change)", source=f"{B3_SUMMARY} summary.R3", params={}, **metrics(b3_r3)),
        rungs=entries, stopped_at=stopped_at, rejected=rejected,
        next=(f"final evaluation at {stopped_at}" if stopped_at else
              f"next rung: {remaining[0]}" if remaining else
              f"no accepted rung reached >= {STOP_SUCCESSES}/{N_TRIALS}" + (f" (rejected: {', '.join(rejected)})" if rejected else "")),
        runtime_s=round(time.perf_counter() - t0, 2), peak_rss_mb=peak_rss_mb())
    (out / "ladder.json").write_text(json.dumps(out_json, indent=2))
    print("ladder: " + "; ".join(f"{e['rung']} {e['success']} (Wilson {e['wilson95']}, accepted {e['accepted']})" for e in entries)
          + f"; stopped at {stopped_at}; next: {out_json['next']}; wrote {out / 'ladder.json'}")


def cmd_ladder(args):
    write_ladder(args.out)


# ---------------------------------------------------------------------------------------------------------------- L1 check
def quat_to_mat(q_wxyz) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=float))
    return R.reshape(3, 3)


def planned_grasp_distances(task: PickPlace, cup_yaw: float) -> dict:
    """Signed distances between the right hand (open) and the object placed where the task's planned grasp puts it.

    Kinematics only. The robot stays in its spawn pose with both hands at HERO's open targets, and the object goes where the
    plan has it RELATIVE TO THE PALM: run() plans palm = cup centre + palm_offset with the palm orientation R_plan, so in the
    palm frame the cup centre is R_plan^T (-palm_offset) and the cup's yaw is taken relative to R_plan (palm z = world z in
    the side grasp). Same relative geometry as at the planned pose; the arm plays no role. Palm = HERO's tracked palm
    (read_state), the frame GRASP_OFFSET is written in.
    """
    s, m, d = task.scene, task.scene.model, task.scene.data
    s.reset(cup_yaw=cup_yaw)                       # spawn pose, both hands open
    st = read_state(s.plant)
    p_palm, R_palm = st.palm_pos_w[1], quat_to_mat(xyzw_to_wxyz(st.palm_quat_w[1]))
    palm_link = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hand_palm_link")
    assert np.allclose(d.xpos[palm_link], p_palm, atol=1e-9) and np.allclose(d.xmat[palm_link].reshape(3, 3), R_palm, atol=1e-9)
    R_plan = quat_to_mat(task.grasp.quat_wxyz)
    c, sn = math.cos(cup_yaw), math.sin(cup_yaw)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (R_palm @ R_plan.T @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])).flatten())
    a = s.cup_qadr
    d.qpos[a:a + 3] = p_palm + R_palm @ (R_plan.T @ -task.grasp.palm_offset)
    d.qpos[a + 3:a + 7] = quat
    mujoco.mj_kinematics(m, d)

    geom = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
    hand = sorted(hand_parts(s))                   # the 8 right-hand collision geoms (palm, thumb 0-2, index 0-1, middle 0-1)
    obj = [g for g in range(m.ngeom) if m.geom_bodyid[g] == s.cup_body]
    assert geom(obj[0]) == "cup_geom"              # the body cylinder first, then the mug's handle capsules
    dist = {(h, o): mujoco.mj_geomDistance(m, d, h, o, DISTMAX, None) for h in hand for o in obj}
    (h_min, o_min), d_min = min(dist.items(), key=lambda kv: kv[1])
    per_hand = {}
    for h in hand:
        o = min(obj, key=lambda o: dist[h, o])
        per_hand[body(h)] = dict(distance_m=dist[h, o], object_geom=geom(o), to_body_cylinder_m=dist[h, obj[0]])
    handle = [dist[h, o] for h in hand for o in obj[1:]]
    return dict(cup_yaw_deg=round(math.degrees(cup_yaw), 2), min_m=d_min,
                min_pair=dict(hand_geom=geom(h_min), hand_body=body(h_min), object_geom=geom(o_min)),
                handle_min_m=min(handle) if handle else None, per_hand_geom=per_hand)


def mm(v) -> float | None:
    return None if v is None else round(v * 1e3, 3)


def report(p: dict) -> dict:
    """``planned_grasp_distances`` in mm, for the JSON."""
    return dict(cup_yaw_deg=p["cup_yaw_deg"], min_signed_distance_mm=mm(p["min_m"]), min_pair=p["min_pair"],
                handle_min_distance_mm=mm(p["handle_min_m"]),
                per_hand_geom_mm={b: dict(distance_mm=mm(v["distance_m"]), object_geom=v["object_geom"],
                                          to_body_cylinder_mm=mm(v["to_body_cylinder_m"])) for b, v in p["per_hand_geom"].items()})


def cmd_check(args):
    t0 = time.perf_counter()
    # (a) the M1 cylinder, legacy offset (= R0); (b) the mug at L1 (= R3 + L1), at each of the 20 trials' handle yaws
    task_a, task_b = make_task("R0", RUNGS["R0"]["params"]), make_task("R3", RUNGS["L1"]["params"])
    a = planned_grasp_distances(task_a, 0.0)
    per_yaw = [planned_grasp_distances(task_b, math.radians(t["handle_yaw_deg"])) for t in fixed_trials(task_b)]
    k_min = int(np.argmin([p["min_m"] for p in per_yaw]))
    b = per_yaw[k_min]
    diff = b["min_m"] - a["min_m"]
    change = {h: dict(a_mm=mm(a["per_hand_geom"][h]["distance_m"]), b_mm=mm(v["distance_m"]),
                      b_minus_a_mm=mm(v["distance_m"] - a["per_hand_geom"][h]["distance_m"])) for h, v in b["per_hand_geom"].items()}
    largest = max(change, key=lambda h: abs(change[h]["b_minus_a_mm"]))
    # L1 must keep B3's trials: the trials it runs (legacy grasp) vs B3 R3's rows, and for contrast those L1's own grasp would give
    b3_r3 = b3_rows("R3")
    n_same = lambda tl: sum(layout_of(t, True) == {f: b3_r3[k][f] for f in LAYOUT_KEYS} for k, t in enumerate(tl))
    trials_check = dict(l1_trials_equal_b3_r3=f"{n_same(fixed_trials(task_b))}/{N_TRIALS}",
                        trials_from_l1_own_grasp_equal_b3_r3=f"{n_same(trials(task_b.grasp)[:N_TRIALS])}/{N_TRIALS}",
                        compared=f"{', '.join(LAYOUT_KEYS)} (as B3 stores them: xy to 0.1 mm, yaw to 0.01 deg)",
                        note="the second line is not used anywhere: it shows that building the trials from L1's grasp would "
                             "re-sample the layouts and yaws")

    # Cross-check of this placement code (information): B1 placed the cup independently (sim/grasp_fix_b1.py, legacy offset,
    # mug handle along the approach direction) and measured the body cylinder; it rounds to 0.01 mm.
    b1 = json.loads(Path("results/m1_grasp_fix/b1_model.json").read_text())["cups"]
    task_mug_legacy = make_task("R3", RUNGS["R0"]["params"])
    yaw_b1 = math.atan2(task_mug_legacy.grasp.approach_dir[1], task_mug_legacy.grasp.approach_dir[0])
    cross = {}
    for name, p in (("cylinder", a), ("mug", planned_grasp_distances(task_mug_legacy, yaw_b1))):
        mine = {h: round(v["to_body_cylinder_m"] * 1e3, 2) for h, v in p["per_hand_geom"].items()}
        ref = {h: v for h, v in b1[name]["nominal_grasp"]["clearance_open_mm"].items() if h in mine}
        cross[f"{name}_legacy"] = dict(this_check_mm=mine, b1_mm=ref, max_abs_diff_mm=round(max(abs(mine[h] - ref[h]) for h in mine), 3))

    out = dict(
        question="L1 check: at the planned grasp pose with the hand open, the minimum signed distance between the right-hand "
                 "geoms and the object, (a) M1 cylinder at legacy settings vs (b) the mug at L1; expected to match within 0.5 mm",
        method=dict(
            placement="kinematics only (mj_kinematics, nothing simulated): robot in its spawn pose, the object placed where run() "
                      "plans it relative to the palm: palm = cup centre + Grasp.palm_offset with palm orientation R_plan, so the cup "
                      "centre is R_plan^T (-palm_offset) in the palm frame (HERO's tracked palm = right_hand_palm_link, asserted) and "
                      "its yaw is taken relative to R_plan (palm z = world z in the side grasp)",
            distance="mujoco.mj_geomDistance (signed: negative = overlap; meshes as their convex hulls), every right-hand collision "
                     "geom (palm, thumb_0/1/2, index_0/1, middle_0/1) x every collision geom of the cup body (cylinder: 1; mug: body "
                     f"cylinder + 3 handle capsules); distmax {DISTMAX} m",
            hand="HERO's open Dex3 targets (TableScene.reset): thumb_1 0.7 rad, all other right-hand joints 0",
            a="TableScene cup_model 'cylinder' (r 0.040, h 0.100 m, 0.15 kg) = B3 R0; PickPlace grasp_offset 'legacy'; yaw 0",
            b="TableScene cup_model 'mug' (config: r 0.0495, h 0.091 m, 0.40 kg) = B3 R3; PickPlace grasp_offset 'radius'; at each of "
              "the 20 trials' handle yaws (B3 R3's), reported: the yaw with the smallest minimum",
            tolerance_mm=CHECK_TOLERANCE_M * 1e3),
        radius_consistent_offset=dict(m1_cylinder=radius_consistent_offset(CUP_RADIUS).tolist(), legacy_GRASP_OFFSET=GRASP_OFFSET.tolist(),
                                      identical=bool(np.array_equal(radius_consistent_offset(CUP_RADIUS), GRASP_OFFSET))),
        a_cylinder_legacy=dict(grasp=grasp_info(task_a), **report(a)),
        b_mug_L1=dict(grasp=grasp_info(task_b), reported_at_trial_yaw=k_min, **report(b),
                      min_over_trial_yaws_mm=dict(min=mm(min(p["min_m"] for p in per_yaw)), max=mm(max(p["min_m"] for p in per_yaw))),
                      same_minimum_pair_at_every_trial_yaw=all(p["min_pair"] == b["min_pair"] for p in per_yaw),
                      handle_min_distance_over_trial_yaws_mm=mm(min(p["handle_min_m"] for p in per_yaw))),
        b_minus_a_mm=mm(diff), match=bool(abs(diff) <= CHECK_TOLERANCE_M),
        per_hand_geom_change=change,
        largest_per_geom_change=dict(hand_body=largest, **change[largest]),
        same_minimum_geom_pair=a["min_pair"] == b["min_pair"],
        trials_check=trials_check,
        cross_check_b1=dict(note="information: this placement code vs B1's independent placement (sim/grasp_fix_b1.py, b1_model.json "
                                 "nominal_grasp.clearance_open_mm, body cylinder only, rounded to 0.01 mm), legacy offset", **cross),
        mujoco=mujoco.__version__, runtime_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), gpu="none (CPU only)")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "l1_check.json").write_text(json.dumps(out, indent=2))
    print(f"(a) cylinder legacy: min {out['a_cylinder_legacy']['min_signed_distance_mm']} mm at {a['min_pair']}")
    print(f"(b) mug L1: min {out['b_mug_L1']['min_signed_distance_mm']} mm at {b['min_pair']} (trial {k_min} yaw)")
    print(f"b - a = {out['b_minus_a_mm']} mm -> match within 0.5 mm: {out['match']}; largest per-geom change: {out['largest_per_geom_change']}")
    print(f"B1 cross-check max |diff| (mm): " + ", ".join(f"{k} {v['max_abs_diff_mm']}" for k, v in cross.items()))
    print(f"wrote {args.out / 'l1_check.json'} ({out['runtime_s']} s, peak RSS {out['peak_rss_mb']} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one rung on trials 0..N-1")
    run.add_argument("--rung", choices=tuple(RUNGS), required=True)
    run.add_argument("--trials", type=int, default=N_TRIALS)
    run.add_argument("--overwrite", action="store_true", help="replace an existing runs file")
    run.add_argument("--tag", default=None, help="R0 only: write runs/R0_<tag>.json (a legacy re-check after a code change)")
    for p in (run, sub.add_parser("check", help="the L1 geometry check"),
              sub.add_parser("ladder", help="rebuild ladder.json from the saved runs files (no simulation)")):
        p.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    dict(run=cmd_run, check=cmd_check, ladder=cmd_ladder)[args.cmd](args)


if __name__ == "__main__":
    main()
