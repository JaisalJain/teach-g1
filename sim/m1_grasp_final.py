"""Track B final: L1 + L3 with HERO's finger closing (grasp profile "mug_v1"), one pre-registered test on B3's R3.

Why: the v3 fix ladder (sim/m1_grasp_ladder.py) was cumulative, so L3 (the contact model) was only tested on top of L2 (IK
closing targets), and L2 had made the grip weaker (lift slip 47 mm against L1's 22 mm). This run tests L1 + L3 with HERO's
closing targets: sim/pick_place.py GRASP_PROFILES["mug_v1"] = grasp_offset "radius" (L1) + closing_targets "hero" (M1's) +
contact_model "grasp" (L3). It is one test, not a search: nothing here is tuned, and failures are results.

Steps, all with mug_v1, each one runs file under results/m1_grasp_fix/v3_final/runs/:
  run1       B3's R3 (full mug from config, 0.40 kg): trials 0-19 of sim.m1_pick_mug.trials() built with the LEGACY grasp,
             R3's handle yaws, layouts asserted against B3's R3 rows (the ladder's worker does all three)
  r0_mug_v1  check: B3's R0 (M1 cylinder, 0.15 kg, yaw 0), its 20 trials; must be 20/20. On the 80 mm cylinder mug_v1 is the
             same physics as the ladder's L3_R0 guard (radius_consistent_offset(0.040) == GRASP_OFFSET exactly, HERO's targets,
             contact model "grasp"), so summarize also compares every row with L3_R0's: equal rows show that the force
             recording below changed nothing.
  fifty      only if Run 1 >= 18/20 (``run`` refuses otherwise): trials 0-49 of trials() (legacy-grasp layouts, the same
             handle-yaw draw R3 used), full mug, 0.40 kg; layouts asserted against results/m1_pick_mug/runs_mass0.37.json,
             whose first 20 are B3 R3's (asserted too)
Not in this script: the legacy check, ``python -m sim.m1_grasp_ladder run --rung R0 --tag after_final --out
results/m1_grasp_fix/v3_final`` (every trial compared with B3's R0 and with M1). The reach test is not rerun: mug_v1's global
contact options are exactly L3's, which passed it (results/m1_grasp_fix/v3/reach_l3/); summarize compares the compiled
contact settings of these runs with the L3 run's.

Instrumentation (read-only; same behaviour): ``ForcePickPlace`` records, per trial, at two moments
  closed  the control step in which the right hand's closure first reaches 1.0 (the first step with the squeeze rule on)
  lift    B3's lift frame (sim/m1_grasp_causes.py LIFT_FRAME)
the state after that 20 ms step, i.e. its last 1 ms substep (as B3's contact rule):
  - the normal contact force on the cup per hand part (thumb, index, middle, palm; B3's PARTS / hand_parts): the sum of
    mj_contactForce's component 0 (the normal force, contact frame) over the contacts between that part's geoms and any cup geom;
  - the 7 right finger-joint torques applied in that substep (TableScene.last_finger_tau) and, per joint, whether
    |tau| >= 0.99 x its effort limit (the limit TableScene.step clips to);
  - the 3 right fingertip touch sensors.
mj_step computes the contacts, their forces and the touch sensors in one mj_forward before it integrates, and the torque is
the one applied in that substep, so the record describes one instant.

Workers: the ladder's (m1_grasp_ladder.run_parallel / run_trials: 4 processes, B3's trial rows, layout asserts) with
ForcePickPlace as the task; nothing of that machinery is repeated here.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.m1_grasp_final ...):
  run --step run1 | r0_mug_v1 | fifty -> runs/<step>.json (refuses to replace one unless --overwrite)
  videos     after fifty: re-runs one success + one failure of it (two successes if nothing failed) with video
             (videos/<label>.mp4, gitignored) and checks the outcome equals the batch's; renders like sim/m1_pick_mug.py
             videos: MUJOCO_GL=egl and __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
  summarize  -> final.json + summary.md, read back from the saved files (nothing is simulated)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")       # before mujoco's first import, which picks the GL backend (only `videos` renders)

import mujoco
import numpy as np

from sim2sim.interactive_scene import FINGER_EFFORT, FINGER_SUFFIXES

from sim.m1_grasp_causes import LIFT_FRAME, PARTS, RUNS, ContactPickPlace, _stats, run_summary
from sim.m1_grasp_ladder import (GUARDS, LAYOUT_KEYS, N_TRIALS, STOP_SUCCESSES, b3_rows, fixed_trials, legacy_defaults, legacy_entry,
                                 make_task, metrics, peak_rss_mb, process_resources, reach_guard, run_parallel, same_rows)
from sim.m1_grasp_ladder import OUT_DIR as LADDER_DIR
from sim.m1_pick_mug import RESULT_KEYS, cylinder_runs, run_trial
from sim.pick_place import GRASP_PROFILES, LIFT_SLIP_FAIL_M
from sim.scene import FINGERTIP_LINKS, SIDES

OUT_DIR = Path("results/m1_grasp_fix/v3_final")
PROFILE_NAME = "mug_v1"
PROFILE = GRASP_PROFILES[PROFILE_NAME]
N_FIFTY = 50
STEPS = {
    "run1": dict(base="R3", n=N_TRIALS, what="Run 1: B3's R3 (full mug from config, 0.40 kg), trials 0-19 of trials() built with "
                                            "the legacy grasp, R3's handle yaws (asserted against B3's R3 rows)"),
    "r0_mug_v1": dict(base="R0", n=N_TRIALS, what="check: B3's R0 (M1 cylinder, 0.15 kg, yaw 0), its 20 trials (asserted against "
                                                 "B3's R0 rows); required 20/20"),
    "fifty": dict(base="R3", n=N_FIFTY, what="trials 0-49 of trials() built with the legacy grasp (the handle-yaw draw R3 used), full "
                                            "mug, 0.40 kg (asserted against results/m1_pick_mug/runs_mass0.37.json); only if Run 1 "
                                            f">= {STOP_SUCCESSES}/{N_TRIALS}"),
}
FIFTY_REFERENCE = Path("results/m1_pick_mug/runs_mass0.37.json")   # trials() 0-49 as M1's mug run stored them (read only)
LEGACY_RECHECK = "R0_after_final"            # written by sim.m1_grasp_ladder (see the module docstring)
L3_RUN, L3_R0_RUN = LADDER_DIR / "runs" / "L3.json", LADDER_DIR / "runs" / "L3_R0.json"   # v3 ladder (read only)

MOMENTS = dict(closed="the control step in which the right hand's closure first reaches 1.0 (close phase: set_hand_closure(1.0), "
                      "then that 20 ms step, the first one with the squeeze rule on); state after the step = its last 1 ms substep",
               lift=f"B3's lift frame: {LIFT_FRAME}")
SATURATION = 0.99                            # |tau| >= 0.99 x the effort limit counts as saturated
assert SIDES == ("left", "right")
RIGHT = slice(len(FINGER_SUFFIXES), 2 * len(FINGER_SUFFIXES))   # TableScene's finger order: the left hand's 7, then the right's
EFFORT = FINGER_EFFORT[RIGHT]                # the limits TableScene.step clips the finger torques to (HERO demo's Dex3 efforts)
RECORD = dict(
    normal_force_N="per hand part (thumb = thumb_0/1/2, index = index_0/1, middle = middle_0/1, palm = palm_link; B3's hand_parts), "
                   "sum of mujoco.mj_contactForce component 0 (normal force in the contact frame) over the contacts in data.contact "
                   "between that part's collision geoms and any geom of the cup body",
    finger_tau_Nm="the 7 right finger-joint torques applied in the step's last substep (TableScene.last_finger_tau, after the "
                  "squeeze rule = what was written to qfrc_applied), FINGER_SUFFIXES order",
    saturated=f"|tau| >= {SATURATION} x the joint's effort limit {dict(zip(FINGER_SUFFIXES, EFFORT.tolist()))} N m",
    touch_N="the right fingertip touch sensors (thumb_2, index_1, middle_1; sim/scene.py: normal force inside a box site around the "
            "distal link's collision geometry)",
    timing="data.contact, the contact forces and the touch sensors all come from the last substep's mj_forward (mj_step computes "
           "them before it integrates); the torque is the one applied in that substep")


class ForcePickPlace(ContactPickPlace):
    """ContactPickPlace (B3's lift-frame contacts) that also records contact forces, finger torques and touch at two moments
    (MOMENTS). It only reads the simulation, so the behaviour is ContactPickPlace's = PickPlace's."""

    def run(self, *args, **kwargs) -> dict:
        self.moments = dict.fromkeys(MOMENTS)            # None = the trial never reached that moment
        result = super().run(*args, **kwargs)
        result["grasp_forces"] = dict(self.moments)
        return result

    def _tick(self, phase: str):
        # decided before the step, like ContactPickPlace's lift test (which sets lift_contacts after it)
        closed = self.moments["closed"] is None and self.scene.hand_closure["right"] >= 1.0
        lift = phase == "lift" and self.lift_contacts is None
        try:
            super()._tick(phase)
        finally:          # also when that step raised (a fall): the step has been simulated
            if closed:
                self.moments["closed"] = self.grasp_forces()
            if lift:
                self.moments["lift"] = self.grasp_forces()

    def grasp_forces(self) -> dict:
        """The record at this instant (RECORD). Reads data only."""
        s, m, d = self.scene, self.scene.model, self.scene.data
        force, f6 = dict.fromkeys(PARTS, 0.0), np.zeros(6)
        for i, (a, b) in enumerate(zip(d.contact.geom1.tolist(), d.contact.geom2.tolist())):
            hand = b if a in self.cup_geoms else a if b in self.cup_geoms else None
            if hand in self.part_of_geom:
                mujoco.mj_contactForce(m, d, i, f6)
                force[self.part_of_geom[hand]] += float(f6[0])
        tau, touch = s.last_finger_tau[RIGHT], s.touch()
        return dict(normal_force_N=force, finger_tau_Nm={f: float(t) for f, t in zip(FINGER_SUFFIXES, tau)},
                    saturated={f: bool(abs(t) >= SATURATION * e) for f, t, e in zip(FINGER_SUFFIXES, tau, EFFORT)},
                    touch_N={f: touch[f"touch_right_{f}"] for f in FINGERTIP_LINKS})


def runs_path(out: Path, step: str) -> Path:
    return out / "runs" / f"{step}.json"


def reference_rows(step: str) -> list[dict] | None:
    """The layouts a step's trials are asserted against. None = the ladder's default, B3's rows of the base run (run1, r0);
    fifty: results/m1_pick_mug's 50 trials (handle_yaw_deg = the cup yaw), whose first 20 must be B3 R3's."""
    if step != "fifty":
        return None
    rows = [dict(r, cup_yaw_deg=r["handle_yaw_deg"]) for r in json.loads(FIFTY_REFERENCE.read_text())["runs"]]
    rows = [{f: r[f] for f in LAYOUT_KEYS} for r in rows]
    assert len(rows) == N_FIFTY and rows[:N_TRIALS] == [{f: r[f] for f in LAYOUT_KEYS} for r in b3_rows("R3")], FIFTY_REFERENCE
    return rows


def load_run(out: Path, step: str, n: int) -> dict | None:
    """A saved runs file, checked to hold trials 0..n-1 (None if it does not exist)."""
    path = runs_path(out, step)
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    if [r["trial"] for r in res["runs"]] != list(range(n)):
        raise SystemExit(f"{path}: expected trials 0-{n - 1}")
    return res


# ---------------------------------------------------------------------------------------------------------------- run
def cmd_run(args):
    step = STEPS[args.step]
    n = args.trials or step["n"]
    path = runs_path(args.out, args.step)
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    if args.step == "fifty":          # spec: the 50-trial evaluation only if Run 1 reached the bar
        run1 = load_run(args.out, "run1", N_TRIALS)
        if run1 is None or run1["summary"]["successes"] < STOP_SUCCESSES:
            raise SystemExit(f"fifty needs Run 1 >= {STOP_SUCCESSES}/{N_TRIALS} in {runs_path(args.out, 'run1')} "
                             f"({'missing' if run1 is None else run1['summary']['successes']})")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    chunks, parts, rows = run_parallel(step["base"], PROFILE, n, args.step, ForcePickPlace, reference_rows(args.step))
    for r in rows:           # trial_row files PickPlace.run's extra keys under "detail"; the force record goes to the top level
        r["grasp_forces"] = r["detail"].pop("grasp_forces")
    cfg = RUNS[step["base"]]
    results = dict(
        step=args.step, what=step["what"], profile=PROFILE_NAME, params=PROFILE, effective_params=legacy_defaults() | PROFILE,
        base_run=step["base"], object=cfg["object"], cup_model=cfg["cup_model"], mass_kg=cfg["mass_kg"],
        cup_yaw="each trial's handle yaw from trials() (= B3 R3 for trials 0-19)" if cfg["use_handle_yaw"] else
        "0 deg (round object, as in M1 and B3 R0)",
        processes=f"{len(chunks)} parallel CPU processes of consecutive trials (sim/m1_grasp_ladder.py run_parallel): {chunks}",
        grasp=parts[0]["grasp"], contact=parts[0]["contact"], mass_properties=parts[0]["mass_properties"],
        cup_height_m=parts[0]["cup_height_m"], moments=MOMENTS, record=RECORD,
        summary=run_summary(rows), forces=force_summary(rows), resources=process_resources(t0, parts), runs=rows)
    path.write_text(json.dumps(results, indent=2))
    s = results["summary"]
    print(f"{args.step}: {s['successes']}/{s['trials']}, Wilson 95 % CI {s['wilson95']}, failures {s['failures_by_category']}; "
          f"lift slip mm {s['lift_slip_in_hand_mm']}; rise cm {s['lift_cup_rise_cm']}; {results['resources']}; wrote {path}")


def force_summary(rows: list[dict]) -> dict:
    """Per moment, over the trials that reached it: normal force on the cup per hand part, each right finger joint's torque
    and in how many trials it was saturated, each right fingertip's touch sensor (n / mean / median / min / max, B3's _stats)."""
    out = {}
    for moment in MOMENTS:
        recs = [r["grasp_forces"][moment] for r in rows if r["grasp_forces"][moment] is not None]
        out[moment] = dict(trials_reached=len(recs),
                           normal_force_N={p: _stats([x["normal_force_N"][p] for x in recs]) for p in PARTS},
                           finger_tau_Nm={f: _stats([x["finger_tau_Nm"][f] for x in recs]) for f in FINGER_SUFFIXES},
                           saturated_trials={f: sum(x["saturated"][f] for x in recs) for f in FINGER_SUFFIXES},
                           touch_N={f: _stats([x["touch_N"][f] for x in recs]) for f in FINGERTIP_LINKS})
    return out


def failed_trials(rows: list[dict]) -> list[dict]:
    """The force / torque / touch record of every failed trial, at both moments (None = never reached)."""
    return [dict(trial=r["trial"], cup_xy=r["cup_xy"], cup_yaw_deg=r["cup_yaw_deg"], category=r["category"], failed_phase=r["failed_phase"],
                 reason=r["reason"], contact_closure=r["contact_closure"], **{m: r["grasp_forces"][m] for m in MOMENTS})
            for r in rows if not r["success"]]


# ---------------------------------------------------------------------------------------------------------------- videos
def gl_renderer() -> str:
    """The OpenGL vendor / renderer EGL gives MuJoCo here, so videos.json says which device rendered (libEGL warns while it
    skips the NVIDIA card, so the warning alone does not tell)."""
    from OpenGL import GL           # PyOpenGL: MuJoCo's own EGL backend depends on it
    ctx = mujoco.GLContext(16, 16)
    ctx.make_current()
    try:
        return f"{GL.glGetString(GL.GL_VENDOR).decode()} / {GL.glGetString(GL.GL_RENDERER).decode()} (EGL)"
    finally:
        ctx.free()


def cmd_videos(args):
    # This laptop's NVIDIA GPU is reserved for other jobs: EGL must load Mesa's driver (the Intel iGPU) instead of NVIDIA's.
    if not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        raise SystemExit("videos: set __EGL_VENDOR_LIBRARY_FILENAMES (here: /usr/share/glvnd/egl_vendor.d/50_mesa.json) so "
                         "rendering does not land on the NVIDIA GPU")
    t0 = time.perf_counter()
    renderer = gl_renderer()
    fifty = load_run(args.out, "fifty", N_FIFTY)
    if fifty is None:
        raise SystemExit(f"no {runs_path(args.out, 'fifty')}: the videos come from the 50-trial run")
    ok, bad = [r for r in fifty["runs"] if r["success"]], [r for r in fifty["runs"] if not r["success"]]
    picks = (ok[:1] + bad[:1]) if ok and bad else ok[:2] if ok else bad[:2]     # first in trial order
    (args.out / "videos").mkdir(parents=True, exist_ok=True)
    task = make_task("R3", PROFILE, ForcePickPlace)
    all_trials = fixed_trials(task, N_FIFTY)
    report = dict(rule="one success + one failure of the 50-trial run (the first of each in trial order), or two successes if "
                       "nothing failed; each re-run alone with video, and its outcome must equal the batch's", videos=[])
    for r in picks:
        label = f"{'success' if r['success'] else 'failure'}_trial{r['trial']:02d}"
        video = args.out / "videos" / f"{label}.mp4"
        again = run_trial(task, all_trials[r["trial"]], video)          # sim/m1_pick_mug.py's row, video on
        assert (again["cup_xy"], again["coaster_xy"], again["handle_yaw_deg"]) == (r["cup_xy"], r["coaster_xy"], r["cup_yaw_deg"])
        same_outcome = all(again[k] == r[k] for k in ("success", "failed_phase", "category"))
        identical = all(again[k] == r[k] for k in RESULT_KEYS) and again["detail"]["grasp_forces"] == r["grasp_forces"]
        report["videos"].append(dict(label=label, video=str(video), trial=r["trial"], batch_success=r["success"],
                                     rerun_success=again["success"], same_outcome=same_outcome, identical_numbers=identical,
                                     compared=list(RESULT_KEYS) + ["grasp_forces"], batch_category=r["category"],
                                     rerun_category=again["category"], wall_s=again["wall_s"]))
        report["resources"] = dict(wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(),
                                   gpu=f"none for physics; rendering: {renderer}")
        (args.out / "videos.json").write_text(json.dumps(report, indent=2))      # before the check, so a mismatch is on record
        print(f"{label}: batch {'SUCCESS' if r['success'] else 'FAIL ' + str(r['category'])}, rerun "
              f"{'SUCCESS' if again['success'] else 'FAIL ' + str(again['category'])}, identical numbers: {identical}")
        assert same_outcome, f"{label}: the re-run's outcome differs from the batch's (see videos.json)"


# ---------------------------------------------------------------------------------------------------------------- summarize
def cmd_summarize(args):
    t0 = time.perf_counter()
    run1, r0 = load_run(args.out, "run1", N_TRIALS), load_run(args.out, "r0_mug_v1", N_TRIALS)
    if run1 is None or r0 is None:
        raise SystemExit("summarize needs runs/run1.json and runs/r0_mug_v1.json")
    fifty = load_run(args.out, "fifty", N_FIFTY)
    legacy_path = args.out / "runs" / f"{LEGACY_RECHECK}.json"
    legacy_run = json.loads(legacy_path.read_text())
    assert legacy_run["tag"] == "after_final" and legacy_run["effective_params"] == legacy_defaults(), legacy_path
    videos_path = args.out / "videos.json"
    videos = json.loads(videos_path.read_text()) if videos_path.exists() else None

    k1 = run1["summary"]["successes"]
    met = k1 >= STOP_SUCCESSES
    strip = lambda rows: [{k: v for k, v in r.items() if k != "grasp_forces"} for r in rows]
    l3, reach = json.loads(L3_RUN.read_text()), reach_guard(GUARDS["L3"])
    ladder = json.loads((LADDER_DIR / "ladder.json").read_text())
    out = dict(
        question=f"Track B final: does L1 + L3 with HERO's closing targets (profile {PROFILE_NAME}) grasp my mug (B3's R3: full "
                 f"mug from config, 0.40 kg) in >= {STOP_SUCCESSES}/{N_TRIALS} of B3's trials?",
        why="the v3 ladder was cumulative, so L3 was tested on top of L2, which had made the grip weaker (lift slip 47 mm vs 22 mm); "
            "this run tests L1 + L3 with HERO's closing targets. One pre-registered test, not a search.",
        profile=dict(name=PROFILE_NAME, params=PROFILE, effective_params=run1["effective_params"], source="sim/pick_place.py GRASP_PROFILES",
                     grasp=run1["grasp"], contact=run1["contact"]),
        protocol=dict(trials=STEPS, moments=MOMENTS, record=RECORD,
                      success="cup tilt < 10 deg, cup centre within coaster radius (5 cm), resting on the coaster (|dz| < 4 mm), robot "
                              "never fell (sim/pick_place.py, unchanged); Wilson score interval, z = 1.96",
                      lift_slip=f"PickPlace lift_slip_in_hand_cm x 10, over the trials that reached the end of the lift (n); "
                                f"over_lift_slip_limit = trials above {LIFT_SLIP_FAIL_M * 1e3:g} mm",
                      contact_fraction_at_lift="B3's definition (thumb AND (index OR middle) AND palm touching at the lift frame), "
                                               "denominator = trials",
                      decision_rule=f"Run 1 >= {STOP_SUCCESSES}/{N_TRIALS}: the 50-trial run + 2 videos; otherwise the force and "
                                    "torque table of the failed trials, and Track B stops (no more fixes)",
                      mujoco=mujoco.__version__),
        run1=dict(run_file=str(runs_path(args.out, "run1")), **metrics(run1["summary"]), forces=run1["forces"],
                  failed_trials=failed_trials(run1["runs"])),
        reference_v3=dict(source=str(LADDER_DIR / "ladder.json"), before_ladder=metrics_only(ladder["before_ladder"]),
                          **{e["rung"]: metrics_only(e) for e in ladder["rungs"]}),
        checks=dict(
            r0_mug_v1=dict(run_file=str(runs_path(args.out, "r0_mug_v1")), **metrics(r0["summary"]), required=f"{N_TRIALS}/{N_TRIALS}",
                           passed=r0["summary"]["successes"] == N_TRIALS, forces=r0["forces"],
                           vs_l3_r0=same_rows(strip(r0["runs"]), json.loads(L3_R0_RUN.read_text())["runs"], L3_R0_RUN, ref_name="l3_r0"),
                           vs_l3_r0_why="on the 80 mm cylinder mug_v1 = the ladder's L3_R0 guard (radius_consistent_offset(0.040) == "
                                        "GRASP_OFFSET exactly, HERO's closing targets, contact model grasp): identical rows show the "
                                        "force recording changed nothing"),
            legacy_r0_after_final=dict(command="python -m sim.m1_grasp_ladder run --rung R0 --tag after_final --out "
                                               f"{args.out}", **legacy_entry(legacy_path)),
            reach_test=dict(rerun=False, why="mug_v1's global contact options are exactly L3's, which passed the reach test "
                                             "(results/m1_grasp_fix/v3/reach_l3/); the reach test uses the cylinder and only the "
                                             "contact model, so the grasp offset and the closing targets play no role there",
                            compiled_contact_settings_equal_l3=run1["contact"] == l3["contact"] == r0["contact"],
                            mug_v1_contact=run1["contact"], l3_contact=l3["contact"],
                            l3_reach_guard={k: reach[k] for k in ("results_file", "table_height_m", "contact_model", "required", "passed",
                                                                  "fell", "pelvis_z_min_m", "pelvis_tilt_max_deg", "root_xy_drift_max_m")})),
        decision=dict(run1_successes=k1, threshold=STOP_SUCCESSES, met=met,
                      next="the 50-trial run + 2 videos" if met else "STOP Track B: report the failed trials' forces and torques; no more fixes"),
        fifty=(dict(run_file=str(runs_path(args.out, "fifty")), **metrics(fifty["summary"]), forces=fifty["forces"],
                    failed_trials=failed_trials(fifty["runs"]), videos=videos,
                    # the same trials in other worker chunks (13/13/12/12 against 5 x 4): equal rows = no order dependence
                    trials_0_19_vs_run1=same_rows(fifty["runs"][:N_TRIALS], run1["runs"], runs_path(args.out, "run1"), ref_name="run1"))
               if fifty else None),
        resources=dict(
            run1=run1["resources"], r0_mug_v1=r0["resources"], legacy_r0_after_final=legacy_run["resources"],
            fifty=fifty["resources"] if fifty else None, videos=videos["resources"] if videos else None,
            summarize=dict(wall_s=round(time.perf_counter() - t0, 2), peak_rss_mb=peak_rss_mb())),
        open_questions=OPEN_QUESTIONS + approach_question(run1, fifty, legacy_run))
    if met and fifty is None:
        print("NOTE: Run 1 met the bar but runs/fifty.json is missing: run --step fifty, then videos, then summarize again")
    (args.out / "final.json").write_text(json.dumps(out, indent=2))
    md = summary_md(out)
    (args.out / "summary.md").write_text(md)
    print(md)
    print(f"wrote {args.out / 'final.json'} and {args.out / 'summary.md'}")


def metrics_only(e: dict) -> dict:
    """A ladder.json entry's headline numbers (for the reference rows)."""
    return {k: e[k] for k in ("success", "wilson95", "failures_by_category", "lift_slip_in_hand_mm", "lift_cup_rise_cm",
                              "contact_fraction_at_lift")} | ({"own_params": e["own_params"]} if "own_params" in e else {})


OPEN_QUESTIONS = [
    "The 'closed' moment is the first step at full closure; the finger targets slew at <= 2 rad/s, so at that step the joint "
    "targets may still lag the closed pose. The 0.5 s hold after it (the close gate) is not recorded.",
    "Normal force per part sums every contact in data.contact; a contact inside the margin with no active constraint has zero "
    "force, so B3's 'touching' (a contact exists) and a force > 0 can differ.",
]


def approach_question(run1: dict, fifty: dict | None, legacy_run: dict) -> list[str]:
    """Not measured, so only listed: the 50-trial run's approach failures (the cup moved > 2 cm before the grasp)."""
    knocked = [r for r in (fifty or {}).get("runs", []) if r["failed_phase"] == "approach"]
    if not knocked:
        return []
    cyl = cylinder_runs()                  # M1's 50 cylinder runs, same layouts in trial order
    return [f"Not measured: why {len(knocked)} of the 50 trials (trials {[r['trial'] for r in knocked]}) knocked the cup during the "
            f"approach, before any finger closed (Run 1: {sum(r['failed_phase'] == 'approach' for r in run1['runs'])}). Their cup "
            f"positions and reasons are in fifty.failed_trials. On the same layouts M1's cylinder (legacy offset) succeeded in "
            f"{sum(cyl[r['trial']]['success'] for r in knocked)}/{len(knocked)}. mug_v1 moves the palm target (L1: approach "
            f"direction {run1['grasp']['approach_dir_in_palm_frame_deg']} instead of {legacy_run['grasp']['approach_dir_in_palm_frame_deg']} "
            "deg in the palm frame) and changes the contact model; which of these matters was not tested (no more fixes in this track)."]


def _ms(v: dict | None, fmt: str = ".2f") -> str:
    return f"{v['mean']:{fmt}} / {v['median']:{fmt}}" if v else "-"


def _metric_row(name: str, m: dict) -> str:
    slip, rise, c = m["lift_slip_in_hand_mm"], m["lift_cup_rise_cm"], m["contact_fraction_at_lift"]
    fails = ", ".join(f"{k} {v}" for k, v in m["failures_by_category"].items()) or "-"
    slip_s = f"{slip['mean']} / {slip['median']} (n {slip['n']}); {slip['over_lift_slip_limit']}" if slip["n"] else "- (n 0)"
    rise_s = f"{rise['mean']} / {rise['median']}" if rise["n"] else "-"
    return (f"| {name} | {m['success']} | {m['wilson95'][0]:.3f}-{m['wilson95'][1]:.3f} | {fails} | {slip_s} | {rise_s} | "
            f"{c['grasp_contact']}/{c['of']} |")


def _force_tables(forces: dict) -> list[str]:
    lines = ["| moment | trials reached | " + " | ".join(f"{p} N (mean / median)" for p in PARTS) + " |", "|---" * (len(PARTS) + 2) + "|"]
    lines += [f"| {mo} | {f['trials_reached']} | " + " | ".join(_ms(f["normal_force_N"][p]) for p in PARTS) + " |" for mo, f in forces.items()]
    lines += ["", "| moment | " + " | ".join(f"{j} (limit {e:g})" for j, e in zip(FINGER_SUFFIXES, EFFORT)) + " |",
              "|---" * (len(FINGER_SUFFIXES) + 1) + "|"]
    lines += [f"| {mo}: tau N m (mean / median) | " + " | ".join(_ms(f["finger_tau_Nm"][j]) for j in FINGER_SUFFIXES) + " |"
              for mo, f in forces.items()]
    lines += [f"| {mo}: saturated (trials of {f['trials_reached']}) | " + " | ".join(str(f["saturated_trials"][j]) for j in FINGER_SUFFIXES)
              + " |" for mo, f in forces.items()]
    lines += ["", "| moment | " + " | ".join(f"touch {t} N (mean / median)" for t in FINGERTIP_LINKS) + " |", "|---" * (len(FINGERTIP_LINKS) + 1) + "|"]
    lines += [f"| {mo} | " + " | ".join(_ms(f["touch_N"][t]) for t in FINGERTIP_LINKS) + " |" for mo, f in forces.items()]
    return lines


def _failed_table(failed: list[dict]) -> list[str]:
    lines = ["| trial (cup xy, m) | yaw (deg) | failure (M1 category: reason) | moment | normal force thumb / index / middle / palm (N) | tau " + " / ".join(FINGER_SUFFIXES)
             + " (N m; * = saturated) | touch " + " / ".join(FINGERTIP_LINKS) + " (N) |", "|---|---|---|---|---|---|---|"]
    for f in failed:
        for mo in MOMENTS:
            x, failure, trial = f[mo], f"{f['category']}: {f['reason']}", f"{f['trial']} ({f['cup_xy'][0]}, {f['cup_xy'][1]})"
            if x is None:
                lines.append(f"| {trial} | {f['cup_yaw_deg']} | {failure} | {mo} | not reached | - | - |")
                continue
            lines.append(f"| {trial} | {f['cup_yaw_deg']} | {failure} | {mo} | "
                         + " / ".join(f"{x['normal_force_N'][p]:.2f}" for p in PARTS) + " | "
                         + " / ".join(f"{x['finger_tau_Nm'][j]:.2f}" + ("*" if x["saturated"][j] else "") for j in FINGER_SUFFIXES) + " | "
                         + " / ".join(f"{x['touch_N'][t]:.2f}" for t in FINGERTIP_LINKS) + " |")
    return lines


def summary_md(o: dict) -> str:
    head = ["| run | success | Wilson 95 % | failures (M1 categories) | lift slip mean / median (mm); > 20 mm | rise mean / median (cm) | "
            "contact at lift |", "|---|---|---|---|---|---|---|"]
    ref = o["reference_v3"]
    c, chk = o["checks"], o["checks"]["reach_test"]
    md = [f"# Track B final - L1 + L3 with HERO's finger closing (profile {PROFILE_NAME})", "",
          f"Why: {o['why']}", "",
          f"Profile `{PROFILE_NAME}` (sim/pick_place.py GRASP_PROFILES): {PROFILE}. Everything else is the M1 oracle; nothing was tuned.",
          "", "## Run 1: B3's R3 (full mug from config, 0.40 kg, B3's 20 trials, R3's handle yaws)", "", *head,
          _metric_row(f"**Run 1 ({PROFILE_NAME})**", o["run1"]),
          *(_metric_row(f"v3 {name} (reference)", ref[key]) for name, key in (("B3 R3, no change", "before_ladder"), ("L1", "L1"),
                                                                            ("L2 (L1 + L2)", "L2"), ("L3 (L1 + L2 + L3)", "L3")) if key in ref),
          "", f"Reference rows: {ref['source']} (read only).", "",
          "## Forces, finger torques and touch in Run 1 (over the trials that reached each moment)", "",
          *(f"- {k}: {v}" for k, v in MOMENTS.items()), *(f"- {k}: {v}" for k, v in RECORD.items()), "",
          *_force_tables(o["run1"]["forces"]), "",
          "## Checks", "",
          f"- R0 with {PROFILE_NAME} (M1 cylinder, its 20 trials): {c['r0_mug_v1']['success']} (required {c['r0_mug_v1']['required']}) -> "
          f"{'PASS' if c['r0_mug_v1']['passed'] else 'FAIL'}. Rows identical to v3's L3_R0 (same physics on the cylinder; every field "
          f"except wall time): {c['r0_mug_v1']['vs_l3_r0']['trials_identical']}/{c['r0_mug_v1']['vs_l3_r0']['trials']}.",
          f"- Legacy R0 after the final edits (`{c['legacy_r0_after_final']['command']}`): {c['legacy_r0_after_final']['success']}, "
          f"trials identical to M1 {c['legacy_r0_after_final']['trials_identical_to_m1']}, to B3 R0 "
          f"{c['legacy_r0_after_final']['trials_identical_to_b3_r0']} -> {'PASS' if c['legacy_r0_after_final']['passed'] else 'FAIL'}.",
          f"- Reach test not rerun: {chk['why']}. Compiled contact settings of these runs equal L3's: "
          f"{chk['compiled_contact_settings_equal_l3']} (cone {chk['mug_v1_contact']['cone']}, impratio {chk['mug_v1_contact']['impratio']}, "
          f"noslip_iterations {chk['mug_v1_contact']['noslip_iterations']}, condim 4 on right palm / thumb_2 / index_1 / middle_1). L3's "
          f"reach guard: passed {chk['l3_reach_guard']['passed']} (fell {chk['l3_reach_guard']['fell']}, pelvis z min "
          f"{chk['l3_reach_guard']['pelvis_z_min_m']} m, tilt max {chk['l3_reach_guard']['pelvis_tilt_max_deg']} deg).", "",
          "## Decision", "",
          f"Run 1 = {o['decision']['run1_successes']}/{N_TRIALS}; threshold {STOP_SUCCESSES}/{N_TRIALS}: "
          f"{'met' if o['decision']['met'] else 'not met'} -> {o['decision']['next']}.", ""]
    if o["run1"]["failed_trials"]:
        md += ["## Failed trials of Run 1: forces, torques and touch at both moments", "", *_failed_table(o["run1"]["failed_trials"]), ""]
    if o["decision"]["met"] and o["fifty"]:
        f = o["fifty"]
        md += [f"## 50 trials (trials 0-49 of trials(), full mug 0.40 kg, {PROFILE_NAME})", "", *head, _metric_row("**50 trials**", f), "",
               f"- Trials 0-19 of this run are Run 1's trials in other worker chunks: identical rows (every field except wall time) "
               f"{f['trials_0_19_vs_run1']['trials_identical']}/{f['trials_0_19_vs_run1']['trials']}."]
        if f["videos"]:
            md += [f"- video: `{v['video']}` (trial {v['trial']}, batch {'success' if v['batch_success'] else 'failure: ' + str(v['batch_category'])}; "
                   f"re-run same outcome {v['same_outcome']}, identical numbers {v['identical_numbers']})" for v in f["videos"]["videos"]]
        if f["failed_trials"]:
            md += ["", "Failed trials of the 50 (forces / torques / touch at both moments):", "", *_failed_table(f["failed_trials"])]
        md += [""]
    res = o["resources"]
    md += ["## Runtime and peak RSS (CPU only)", "", "| step | wall (s) | peak RSS (MB) |", "|---|---|---|"]
    for name, r in res.items():
        if r is None:
            md.append(f"| {name} | not run | - |")
        elif "per_process" in r:
            md.append(f"| {name} | {r['wall_s']} | max process {max(p['peak_rss_mb'] for p in r['per_process'])}, sum "
                      f"{r['peak_rss_mb_sum_of_processes']}, parent {r['parent_peak_rss_mb']} |")
        else:
            md.append(f"| {name} | {r['wall_s']} | {r['peak_rss_mb']} |")
    md += ["", *(f"- Open question: {q}" for q in o["open_questions"]), "", "All numbers: final.json (raw rows: runs/)."]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one step's trials with the mug_v1 profile")
    run.add_argument("--step", choices=tuple(STEPS), required=True)
    run.add_argument("--trials", type=int, default=None, help="default: the step's own count (smoke tests: fewer, another --out)")
    run.add_argument("--overwrite", action="store_true", help="replace an existing runs file")
    for p in (run, sub.add_parser("videos"), sub.add_parser("summarize")):
        p.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    dict(run=cmd_run, videos=cmd_videos, summarize=cmd_summarize)[args.cmd](args)


if __name__ == "__main__":
    main()
