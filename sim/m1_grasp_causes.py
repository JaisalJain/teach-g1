"""Track B, B3: why does the M1 oracle grasp drop my mug -- its size, its mass or its handle?

The oracle is sim/pick_place.py ``PickPlace``, unchanged (HERO side grasp, phases, touch logic, success criterion). Four runs
change the object one property at a time, on the same 20 layouts = trials 0-19 of ``sim.m1_pick_mug.trials()`` (M1's
results/m1_pick random_00-09 followed by results/m1_pick_extended random_00-09):

  R0  M1 cylinder (D 0.080, h 0.100 m), 0.15 kg                    baseline; must reproduce M1's 20/20
  R1  mug body cylinder (D 0.099, h 0.091 m from config), no handle, 0.15 kg     R0 -> R1 isolates the size
  R2  the same cylinder, config mass (0.40 kg)                     R1 -> R2 isolates the mass
  R3  full mug from config (body + handle 0.06 x 0.010 m), 0.40 kg R2 -> R3 isolates the handle

Cup yaw: R0-R2 are round with no handle, so they run at yaw 0 as M1 did (that keeps R0 identical to M1); R3 uses each trial's
handle yaw from ``trials()`` (the same draw as results/m1_pick_mug).

Besides PickPlace's own numbers (lift slip in the hand, cup rise), each trial records which parts of the right hand touch the
cup at the LIFT FRAME = the first control step of the oracle's "lift" phase: the hand is fully closed (after the 0.5 s hold at
the end of "close") and the lift reference has just started, so the palm has not risen yet (the min-jerk path starts at rest).
The contacts are read from ``data.contact`` after that 20 ms step, i.e. the contact list of its last 1 ms substep, by geom
pair: a right-hand geom is classified by its body (thumb_0/1/2 -> thumb, index_0/1 -> index, middle_0/1 -> middle,
palm_link -> palm; every one of these bodies has exactly one collision geom), the object is any geom of the cup body.
"Grasp contact" = thumb AND (index OR middle) AND palm. Reading ``data.contact`` changes nothing in the simulation.

Hand-cup contacts flicker from one 1 ms substep to the next, so a single substep is a snapshot. As information only (the
definition above stays the headline), each trial also counts, over all 20 substeps of the lift frame, how many had grasp
contact and how many had each part touching (``TableScene.on_substep``, read-only). Open question for the user: should "at the
lift frame" mean the last substep (used), any substep or all substeps?

Lift slip and rise: mean (the spec's number) plus median, and how many trials slipped more than the oracle's lift-slip limit
(sim/pick_place.py LIFT_SLIP_FAIL_M, 2 cm): a few lost cups pull the mean up.

Run from the repo root (env -u PYTHONPATH conda run -n hoi-sim python -m sim.m1_grasp_causes ...):
  run --run R0 (R1, R2, R3)  -> results/m1_grasp_fix/b3_runs/<run>.json (saved after every trial; refuses to replace an
                                existing file unless --overwrite). The four runs are independent processes.
  summarize                  -> results/m1_grasp_fix/b3_causes.json (per-trial rows, per-run summary with Wilson 95 % CI and
                                M1's failure categories, R0 vs M1 check) + b3_causes.md (one table)
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import mujoco
import numpy as np

from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.policy_hero_export import HeroExportPolicy

from sim.m1_pick_mug import CYLINDER_RESULTS, mass_properties, success_rate, trials
from sim.pick_place import CATEGORY, LIFT_SLIP_FAIL_M, TABLE_HEIGHT, PickPlace
from sim.scene import CUP_HEIGHT, CUP_MASS, CUP_RADIUS, mug_config

OUT_DIR = Path("results/m1_grasp_fix")
N_TRIALS = 20
MUG = mug_config()
RUNS = {
    "R0": dict(cup_model="cylinder", mass_kg=CUP_MASS, use_handle_yaw=False, changed="- (M1 baseline)",
               object=f"M1 cylinder, D {2 * CUP_RADIUS:.3f} m, h {CUP_HEIGHT:.3f} m"),
    "R1": dict(cup_model="mug_body", mass_kg=CUP_MASS, use_handle_yaw=False, changed="size",
               object=f"cylinder, D {MUG['body_outer_diameter_m']:.3f} m, h {MUG['height_m']:.3f} m, no handle"),
    "R2": dict(cup_model="mug_body", mass_kg=MUG["mass_kg"], use_handle_yaw=False, changed="mass",
               object=f"cylinder, D {MUG['body_outer_diameter_m']:.3f} m, h {MUG['height_m']:.3f} m, no handle"),
    "R3": dict(cup_model="mug", mass_kg=MUG["mass_kg"], use_handle_yaw=True, changed="handle",
               object=f"full mug (config), handle {MUG['handle_height_m']:.2f} x {MUG['handle_thickness_m']:.3f} m"),
}
PARTS = ("thumb", "index", "middle", "palm")
# Per-trial fields copied from PickPlace.run (None when the run stopped before that phase); the rest goes under "detail".
RESULT_KEYS = ("success", "failed_phase", "reason", "category", "contact_closure", "lift_slip_in_hand_cm", "lift_cup_rise_cm",
               "cup_off_coaster_center_cm", "cup_tilt_deg")
LIFT_FRAME = ("first control step of the oracle's 'lift' phase (hand fully closed after the 0.5 s close hold, lift reference just "
              "started, palm not yet risen); contacts = data.contact after that step (its last 1 ms substep)")
CONTACT_RULE = ("by geom pair in data.contact: right-hand geom -> its body (thumb_0/1/2 = thumb, index_0/1 = index, middle_0/1 = "
                "middle, palm_link = palm); object = any geom of the cup body; grasp contact = thumb AND (index OR middle) AND palm")
SUBSTEP_NOTE = ("information only: the same contact rule read after each of the lift frame's 20 substeps of 1 ms; any = grasp contact "
                "in at least one substep, all = in every substep. Contacts flicker between substeps, so the last-substep count is a "
                "snapshot")
OPEN_QUESTIONS = ["Should 'at the lift frame' mean the last 1 ms substep of that 20 ms step (used), any of its 20 substeps, or all "
                  "of them? Contacts flicker between substeps; summary.*.lift_frame.substep_sensitivity has all three counts."]


def hand_parts(scene) -> dict[int, str]:
    """Right-hand collision geom id -> "thumb" | "index" | "middle" | "palm", from the name of the geom's body."""
    m = scene.model
    parts = {}
    for g in scene.hand_geoms:
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        if body.startswith("right_hand_"):
            parts[g] = body.removeprefix("right_hand_").split("_")[0]     # e.g. right_hand_thumb_2_link -> thumb
    assert set(parts.values()) == set(PARTS), parts
    return parts


class ContactPickPlace(PickPlace):
    """PickPlace that also records which hand parts touch the cup at the lift frame. Only reads data.contact: same behaviour."""

    def __init__(self, policy, cup_model: str, cup_mass: float):
        super().__init__(policy, cup_model=cup_model, cup_mass=cup_mass)
        m = self.scene.model
        self.part_of_geom = hand_parts(self.scene)
        self.cup_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.scene.cup_body}
        self.lift_contacts = None

    def run(self, *args, **kwargs) -> dict:
        self.lift_contacts, self.lift_substeps = None, []
        result = super().run(*args, **kwargs)
        result["lift_frame_contacts"] = self.lift_contacts          # None = the trial never reached the lift
        result["lift_frame_substeps"] = substep_counts(self.lift_substeps) if self.lift_contacts is not None else None
        return result

    def _tick(self, phase: str):
        first_lift = phase == "lift" and self.lift_contacts is None
        if first_lift:      # information only: the contacts after each 1 ms substep of this step
            self.scene.on_substep = lambda: self.lift_substeps.append(self.object_contacts())
        try:
            super()._tick(phase)
        finally:          # also when that step raised (a fall): the step has been simulated
            if first_lift:
                self.scene.on_substep = None
                self.lift_contacts = self.object_contacts()
                assert self.lift_substeps[-1] == self.lift_contacts      # the definition = the last substep

    def object_contacts(self) -> dict[str, bool]:
        """Which hand parts are in contact with any cup geom right now."""
        touching = dict.fromkeys(PARTS, False)
        for c in self.scene.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            hand = b if a in self.cup_geoms else a if b in self.cup_geoms else None
            if hand in self.part_of_geom:
                touching[self.part_of_geom[hand]] = True
        return touching


def grasp_contact(contacts: dict | None) -> bool:
    return bool(contacts and contacts["thumb"] and (contacts["index"] or contacts["middle"]) and contacts["palm"])


def substep_counts(substeps: list[dict]) -> dict:
    """Over the lift frame's substeps: how many had grasp contact, and how many had each part touching."""
    return dict(n=len(substeps), grasp_contact=sum(map(grasp_contact, substeps)), parts={p: sum(c[p] for c in substeps) for p in PARTS})


def runs_path(out: Path, run: str) -> Path:
    return out / "b3_runs" / f"{run}.json"


def m1_cylinder_runs() -> list[dict]:
    """M1's runs for trials 0-19: results/m1_pick random_00-09, then results/m1_pick_extended random_00-09."""
    runs = []
    for path in CYLINDER_RESULTS:
        runs += [r for r in json.loads(path.read_text())["runs"] if r["name"].startswith("random")][:10]
    return runs


# ---------------------------------------------------------------------------------------------------------------- run
def cmd_run(args):
    cfg = RUNS[args.run]
    path = runs_path(args.out, args.run)
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    task = ContactPickPlace(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), cfg["cup_model"], cfg["mass_kg"])
    trial_list = trials(task.grasp)[:args.trials]
    # Same layouts as M1 (the layouts depend only on the grasp's horizontal palm offset, which no cup height changes)
    for t, m1 in zip(trial_list, m1_cylinder_runs()):
        assert [round(float(v), 4) for v in (*t["cup_xy"], *t["coaster_xy"])] == m1["cup_xy"] + m1["coaster_xy"], t["trial"]
    results = dict(run=args.run, config=cfg, mass_properties=mass_properties(task.scene), cup_height_m=task.scene.cup_height,
                   grasp_palm_offset_m=task.grasp.palm_offset.round(4).tolist(), runs=[])
    for t in trial_list:
        yaw_deg = t["handle_yaw_deg"] if cfg["use_handle_yaw"] else 0.0
        t0 = time.perf_counter()
        r = task.run(t["cup_xy"], t["coaster_xy"], cup_yaw=math.radians(yaw_deg))
        row = dict(trial=t["trial"], layout_seed=t["layout_seed"], layout_index=t["layout_index"], cup_xy=r["cup_xy"],
                   coaster_xy=r["coaster_xy"], cup_yaw_deg=round(yaw_deg, 2), mass_kg=task.scene.cup_mass,
                   **{k: r.get(k) for k in RESULT_KEYS}, lift_frame_contacts=r["lift_frame_contacts"],
                   lift_frame_substeps=r["lift_frame_substeps"], wall_s=round(time.perf_counter() - t0, 1))
        row["detail"] = {k: v for k, v in r.items() if k not in row}
        results["runs"].append(row)
        path.write_text(json.dumps(results, indent=2))          # saved after every trial
        print(f"{args.run} trial {row['trial']:2d} -> " + ("SUCCESS" if row["success"] else f"FAIL {row['failed_phase']}: {row['reason']}")
              + f"  lift contacts {row['lift_frame_contacts']}  ({row['wall_s']} s)", flush=True)
    # CPU only (the NVIDIA GPU belongs to perception); ru_maxrss is in kB on Linux
    results["resources"] = dict(wall_s=round(time.perf_counter() - t_start, 1), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                                gpu="none (CPU only, onnxruntime CPU)")
    path.write_text(json.dumps(results, indent=2))
    rate = success_rate(results["runs"])
    print(f"{args.run}: {rate['successes']}/{rate['trials']}, Wilson 95 % CI {rate['wilson95']}; {results['resources']}; wrote {path}")


# ---------------------------------------------------------------------------------------------------------------- summarize
def run_summary(runs: list[dict]) -> dict:
    reached = [r for r in runs if r["lift_frame_contacts"] is not None]
    slips = [r["lift_slip_in_hand_cm"] * 10 for r in runs if r["lift_slip_in_hand_cm"] is not None]
    rises = [r["lift_cup_rise_cm"] for r in runs if r["lift_cup_rise_cm"] is not None]
    k = sum(grasp_contact(r["lift_frame_contacts"]) for r in runs)
    sub = [r["lift_frame_substeps"] for r in reached]
    slip_stats = _stats(slips)
    if slip_stats:
        slip_stats["over_lift_slip_limit"] = sum(v > LIFT_SLIP_FAIL_M * 1e3 for v in slips)
    return dict(**success_rate(runs),
                lift_slip_in_hand_mm=slip_stats, lift_cup_rise_cm=_stats(rises),
                lift_frame=dict(reached=len(reached), grasp_contact=k,
                                fraction_of_all=round(k / len(runs), 3),
                                fraction_of_reached=round(k / len(reached), 3) if reached else None,
                                part_touching_counts={p: sum(r["lift_frame_contacts"][p] for r in reached) for p in PARTS},
                                substep_sensitivity=dict(
                                    grasp_contact_last_substep=k,
                                    grasp_contact_any_substep=sum(s["grasp_contact"] > 0 for s in sub),
                                    grasp_contact_all_substeps=sum(s["grasp_contact"] == s["n"] for s in sub),
                                    part_touching_any_substep={p: sum(s["parts"][p] > 0 for s in sub) for p in PARTS})))


def _stats(v: list) -> dict | None:
    return dict(n=len(v), mean=round(float(np.mean(v)), 2), median=round(float(np.median(v)), 2), min=round(float(np.min(v)), 2),
                max=round(float(np.max(v)), 2)) if v else None


def r0_vs_m1(r0: list[dict]) -> dict:
    """R0 must be M1: compare every trial's outcome and numbers with M1's stored run."""
    keys = ("success", "contact_closure", "lift_slip_in_hand_cm", "lift_cup_rise_cm", "cup_off_coaster_center_cm", "cup_tilt_deg")
    diffs = []
    for r, m in zip(r0, m1_cylinder_runs()):
        mine = {k: r[k] for k in keys} | dict(cup_final=r["detail"]["cup_final"], sim_seconds=r["detail"]["sim_seconds"])
        theirs = {k: m.get(k) for k in keys} | dict(cup_final=m["cup_final"], sim_seconds=m["sim_seconds"])
        if mine != theirs:
            diffs.append(dict(trial=r["trial"], r0={k: v for k, v in mine.items() if theirs[k] != v},
                              m1={k: v for k, v in theirs.items() if mine[k] != v}))
    return dict(m1_successes=sum(m["success"] for m in m1_cylinder_runs()), r0_successes=sum(r["success"] for r in r0),
                fields_compared=list(keys) + ["cup_final", "sim_seconds"], trials_identical=len(r0) - len(diffs), differences=diffs)


def cmd_summarize(args):
    data = {}
    for run in RUNS:
        res = json.loads(runs_path(args.out, run).read_text())
        if [r["trial"] for r in res["runs"]] != list(range(N_TRIALS)):
            raise SystemExit(f"{runs_path(args.out, run)}: expected trials 0-{N_TRIALS - 1}")
        data[run] = res
    layout = lambda r: (r["cup_xy"], r["coaster_xy"])
    assert all(list(map(layout, d["runs"])) == list(map(layout, data["R0"]["runs"])) for d in data.values()), "runs differ in layouts"
    # R3's handle yaws are the ones results/m1_pick_mug used for the same trials
    mug_v1 = json.loads(Path("results/m1_pick_mug/runs_mass0.37.json").read_text())["runs"][:N_TRIALS]
    assert [r["cup_yaw_deg"] for r in data["R3"]["runs"]] == [r["handle_yaw_deg"] for r in mug_v1], "R3 handle yaws differ from m1_pick_mug"

    summary = {run: dict(object=d["config"]["object"], cup_model=d["config"]["cup_model"], mass_kg=d["config"]["mass_kg"],
                         changed_from_previous=d["config"]["changed"], mass_properties=d["mass_properties"],
                         cup_height_m=d["cup_height_m"], grasp_palm_offset_m=d["grasp_palm_offset_m"], **run_summary(d["runs"]),
                         resources=d["resources"])
               for run, d in data.items()}
    rows = {run: [{k: v for k, v in r.items() if k != "detail"} | dict(grasp_contact_at_lift=grasp_contact(r["lift_frame_contacts"]))
                  for r in d["runs"]] for run, d in data.items()}
    out = dict(
        protocol=dict(oracle="sim/pick_place.py PickPlace, unchanged (HERO side grasp, phases, touch logic, success criterion)",
                      trials=f"trials 0-{N_TRIALS - 1} of sim.m1_pick_mug.trials() = M1's results/m1_pick random_00-09 + "
                             "results/m1_pick_extended random_00-09 (same layouts, asserted)",
                      cup_yaw="R0-R2: 0 deg (round, no handle; as in M1); R3: the trial's handle yaw from trials() (= results/m1_pick_mug, asserted)",
                      steps="R0 -> R1 isolates size (diameter and height), R1 -> R2 isolates mass, R2 -> R3 isolates the handle",
                      lift_frame=LIFT_FRAME, contact_rule=CONTACT_RULE,
                      lift_slip="PickPlace lift_slip_in_hand_cm x 10 (cup displacement relative to the palm over the lift), mean (the spec's "
                                "number) and median over trials that reached the end of the lift; over_lift_slip_limit = trials above "
                                f"LIFT_SLIP_FAIL_M = {LIFT_SLIP_FAIL_M * 1e3:g} mm (sim/pick_place.py)",
                      lift_rise="PickPlace lift_cup_rise_cm (cup centre rise over the 10 cm palm lift), mean and median over trials that reached the end of the lift",
                      substep_sensitivity=SUBSTEP_NOTE,
                      success="cup tilt < 10 deg, cup centre within coaster radius (5 cm), resting on the coaster (|dz| < 4 mm), robot never fell",
                      wilson="Wilson score interval, z = 1.96", m1_failure_categories=CATEGORY, table_height_m=TABLE_HEIGHT,
                      mug_config=MUG, mujoco=mujoco.__version__),
        summary=summary, r0_vs_m1=r0_vs_m1(data["R0"]["runs"]), open_questions=OPEN_QUESTIONS, trials=rows)
    (args.out / "b3_causes.json").write_text(json.dumps(out, indent=2))

    head = ("| run | object | mass (kg) | changed vs previous run | success | Wilson 95 % CI | failures (M1 categories) | lift slip in hand, "
            f"mean (median) (mm) [n]; > {LIFT_SLIP_FAIL_M * 1e3:g} mm | cup rise for 10 cm lift, mean (median) (cm) [n] | thumb + finger + palm "
            "at lift frame: of 20 | of trials reaching the lift | parts touching at lift frame (thumb / index / middle / palm) |")
    lines = [head, "|" + "---|" * head.count(" | ") + "---|"]
    for run, s in summary.items():
        lf = s["lift_frame"]
        stat = lambda v, unit_fmt: f"{v['mean']:{unit_fmt}} ({v['median']:{unit_fmt}}) [{v['n']}]" if v else "- [0]"
        lines.append(f"| {run} | {s['object']} | {s['mass_kg']:.2f} | {s['changed_from_previous']} | {s['successes']}/{s['trials']} | "
                     f"{s['wilson95'][0] * 100:.0f}-{s['wilson95'][1] * 100:.0f} % | "
                     + (", ".join(f"{c}: {n}" for c, n in s["failures_by_category"].items()) or "-") + " | "
                     f"{stat(s['lift_slip_in_hand_mm'], '.1f')}; {(s['lift_slip_in_hand_mm'] or {}).get('over_lift_slip_limit', 0)} | "
                     f"{stat(s['lift_cup_rise_cm'], '.2f')} | "
                     f"{lf['grasp_contact']}/{s['trials']} | " + (f"{lf['grasp_contact']}/{lf['reached']}" if lf["reached"] else "-") + " | "
                     + " / ".join(str(lf["part_touching_counts"][p]) for p in PARTS) + f" (of {lf['reached']}) |")
    chk = out["r0_vs_m1"]
    sub = {run: s["lift_frame"]["substep_sensitivity"] for run, s in summary.items()}
    md = ["# B3 - cause separation (oracle unchanged, trials 0-19)", "",
          "R0 -> R1 isolates the size (diameter and height), R1 -> R2 the mass, R2 -> R3 the handle.", "", *lines, "",
          f"- Lift frame: {LIFT_FRAME}.", f"- Contact: {CONTACT_RULE}. Trials that never reached the lift count as no contact in the "
          "'of 20' column.",
          "- Lift slip and rise: PickPlace's own measurements, over the trials that reached the end of the lift ([n]). The mean is the "
          f"spec's number; the median and the count above the oracle's {LIFT_SLIP_FAIL_M * 1e3:g} mm lift-slip limit show the typical trial "
          "(a few lost cups pull the mean up).",
          "- Lift-frame contacts are a 1 ms snapshot; contacts flicker between substeps. Thumb + finger + palm in the last / any / all of "
          "the frame's 20 substeps (information only): " + ", ".join(
              f"{run} {ss['grasp_contact_last_substep']}/{ss['grasp_contact_any_substep']}/{ss['grasp_contact_all_substeps']}"
              for run, ss in sub.items()) + "; index finger touching in the last / any substep: " + ", ".join(
              f"{run} {summary[run]['lift_frame']['part_touching_counts']['index']}/{ss['part_touching_any_substep']['index']}"
              for run, ss in sub.items()) + ".",
          *(f"- Open question: {q}" for q in OPEN_QUESTIONS),
          f"- Cup yaw: {out['protocol']['cup_yaw']}.",
          f"- R0 vs M1 (results/m1_pick + m1_pick_extended, runs 0-19): R0 {chk['r0_successes']}/20, M1 {chk['m1_successes']}/20; "
          f"{chk['trials_identical']}/20 trials identical in {', '.join(chk['fields_compared'])}.",
          "- Runtime (four parallel CPU processes, no GPU): " + ", ".join(
              f"{run} {s['resources']['wall_s']:.0f} s / peak RSS {s['resources']['peak_rss_mb']} MB" for run, s in summary.items()) + ".",
          "- All numbers: results/m1_grasp_fix/b3_causes.json (raw runs: b3_runs/R0-R3.json)."]
    (args.out / "b3_causes.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run trials 0..N-1 of one of R0-R3")
    run.add_argument("--run", choices=tuple(RUNS), required=True)
    run.add_argument("--trials", type=int, default=N_TRIALS)
    run.add_argument("--overwrite", action="store_true", help="replace an existing runs file")
    for p in (run, sub.add_parser("summarize")):
        p.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    dict(run=cmd_run, summarize=cmd_summarize)[args.cmd](args)


if __name__ == "__main__":
    main()
