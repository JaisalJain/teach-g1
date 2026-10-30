"""M1 oracle re-run with my mug: the scripted pick-and-place of sim/pick_place.py, unchanged, on the mug model.

The oracle (HERO side grasp, phases, touch logic, success criterion) is exactly M1's; only the object changes: the mug from
config/objects.yaml (sim/scene.py, cup_model="mug") instead of the 8 x 10 cm cylinder. The trials are paired with M1's 50/50:
- layouts: M1's 50 in-workspace layouts, in order = sample_layouts(grasp, 10, seed=0) (results/m1_pick) followed by
  sample_layouts(grasp, 40, seed=1) (results/m1_pick_extended), default PALM_BOX; trial k = 0..49. The layouts depend only on
  the grasp's horizontal palm offset, which does not depend on the cup's height, so they are the same cup / coaster positions.
- handle yaw: the oracle grasps the body and ignores the handle; the handle points away from the palm (along the oracle's
  approach direction ``Grasp.approach_dir``, palm -> cup) give or take U(-45, +45) deg, drawn once for all 50 trials with
  np.random.default_rng(2026). Trial k gets the same yaw at every mass.
- mass: config mass_kg (0.37 kg, an unverified guess) for the 50 trials; sensitivity: trials 0-19 at 0.25 and 0.45 kg
  (0.37 kg reuses trials 0-19 of the main run, not rerun).
The cylinder baseline differs in more than shape (mass, size, no handle); summarize says so next to the paired comparison.

Run from the repo root (env -u PYTHONPATH conda run -n hoi-sim python -m sim.m1_pick_mug ...):
  run --mass 0.37               -> results/m1_pick_mug/runs_mass0.37.json (trials 0-49, saved after every trial)
  run --mass 0.25 / --mass 0.45 -> runs_mass0.25.json / runs_mass0.45.json (trials 0-19). ``run`` refuses to replace an
                                   existing runs file unless --overwrite is given (a short run must not wipe the main batch)
  summarize                     -> summary.json + table.md: success + Wilson 95 % CI per run, failure categories (M1's),
                                   paired comparison with the cylinder's 50/50 (flipped layouts), mass sweep;
                                   stops if a runs file is incomplete or at the wrong mass
  videos                        -> re-runs one success and one failure of the 0.37 kg run with video (videos/<label>.mp4,
                                   gitignored; with no success: the first failure of each of two categories) and checks the
                                   outcome equals the batch's; renders with MUJOCO_GL=egl and needs
                                   __EGL_VENDOR_LIBRARY_FILENAMES (Mesa's, on this laptop) so rendering stays off the NVIDIA GPU
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")       # before mujoco's first import, which picks the GL backend (only `videos` renders)

import mujoco
import numpy as np

from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.policy_hero_export import HeroExportPolicy

from sim.pick_place import GRASP_Z_ABOVE_TABLE, PALM_BOX, TABLE_HEIGHT, Grasp, PickPlace, sample_layouts
from sim.scene import CUP_HEIGHT, CUP_MASS, CUP_RADIUS, TableScene, mug_config

OUT_DIR = Path("results/m1_pick_mug")
LAYOUT_SETS = ((0, 10), (1, 40))          # (sample_layouts seed, count) of M1's results/m1_pick and results/m1_pick_extended
N_TRIALS = sum(n for _, n in LAYOUT_SETS)
CYLINDER_RESULTS = (Path("results/m1_pick/pick_results.json"), Path("results/m1_pick_extended/pick_results.json"))
YAW_SEED = 2026
YAW_SPREAD_DEG = 45.0
MAIN_MASS = mug_config()["mass_kg"]
SWEEP_MASSES, SWEEP_TRIALS = (0.25, 0.45), 20
# Per-trial fields copied from PickPlace.run (None when the run stopped before that phase); the rest goes under "detail".
GRASP_KEYS = ("contact_closure", "lift_slip_in_hand_cm", "lift_cup_rise_cm", "cup_off_coaster_center_cm")
RESULT_KEYS = ("success", "failed_phase", "reason", "category", "cup_off_coaster_center_cm", "cup_tilt_deg", "contact_closure",
               "lift_slip_in_hand_cm", "lift_cup_rise_cm")


def trials(grasp: Grasp) -> list[dict]:
    """The 50 trials: M1's layouts in M1's order, each with its handle yaw (deg)."""
    layouts = [(seed, i, cup, coaster) for seed, n in LAYOUT_SETS for i, (cup, coaster) in enumerate(sample_layouts(grasp, n, seed))]
    approach_deg = math.degrees(math.atan2(grasp.approach_dir[1], grasp.approach_dir[0]))
    spread = np.random.default_rng(YAW_SEED).uniform(-YAW_SPREAD_DEG, YAW_SPREAD_DEG, size=len(layouts))
    return [dict(trial=k, layout_seed=seed, layout_index=i, cup_xy=cup, coaster_xy=coaster, handle_yaw_deg=approach_deg + float(d))
            for k, ((seed, i, cup, coaster), d) in enumerate(zip(layouts, spread))]


def run_trial(task: PickPlace, t: dict, video_path: Path | None = None) -> dict:
    t0 = time.perf_counter()
    r = task.run(t["cup_xy"], t["coaster_xy"], video_path, cup_yaw=math.radians(t["handle_yaw_deg"]))
    row = dict(trial=t["trial"], layout_seed=t["layout_seed"], layout_index=t["layout_index"], cup_xy=r["cup_xy"], coaster_xy=r["coaster_xy"],
               handle_yaw_deg=round(t["handle_yaw_deg"], 2), mass_kg=task.scene.cup_mass, **{k: r.get(k) for k in RESULT_KEYS},
               wall_s=round(time.perf_counter() - t0, 1))
    row["detail"] = {k: v for k, v in r.items() if k not in row}
    return row


def runs_path(out: Path, mass: float) -> Path:
    return out / f"runs_mass{mass:.2f}.json"


def make_task(mass: float) -> PickPlace:
    return PickPlace(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), cup_model="mug", cup_mass=mass)


def mass_properties(scene: TableScene) -> dict:
    """What MuJoCo derived from the mug's geoms: COM offset and inertia about the COM, in the cup frame (x = handle)."""
    m, b = scene.model, scene.cup_body
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    return dict(mass_kg=round(float(m.body_mass[b]), 6), com_offset_m=m.body_ipos[b].round(5).tolist(),
                inertia_diag_kgm2=(R @ np.diag(m.body_inertia[b]) @ R.T).diagonal().round(7).tolist())


def protocol(task: PickPlace) -> dict:
    g = task.grasp
    return dict(cup_model="mug", mug=mug_config(), mass_kg=task.scene.cup_mass, mass_properties=mass_properties(task.scene),
                table_height_m=TABLE_HEIGHT,
                oracle="sim/pick_place.py PickPlace, unchanged: HERO Dex3 side grasp on the body (handle ignored), phases, touch "
                       "logic and success criterion as in M1; the cup's own height replaces the cylinder's",
                grasp=dict(palm_offset_m=g.palm_offset.round(4).tolist(), palm_quat_wxyz=g.quat_wxyz.round(4).tolist(),
                           approach_dir=g.approach_dir.round(4).tolist()),
                layouts="sample_layouts(grasp, 10, seed=0) + sample_layouts(grasp, 40, seed=1) = M1's 50 (results/m1_pick + "
                        "results/m1_pick_extended); trial k = 0..49 in that order", palm_box=PALM_BOX,
                handle_yaw=f"atan2(approach_dir) + U(-{YAW_SPREAD_DEG:.0f}, +{YAW_SPREAD_DEG:.0f}) deg, np.random.default_rng({YAW_SEED}), "
                           "drawn once for all 50 trials; handle = mug body +x",
                approach_yaw_deg=round(math.degrees(math.atan2(g.approach_dir[1], g.approach_dir[0])), 2),
                rerun_source="trials() in sim/m1_pick_mug.py regenerates every trial at full precision; this file rounds cup / "
                             "coaster xy to 0.1 mm and the handle yaw to 0.01 deg, so re-run from trials(), not from these numbers",
                success="cup tilt < 10 deg, body centre within coaster radius (5 cm), resting on the coaster (|dz| < 4 mm), robot never fell",
                mujoco=mujoco.__version__)


# ---------------------------------------------------------------------------------------------------------------- run
def cmd_run(args):
    n = args.trials or (N_TRIALS if args.mass == MAIN_MASS else SWEEP_TRIALS)
    path = runs_path(args.out, args.mass)
    if path.exists() and not args.overwrite:      # e.g. "run --mass 0.37 --trials 20" must not wipe the 50-trial main batch
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    args.out.mkdir(parents=True, exist_ok=True)
    task = make_task(args.mass)
    results = dict(protocol=protocol(task), runs=[])
    for t in trials(task.grasp)[:n]:
        r = run_trial(task, t)
        results["runs"].append(r)
        path.write_text(json.dumps(results, indent=2))          # saved after every trial, so a stopped batch keeps its trials
        print(f"trial {r['trial']:2d} yaw {r['handle_yaw_deg']:7.2f}  -> " + ("SUCCESS" if r["success"] else f"FAIL {r['failed_phase']}: {r['reason']}")
              + f"  (cup {r['cup_off_coaster_center_cm']} cm off centre, tilt {r['cup_tilt_deg']} deg, {r['wall_s']} s)", flush=True)
    rate = success_rate(results["runs"])
    print(f"mass {args.mass} kg: {rate['successes']}/{rate['trials']}, Wilson 95 % CI {rate['wilson95']}; wrote {path}")


# ---------------------------------------------------------------------------------------------------------------- summarize
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (z = 1.96: 95 %)."""
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def success_rate(runs: list[dict]) -> dict:
    k, n = sum(bool(r["success"]) for r in runs), len(runs)
    lo, hi = wilson(k, n)
    return dict(successes=k, trials=n, rate=round(k / n, 3), wilson95=[round(lo, 3), round(hi, 3)],
                failures_by_category=dict(sorted(Counter(r["category"] for r in runs if not r["success"]).items())))


def cylinder_runs() -> list[dict]:
    """M1's 50 random in-workspace runs (cylinder, 0.15 kg) in trial order."""
    runs = []
    for path, (seed, n) in zip(CYLINDER_RESULTS, LAYOUT_SETS):
        res = json.loads(path.read_text())
        assert res["protocol"]["seed"] == seed and res["protocol"]["palm_box"] == {k: list(v) for k, v in PALM_BOX.items()}, path
        rand = [r for r in res["runs"] if r["name"].startswith("random")]
        assert len(rand) == n, path
        runs += rand
    return runs


def load_runs(out: Path, mass: float, n: int) -> list[dict] | None:
    """One mass's runs file, checked to hold exactly trials 0..n-1, all at that mass (None if it does not exist yet)."""
    path = runs_path(out, mass)
    if not path.exists():
        return None
    runs = json.loads(path.read_text())["runs"]
    if [r["trial"] for r in runs] != list(range(n)) or any(r["mass_kg"] != mass for r in runs):
        raise SystemExit(f"{path}: expected trials 0-{n - 1} at {mass} kg, found {len(runs)} trials at "
                         f"{sorted({r['mass_kg'] for r in runs})} kg (stopped batch or wrong --mass/--trials?)")
    return runs


def cmd_summarize(args):
    main = load_runs(args.out, MAIN_MASS, N_TRIALS)
    if main is None:
        raise SystemExit(f"no {runs_path(args.out, MAIN_MASS)}: run the main batch first")
    cyl = cylinder_runs()
    for r, c in zip(main, cyl):          # paired layouts: the mug's trial k must be the cylinder's run k
        assert (c["cup_xy"], c["coaster_xy"]) == (r["cup_xy"], r["coaster_xy"]), f"trial {r['trial']}: layout differs from the cylinder's"
    sweep = {MAIN_MASS: main[:SWEEP_TRIALS]}
    trial_key = lambda r: (r["cup_xy"], r["coaster_xy"], r["handle_yaw_deg"])
    for m in SWEEP_MASSES:
        v = load_runs(args.out, m, SWEEP_TRIALS)
        if v is None:
            print(f"mass sweep: no {runs_path(args.out, m)} yet, left out")
            continue
        assert [trial_key(r) for r in v] == [trial_key(r) for r in main[:SWEEP_TRIALS]], f"{m} kg: not the main run's trials"
        sweep[m] = v

    mug_rate, cyl_rate = success_rate(main), success_rate(cyl)
    rates = {f"mug {MAIN_MASS:.2f} kg, {len(main)} layouts": mug_rate, f"cylinder {CUP_MASS:.2f} kg (M1), same {len(cyl)} layouts": cyl_rate}
    rates |= {f"mug {m:.2f} kg, trials 0-{len(v) - 1}" + (" (= main run)" if m == MAIN_MASS else ""): success_rate(v) for m, v in sorted(sweep.items())}
    paired = [dict(trial=r["trial"], cylinder=c["success"], mug=r["success"], mug_category=r["category"], mug_reason=r["reason"],
                   handle_yaw_deg=r["handle_yaw_deg"], cup_xy=r["cup_xy"], coaster_xy=r["coaster_xy"]) for r, c in zip(main, cyl)]
    flips = [p for p in paired if p["cylinder"] != p["mug"]]
    cyl_only, mug_only = sum(p["cylinder"] for p in flips), sum(p["mug"] for p in flips)
    mug = mug_config()
    confounds = (f"the cylinder differs from the mug in more than shape: mass {CUP_MASS} vs {MAIN_MASS} kg, body radius "
                 f"{CUP_RADIUS * 100:.2f} vs {mug['body_outer_diameter_m'] * 50:.2f} cm, height {CUP_HEIGHT * 100:.1f} vs "
                 f"{mug['height_m'] * 100:.1f} cm, no handle; the palm is at the same {GRASP_Z_ABOVE_TABLE * 100:.1f} cm above the table, "
                 f"i.e. {(GRASP_Z_ABOVE_TABLE - CUP_HEIGHT / 2) * 100:.2f} vs {(GRASP_Z_ABOVE_TABLE - mug['height_m'] / 2) * 100:.2f} cm above the cup centre")
    summary = dict(
        protocol=json.loads(runs_path(args.out, MAIN_MASS).read_text())["protocol"],
        success=rates,
        comparison_with_cylinder=dict(cylinder=f"{cyl_rate['successes']}/{cyl_rate['trials']}", mug=f"{mug_rate['successes']}/{mug_rate['trials']}",
                                      both_succeeded=sum(p["cylinder"] and p["mug"] for p in paired),
                                      both_failed=sum(not p["cylinder"] and not p["mug"] for p in paired),
                                      cylinder_only_succeeded=cyl_only, mug_only_succeeded=mug_only,
                                      confounds=confounds, flipped=flips),
        mass_sweep=[dict(trial=k, **{f"{m:.2f} kg": sweep[m][k]["success"] for m in sorted(sweep)}) for k in range(SWEEP_TRIALS)],
        # how the grasp behaved, per run (over the trials that reached each measurement; mean / min / max)
        grasp_stats={name: {k: _stats([r[k] for r in runs if r.get(k) is not None]) for k in GRASP_KEYS}
                     for name, runs in [(f"mug {MAIN_MASS:.2f} kg, {len(main)} layouts", main), (f"cylinder {CUP_MASS:.2f} kg (M1)", cyl)]
                     + [(f"mug {m:.2f} kg, trials 0-{len(v) - 1}", v) for m, v in sorted(sweep.items())]},
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    table = ["| run | success | Wilson 95 % CI | failures by category |", "|---|---|---|---|"]
    table += [f"| {name} | {v['successes']}/{v['trials']} | {v['wilson95'][0] * 100:.0f}-{v['wilson95'][1] * 100:.0f} % | "
              + (", ".join(f"{c}: {n}" for c, n in v["failures_by_category"].items()) or "-") + " |" for name, v in rates.items()]
    table += ["", f"Paired with the cylinder (same layouts): {len(flips)} flipped ({cyl_only} cylinder-only successes, "
              f"{mug_only} mug-only).", f"Caveat: {confounds}."]
    table += [f"- trial {p['trial']}: cylinder {'success' if p['cylinder'] else 'fail'} -> mug "
              + ("success" if p["mug"] else f"fail ({p['mug_category']}: {p['mug_reason']})") + f", handle yaw {p['handle_yaw_deg']} deg" for p in flips]
    (args.out / "table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"wrote {args.out / 'summary.json'} and {args.out / 'table.md'}")


def _stats(v: list) -> dict | None:
    return dict(n=len(v), mean=round(float(np.mean(v)), 3), min=round(float(np.min(v)), 3), max=round(float(np.max(v)), 3)) if v else None


# ---------------------------------------------------------------------------------------------------------------- videos
def cmd_videos(args):
    # This laptop's NVIDIA GPU is reserved for other jobs: EGL must load Mesa's driver (the Intel iGPU) instead of NVIDIA's.
    if not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        raise SystemExit("videos: set __EGL_VENDOR_LIBRARY_FILENAMES (here: /usr/share/glvnd/egl_vendor.d/50_mesa.json) so "
                         "rendering does not land on the NVIDIA GPU")
    main = json.loads(runs_path(args.out, MAIN_MASS).read_text())["runs"]
    ok, bad = [r for r in main if r["success"]], [r for r in main if not r["success"]]
    first_per_category = []                     # the first failure of each failure category, in trial order
    for r in bad:
        if r["category"] not in [f["category"] for f in first_per_category]:
            first_per_category.append(r)
    picks = (ok[:1] + bad[:1]) if ok and bad else ok[:2] if ok else first_per_category[:2]
    (args.out / "videos").mkdir(parents=True, exist_ok=True)
    task = make_task(MAIN_MASS)
    all_trials = trials(task.grasp)
    report = []
    for r in picks:
        label = f"{'success' if r['success'] else 'failure'}_trial{r['trial']:02d}"
        again = run_trial(task, all_trials[r["trial"]], args.out / "videos" / f"{label}.mp4")
        same_outcome = all(again[k] == r[k] for k in ("success", "failed_phase", "category"))
        identical = all(again[k] == r[k] for k in RESULT_KEYS)
        report.append(dict(label=label, video=str(args.out / "videos" / f"{label}.mp4"), trial=r["trial"], batch_success=r["success"],
                           rerun_success=again["success"], same_outcome=same_outcome, identical_numbers=identical,
                           batch_category=r["category"], rerun_category=again["category"]))
        (args.out / "videos.json").write_text(json.dumps(report, indent=2))      # before the check, so a mismatch is on record
        print(f"{label}: batch {'SUCCESS' if r['success'] else 'FAIL ' + str(r['category'])}, rerun "
              f"{'SUCCESS' if again['success'] else 'FAIL ' + str(again['category'])}, identical numbers: {identical}")
        assert same_outcome, f"{label}: the re-run's outcome differs from the batch's (see videos.json)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run trials 0..N-1 at one mass")
    run.add_argument("--mass", type=float, default=MAIN_MASS)
    run.add_argument("--trials", type=int, default=None, help=f"default: {N_TRIALS} at {MAIN_MASS} kg, {SWEEP_TRIALS} at other masses")
    run.add_argument("--overwrite", action="store_true", help="replace an existing runs file for this mass")
    for p in (run, sub.add_parser("summarize"), sub.add_parser("videos")):
        p.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    dict(run=cmd_run, summarize=cmd_summarize, videos=cmd_videos)[args.cmd](args)


if __name__ == "__main__":
    main()
