"""M4b Step 3: heading sweep -- can the arm execute the oracle's grasp at another palm heading? Scripted oracle, no human data.

The oracle is Track B final's mug_v1 (sim/m1_grasp_final.py ForcePickPlace = PickPlace with GRASP_PROFILES["mug_v1"], built by
sim/m1_grasp_ladder.py make_task("R3", ...): my mug from config at 0.40 kg). Its grasp is turned about the cup axis by
delta = phi - 45 deg (``rotated_grasp``): palm_offset -> R_z(delta) palm_offset (the horizontal part turns, z is unchanged),
quat -> R_z(delta) R_ora, approach_dir -> R_z(delta) approach_dir. Height, settles, checks and everything else are the oracle's
(PickPlace.run reads only ``task.grasp`` for these three). The handle turns with the grasp: handle yaw = the trial's + delta.
Layouts: trials 0-19 of sim.m1_pick_mug.trials() (the legacy-grasp layouts, asserted against fifty.json's), phi in
{45, 55, 65, 75, 85} deg: 100 runs.

Check (spec): phi = 45 deg (delta = 0, the grasp asserted bit for bit equal to the oracle's) must reproduce Track B final's
mug_v1 rows for trials 0-19 exactly (results/m1_grasp_fix/v3_final/runs/fifty.json, every field but wall_s, via
sim/m1_grasp_ladder.py same_rows). phi = 45 runs first; if it does not reproduce, nothing else runs and the command exits 1.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.m4b_heading_sweep):
  -> results/m4b/heading_sweep/runs.json (100 rows: Track B final's row fields + phi_deg, delta_deg, trial_handle_yaw_deg)
     and results/m4b/heading_sweep/phi45_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data_tools import reach_specs as rs
from sim.m1_grasp_causes import trial_row
from sim.m1_grasp_final import N_FIFTY, PROFILE as FINAL_PROFILE, ForcePickPlace, reference_rows
from sim.m1_grasp_ladder import LAYOUT_KEYS, fixed_trials, layout_of, make_task, peak_rss_mb, same_rows
from sim.m4_execute import FIFTY, MAX_WORKERS, MUG_MASS_KG, RSS_CAP_MB
from sim.pick_place import GRASP_PROFILES, WRIST_YAW, Grasp

from retarget.frames import rot_z

PHIS_DEG = (45, 55, 65, 75, 85)
PHI_ORACLE_DEG = 45
N_SWEEP = 20                       # trials 0-19
OUT = Path("results/m4b/heading_sweep")


def rotated_grasp(g: Grasp, delta: float) -> Grasp:
    """The oracle's grasp turned by ``delta`` [rad] about the cup axis (module docstring). R_ora is rebuilt the way
    Grasp.hero_side_grasp builds it (asserted to give the grasp's own quaternion), so delta = 0 returns the same numbers."""
    R_ora = rs.palm_rotation(WRIST_YAW, 0.0, 0.0)
    assert np.array_equal(rs.mat_to_quat_wxyz(R_ora), g.quat_wxyz), "R_ora does not give the grasp's quaternion"
    Rz = rot_z(delta)
    offset = Rz @ g.palm_offset
    assert offset[2] == g.palm_offset[2], "R_z changed the offset's z"
    return Grasp(offset, rs.mat_to_quat_wxyz(Rz @ R_ora), Rz @ g.approach_dir)


def run_chunk(phi_deg: int, trial_ids: list[int]) -> dict:
    """One worker: Track B final's task with the grasp turned to ``phi_deg``, trials ``trial_ids``."""
    t0 = time.perf_counter()
    task = make_task("R3", FINAL_PROFILE, ForcePickPlace)
    assert FINAL_PROFILE == GRASP_PROFILES["mug_v1"] and task.scene.cup_mass == MUG_MASS_KG
    oracle_grasp = task.grasp
    delta_deg = phi_deg - PHI_ORACLE_DEG
    task.grasp = rotated_grasp(oracle_grasp, math.radians(delta_deg))
    if delta_deg == 0:
        assert all(np.array_equal(getattr(task.grasp, k), getattr(oracle_grasp, k)) for k in ("palm_offset", "quat_wxyz", "approach_dir"))
    ref = reference_rows("fifty")
    all_trials = fixed_trials(task, N_FIFTY)
    rows = []
    for k in trial_ids:
        t = all_trials[k]
        assert layout_of(t, True) == {f: ref[k][f] for f in LAYOUT_KEYS}, f"trial {k}: layout is not fifty.json's"
        row = trial_row(task, dict(t, handle_yaw_deg=t["handle_yaw_deg"] + delta_deg), True)
        row["grasp_forces"] = row["detail"].pop("grasp_forces")          # as sim/m1_grasp_final.py stores its rows
        rows.append(row)
        print(f"phi {phi_deg} trial {k:2d} -> " + ("SUCCESS" if row["success"] else f"FAIL {row['category']}: {row['reason']}")
              + f" ({row['wall_s']} s)", flush=True)
    rss = peak_rss_mb()
    if rss > RSS_CAP_MB:
        raise MemoryError(f"peak RSS {rss} MB > {RSS_CAP_MB} MB (spec cap)")
    grasp = {k: np.round(getattr(task.grasp, k), 6).tolist() for k in ("palm_offset", "quat_wxyz", "approach_dir")}
    return dict(phi_deg=phi_deg, trials=trial_ids, rows=rows, grasp=grasp, wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=rss)


def run_phis(phis: list[int], workers: int) -> list[dict]:
    """Every (phi, trial) of ``phis`` on ``workers`` processes, in chunks of consecutive trials. Returns the chunk results."""
    per_phi = max(1, workers // len(phis))
    jobs = [(phi, c.tolist()) for phi in phis for c in np.array_split(np.arange(N_SWEEP), per_phi)]
    # fork: the parent has built nothing (no scene, no ONNX session), so every worker starts clean (as sim/m1_grasp_ladder.py)
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(run_chunk, [j[0] for j in jobs], [j[1] for j in jobs]))


def cmd_run(args):
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "runs.json").exists() and not args.overwrite:
        raise SystemExit(f"{args.out / 'runs.json'} exists; pass --overwrite")
    t0 = time.perf_counter()
    parts = run_phis([PHI_ORACLE_DEG], args.workers)
    rows45 = [r for p in parts for r in p["rows"]]
    assert [r["trial"] for r in rows45] == list(range(N_SWEEP))
    fifty = json.loads(FIFTY.read_text())["runs"]
    check = same_rows(rows45, fifty[:N_SWEEP], FIFTY, ref_name="track_b_fifty")
    check.update(what=f"phi = {PHI_ORACLE_DEG} deg (delta = 0) vs Track B final's mug_v1 trials 0-{N_SWEEP - 1}", passed=check["trials_identical"] == N_SWEEP,
                 wall_s=round(time.perf_counter() - t0, 1))
    (args.out / "phi45_check.json").write_text(json.dumps(check, indent=2))
    print(f"phi 45 check: {check['trials_identical']}/{N_SWEEP} identical to fifty.json (differences {check['differences']})")
    if not check["passed"]:
        raise SystemExit("phi = 45 does not reproduce Track B final's rows: STOP (spec), nothing else is run")
    parts += run_phis([p for p in PHIS_DEG if p != PHI_ORACLE_DEG], args.workers)
    rows = []
    for p in parts:
        for r in p["rows"]:
            delta = p["phi_deg"] - PHI_ORACLE_DEG
            rows.append(dict(phi_deg=p["phi_deg"], delta_deg=delta, trial_handle_yaw_deg=round(r["cup_yaw_deg"] - delta, 2), **r))
    rows.sort(key=lambda r: (r["phi_deg"], r["trial"]))
    out = dict(what="M4b Step 3 heading sweep (sim/m4b_heading_sweep.py): Track B final's mug_v1 oracle with its grasp turned about the cup "
                    "axis by phi - 45 deg; handle yaw = trial + (phi - 45); trials 0-19 of trials()",
               phis_deg=list(PHIS_DEG), trials=list(range(N_SWEEP)), mass_kg=MUG_MASS_KG, profile="mug_v1", phi45_check=check["passed"],
               grasps={str(p["phi_deg"]): p["grasp"] for p in parts},
               resources=dict(wall_s=round(time.perf_counter() - t0, 1), workers=args.workers, per_process=[dict(phi_deg=p["phi_deg"], trials=p["trials"],
                              wall_s=p["wall_s"], peak_rss_mb=p["peak_rss_mb"]) for p in parts], gpu="none (CPU only)"),
               runs=rows)
    (args.out / "runs.json").write_text(json.dumps(out, indent=2))
    for phi in PHIS_DEG:
        rs_ = [r for r in rows if r["phi_deg"] == phi]
        print(f"phi {phi}: {sum(r['success'] for r in rs_)}/{len(rs_)}")
    print(f"wrote {args.out / 'runs.json'} ({out['resources']['wall_s']} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overwrite", action="store_true")
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()
