"""M5: the learned grasp selector -- the hoi-sim side (layouts, rollouts, candidate table, selection, evaluation, summary).

Plan (CLAUDE.md "The approach" step 5, M5): per layout, choose which of my clips and which grasp heading phi the robot uses with
the b_hybrid motion (my approach to the pre-grasp point PG, the oracle's final approach, my transport). b_hybrid_phi = b_hybrid
with the grasp heading set to phi (retarget/paths_v2.py hybrid_path, sim/m4_execute.py candidate_phi). Every layout has ONE
handle yaw, whatever the grasp: eval trials their own handle_yaw_deg, training layouts their drawn yaw.

The 50 evaluation layouts (trials 0-49 of sim.m1_pick_mug.trials()) are TEST ONLY: nothing here trains, selects or sets a
threshold on them.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m policy.m5 ...):
  layouts   Step 1: 120 training layouts (sample_layouts(legacy grasp, 120, seed 100); handle yaw = trials()'s approach angle +
            U(-45, 45) deg with default_rng(2100)), disjoint from the eval layouts -> results/m5/train_layouts.json
  step0     Step 0 checks: b_hybrid_phi at phi_h / 45 deg vs b_hybrid / b_hybrid_h45 (paths bit for bit; 2 re-runs each vs the
            saved M4b rows) -> results/m5/step0_check.json
  items     Step 2: the (clip, candidate, training layout) items -> results/m5/rollout_items.json (run them with
            python -m sim.m4_execute run --items results/m5/rollout_items.json --out results/m5/rollouts)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.m1_pick_mug import YAW_SPREAD_DEG, trials
from sim.pick_place import PALM_BOX, Grasp, sample_layouts
from sim.scene import mug_config

OUT = Path("results/m5")
N_TRAIN, TRAIN_SEED, YAW_SEED = 120, 100, 2100            # spec Step 1
DISJOINT_M = 0.001                                         # no training cup / coaster within 1 mm of an eval cup / coaster
CLIP_PHI_H_DEG = None                                      # read from results/m3_retarget/human_values.json (candidates)
HUMAN_VALUES = Path("results/m3_retarget/human_values.json")
FIXED_DEG = (75, 65, 55)                                   # the candidates strictly between phi_h and 45 (plus phi_h and 45)


# ---------------------------------------------------------------------------------------------------------------- Step 1
def eval_layouts() -> list[dict]:
    """trials 0-49 (the test set) as dict(name, cup_xy, coaster_xy, handle_yaw_deg)."""
    legacy = Grasp.hero_side_grasp(mug_config()["height_m"])
    return [dict(name=f"trial{t['trial']:02d}", cup_xy=np.asarray(t["cup_xy"], float), coaster_xy=np.asarray(t["coaster_xy"], float),
                 handle_yaw_deg=float(t["handle_yaw_deg"])) for t in trials(legacy)]


def cmd_layouts(args):
    legacy = Grasp.hero_side_grasp(mug_config()["height_m"])
    # trials() draws the yaw around ITS grasp's approach angle (the legacy grasp's); same here, so the distribution is trials()'s
    approach_deg = math.degrees(math.atan2(legacy.approach_dir[1], legacy.approach_dir[0]))
    ev = eval_layouts()
    ev_pts = np.array([p for e in ev for p in (e["cup_xy"], e["coaster_xy"])])
    near = lambda p: float(np.linalg.norm(ev_pts - p, axis=1).min())
    # The spec's assert (no training cup or coaster within 1 mm of any eval cup or coaster) fails on the first 120 draws of seed 100:
    # with ~24 000 point pairs in a 10 x 20 cm box, about 4 pairs closer than 1 mm are expected by chance. Deviation (logged): the
    # draws that break it are dropped and the SAME seeded sampler stream continues until N_TRAIN layouts are left (sample_layouts
    # with a larger n returns the same first draws, asserted), so the training set is disjoint as the spec wants and nothing
    # depends on any outcome. The yaw is then drawn as specified, N_TRAIN values in order.
    n_draw = N_TRAIN
    while True:
        pairs = sample_layouts(legacy, n_draw, TRAIN_SEED)
        kept = [(i, cup, coaster) for i, (cup, coaster) in enumerate(pairs) if min(near(cup), near(coaster)) > DISJOINT_M]
        if len(kept) >= N_TRAIN:
            break
        n_draw += 1
    first = sample_layouts(legacy, N_TRAIN, TRAIN_SEED)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(first, pairs)), "sampler prefix changed"
    dropped = [dict(draw=i, closest_mm=round(min(near(c), near(k)) * 1e3, 3)) for i, (c, k) in enumerate(pairs) if min(near(c), near(k)) <= DISJOINT_M]
    kept = kept[:N_TRAIN]
    spread = np.random.default_rng(YAW_SEED).uniform(-YAW_SPREAD_DEG, YAW_SPREAD_DEG, size=N_TRAIN)
    lays, closest = [], np.inf
    for j, ((i, cup, coaster), d) in enumerate(zip(kept, spread)):
        closest = min(closest, near(cup), near(coaster))
        assert min(near(cup), near(coaster)) > DISJOINT_M
        lays.append(dict(name=f"train{j:03d}", index=j, sampler_draw=i, layout_seed=TRAIN_SEED, cup_xy=[float(v) for v in cup],
                         coaster_xy=[float(v) for v in coaster], handle_yaw_deg=approach_deg + float(d)))
    out = dict(what="M5 Step 1 training layouts (policy/m5.py layouts)", sampler=f"sim.pick_place.sample_layouts(legacy grasp, {N_TRAIN}, seed={TRAIN_SEED})",
               palm_box=PALM_BOX, yaw=f"legacy grasp approach angle {approach_deg:.4f} deg + U(-{YAW_SPREAD_DEG}, {YAW_SPREAD_DEG}) deg, "
                                     f"np.random.default_rng({YAW_SEED}) (trials()'s distribution)",
               disjoint=dict(rule=f"every training cup and coaster > {DISJOINT_M * 1e3:.0f} mm from every eval (trials 0-49) cup and coaster",
                             closest_mm=round(closest * 1e3, 3), passed=True, draws=n_draw, dropped_draws=dropped,
                             deviation="the first 120 draws break the rule (chance near-coincidences); the breaking draws are dropped and "
                                       "the same sampler stream continues (policy/m5.py cmd_layouts)"),
               layouts=lays)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_layouts.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out / 'train_layouts.json'}: {len(lays)} layouts, closest point to an eval cup / coaster {closest * 1e3:.1f} mm")


# ---------------------------------------------------------------------------------------------------------------- Step 0
STEP0_LAYOUTS = ("trial00", "trial01")
STEP0_CLIP = "26d4118b0c"
# fields that name the run rather than describe it: the run name, the variant, M5's candidate fields, and in detail.path.info the
# variant, phi_rad and b_hybrid's own self-check record (computed only when the variant is b_hybrid)
IDENTITY = ("run", "variant", "candidate", "phi_deg", "phi_h_deg", "departure_deg", "yaw", "path_xy", "tag", "train_index")
IDENTITY_INFO = ("variant", "phi_rad", "check_against_b_level")


def _path_compare(clip: str, layout: str) -> dict:
    """b_hybrid_phi's path at phi_h and at 45 deg vs b_hybrid's and b_hybrid_h45's, bit for bit, from one standing palm."""
    from retarget import paths_v2
    from retarget.warp import standing_palm
    from sim.m4_execute import candidate_phi, clip_context, variant_layout, worker_task

    task, ctx = worker_task(), clip_context(clip)
    out = {}
    for cand, ref_variant, yaw in (("phih", "b_hybrid", "m4"), ("45", "b_hybrid_h45", "own")):
        lay_ref = variant_layout(ref_variant, ctx["layouts"][layout])
        lay_phi = variant_layout(paths_v2.PHI_VARIANT, ctx["layouts"][layout], yaw)
        assert lay_ref["handle_yaw"] == lay_phi["handle_yaw"]
        p_s, q_now, _ = standing_palm(task, lay_ref)
        a = paths_v2.hybrid_path(clip, ref_variant, lay_ref, p_s, q_now, oracle=ctx["oracle"])
        b = paths_v2.hybrid_path(clip, paths_v2.PHI_VARIANT, lay_phi, p_s, q_now, oracle=ctx["oracle"], phi=candidate_phi(cand, ctx))
        arrays = all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ("palm_pos", "palm_quat_wxyz", "closure", "phase"))
        idx = all(a[k] == b[k] for k in a if k.startswith("i_")) and a["follow"] == b["follow"]
        ia = {k: v for k, v in a["info"].items() if k not in IDENTITY_INFO}
        ib = {k: v for k, v in b["info"].items() if k not in IDENTITY_INFO}
        info = json.dumps(ia, default=str, sort_keys=True) == json.dumps(ib, default=str, sort_keys=True)
        out[f"{cand}_vs_{ref_variant}"] = dict(arrays_identical=arrays, indices_identical=idx, info_identical=info, identical=arrays and idx and info)
    return dict(clip=clip, layout=layout, **out)


def _rerun_compare(clip: str, cand: str, ref_variant: str, yaw: str, layout: str) -> dict:
    """One b_hybrid_phi re-run vs the saved M4b row of ``ref_variant`` (row_differences, identity fields left out)."""
    from sim.m4_execute import PHI, row_differences, run_name, run_one

    saved_path = Path("results/m4b/runs") / f"{run_name(clip, ref_variant, layout)}.json"
    saved = json.loads(saved_path.read_text())
    row = json.loads(json.dumps(run_one(clip, PHI, layout, candidate=cand, yaw=yaw, diagnostics=tuple(saved["diagnostics_run"]))))
    strip = lambda r: {k: v for k, v in r.items() if k not in IDENTITY} | {"detail": {**r["detail"], "path": {**r["detail"]["path"], "info": {
        k: v for k, v in r["detail"]["path"]["info"].items() if k not in IDENTITY_INFO}}}}
    diff = row_differences(strip(row), strip(saved))
    return dict(run=row["run"], saved_row=str(saved_path), handle_yaw_deg=row["handle_yaw_deg"], saved_handle_yaw_deg=saved["handle_yaw_deg"],
                success=row["success"], saved_success=saved["success"], fields_differing=diff, identical=not diff)


def cmd_step0(args):
    t0 = time.perf_counter()
    jobs = [(STEP0_CLIP, c, v, y, l) for c, v, y in (("phih", "b_hybrid", "m4"), ("45", "b_hybrid_h45", "own")) for l in STEP0_LAYOUTS]
    with ProcessPoolExecutor(len(jobs) + 1) as pool:
        paths_f = [pool.submit(_path_compare, clip, lay) for clip in ("26d4118b0c", "4e103843db", "49aff04f7a", "e8825dea53") for lay in ("trial00",)]
        reruns_f = [pool.submit(_rerun_compare, *j) for j in jobs]
        paths, reruns = [f.result() for f in paths_f], [f.result() for f in reruns_f]
    out = dict(what="M5 Step 0: b_hybrid_phi reproduces b_hybrid (phi = phi_h) and b_hybrid_h45 (phi = 45 deg = phi_ora)",
               candidate_45="phi_ora exactly (0.7853981633974484 rad = 45 deg + 1 ulp), so that it is b_hybrid_h45's R_ora bit for bit",
               identity_fields_left_out=dict(row=list(IDENTITY), path_info=list(IDENTITY_INFO)),
               paths=paths, reruns=reruns, passed=all(p[k]["identical"] for p in paths for k in p if k.endswith(("b_hybrid", "h45"))) and all(r["identical"] for r in reruns),
               wall_s=round(time.perf_counter() - t0, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "step0_check.json").write_text(json.dumps(out, indent=2))
    for p in paths:
        print(p)
    for r in reruns:
        print(f"{r['run']}: identical {r['identical']} (success {r['success']} / saved {r['saved_success']}, yaw {r['handle_yaw_deg']} / {r['saved_handle_yaw_deg']})"
              + (f" differs {r['fields_differing']}" if r["fields_differing"] else ""))
    print(f"Step 0 passed: {out['passed']} ({out['wall_s']} s)")
    if not out["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------- Step 2
def candidates() -> dict[str, list[str]]:
    """Per clip: phi_h, every value of FIXED_DEG strictly between phi_h and 45, and 45 (labels for sim.m4_execute.candidate_phi)."""
    hv = json.loads(HUMAN_VALUES.read_text())["clips"]
    out = {}
    for clip, v in hv.items():
        phi_h = v["phi_h_deg"]
        out[clip] = ["phih"] + [str(d) for d in FIXED_DEG if 45 < d < phi_h] + ["45"]
    return out


def cmd_items(args):
    lays = json.loads((args.out / "train_layouts.json").read_text())["layouts"][:args.n_layouts]
    cand = candidates()
    items = [dict(clip=c, variant="b_hybrid_phi", layout=l["name"], candidate=k) for l in lays for c in cand for k in cand[c]]
    (args.out / "rollout_items.json").write_text(json.dumps(items))
    print(f"{len(items)} items ({len(lays)} layouts x {sum(map(len, cand.values()))} candidates: {cand}) -> {args.out / 'rollout_items.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("layouts", "step0"):
        sub.add_parser(name).add_argument("--out", type=Path, default=OUT)
    it = sub.add_parser("items")
    it.add_argument("--out", type=Path, default=OUT)
    it.add_argument("--n-layouts", type=int, default=N_TRAIN)
    args = ap.parse_args()
    dict(layouts=cmd_layouts, step0=cmd_step0, items=cmd_items)[args.cmd](args)


if __name__ == "__main__":
    main()
