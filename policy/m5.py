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
  candidates  Branch D1 + the training data: results/m5/candidates.md / .json (success per candidate and per clip x phi, the
            candidate-oracle ceiling, the fixed-best candidate F) and results/m5/train_data.npz for policy/train_selector.py
  onnx-check  onnxruntime vs torch on the 1000 saved feature rows (|dp| <= 1e-5) -> results/m5/onnx_check.json
  select    Step 4: the selector's choices on the 50 test layouts from NOISY inputs (seed 2026) -> results/m5/choices.json and
            results/m5/eval_items.json (clean runs of R1 / R2 / F / b_hybrid at the original yaw; noisy-path runs of R2 / F /
            b_hybrid_h45 of the best 45-deg clip). Run them with sim.m4_execute run --items ... --out results/m5/eval
  reproduce-r2  for scripts/reproduce.sh: R2's choice on one test trial recomputed from selector.onnx (same noise draw) and
            checked against choices.json, then that run re-run and compared with its saved eval row (exit 1 if anything differs)
  summarize Step 5 -> results/m5/summary.md + summary.json (ablation table, paired comparisons, R2's choices, CV and candidate
            tables, branches D2 / D3, confounds, video picks); reads only saved files
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


# ---------------------------------------------------------------------------------------------------------------- D1 + data
ROLLOUTS = OUT / "rollouts" / "runs"
ERROR_STOP = 0.05                                          # D1: STOP if more than 5 % of the runs raised exceptions


def phis() -> tuple[dict, float]:
    """(clip -> phi_h [rad], phi_ora [rad]): the values sim.m4_execute.candidate_phi uses (human_values.json = human_at_grasp's
    phi_h; phi_ora = oracle_reference's heading, which does not depend on the mug radius)."""
    from retarget.grasp import oracle_reference
    hv = json.loads(HUMAN_VALUES.read_text())["clips"]
    return {c: float(v["phi_h_rad"]) for c, v in hv.items()}, float(oracle_reference(None)["phi_ora"])


def table() -> list[tuple[str, str]]:
    """The 18 candidates (clip, label) in table order: clip order, then phi_h, 75, 65, 55, 45."""
    from policy.features import CLIPS
    cand = candidates()
    return [(c, k) for c in CLIPS for k in cand[c]]


def phi_of(clip: str, label: str, phi_h: dict, phi_ora: float) -> float:
    from sim.m4_execute import candidate_phi
    return candidate_phi(label, dict(human=dict(phi_h=phi_h[clip]), oracle=dict(phi_ora=phi_ora)))


def wilson_str(k: int, n: int) -> tuple[str, list[float]]:
    from sim.m1_pick_mug import wilson
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo:.2f}-{hi:.2f})", [round(lo, 3), round(hi, 3)]


def cmd_candidates(args):
    from collections import Counter
    from policy.features import CLIPS
    from retarget.m4b_summary import category_v2
    from sim.m4_execute import run_name

    lays = json.loads((args.out / "train_layouts.json").read_text())["layouts"]
    rows = [json.loads(p.read_text()) for p in sorted(ROLLOUTS.glob("*.json"))]
    errors = [r["run"] for r in rows if r.get("success") is None]
    used = sorted({r["layout"] for r in rows})
    lays = [l for l in lays if l["name"] in used]
    tab = table()
    expected = {run_name(c, "b_hybrid_phi", l["name"], k) for l in lays for c, k in tab}
    by_run = {r["run"]: r for r in rows}
    assert set(by_run) == expected, f"rollouts: missing {sorted(expected - set(by_run))[:5]}, extra {sorted(set(by_run) - expected)[:5]}"
    d1 = dict(runs=len(rows), exceptions=len(errors), exception_runs=errors, fraction=len(errors) / len(rows), stop_above=ERROR_STOP,
              branch="STOP" if len(errors) / len(rows) > ERROR_STOP else "continue")
    if d1["branch"] == "STOP":
        (args.out / "candidates.json").write_text(json.dumps(dict(d1=d1), indent=2))
        raise SystemExit(f"D1: {len(errors)}/{len(rows)} runs raised exceptions (> {ERROR_STOP:.0%}): STOP")
    phi_h, phi_ora = phis()
    cand_rows = []
    for c, k in tab:
        rs_ = [by_run[run_name(c, "b_hybrid_phi", l["name"], k)] for l in lays]
        phi = phi_of(c, k, phi_h, phi_ora)
        assert all(abs(r["phi_deg"] - math.degrees(phi)) < 1e-9 for r in rs_)
        ok = sum(bool(r["success"]) for r in rs_)
        txt, ci = wilson_str(ok, len(rs_))
        cand_rows.append(dict(clip=c, candidate=k, phi_deg=math.degrees(phi), departure_deg=math.degrees(phi_h[c] - phi), successes=ok, n=len(rs_),
                              rate=ok / len(rs_), wilson95=ci, text=txt, category_v2=dict(Counter(category_v2(r) for r in rs_ if not r["success"]).most_common())))
    from policy.rules import fixed_best
    F = fixed_best([x["rate"] for x in cand_rows], [x["phi_deg"] for x in cand_rows])
    any_ok = [any(by_run[run_name(c, "b_hybrid_phi", l["name"], k)]["success"] for c, k in tab) for l in lays]
    ceil_txt, ceil_ci = wilson_str(sum(any_ok), len(lays))
    # training data for policy/train_selector.py: exact layout values from train_layouts.json, phi from candidate_phi's formulas
    rec = dict(layout_index=[], clip_index=[], candidate=[], phi=[], phi_h=[], cup_xy=[], coaster_xy=[], yaw=[], success=[])
    for l in lays:
        for c, k in tab:
            r = by_run[run_name(c, "b_hybrid_phi", l["name"], k)]
            assert r["cup_xy"] == [round(v, 4) for v in l["cup_xy"]] and r["coaster_xy"] == [round(v, 4) for v in l["coaster_xy"]]
            assert abs(r["handle_yaw_deg"] - round(l["handle_yaw_deg"], 2)) < 1e-9
            for key, v in (("layout_index", l["index"]), ("clip_index", CLIPS.index(c)), ("candidate", k), ("phi", phi_of(c, k, phi_h, phi_ora)),
                           ("phi_h", phi_h[c]), ("cup_xy", l["cup_xy"]), ("coaster_xy", l["coaster_xy"]), ("yaw", math.radians(l["handle_yaw_deg"])),
                           ("success", int(bool(r["success"])))):
                rec[key].append(v)
    np.savez(args.out / "train_data.npz", **{k: np.asarray(v) for k, v in rec.items()})
    out = dict(what="M5 Step 2 candidate rollouts on the training layouts (policy/m5.py candidates)", d1=d1, layouts=len(lays),
               candidates=cand_rows, ceiling=dict(layouts_with_a_success=sum(any_ok), n=len(lays), wilson95=ceil_ci, text=ceil_txt),
               fixed_best=dict(clip=cand_rows[F]["clip"], candidate=cand_rows[F]["candidate"], phi_deg=cand_rows[F]["phi_deg"], text=cand_rows[F]["text"],
                               rule="highest training success; ties -> larger phi; then table order"),
               resources=[json.loads(p.read_text()) for p in sorted((args.out / "rollouts").glob("run_resources_*.json"))])
    (args.out / "candidates.json").write_text(json.dumps(out, indent=2))
    L = ["# M5 candidate rollouts (training layouts only)", "", f"`python -m policy.m5 candidates`; {len(lays)} training layouts x {len(tab)} candidates = {len(rows)} runs; "
         f"exceptions {len(errors)} ({d1['fraction']:.1%}; D1 stops above 5 %): branch **{d1['branch']}**.", "",
         "b_hybrid_phi = b_hybrid (my approach to PG, the oracle's final approach, my transport) with the grasp heading phi. Each layout has one handle yaw.", "",
         "| clip | candidate | phi [deg] | departure from my heading [deg] | success (Wilson 95 %) | failures (category_v2) |", "|---|---|---|---|---|---|"]
    for x in cand_rows:
        L.append(f"| {x['clip']} | {x['candidate']} | {x['phi_deg']:.1f} | {x['departure_deg']:.1f} | {x['text']} | " + (", ".join(f"{k} {v}" for k, v in x["category_v2"].items()) or "-") + " |")
    L += ["", "Success per clip x phi (rows: clips; columns: phi):", "", "| clip | phi_h | 75 | 65 | 55 | 45 |", "|---|---|---|---|---|---|"]
    for c in CLIPS:
        cells = {x["candidate"]: f"{x['successes']}/{x['n']}" for x in cand_rows if x["clip"] == c}
        L.append(f"| {c} ({math.degrees(phi_h[c]):.1f}) | " + " | ".join(cells.get(k, "-") for k in ("phih", "75", "65", "55", "45")) + " |")
    L += ["", f"Candidate-oracle ceiling (at least one candidate succeeds): **{ceil_txt}** of the training layouts.", "",
          f"Fixed-best candidate F (highest training success, ties -> larger phi): **{out['fixed_best']['clip']} {out['fixed_best']['candidate']}** "
          f"(phi {out['fixed_best']['phi_deg']:.1f} deg), {out['fixed_best']['text']}."]
    (args.out / "candidates.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_onnx_check(args):
    import onnxruntime as ort
    d = np.load(args.out / "onnx_check_rows.npz")
    sess = ort.InferenceSession(str(args.out / "selector.onnx"), providers=["CPUExecutionProvider"])
    p = sess.run(["p"], {"features": d["features"]})[0][:, 0]
    diff = float(np.abs(p - d["p_torch"]).max())
    out = dict(rows=int(len(p)), max_abs_diff=diff, tolerance=1e-5, passed=diff <= 1e-5, onnxruntime=ort.__version__, numpy=np.__version__)
    (args.out / "onnx_check.json").write_text(json.dumps(out, indent=2))
    print(out)
    if not out["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------- Step 4
EVAL_NOISE_SEED = 2026


def selector_p(sess, clips_idx, labels, cup, coaster, yaw, phi_h: dict, phi_ora: float) -> np.ndarray:
    """p for every candidate of ONE layout (inputs as the selector sees them)."""
    from policy.features import CLIPS, features
    tab = table()
    X = features(np.repeat(cup[None], len(tab), 0), np.repeat(coaster[None], len(tab), 0), np.full(len(tab), yaw), [CLIPS.index(c) for c, _ in tab],
                 [phi_of(c, k, phi_h, phi_ora) for c, k in tab], [phi_h[c] for c, _ in tab])
    return sess.run(["p"], {"features": X.astype(np.float32)})[0][:, 0].astype(float)


def cmd_select(args):
    import onnxruntime as ort
    from policy.features import noisy
    from policy.rules import r1, r2

    sess = ort.InferenceSession(str(args.out / "selector.onnx"), providers=["CPUExecutionProvider"])
    cj = json.loads((args.out / "candidates.json").read_text())
    tab = table()
    phi_h, phi_ora = phis()
    dep = [math.degrees(phi_h[c] - phi_of(c, k, phi_h, phi_ora)) for c, k in tab]
    F = next(j for j, (c, k) in enumerate(tab) if (c, k) == (cj["fixed_best"]["clip"], cj["fixed_best"]["candidate"]))
    h45 = max((x for x in cj["candidates"] if x["candidate"] == "45"), key=lambda x: (x["rate"], -[c for c, _ in tab].index(x["clip"])))
    ev = eval_layouts()
    cups, coasters = np.array([e["cup_xy"] for e in ev]), np.array([e["coaster_xy"] for e in ev])
    yaws = np.radians([e["handle_yaw_deg"] for e in ev])
    ncup, ncoast, nyaw = noisy(np.random.default_rng(EVAL_NOISE_SEED), cups, coasters, yaws)      # one draw per trial, trial order
    choices, clean, noisy_items = [], {}, {}
    item = lambda c, k, lay, **kw: dict(clip=c, variant="b_hybrid_phi", layout=lay, candidate=k, **kw)
    for t, e in enumerate(ev):
        p = selector_p(sess, None, None, ncup[t], ncoast[t], nyaw[t], phi_h, phi_ora)
        pick = dict(R1=r1(p), R2=r2(p, dep), F=F)
        choices.append(dict(trial=t, layout=e["name"], noisy_inputs=dict(cup_xy=ncup[t].tolist(), coaster_xy=ncoast[t].tolist(), handle_yaw_deg=math.degrees(nyaw[t])),
                            p={f"{c}/{k}": round(float(v), 5) for (c, k), v in zip(tab, p)},
                            **{rule: dict(clip=tab[j][0], candidate=tab[j][1], phi_deg=math.degrees(phi_of(*tab[j], phi_h, phi_ora)), departure_deg=dep[j],
                                          p=float(p[j])) for rule, j in pick.items()}))
        for c in phi_h:                                    # baseline: b_hybrid (phi_h) of every clip at the original yaw
            clean[(c, "phih", e["name"])] = item(c, "phih", e["name"])
        for rule in ("R1", "R2", "F"):
            c, k = tab[pick[rule]]
            clean[(c, k, e["name"])] = item(c, k, e["name"])
        path_xy = dict(cup_xy=ncup[t].tolist(), coaster_xy=ncoast[t].tolist())
        for c, k in (tab[pick["R2"]], tab[F], (h45["clip"], "45")):
            noisy_items[(c, k, e["name"])] = item(c, k, e["name"], path_xy=path_xy, tag="noisy")
    items = list(clean.values()) + list(noisy_items.values())
    out = dict(what="M5 Step 4 selector choices on trials 0-49 (policy/m5.py select)", noise=f"one draw per trial, default_rng({EVAL_NOISE_SEED}), "
               "cup / coaster xy N(0, 1 cm) per axis, yaw N(0, 10 deg) (policy/features.py noisy); the same noisy layout feeds the selector and the noisy-path runs",
               F=dict(clip=tab[F][0], candidate=tab[F][1]), h45_best_clip=h45["clip"], h45_rule="the 45-deg candidate with the highest training success",
               clean_runs=len(clean), noisy_runs=len(noisy_items), choices=choices)
    (args.out / "choices.json").write_text(json.dumps(out, indent=2))
    (args.out / "eval_items.json").write_text(json.dumps(items))
    print(f"{len(clean)} clean + {len(noisy_items)} noisy runs -> {args.out / 'eval_items.json'}; F = {tab[F]}, h45 clip {h45['clip']}")


def cmd_reproduce_r2(args):
    import onnxruntime as ort
    from policy.features import noisy
    from policy.rules import r2
    from sim.m4_execute import row_differences, run_name, run_one

    sess = ort.InferenceSession(str(args.out / "selector.onnx"), providers=["CPUExecutionProvider"])
    tab = table()
    phi_h, phi_ora = phis()
    dep = [math.degrees(phi_h[c] - phi_of(c, k, phi_h, phi_ora)) for c, k in tab]
    ev = eval_layouts()
    ncup, ncoast, nyaw = noisy(np.random.default_rng(EVAL_NOISE_SEED), np.array([e["cup_xy"] for e in ev]), np.array([e["coaster_xy"] for e in ev]),
                               np.radians([e["handle_yaw_deg"] for e in ev]))
    t = args.trial
    c, k = tab[r2(selector_p(sess, None, None, ncup[t], ncoast[t], nyaw[t], phi_h, phi_ora), dep)]
    saved_choice = json.loads((args.out / "choices.json").read_text())["choices"][t]["R2"]
    same_choice = (c, k) == (saved_choice["clip"], saved_choice["candidate"])
    saved_path = EVAL / f"{run_name(c, 'b_hybrid_phi', f'trial{t:02d}', k)}.json"
    saved = json.loads(saved_path.read_text())
    again = json.loads(json.dumps(run_one(c, "b_hybrid_phi", f"trial{t:02d}", candidate=k, diagnostics=tuple(saved["diagnostics_run"]))))
    diff = row_differences(again, saved)
    print(f"R2 trial {t}: choice {c} {k} matches choices.json: {same_choice}; re-run matches {saved_path}: {not diff}"
          + (f" (differs in {diff})" if diff else ""))
    if not (same_choice and not diff):
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------- Step 5
EVAL = OUT / "eval" / "runs"
D2_MIN_AUC = 0.6
D3_MIN = 30


def _rate(rows: list[dict]) -> dict:
    from collections import Counter
    from retarget.m4b_summary import category_v2
    k = sum(bool(r["success"]) for r in rows)
    txt, ci = wilson_str(k, len(rows))
    return dict(successes=k, n=len(rows), wilson95=ci, text=txt,
                category_v2=dict(Counter(category_v2(r) for r in rows if not r["success"]).most_common()))


def _paired(a: list[dict], b: list[dict]) -> dict:
    for x, y in zip(a, b):
        assert x["trial"] == y["trial"] and x["cup_xy"] == y["cup_xy"] and x["coaster_xy"] == y["coaster_xy"]
    sa, sb = [bool(x["success"]) for x in a], [bool(y["success"]) for y in b]
    return dict(both=sum(x and y for x, y in zip(sa, sb)), first_only=sum(x and not y for x, y in zip(sa, sb)),
                second_only=sum(y and not x for x, y in zip(sa, sb)), neither=sum(not x and not y for x, y in zip(sa, sb)))


def cmd_summarize(args):
    from collections import Counter
    from retarget.m4b_summary import category_v2
    from sim.m4_execute import FIFTY, run_name

    load = lambda d: {r["run"]: r for r in (json.loads(p.read_text()) for p in sorted(d.glob("*.json")))}
    ev_rows, m4, m4b = load(EVAL), load(Path("results/m4_execute/runs")), load(Path("results/m4b/runs"))
    assert all(r.get("success") is not None for r in ev_rows.values()), "an eval run raised an exception"
    oracle = json.loads(FIFTY.read_text())["runs"]
    ch = json.loads((args.out / "choices.json").read_text())
    cv = json.loads((args.out / "cv.json").read_text())
    cj = json.loads((args.out / "candidates.json").read_text())
    clips = list(candidates())
    trials_ = [f"trial{t:02d}" for t in range(50)]
    def pick(rule: str, noisy: bool = False) -> list[dict]:
        """The eval row of ``rule``'s chosen candidate on each trial."""
        return [ev_rows[run_name(c[rule]["clip"], "b_hybrid_phi", trials_[t], c[rule]["candidate"], "noisy" if noisy else None)]
                for t, c in enumerate(ch["choices"])]
    h45c = ch["h45_best_clip"]
    groups = dict(
        naive_M4=[m4[run_name(c, "naive", t)] for c in clips for t in trials_],
        b_level_M4=[m4[run_name(c, "b_level", t)] for c in clips for t in trials_],
        b_hybrid_original_yaw=[ev_rows[run_name(c, "b_hybrid_phi", t, "phih")] for c in clips for t in trials_],
        b_hybrid_h45_M4b=[m4b[run_name(c, "b_hybrid_h45", t)] for c in clips for t in trials_],
        **{f"b_hybrid_h45_M4b_{h45c}": [m4b[run_name(h45c, "b_hybrid_h45", t)] for t in trials_]},
        R1_clean=pick("R1"), R2_clean=pick("R2"), F_clean=pick("F"),
        R2_noisy=pick("R2", True), F_noisy=pick("F", True),
        **{f"b_hybrid_h45_{h45c}_noisy": [ev_rows[run_name(h45c, "b_hybrid_phi", t, "45", "noisy")] for t in trials_]},
        oracle=oracle)
    abl = {k: _rate(v) for k, v in groups.items()}
    per_clip = {v: {c: _rate([r for r in groups[v] if r.get("clip") == c]) for c in clips} for v in ("naive_M4", "b_level_M4", "b_hybrid_original_yaw", "b_hybrid_h45_M4b")}
    paired = {"R2_vs_F": _paired(groups["R2_clean"], groups["F_clean"]), "R2_vs_oracle": _paired(groups["R2_clean"], oracle),
              f"R2_vs_b_hybrid_h45_{h45c}": _paired(groups["R2_clean"], groups[f"b_hybrid_h45_M4b_{h45c}"]),
              "R1_vs_F": _paired(groups["R1_clean"], groups["F_clean"])}
    r2 = [c["R2"] for c in ch["choices"]]
    hist = Counter(f"{x['clip']} {x['candidate']}" for x in r2)
    r2_choice = dict(histogram=dict(sorted(hist.items())), mean_departure_deg=float(np.mean([x["departure_deg"] for x in r2])),
                     max_departure_deg=float(max(x["departure_deg"] for x in r2)), chose_my_heading=sum(x["candidate"] == "phih" for x in r2),
                     clips=dict(Counter(x["clip"] for x in r2)), fell_back_to_R1=sum(x["p"] < 0.8 for x in r2))
    d2 = dict(auc=cv["auc"], auc_below_0_6=cv["auc"] < D2_MIN_AUC, R1_cv=cv["selection"]["R1"], F_cv=cv["selection"]["F"],
              R1_not_better_than_F=cv["selection"]["R1"] <= cv["selection"]["F"])
    d2["notes"] = ([f"the selector does not predict success better than chance (CV AUC {cv['auc']:.3f} < 0.6): R1 / R2 are unreliable"] if d2["auc_below_0_6"] else []) + \
                  ([f"learning does not beat the fixed best choice in CV (R1 {d2['R1_cv']} <= F {d2['F_cv']})"] if d2["R1_not_better_than_F"] else [])
    low = [k for k in ("R1_clean", "R2_clean", "F_clean") if abl[k]["successes"] < D3_MIN]
    what_failed = {}                                       # D3: failures per chosen candidate, by category_v2
    for k in low:
        rule, counts = k.split("_")[0], {}
        for row, c in zip(groups[k], ch["choices"]):
            if not row["success"]:
                counts.setdefault(f"{c[rule]['clip']} {c[rule]['candidate']}", Counter())[category_v2(row)] += 1
        what_failed[k] = {name: dict(cnt) for name, cnt in counts.items()}
    r2_succ = [r["run"] for r in groups["R2_clean"] if r["success"]]
    r2_fail = [r["run"] for r in groups["R2_clean"] if not r["success"]]
    choice_clips = {rule: dict(Counter(c[rule]["clip"] for c in ch["choices"]).most_common()) for rule in ("R1", "R2", "F")}
    choice_hist = {rule: dict(sorted(Counter(f"{c[rule]['clip']} {c[rule]['candidate']}" for c in ch["choices"]).items())) for rule in ("R1", "R2", "F")}
    # R2 by the clip it chose: on those layouts, how R2, F and the oracle did, and R2's wins over the oracle and over F
    r2_by_clip = {}
    for clip in sorted({c["R2"]["clip"] for c in ch["choices"]}):
        ts = [t for t, c in enumerate(ch["choices"]) if c["R2"]["clip"] == clip]
        r2s, fs, os_ = ([bool(groups[g][t]["success"]) for t in ts] for g in ("R2_clean", "F_clean", "oracle"))
        r2_by_clip[clip] = dict(layouts=len(ts), candidates=dict(sorted(Counter(ch["choices"][t]["R2"]["candidate"] for t in ts).items())),
                                R2_successes=sum(r2s), F_successes=sum(fs), oracle_successes=sum(os_),
                                R2_wins_over_oracle=[t for t, a, b in zip(ts, r2s, os_) if a and not b], R2_wins_over_F=[t for t, a, b in zip(ts, r2s, fs) if a and not b])
    summary = dict(what="M5 Step 5 summary (policy/m5.py summarize)", ablation=abl, per_clip=per_clip, paired=paired, r2_choices=r2_choice,
                   choice_clips=choice_clips, choice_histogram=choice_hist, r2_by_clip=r2_by_clip,
                   cv=dict(auc=cv["auc"], brier=cv["brier"], base_rate=cv["base_rate"], selection=cv["selection"], n_layouts=cv["n_layouts"]),
                   candidates=dict(rows=[{k: x[k] for k in ("clip", "candidate", "phi_deg", "departure_deg", "text", "rate")} for x in cj["candidates"]],
                                   ceiling=cj["ceiling"], fixed_best=cj["fixed_best"]),
                   branches=dict(D1=cj["d1"]["branch"], D2=d2, D3=dict(below_30=low, what_failed=what_failed)),
                   h45_best_clip=h45c, F=ch["F"],
                   video_picks=dict(r2_success=r2_succ[0] if r2_succ else None, r2_failure=r2_fail[0] if r2_fail else None),
                   confounds=["the 4-clip pooled rows (naive, b_level, b_hybrid, b_hybrid_h45) reuse the same 50 layouts 4 times, so their Wilson "
                              "intervals (which treat the 200 runs as independent) are too narrow",
                              "the place correction reads privileged simulator state (the cup's true pose in the hand) in every variant, the "
                              "selector rules and the oracle alike; in the noisy runs only the PATH is built from the noisy layout's cup and "
                              "coaster xy (the noisy handle yaw feeds only the selector's features; the scene keeps the true layout)",
                              "the 50 eval layouts were also used in M4 / M4b (to compare variants), but never to train, select or set a "
                              "threshold here: the selector, F and the h45 clip come from the 120 training layouts only. The candidate SET was "
                              "designed from results on these layouts, though: the b_hybrid motion from M4's failures on trials 0-49, the 45-75 "
                              "deg heading grid from the M4b sweep on trials 0-19",
                              "'F, noisy path' and 'b_hybrid_h45 4e103843db noisy' are the same 50 runs (F is 4e103843db at 45 deg)",
                              "b_hybrid_h45 (M4b rows) and b_hybrid at the original yaw use the trial's own handle yaw, as do the selector runs"],
                   videos=json.loads((OUT / "videos" / "videos.json").read_text()) if (OUT / "videos" / "videos.json").exists() else None)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text(render_summary(summary))
    print(render_summary(summary))


LABELS = dict(naive_M4="(a) naive kinematic replay (M4), 4 clips", b_level_M4="b_level: my full approach (M4), 4 clips",
              b_hybrid_original_yaw="b_hybrid: my heading + robot final approach, 4 clips", b_hybrid_h45_M4b="b_hybrid_h45: 45 deg heading (M4b), 4 clips",
              R1_clean="(c) selector R1 (argmax p), clean", R2_clean="(c) selector R2 (human-first), clean", F_clean="fixed-best candidate F, clean",
              R2_noisy="selector R2, noisy path", F_noisy="F, noisy path", oracle="scripted oracle (Track B final)")


def render_summary(s: dict) -> str:
    unreliable = s["branches"]["D2"]["auc_below_0_6"]
    lab = lambda k: LABELS.get(k, k.replace("_", " ")) + (" [unreliable: CV AUC < 0.6]" if unreliable and k.startswith(("R1", "R2")) else "")
    L = ["# M5 results: learned grasp selector", "", "Generated by `python -m policy.m5 summarize` from saved rows only. Test set = trials 0-49 (each trial's own "
         "handle yaw), never used for training, selection or thresholds. Wilson 95 % intervals.", "", "## Ablation (success / N)", "",
         "| variant | success (Wilson 95 %) | failures (category_v2) |", "|---|---|---|"]
    for k, v in s["ablation"].items():
        L.append(f"| {lab(k)} | {v['text']} | " + (", ".join(f"{a} {b}" for a, b in v["category_v2"].items()) or "-") + " |")
    L += ["", "Per clip (/50): " + "; ".join(f"{v}: " + ", ".join(f"{c[:5]} {x['successes']}" for c, x in d.items()) for v, d in s["per_clip"].items()), "",
          "## Paired comparisons (same 50 trials)", "", "| comparison | both | first only | second only | neither |", "|---|---|---|---|---|"]
    for k, v in s["paired"].items():
        L.append(f"| {k.replace('_vs_', ' vs ')} | {v['both']} | {v['first_only']} | {v['second_only']} | {v['neither']} |")
    c = s["r2_choices"]
    L += ["", "## R2's choices (from noisy inputs)", "", f"Mean departure from my heading {c['mean_departure_deg']:.1f} deg (max {c['max_departure_deg']:.1f}); "
          f"my own heading chosen in {c['chose_my_heading']}/50 trials; no candidate with p >= 0.8 (fell back to R1) in {c['fell_back_to_R1']}/50.", "",
          "| clip candidate | trials |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c["histogram"].items()] + \
         ["", "Layouts per chosen clip: " + "; ".join(f"{rule}: " + ", ".join(f"{k} {v}" for k, v in d.items()) for rule, d in s["choice_clips"].items()),
          "", "Choices per rule (clip candidate: layouts): " + "; ".join(f"{rule}: " + ", ".join(f"{k} {v}" for k, v in d.items()) for rule, d in s["choice_histogram"].items()),
          "", "R2 by the clip it chose (on those layouts):", "", "| chosen clip | layouts | candidates | R2 successes | F successes | oracle successes | R2 wins over oracle (trials) | R2 wins over F (trials) |",
          "|---|---|---|---|---|---|---|---|"] + \
         [f"| {k} | {v['layouts']} | " + ", ".join(f"{a} {b}" for a, b in v["candidates"].items()) + f" | {v['R2_successes']} | {v['F_successes']} | {v['oracle_successes']} | "
          f"{v['R2_wins_over_oracle']} | {v['R2_wins_over_F']} |" for k, v in s["r2_by_clip"].items()]
    cv = s["cv"]
    L += ["", f"## Cross-validation (5 folds by layout, {cv['n_layouts']} training layouts, noisy held-out inputs)", "",
          f"AUC {cv['auc']:.3f}, Brier {cv['brier']:.3f} (base rate {cv['base_rate']:.3f}). Held-out selection success: " +
          ", ".join(f"{k} {v}/{cv['n_layouts']}" for k, v in cv["selection"].items()) + ".", "",
          "## Candidates (training layouts)", "", "| clip | candidate | phi | departure | success |", "|---|---|---|---|---|"]
    for x in s["candidates"]["rows"]:
        L.append(f"| {x['clip']} | {x['candidate']} | {x['phi_deg']:.1f} | {x['departure_deg']:.1f} | {x['text']} |")
    L += ["", f"Candidate-oracle ceiling: {s['candidates']['ceiling']['text']}. Fixed-best F: {s['F']['clip']} {s['F']['candidate']}. "
          f"b_hybrid_h45 noisy / best clip: {s['h45_best_clip']}.", "", "## Branches", "", f"- D1: {s['branches']['D1']}",
          "- D2: " + ("; ".join(s["branches"]["D2"]["notes"]) or f"continue (CV AUC {cv['auc']:.3f} >= 0.6, R1 {cv['selection']['R1']} > F {cv['selection']['F']})"),
          "- D3: " + (f"below 30/50: {s['branches']['D3']['below_30']}" if s["branches"]["D3"]["below_30"] else "R1, R2 and F all >= 30/50")]
    for k, d in s["branches"]["D3"]["what_failed"].items():
        L.append(f"  - {k} failures per chosen candidate: " + "; ".join(f"{c}: {dict(v)}" for c, v in d.items() if v))
    L += ["", "## Confounds", ""] + [f"- {x}" for x in s["confounds"]]
    if s["videos"]:
        L += ["", "## Videos", ""] + [f"- `{v['video']}` (gitignored): saved {'success' if v['saved_success'] else v['saved_category']}, re-run identical {v['identical']}"
                                      for v in s["videos"]["videos"]]
        if s["videos"].get("gif"):
            L.append(f"- GIF `{s['videos']['gif']['file']}` ({s['videos']['gif']['chosen']['size_bytes'] / 1e6:.2f} MB)")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("layouts", "step0", "candidates", "onnx-check", "select", "summarize"):
        sub.add_parser(name).add_argument("--out", type=Path, default=OUT)
    rr = sub.add_parser("reproduce-r2")
    rr.add_argument("--out", type=Path, default=OUT)
    rr.add_argument("--trial", type=int, default=0)
    it = sub.add_parser("items")
    it.add_argument("--out", type=Path, default=OUT)
    it.add_argument("--n-layouts", type=int, default=N_TRAIN)
    args = ap.parse_args()
    dict(layouts=cmd_layouts, step0=cmd_step0, items=cmd_items, candidates=cmd_candidates, select=cmd_select, summarize=cmd_summarize, **{"onnx-check": cmd_onnx_check, "reproduce-r2": cmd_reproduce_r2})[args.cmd](args)


if __name__ == "__main__":
    main()
