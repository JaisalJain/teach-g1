"""M4 Steps 0, 1 and 3 (kinematic): two corrections to M3, the v2 path timing for M3's runs, and the oracle's own path under
M3's reachability rule. Outputs in results/m3_retarget/v2/; M3's files are only read.

Step 0.1 (place-palm sign). The nominal oracle palm for a cup centre at xy c is c + the legacy palm_offset[:2], the mapping
  sample_layouts uses (it draws palms in PALM_BOX and puts the cup at palm - offset). M3 flagged its recorded layouts with
  coaster - offset. Recomputed here with the correct sign for each clip's recorded layout (M3's: cup = PALM_BOX centre - offset,
  coaster = cup + my coaster offset).
Step 0.2 (the oracle under M3's rule). paths_v2.oracle_path (the oracle's nominal waypoints through min_jerk_path) on M3's 10
  random layouts (trials 0-9 of sim.m1_pick_mug.trials(), legacy-grasp cup and coaster xy) with each trial's ORIGINAL handle
  yaw (the oracle's heading is phi_ora, so no turn), run through M3's harness unchanged: run_m3.run_one with only its path
  builder swapped for the oracle's (``harness_builder``; run_one looks build_path up when it runs). Same setup, single plan,
  FK errors, reachability rule, joint limits and distances at G as M3's 92 runs. Also, as information only: every variant's
  largest FK errors outside the robot-specific prelude, from the runs' max_by_phase.
Step 1 (v2 timing). For each of M3's 92 runs: M3's path rebuilt by M3's own builder from the run's saved standing palm (checked
  byte-identical to the saved M3 npz), then paths_v2's v2 timing; v2 npz in v2/paths/. Also a dry run of the builder on the
  M4 layouts (trials 0-49 x 4 clips x 3 variants) with the recorded run's standing palm, so a segment that cannot be re-timed
  shows up before the physics runs.
Step 3 (collision diagnostic) on the 10 oracle rows: retarget.collision on the very clip run_one planned.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.run_m3_v2 ...):
  paths      -> v2/paths/<M3 run>.npz (92), v2/paths.json (per run: indices, segments, checks; the dry run; the sigma check)
  oracle     -> v2/runs/oracle__oracle__trialXX.json + v2/paths/oracle__oracle__trialXX.npz (10), v2/oracle.json (checks, resources)
  summarize  -> v2/summary.json + v2/summary.md, read back from the saved files (nothing simulated)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from retarget import run_m3
from retarget.collision import collision_min, collision_min_from_clip
from retarget.frames import load_demo
from retarget.grasp import human_at_grasp, oracle_reference
from retarget.paths_v2 import (B_VARIANTS, SAVE_FIELDS, default_oracle, m3_path, oracle_path, oracle_waypoints, save_path, sigma_check,
                               v2_from_m3_b, v2_from_m3_naive, v2_path)
from retarget.run_m3 import CLIPS, PATH_FIELDS, jsonable, run_names
from sim.m1_grasp_ladder import peak_rss_mb
from sim.m1_pick_mug import trials
from sim.pick_place import PALM_BOX, Grasp

M3 = Path("results/m3_retarget")
OUT = M3 / "v2"
N_ORACLE = 10                     # M3's random layouts = trials 0-9
N_DRY = 50                        # M4's layouts = trials 0-49
ORACLE = "oracle"                 # clip and variant name of the oracle rows (run name oracle__oracle__trialXX)
INFO_PATH_M = 0.020               # M3's path-max rule (2 cm), used only to COUNT runs below it after the prelude (information)
CODE = ("retarget/paths_v2.py", "retarget/collision.py", "retarget/run_m3_v2.py", *run_m3.CODE)


# ---------------------------------------------------------------------------------------------------------------- helpers
def code_hashes() -> dict:
    """SHA-256 of the code a step ran with (summarize checks it still matches)."""
    root = Path(__file__).resolve().parents[1]
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in CODE}


def m3_row(name: str) -> dict:
    return json.loads((M3 / "runs" / f"{name}.json").read_text())


def in_box(p) -> bool:
    return bool(PALM_BOX["x"][0] <= p[0] <= PALM_BOX["x"][1] and PALM_BOX["y"][0] <= p[1] <= PALM_BOX["y"][1])


def outside_by(p) -> list[float]:
    """How far (x, y) a point lies outside PALM_BOX (0 = within the box's range on that axis) [m]."""
    return [float(max(lo - v, 0.0, v - hi)) for v, (lo, hi) in zip(p, (PALM_BOX["x"], PALM_BOX["y"]))]


def same_path(saved, path: dict, fields=PATH_FIELDS) -> list[str]:
    """Fields in which a path dict differs from a saved npz, byte for byte (phase compared as fixed-width unicode)."""
    norm = lambda k: np.asarray([str(p) for p in path[k]], dtype=str) if k == "phase" else np.asarray(path[k])
    return [k for k in fields if not (saved[k].dtype == norm(k).dtype and saved[k].shape == norm(k).shape and np.array_equal(saved[k], norm(k)))]


def layout_from_row(row: dict) -> dict:
    lay = dict(row["layout"])
    lay["cup_xy"], lay["coaster_xy"] = np.asarray(lay["cup_xy"], float), np.asarray(lay["coaster_xy"], float)
    return lay


def oracle_layouts(cup_height: float, n: int = N_ORACLE) -> dict[str, dict]:
    """M3's random layouts (trials 0..n-1 of trials(), legacy grasp) with each trial's own handle yaw (the oracle's)."""
    out = {}
    for t in trials(Grasp.hero_side_grasp(cup_height))[:n]:
        name = f"trial{t['trial']:02d}"
        out[name] = dict(name=name, cup_xy=np.asarray(t["cup_xy"], float), coaster_xy=np.asarray(t["coaster_xy"], float),
                         handle_yaw=math.radians(t["handle_yaw_deg"]), handle_yaw_source="trials() handle_yaw_deg (the oracle's own, no turn)",
                         trial_handle_yaw_deg=t["handle_yaw_deg"], cup_height=cup_height)
    return out


@contextlib.contextmanager
def harness_builder(builder):
    """run_m3.run_one with its path builder swapped: run_one calls the module-level ``build_path`` when it runs, so everything
    else it does (setup, single plan, assert at the start, FK errors, reachability, joint limits, distances at G) is M3's code."""
    saved = run_m3.build_path
    run_m3.build_path = builder
    try:
        yield
    finally:
        run_m3.build_path = saved


def oracle_builder(variant, demo, layouts, layout, oracle, p_s, q_now):
    """build_path's signature (variant, demo, layouts, layout, oracle, p_s, q_now) -> the oracle's nominal path."""
    assert variant == ORACLE
    return oracle_path(layout, p_s, q_now, oracle=oracle)


# ---------------------------------------------------------------------------------------------------------------- paths
def run_record(v2: dict) -> dict:
    """The per-run entry of paths.json."""
    info = v2["info"]
    return dict(frames=info["frames"], m3_frames=info["m3_frames"], **{k: int(v2[k]) for k in SAVE_FIELDS if k.startswith("i_")},
                follow=v2["follow"], segments=info["v2"]["segments"], j_pause=info["v2"].get("j_pause"), j_peak=info["v2"].get("j_peak"),
                k=info["k"])


def trial_layouts(demo: dict, oracle: dict, cup_height: float, n: int) -> list[dict]:
    """M4's layouts (dry run only): trials 0..n-1 (legacy-grasp xy), handle yaw = trial yaw + (phi_h - phi_ora), as M3's."""
    turn = human_at_grasp(demo, cup_height)["phi_h"] - oracle["phi_ora"]
    return [dict(name=f"trial{t['trial']:02d}", cup_xy=np.asarray(t["cup_xy"], float), coaster_xy=np.asarray(t["coaster_xy"], float),
                 handle_yaw=math.radians(t["handle_yaw_deg"]) + turn, cup_height=cup_height)
            for t in trials(Grasp.hero_side_grasp(cup_height))[:n]]


def cmd_paths(args):
    t0 = time.perf_counter()
    (OUT / "paths").mkdir(parents=True, exist_ok=True)
    names = run_names()
    existing = [n for n in names if (OUT / "paths" / f"{n}.npz").exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} v2 paths exist in {OUT / 'paths'}; pass --overwrite to replace them")
    runs, ctx = {}, {}
    for name in names:
        row = m3_row(name)
        clip, variant = row["clip"], row["variant"]
        h = float(row["layout"]["cup_height"])
        if clip not in ctx:
            demo = load_demo(clip)
            oracle = default_oracle(h)
            # M3's runs took r from the compiled cup_geom; the config value must be the same number
            assert jsonable(oracle) == row["oracle"], f"{clip}: config oracle reference differs from M3's run"
            layouts = run_m3.layouts_for(demo, human_at_grasp(demo, h), oracle, h)
            ctx[clip] = (demo, oracle, layouts)
        demo, oracle, layouts = ctx[clip]
        layout = layouts[row["layout"]["name"]]
        assert jsonable(layout) == row["layout"], f"{name}: rebuilt layout differs from M3's"
        p_s, q_now = np.asarray(row["setup"]["p_s"], float), np.asarray(row["setup"]["q_now_wxyz"], float)
        m3 = m3_path(clip, variant, layout, p_s, q_now, oracle=oracle)
        with np.load(M3 / "paths" / f"{name}.npz", allow_pickle=False) as saved:
            differing = same_path(saved, m3)
        assert not differing, f"{name}: M3's path rebuilt differs from the saved one in {differing}"
        v2 = v2_from_m3_naive(m3) if variant == "a_naive" else v2_from_m3_b(m3)
        save_path(v2, OUT / "paths" / f"{name}.npz")
        runs[name] = dict(clip=clip, variant=variant, layout=row["layout"]["name"], m3_path_rebuilt_identical=True, **run_record(v2))
        print(f"{name:<36} M3 {v2['info']['m3_frames']['total']} -> v2 {len(v2['palm_pos'])} frames; segments "
              + ", ".join(f"{s['name']} {s['D0_s']:.2f}->{s['D_s']:.2f} s" for s in v2["info"]["v2"]["segments"]), flush=True)

    # the oracle's nominal legs = warp.time_scale's (the same formulas): checked at every clip's recorded layout
    legs = {}
    for clip in CLIPS:
        row = m3_row(run_m3.run_name(clip, "b_level", "recorded"))
        ts = row["path_info"]["time_scale"]
        w = oracle_waypoints(layout_from_row(row), row["setup"]["p_s"], row["setup"]["q_now_wxyz"], default_oracle(row["layout"]["cup_height"]))
        legs[clip] = dict(pregrasp_dist_m=[w["pregrasp_dist_m"], ts["oracle_pregrasp_leg"]["dist_m"]],
                          transport_dist_m=[w["transport_dist_m"], ts["oracle_transport_leg"]["dist_m"]],
                          transport_s=[w["transport_s"], ts["oracle_transport_leg"]["T_s"]], z_clear=[w["z_clear"], ts["oracle_pregrasp_leg"]["z_clear"]])
        legs[clip]["equal"] = all(abs(a - b) < 1e-12 for a, b in legs[clip].values())

    # dry run on M4's layouts (trials 0-49): every v2 path must build (re-timing asserts included)
    dry = dict(layouts=f"trials 0-{N_DRY - 1}, handle yaw turned by phi_h - phi_ora", standing_palm="each clip's recorded b_level run's p_s / q_now",
               built=0, failed=[], min_segment_D0_s={})
    for clip in CLIPS:
        row = m3_row(run_m3.run_name(clip, "b_level", "recorded"))
        p_s, q_now = np.asarray(row["setup"]["p_s"], float), np.asarray(row["setup"]["q_now_wxyz"], float)
        h = float(row["layout"]["cup_height"])
        demo, oracle, _ = ctx[clip]
        for variant in (*B_VARIANTS, "naive"):
            mins = {}
            for lay in trial_layouts(demo, oracle, h, N_DRY):
                try:
                    v2 = v2_path(clip, variant, lay, p_s, q_now, oracle=oracle)
                except (AssertionError, ValueError) as e:
                    dry["failed"].append(dict(clip=clip, variant=variant, layout=lay["name"], error=str(e)))
                    continue
                dry["built"] += 1
                for s in v2["info"]["v2"]["segments"]:
                    mins[s["name"]] = min(mins.get(s["name"], np.inf), s["D0_s"])
            dry["min_segment_D0_s"][f"{clip}__{variant}"] = mins
    dry["of"] = len(CLIPS) * 3 * N_DRY

    out = dict(step="paths", runs_n=len(runs), all_m3_paths_rebuilt_identical=all(r["m3_path_rebuilt_identical"] for r in runs.values()),
               segment_asserts="retime: first and last 50 Hz step < 1 mm, end positions = M3's to 1e-9; assembled path ends = M3's to 1e-9 "
                               "(an assert failure stops this step, so every saved path passed them)",
               sigma_check=sigma_check(), oracle_legs_equal_time_scale=legs, dry_run=jsonable(dry), runs=jsonable(runs),
               wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), code_sha256=code_hashes())
    (OUT / "paths.json").write_text(json.dumps(out, indent=2))
    print(f"{len(runs)} v2 paths; M3 rebuilt identical: {out['all_m3_paths_rebuilt_identical']}; sigma check {out['sigma_check']['passed']}; "
          f"oracle legs = time_scale: {all(v['equal'] for v in legs.values())}; dry run built {dry['built']}/{dry['of']}, failed {len(dry['failed'])}; "
          f"wrote {OUT / 'paths.json'} ({out['wall_s']} s, peak RSS {out['peak_rss_mb']} MB)")


# ---------------------------------------------------------------------------------------------------------------- oracle
def cmd_oracle(args):
    t0 = time.perf_counter()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    existing = list((OUT / "runs").glob(f"{ORACLE}__*.json"))
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} oracle rows exist in {OUT / 'runs'}; pass --overwrite to replace them")
    task = run_m3.make_task()                                   # M3's: SetupOnly(mug, 0.40 kg from config, mug_v1)
    oracle = oracle_reference(task.scene)
    layouts = oracle_layouts(task.scene.cup_height)
    rows, refs = [], {}
    with harness_builder(oracle_builder):
        for name, layout in layouts.items():
            row, ref, path = run_m3.run_one(task, ORACLE, ORACLE, name, None, {}, oracle, layouts)
            t1 = time.perf_counter()
            row["collision_min"] = jsonable(collision_min_from_clip(task.scene, ref, path, layout))
            row["collision_wall_s"] = round(time.perf_counter() - t1, 2)
            save_path(path, OUT / "paths" / f"{row['run']}.npz")
            (OUT / "runs" / f"{row['run']}.json").write_text(json.dumps(row, indent=2))
            rows.append(row)
            refs[name] = path
            fe = row["fk_error"]
            print(f"{row['run']:<26} reachable {str(row['reachable']):<5} pos cm max {fe['pos_cm']['max']:5.2f} ({fe['pos_cm']['argmax_phase']}) "
                  f"G {fe['pos_cm']['at_G']:4.2f} P {fe['pos_cm']['at_P']:4.2f}; collision min {row['collision_min']['min_mm']:6.1f} mm "
                  f"({row['collision_min']['object_geom']}, frame {row['collision_min']['frame']}); hand-mug at G "
                  f"{row['distances_at_G']['hand_mug']['min_m'] * 1e3:5.1f} mm ({row['wall_s']} s)", flush=True)

    # check: collision_min (its own setup + plan) gives the same numbers as the diagnostic on run_one's clip
    name = next(iter(layouts))
    fresh = jsonable(collision_min(task, layouts[name], refs[name]))
    keep = lambda d: {k: v for k, v in d.items() if k not in ("plan", "path_start_minus_palm_start_mm", "path_start_minus_p_s_m")}
    first = rows[0]["collision_min"]
    # check: the standing palm does not depend on the handle yaw (oracle rows: trial yaw; M3's b rows: turned yaw)
    p_s_same = {r["layout"]["name"]: r["setup"]["p_s"] == m3_row(run_m3.run_name(CLIPS[0], "b_level", r["layout"]["name"]))["setup"]["p_s"]
                for r in rows}
    out = dict(step="oracle", rows=len(rows), task="run_m3.make_task(): SetupOnly(cup_model='mug', mass from config, mug_v1)",
               harness="run_m3.run_one with build_path swapped for paths_v2.oracle_path (harness_builder)",
               collision_min_equals_run_one_clip=dict(layout=name, identical=keep(fresh) == first,
                                                      fresh_plan_info=fresh["plan"], run_one_plan_info=rows[0]["plan"]),
               p_s_identical_to_m3_b_rows_same_trial=p_s_same,
               wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), gpu="none (CPU only)", code_sha256=code_hashes())
    (OUT / "oracle.json").write_text(json.dumps(jsonable(out), indent=2))
    print(f"collision_min (fresh setup + plan) = run_one clip's: {out['collision_min_equals_run_one_clip']['identical']}; p_s identical to M3 "
          f"b rows: {all(p_s_same.values())}; wrote {OUT / 'oracle.json'} ({out['wall_s']} s, peak RSS {out['peak_rss_mb']} MB)")


# ---------------------------------------------------------------------------------------------------------------- summarize
def stats(v) -> dict | None:
    v = [x for x in v if x is not None]
    return dict(n=len(v), min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v))) if v else None


def after_prelude(row: dict) -> dict:
    """The run's largest FK errors outside the prelude (max over the other phases of max_by_phase), and where."""
    mbp = {p: v for p, v in row["fk_error"]["max_by_phase"].items() if p != "prelude"}
    p_pos = max(mbp, key=lambda p: mbp[p]["pos_cm"])
    p_ori = max(mbp, key=lambda p: mbp[p]["ori_deg"])
    return dict(pos_cm=mbp[p_pos]["pos_cm"], pos_phase=p_pos, ori_deg=mbp[p_ori]["ori_deg"], ori_phase=p_ori,
                prelude_pos_cm=row["fk_error"]["max_by_phase"]["prelude"]["pos_cm"])


def cmd_summarize(args):
    t0 = time.perf_counter()
    m3_rows = [m3_row(n) for n in run_names()]
    or_rows = [json.loads((OUT / "runs" / f"{ORACLE}__{ORACLE}__trial{k:02d}.json").read_text()) for k in range(N_ORACLE)]
    paths = json.loads((OUT / "paths.json").read_text())
    orc = json.loads((OUT / "oracle.json").read_text())

    # Step 0.1: the place-palm flag with the correct sign
    place = {}
    for clip in CLIPS:
        lay = m3_row(run_m3.run_name(clip, "b_level", "recorded"))["layout"]
        off = Grasp.hero_side_grasp(lay["cup_height"]).palm_offset[:2]
        grasp_palm = np.asarray(lay["cup_xy"]) + off
        place_palm = np.asarray(lay["coaster_xy"]) + off
        place[clip] = dict(cup_xy=lay["cup_xy"], coaster_xy=lay["coaster_xy"], legacy_palm_offset_xy=off.tolist(),
                           grasp_palm_xy=grasp_palm.tolist(), grasp_palm_in_palm_box=in_box(grasp_palm),
                           place_palm_xy=place_palm.tolist(), place_palm_in_palm_box=in_box(place_palm), place_palm_outside_by_m=outside_by(place_palm),
                           m3_flag=dict(formula="coaster - offset (M3's spec sign)", xy=lay["coaster_place_palm_xy"], in_box=lay["coaster_place_palm_in_palm_box"]))

    # Step 0.2: reachability with the oracle rows; max errors after the prelude (information)
    groups = {(c, v): [r for r in m3_rows if r["clip"] == c and r["variant"] == v] for c in CLIPS for v in run_m3.VARIANTS}
    groups[(ORACLE, ORACLE)] = or_rows
    reach, after = {}, {}
    for (c, v), rows in groups.items():
        key = f"{c}__{v}"
        reach[key] = dict(clip=c, variant=v, reachable=sum(r["reachable"] for r in rows), of=len(rows),
                          parts_passed={k: sum(r["reachable_parts"][k] for r in rows) for k in ("at_G", "at_P", "path")},
                          layouts="recorded" if v == "a_naive" else "trials 0-9" if v == ORACLE else "recorded + trials 0-9")
        ap = [after_prelude(r) for r in rows]
        after[key] = dict(clip=c, variant=v, runs=len(rows), pos_cm=stats([a["pos_cm"] for a in ap]), ori_deg=stats([a["ori_deg"] for a in ap]),
                          prelude_pos_cm=stats([a["prelude_pos_cm"] for a in ap]),
                          pos_phase_counts={p: sum(a["pos_phase"] == p for a in ap) for p in dict.fromkeys(a["pos_phase"] for a in ap)},
                          runs_below_2cm_after_prelude_information=sum(a["pos_cm"] < INFO_PATH_M * 100 for a in ap))
    per_variant = {}
    for v in (*run_m3.VARIANTS, ORACLE):
        rows = [r for (c, vv), rr in groups.items() if vv == v for r in rr]
        ap = [after_prelude(r) for r in rows]
        per_variant[v] = dict(runs=len(rows), reachable=sum(r["reachable"] for r in rows), pos_cm_after_prelude=stats([a["pos_cm"] for a in ap]),
                              ori_deg_after_prelude=stats([a["ori_deg"] for a in ap]), prelude_pos_cm=stats([a["prelude_pos_cm"] for a in ap]),
                              runs_below_2cm_after_prelude_information=sum(a["pos_cm"] < INFO_PATH_M * 100 for a in ap))
    oracle_table = [dict(run=r["run"], layout=r["layout"]["name"], handle_yaw_deg=math.degrees(r["layout"]["handle_yaw"]), reachable=r["reachable"],
                         parts=r["reachable_parts"], pos_cm=dict(max=r["fk_error"]["pos_cm"]["max"], at_G=r["fk_error"]["pos_cm"]["at_G"],
                                                                  at_P=r["fk_error"]["pos_cm"]["at_P"], argmax_phase=r["fk_error"]["pos_cm"]["argmax_phase"]),
                         ori_deg=dict(max=r["fk_error"]["ori_deg"]["max"], at_G=r["fk_error"]["ori_deg"]["at_G"], at_P=r["fk_error"]["ori_deg"]["at_P"]),
                         max_by_phase_pos_cm={p: v["pos_cm"] for p, v in r["fk_error"]["max_by_phase"].items()}, after_prelude=after_prelude(r),
                         qp_failures=r["qp_failures"], joint_limit_hit_frames=r["joint_limits"]["hit_frames"],
                         hand_mug_at_G_mm=r["distances_at_G"]["hand_mug"]["min_m"] * 1e3, hand_handle_at_G_mm=r["distances_at_G"]["hand_handle"]["min_m"] * 1e3,
                         collision_min_mm=r["collision_min"]["min_mm"], collision_min_where=dict(frame=r["collision_min"]["frame"], phase=r["collision_min"]["phase"],
                                                                                             remaining_arc_to_G_m=r["collision_min"]["remaining_arc_to_G_m"],
                                                                                             hand_body=r["collision_min"]["hand_body"], object_geom=r["collision_min"]["object_geom"]),
                         path_start_minus_palm_start_mm=r["setup"]["path_start_minus_palm_start_mm"], frames=r["path"]["T"])
                    for r in or_rows]

    # Step 1: v2 paths
    pr = paths["runs"]
    seg_tab = {}
    for name, r in pr.items():
        for s in r["segments"]:
            k = (r["clip"], r["variant"], s["name"])
            e = seg_tab.setdefault(k, dict(D0=[], first=[], last=[], m3_first=[], m3_last=[]))
            e["D0"].append(s["D0_s"]); e["first"].append(s["first_step_mm"]); e["last"].append(s["last_step_mm"])
            e["m3_first"].append(s["m3_first_last_speed_m_s"][0]); e["m3_last"].append(s["m3_first_last_speed_m_s"][1])
    segments = [dict(clip=c, variant=v, segment=s, runs=len(e["D0"]), D0_s=stats(e["D0"]), first_step_mm_max=max(e["first"]), last_step_mm_max=max(e["last"]),
                     m3_start_speed_m_s=stats(e["m3_first"]), m3_end_speed_m_s=stats(e["m3_last"])) for (c, v, s), e in seg_tab.items()]
    recorded = {n: dict(m3_frames=r["m3_frames"]["total"], v2_frames=r["frames"]["total"], **{k: r[k] for k in ("i_prelude_end", "i_pause", "i_G", "i_close_end", "i_peak", "i_P", "i_release")})
                for n, r in pr.items() if r["layout"] == "recorded"}

    checks = dict(
        m3_paths_rebuilt_identical=f"{sum(r['m3_path_rebuilt_identical'] for r in pr.values())}/{len(pr)}",
        sigma_closed_form_vs_numeric=paths["sigma_check"], oracle_legs_equal_warp_time_scale={c: v["equal"] for c, v in paths["oracle_legs_equal_time_scale"].items()},
        dry_run_trials_0_49=dict(built=paths["dry_run"]["built"], of=paths["dry_run"]["of"], failed=paths["dry_run"]["failed"],
                                 min_segment_D0_s=min(v for d in paths["dry_run"]["min_segment_D0_s"].values() for v in d.values())),
        collision_min_equals_run_one_clip=orc["collision_min_equals_run_one_clip"]["identical"],
        oracle_p_s_identical_to_m3_b_rows=all(orc["p_s_identical_to_m3_b_rows_same_trial"].values()),
        oracle_path_start_vs_palm_start_mm_max=max(r["setup"]["path_start_minus_palm_start_mm"] for r in or_rows),
        oracle_mapping_at_G_passed=sum(r["mapping_check"]["passed"] for r in or_rows),
        oracle_collision_mapping_passed=sum(r["collision_min"]["mapping_passed"] for r in or_rows),
        same_code=dict(paths=paths["code_sha256"] == code_hashes(), oracle=orc["code_sha256"] == code_hashes()),
        # M3's code is only imported, never edited: its files must still hash as when M3's 92 runs were made
        m3_code_unchanged=json.loads((M3 / "run_resources.json").read_text())["code_sha256"] == run_m3.code_hashes())
    coll = dict(per_row_mm=[r["collision_min_mm"] for r in oracle_table], stats_mm=stats([r["collision_min_mm"] for r in oracle_table]),
                body_stats_mm=stats([r["collision_min"]["body"]["min_mm"] for r in or_rows]),
                handle_stats_mm=stats([r["collision_min"]["handle"]["min_mm"] for r in or_rows]),
                method=or_rows[0]["collision_min"]["method"])
    summary = dict(
        protocol=dict(
            place_palm="nominal oracle palm for a cup centre at xy c = c + legacy palm_offset[:2] (sample_layouts' mapping); place palm = coaster + offset",
            oracle_rows="paths_v2.oracle_path on trials 0-9 (legacy-grasp xy, each trial's own handle yaw) through run_m3.run_one unchanged "
                        "(build_path swapped); reachable = M3's rule (FK error at G and at P < 1 cm and < 10 deg, path max < 2 cm)",
            after_prelude="information only: per run, the largest FK position / orientation error over the phases other than 'prelude' "
                          "(max_by_phase); 'below 2 cm' counts runs whose after-prelude maximum is < 2 cm (not a rule)",
            v2="paths_v2: M3's path, pause removed, approach split at i_pause, transport at its peak, each human segment re-timed rest to "
               "rest (D = D0 + 0.4 s, smoothstep speed ramps), geometry / warps / k unchanged",
            collision="retarget.collision (Step 3), information only"),
        checks=checks, place_palm=place, reachability=reach, after_prelude=after, after_prelude_per_variant=per_variant, oracle_rows=oracle_table,
        collision_oracle=coll, v2_segments=segments, v2_recorded=recorded,
        resources=dict(paths=dict(wall_s=paths["wall_s"], peak_rss_mb=paths["peak_rss_mb"]), oracle=dict(wall_s=orc["wall_s"], peak_rss_mb=orc["peak_rss_mb"]),
                       gpu="none (CPU only)"))
    summary["resources"]["summarize"] = dict(wall_s=round(time.perf_counter() - t0, 2), peak_rss_mb=peak_rss_mb())
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    (OUT / "summary.md").write_text(summary_md(jsonable(summary)))
    print(summary_md(jsonable(summary)))
    print(f"wrote {OUT / 'summary.json'} and {OUT / 'summary.md'}")


def fmt(v, spec=".2f") -> str:
    return "-" if v is None else format(v, spec)


def summary_md(s: dict) -> str:
    c = s["checks"]
    L = ["# M3 v2: place-palm sign, the oracle under M3's rule, v2 path timing", "",
         "Made by `python -m retarget.run_m3_v2 summarize` from results/m3_retarget/runs/*.json (M3, read only), v2/runs/oracle__*.json, "
         "v2/paths.json and v2/oracle.json. Kinematic only (no physics beyond PickPlace.run's 2 s setup hold).", "",
         "## Checks", "",
         f"- M3's 92 paths rebuilt from the saved standing palms by M3's own builders: {c['m3_paths_rebuilt_identical']} byte-identical to the saved npz.",
         f"- sigma (closed form) vs a numeric integral of v: max diff {c['sigma_closed_form_vs_numeric']['max_abs_diff_s']:.1e} s "
         f"(pass {c['sigma_closed_form_vs_numeric']['passed']}); sigma(D) - D0 = {c['sigma_closed_form_vs_numeric']['sigma_D_minus_D0_s']:.1e} s.",
         "- Per re-timed segment (asserted while building, so every saved path passed): first and last 50 Hz step < 1 mm, end positions = M3's to 1e-9.",
         f"- Dry run of v2_path on M4's layouts (trials 0-49 x 4 clips x b_level / b_tilt / naive): built {c['dry_run_trials_0_49']['built']}/"
         f"{c['dry_run_trials_0_49']['of']}, failed {len(c['dry_run_trials_0_49']['failed'])}; shortest segment D0 {c['dry_run_trials_0_49']['min_segment_D0_s']:.2f} s (T_R = 0.4 s).",
         f"- Oracle nominal legs = warp.time_scale's at each recorded layout (distances, transport time, z_clear to 1e-12): "
         f"{all(c['oracle_legs_equal_warp_time_scale'].values())}.",
         f"- Oracle rows: path sample 0 vs palm_start max {c['oracle_path_start_vs_palm_start_mm_max']:.3f} mm (limit 1 mm); clip -> plant mapping at G "
         f"{c['oracle_mapping_at_G_passed']}/10, over the collision window {c['oracle_collision_mapping_passed']}/10 (< 1 mm).",
         f"- collision_min with its own setup + plan = the diagnostic on run_one's clip: {c['collision_min_equals_run_one_clip']}. "
         f"Standing palm of the oracle rows = M3's b rows on the same trials (other handle yaw): {c['oracle_p_s_identical_to_m3_b_rows']}.",
         f"- Code unchanged since the steps ran (SHA-256): paths {c['same_code']['paths']}, oracle {c['same_code']['oracle']}. "
         f"M3's 8 code files = the hashes in results/m3_retarget/run_resources.json: {c['m3_code_unchanged']}.", "",
         "## Step 0.1 — place-palm flag with the correct sign (recorded layouts)", "",
         "Nominal oracle palm for a cup centre at xy c = c + legacy palm_offset[:2] (what sample_layouts uses). M3 flagged coaster - offset.", "",
         "| clip | cup xy | coaster xy | grasp palm xy (in box) | place palm xy = coaster + offset | in PALM_BOX | outside by x / y (cm) | M3's flag (coaster - offset) |",
         "|---|---|---|---|---|---|---|---|"]
    for clip, p in s["place_palm"].items():
        xy = lambda v: f"({v[0]:.3f}, {v[1]:.3f})"
        L.append(f"| {clip} | {xy(p['cup_xy'])} | {xy(p['coaster_xy'])} | {xy(p['grasp_palm_xy'])} ({'yes' if p['grasp_palm_in_palm_box'] else 'no'}) | "
                 f"{xy(p['place_palm_xy'])} | {'yes' if p['place_palm_in_palm_box'] else 'no'} | {p['place_palm_outside_by_m'][0] * 100:.1f} / "
                 f"{p['place_palm_outside_by_m'][1] * 100:.1f} | {xy(p['m3_flag']['xy'])} {'in' if p['m3_flag']['in_box'] else 'out'} |")
    L += ["", f"PALM_BOX: x {PALM_BOX['x']}, y {PALM_BOX['y']} m.", "",
          "## Step 0.2 — reachability under M3's rule, with the oracle's own nominal path", "",
          "Reachable = FK error at G and at P < 1 cm and < 10 deg AND path max position error < 2 cm (M3's rule, unchanged). Oracle: "
          "paths_v2.oracle_path through run_m3.run_one on trials 0-9 with each trial's own handle yaw.", "",
          "| clip | variant | layouts | at G | at P | path max < 2 cm | reachable |", "|---|---|---|---|---|---|---|"]
    for r in s["reachability"].values():
        pp = r["parts_passed"]
        L.append(f"| {r['clip']} | {r['variant']} | {r['layouts']} | {pp['at_G']}/{r['of']} | {pp['at_P']}/{r['of']} | {pp['path']}/{r['of']} | {r['reachable']}/{r['of']} |")
    L += ["", "Oracle rows:", "",
          "| run | handle yaw (deg) | reachable | pos err max / G / P (cm) | where the max is | max after prelude (cm, phase) | ori max (deg) | hand-mug at G (mm) | collision min (mm) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in s["oracle_rows"]:
        a = r["after_prelude"]
        L.append(f"| {r['run']} | {r['handle_yaw_deg']:.1f} | {'yes' if r['reachable'] else 'no'} | {r['pos_cm']['max']:.2f} / {r['pos_cm']['at_G']:.2f} / "
                 f"{r['pos_cm']['at_P']:.2f} | {r['pos_cm']['argmax_phase']} | {a['pos_cm']:.2f} ({a['pos_phase']}) | {r['ori_deg']['max']:.1f} | "
                 f"{r['hand_mug_at_G_mm']:.1f} | {fmt(r['collision_min_mm'], '.1f')} |")
    L += ["", "### Largest FK errors outside the prelude (information only; from each run's max_by_phase)", "",
          "Per run: the maximum over every phase except 'prelude'. Columns give min / median / max over the group's runs; 'below 2 cm' counts runs "
          "whose after-prelude maximum is < 2 cm (a count, not a rule).", "",
          "| clip | variant | runs | pos after prelude (cm) | phase of the max (runs) | ori after prelude (deg) | prelude pos max (cm) | below 2 cm after prelude |",
          "|---|---|---|---|---|---|---|---|"]
    mmm = lambda st, spec=".2f": "-" if st is None else f"{st['min']:{spec}} / {st['median']:{spec}} / {st['max']:{spec}}"
    for a in s["after_prelude"].values():
        L.append(f"| {a['clip']} | {a['variant']} | {a['runs']} | {mmm(a['pos_cm'])} | {', '.join(f'{p} {n}' for p, n in a['pos_phase_counts'].items())} | "
                 f"{mmm(a['ori_deg'], '.1f')} | {mmm(a['prelude_pos_cm'])} | {a['runs_below_2cm_after_prelude_information']}/{a['runs']} |")
    L += ["", "| variant (all clips) | runs | reachable | pos after prelude (cm) | ori after prelude (deg) | prelude pos max (cm) | below 2 cm after prelude |",
          "|---|---|---|---|---|---|---|"]
    for v, a in s["after_prelude_per_variant"].items():
        L.append(f"| {v} | {a['runs']} | {a['reachable']} | {mmm(a['pos_cm_after_prelude'])} | {mmm(a['ori_deg_after_prelude'], '.1f')} | "
                 f"{mmm(a['prelude_pos_cm'])} | {a['runs_below_2cm_after_prelude_information']}/{a['runs']} |")
    co = s["collision_oracle"]
    L += ["", "## Step 3 — collision diagnostic on the oracle rows (kinematic, information only)", "",
          f"Method: {co['method']}.", "",
          f"Min signed distance over the window (mm; negative = overlap), 10 rows: min / median / max {mmm(co['stats_mm'], '.1f')}; "
          f"to the body {mmm(co['body_stats_mm'], '.1f')}; to the handle {mmm(co['handle_stats_mm'], '.1f')}. Per row in the oracle table above.", "",
          "## Step 1 — v2 path timing (M3's 92 runs; v2/paths/<run>.npz)", "",
          "Per segment over the runs (b: 11 layouts per clip and variant; naive: recorded only). D0 = M3's duration (D = D0 + 0.4 s). "
          "M3 start / end speed = the speed of M3's first / last step of the segment: the jump v2 removes.", "",
          "| clip | variant | segment | runs | D0 (s) min / median / max | first / last step max (mm) | M3 start speed (m/s) | M3 end speed (m/s) |",
          "|---|---|---|---|---|---|---|---|"]
    for g in s["v2_segments"]:
        L.append(f"| {g['clip']} | {g['variant']} | {g['segment']} | {g['runs']} | {mmm(g['D0_s'])} | {g['first_step_mm_max']:.3f} / {g['last_step_mm_max']:.3f} | "
                 f"{mmm(g['m3_start_speed_m_s'], '.3f')} | {mmm(g['m3_end_speed_m_s'], '.3f')} |")
    L += ["", "Recorded layout, frame indices of the v2 paths (-1 = none):", "",
          "| run | M3 frames | v2 frames | i_prelude_end | i_pause | i_G | i_close_end | i_peak | i_P | i_release |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in s["v2_recorded"].items():
        L.append(f"| {n} | {r['m3_frames']} | {r['v2_frames']} | {r['i_prelude_end']} | {r['i_pause']} | {r['i_G']} | {r['i_close_end']} | {r['i_peak']} | "
                 f"{r['i_P']} | {r['i_release']} |")
    r = s["resources"]
    L += ["", "## Runtime and peak RSS (CPU only)", "", "| step | wall (s) | peak RSS (MB) |", "|---|---|---|",
          f"| paths | {r['paths']['wall_s']} | {r['paths']['peak_rss_mb']} |", f"| oracle (1 process) | {r['oracle']['wall_s']} | {r['oracle']['peak_rss_mb']} |",
          f"| summarize | {r['summarize']['wall_s']} | {r['summarize']['peak_rss_mb']} |"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("paths", "v2 paths of M3's 92 runs (+ dry run on trials 0-49)"), ("oracle", "the oracle's nominal path through M3's harness, trials 0-9")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--overwrite", action="store_true")
    sub.add_parser("summarize", help="summary.json + summary.md from the saved files")
    args = ap.parse_args()
    dict(paths=cmd_paths, oracle=cmd_oracle, summarize=cmd_summarize)[args.cmd](args)


if __name__ == "__main__":
    main()
