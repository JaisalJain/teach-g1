"""M4b: the re-categorisation of failures (Step 1) and the M4b results (Step 4), read back from saved row files (nothing is
simulated here).

category_v2 (Step 1). M4's categories file most runs that pushed the mug away on the way in as "palm could not reach the target",
because the executor's grasp settle runs before check_cup_undisturbed (CLAUDE.md "Check order"). category_v2 = "pushed mug
before close" when the run failed BEFORE the close phase AND the horizontal distance between the mug's final position
(detail.cup_final) and its layout position (cup_xy) is more than 2 cm; otherwise it is the run's own ``category``. The run files
are never edited: category_v2 is computed here, from them.

"Failed before the close phase" = the Failure was raised before close_hand started: failed_phase "approach" (the oracle's
check_cup_undisturbed, after the grasp settle), a "reach" failure of the pre-grasp or grasp settle (the run's last settle record
names it), or a fall during one of the phases before the close.

Run from the repo root (env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.m4b_summary ...):
  m4-v2      Step 1 on M4's rows (results/m4_execute/runs) -> results/m4b/m4_category_v2.json
  summarize  Step 4 -> results/m4b/summary.md + summary.json + heading_sweep.png. Inputs (read only): results/m4b/runs (the 400
             hybrid rows), results/m4_execute/runs (M4's b_level), fifty.json (the oracle), results/m4b/heading_sweep/runs.json,
             results/m3_retarget/human_values.json (each clip's phi_h), results/m4b/videos/videos.json if it exists. It also
             picks the video runs by fixed rules (``video_picks``) and writes them to summary.json for ``sim.m4_execute render``.

Main table: success / 50 per clip (and / 200 pooled) with the Wilson 95 % interval; the pooled interval treats the 200 runs as
independent although they reuse the same 50 layouts (the real uncertainty is larger). Paired tables count the same layout (trial)
in both groups. The heading sweep's rows are the oracle's (no human data); its plot marks each clip's phi_h.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sim.m1_pick_mug import wilson
from sim.m4_execute import CLIPS, FIFTY, HYBRID, M4_VARIANTS, N_TRIALS, OUT_DIR as M4_DIR, run_name
from sim.pick_place import CATEGORY

OUT = Path("results/m4b")
PUSHED = "pushed mug before close"
PUSH_M = 0.02                                   # spec: horizontal mug displacement > 2 cm
PRE_CLOSE_SETTLES = ("settle_pregrasp", "settle_grasp")
PRE_CLOSE_PHASES = ("stand", "pregrasp", *PRE_CLOSE_SETTLES, "approach")     # tick phases before close_hand (a fall there)


def failed_before_close(r: dict) -> bool:
    """True if the run failed before close_hand started (module docstring)."""
    if r["success"]:
        return False
    phase = r["failed_phase"]
    if phase == "reach":
        settles = r["settles"] if "settles" in r else r["detail"]["settles"]      # oracle rows keep them under detail
        settle = settles[-1]["phase"]
        assert r["reason"].startswith(settle + ":"), f"{r.get('run', r.get('trial'))}: the last settle {settle} is not the failing one ({r['reason']})"
        return settle in PRE_CLOSE_SETTLES
    if r["category"] == "fell":
        return phase in PRE_CLOSE_PHASES
    return phase == "approach"


def cup_shift_m(r: dict) -> float:
    """Horizontal distance between the mug's final position and its layout position."""
    return float(np.linalg.norm(np.asarray(r["detail"]["cup_final"][:2], float) - np.asarray(r["cup_xy"], float)))


def category_v2(r: dict) -> str | None:
    """None for a success, as ``category``."""
    if r["success"]:
        return None
    return PUSHED if failed_before_close(r) and cup_shift_m(r) > PUSH_M else r["category"]


def load_rows(runs_dir: Path) -> dict[str, dict]:
    rows = {}
    for p in sorted(runs_dir.glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("success") is None:
            raise SystemExit(f"{p}: error row (no outcome)")
        rows[r["run"]] = r
    return rows


def recategorised(rows: list[dict]) -> dict:
    """Both categorisations of a group of rows, and how category maps onto category_v2."""
    fails = [r for r in rows if not r["success"]]
    return dict(n=len(rows), successes=len(rows) - len(fails), failures=len(fails),
                category=dict(Counter(r["category"] for r in fails).most_common()),
                category_v2=dict(Counter(category_v2(r) for r in fails).most_common()),
                category_to_v2={f"{a} -> {b}": n for (a, b), n in Counter((r["category"], category_v2(r)) for r in fails).most_common()})


def cmd_m4_v2(args):
    rows = load_rows(args.runs)
    assert len(rows) == 604, f"expected M4's 604 rows, found {len(rows)}"
    groups = {}
    for v in M4_VARIANTS:
        for c in CLIPS:
            groups[f"{v}/{c}/trials"] = recategorised([r for r in rows.values() if r["variant"] == v and r["clip"] == c and r["trial"] is not None])
        groups[f"{v}/pool/trials"] = recategorised([r for r in rows.values() if r["variant"] == v and r["trial"] is not None])
    groups["b_level/recorded_v2"] = recategorised([r for r in rows.values() if r["layout"] == "recorded_v2"])
    out = dict(rule=f"category_v2 = '{PUSHED}' if the run failed before the close phase and |cup_final xy - cup_xy| > {PUSH_M} m, "
                    "else category (retarget/m4b_summary.py)", runs_dir=str(args.runs), rows=len(rows), groups=groups,
               runs={n: dict(category=r["category"], category_v2=category_v2(r), failed_before_close=failed_before_close(r),
                             cup_shift_cm=round(cup_shift_m(r) * 100, 2)) for n, r in rows.items() if not r["success"]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "m4_category_v2.json").write_text(json.dumps(out, indent=2))
    for k, g in groups.items():
        print(f"{k:<28} {g['failures']:3d} failures  category {g['category']}  ->  category_v2 {g['category_v2']}")
    print(f"wrote {args.out / 'm4_category_v2.json'}")


# ---------------------------------------------------------------------------------------------------------------- Step 4
SWEEP = OUT / "heading_sweep" / "runs.json"
HUMAN_VALUES = Path("results/m3_retarget/human_values.json")
VIDEOS = OUT / "videos" / "videos.json"
VIEW = {"26d4118b0c": "top-down", "4e103843db": "90 deg", "49aff04f7a": "45 deg", "e8825dea53": "45 deg"}   # CLAUDE.md "Recorded data"
VARIANTS = ("b_level", *HYBRID)                 # b_level = M4's rows
SETTLES = ("settle_pregrasp", "settle_grasp", "settle_place")


def rate(rows: list[dict]) -> dict:
    k, n = sum(bool(r["success"]) for r in rows), len(rows)
    lo, hi = wilson(k, n)
    fails = [r for r in rows if not r["success"]]
    return dict(successes=k, n=n, wilson95=[round(lo, 3), round(hi, 3)], category=dict(Counter(r["category"] for r in fails).most_common()),
                category_v2=dict(Counter(category_v2(r) for r in fails).most_common()))


def by_trial(clip: str, variant: str, runs_from: dict) -> list[dict]:
    src = runs_from[variant]
    out = [src[run_name(clip, variant, f"trial{t:02d}")] for t in range(N_TRIALS)]
    assert [r["trial"] for r in out] == list(range(N_TRIALS))
    return out


def paired(a: list[dict], b: list[dict], names: tuple[str, str]) -> dict:
    """Same layouts in both lists (asserted): both succeed / only a / only b / neither."""
    for x, y in zip(a, b):
        assert x["trial"] == y["trial"] and x["cup_xy"] == y["cup_xy"] and x["coaster_xy"] == y["coaster_xy"], (x["trial"], y["trial"])
    sa, sb = [bool(x["success"]) for x in a], [bool(y["success"]) for y in b]
    return {"both": sum(x and y for x, y in zip(sa, sb)), f"{names[0]}_only": sum(x and not y for x, y in zip(sa, sb)),
            f"{names[1]}_only": sum(y and not x for x, y in zip(sa, sb)), "neither": sum(not x and not y for x, y in zip(sa, sb))}


def stats_mm(v: list[float]) -> dict | None:
    if not v:
        return None
    a = np.asarray(v, float)
    return dict(n=len(a), median=round(float(np.median(a)), 1), min=round(float(a.min()), 1), max=round(float(a.max()), 1),
                overlap=int((a < 0).sum()))


def collision(rows: list[dict]) -> dict:
    """min_mm of the collision diagnostic (negative = overlap): successes vs failures, and failures by category_v2."""
    mm = lambda r: r["collision_min"]["min_mm"]
    assert all(r["collision_min"] and r["collision_min"]["mapping_passed"] for r in rows), "a collision diagnostic is missing or unmapped"
    fails = [r for r in rows if not r["success"]]
    return dict(success=stats_mm([mm(r) for r in rows if r["success"]]), failure=stats_mm([mm(r) for r in fails]),
                failure_by_category_v2={c: stats_mm([mm(r) for r in fails if category_v2(r) == c]) for c in sorted({category_v2(r) for r in fails})})


def reach_settles(rows: list[dict]) -> dict:
    c = Counter(r["settles"][-1]["phase"] for r in rows if not r["success"] and r["failed_phase"] == "reach")
    return {k: c.get(k, 0) for k in SETTLES}


def video_picks(runs: dict[str, dict]) -> dict:
    """best_success = the first b_hybrid success (trial order) of the clip with the most b_hybrid successes (ties: CLIPS order);
    failure = the first b_hybrid failure (trial order, then CLIPS order) whose category_v2 is the most common among b_hybrid's
    failures. None if there is no such run."""
    hy = [r for r in runs.values() if r["variant"] == "b_hybrid"]
    wins = {c: sum(r["success"] for r in hy if r["clip"] == c) for c in CLIPS}
    best = max(CLIPS, key=lambda c: (wins[c], -CLIPS.index(c)))
    succ = sorted((r for r in hy if r["clip"] == best and r["success"]), key=lambda r: r["trial"])
    fails = [r for r in hy if not r["success"]]
    top = Counter(category_v2(r) for r in fails).most_common(1)[0][0] if fails else None
    fail = sorted((r for r in fails if category_v2(r) == top), key=lambda r: (r["trial"], CLIPS.index(r["clip"])))
    return dict(best_clip=best, best_clip_successes=wins[best], best_success=succ[0]["run"] if succ else None,
                failure_category_v2=top, failure=fail[0]["run"] if fail else None)


def sweep_plot(sweep: dict, phi_h: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from retarget.m4_summary import BLUE, ORANGE

    phis = [s["phi_deg"] for s in sweep]
    k = np.array([s["successes"] for s in sweep]); n = np.array([s["n"] for s in sweep])
    lo = np.array([s["wilson95"][0] for s in sweep]); hi = np.array([s["wilson95"][1] for s in sweep])
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.errorbar(phis, k / n, yerr=[k / n - lo, hi - k / n], color=BLUE, marker="o", capsize=4, lw=2, label="oracle mug_v1, grasp turned to phi (20 layouts each)")
    for i, (clip, phi) in enumerate(sorted(phi_h.items(), key=lambda x: x[1])):
        ax.axvline(phi, color=ORANGE, ls="--", lw=1)
        ax.text(phi, 1.02 + 0.06 * (i % 2), f"{clip[:5]} ({VIEW[clip]})", rotation=0, ha="center", va="bottom", fontsize=7, color=ORANGE)
    ax.set(xlabel="palm heading phi [deg] (oracle 45)", ylabel="success rate (Wilson 95 %)", ylim=(0, 1.0), xticks=phis)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Heading sweep; dashed = my clips' phi_h", fontsize=9, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_summarize(args):
    hyb = load_rows(args.runs)
    m4 = {n: r for n, r in load_rows(M4_DIR / "runs").items() if r["variant"] == "b_level" and r["trial"] is not None}
    expected = {run_name(c, v, f"trial{t:02d}") for c in CLIPS for v in HYBRID for t in range(N_TRIALS)}
    assert set(hyb) == expected, f"hybrid rows: missing {sorted(expected - set(hyb))[:5]}, extra {sorted(set(hyb) - expected)[:5]}"
    assert len(m4) == 200
    runs_from = {"b_level": m4, **{v: hyb for v in HYBRID}}
    oracle = json.loads(FIFTY.read_text())["runs"]
    assert [o["trial"] for o in oracle] == list(range(N_TRIALS))
    sw = json.loads(SWEEP.read_text())
    assert sw["phi45_check"] is True
    phi_h = {c: v["phi_h_deg"] for c, v in json.loads(HUMAN_VALUES.read_text())["clips"].items()}

    main = dict(oracle=rate(oracle))
    for v in VARIANTS:
        main[v] = {c: rate(by_trial(c, v, runs_from)) for c in CLIPS}
        main[v]["pool"] = rate([r for c in CLIPS for r in by_trial(c, v, runs_from)])
    pairs = {}
    for c in CLIPS:
        rows = {v: by_trial(c, v, runs_from) for v in VARIANTS}
        pairs[c] = {f"{v}_vs_oracle": paired(rows[v], oracle, (v, "oracle")) for v in VARIANTS}
        pairs[c]["b_hybrid_vs_b_level"] = paired(rows["b_hybrid"], rows["b_level"], ("b_hybrid", "b_level"))
        pairs[c]["b_hybrid_vs_b_hybrid_h45"] = paired(rows["b_hybrid"], rows["b_hybrid_h45"], ("b_hybrid", "b_hybrid_h45"))
    sweep = []
    for phi in sw["phis_deg"]:
        rs_ = [r for r in sw["runs"] if r["phi_deg"] == phi]
        assert [r["trial"] for r in rs_] == list(range(len(rs_)))
        sweep.append(dict(phi_deg=phi, **rate(rs_), reach_settle=dict(Counter(r["detail"]["settles"][-1]["phase"] for r in rs_ if r["failed_phase"] == "reach"))))
    sweep_plot(sweep, phi_h, args.out / "heading_sweep.png")
    coll = {v: collision([r for r in hyb.values() if r["variant"] == v]) for v in HYBRID}
    coll["b_level (M4; window ends 1 cm before G)"] = collision(list(m4.values()))
    s = dict(what="M4b Step 4 summary (retarget/m4b_summary.py summarize)",
             sources=dict(hybrid_rows=str(args.runs), m4_rows=str(M4_DIR / "runs"), oracle=str(FIFTY), heading_sweep=str(SWEEP), phi_h=str(HUMAN_VALUES)),
             definitions=dict(category_v2=f"'{PUSHED}' if the run failed before the close phase and |cup_final xy - cup_xy| > {PUSH_M} m, else category",
                              wilson="Wilson score interval, z = 1.96", collision_min="retarget/collision.py, mm, negative = overlap; hybrids: window ends 1 cm before PG",
                              categories=dict(CATEGORY)),
             m4_category_v2={f"{v}/{c}": recategorised([r for r in load_rows(M4_DIR / "runs").values() if r["variant"] == v and r["clip"] == c and r["trial"] is not None])
                             for v in ("b_level", "b_tilt") for c in [*CLIPS]},
             phi_h_deg=phi_h, main=main, paired=pairs, reach_settles={v: {c: reach_settles(by_trial(c, v, runs_from)) for c in CLIPS} for v in VARIANTS},
             heading_sweep=dict(phi45_reproduces_track_b=sw["phi45_check"], rows=sweep, plot="heading_sweep.png"), collision=coll,
             video_picks=video_picks(hyb), videos=json.loads(VIDEOS.read_text()) if VIDEOS.exists() else None,
             resources_runs=[json.loads(p.read_text()) for p in sorted(args.out.glob("run_resources_*.json"))], resources_sweep=sw["resources"])
    (args.out / "summary.json").write_text(json.dumps(s, indent=2))
    (args.out / "summary.md").write_text(render(s))
    print(render(s))


def ci(x: dict) -> str:
    return f"{x['successes']}/{x['n']} ({x['wilson95'][0]:.2f}-{x['wilson95'][1]:.2f})"


def cats(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "-"


def render(s: dict) -> str:
    m, L = s["main"], []
    L += ["# M4b results: robot-specific final approach + heading test", "",
          "Generated by `python -m retarget.m4b_summary summarize` from the saved rows (nothing simulated here). Profile mug_v1, my mug at "
          "0.40 kg, trials 0-49 (the oracle's 50 layouts). Wilson 95 % intervals; the pooled one treats the 200 runs as independent although "
          "they reuse the same 50 layouts, so the real uncertainty is larger. The place correction of every variant and of the oracle reads "
          "privileged simulator state (the cup's true pose in the hand).", "",
          f"category_v2 = {s['definitions']['category_v2']}.", "",
          "## Step 1: M4 re-categorised (category -> category_v2, trial runs)", "", "| variant / clip | failures | category | category_v2 |", "|---|---|---|---|"]
    for k, g in s["m4_category_v2"].items():
        L.append(f"| {k} | {g['failures']} | {cats(g['category'])} | {cats(g['category_v2'])} |")
    L += ["", "## Main table (success / 50, Wilson 95 %)", "", f"Oracle (Track B final mug_v1, fifty.json): **{ci(m['oracle'])}**; failures: {cats(m['oracle']['category'])} "
          f"(category_v2: {cats(m['oracle']['category_v2'])}).", "", "| clip | phi_h | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for c in [*CLIPS, "pool"]:
        head = f"| {c} ({VIEW[c]}) | {s['phi_h_deg'][c]:.1f} |" if c != "pool" else "| 4-clip pool | |"
        L.append(head + " | ".join(f" {ci(m[v][c])}" for v in VARIANTS) + " |")
    L += ["", "Failure categories (category / category_v2):", ""]
    for v in VARIANTS:
        for c in [*CLIPS, "pool"]:
            L.append(f"- {v} {c}: {cats(m[v][c]['category'])} / v2: {cats(m[v][c]['category_v2'])}")
    L += ["", "Reach failures by the settle that gave up (pre-grasp / grasp / place):", ""]
    for v in VARIANTS:
        L.append(f"- {v}: " + "; ".join(f"{c} {r['settle_pregrasp']}/{r['settle_grasp']}/{r['settle_place']}" for c, r in s["reach_settles"][v].items()))
    L += ["", "## Paired comparisons (same 50 layouts)", "", "| clip | comparison | both | first only | second only | neither |", "|---|---|---|---|---|---|"]
    for c, d in s["paired"].items():
        for name, p in d.items():
            vals = list(p.values())
            L.append(f"| {c} | {name.replace('_vs_', ' vs ')} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    L += ["", "## Heading sweep (oracle mug_v1, grasp turned about the cup axis; trials 0-19, handle yaw = trial + (phi - 45))", "",
          f"phi = 45 reproduces Track B final's trials 0-19 exactly: {s['heading_sweep']['phi45_reproduces_track_b']}. Clips' phi_h: "
          + ", ".join(f"{c} {v:.1f}" for c, v in s["phi_h_deg"].items()) + ". Plot: `heading_sweep.png`.", "",
          "| phi [deg] | success / 20 | category | category_v2 | reach: settle that gave up |", "|---|---|---|---|---|"]
    for r in s["heading_sweep"]["rows"]:
        L.append(f"| {r['phi_deg']} | {ci(r)} | {cats(r['category'])} | {cats(r['category_v2'])} | {cats(r['reach_settle'])} |")
    L += ["", "## Collision diagnostic (min hand-mug distance along the planned approach, mm; negative = overlap; information only)", "",
          "Hybrids: window from the prelude's end to 1 cm of arc before PG (PG -> G is the oracle's leg). M4 b_level for reference: its window "
          "ended 1 cm before G.", "", "| variant | successes: n, median (min..max), overlapping | failures: n, median (min..max), overlapping |", "|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x['n']}, {x['median']} ({x['min']}..{x['max']}), {x['overlap']}"
    for v, d in s["collision"].items():
        L.append(f"| {v} | {f(d['success'])} | {f(d['failure'])} |")
    for v, d in s["collision"].items():
        L.append(f"- {v} failures by category_v2: " + "; ".join(f"{c}: {f(x)}" for c, x in d["failure_by_category_v2"].items()))
    vp = s["video_picks"]
    L += ["", "## Videos", "", f"Picks (fixed rules, `video_picks`): best clip {vp['best_clip']} ({vp['best_clip_successes']}/50 b_hybrid successes): "
          f"success {vp['best_success']}; failure ({vp['failure_category_v2']}): {vp['failure']}."]
    if s["videos"]:
        for v in s["videos"]["videos"]:
            L.append(f"- `{v['video']}` (gitignored): saved {'success' if v['saved_success'] else v['saved_category']}, re-run identical to the row: {v['identical']}")
        if s["videos"]["gif"]:
            L.append(f"- GIF `{s['videos']['gif']['file']}`, {s['videos']['gif']['chosen']['size_bytes'] / 1e6:.2f} MB, from {s['videos']['gif']['run']}")
    else:
        L.append("- not rendered yet (`sim.m4_execute render`)")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m4 = sub.add_parser("m4-v2", help="Step 1: category_v2 of M4's rows")
    m4.add_argument("--runs", type=Path, default=M4_DIR / "runs")
    m4.add_argument("--out", type=Path, default=OUT)
    sm = sub.add_parser("summarize", help="Step 4: results/m4b/summary.md + summary.json + heading_sweep.png")
    sm.add_argument("--runs", type=Path, default=OUT / "runs")
    sm.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dict(summarize=cmd_summarize, **{"m4-v2": cmd_m4_v2})[args.cmd](args)


if __name__ == "__main__":
    main()
