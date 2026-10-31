"""Polish Step 1: naive, b_level and b_tilt at each layout's own handle yaw, read back from the saved rows (nothing is simulated here).

Why: M4 ran these three variants with handle yaw = trial yaw + (phi_h - phi_ora): the handle turned with my grasp heading. The
oracle, b_hybrid_h45 and every M5 row use the trial's own yaw. The rows read here (``sim.m4_execute run --own-yaw``) use the
trial's own yaw too, so every row of the README's table runs the same layouts, handle included.

Inputs (read only): results/final/own_yaw/runs/*.json (600 rows: 3 variants x 4 clips x trials 0-49), results/m4_execute/runs/
(the same runs at M4's turned yaw), results/m1_grasp_fix/v3_final/runs/fifty.json (the oracle), the run_resources file(s) of the batch.
Outputs: <out>/summary.md + summary.json (default results/final/own_yaw/).
  - Branch E1: if more than 1 % of the rows are exception rows, only the E1 record is written and the script exits 1.
  - Per clip and pooled: successes with the Wilson 95 % interval; failures by category and by category_v2 (retarget/m4b_summary.py).
  - Paired, same trial: against the oracle, and against the same variant's M4 row (both / own yaw only / other only / neither,
    with the trial numbers of the discordant layouts).
The pooled interval treats the 200 runs as independent although the 4 clips reuse the same 50 layouts.

Run from the repo root: env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.own_yaw_summary [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from retarget.m4b_summary import category_v2
from sim.m1_pick_mug import wilson
from sim.m4_execute import CLIPS, FIFTY, M4_VARIANTS, N_TRIALS, OUT_DIR as M4_DIR, run_name

OWN = Path("results/final/own_yaw")
E1_MAX = 0.01                      # spec: stop if more than 1 % of the runs raised exceptions
VIEW = {"26d4118b0c": "top-down", "4e103843db": "90 deg", "49aff04f7a": "45 deg", "e8825dea53": "45 deg"}   # CLAUDE.md "Recorded data"
TRIALS = [f"trial{t:02d}" for t in range(N_TRIALS)]


def load(runs_dir: Path) -> dict[str, dict]:
    return {r["run"]: r for r in (json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json")))}


def rate(rows: list[dict]) -> dict:
    k, n = sum(bool(r["success"]) for r in rows), len(rows)
    lo, hi = wilson(k, n)
    fails = [r for r in rows if not r["success"]]
    return dict(successes=k, n=n, wilson95=[round(lo, 3), round(hi, 3)], text=f"{k}/{n} ({lo:.2f}-{hi:.2f})",
                category=dict(Counter(r["category"] for r in fails).most_common()),
                category_v2=dict(Counter(category_v2(r) for r in fails).most_common()))


def paired(a: list[dict], b: list[dict]) -> dict:
    """a, b: rows of the same 50 trials in trial order."""
    sa, sb = [bool(r["success"]) for r in a], [bool(r["success"]) for r in b]
    return dict(both=sum(x and y for x, y in zip(sa, sb)), first_only=sum(x and not y for x, y in zip(sa, sb)),
                second_only=sum(y and not x for x, y in zip(sa, sb)), neither=sum(not x and not y for x, y in zip(sa, sb)),
                first_only_trials=[t for t, (x, y) in enumerate(zip(sa, sb)) if x and not y],
                second_only_trials=[t for t, (x, y) in enumerate(zip(sa, sb)) if y and not x])


def summarize(out: Path) -> dict:
    own, m4 = load(OWN / "runs"), load(M4_DIR / "runs")
    oracle = json.loads(FIFTY.read_text())["runs"]
    assert [o["trial"] for o in oracle] == list(range(N_TRIALS)), "fifty.json is not trials 0-49 in order"
    expected = {run_name(c, v, t) for c in CLIPS for v in M4_VARIANTS for t in TRIALS}
    assert set(own) == expected, f"own-yaw rows: missing {sorted(expected - set(own))[:5]}, extra {sorted(set(own) - expected)[:5]}"
    errors = sorted(n for n, r in own.items() if r.get("success") is None)
    e1 = dict(rows=len(own), exceptions=len(errors), exception_runs=errors, fraction=round(len(errors) / len(own), 4),
              stop_above=E1_MAX, branch="STOP" if len(errors) / len(own) > E1_MAX else "continue")
    out.mkdir(parents=True, exist_ok=True)
    if e1["branch"] == "STOP":
        (out / "summary.json").write_text(json.dumps(dict(E1=e1), indent=2) + "\n")
        raise SystemExit(f"E1: {len(errors)}/{len(own)} runs raised exceptions (> {E1_MAX:.0%}): STOP")
    assert not errors, f"{len(errors)} exception rows (under the E1 limit): decide how to count them before summarizing: {errors}"

    # every own-yaw row: the trial's own yaw (= the oracle's), the same cup / coaster as the oracle and the M4 row, the M4 setup
    turn = {}
    for c in CLIPS:
        for v in M4_VARIANTS:
            for t, name in enumerate(TRIALS):
                r, m, o = own[run_name(c, v, name)], m4[run_name(c, v, name)], oracle[t]
                assert r["own_yaw"] is True and r["trial"] == t and r["paths"] == "v2" and r["profile"] == "mug_v1" and r["mass_kg"] == 0.4, r["run"]
                assert r["handle_yaw_deg"] == r["trial_handle_yaw_deg"] == o["cup_yaw_deg"], r["run"]
                assert r["cup_xy"] == m["cup_xy"] == o["cup_xy"] and r["coaster_xy"] == m["coaster_xy"] == o["coaster_xy"], r["run"]
                assert "own_yaw" not in m and m["trial_handle_yaw_deg"] == r["trial_handle_yaw_deg"], m["run"]
                turn.setdefault(c, set()).add(round(m["handle_yaw_deg"] - r["handle_yaw_deg"], 1))
    checks = dict(rows=len(own), every_row="own_yaw true; handle_yaw_deg = trial_handle_yaw_deg = the oracle's cup_yaw_deg; cup and coaster xy = "
                                           "the oracle's and the M4 row's; paths v2, profile mug_v1, mass 0.4 kg (asserted)",
                  m4_turn_deg={c: sorted(v) for c, v in turn.items()})

    rows = lambda rs, c, v: [rs[run_name(c, v, t)] for t in TRIALS]
    table = {v: {**{c: rate(rows(own, c, v)) for c in CLIPS}, "pool": rate([r for c in CLIPS for r in rows(own, c, v)])} for v in M4_VARIANTS}
    m4_table = {v: {**{c: rate(rows(m4, c, v)) for c in CLIPS}, "pool": rate([r for c in CLIPS for r in rows(m4, c, v)])} for v in M4_VARIANTS}
    vs_oracle = {v: {c: paired(rows(own, c, v), oracle) for c in CLIPS} for v in M4_VARIANTS}
    vs_m4 = {v: {c: paired(rows(own, c, v), rows(m4, c, v)) for c in CLIPS} for v in M4_VARIANTS}
    for d in (vs_oracle, vs_m4):
        for v in M4_VARIANTS:
            d[v]["pool"] = {k: sum(d[v][c][k] for c in CLIPS) for k in ("both", "first_only", "second_only", "neither")}
    o = rate(oracle)
    s = dict(what="Polish Step 1: naive / b_level / b_tilt at each layout's own handle yaw (retarget/own_yaw_summary.py)", E1=e1, checks=checks,
             oracle=o, own_yaw=table, m4_turned_yaw=m4_table, paired_vs_oracle=vs_oracle, paired_vs_m4=vs_m4,
             resources=[json.loads(p.read_text()) for p in sorted(OWN.glob("run_resources_*.json"))],
             wilson="Wilson score interval, z = 1.96; the pooled one treats the 200 runs as independent (they reuse the same 50 layouts)")
    (out / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    (out / "summary.md").write_text(render(s))
    return s


def render(s: dict) -> str:
    cats = lambda d: ", ".join(f"{k} {v}" for k, v in d.items()) or "-"
    L = ["# naive, b_level and b_tilt at each layout's own handle yaw", "",
         "Made by `python -m retarget.own_yaw_summary` from the saved rows (nothing is simulated here). Rows: `results/final/own_yaw/runs/` "
         "(`python -m sim.m4_execute run --own-yaw`). M4 ran these variants with the handle turned with my grasp heading (trial yaw + "
         "(phi_h - phi_ora)); here the handle has the trial's own yaw, as for the oracle, b_hybrid_h45 and every M5 run. The paths and "
         "everything else are M4's. Success / 50 per clip, / 200 pooled, Wilson 95 %; the pooled interval treats the 200 runs as "
         "independent although the 4 clips reuse the same 50 layouts, so the real uncertainty is larger.", "",
         f"- E1 (stop if more than 1 % of the runs raised exceptions): {s['E1']['exceptions']}/{s['E1']['rows']} exceptions, branch **{s['E1']['branch']}**.",
         f"- Checks: {s['checks']['rows']} rows; {s['checks']['every_row']}.",
         "- M4's handle turn per clip (M4 yaw - own yaw, deg): " + "; ".join(f"{c} {v}" for c, v in s["checks"]["m4_turn_deg"].items()) + ".",
         f"- Oracle (Track B final, fifty.json): {s['oracle']['text']}.", "",
         "## Success", "", "| variant | " + " | ".join(f"{c} ({VIEW[c]})" for c in CLIPS) + " | 4-clip pool |", "|---|---|---|---|---|---|"]
    for v in M4_VARIANTS:
        t = s["own_yaw"][v]
        L.append(f"| {v} | " + " | ".join(f"{t[c]['successes']}/50" for c in CLIPS) + f" | {t['pool']['text']} |")
    L += ["", "The same variants in M4 (handle turned with the grasp), for comparison:", "",
          "| variant (M4) | " + " | ".join(CLIPS) + " | 4-clip pool |", "|---|---|---|---|---|---|"]
    for v in M4_VARIANTS:
        t = s["m4_turned_yaw"][v]
        L.append(f"| {v} | " + " | ".join(f"{t[c]['successes']}/50" for c in CLIPS) + f" | {t['pool']['text']} |")
    L += ["", "## Failures", "", "category_v2 = 'pushed mug before close' if the run failed before the close and the mug ended more than 2 cm "
          "from its layout position, else the run's category (retarget/m4b_summary.py).", "",
          "| variant | clip | failures | category | category_v2 |", "|---|---|---|---|---|"]
    for v in M4_VARIANTS:
        for c in (*CLIPS, "pool"):
            x = s["own_yaw"][v][c]
            L.append(f"| {v} | {c} | {x['n'] - x['successes']} | {cats(x['category'])} | {cats(x['category_v2'])} |")
    for key, other, title in (("paired_vs_oracle", "oracle", "Paired with the oracle (same 50 trials)"),
                              ("paired_vs_m4", "M4 (turned handle)", "Paired with the same variant's M4 row (same 50 trials)")):
        L += ["", f"## {title}", "", f"| variant | clip | both succeed | own yaw only | {other} only | neither | own yaw only (trials) | {other} only (trials) |",
              "|---|---|---|---|---|---|---|---|"]
        for v in M4_VARIANTS:
            for c in (*CLIPS, "pool"):
                p = s[key][v][c]
                L.append(f"| {v} | {c} | {p['both']} | {p['first_only']} | {p['second_only']} | {p['neither']} | "
                         f"{p.get('first_only_trials', '') if c != 'pool' else ''} | {p.get('second_only_trials', '') if c != 'pool' else ''} |")
    for r in s["resources"]:
        L += ["", f"Batch: {r['items']} runs, {r['workers']} workers, {r['wall_s']} s, {len(r['errors'])} errors, peak RSS per process "
              f"{min(r['per_process_peak_rss_mb'])}-{max(r['per_process_peak_rss_mb'])} MB, {r['gpu']}."]
    L += ["", "The place correction reads privileged simulator state (the cup's true pose in the hand), as in M4 and for the oracle."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OWN)
    s = summarize(ap.parse_args().out)
    for v in M4_VARIANTS:
        print(f"{v:<8} own yaw " + ", ".join(f"{c[:5]} {s['own_yaw'][v][c]['successes']}" for c in CLIPS) + f", pool {s['own_yaw'][v]['pool']['text']}"
              f"  (M4 pool {s['m4_turned_yaw'][v]['pool']['text']})")


if __name__ == "__main__":
    main()
