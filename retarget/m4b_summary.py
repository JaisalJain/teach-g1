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
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sim.m4_execute import CLIPS, OUT_DIR as M4_DIR, VARIANTS as M4_VARIANTS

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
        settle = r["settles"][-1]["phase"]
        assert r["reason"].startswith(settle + ":"), f"{r['run']}: the last settle {settle} is not the failing one ({r['reason']})"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m4 = sub.add_parser("m4-v2", help="Step 1: category_v2 of M4's rows")
    m4.add_argument("--runs", type=Path, default=M4_DIR / "runs")
    m4.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dict(**{"m4-v2": cmd_m4_v2})[args.cmd](args)


if __name__ == "__main__":
    main()
