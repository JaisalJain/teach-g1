"""M4 Step 5: the results of the M4 runs, read back from the saved row files (nothing is simulated here).

Inputs (read only):
  results/m4_execute/runs/*.json                 604 rows: b_level / b_tilt / naive x 4 clips x trials 0-49, plus the 4
                                                 recorded_v2 runs (b_level). Written by ``python -m sim.m4_execute run``.
  results/m1_grasp_fix/v3_final/runs/fifty.json  the oracle: Track B final's 50 mug_v1 trials (same cup / coaster xy as trials 0-49)
  results/m3_retarget/v2/runs/oracle__oracle__trial00-09.json   the oracle's collision minimum (Step 0 rows, trials 0-9 only)
Outputs: results/m4_execute/summary.md + summary.json + failure_map_<variant>.png (one per variant).

What is reported (spec, Step 5):
  - main table: success / 50 with the Wilson 95 % CI for the oracle, each clip x variant, and the 4-clip pool of each variant;
    failure categories = M1's (sim/pick_place.py CATEGORY) plus "reach" (palm could not reach the target, a settle failure)
  - paired with the oracle, per clip and variant: both succeed / oracle only / variant only / both fail (same 50 layouts)
  - collision diagnostic (retarget/collision.py, information only): per-variant distribution of the minimum hand-mug distance
    for successes vs approach knocks (reach failures and the rest shown too, so no group is left out), and the oracle's
  - failure maps: cup xy coloured by outcome, one PNG per variant, with the PALM_BOX cup region; the cluster description is
    labelled "descriptive, found after looking". Its numbers are computed here (``regions``) and every sentence is asserted
    against them before it is written, so the text cannot drift from the rows
  - recorded_v2 outcomes per clip

Run from the repo root: env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.m4_summary
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from sim.m1_pick_mug import success_rate, wilson
from sim.m4_execute import CLIPS, FIFTY, MUG_MASS_KG, N_TRIALS, OUT_DIR, PRIVILEGED, PROFILE, VARIANTS, run_name
from sim.pick_place import CATEGORY, PALM_BOX, Grasp
from sim.scene import mug_config

ORACLE_COLLISION = Path("results/m3_retarget/v2/runs")           # oracle__oracle__trialNN.json (trials 0-9)
N_ORACLE_COLLISION = 10
B_VARIANTS = ("b_level", "b_tilt")
REACH = CATEGORY["reach"]
VIEW = {"26d4118b0c": "top-down", "4e103843db": "90 deg", "49aff04f7a": "45 deg", "e8825dea53": "45 deg"}   # CLAUDE.md "Recorded data"

# Failure maps. Categorical colours = the first three slots of the validated palette (dataviz skill: the first three validate
# all-pairs; a fourth fails the normal-vision floor), so more than three outcomes are told apart by marker shape too, and the
# legend lists every one. Colour follows the outcome, never its rank.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STYLE = {                           # outcome -> (colour, marker)
    "success": (BLUE, "o"),
    REACH: (ORANGE, "X"),
    CATEGORY["approach"]: (AQUA, "^"),                 # missed/knocked cup (the oracle's knocks are in its approach)
    CATEGORY["transport"]: (AQUA, "s"),                # dropped while moving
    CATEGORY["place"]: (AQUA, "D"),                    # released but not on the coaster
    CATEGORY["lift"]: (AQUA, "P"),                     # slip during lift
    CATEGORY["fell"]: (AQUA, "h"),                     # fell
}
# Descriptive split of the 50 cups, chosen after looking (labelled as such in the summary): the cups form two groups in y with
# an empty band between them (checked below), and the farthest ten cups (x >= X_FAR) behaved differently for e8825dea53 b_level.
Y_SPLIT, X_FAR = -0.25, 0.52


# ---------------------------------------------------------------------------------------------------------------- loading
def load_rows(runs_dir: Path) -> dict[str, dict]:
    rows = {}
    for f in sorted(runs_dir.glob("*.json")):
        r = json.loads(f.read_text())
        assert r["run"] == f.stem, f"{f}: row says {r['run']}"
        rows[f.stem] = r
    return rows


def expected_runs() -> list[str]:
    trial_runs = [run_name(c, v, f"trial{t:02d}") for c in CLIPS for v in VARIANTS for t in range(N_TRIALS)]
    return trial_runs + [run_name(c, "b_level", "recorded_v2") for c in CLIPS]


def check_rows(rows: dict[str, dict]) -> dict:
    """Completeness and consistency of the row set; stops the summary if anything is off (a partial table must never be written)."""
    want = expected_runs()
    missing, extra = [n for n in want if n not in rows], sorted(set(rows) - set(want))
    errors = [n for n, r in rows.items() if r.get("error") or r.get("success") is None]
    ok_rows = [r for n, r in rows.items() if n in want and n not in errors]
    b = [r for r in ok_rows if r["variant"] in B_VARIANTS]
    naive = [r for r in ok_rows if r["variant"] == "naive"]
    c = dict(expected=len(want), found=len(rows), missing=missing, extra=extra, error_rows=errors,
             paths_v2=all(r["paths"] == "v2" for r in ok_rows), profile=all(r["profile"] == PROFILE for r in ok_rows),
             mass_kg=all(r["mass_kg"] == MUG_MASS_KG for r in ok_rows),
             b_rows_with_collision_min=sum(isinstance(r.get("collision_min"), dict) and r["collision_min"].get("min_mm") is not None
                                           and r["collision_min"].get("mapping_passed") is True for r in b),
             b_rows=len(b), naive_rows_without_collision_min=sum(r.get("collision_min") is None for r in naive), naive_rows=len(naive))
    c["passed"] = (not missing and not extra and not errors and c["paths_v2"] and c["profile"] and c["mass_kg"]
                   and c["b_rows_with_collision_min"] == c["b_rows"] and c["naive_rows_without_collision_min"] == c["naive_rows"])
    if not c["passed"]:
        raise SystemExit(f"row check failed, nothing written: {json.dumps({k: v for k, v in c.items() if k != 'passed'}, indent=1)}")
    return c


def load_oracle() -> list[dict]:
    fifty = json.loads(FIFTY.read_text())
    runs = fifty["runs"]
    assert [r["trial"] for r in runs] == list(range(N_TRIALS)), "fifty.json is not trials 0-49 in order"
    s = success_rate(runs)
    assert s["wilson95"] == fifty["summary"]["wilson95"] and s["successes"] == fifty["summary"]["successes"], "fifty.json summary disagrees"
    return runs


def oracle_collision() -> dict[int, float]:
    """The oracle's collision minimum (mm) per trial, from the Step 0 rows (M3's harness on the oracle's nominal path)."""
    out = {}
    for t in range(N_ORACLE_COLLISION):
        r = json.loads((ORACLE_COLLISION / f"oracle__oracle__trial{t:02d}.json").read_text())
        out[t] = r["collision_min"]["min_mm"]
    return out


def trial_rows(rows: dict, clip: str, variant: str) -> list[dict]:
    return [rows[run_name(clip, variant, f"trial{t:02d}")] for t in range(N_TRIALS)]


def outcome(r: dict) -> str:
    return "success" if r["success"] else r["category"]


# ---------------------------------------------------------------------------------------------------------------- tables
def main_table(rows: dict, oracle: list[dict]) -> list[dict]:
    t = [dict(name="oracle (Track B final, mug_v1)", clip=None, variant="oracle", **success_rate(oracle))]
    for c in CLIPS:
        for v in VARIANTS:
            t.append(dict(name=f"{c} {v}", clip=c, variant=v, **success_rate(trial_rows(rows, c, v))))
    for v in VARIANTS:
        t.append(dict(name=f"4-clip pool {v}", clip="pool", variant=v, **success_rate([r for c in CLIPS for r in trial_rows(rows, c, v)])))
    return t


def paired_table(rows: dict, oracle: list[dict]) -> list[dict]:
    """Per clip x variant: the same 50 layouts, oracle vs variant (cup and coaster xy asserted equal; the handle yaw differs, see notes)."""
    out = []
    for c in CLIPS:
        for v in VARIANTS:
            rs = trial_rows(rows, c, v)
            for o, r in zip(oracle, rs):
                assert (o["cup_xy"], o["coaster_xy"]) == (r["cup_xy"], r["coaster_xy"]), f"{r['run']}: layout differs from the oracle's trial {o['trial']}"
            cell = lambda so, sv: [o["trial"] for o, r in zip(oracle, rs) if bool(o["success"]) == so and bool(r["success"]) == sv]
            both, only_o, only_v, none = cell(True, True), cell(True, False), cell(False, True), cell(False, False)
            assert len(both) + len(only_o) + len(only_v) + len(none) == N_TRIALS
            out.append(dict(clip=c, variant=v, both_succeed=len(both), oracle_only=len(only_o), variant_only=len(only_v), both_fail=len(none),
                            oracle_only_trials=only_o, variant_only_trials=only_v))
    return out


SETTLES = ("settle_pregrasp", "settle_grasp", "settle_place")


def reach_by_settle(rows: dict) -> list[dict]:
    """Which settle gave up, for the reach failures of each clip x variant (b trial runs): the failed settle is the failure's own
    last entry in ``settles`` and its reason's prefix (the two are asserted to agree)."""
    out = []
    for c in CLIPS:
        for v in B_VARIANTS:
            cnt = Counter()
            for r in trial_rows(rows, c, v):
                if r["failed_phase"] == "reach":
                    last = r["settles"][-1]["phase"]
                    assert r["reason"].startswith(last + ":") and last in SETTLES, f"{r['run']}: {r['reason']}"
                    cnt[last] += 1
            out.append(dict(clip=c, variant=v, **{k: cnt.get(k, 0) for k in SETTLES}))
    return out


def group_of(r: dict) -> str:
    if r["success"]:
        return "success"
    if r["failed_phase"] == "approach" and r["category"] == CATEGORY["approach"]:
        return "approach knock"
    if r["failed_phase"] == "reach":
        return "reach (a settle gave up)"
    return f"other failure: {r['category']}"


def stats_mm(v: list[float]) -> dict:
    a = np.asarray(v, float)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return dict(n=len(a), min=round(float(q[0]), 1), p25=round(float(q[1]), 1), median=round(float(q[2]), 1), p75=round(float(q[3]), 1),
                max=round(float(q[4]), 1), overlapping=int((a < 0).sum()))


def collision_table(rows: dict, oracle: list[dict]) -> dict:
    """Per b variant and outcome group, the distribution of the minimum signed hand-mug distance (mm, negative = overlap) over the
    approach window, on the 50-layout trial rows of all 4 clips. ``clips`` shows which clips fill the group (a group can come
    from one clip only). Oracle: its 10 Step-0 rows, outcome from fifty.json for the same trial."""
    per_variant = {}
    for v in B_VARIANTS:
        rs = [r for c in CLIPS for r in trial_rows(rows, c, v)]
        groups = {}
        for g in ("success", "approach knock", "reach (a settle gave up)"):
            sel = [r for r in rs if group_of(r) == g]
            groups[g] = dict(**stats_mm([r["collision_min"]["min_mm"] for r in sel]), clips=dict(Counter(r["clip"] for r in sel))) if sel else dict(n=0)
        other = [r for r in rs if group_of(r).startswith("other")]
        if other:
            groups["other failure (" + ", ".join(sorted({r["category"] for r in other})) + ")"] = dict(
                **stats_mm([r["collision_min"]["min_mm"] for r in other]), clips=dict(Counter(r["clip"] for r in other)))
        per_variant[v] = groups
    ocol = oracle_collision()
    by_group = {}
    for t, mm in ocol.items():
        by_group.setdefault(group_of_oracle(oracle[t]), []).append(mm)
    per_variant["oracle"] = {g: stats_mm(v) for g, v in by_group.items()}
    no_value = [o["trial"] for o in oracle if o["trial"] not in ocol and not o["success"]]
    return dict(per_variant=per_variant, oracle_trials_measured=sorted(ocol), oracle_values_mm={str(t): round(v, 2) for t, v in ocol.items()},
                oracle_failed_trials_without_value=no_value)


def group_of_oracle(o: dict) -> str:
    return "success" if o["success"] else ("approach knock" if o["failed_phase"] == "approach" and o["category"] == CATEGORY["approach"]
                                             else f"other failure: {o['category']}")


def recorded_v2_table(rows: dict) -> list[dict]:
    out = []
    for c in CLIPS:
        r = rows[run_name(c, "b_level", "recorded_v2")]
        d = r["detail"]
        entry = dict(clip=c, view=VIEW[c], success=r["success"], category=r["category"], failed_phase=r["failed_phase"], reason=r["reason"],
                     cup_xy=r["cup_xy"], coaster_xy=r["coaster_xy"], handle_yaw_deg=r["handle_yaw_deg"],
                     collision_min_mm=round(r["collision_min"]["min_mm"], 1))
        if r["success"]:
            entry.update(cup_off_coaster_center_cm=r["cup_off_coaster_center_cm"], lift_slip_in_hand_cm=r["lift_slip_in_hand_cm"],
                         lift_cup_rise_cm=r["lift_cup_rise_cm"], place_shift_mm=d["place_correction"]["shift_mm"])
        else:
            last = r["settles"][-1]
            entry.update(settle_phase=last["phase"], settle_palm_err_cm=last["palm_err_cm"], settle_seconds=last["seconds"])
        out.append(entry)
    return out


# ---------------------------------------------------------------------------------------------------------------- failure maps
def cup_region() -> tuple[tuple[float, float], tuple[float, float], list[float]]:
    """PALM_BOX translated to cup centres: a cup at c has its nominal grasp palm at c + legacy palm_offset[:2] (what sample_layouts uses)."""
    off = Grasp.hero_side_grasp(mug_config()["height_m"]).palm_offset[:2]
    return (PALM_BOX["x"][0] - off[0], PALM_BOX["x"][1] - off[0]), (PALM_BOX["y"][0] - off[1], PALM_BOX["y"][1] - off[1]), [float(off[0]), float(off[1])]


def draw_panel(ax, runs: list[dict], title: str):
    (x0, x1), (y0, y1), _ = cup_region()
    ax.add_patch(Rectangle((y1, x0), y0 - y1, x1 - x0, fill=False, ls="--", lw=1.0, ec="#6b6a66"))     # horizontal axis = y (inverted)
    for name, (colour, marker) in STYLE.items():
        sel = [r for r in runs if outcome(r) == name]
        if sel:
            xy = np.array([r["cup_xy"] for r in sel])
            ax.scatter(xy[:, 1], xy[:, 0], s=58, c=colour, marker=marker, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlim(-0.10, -0.37)                  # y inverted: the robot's left (+y) is on the left of the plot, x (forward) points up
    ax.set_ylim(0.425, 0.565)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, color="#d9d8d3", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def failure_map(variant: str, rows: dict, oracle: list[dict], path: Path) -> dict:
    """One PNG: a panel per clip (this variant) plus the oracle for reference. Returns what was drawn (counts per panel)."""
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 8.6), facecolor="white")
    axes = axes.ravel()
    drawn = {}
    for ax, c in zip(axes, CLIPS):
        rs = trial_rows(rows, c, variant)
        k = sum(bool(r["success"]) for r in rs)
        draw_panel(ax, rs, f"{c} ({VIEW[c]}): {k}/{N_TRIALS} succeed")
        drawn[c] = dict(Counter(outcome(r) for r in rs))
    k = sum(bool(o["success"]) for o in oracle)
    draw_panel(axes[4], oracle, f"oracle (reference, own handle yaws): {k}/{N_TRIALS} succeed")
    drawn["oracle"] = dict(Counter(outcome(o) for o in oracle))
    for ax in axes[:5]:
        ax.set_xlabel("y (m); the robot's left is to the left", fontsize=8)
        ax.set_ylabel("x (m), forward", fontsize=8)
    shown = [n for n in STYLE if any(n in d for d in drawn.values())]
    total = Counter()
    for d in drawn.values():
        total.update(d)
    handles = [Line2D([], [], ls="", marker=STYLE[n][1], ms=8, mfc=STYLE[n][0], mec="white", label=f"{n} ({total[n]})") for n in shown]
    handles.append(Line2D([], [], ls="--", color="#6b6a66", label="PALM_BOX cup region\n(box minus legacy palm offset)"))
    axes[5].axis("off")
    axes[5].legend(handles=handles, loc="upper left", fontsize=9, frameon=False, title="outcome (count over the 5 panels)", title_fontsize=9)
    fig.suptitle(f"{variant}: cup position vs outcome, trials 0-49 (the same 50 cups in every panel)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return drawn


def regions(rows: dict) -> dict:
    """Outcome counts of the b variants in four cup regions, per clip: y < Y_SPLIT ("outer": farther to the robot's right) / "inner"
    (nearer its midline; both groups have y < 0), and x < X_FAR ("near") / x >= X_FAR ("far"). The split values were chosen after
    looking at the maps; they are not a model."""
    layout = trial_rows(rows, CLIPS[0], "b_level")           # the 50 cups are the same for every clip and variant
    ys = sorted(r["cup_xy"][1] for r in layout)
    below, above = [y for y in ys if y < Y_SPLIT], [y for y in ys if y >= Y_SPLIT]
    out = dict(y_gap=[max(below), min(above)], n_outer=len(below), n_inner=len(above), n_far=sum(r["cup_xy"][0] >= X_FAR for r in layout), cells={})
    for v in B_VARIANTS:
        for c in CLIPS:
            cells = {}
            for r in trial_rows(rows, c, v):
                x, y = r["cup_xy"]
                key = f"{'outer' if y < Y_SPLIT else 'inner'}-{'far' if x >= X_FAR else 'near'}"
                cells.setdefault(key, Counter())[outcome(r)] += 1
            out["cells"][f"{c} {v}"] = {k: dict(sorted(cnt.items())) for k, cnt in sorted(cells.items())}
    return out


CELLS = ("outer-near", "outer-far", "inner-near", "inner-far")


def count(reg: dict, run: str, name: str, cells: tuple[str, ...] = CELLS) -> int:
    """Number of the run's cups with outcome ``name`` in the given regions."""
    return sum(reg["cells"][run].get(cell, {}).get(name, 0) for cell in cells)


def knock_text(rows: dict, clip: str, variant: str) -> str:
    xy = np.array([r["cup_xy"] for r in trial_rows(rows, clip, variant) if outcome(r) == CATEGORY["approach"]])
    span = lambda v: f"{v.min():.3f}" if v.min() == v.max() else f"{v.min():.3f} to {v.max():.3f}"
    return f"{clip} {len(xy)} (cup x {span(xy[:, 0])}, y {span(xy[:, 1])})"


def cluster_claims(rows: dict, reg: dict, oracle: list[dict]) -> list[tuple[str, bool]]:
    """The cluster description: (sentence, whether the statement in it holds). Written after looking at the maps. The numbers are
    computed from the rows; every check is asserted by the writer, so a sentence that stops being true when the rows change stops
    the summary instead of being written."""
    n = lambda run: sum(reg["cells"][run][cell].get("success", 0) for cell in reg["cells"][run])
    tot = lambda run, cells: sum(sum(reg["cells"][run].get(cell, {}).values()) for cell in cells)
    near, far, inner, outer = ("outer-near", "inner-near"), ("outer-far", "inner-far"), ("inner-near", "inner-far"), ("outer-near", "outer-far")
    e88l, e88t, a49l, a49t, c26t = ("e8825dea53 b_level", "e8825dea53 b_tilt", "49aff04f7a b_level", "49aff04f7a b_tilt", "26d4118b0c b_tilt")
    reach = [r for c in CLIPS for v in B_VARIANTS for r in trial_rows(rows, c, v) if r["failed_phase"] == "reach"]
    reach_err = [r["settles"][-1]["palm_err_cm"] for r in reach]
    o_knock = [o for o in oracle if not o["success"] and o["category"] == CATEGORY["approach"]]
    a49_knocks = [r for r in trial_rows(rows, "49aff04f7a", "b_tilt") if outcome(r) == CATEGORY["approach"]]
    other_knocks = [r for c in CLIPS for v in B_VARIANTS for r in trial_rows(rows, c, v) if outcome(r) == CATEGORY["approach"] and r not in a49_knocks]
    in_corner = lambda r: r["cup_xy"][1] < Y_SPLIT and r["cup_xy"][0] < max(o["cup_xy"][0] for o in o_knock) + 0.01      # outer group, x within 1 cm of the oracle's highest knock
    shared = [r["trial"] for r in a49_knocks if r["trial"] in {o["trial"] for o in o_knock}]
    naive = [r for c in CLIPS for r in trial_rows(rows, c, "naive")]
    moved = np.array([math.hypot(r["detail"]["cup_final"][0] - r["cup_xy"][0], r["detail"]["cup_final"][1] - r["cup_xy"][1]) for r in naive]) * 100
    off = np.array([r["cup_off_coaster_center_cm"] for r in naive])
    return [
        (f"Layout: the 50 cups form two groups in y with an empty band between them (no cup with y between {reg['y_gap'][0]:.3f} and {reg['y_gap'][1]:.3f} m): "
         f"{reg['n_outer']} cups with y < {Y_SPLIT} (\"outer\", farther to the robot's right) and {reg['n_inner']} nearer its midline (\"inner\"); "
         f"{reg['n_far']} cups lie at x >= {X_FAR} m (\"far\"). These are properties of the layouts, not of the runs.",
         reg["n_outer"] + reg["n_inner"] == N_TRIALS and reg["y_gap"][1] - reg["y_gap"][0] > 0.05),
        (f"Reach: the {len(reach)} reach failures of the b trial runs are failed settles: settle_pregrasp {sum(r['settles'][-1]['phase'] == 'settle_pregrasp' for r in reach)}, "
         f"settle_grasp {sum(r['settles'][-1]['phase'] == 'settle_grasp' for r in reach)}, settle_place {sum(r['settles'][-1]['phase'] == 'settle_place' for r in reach)} "
         f"(the palm was {min(reach_err)} to {max(reach_err)} cm from the target when the settle gave up).",
         len(reach) > 0 and all(r["settles"][-1]["phase"] in SETTLES for r in reach)),
        (f"b_level, 26d4118b0c and 4e103843db: reach failure at all 50 cups (the whole map is one outcome), and 4e103843db also at all 50 cups for b_tilt.",
         all(count(reg, f"{c} b_level", REACH) == N_TRIALS for c in ("26d4118b0c", "4e103843db")) and count(reg, "4e103843db b_tilt", REACH) == N_TRIALS),
        (f"b_level, e8825dea53 (the only clip with successes): the {reg['n_far']} far cups all succeeded ({count(reg, e88l, 'success', far)}/{tot(e88l, far)}); "
         f"the near cups (x < {X_FAR}) succeeded {count(reg, e88l, 'success', near)}/{tot(e88l, near)}, of them {count(reg, e88l, 'success', ('inner-near',))}/"
         f"{tot(e88l, ('inner-near',))} inner and {count(reg, e88l, 'success', ('outer-near',))}/{tot(e88l, ('outer-near',))} outer.",
         count(reg, e88l, "success", far) == tot(e88l, far) == reg["n_far"]),
        (f"b_level, 49aff04f7a: no success; reach failure at {count(reg, a49l, REACH)}/50 cups and the other {count(reg, a49l, CATEGORY['approach'])} runs are approach "
         f"knocks, all of them at far cups ({count(reg, a49l, CATEGORY['approach'], far)} of {tot(a49l, far)} far cups).",
         count(reg, a49l, "success") == 0 and count(reg, a49l, CATEGORY["approach"]) == count(reg, a49l, CATEGORY["approach"], far) > 0),
        (f"b_tilt, e8825dea53: its {n(e88t)} successes are all in the outer group ({count(reg, e88t, 'success', outer)}/{tot(e88t, outer)} outer cups; "
         f"{count(reg, e88t, 'success', inner)}/{tot(e88t, inner)} inner cups).", count(reg, e88t, "success", inner) == 0 < count(reg, e88t, "success", outer)),
        (f"b_tilt, 49aff04f7a and 26d4118b0c: the successes are mostly in the inner group (49aff04f7a {count(reg, a49t, 'success', inner)}/{tot(a49t, inner)} inner vs "
         f"{count(reg, a49t, 'success', outer)}/{tot(a49t, outer)} outer; 26d4118b0c {count(reg, c26t, 'success', inner)}/{tot(c26t, inner)} inner vs "
         f"{count(reg, c26t, 'success', outer)}/{tot(c26t, outer)} outer). So the side on which b_tilt succeeds differs between clips.",
         count(reg, a49t, "success", inner) > count(reg, a49t, "success", outer) and count(reg, c26t, "success", inner) > count(reg, c26t, "success", outer)),
        ("Approach knocks, where the cup was hit before the grasp (cup xy of the knocked cups): "
         f"b_level: {knock_text(rows, '49aff04f7a', 'b_level')}; {knock_text(rows, 'e8825dea53', 'b_level')}. "
         f"b_tilt: {knock_text(rows, '26d4118b0c', 'b_tilt')}; {knock_text(rows, '49aff04f7a', 'b_tilt')}. "
         f"No other clip-variant has a knock, and no knock occurs at the close.",
         sum(r["category"] == CATEGORY["approach"] for c in CLIPS for v in B_VARIANTS for r in trial_rows(rows, c, v))
         == sum(outcome(r) == CATEGORY["approach"] and r["failed_phase"] == "approach" for c in CLIPS for v in B_VARIANTS for r in trial_rows(rows, c, v)) == 12),
        (f"The oracle's own {len(o_knock)} approach knocks (trials {', '.join(str(o['trial']) for o in o_knock)}) all lie in the outer group at the low-x edge "
         f"(cup x {min(o['cup_xy'][0] for o in o_knock):.3f} to {max(o['cup_xy'][0] for o in o_knock):.3f}). In that same corner b_tilt 49aff04f7a knocked the cup at "
         f"trials {', '.join(str(r['trial']) for r in a49_knocks)}, of which {', '.join(str(t) for t in shared) or 'none'} are also oracle knock trials; the "
         f"other b knocks are elsewhere.",
         len(o_knock) > 0 and all(o["cup_xy"][1] < Y_SPLIT for o in o_knock) and all(in_corner(r) for r in a49_knocks) and not any(in_corner(r) for r in other_knocks)),
        (f"b_tilt, e8825dea53 also has 2 'dropped while moving' (the cup slipped in the hand during the transport, trials "
         f"{', '.join(str(r['trial']) for r in trial_rows(rows, 'e8825dea53', 'b_tilt') if r['category'] == CATEGORY['transport'])}).",
         sum(r["category"] == CATEGORY["transport"] for c in CLIPS for v in B_VARIANTS for r in trial_rows(rows, c, v)) == 2),
        (f"naive: the map has no structure, because all {len(naive)} runs fail the same way (released but not on the coaster) at every cup position. "
         f"The cup ended {off.min():.1f} to {off.max():.1f} cm (median {np.median(off):.1f}) from the coaster centre and had moved "
         f"{moved.min():.1f} to {moved.max():.1f} cm (median {np.median(moved):.1f}) from its start in xy.",
         len(naive) == 200 and all(not r["success"] for r in naive)),
    ]


# ---------------------------------------------------------------------------------------------------------------- summarize
def fmt_ci(ci) -> str:
    return f"{ci[0]:.3f}-{ci[1]:.3f}"


def fmt_cats(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "-"


def cell_str(cell: dict | None) -> str:
    return "-" if not cell else ", ".join(f"{k} {v}" for k, v in cell.items())


def render(s: dict) -> str:
    L = []
    c = s["checks"]
    L += ["# M4 results: the retargeted paths executed by HERO (mug_v1)", "",
          f"Made by `python -m retarget.m4_summary` from the saved rows ({c['found']} row files in results/m4_execute/runs/; the oracle is "
          f"{FIFTY}, not rerun). Nothing is simulated by this script; every number below is in summary.json.", "",
          "Variants: **b_level** and **b_tilt** = object-aware paths in four slices with the oracle's settles and grasp checks between them; "
          "**naive** = the mapped human path played in one go with its own finger closure, no settle, no grasp check, no place correction. "
          "Success is the oracle's own `evaluate()` (cup upright, on the coaster within 5 cm, at rest on it, robot never fell). "
          "Failure categories are M1's plus **reach** = `palm could not reach the target` (a failed settle). Layouts: trials 0-49, the same cup and "
          "coaster positions for the oracle and every variant.", ""]

    L += ["## Headline (read off the tables below)", ""]
    pool = {r["variant"]: r for r in s["main_table"] if r["clip"] == "pool"}
    o = s["main_table"][0]
    L += [f"- Oracle (Track B final): {o['successes']}/{o['trials']}, Wilson 95 % {fmt_ci(o['wilson95'])}.",
          *[f"- {v}: {p['successes']}/{p['trials']} over the 4 clips, Wilson 95 % {fmt_ci(p['wilson95'])} (pooled; see caveat 1)." for v, p in pool.items()],
          f"- Reach failures: {sum(r['failures_by_category'].get(REACH, 0) for r in s['main_table'] if r['clip'] in CLIPS and r['variant'] in B_VARIANTS)} of the "
          f"{sum(r['trials'] for r in s['main_table'] if r['clip'] in CLIPS and r['variant'] in B_VARIANTS)} b trial runs.", ""]

    L += ["## Checks", "",
          f"- Row set: {c['found']}/{c['expected']} rows present, {len(c['missing'])} missing, {len(c['extra'])} extra, {len(c['error_rows'])} error rows.",
          f"- Every row: paths v2 {c['paths_v2']}, profile {PROFILE} {c['profile']}, mug mass {MUG_MASS_KG} kg {c['mass_kg']}.",
          f"- Collision minimum present with the IK mapping check passed: {c['b_rows_with_collision_min']}/{c['b_rows']} b rows; "
          f"absent as specified on {c['naive_rows_without_collision_min']}/{c['naive_rows']} naive rows.",
          f"- Oracle: {s['oracle_check']}", ""]

    L += ["## Main table: success over 50 layouts", "",
          "| run | view | success | Wilson 95 % CI | failures (category n) |", "|---|---|---|---|---|"]
    for r in s["main_table"]:
        view = VIEW.get(r["clip"], "-")
        label = f"**{r['name']}**" if r["clip"] in (None, "pool") else r["name"]
        L.append(f"| {label} | {view} | {r['successes']}/{r['trials']} | {fmt_ci(r['wilson95'])} | {fmt_cats(r['failures_by_category'])} |")
    L += ["", "Categories: `palm could not reach the target` = reach (a settle did not converge, see the next table); `missed/knocked cup` = knocked the cup in the approach "
          "(every one of them: failed phase approach); `dropped while moving` = the cup slipped in the hand during the transport; "
          "`released but not on the coaster` = naive's only failure.", ""]

    L += ["### Which settle gave up (reach failures of the b trial runs)", "",
          "| clip | variant | settle_pregrasp | settle_grasp | settle_place | total |", "|---|---|---|---|---|---|"]
    for r in s["reach_by_settle"]:
        L.append(f"| {r['clip']} | {r['variant']} | {r['settle_pregrasp']} | {r['settle_grasp']} | {r['settle_place']} | "
                 f"{r['settle_pregrasp'] + r['settle_grasp'] + r['settle_place']} |")
    L += [""]

    L += ["## Paired with the oracle (same 50 layouts)", "",
          "| clip | variant | both succeed | oracle only | variant only | both fail |", "|---|---|---|---|---|---|"]
    for p in s["paired"]:
        L.append(f"| {p['clip']} | {p['variant']} | {p['both_succeed']} | {p['oracle_only']} | {p['variant_only']} | {p['both_fail']} |")
    L += ["", f"Trial numbers of the oracle-only and variant-only cells are in summary.json (`paired`). The oracle's {len(s['oracle_failed_trials'])} failures are trials "
          f"{', '.join(str(t) for t in s['oracle_failed_trials'])}.", ""]

    L += ["## Collision diagnostic (kinematic, information only, never a gate)", "",
          "Minimum signed distance (mm; negative = overlap) between the right-hand geoms and the mug (body and handle) over the approach window "
          "(from the prelude's end to 1 cm of arc length before G), on M3's single-plan IK clip of each run's path (retarget/collision.py). "
          "Trial rows of the 4 clips; `clips` = which clips fill the group. The window stops 1 cm before G, so a late overlap can be the "
          "intended grasp contact as well as a knock; the diagnostic cannot tell them apart.", "",
          "| variant | outcome group | n | min | p25 | median | p75 | max | overlapping (< 0) | clips |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, groups in s["collision"]["per_variant"].items():
        for g, d in groups.items():
            if d["n"] == 0:
                L.append(f"| {v} | {g} | 0 | - | - | - | - | - | - | - |")
                continue
            clips = ", ".join(f"{k} {n}" for k, n in d["clips"].items()) if "clips" in d else f"trials 0-9"
            L.append(f"| {v} | {g} | {d['n']} | {d['min']} | {d['p25']} | {d['median']} | {d['p75']} | {d['max']} | {d['overlapping']} | {clips} |")
    col = s["collision"]
    oracle_groups = ", ".join(f"{g} {d['n']}" for g, d in col["per_variant"]["oracle"].items())
    L += ["", f"Oracle: the Step 0 rows only (trials {col['oracle_trials_measured'][0]}-{col['oracle_trials_measured'][-1]}, the oracle's nominal path through the same "
          f"function); their outcomes in fifty.json: {oracle_groups}. Its failed trials {col['oracle_failed_trials_without_value']} have no value (not measured). "
          "naive: not computed (Step 3 is for b runs).", ""]

    L += ["## Failure maps", ""]
    for v in VARIANTS:
        L += [f"![{v}]({s['failure_maps'][v]['png']})", ""]
    L += ["Cup xy (sim world, x forward, y to the robot's left) coloured by outcome (blue success, orange reach, aqua any other failure; the marker "
          "shape names the failure, see the legend); one panel per clip and the oracle for reference; dashed rectangle = "
          f"the PALM_BOX cup region (PALM_BOX minus the legacy palm offset {s['cup_region']['palm_offset_xy']}).", "",
          "### Cluster description (descriptive, found after looking)", "",
          f"Written after looking at the maps and at the counts below; the splits (y < {Y_SPLIT} m, x >= {X_FAR} m) were chosen after looking and are "
          "not a model. Every number in these sentences is asserted against the rows when this file is written.", ""]
    L += [f"- {text}" for text in s["cluster_description"]]
    L += ["", "Outcome counts in the four cup regions (S = success; r = reach; K = missed/knocked cup; D = dropped while moving):", "",
          f"| run | outer, x < {X_FAR} | outer, x >= {X_FAR} | inner, x < {X_FAR} | inner, x >= {X_FAR} |", "|---|---|---|---|---|"]
    short = {"success": "S", REACH: "r", CATEGORY["approach"]: "K", CATEGORY["transport"]: "D"}
    reg = s["regions"]
    for run, cells in reg["cells"].items():
        L.append(f"| {run} | " + " | ".join(", ".join(f"{short[k]} {n}" for k, n in cells.get(cell, {}).items()) or "-"
                                             for cell in CELLS) + " |")
    L += ["", f"Region sizes over the 50 cups: outer (y < {Y_SPLIT}) {reg['n_outer']}, inner {reg['n_inner']}, far (x >= {X_FAR}) {reg['n_far']}.", ""]

    L += ["## Recorded layout v2 (b_level, one run per clip)", "",
          "Cup at the PALM_BOX centre minus the legacy palm offset minus half my coaster offset; coaster = cup + my offset (both nominal palms centred on the box).", "",
          "| clip | view | outcome | detail | cup xy | coaster xy | collision min (mm) |", "|---|---|---|---|---|---|---|"]
    for r in s["recorded_v2"]:
        if r["success"]:
            detail = (f"cup {r['cup_off_coaster_center_cm']} cm from the coaster centre, slip in hand {r['lift_slip_in_hand_cm']} cm, "
                      f"rise {r['lift_cup_rise_cm']} cm, place correction {r['place_shift_mm']} mm")
            res = "success"
        else:
            detail = r["reason"]
            res = f"fail: {r['category']}"
        L.append(f"| {r['clip']} | {r['view']} | {res} | {detail} | ({r['cup_xy'][0]:.3f}, {r['cup_xy'][1]:.3f}) | "
                 f"({r['coaster_xy'][0]:.3f}, {r['coaster_xy'][1]:.3f}) | {r['collision_min_mm']} |")
    L += ["", "## Caveats", ""]
    L += [f"{i}. {text}" for i, text in enumerate(s["caveats"], 1)]
    return "\n".join(L) + "\n"


def caveats() -> list[str]:
    return [
        "The 4-clip pool's Wilson interval treats its 200 runs as independent. They are not: the 4 clips reuse the same 50 cup and coaster positions, "
        "and the clips are 4 separate executions (n = 4), so the real uncertainty is larger than the interval says.",
        "Viewpoint and execution vary together (each clip is a different view of a different execution), so a difference between clips cannot be "
        "attributed to the viewpoint.",
        "The oracle runs the trial's own handle yaw, the variants run the trial's yaw plus (phi_h - phi_ora) for that clip (as in M3). The oracle grasps the "
        "body and ignores the handle, so this changes the handle's position only, but it means the paired runs are not identical in every input.",
        f"The place correction of the b variants uses privileged simulator state, as the oracle's place target does: {PRIVILEGED.split(': ', 1)[1]}.",
        "The oracle's collision values come from 10 runs (trials 0-9) of the oracle's nominal path through M3's harness; they are not a recording from the "
        "Track B physics runs, and the oracle's failed trials were not measured.",
        "The naive variant replays at a fixed recorded-layout translation (M3), so it ignores where the trial's cup and coaster are by construction.",
    ]


def summarize(args):
    rows = load_rows(args.out / "runs")
    checks = check_rows(rows)
    oracle = load_oracle()
    s = dict(what="M4 Step 5 summary", sources=dict(rows=str(args.out / "runs"), oracle=str(FIFTY), oracle_collision=str(ORACLE_COLLISION)),
             definitions=dict(success="sim/pick_place.py evaluate(): cup tilt < 10 deg, cup body centre within 5 cm of the coaster centre, "
                                      "resting on it (|dz| < 4 mm), robot never fell", wilson="Wilson score interval, z = 1.96",
                              categories={k: v for k, v in CATEGORY.items()}, approach_knock="failed_phase approach, category missed/knocked cup",
                              collision_min="retarget/collision.py collision_min, mm, negative = overlap, information only"),
             checks=checks,
             oracle_check=f"fifty.json = trials 0-49 in order, its saved summary equals the recomputed one ({success_rate(oracle)['successes']}/50); the cup and "
                          f"coaster xy of every variant row equal the oracle's trial row (asserted in the paired table).")
    s["oracle_failed_trials"] = [o["trial"] for o in oracle if not o["success"]]
    s["main_table"] = main_table(rows, oracle)
    s["reach_by_settle"] = reach_by_settle(rows)
    s["paired"] = paired_table(rows, oracle)
    s["collision"] = collision_table(rows, oracle)
    s["recorded_v2"] = recorded_v2_table(rows)
    (x0, x1), (y0, y1), off = cup_region()
    s["cup_region"] = dict(palm_box=PALM_BOX, palm_offset_xy=[round(off[0], 4), round(off[1], 4)], x=[round(x0, 4), round(x1, 4)], y=[round(y0, 4), round(y1, 4)])
    s["failure_maps"] = {}
    for v in VARIANTS:
        png = f"failure_map_{v}.png"
        s["failure_maps"][v] = dict(png=png, drawn=failure_map(v, rows, oracle, args.out / png))
    s["regions"] = regions(rows)
    claims = cluster_claims(rows, s["regions"], oracle)
    bad = [text for text, holds in claims if not holds]
    if bad:
        raise SystemExit("cluster description no longer matches the rows, nothing written (rewrite it after looking again):\n  " + "\n  ".join(bad))
    s["cluster_description"] = [text for text, _ in claims]
    s["caveats"] = caveats()
    s["runs"] = [dict(run=r["run"], clip=r["clip"], variant=r["variant"], layout=r["layout"], cup_xy=r["cup_xy"], coaster_xy=r["coaster_xy"],
                      handle_yaw_deg=r["handle_yaw_deg"], success=r["success"], category=r["category"], failed_phase=r["failed_phase"],
                      reason=r["reason"], collision_min_mm=None if r["collision_min"] is None else round(r["collision_min"]["min_mm"], 2))
                 for r in (rows[n] for n in expected_runs())]
    (args.out / "summary.json").write_text(json.dumps(s, indent=2))
    (args.out / "summary.md").write_text(render(s))
    print(f"wrote {args.out / 'summary.md'}, {args.out / 'summary.json'} and {len(VARIANTS)} failure maps")
    for r in s["main_table"]:
        print(f"{r['name']:<36} {r['successes']:>3}/{r['trials']:<3} {fmt_ci(r['wilson95'])}  {fmt_cats(r['failures_by_category'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    summarize(ap.parse_args())


if __name__ == "__main__":
    main()
