"""M6 figures for the README, drawn only from saved summaries (nothing simulated here).

  results/final/ablation.png  one consistent set of bars, every run at each test layout's own handle yaw: naive, b_level, b_tilt
                              (results/final/own_yaw/summary.json, the polish re-run), b_hybrid at the original yaw, b_hybrid_h45, F, R1,
                              R2, F noisy, R2 noisy (results/m5/summary.json "ablation"): success rate with its Wilson 95 % interval, the
                              oracle as a dashed reference line. N is in each label (the 4-clip pools are /200 and reuse the same 50
                              layouts, so their intervals are too narrow). M4's turned-handle naive / b_level / b_tilt rows are not drawn.
  results/final/heading.png   M4b's heading sweep (results/m4b/summary.json "heading_sweep": the oracle with its grasp turned to phi,
                              20 layouts each) plus R2's choices on the test layouts (results/m5/choices.json + the eval rows): one marker
                              per (clip, chosen phi) at that choice's success rate, its size = how often R2 chose it.

Run: env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m retarget.final_figures
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retarget.m4_summary import AQUA, BLUE, ORANGE

OUT = Path("results/final")
M5 = Path("results/m5")
VIEW = {"26d4118b0c": "top-down", "4e103843db": "90°", "49aff04f7a": "45° (a)", "e8825dea53": "45° (b)"}
OWN_YAW = Path("results/final/own_yaw/summary.json")
ORDER = ("naive", "b_level", "b_tilt", "b_hybrid_original_yaw", "b_hybrid_h45_M4b", "F_clean", "R1_clean", "R2_clean", "F_noisy", "R2_noisy")
NAMES = dict(naive="naive: my palm path replayed", b_level="b_level: my full approach", b_tilt="b_tilt: b_level + my wrist tilt",
             b_hybrid_original_yaw="b_hybrid: my heading + robot final approach", b_hybrid_h45_M4b="b_hybrid_h45: 45° heading",
             F_clean="fixed-best candidate F", R1_clean="selector R1 (argmax)", R2_clean="selector R2 (human-first)",
             F_noisy="F, noisy layout", R2_noisy="R2, noisy layout")


def ablation(s: dict, own: dict, path: Path):
    """s = results/m5/summary.json, own = results/final/own_yaw/summary.json (its 4-clip pools)."""
    abl = dict(s["ablation"], **{v: own["own_yaw"][v]["pool"] for v in ("naive", "b_level", "b_tilt")})
    keys = list(ORDER)
    names = [NAMES[k] for k in keys]
    rate = np.array([abl[k]["successes"] / abl[k]["n"] for k in keys])
    lo = np.array([abl[k]["wilson95"][0] for k in keys])
    hi = np.array([abl[k]["wilson95"][1] for k in keys])
    labels = [f"{n}  ({abl[k]['successes']}/{abl[k]['n']})" for n, k in zip(names, keys)]
    colour = [AQUA if k.startswith(("R1", "R2")) else BLUE for k in keys]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(keys) + 1.4), dpi=150)
    y = np.arange(len(keys))[::-1]
    ax.barh(y, rate, color=colour, height=0.62)
    ax.errorbar(rate, y, xerr=[rate - lo, hi - rate], fmt="none", ecolor="#333333", capsize=3, lw=1)
    o = abl["oracle"]
    ax.axvline(o["successes"] / o["n"], color=ORANGE, ls="--", lw=1.5)
    ax.text(o["successes"] / o["n"] - 0.01, len(keys) - 0.45, f"scripted oracle {o['successes']}/{o['n']}", color=ORANGE, fontsize=8, va="bottom", ha="right")
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.0), xlabel="success rate on the 50 test layouts, each at its own handle yaw (Wilson 95 %)")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def heading(path: Path):
    sweep = json.loads(Path("results/m4b/summary.json").read_text())["heading_sweep"]["rows"]
    ch = json.loads((M5 / "choices.json").read_text())["choices"]
    rows = {json.loads(p.read_text())["run"]: json.loads(p.read_text()) for p in (M5 / "eval" / "runs").glob("*.json")}
    from sim.m4_execute import run_name
    picks = defaultdict(list)
    for t, c in enumerate(ch):
        r = rows[run_name(c["R2"]["clip"], "b_hybrid_phi", f"trial{t:02d}", c["R2"]["candidate"])]
        picks[(c["R2"]["clip"], round(c["R2"]["phi_deg"], 1))].append(bool(r["success"]))
    phis = [r["phi_deg"] for r in sweep]
    k = np.array([r["successes"] for r in sweep]); n = np.array([r["n"] for r in sweep])
    lo = np.array([r["wilson95"][0] for r in sweep]); hi = np.array([r["wilson95"][1] for r in sweep])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.errorbar(phis, k / n, yerr=[k / n - lo, hi - k / n], color=BLUE, marker="o", capsize=4, lw=2, label="oracle, grasp turned to phi (M4b, 20 layouts each)")
    markers = dict(zip(VIEW, ("s", "^", "D", "P")))
    for clip in VIEW:
        xs = [(phi, v) for (c, phi), v in picks.items() if c == clip]
        if xs:
            ax.scatter([phi for phi, _ in xs], [np.mean(v) for _, v in xs], s=[30 + 12 * len(v) for _, v in xs], marker=markers[clip],
                       color=AQUA, edgecolor="#1a1a1a", lw=0.6, zorder=3, label=f"R2 picks {clip[:5]} ({VIEW[clip]})")
    ax.set(xlabel="grasp heading phi [deg] (oracle 45)", ylabel="success rate", ylim=(0, 1.05))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Heading sweep and the selector's choices (marker size = how often R2 chose it; y = their success rate)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {f"{c} {phi}": dict(chosen=len(v), successes=sum(v)) for (c, phi), v in sorted(picks.items())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((M5 / "summary.json").read_text())
    ablation(s, json.loads(OWN_YAW.read_text()), OUT / "ablation.png")
    r2 = heading(OUT / "heading.png")
    (OUT / "figures.json").write_text(json.dumps(dict(what="retarget/final_figures.py", sources=[str(OWN_YAW), "results/m5/summary.json", "results/m4b/summary.json",
                                                                                             "results/m5/choices.json", "results/m5/eval/runs"],
                                                      figures=["ablation.png", "heading.png"], r2_choices_plotted=r2), indent=2))
    print(f"wrote {OUT / 'ablation.png'}, {OUT / 'heading.png'}")


if __name__ == "__main__":
    main()
