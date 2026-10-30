"""Shared bits of the M2 perception pipeline: paths, clip list, object config, and the per-stage runner.

Every stage module (frames, hand_wilor, masks_sam2, ...) exposes ``run(clip) -> stats`` and calls ``stage_cli`` from its
``__main__``. ``stage_cli`` caches: a stage whose ``data/work/<clip>/<stage>/done.json`` exists is skipped unless --force.
done.json also records the stage's runtime, peak RSS and (for GPU stages) peak VRAM. Each stage runs in its own process
(``run_clip`` starts them one by one), so these peaks are per stage and GPU memory is freed when the stage exits.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import threading
import time
from pathlib import Path

import yaml

RAW = Path("data/raw")
WORK = Path("data/work")
RESULTS = Path("results/m2_perception_v2")        # M2 v2 outputs; v1 stays untouched in RESULTS_V1
RESULTS_V1 = Path("results/m2_perception")
DEMOS = Path("data/demos")
CONFIG = Path("config/objects.yaml")
CLIPS = {"26d4118b0c": "top-down", "4e103843db": "90deg", "49aff04f7a": "45deg", "e8825dea53": "45deg"}

RSS_LIMIT_GB = 16.0


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def cfg(c: dict, *keys) -> float:
    """cfg(c, "mug", "height_m") -> the value field of that entry."""
    for k in keys:
        c = c[k]
    return float(c["value"])


def stage_dir(clip: str, stage: str) -> Path:
    d = WORK / clip / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_done(clip: str, stage: str) -> bool:
    return (WORK / clip / stage / "done.json").exists()


def _rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def _watchdog():
    """Stop the whole process if its resident memory exceeds RSS_LIMIT_GB (the run is reported as failed by run_clip)."""
    while True:
        if _rss_gb() > RSS_LIMIT_GB:
            print(f"RSS above {RSS_LIMIT_GB} GB -- stopping this stage", flush=True)
            os._exit(3)
        time.sleep(0.5)


def stage_cli(stage, run=None, gpu: bool = False) -> None:
    """Standard main() of a stage module. A module with several stages passes a dict {--stage value: (stage name, run, gpu)};
    the first entry is the default."""
    stages = stage if isinstance(stage, dict) else {stage: (stage, run, gpu)}
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, choices=sorted(CLIPS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", choices=list(stages), default=next(iter(stages)))
    args = ap.parse_args()
    stage, run, gpu = stages[args.stage]
    if is_done(args.clip, stage) and not args.force:
        print(f"{stage} {args.clip}: cached, skipping (--force to redo)")
        return
    (WORK / args.clip / stage / "done.json").unlink(missing_ok=True)
    threading.Thread(target=_watchdog, daemon=True).start()
    t0 = time.perf_counter()
    stats = run(args.clip) or {}
    record = {"stage": stage, "clip": args.clip, "seconds": round(time.perf_counter() - t0, 2),
              "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)}
    if gpu:
        import torch
        record["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    record.update(stats)
    (stage_dir(args.clip, stage) / "done.json").write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps(record, default=float))
