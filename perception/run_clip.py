"""Run the M2 Track A pipeline (S1-S11) for one clip, one stage per process.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.run_clip --clip <id> [--force]

Order: frames, hand_wilor, masks_sam2, table, hand_metric, objects, table_frame, events, palm, build_demo, report.
Each stage runs as its own `python -m perception.<module> --clip <id>` process, so a GPU stage frees its memory when it
exits and every done.json records that stage's own peaks. Caching is done by the stages themselves (a stage with a
done.json is skipped); --force is passed to every stage. The run stops at the first stage that exits non-zero.
So "S10 always writes the demo file, even for failing clips" means clips that fail their GATES: if a stage crashes, the
later stages (build_demo, report included) do not run. build_demo can then be run by hand; it writes the file with the
crashed stage's fields NaN and lists that stage in its done.json ("missing").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from perception.common import CLIPS

MODULES = ["frames", "hand_wilor", "masks_sam2", "table", "hand_metric", "objects", "table_frame", "events", "palm",
           "build_demo", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, choices=sorted(CLIPS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    for module in MODULES:
        cmd = [sys.executable, "-m", f"perception.{module}", "--clip", args.clip] + (["--force"] if args.force else [])
        print(f"== {module}", flush=True)
        t0 = time.perf_counter()
        code = subprocess.run(cmd).returncode
        print(f"== {module}: exit {code}, {time.perf_counter() - t0:.1f} s", flush=True)
        if code != 0:
            print(f"stopped: {module} failed on {args.clip}; later stages not run", file=sys.stderr)
            sys.exit(code)
    print(f"== {args.clip}: all {len(MODULES)} stages done in {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
