"""Run the M2 Track A pipeline (S1-S11 with the v2 stages) on the clips, one stage per process, in two passes.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.run_clip \
        [--clip <id> ...] [--force]          (default: all 4 clips)

Two numbers of v2 need ALL 4 clips before any clip can go on: s_hand (S5b pass 2 pools the pass-1 hand scale of every
clip) and the A4 mask-source decision (S6 applies one source to all clips, chosen from the test on every clip). So:
  pass A, clip by clip: frames, hand_wilor, masks_sam2, table, hand_metric (S5), hand_metric --stage scale (S5b pass 1),
                        masks_sam2 --stage full (A4 masks), objects --stage test (A4 fits)
  pass B, clip by clip: hand_metric --stage fit (S5b pass 2), objects (S6), table_frame, events, palm, build_demo, report
  then: report --summary
Each step runs as its own `python -m perception.<module> --clip <id> [--stage ...]` process, so a GPU stage frees its
memory when it exits and every done.json records that stage's own peaks. Caching is done by the stages themselves (a
stage with a done.json is skipped). --force is passed to EVERY stage, so `run_clip --force` also reruns S1-S5,
including both GPU models (WiLoR, SAM 2). To rerun only from S5b on, either call those modules by hand with --force
(hand_metric --stage scale / fit, masks_sam2 --stage full, objects --stage test / objects, table_frame, events, palm,
build_demo, report), or delete their done.json files and run run_clip without --force.
The run stops at the first step that exits non-zero and exits with that step's code (128 + N for a step killed by
signal N, the shell convention, e.g. 137 for the OOM killer's SIGKILL). A crashed stage therefore stops every later
step of every clip (build_demo and report included); build_demo can then be run by hand: it writes the file with the
crashed stage's fields NaN and lists that stage in its done.json ("missing").
With --clip on fewer than 4 clips, the barrier stages (hand_metric --stage fit, objects) use the other clips' cached
pass-A outputs and stop with a message if those do not exist. The other clips' pass-B outputs are NOT rerun: if a pass-A
change (a subset run, or one rerun of hand_scale or mask_test) moves s_hand or flips the mask decision, rerun pass B on
all 4 clips. report --summary prints an ERROR line when the demo files disagree on s_hand or on the mug mask source; it
cannot see a stale mask decision that did not flip.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from perception.common import CLIPS

# (module, extra arguments); a module without --stage runs its default stage
PASS_A = [("frames", []), ("hand_wilor", []), ("masks_sam2", []), ("table", []), ("hand_metric", []),
          ("hand_metric", ["--stage", "scale"]), ("masks_sam2", ["--stage", "full"]), ("objects", ["--stage", "test"])]
PASS_B = [("hand_metric", ["--stage", "fit"]), ("objects", []), ("table_frame", []), ("events", []), ("palm", []),
          ("build_demo", []), ("report", [])]


def step(name: str, cmd: list[str]) -> None:
    """Run one stage process; exit with its code if it fails."""
    print(f"== {name}", flush=True)
    t0 = time.perf_counter()
    code = subprocess.run(cmd).returncode
    print(f"== {name}: exit {code}, {time.perf_counter() - t0:.1f} s", flush=True)
    if code != 0:
        print(f"stopped: {name} failed; later steps not run", file=sys.stderr)
        # subprocess reports a signal death as -N; sys.exit(-9) would exit with 247, so use the shell's 128 + N
        sys.exit(128 - code if code < 0 else code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="+", choices=sorted(CLIPS), default=list(CLIPS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    for steps in (PASS_A, PASS_B):
        for clip in args.clip:
            for module, extra in steps:
                cmd = [sys.executable, "-m", f"perception.{module}", "--clip", clip, *extra]
                step(" ".join([module, *extra, clip]), cmd + (["--force"] if args.force else []))
    step("report --summary", [sys.executable, "-m", "perception.report", "--summary"])
    print(f"== {', '.join(args.clip)}: both passes and the summary done in {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
