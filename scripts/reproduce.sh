#!/usr/bin/env bash
# Re-runs a few results from committed files only (the demo .npz files, the saved rows, the ONNX selector); no GPU, no perception.
# Each re-run prints whether it matches its saved row; the script exits non-zero if any does not.
#   1. one oracle trial (Track B final mug_v1 trial 0) vs results/m1_grasp_fix/v3_final/runs/fifty.json
#   2. one b_hybrid_h45 trial (M4b, e8825dea53 trial00) vs its row in results/m4b/runs/
#   3. one R2 trial: the selector's choice on test trial 0 recomputed from results/m5/selector.onnx, then that run vs its row
#   4. the M4, M4b and M5 summaries regenerated from the saved rows; their .md / .json must equal the committed files
# Needs the hoi-sim conda env and third_party/HERO (docs/INSTALL.md). Run from anywhere: bash scripts/reproduce.sh
set -uo pipefail
cd "$(dirname "$0")/.."
S="env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python"
TMP="$(mktemp -d)"
fail=0
step() { echo; echo "== $1"; }

step "1. oracle trial 0"
$S -m sim.m4_execute check-oracle --trials 1 --out "$TMP" || fail=1

step "2. b_hybrid_h45 e8825dea53 trial00"
$S -m sim.m4_execute rerun --row results/m4b/runs/e8825dea53__b_hybrid_h45__trial00.json --report "$TMP/h45.json" || fail=1

step "3. R2 (ONNX selector) on test trial 0"
$S -m policy.m5 reproduce-r2 --trial 0 || fail=1

step "4. summaries from the saved rows"
mkdir -p "$TMP/m4" && ln -s "$PWD/results/m4_execute/runs" "$TMP/m4/runs"
$S -m retarget.m4_summary --out "$TMP/m4" > /dev/null || fail=1
mkdir -p "$TMP/m4b"
$S -m retarget.m4b_summary summarize --out "$TMP/m4b" > /dev/null || fail=1       # summary.md only: summary.json also lists run_resources files
for f in summary.md; do
  if cmp -s "results/m4_execute/$f" "$TMP/m4/$f"; then echo "M4 $f matches"; else echo "M4 $f DIFFERS"; fail=1; fi
done
if cmp -s results/m4b/summary.md "$TMP/m4b/summary.md"; then echo "M4b summary.md matches"; else echo "M4b summary.md DIFFERS"; fail=1; fi
$S -m policy.m5 summarize > /dev/null || fail=1
if git diff --quiet -- results/m5/summary.md results/m5/summary.json; then echo "M5 summary matches"; else echo "M5 summary DIFFERS (see git diff)"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "ALL MATCH"; else echo "SOMETHING DIFFERS (see above)"; fi
rm -rf "$TMP"
exit "$fail"
