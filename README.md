# teach-g1

From 4 iPhone LiDAR clips of my own hand, a standing Unitree G1 (MuJoCo sim, HERO whole-body controller) learns to pick up my
mug and place it on a coaster. Simulation only, no real robot. Work in progress; `docs/log.md` has every step and number.

## Setup
- Two conda envs, `hoi-perception` (WiLoR, SAM 2) and `hoi-sim` (HERO, MuJoCo); they only exchange `data/demos/*.npz`.
  Pins in `envs/`, install steps in `docs/log.md` (2026-10-29 hoi-sim, 2026-10-30 hoi-perception).
- `third_party/` (gitignored): HERO `416c4e0`, WiLoR `fcb9113`, SAM 2 `2b90b9f`, plus their weights and `MANO_RIGHT.pkl`
  (licensed, register at mano.is.tue.mpg.de).
- Always run with `env -u PYTHONPATH` (a ROS install on this machine leaks into every env).

```bash
P="env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python"
S="env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python"
```

## Perception: my clips → demo files
```bash
$P -m perception.run_clip        # all stages, all 4 clips: hand (WiLoR + LiDAR), mug/coaster masks (SAM 2), table frame, events
                                 # -> data/demos/<clip>.npz and results/m2_perception_v3/ (stages cached in data/work/, --force reruns)
$P tests/test_geometry.py        # unit test of the upright <-> original image maths
```
Raw Stray Scanner recordings go in `data/raw/<clip>/`.

## Simulation: G1 + HERO
```bash
$S -m sim.m1_reach_test --heights 0.74 --out /tmp/teach_g1/reach          # G1 stands and reaches 6 palm goals over a table
$S -m sim.pick_place --cup-model cylinder --trials 10 --out /tmp/teach_g1/pick   # scripted oracle pick-and-place (M1 cylinder)
$S -m sim.b2_capacity --version v3                                        # can the Dex3 hand wrap cylinders of each diameter?
                                                                          # (no --out: rewrites results/m1_grasp_fix/v3/b2v3_capacity.*)
$S -m sim.m1_grasp_ladder run --rung L3 --out /tmp/teach_g1/ladder        # grasp-fix rung on my mug (20 layouts)
```
Videos render offscreen with `MUJOCO_GL=egl` (here also `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`
to keep rendering off the NVIDIA GPU).

## Results so far
`results/m2_perception_v3/summary.md` (perception gates per clip) and `results/m1_grasp_fix/` (grasp feasibility and fix ladder).
