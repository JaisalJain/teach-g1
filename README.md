# teach-g1: a standing Unitree G1 picks and places my mug, using hand motion from four iPhone clips of me

![My hand (left) and the G1 (right)](results/extra/hand_to_robot.gif)

*Left: my clip `e8825dea53` with the perception overlay. Right: the G1 runs the path built from it on a new layout (my clip is time-warped to match).*

I recorded four iPhone LiDAR clips of my right hand moving my mug onto a coaster. The pipeline turns them into a palm path
relative to the mug and the coaster, and a Unitree G1 that balances on its own legs (HERO whole-body controller, MuJoCo) runs it on
50 random layouts it never saw. Simulation only.

Copying my hand exactly fails: replaying my palm path succeeds on 0/200 runs and following my whole approach on 25/200. The
Dex3's open fingers reach about 8 cm past the palm and sweep through the mug. Replacing only my last 8 cm with a straight robot move
fixes that (`b_hybrid`, 98/200). My palm heading only transfers below about 65 deg (the scripted grasp turned to 65 deg succeeds on
20/20 layouts, at 75 deg on 13/20), and three of my clips grasp at 82-86 deg, so turning the grasp to 45 deg gets 179/200.

The best rule (R2, a human-first selector) succeeds on 49/50 test layouts and keeps my own heading on 35/50. A single fixed choice
with no learning gets 46/50, and the scripted grasp with no human data 44/50. So my data decides how the robot moves (approach
shape, timing, grasp height, transport, and heading where the arm can reach it), not whether it succeeds; the robot-specific parts
do that.

## Results

| variant | what comes from my clips | successes (Wilson 95 %) |
|---|---|---|
| `naive` | my whole palm path and wrist, replayed where I did it | 0/200 (0.00-0.02) |
| `b_level` | approach and transport shape bent to the new mug and coaster; palm heading, grasp height, timing | 25/200 (0.09-0.18) |
| `b_tilt` | as `b_level`, plus my wrist tilt | 39/200 (0.15-0.26) |
| `b_hybrid` | as `b_level`, but the robot does the last 8 cm of the approach | 98/200 (0.42-0.56) |
| `b_hybrid_h45` | as `b_hybrid`, but the grasp heading is 45 deg, not mine | 179/200 (0.84-0.93) |
| scripted oracle | nothing | 44/50 (0.76-0.94) |
| F: one fixed choice | `b_hybrid_h45` of clip `4e103843db` on every layout | 46/50 (0.81-0.97) |
| R1: selector, highest predicted success | per layout, a clip and a heading (never my own) | 47/50 (0.84-0.98) |
| R2: selector, human-first | per layout, the clip and heading closest to mine among those predicted to succeed | 49/50 (0.90-1.00) |
| R2, noisy layout | as R2, with the path built from a layout perturbed by 1 cm / 10 deg | 44/50 (0.76-0.94) |
| F, noisy layout | as F, with the same noise | 38/50 (0.63-0.86) |

*Every row runs the same 50 test layouts, each with its own handle direction. The /200 rows pool my four clips, which reuse those
50 layouts, so their intervals are too narrow. Sources: [own-yaw summary](results/final/own_yaw/summary.md) (first three rows) and
[M5 summary](results/m5/summary.md) (the rest); per-clip numbers in [docs/details.md](docs/details.md).*

![Success rate per variant](results/final/ablation.png)

## What I recorded

- The same task four times, each a separate execution filmed from a different viewpoint: I wrap my right hand round the side of
  my mug, place it on the coaster, release it and take the hand away. I sat at a lap desk on a bed; the phone was propped by hand.
- iPhone with LiDAR and the [Stray Scanner](https://github.com/StrayRobots/scanner) app: RGB video, LiDAR depth, depth confidence,
  camera intrinsics and poses, IMU. Mug: glossy dark-green ceramic, 9.9 cm wide (label), 0.40 kg (estimated); coaster about
  10.4 cm across (estimated); all sizes in [config/objects.yaml](config/objects.yaml).
- All four clips are committed unchanged in `data/raw/<id>/`; [data/raw/README.md](data/raw/README.md) explains the format and the
  traps I found. The extracted demo files `data/demos/*.npz` are committed too.

| clip | view | capture rate | length |
|---|---|---|---|
| `26d4118b0c` | top-down | 30 Hz | 8.30 s |
| `4e103843db` | 90 deg (at my side) | 60 Hz | 8.87 s |
| `49aff04f7a` | 45 deg | 60 Hz | 7.97 s |
| `e8825dea53` | 45 deg | 60 Hz | 7.89 s |

## How it works

- **Perception** (`perception/`, GPU): WiLoR fits a MANO hand to my right hand in every frame, and the LiDAR depth moves it to
  its metric position (median residual 0.47-0.76 cm). SAM 2 masks the mug, its handle and the coaster; a RANSAC plane gives the table.
  Output: one `data/demos/<clip>.npz` with the palm path, the mug and coaster positions and the events (contact, lift, place,
  release).
- **Retargeting** (`retarget/`): the palm path is expressed relative to the mug (approach, grasp) and the coaster (transport,
  place), an idea from OKAMI (CoRL 2024). For each new layout it is bent smoothly so it starts at the robot's standing palm and ends
  at the grasp and place poses. The grasp keeps my palm heading (61.7-85.8 deg) and grasp height, and time is stretched by
  1.09-1.74 so that my palm speed (95th percentile) is no faster than the scripted grasp's peak. No finger angles are used.
- **Execution** (`sim/`, CPU): HERO's example policy drives the G1's 29 body joints at 50 Hz and keeps it balanced. It tracks a
  reference clip that HERO's own IK builds from my palm path. Our PD loop closes the Dex3 with HERO's closing targets until the
  touch sensors we added read more than 0.3 N on the thumb and one finger. Success = mug upright on the coaster within 5 cm, at
  rest, robot still standing.
- **Final-approach fix** (`b_hybrid`): my approach ends 8 cm before the grasp; the palm settles there and the robot moves
  straight in for 2 s, as the scripted grasp does.
- **Selector** (`policy/`): 18 candidates (clip x heading: mine, 75, 65, 55 or 45 deg) were run on 120 training layouts (2160
  runs). A small MLP predicts success from the noisy layout and the candidate; it trains in PyTorch and runs as ONNX. R1 takes
  the highest predicted success, R2 the candidate closest to my heading among those with p >= 0.8, F one fixed candidate.

![Heading sweep and the selector's choices](results/final/heading.png)

*Blue: the scripted grasp turned to each heading, 20 layouts each ([M4b summary](results/m4b/summary.md)). Markers: R2's choices
on the test layouts, sized by how often it chose them. My four headings are 61.7, 81.7, 81.9 and 85.8 deg.*

## What worked and what didn't

- **Copying my last centimetres knocks the mug.** The Dex3 fingertips sit about 8 cm ahead of the palm point HERO tracks, so my
  final approach sweeps them through the mug: 131 of the 175 `b_level` failures pushed the mug more than 2 cm before the hand
  closed. The robot's straight last 8 cm fixed it (25/200 -> 98/200).
- **My heading transfers only up to about 65 deg.** The scripted grasp holds 20/20 up to 65 deg and drops to 13/20 at 75 deg. On
  the 120 training layouts my three high-heading clips succeed on 31-41/120 at their own heading and on 111-115/120 at 45 deg.
- **Learning ties the fixed choice.** In cross-validation the selector's best pick (R1) and the fixed candidate F both succeed on
  115 of 120 held-out layouts. On the test layouts R2 and F differ on only 5 layouts (4 vs 1).
- **R2's heading comes mostly from one clip.** 33 of its 35 own-heading picks are `e8825dea53`, the clip whose heading (61.7 deg)
  is already inside the arm's range.
- **The mug fit reads 6-7 mm wide.** The fitted diameter is 105.1-106.1 mm against the 99 mm label, in every view; full-resolution
  masks did not fix it. The simulator uses the label size.
- **Per-digit finger IK made the grip worse.** Mean lift slip 47 mm against 22 mm without it (0/20 grasps either way), so the
  Dex3 uses HERO's own closing targets.
- **No grip margin.** In the final grasp three finger joints sit at their torque limit (thumb_1 and middle_0 in 20 of 20 grasps,
  index_0 in 12 of 20).

Sources and longer notes: [docs/details.md](docs/details.md); [docs/log.md](docs/log.md) has every step.

## How to run

Install: [docs/INSTALL.md](docs/INSTALL.md) (two conda envs: `hoi-sim` for simulation, CPU only; `hoi-perception` for WiLoR and
SAM 2, GPU; `env -u PYTHONPATH` keeps a ROS install on my machine out of them). Start with the check:

```bash
bash scripts/reproduce.sh
```

It uses committed files only (no GPU, no weights). It re-runs one scripted-oracle trial, one `b_hybrid_h45` trial, one own-yaw
`b_level` trial and one R2 trial (its choice recomputed from the ONNX selector) and compares each with its saved row, field by
field. Then it regenerates the M4, M4b, M5 and own-yaw summaries and compares them with the committed files. It prints ALL MATCH
when everything matches. On my laptop it took 47 s ([output](results/final/reproduce_check.txt)).

```bash
P="env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python"
S="env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python"
$P -m perception.run_clip                                          # my 4 clips -> data/demos/*.npz
$S -m sim.m4_execute run --own-yaw --workers 8 --out /tmp/own_yaw  # naive, b_level, b_tilt x 4 clips x 50 layouts
$S -m sim.m4_execute run --variants b_hybrid_h45 --workers 8 --out /tmp/m4b
$S -m policy.m5 summarize                                          # selector results from the saved rows
$S -m retarget.own_yaw_summary                                     # own-yaw results from the saved rows
```

The full command list is in [docs/details.md](docs/details.md#how-to-run).

## Limitations

- **Simulation only.** There is no real robot.
- **The mug's mass and the coaster's size are estimates**, and the contact settings come from MuJoCo Menagerie's hand models, not
  from a measurement of my mug.
- **Privileged place correction.** For the oracle and every `b` variant, the place pose uses the mug's true pose in the hand, read
  from the simulator; perception does not provide it. Apart from the noisy runs, the robot also gets exact mug and coaster positions.
- **Small workspace.** Layouts keep the palm inside x 0.35-0.45 m and y -0.32 to -0.12 m in front of the robot.
- **The test layouts shaped the design.** The same 50 layouts were used in M4, M4b and M5; the final-approach fix came from M4's
  failures on them and the heading grid from a sweep on 20 of them. The selector, F and the 0.8 threshold never saw them.
- **n = 4 clips, with viewpoint and execution changing together**, so the effect of the viewpoint cannot be separated.

## How I built this

I recorded the four clips, measured the mug and desk, and chose between options at every step. I planned the work and wrote the
step prompts in conversation with Claude (Anthropic), and Claude Code wrote and ran the code on my laptop from those prompts and
CLAUDE.md. I reviewed every result before the next step; docs/log.md is the running record of what was tried, what broke and
every number.

## Licence and third-party code

The code is MIT ([LICENSE](LICENSE)). My recordings in `data/raw/` are CC BY 4.0 ([data/raw/README.md](data/raw/README.md)).
Third-party code and weights keep their own licences; none of them is redistributed here.

| repo | commit | licence | used for |
|---|---|---|---|
| [RunpeiDong/HERO](https://github.com/RunpeiDong/HERO) | `416c4e0` | MIT | whole-body policy, G1 + Dex3 model, IK / FK |
| [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) | `fcb9113` | CC BY-NC-ND 4.0 (code and models) | 3D hand per frame |
| [MANO](https://mano.is.tue.mpg.de) | - | non-commercial | hand model inside WiLoR |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | `2b90b9f` | Apache-2.0 | mug, handle and coaster masks |
| [StrayRobots/scanner](https://github.com/StrayRobots/scanner) | `ec3e1dc` | MIT | the recording app |

WiLoR's models and MANO are for non-commercial use only, so the perception half cannot be used commercially. Python packages,
licence files and everything else: [docs/details.md](docs/details.md#third-party-code).

## Gallery

| | |
|---|---|
| ![R2 success](results/m5/r2_success.gif) | ![b_hybrid success](results/m4b/b_hybrid_success.gif) |
| **The selector (R2)** on test layout trial00: it picked `e8825dea53` at 55 deg | **`b_hybrid`**: my heading and approach, the robot's final 8 cm, my transport (`e8825dea53`, trial00, an M4b run) |
| ![oracle success](results/final/oracle_success.gif) | ![b_level failure](results/final/b_level_failure.gif) |
| **Scripted oracle** (no human data), trial 0 | **`b_level` failure**: following my whole approach, the open fingers push the mug away before the grasp (an M4 run) |
