# Track B final - L1 + L3 with HERO's finger closing (profile mug_v1)

Why: the v3 ladder was cumulative, so L3 was tested on top of L2, which had made the grip weaker (lift slip 47 mm vs 22 mm); this run tests L1 + L3 with HERO's closing targets. One pre-registered test, not a search.

Profile `mug_v1` (sim/pick_place.py GRASP_PROFILES): {'grasp_offset': 'radius', 'closing_targets': 'hero', 'contact_model': 'grasp'}. Everything else is the M1 oracle; nothing was tuned.

## Run 1: B3's R3 (full mug from config, 0.40 kg, B3's 20 trials, R3's handle yaws)

| run | success | Wilson 95 % | failures (M1 categories) | lift slip mean / median (mm); > 20 mm | rise mean / median (cm) | contact at lift |
|---|---|---|---|---|---|---|
| **Run 1 (mug_v1)** | 20/20 | 0.839-1.000 | - | 5.3 / 5.25 (n 20); 0 | 6.81 / 6.83 | 12/20 |
| v3 B3 R3, no change (reference) | 0/20 | 0.000-0.161 | dropped while moving 8, slip during lift 12 | 20.75 / 20.1 (n 20); 10 | 5.11 / 4.98 | 13/20 |
| v3 L1 (reference) | 0/20 | 0.000-0.161 | dropped while moving 5, slip during lift 15 | 21.96 / 22.7 (n 20); 12 | 4.99 / 4.96 | 14/20 |
| v3 L2 (L1 + L2) (reference) | 0/20 | 0.000-0.161 | dropped while moving 1, slip during lift 19 | 47.06 / 30.3 (n 20); 19 | 3.21 / 4.34 | 14/20 |
| v3 L3 (L1 + L2 + L3) (reference) | 11/20 | 0.342-0.742 | missed/knocked cup 9 | 6.8 / 6.6 (n 11); 0 | 6.87 / 6.86 | 5/20 |

Reference rows: results/m1_grasp_fix/v3/ladder.json (read only).

## Forces, finger torques and touch in Run 1 (over the trials that reached each moment)

- closed: the control step in which the right hand's closure first reaches 1.0 (close phase: set_hand_closure(1.0), then that 20 ms step, the first one with the squeeze rule on); state after the step = its last 1 ms substep
- lift: B3's lift frame: first control step of the oracle's 'lift' phase (hand fully closed after the 0.5 s close hold, lift reference just started, palm not yet risen); contacts = data.contact after that step (its last 1 ms substep)
- normal_force_N: per hand part (thumb = thumb_0/1/2, index = index_0/1, middle = middle_0/1, palm = palm_link; B3's hand_parts), sum of mujoco.mj_contactForce component 0 (normal force in the contact frame) over the contacts in data.contact between that part's collision geoms and any geom of the cup body
- finger_tau_Nm: the 7 right finger-joint torques applied in the step's last substep (TableScene.last_finger_tau, after the squeeze rule = what was written to qfrc_applied), FINGER_SUFFIXES order
- saturated: |tau| >= 0.99 x the joint's effort limit {'thumb_0': 2.45, 'thumb_1': 1.4, 'thumb_2': 1.4, 'index_0': 1.4, 'index_1': 1.4, 'middle_0': 1.4, 'middle_1': 1.4} N m
- touch_N: the right fingertip touch sensors (thumb_2, index_1, middle_1; sim/scene.py: normal force inside a box site around the distal link's collision geometry)
- timing: data.contact, the contact forces and the touch sensors all come from the last substep's mj_forward (mj_step computes them before it integrates); the torque is the one applied in that substep

| moment | trials reached | thumb N (mean / median) | index N (mean / median) | middle N (mean / median) | palm N (mean / median) |
|---|---|---|---|---|---|
| closed | 20 | 68.61 / 74.81 | 19.29 / 22.90 | 24.98 / 24.87 | 18.22 / 6.49 |
| lift | 20 | 60.28 / 64.28 | 20.44 / 26.10 | 29.15 / 28.40 | 22.67 / 17.02 |

| moment | thumb_0 (limit 2.45) | thumb_1 (limit 1.4) | thumb_2 (limit 1.4) | index_0 (limit 1.4) | index_1 (limit 1.4) | middle_0 (limit 1.4) | middle_1 (limit 1.4) |
|---|---|---|---|---|---|---|---|
| closed: tau N m (mean / median) | -0.11 / -0.10 | -1.40 / -1.40 | -1.14 / -1.19 | 1.00 / 1.40 | 0.64 / 0.84 | 1.40 / 1.40 | 0.97 / 0.94 |
| lift: tau N m (mean / median) | -0.09 / -0.06 | -1.40 / -1.40 | -1.04 / -1.05 | 0.97 / 1.40 | 0.52 / 0.51 | 1.40 / 1.40 | 1.02 / 1.02 |
| closed: saturated (trials of 20) | 0 | 20 | 0 | 12 | 0 | 20 | 0 |
| lift: saturated (trials of 20) | 0 | 20 | 0 | 12 | 0 | 20 | 0 |

| moment | touch thumb_2 N (mean / median) | touch index_1 N (mean / median) | touch middle_1 N (mean / median) |
|---|---|---|---|
| closed | 34.11 / 35.44 | 17.16 / 21.53 | 23.80 / 24.87 |
| lift | 31.60 / 32.86 | 18.45 / 24.93 | 29.10 / 28.40 |

## Checks

- R0 with mug_v1 (M1 cylinder, its 20 trials): 20/20 (required 20/20) -> PASS. Rows identical to v3's L3_R0 (same physics on the cylinder; every field except wall time): 20/20.
- Legacy R0 after the final edits (`python -m sim.m1_grasp_ladder run --rung R0 --tag after_final --out results/m1_grasp_fix/v3_final`): 20/20, trials identical to M1 20/20, to B3 R0 20/20 -> PASS.
- Reach test not rerun: mug_v1's global contact options are exactly L3's, which passed the reach test (results/m1_grasp_fix/v3/reach_l3/); the reach test uses the cylinder and only the contact model, so the grasp offset and the closing targets play no role there. Compiled contact settings of these runs equal L3's: True (cone mjCONE_ELLIPTIC, impratio 10.0, noslip_iterations 3, condim 4 on right palm / thumb_2 / index_1 / middle_1). L3's reach guard: passed True (fell False, pelvis z min 0.686 m, tilt max 12.8 deg).

## Decision

Run 1 = 20/20; threshold 18/20: met -> the 50-trial run + 2 videos.

## 50 trials (trials 0-49 of trials(), full mug 0.40 kg, mug_v1)

| run | success | Wilson 95 % | failures (M1 categories) | lift slip mean / median (mm); > 20 mm | rise mean / median (cm) | contact at lift |
|---|---|---|---|---|---|---|
| **50 trials** | 44/50 | 0.762-0.944 | missed/knocked cup 5, slip during lift 1 | 4.97 / 4.9 (n 45); 0 | 6.81 / 6.85 | 29/50 |

- Trials 0-19 of this run are Run 1's trials in other worker chunks: identical rows (every field except wall time) 20/20.
- video: `results/m1_grasp_fix/v3_final/videos/success_trial00.mp4` (trial 0, batch success; re-run same outcome True, identical numbers True)
- video: `results/m1_grasp_fix/v3_final/videos/failure_trial22.mp4` (trial 22, batch failure: slip during lift; re-run same outcome True, identical numbers True)

Failed trials of the 50 (forces / torques / touch at both moments):

| trial (cup xy, m) | yaw (deg) | failure (M1 category: reason) | moment | normal force thumb / index / middle / palm (N) | tau thumb_0 / thumb_1 / thumb_2 / index_0 / index_1 / middle_0 / middle_1 (N m; * = saturated) | touch thumb_2 / index_1 / middle_1 (N) |
|---|---|---|---|---|---|---|
| 22 (0.5104, -0.197) | 5.22 | slip during lift: cup slipped 0.7 cm in the hand, rose 4.3 of 10 cm | closed | 81.84 / 18.70 / 20.68 / 0.00 | -0.14 / -1.40* / -1.21 / 1.40* / 0.84 / 1.40* / 0.86 | 35.87 / 18.70 / 20.68 |
| 22 (0.5104, -0.197) | 5.22 | slip during lift: cup slipped 0.7 cm in the hand, rose 4.3 of 10 cm | lift | 68.31 / 19.17 / 0.00 / 0.00 | -0.13 / -1.40* / -0.98 / 1.40* / 0.84 / 1.40* / 0.95 | 29.32 / 19.17 / 0.00 |
| 34 (0.4508, -0.3217) | -23.4 | missed/knocked cup: knocked the cup before grasping (moved 4.2 cm, tilt 0 deg) | closed | not reached | - | - |
| 34 (0.4508, -0.3217) | -23.4 | missed/knocked cup: knocked the cup before grasping (moved 4.2 cm, tilt 0 deg) | lift | not reached | - | - |
| 38 (0.4685, -0.3134) | 26.58 | missed/knocked cup: knocked the cup before grasping (moved 2.4 cm, tilt 0 deg) | closed | not reached | - | - |
| 38 (0.4685, -0.3134) | 26.58 | missed/knocked cup: knocked the cup before grasping (moved 2.4 cm, tilt 0 deg) | lift | not reached | - | - |
| 44 (0.4469, -0.3284) | -33.95 | missed/knocked cup: knocked the cup before grasping (moved 2.7 cm, tilt 0 deg) | closed | not reached | - | - |
| 44 (0.4469, -0.3284) | -33.95 | missed/knocked cup: knocked the cup before grasping (moved 2.7 cm, tilt 0 deg) | lift | not reached | - | - |
| 45 (0.4601, -0.3206) | -10.08 | missed/knocked cup: knocked the cup before grasping (moved 4.9 cm, tilt 0 deg) | closed | not reached | - | - |
| 45 (0.4601, -0.3206) | -10.08 | missed/knocked cup: knocked the cup before grasping (moved 4.9 cm, tilt 0 deg) | lift | not reached | - | - |
| 49 (0.4505, -0.3043) | -5.1 | missed/knocked cup: knocked the cup before grasping (moved 2.6 cm, tilt 0 deg) | closed | not reached | - | - |
| 49 (0.4505, -0.3043) | -5.1 | missed/knocked cup: knocked the cup before grasping (moved 2.6 cm, tilt 0 deg) | lift | not reached | - | - |

## Runtime and peak RSS (CPU only)

| step | wall (s) | peak RSS (MB) |
|---|---|---|
| run1 | 37.0 | max process 1679, sum 6636, parent 147 |
| r0_mug_v1 | 35.9 | max process 1733, sum 6847, parent 147 |
| legacy_r0_after_final | 29.3 | max process 1702, sum 6641, parent 63 |
| fifty | 89.1 | max process 1805, sum 7069, parent 149 |
| videos | 24.8 | 1718 |
| summarize | 0.01 | 149 |

- Open question: The 'closed' moment is the first step at full closure; the finger targets slew at <= 2 rad/s, so at that step the joint targets may still lag the closed pose. The 0.5 s hold after it (the close gate) is not recorded.
- Open question: Normal force per part sums every contact in data.contact; a contact inside the margin with no active constraint has zero force, so B3's 'touching' (a contact exists) and a force > 0 can differ.
- Open question: Not measured: why 5 of the 50 trials (trials [34, 38, 44, 45, 49]) knocked the cup during the approach, before any finger closed (Run 1: 0). Their cup positions and reasons are in fifty.failed_trials. On the same layouts M1's cylinder (legacy offset) succeeded in 5/5. mug_v1 moves the palm target (L1: approach direction 39.95 instead of 35.54 deg in the palm frame) and changes the contact model; which of these matters was not tested (no more fixes in this track).

All numbers: final.json (raw rows: runs/).
