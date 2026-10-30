# CLAUDE.md — teach-g1

## What this project is

Entry for the **"Humanoid Challenge: Teach a Humanoid With Your Own Hands"** (Awign × Armature AI Labs × RIC × Tanna TechBiz).

**Deadline: Wed 4 Nov 2026, 23:59 IST.** Submission = a public GitHub repo. Code freeze target: Nov 4, 12:00 IST.

**One-line pitch:** From 4 iPhone LiDAR clips of my own hand, a standing Unitree G1 picks up a mug and places it on a coaster anywhere in its reachable workspace, with a measured success rate.

### Challenge rules that constrain design

- Hard rule: **data I personally recorded must play a real role** in the approach.
- G1 + MuJoCo preferred. **No fixed-base robot** — organizers asked for a whole-body controller (the G1 must balance on its legs).
- Finger dexterity is "good to have"; fewer than 5 fingers is allowed.
- Offline recordings are fine (no real-time requirement).
- AI tools are allowed for boilerplate. Judges explicitly say **"no AI slop"** — keep the code simple and the README honest.

### Judging

1. **Creativity** — as long as my own data genuinely matters.
2. **Results** — how well the robot performs in sim (success rate).
3. **Clarity** — simple implementation, honest write-up of what worked and what didn't.

Organizers value **repeatable end-to-end task completion over bold but flaky attempts.**

### Submission must contain

- README (or short deck): how to run it, example videos/GIFs, design choices, what worked / what didn't.
- My recorded data (or a representative sample) plus a note on how it was collected.

### Recorded data

- The project uses the **4 existing clips** in `data/raw/<id>/` (the earlier plan of 15–20 new demos is dropped). Each clip is the same task —
  side-wrap pick of my mug, place it on the coaster, release, hand out — filmed from a different viewpoint, and each is a separate execution:
  - `26d4118b0c`: top-down (captured at 30 Hz)
  - `4e103843db`: 90° (camera at my side, square to the desk edge, ~37° from vertical)
  - `49aff04f7a`: 45° (left/right unconfirmed)
  - `e8825dea53`: 45° (left/right unconfirmed; the earlier view check saw the hand cut off during the grasp and lift)
- **Me and the scene:** I used my right hand and sat at the desk's near long edge in all 4 clips. Object: glossy dark-green ceramic mug
  with a handle. Coaster: flat resin disc. Surface: a lap desk (60 cm long, ~38.5 cm deep; the near long edge has a shallow curve) on a bed.
  The phone was propped by hand.
- **`config/objects.yaml` is the single source of object and desk geometry** (mug, coaster, desk; each value carries a `source` flag:
  label / measured / derived / estimated). Perception and sim read it; never hard-code these dimensions.
- **Secondary question:** how the capture viewpoint affects the extracted hand–object trajectory and later robot success.
  Confound: viewpoint and execution vary together; n = 4.

---

## The approach

1. **Record:** iPhone (LiDAR) + **Stray Scanner** app (Record3D is paywalled) gives RGB video + LiDAR depth + confidence + intrinsics + camera odometry + IMU. Third-person view; in frame: mug, coaster, my hand and forearm. The data set is the 4 existing clips (see "Recorded data" above).
2. **Perceive** (`hoi-perception` env): WiLoR for 3D hand pose (MANO), shifted to metric depth with the LiDAR; SAM 2 masks for the mug body, mug handle and coaster; RANSAC plane fit for the table; the known mug geometry (`config/objects.yaml`) fitted to the masks (the glossy mug's depth only initialises the fit). Output: one `.npz` per demo (schema v2).
3. **Retarget** (`hoi-sim` env):
   - Build a **table frame** (z up, origin on the table plane).
   - Express the wrist trajectory **relative to the cup** (approach/grasp/lift) and **relative to the coaster** (transport/place). This object-centric step is what lets the motion generalize to new layouts (the idea comes from OKAMI, CoRL 2024).
   - Map fingers to the Dex3 joints by **fingertip IK with `mink` on the same MuJoCo Dex3 hand model**: human thumb → thumb, index → index, middle (+ring/pinky) → middle. Fallback: `dex-retargeting` run offline in `hoi-perception`, joint angles passed via the `.npz`.
4. **Execute:**
   - **HERO** whole-body controller drives the G1's 29 body joints; the G1 balances itself. HERO tracks a **reference clip**, not a palm target: we build the clip from **our retargeted palm path** (replacing the smooth-path step of HERO's `BenchReplanner`, reusing its IK → FK → `ClipReference` code). HERO's point-to-point replanner is the fallback and an ablation baseline.
   - We drive the Dex3 fingers ourselves (own finger PD loop on HERO's movable-finger model).
   - MuJoCo touch sensors (added by us) on the fingertips confirm the grasp before the lift.
5. **Learn:** generate a few hundred randomized sim demos with object-aware replay, keep only successes, train a small ACT or Diffusion Policy (LeRobot, in `hoi-policy`), **export it to ONNX** and evaluate it closed-loop inside `hoi-sim` alongside HERO.
6. **Prove it:** success rate over 50 random layouts for three variants:
   - (a) naive kinematic replay
   - (b) object-aware replay + contact grasp
   - (c) learned policy

   **Stretch:** the same policy trained on a hand-coded straight-down grasp instead of my motion, to test whether my data matters.

7. **Bonus:** live mode — webcam → MediaPipe → the same retargeting → G1.

### HERO (whole-body controller) — key facts, VERIFIED from code at commit `416c4e0` (details in docs/log.md, 2026-10-28)

- Repo: https://github.com/RunpeiDong/HERO (MIT), cloned in `third_party/HERO`. Model: https://huggingface.co/RunpeiDong/HERO. Inference only — **we do NOT train HERO** (training needs Isaac).
- The example policy (`checkpoints/example/model.onnx` + `model_hero.json`) ships inside the repo. Loaded by `sim2sim.policy_hero_export.HeroExportPolicy(dir, stand_flag=0.0)` on CPU onnxruntime; 50 Hz; outputs 29 body joint targets (not fingers).
- **It does NOT take a palm target.** It tracks a 50 fps reference clip (`sim2sim.reference.ClipReference`): reference arm joints at t+1, the reference palm pose (in the clip's own pelvis frame → palm error inputs), and the reference root pose at t+{0,5,10,15,20}. A world palm goal becomes a clip via mink IK: `sim2sim.bench.replan.BenchReplanner.plan_from_state` (palm path → IK → FK → clip), with closed-loop replanning + goal adjust `g ← g − 0.6·e`.
- Action: `q_target = q_default + action_scale_j · clip(a, ±100)`, plus `(q_ref − q_default)` on the 14 arm joints. `action_scale_j = 0.25 · effort_limit_j / kp_j` (0.075–0.55 per joint, **not** a flat 0.25).
- PD runs in Python, not MuJoCo actuators: `tau = kp(q* − q) − kd·q̇`, clipped to effort limits, written to `qfrc_applied`, 20 × 1 ms substeps per policy step. Use the sidecar kp/kd/effort limits.
- `model_hero.json` (the sidecar) is **authoritative** for observation layout, PD gains, action scales and joint tables. Some things are hard-coded in Python instead (obs scales h01×0.25, h03×2, h10×0.05; joint order; palm offset) — read the code, never assume.
- Tracked "palm" = `wrist_yaw_link` + (0.0415, ±0.003, 0) m = the Dex3 palm origin. Finger bases are ~8 cm further forward.
- Robot model: trained with **fixed** Dex3 fingers (`g1_29dof_dex3fixed_hero.urdf`) and **capsule feet** (`sonic_train`). We use `g1_29dof_with_hand_rev_1_0.urdf` (same masses/geometry, 14 movable finger joints) with capsule feet; box sole (`sonic_box`) only if standing is unstable.
- Object inputs (h17–h19) only activate for ~0.30 m boxes near the floor → we feed zeros (as HERO's own table demo does). No touch sensors in HERO's models — we add them. HERO's benchmark has no table collision.
- Has a MuJoCo benchmark (`scripts/hero_bench.py`) and `sim2sim/` code. Brand new repo (few stars). Expect rough edges.
- **Fallback, if HERO isn't working by end of Day 1:** LeRobot's Unitree G1 MuJoCo sim with `GrootLocomotionController` or `HolosomaLocomotionController` handling the lower body; arms via IK (`mink`). Docs: https://huggingface.co/docs/lerobot/en/unitree_g1

---

## Machine & environments

- **Machine:** Lenovo Legion, Ubuntu (24.04), Intel Core Ultra 9 275HX, 32 GB RAM, **RTX 5060 Laptop GPU (8 GB VRAM, Blackwell)**.
- **Blackwell needs CUDA ≥ 12.8 PyTorch builds.** Repos that pin old torch will fail with "no kernel image". Never let a requirements file downgrade torch — install their other deps with care (e.g. `pip install --no-deps`, then add missing ones).

| env              | python                                            | status                                         | purpose                                                                                                                                                                                                                                           |
| ---------------- | ------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hoi-perception` | 3.10                                              | **exists**, torch 2.11.0+cu128 verified on GPU | WiLoR, SAM 2, Open3D, Stray Scanner parsing                                                                                                                                                                                                            |
| `hoi-sim`        | 3.11                                              | created 2026-10-28; lock in `envs/`            | HERO (editable, `[bench]`), `numpy<2` (HERO pin), `mujoco==3.13.0` (the version HERO names), mink, daqp, onnxruntime (CPU build — HERO's policy is tiny; avoids CUDA/cuDNN mismatches on Blackwell), pytest, imageio. **No torch.** Every later install: `pip install ... -c envs/hoi-sim.constraints.txt`. |
| `hoi-policy`     | 3.12                                              | to create at M5                                | LeRobot (needs Python ≥3.12, numpy ≥2 — conflicts with HERO's `numpy<2`) + torch 2.11.0+cu128. Trains ACT/DP; the policy is exported to ONNX and run inside `hoi-sim`.                                                                       |

### Environment rules

- **Never install anything into `lensing` or `base`** (other project).
- **No ROS 2.** Plain Python + MuJoCo only. Our code must never depend on ROS.
- **Always run our envs with `env -u PYTHONPATH`** (e.g. `env -u PYTHONPATH conda run -n hoi-sim python -m sim.m1_reach_test`): `~/.bashrc` sources
  ROS 2 Jazzy, which puts `/opt/ros/jazzy/...` (Python 3.12 packages) on `PYTHONPATH` ahead of every conda env's site-packages.
- No venv on top of conda; use `pip install` inside the conda env.
- The two envs communicate **only through files** (`data/demos/*.npz`).

---

## Repo layout

```
~/hoi/   (project root; "teach-g1" is the project name)
  CLAUDE.md
  README.md
  third_party/          # cloned repos (HERO, WiLoR, ...) — gitignored; record each repo's exact commit hash in docs/log.md and the README so judges can reproduce
  data/
    raw/                # Stray Scanner recordings (only a small sample committed)
    demos/              # extracted .npz, one per demo
  perception/           # Stray Scanner parsing, hand + object extraction, viz overlays
  retarget/             # table frame, object-centric wrist, finger mapping
  sim/                  # MuJoCo scene (G1 + table + cup + coaster), HERO wrapper, grasp logic
  policy/               # randomized demo generation, training, closed-loop eval
  live/                 # webcam teleop bonus
  results/              # eval logs, metrics json, videos/GIFs
  envs/                 # pip lock + constraints files per conda env (reproducible installs)
  docs/log.md           # running log: what was tried, what broke, numbers
```

---

## Data conventions

- **Units:** meters, radians, seconds.
- **Sim world frame:** MuJoCo, z up. Robot stance centre at the origin, x forward (HERO's benchmark frame).
- **Sim table height: 0.74 m** from now on (one of HERO's three benchmark heights; table near edge at x = 0.30 m, 0.60 m deep).
- **Capture app: Stray Scanner** (https://github.com/StrayRobots/scanner). Verified on the phone: each recording folder has `rgb.mp4`,
  `camera_matrix.csv`, `odometry.csv`, `imu.csv`, `depth/` and `confidence/`.
  - **Format VERIFIED on 4 test clips (2026-10-30, parser `perception/stray.py`, details in docs/log.md):**
    RGB 1920×1440 HEVC; depth uint16 **mm**, 256×192, registered to RGB (scale K by 256/1920); confidence 0/1/2;
    `odometry.csv` = timestamp, frame, x y z, qx qy qz qw (**scalar last**), per-frame fx fy cx cy (RGB pixels); IMU accel in **g**.
  - **Camera poses are camera-to-world with OpenCV camera axes (x right, y down, z forward)** — the app already converts from ARKit.
    World = ARKit session frame, **y up** (gravity −y). Verified: table normal from depth vs odometry up 1.8–5.7°, alternatives ≥ 54°.
  - **Frame alignment:** ffmpeg decodes N−1 video frames for N depth/odometry rows (first frame hidden by the MP4 edit list) →
    decoded frame k = index k+1; index 0 has no image. Use odometry timestamps, never the video clock (it claims 60 fps even for 30 Hz).
  - Phone held in portrait → frames come out sideways; rotate with `stray.upright_rot90` for viewing / detectors.
- **Table frame** (perception, per clip; built in `perception` stage S7, `T_table_world` in the `.npz`):
  - z = table-plane normal (pointing up, away from the desk), origin = initial mug base centre (on the plane).
  - x = **my forward** direction: in-plane perpendicular to the desk's long edges (fitted to the desk boundary in depth), signed so the
    first metric palm position has x < 0 (I sit at the near long edge). y = z × x (so y points to my left).
  - `x_axis_method` records how x was found (long-edge rule, largest-extent fallback, reference-direction fallback, or `failed`).
- **Demo `.npz` schema v2** (`data/demos/<clip>.npz`, written by `perception.build_demo` = S10; `schema_version = 2`; the field
  list below is copied from that module's docstring, which is the master copy). Always written, also for clips that fail
  their gates: what could not be produced is NaN (False for flags, -1 for indices / phase).

```
Frames: "table" = the S7 table frame (z = table normal, up; origin = initial mug base centre on the table; x = my
forward, away from the near desk edge; y = z x x). "cam" = ORIGINAL (sideways) camera frame, OpenCV axes (x right, y down,
z forward). "world" = ARKit world, y up. Units: metres, radians, seconds. T = number of canonical frames i (odometry
rows); i = 0 has no image, so every per-image field is NaN / False there.

Metadata
  schema_version          () int     2
  clip_id, view_label     () str     view_label: "top-down" | "90deg" | "45deg"
  t                       (T,)       odometry timestamps [s], ARKit clock (absolute; t - t[0] = clip time)
  K                       (T,3,3)    RGB intrinsics [px] of the ORIGINAL 1920x1440 image
  T_world_cam             (T,4,4)    camera-to-world
  T_table_world           (4,4)      p_table = T_table_world @ p_world; all NaN if S7 found no table x, and then so is
                                     every table-frame field below
  table_plane_world       (4,)       [nx, ny, nz, d] with n.p + d = 0 (world), n.(0,1,0) > 0
  table_rms_m             ()         table-plane inlier RMS (S4)
  x_axis_method           () str     how S7 found table x: long_edge_extent | largest_total_extent | reference_fallback |
                                     failed ("missing" if S7 has no output)
  x_axis_vs_reference_deg ()         signed angle from the S7 reference direction to table x, about z [deg]
Hand (WiLoR MANO right hand, depth-corrected with the LiDAR in S5)
  hand_valid              (T,) bool  a right hand was detected and reconstructed (S2)
  metric_valid            (T,) bool  hand_valid and >= 30 LiDAR pixels in the S5 correction; use only these frames
  hand_kp_cam             (T,21,3)   21 keypoints after the S5 depth correction, cam frame; NaN where not metric_valid
                                     (S5 corrects only those frames)
  hand_kp_table           (T,21,3)   the same keypoints in the table frame (NaN where not metric_valid)
                                     order (OpenPose, verified in WiLoR's mano_wrapper.py): wrist 0, thumb 1-4,
                                     index 5-8, middle 9-12, ring 13-16, pinky 17-20 (4, 8, 12, 16, 20 = fingertips)
  palm_pos_table_raw      (T,3)      0.5 (kp0 + kp9), metric_valid frames (S9)
  palm_pos_table          (T,3)      gap-filled (gaps <= 0.3 s) and Savitzky-Golay smoothed (S9); finite on the
                                     metric_valid frames AND the filled gap frames
  palm_rot_table_raw      (T,3,3)    columns [x_f, y, n]: x_f = wrist -> middle-finger base, n = palm normal (points
                                     toward the cup at grasp), y = n x x_f
  palm_rot_table          (T,3,3)    gap-filled (SLERP), smoothed, re-orthonormalised (S9)
  palm_normal_flipped     () bool    S9 flipped n in every frame. Also False when the flip was NOT decided (grasp frame
                                     undefined) or S9 has no output: palm/palm.npz flip_decided tells them apart
  mano_global_orient      (T,3,3)    MANO root rotation in the table frame (R_table_cam @ WiLoR's cam-frame global_orient)
  mano_hand_pose          (T,45)     15 MANO joint rotations, axis-angle [rad], each relative to its parent joint
  mano_betas              (T,10)     MANO shape coefficients
                                     (mano_*: WiLoR's own values -- S5 only translates the hand -- on every hand_valid
                                     frame, also where the hand is not metric_valid)
  depth_shift_m           (T,)       S5 depth shift D = median(lidar - mesh) (+ = WiLoR's hand was too close); finite
                                     wherever S5 could compute D, i.e. also on hand_valid frames that are not metric_valid
  lidar_residual_m        (T,)       median |lidar - (mesh + D)| over the S5 pass-2 pixels (finite where D is)
  reproj_err_px           (T,)       plumbing check: corrected kp3d projected with K vs WiLoR's own 2D keypoints
                                     (tests our transforms, NOT WiLoR's accuracy)
Objects (table frame)
  cup_init_pos_table, cup_final_pos_table (3,)  mug base centre on the table plane, initial / final window (S6)
  cup_init_yaw, cup_final_yaw  ()    handle direction: angle of (handle point P - cup axis) from table x, about z [rad]
  cup_yaw_valid           (2,) bool  [init, final]: handle mask >= 0.2 % of the body mask (else do not trust the yaw)
  cup_init_iou, cup_final_iou ()     silhouette IoU of the 2-DOF cylinder fit
  cup_free_fit_r_h        (2,)       [r, h] of the 4-DOF validation fit, initial window; h NaN = unobservable
  coaster_pos_table       (3,)       coaster centre on the table plane (initial window)
  coaster_diameter_fit_m  ()         diameter of the RANSAC circle fit
Events (S8, from the raw metric palm)
  phase                   (T,) int8  0 approach, 1 grasp, 2 transport, 3 place, 4 release/retreat, -1 undefined
  t_contact, t_lift, t_place, t_release ()   [s], same clock as t (NaN = not detected)
  grasp_index             () int     grasp frame g (index into t), -1 = undefined
  T_palm_in_cup_at_grasp  (4,4)      smoothed palm pose at g in the cup frame (origin = initial base centre, z up,
                                     x toward the handle at the initial yaw; table x if that yaw is invalid). Raw pose
                                     if g's valid run is shorter than the smoothing window. NaN if g is undefined, or
                                     if S7 found no table x and the handle yaw is invalid
Dimensions
  mug_dims, coaster_dims  () str     JSON of the config/objects.yaml blocks, copied as they are (value + source flag per
                                     entry; the mug's capacity_ml is a bare number in the config)
```

---

## Milestones (one per day — stop and report at each)

- **M1 — Oct 29 — Sim foundation**
  - `hoi-sim` env created.
  - HERO running in MuJoCo: G1 standing and balancing, tracking a **scripted** palm target (e.g. a circle in front of the chest).
  - Scene with table + cup + coaster.
  - Fallback decision by end of day.
- **M2 — Oct 30 — Perception**
  - Parse Stray Scanner recordings.
  - WiLoR hand + SAM 2 masks + depth produce `.npz` per demo.
  - Overlay video confirming 3D keypoints project back onto the RGB frames.
- **M3 — Oct 31 — Retargeting**
  - Table frame + object-centric wrist trajectory + Dex3 finger mapping.
  - Kinematic replay video in MuJoCo.
- **M4 — Nov 1 — Physical execution**
  - HERO tracks the retargeted palm path, fingers close until fingertip contact, cup lifted and placed.
  - Works on the recorded layouts.
- **M5 — Nov 2 — Learning**
  - Randomized layouts → success-filtered demos → ACT / Diffusion Policy → closed-loop eval on 50 layouts.
- **M6 — Nov 3 — Polish**
  - Ablation table (a/b/c), GIFs.
  - Live webcam mode if time allows.
- **M7 — Nov 4 — Ship**
  - README, data sample, clean install instructions.
  - Push public repo well before 23:59 IST.

---

## Working rules for Claude Code

- **Step by step.** Finish one milestone sub-task, show the result (command output, image, video path), then stop and let me check before moving on.
- **I must understand everything** — I'll be interviewed on this repo. When you make a design choice, explain it in 1–3 sentences. Prefer simple, readable code over clever abstractions.
- **Never fabricate results.**
  - Every number in the README must come from a script in the repo whose output is saved under `results/`.
  - If something fails, say so plainly.
- **Log as you go.** Append to `docs/log.md`: what was tried, what broke, fixes, metrics. This becomes the README's "what worked / what didn't".
- **Verify before assuming.**
  - Read the actual repo code and docs (especially HERO's sidecar and `sim2sim/`) before writing integration code.
  - Mark anything uncertain.
- **Ask before:** downloads over ~2 GB, installing system packages (apt), or changing the plan or task.
- **Git:**
  - Commit at each working step with clear messages.
  - Never commit large raw data or model weights. GitHub's 100 MB file limit applies; keep committed data samples small.
  - do not add claude as coauthor in any commits
- **GPU budget:** 8 GB VRAM. Run perception models one at a time; free memory between stages.
- **Time is the main constraint.** If a sub-task stalls for over ~2 hours, stop and propose the simplest working alternative.
