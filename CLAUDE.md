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
2. **Perceive** (`hoi-perception` env): WiLoR for 3D hand pose (MANO), shifted to metric depth with the LiDAR; SAM 2 masks for the mug body, mug handle and coaster; RANSAC plane fit for the table; the known mug geometry (`config/objects.yaml`) fitted to the masks (the glossy mug's depth only initialises the fit). Output: one `.npz` per demo (schema v4, see below).
3. **Retarget** (`hoi-sim` env):
   - Build a **table frame** (z up, origin on the table plane).
   - Express the wrist trajectory **relative to the cup** (approach/grasp/lift) and **relative to the coaster** (transport/place). This object-centric step is what lets the motion generalize to new layouts (the idea comes from OKAMI, CoRL 2024).
   - **Dropped: the per-digit finger mapping** (fingertip IK with `mink`, fallback `dex-retargeting`). Per-digit IK closing targets were tried in the grasp-fix ladder (Track B L2) and made the grip worse; the Dex3 closes with HERO's own closing targets until the fingertips touch (profile `mug_v1`). No human finger angles are used.
4. **Execute:**
   - **HERO** whole-body controller drives the G1's 29 body joints; the G1 balances itself. HERO tracks a **reference clip**, not a palm target: we build the clip from **our retargeted palm path** (replacing the smooth-path step of HERO's `BenchReplanner`, reusing its IK → FK → `ClipReference` code). The scripted oracle (`sim/pick_place.py`, profile `mug_v1`) is the baseline with no human data.
   - We drive the Dex3 fingers ourselves (own finger PD loop on HERO's movable-finger model).
   - MuJoCo touch sensors (added by us) on the fingertips confirm the grasp before the lift.
5. **Learn (M5 plan change: a learned grasp selector; the ACT / Diffusion Policy / LeRobot plan is dropped):** per layout, choose which of my clips and which grasp heading phi (my phi_h, 75, 65, 55 or 45 deg) the robot uses with the b_hybrid motion.
   - Training data: 18 candidates on 120 disjoint training layouts in `hoi-sim` (2160 rollouts).
   - Model: one pre-registered MLP predicting success, trained with PyTorch in `hoi-perception`, exported to ONNX and run with onnxruntime in `hoi-sim`.
   - Why: M4b showed my heading transfers only up to about 65 deg on this arm.
6. **Prove it (as built):** success over the 50 test layouts (trials 0-49), every row at each layout's own handle yaw:
   - (a) naive kinematic replay; object-aware `b_level` and `b_tilt` (`results/final/own_yaw/`)
   - (b) `b_hybrid` (robot-specific final approach) and `b_hybrid_h45` (also turned to 45 deg) (`results/m5/`, `results/m4b/`)
   - (c) the selector rules R1, R2 and the fixed choice F (`results/m5/`)
   - the scripted oracle (no human data) as the reference line.

   **Dropped:** the stretch comparison (the same policy trained on a hand-coded straight-down grasp).

7. **Dropped:** live mode (webcam → MediaPipe → the same retargeting → G1). Not built; `live/` is an empty placeholder.

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
- **Fallback (not needed: HERO worked on Day 1):** LeRobot's Unitree G1 MuJoCo sim with `GrootLocomotionController` or `HolosomaLocomotionController` handling the lower body; arms via IK (`mink`). Docs: https://huggingface.co/docs/lerobot/en/unitree_g1

---

## Machine & environments

- **Machine:** Lenovo Legion, Ubuntu (24.04), Intel Core Ultra 9 275HX, 32 GB RAM, **RTX 5060 Laptop GPU (8 GB VRAM, Blackwell)**.
- **Blackwell needs CUDA ≥ 12.8 PyTorch builds.** Repos that pin old torch will fail with "no kernel image". Never let a requirements file downgrade torch — install their other deps with care (e.g. `pip install --no-deps`, then add missing ones).

| env              | python                                            | status                                         | purpose                                                                                                                                                                                                                                           |
| ---------------- | ------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hoi-perception` | 3.10                                              | **exists**, torch 2.11.0+cu128 verified on GPU | WiLoR, SAM 2, Open3D, Stray Scanner parsing                                                                                                                                                                                                            |
| `hoi-sim`        | 3.11                                              | created 2026-10-28; lock in `envs/`            | HERO (editable, `[bench]`), `numpy<2` (HERO pin), `mujoco==3.13.0` (the version HERO names), mink, daqp, onnxruntime (CPU build — HERO's policy is tiny; avoids CUDA/cuDNN mismatches on Blackwell), pytest, imageio. **No torch.** Every later install: `pip install ... -c envs/hoi-sim.constraints.txt`. |
| `hoi-policy`     | 3.12                                              | **not created** (M5 plan change)               | Was planned for LeRobot ACT/DP. M5 trains its small selector MLP in `hoi-perception` instead (onnx 1.23.2 added there for the export) and runs it with onnxruntime in `hoi-sim`.                                                                       |
| `hoi-sim-fresh`  | 3.11                                              | created 2026-10-31 (polish Step 7)             | Fresh install of `hoi-sim` from docs/INSTALL.md to check it; `pip freeze` equals `envs/hoi-sim.lock.txt`. Its HERO is an editable install from a temporary folder, so it breaks when that folder is deleted. Remove with `conda env remove -n hoi-sim-fresh`. |

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
    raw/                # my four Stray Scanner recordings (all committed; data/raw/README.md)
    demos/              # extracted .npz, one per demo
  perception/           # Stray Scanner parsing, hand + object extraction, viz overlays
  retarget/             # table frame, object-centric palm paths, summaries and figures
  sim/                  # MuJoCo scene (G1 + table + cup + coaster), HERO wrapper, grasp logic, executor
  policy/               # M5 grasp selector: layouts, features, rules, training, evaluation
  live/                 # empty placeholder (live mode was dropped)
  config/objects.yaml   # mug, coaster and desk geometry
  scripts/              # reproduce.sh (re-runs a few results), hygiene.py
  tests/
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
- **Demo `.npz` schema v4** (`data/demos/<clip>.npz`, written by `perception.build_demo` = S10; `schema_version = 4` since M2 v3,
  perception frozen; v3 removed hand_scale_per_frame / s_hand and added palm_placement, reproj_err_palm_mm, t_rise; the field list
  below is copied from that module's docstring, which is the master copy). Always written, also for clips that fail their gates:
  what could not be produced is NaN (False for flags, -1 for indices / phase).

```
Frames: "table" = the S7 table frame (z = table normal, up; origin = initial mug base centre on the table; x = my
forward, away from the near desk edge; y = z x x). "cam" = ORIGINAL (sideways) camera frame, OpenCV axes (x right, y down,
z forward). "world" = ARKit world, y up. Units: metres, radians, seconds. T = number of canonical frames i (odometry
rows); i = 0 has no image, so every per-image field is NaN / False there.

Metadata
  schema_version          () int     4
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
Hand (WiLoR MANO right hand; S5 moves it along its viewing ray to the LiDAR depth, then the A1v3 palm placement slides
     it parallel to the image plane, t = (t_x, t_y, 0), so that the mean of the 5 palm keypoints {0, 5, 9, 13, 17}
     projects exactly onto the mean of WiLoR's 2D palm keypoints. The scale is 1, the depth stays where LiDAR put it)
  palm_placement          () str     "palm_centroid_ray" ("missing" if there is no hand_place output)
  hand_valid              (T,) bool  a right hand was detected and reconstructed (S2)
  metric_valid            (T,) bool  hand_valid and >= 30 LiDAR pixels in the S5 correction; use only these frames
                                     (the placement keeps S5's metric_valid)
  hand_kp_cam             (T,21,3)   21 keypoints of the placed hand (S5 depth shift, then the sideways placement),
                                     cam frame; NaN where not metric_valid (S5 and the placement act only there)
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
                                     (mano_*: WiLoR's own values on every hand_valid frame, also where the hand is not
                                     metric_valid. S5 and the placement only move the hand, so the metric hand is the
                                     MANO hand of these values plus a per-frame translation: align its keypoints to
                                     hand_kp_cam / hand_kp_table)
  depth_shift_m           (T,)       S5 depth shift D = median(lidar - mesh) (+ = WiLoR's hand was too close); finite
                                     wherever S5 could compute D, i.e. also on hand_valid frames that are not metric_valid
  lidar_residual_m        (T,)       re-measured on the placed hand (no new depth shift): median |lidar - mesh| over the
                                     hand pixels within 3 cm of the mesh; metric_valid frames only, NaN where no pixel is
                                     left
  reproj_err_palm_px      (T,)       median over the 5 palm keypoints (wrist, index / middle / ring / pinky bases) of
                                     |project(K, kp) - WiLoR's 2D keypoint| [px], after the placement; metric_valid frames
  reproj_err_palm_mm      (T,)       reproj_err_palm_px * z_c / fx * 1000 [mm]: the same error at the hand's distance (z_c = depth
                                     of the S5 mesh centroid, fx of that frame); metric_valid frames. Information only
                                     (monocular hand-model limitation: orthographic model vs a 35-45 cm perspective camera)
  reproj_err_all_px       (T,)       the same median over all 21 keypoints (the fingers included)
  reproj_err_px           (T,)       S5's plumbing check from BEFORE the placement: the S5 hand's 21 keypoints projected
                                     with K vs WiLoR's own 2D keypoints (tests our transforms, NOT WiLoR's accuracy)
Objects (table frame)
  cup_init_pos_table, cup_final_pos_table (3,)  mug base centre on the table plane, initial / final window (S6)
  cup_init_yaw, cup_final_yaw  ()    handle direction: angle of (handle point P - cup axis) from table x, about z [rad]
  cup_yaw_valid           (2,) bool  [init, final]: handle mask >= 0.2 % of the body mask (else do not trust the yaw)
  cup_init_iou, cup_final_iou ()     silhouette IoU of the 2-DOF cylinder fit
  cup_free_fit_r_h        (2,)       [r, h] of the 4-DOF validation fit, initial window; h NaN = unobservable
  coaster_pos_table       (3,)       coaster centre on the table plane (initial window)
  coaster_diameter_fit_m  ()         diameter of the RANSAC circle fit
  mug_mask_source         () str     the mug body masks behind the cup fits above (S6, chosen by the A4 rule for all
                                     clips): "video" (SAM 2 video predictor, half res) | "full_res_image" (SAM 2 image
                                     predictor on the full-res frames of the two static windows)
Events (S8, from the raw metric palm)
  phase                   (T,) int8  0 approach, 1 grasp, 2 transport, 3 place, 4 release/retreat, -1 undefined
  t_contact, t_lift, t_place, t_release ()   [s], same clock as t (NaN = not detected)
  t_rise                  ()         [s], same clock as t: the first NEAR frame (d_axis <= r + 8 cm) whose palm z is
                                     >= 3 cm above the running minimum of z over the near frames so far, and stays so
                                     on the next 4 metric_valid frames (A2v3). NaN = not found; then every event is
                                     undefined
  contact_rule            () str     "pre_rise_min_plus_1cm": D_min = min d_axis over the metric_valid frames BEFORE
                                     t_rise (d_axis = horizontal palm distance to the initial cup axis);
                                     t_contact = first metric_valid frame with d_axis <= D_min + 1 cm
  d_axis_min_m            ()         D_min, as above (NaN = undefined)
  d_axis_min_index        () int     the frame of D_min (index into t), -1 = undefined
  grasp_index             () int     grasp frame g (index into t), -1 = undefined. g = last valid frame <= lift - 3 (S8),
                                     which is not forced to come after t_contact; if it comes before, phase[g] is 0
                                     (approach)
  T_palm_in_cup_at_grasp  (4,4)      smoothed palm pose at g in the cup frame (origin = initial base centre, z up,
                                     x toward the handle at the initial yaw; table x if that yaw is invalid). Raw pose
                                     if g's valid run is shorter than the smoothing window. NaN if g is undefined, or
                                     if S7 found no table x and the handle yaw is invalid
Dimensions
  mug_dims, coaster_dims  () str     JSON of the config/objects.yaml blocks, copied as they are (value + source flag per
                                     entry; the mug's capacity_ml is a bare number in the config)
```


---

## Retarget conventions

Code in `retarget/` (hoi-sim, kinematic only), outputs in `results/m3_retarget/`. Details and results are in docs/log.md "M3".

- **Frames:** human table frame to sim world = identity rotation (x forward, y left, z up) plus a per-layout xy translation; human table z = 0 maps to the sim table top z = 0.74. Human cup positions are base centres, the sim cup pose is the body centre (cup_height / 2 higher); convert explicitly (`frames.base_to_centre`).
- **Palm axes (verified from the model at the oracle grasp):** robot palm x points toward the finger bases, palm y (the palm normal) toward the object, z = x x y. The cup axis is `o = (0.077, r + 0.015, 0)` in the palm frame.
- **M:** `R_robot = R_human @ M`, M columns (e1, e3, -e2), det = 1 (robot x = x_f, y = n, z = x_f x n). Human palm frame columns are [x_f, y_h, n].
- **Heading:** `phi(R) = atan2(R[1,1], R[0,1])` for robot rotations; for human rotations the n column `atan2(R_h[1,2], R_h[0,2])`.
- **Grasp variants (both: q = cup axis xy, table top + h_frac_h * h; p = q - R * o):** `b_level` R = R_z(phi_h - phi_ora) @ R_ora (my heading, level wrist); `b_tilt` R = R_h(g) @ M (full wrist). `a_naive` replays the mapped human palm and R_h(t) @ M at a fixed recorded-layout translation. phi_ora = 45 deg, h_frac_ora = 0.714.
- **a0 rule:** a0 = first frame of the last contiguous run ending at g with finite palm_pos_table and z >= 0.02 m (removes the below-table entry artefact).
- **Time scale k:** per clip, at the recorded layout: `k = max(1, v_h / v_o)`; v_h = p95 of smoothed palm speed over approach + transport, v_o = larger oracle min-jerk peak (pregrasp 0.244 m/s, transport 0.15 m/s). Human times x k, resampled to 50 Hz.
- **Prelude (robot-specific):** standing palm to (p_s.x, p_s.y, z_clear) over 1.2 s ending at R_G, like the oracle's first waypoint.
- **Warps (b):** approach = mapped human path plus a smooth-step offset blending the start to p_r0 and the end to G (w = 10s^3 - 15s^4 + 6s^5), z clamped >= table + 0.03 m (count recorded), orientation held at R_G, 0.5 s pause (`i_pause`) at 0.08 m arc length to G; close 1.75 s at G (closure ramps 0 to 1 in 1.25 s); transport = mapped path blended linearly from G to P, bump over the G to P line capped at 0.10 m; release and retreat as the oracle.
- **Reachable:** FK error at G and at P < 1 cm and < 10 deg, and path max < 2 cm. Joint-limit hit = any arm joint within 0.01 rad of its limit.
- **Output format:** `results/m3_retarget/paths/<clip>__<variant>__<layout>.npz` with `palm_pos` (T,3), `palm_quat_wxyz` (T,4) (world frame, 50 Hz), `closure` (T,) in [0, 1] (0 = HERO open), `phase` (T,) fixed-width unicode strings (prelude, approach, pause, close, transport, release, retreat; naive: prelude, to_a0, replay, release), and the indices `i_pause` (-1 for naive), `i_G`, `i_P`, `i_release`. Per-run metrics in `runs/<same name>.json`; summary in `summary.md` / `summary.json`.
- **Plan API:** `PathPlanner.plan_path(..., return_fk=True)` additionally returns the per-frame FK palm position and wxyz quaternion; default output unchanged.
- **v2 timing (M4, `retarget/paths_v2.py`):** M3's path with the pause hold removed. The approach is split at i_pause and the transport at its highest palm sample (i_peak). Each human segment (approach 1, approach 2, transport 1, transport 2) is re-timed rest to rest: D = D0 + 0.4 s, speed factor smoothstep(t/0.4) at both ends, sigma(D) = D0. Geometry, warps and k are M3's (built by M3's own functions, unchanged). Naive: the replay is re-timed as one segment, closure rule kept at the mapped frames. v2 npz = M3's fields + i_prelude_end, i_close_end, i_peak (-1 = none) in `results/m3_retarget/v2/paths/`; the returned dict also has `follow` (inclusive frame ranges: approach1 [0, i_pause], approach2 [i_pause, i_G], transport1 [i_close_end, i_peak], transport2 [i_peak, i_P]; naive: all).
- **Oracle path:** `paths_v2.oracle_path` = the oracle's nominal waypoints through `min_jerk_path` in M3's phase names (cup at rest, nominal in-hand offset as in `warp.time_scale`). It goes through M3's harness unchanged (`run_m3.run_one` with `build_path` swapped).
- **Place palm:** the nominal oracle palm = cup + legacy palm_offset[:2], so place palm = coaster + offset (M3's spec sign, coaster - offset, was wrong). With the right sign all 4 recorded place palms are still outside PALM_BOX, on y only (3.7-6.8 cm).
- **Executor (`sim/m4_execute.py`, `Execute(PickPlace)`, mug_v1, 0.40 kg):** `follow(phase, pos, quat, base, base_goal)` plays a dense 50 Hz slice (one `plan_path`, one `_tick` per frame) and bends frame 0 onto the live IK palm with w = 10s^3 - 15s^4 + 6s^5; the end frame is unchanged. b_level / b_tilt: follow prelude+approach 1, `settle_pregrasp`; follow approach 2, `settle_grasp`, `check_cup_undisturbed`; `close_hand`, `check_grasp`; follow transport 1 to the peak (lift check), re-warp transport 2 to P' = cup_place - cup_in_hand, follow it, `check_in_hand`, `settle_place`, `check_in_hand`; open, retreat, `evaluate()`; every check is the oracle's own. naive: one follow with per-tick closure, no settles, checks or place correction, then retreat and `evaluate()`. The place correction uses privileged simulator state (the cup's true pose in the hand at the peak), exactly as the oracle's place target does; say so in the README for both.
- **Check order:** `settle_grasp` runs before `check_cup_undisturbed`, so a mug pushed by the open hand is usually filed as "reach" (palm could not reach the target), not "missed/knocked cup"; look at `detail.cup_final` vs `cup_xy` before reading "reach" as unreachable.
- **Collision diagnostic:** `retarget.collision.collision_min` (min signed distance, right-hand geoms to the mug, over the approach window on M3's single-plan IK clip), information only, never a gate.
- **M4 results:** `results/m4_execute/summary.md` (62/600 trial-run successes; oracle 44/50 on the same layouts; b_level 23/200, b_tilt 39/200, naive 0/200; 1/4 on the recorded layout v2). Regenerate with `python -m retarget.m4_summary`; rerun any row with `python -m sim.m4_execute rerun --row <row.json> --report <file>`.
- **M4b hybrid (`paths_v2.hybrid_path`; variants `b_hybrid`, `b_hybrid_h45`):**
  - Grasp: b_hybrid uses b_level's grasp; h45 uses R_ora at my grasp height (`hybrid_grasp`).
  - u = M3's horizontal unit vector of (q - G); PG = G - 0.08 u.
  - Approach: my a0..g warped onto p_r0 -> PG (M3's formula and z clamp), no pause, re-timed as one segment (i_pause = PG). Then the
    oracle's settle at PG (5 s) and its own `move` PG -> G (2.0 s); path frames i_pause..i_G (phase "final_approach") are that leg's
    nominal record.
  - Close to retreat: M4's b executor.
  - Handle yaw: h45 uses the trial's own (`variant_layout`).
  - Collision window ends 1 cm of arc before PG.
  - b_hybrid self-checks against v2 b_level on every build.
- **category_v2 (`retarget/m4b_summary.py`):** "pushed mug before close" if the run failed before close_hand (approach check, a
  pre-grasp / grasp settle, or a fall before the close) and |cup_final xy - cup_xy| > 2 cm; otherwise `category`. It is computed from
  the rows; the rows are never edited.
- **Heading sweep (`sim/m4b_heading_sweep.py`):**
  - Track B final's mug_v1 oracle with `task.grasp` turned by phi - 45 deg about the cup axis (offset, quaternion, approach_dir);
    handle yaw = trial + delta.
  - Trials 0-19.
  - phi = 45 must reproduce fifty.json exactly.
- **M4b results:** `results/m4b/summary.md`.
  - b_hybrid 112/200; b_hybrid_h45 179/200; oracle 44/50.
  - Sweep: phi 45/55/65 20/20, 75 13/20, 85 8/20.
  - The comparisons are a sequential ablation (approach bundle, then heading), so the ranking of the two factors is not tested.
  - Regenerate with `python -m retarget.m4b_summary summarize`.
- **M5 selector (`policy/`):**
  - `features.py` / `rules.py`: numpy only, shared by both envs.
  - `m5.py`: hoi-sim layouts, items, the candidate table, the ONNX check, selection, evaluation and the summary.
  - `train_selector.py`: hoi-perception CV and training.
  - `b_hybrid_phi` = b_hybrid with the grasp heading phi. Candidates: "phih", or phi_ora + (label - 45) deg; "45" = phi_ora exactly.
  - One handle yaw per layout (its own). Run names `<clip>__phi<label>__<layout>[__noisy]`.
  - Training layouts `train000-119` (results/m5/train_layouts.json). Test = trials 0-49, never used for training or selection.
  - Inputs: noisy (cup / coaster N(0, 1 cm), yaw N(0, 10 deg)). Training: 5 copies per row, seed 0. CV held-out: per layout, seed 1.
    Test: per trial, seed 2026.
  - Rules: R1 = argmax p; R2 = the smallest departure among p >= 0.8, else R1; F = the fixed-best training candidate.
- **M5 results:** `results/m5/summary.md`.
  - Clean: R2 49/50, R1 47/50, F 46/50.
  - Noisy path: R2 44/50, F 38/50.
  - CV: AUC 0.916; R1 = F = 115/120 held out (D2: learning does not beat F in CV).
  - Regenerate with `python -m policy.m5 summarize`. Reproduce check: `bash scripts/reproduce.sh`.
- **Own yaw (polish, `run --own-yaw`):** naive / b_level / b_tilt re-run at each trial's own handle yaw (M4 turned the handle with
  my heading: trial + (phi_h - phi_ora)). Rows keep M4's names, in `results/final/own_yaw/runs/` (field `own_yaw: true`).
  Summary: `python -m retarget.own_yaw_summary` -> `results/final/own_yaw/summary.md`. The README's table uses these rows, M5's
  b_hybrid at the original yaw (98/200) and M4b's b_hybrid_h45; the M4 / M4b turned-handle numbers stay in docs/details.md.

---

## Milestones: what happened each day (from docs/log.md and the git history)

- **Wed 28 Oct:** repo skeleton; HERO cloned and read (the key facts above); hoi-sim env plan.
- **Thu 29 Oct (M1):** hoi-sim env; MuJoCo scene with table, mug and coaster; HERO standing and reaching; reference builder and the
  first scripted pick-and-place with a touch-gated grasp. HERO worked, so the LeRobot fallback was not needed.
- **Fri 30 Oct (M2 + grasp fix):** hoi-perception env (WiLoR, SAM 2, MANO); Stray Scanner parser and format checks; perception on
  all 4 clips in three versions (v1 → v2 → v3, frozen after v3, schema 4). In parallel (Track B) the scripted grasp failed on my
  mug's model (0/50, against 50/50 on the test cylinder); the fix ladder ended in the `mug_v1` profile (44/50, just after midnight).
- **Sat 31 Oct (M3-M7):** M3 kinematic retargeting; M4 physical execution (62/600 trial runs); M4b fix (b_hybrid, heading sweep);
  M5 learned selector overnight (2160 training rollouts, R2 49/50); figures, GIFs, README draft, reproduce.sh, clean-clone check.
  Then the polish step: own-yaw ablation, README restructure, LICENSE, all four raw clips committed, fresh-env check.
- **Sun 1 to Wed 4 Nov:** owner review, then push the public repo (code freeze 4 Nov 12:00 IST, deadline 23:59 IST).

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
