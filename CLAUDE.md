# CLAUDE.md — teach-g1

## What this project is

Entry for the **"Humanoid Challenge: Teach a Humanoid With Your Own Hands"** (Awign × Armature AI Labs × RIC × Tanna TechBiz).

**Deadline: Wed 4 Nov 2026, 23:59 IST.** Submission = a public GitHub repo. Code freeze target: Nov 4, 12:00 IST.

**One-line pitch:** From ~15 iPhone LiDAR clips of my own hand, a standing Unitree G1 learns to pick up a cup and place it on a coaster from anywhere on the table — with a measured success rate.

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

---

## The approach

1. **Record:** iPhone (LiDAR) + **Stray Scanner** app (Record3D is paywalled) gives RGB video + LiDAR depth + confidence + intrinsics + camera odometry + IMU. Static third-person view, about 60–80 cm from the table at about 45°. In frame: cup, coaster, my hand and forearm. 15–20 demos, cup and coaster in varied positions. Matte cup.
2. **Perceive** (`hoi-perception` env): WiLoR for 3D hand pose (MANO), SAM 2 for cup and coaster masks, LiDAR depth for metric object positions, RANSAC plane fit for the table. Output: one `.npz` per demo.
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
  - **Format details NOT yet confirmed** — check against the repo's `docs/format.md` and the real test clips before writing the parser
    (depth encoding/units/resolution, odometry column order + quaternion convention, timestamps and how frames align with `rgb.mp4`).
  - ARKit camera frame is expected to be y-up, −z-forward (VERIFY on real data). Convert explicitly and document the convention in code.
  - LiDAR depth is low-res (likely 256×192) vs RGB — upsample depth or downsample RGB consistently.
- **Demo `.npz` schema** (camera frame unless noted). Keep fields stable once the retargeter depends on them:
  - `t` (T,) — timestamps
  - `K` (3,3) — intrinsics
  - `hand_kp` (T,21,3) — MANO keypoints
  - `wrist_pos` (T,3), `wrist_rot` (T,3,3)
  - `mano_pose` (T,45) — optional
  - `hand_valid` (T,) bool
  - `cup_pos` (T,3), `coaster_pos` (T,3), `cup_valid` (T,)
  - `table_plane` (4,) — [nx, ny, nz, d]
  - `phase` (T,) int — 0 approach, 1 grasp, 2 lift/transport, 3 place, 4 release/retreat. Segment from finger closure + cup motion.

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
