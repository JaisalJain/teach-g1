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

1. **Record:** iPhone (LiDAR) + Record3D app gives RGB-D + intrinsics + camera pose. Static third-person view, about 60–80 cm from the table at about 45°. In frame: cup, coaster, my hand and forearm. 15–20 demos, cup and coaster in varied positions. Matte cup.
2. **Perceive** (`hoi-perception` env): WiLoR for 3D hand pose (MANO), SAM 2 for cup and coaster masks, LiDAR depth for metric object positions, RANSAC plane fit for the table. Output: one `.npz` per demo.
3. **Retarget** (`hoi-sim` env):
   - Build a **table frame** (z up, origin on the table plane).
   - Express the wrist trajectory **relative to the cup** (approach/grasp/lift) and **relative to the coaster** (transport/place). This object-centric step is what lets the motion generalize to new layouts (the idea comes from OKAMI, CoRL 2024).
   - Map fingers to the Dex3 joints with `dex-retargeting`: human thumb → thumb, index → index, middle (+ring/pinky) → middle.
4. **Execute:**
   - **HERO** whole-body controller drives the G1's 29 body joints from palm targets; the G1 balances itself.
   - We drive the Dex3 fingers ourselves.
   - MuJoCo touch sensors on the fingertips confirm the grasp before the lift.
5. **Learn:** generate a few hundred randomized sim demos with object-aware replay, keep only successes, train a small ACT or Diffusion Policy (LeRobot), evaluate closed-loop.
6. **Prove it:** success rate over 50 random layouts for three variants:
   - (a) naive kinematic replay
   - (b) object-aware replay + contact grasp
   - (c) learned policy

   **Stretch:** the same policy trained on a hand-coded straight-down grasp instead of my motion, to test whether my data matters.

7. **Bonus:** live mode — webcam → MediaPipe → the same retargeting → G1.

### HERO (whole-body controller) — key facts, mostly UNVERIFIED by us

- Repo: https://github.com/RunpeiDong/HERO (MIT). Model: https://huggingface.co/RunpeiDong/HERO
- Pretrained ONNX policy for G1 + Dex3. It tracks **world-fixed palm targets** and outputs 29 body joint targets at 50 Hz.
- Actions are residuals around reference arm joints. Joint target = default pose + 0.25 · clip(action), plus the arm reference.
- `model_hero.json` (the sidecar) is **authoritative** for observation layout, scales, PD gains and joint tables. Never assume a layout.
- Has a MuJoCo benchmark (`scripts/hero_bench.py`, `pip install -e ".[bench]"`) and `sim2sim/` code. Training needs Isaac — **we do NOT train HERO**, inference only.
- Brand new repo (few stars). Expect rough edges.
- **Fallback, if HERO isn't working by end of Day 1:** LeRobot's Unitree G1 MuJoCo sim with `GrootLocomotionController` or `HolosomaLocomotionController` handling the lower body; arms via IK (`mink`). Docs: https://huggingface.co/docs/lerobot/en/unitree_g1

---

## Machine & environments

- **Machine:** Lenovo Legion, Ubuntu (24.04), Intel Core Ultra 9 275HX, 32 GB RAM, **RTX 5060 Laptop GPU (8 GB VRAM, Blackwell)**.
- **Blackwell needs CUDA ≥ 12.8 PyTorch builds.** Repos that pin old torch will fail with "no kernel image". Never let a requirements file downgrade torch — install their other deps with care (e.g. `pip install --no-deps`, then add missing ones).

| env              | python                                            | status                                         | purpose                                                                                                                                                                                                                                           |
| ---------------- | ------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hoi-perception` | 3.10                                              | **exists**, torch 2.11.0+cu128 verified on GPU | WiLoR, SAM 2, Open3D, Record3D parsing                                                                                                                                                                                                            |
| `hoi-sim`        | 3.11 (verify against HERO / LeRobot requirements) | to create                                      | mujoco, onnxruntime (CPU build — HERO's policy is tiny; avoids CUDA/cuDNN mismatches on Blackwell), dex-retargeting, HERO, LeRobot. If LeRobot's deps conflict with HERO's, give LeRobot its own env (`hoi-policy`) rather than forcing versions. |

### Environment rules

- **Never install anything into `lensing` or `base`** (other project).
- **No ROS 2.** Plain Python + MuJoCo only.
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
    raw/                # Record3D exports (only a small sample committed)
    demos/              # extracted .npz, one per demo
  perception/           # Record3D parsing, hand + object extraction, viz overlays
  retarget/             # table frame, object-centric wrist, finger mapping
  sim/                  # MuJoCo scene (G1 + table + cup + coaster), HERO wrapper, grasp logic
  policy/               # randomized demo generation, training, closed-loop eval
  live/                 # webcam teleop bonus
  results/              # eval logs, metrics json, videos/GIFs
  docs/log.md           # running log: what was tried, what broke, numbers
```

---

## Data conventions

- **Units:** meters, radians, seconds.
- **Sim world frame:** MuJoCo, z up.
- **Record3D / ARKit camera frame is y-up, −z-forward** (VERIFY on real export). Convert explicitly and document the convention in code.
- **Record3D export format:** VERIFY on the first real file before writing parsers.
  - Expected: `.r3d` (a zip with `metadata` json, RGB jpgs, compressed depth, confidence maps), or "EXR + JPG" export.
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
  - Parse Record3D.
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
