# teach-g1 — running log

What was tried, what broke, fixes, numbers. Newest entries at the bottom.

---

## 2026-10-28 (Wed) — M1 sub-task 1: repo skeleton, HERO clone, hoi-sim env plan

### Repo
- `git init` (branch `main`), `.gitignore` (third_party/, data/raw/ except `data/raw/sample/`, `*.r3d`, weights
  `*.onnx *.pt *.pth *.ckpt *.safetensors *.bin`, videos `*.mp4 *.mov ...`; GIFs are allowed), layout from CLAUDE.md with `.gitkeep`s.

### Third-party repos (gitignored; exact commits for reproducibility)
| repo | path | commit |
|---|---|---|
| RunpeiDong/HERO | `third_party/HERO` | `416c4e007e2254524322a1771f267a57b6a69994` (2026-10-15, "Code release") |

- HERO ships its own G1 + Dex3 assets (`assets/robots/g1_modified/`) and the example policy
  (`checkpoints/example/model.onnx` 5.5 MB + `model_hero.json` sidecar) inside the git repo — no separate download needed.
- **MuJoCo Menagerie** is already cloned at `/home/jaisal/mujoco_menagerie` (commit `accb6df4`, has `unitree_g1/`).
  NOT used for HERO (HERO must use its bundled G1+Dex3 model, which matches the policy's training URDF).
  Available as a reference / fallback model (e.g. for the LeRobot fallback or for comparing collision geometry).

### HERO facts verified by reading the code/docs (commit 416c4e0)
- **Python:** `requires-python = ">=3.10"` (pyproject). README: "Python 3.10+". Training side is Isaac Sim 5.1 / Isaac Lab 2.3 (Python 3.11) — not used.
- **Core deps** (installed by plain `pip install -e .`): `numpy>=1.24,<2`, `scipy>=1.10`, `mujoco>=3.3,<4`, `mink>=1.3,<2`, `daqp>=0.6`,
  `onnxruntime>=1.18,<2`. **No torch.**
- **`[bench]` extra** = `numpy>=1.24,<2`, `mujoco>=3.3,<4`, `mink>=1.3,<2`, `daqp>=0.6`, `onnxruntime>=1.18,<2` — i.e. the core deps
  minus scipy, listed again "so the scoring stack is explicit". It adds nothing beyond the core install.
- MuJoCo-side code (`sim2sim/`, `data_tools/`, `scripts/hero_bench.py`) imports only numpy, mujoco, mink, onnxruntime (+ stdlib) and
  `hero_isaacsim.constants` / `hero_isaacsim.paths` (both pure-Python, no torch/Isaac). torch only appears in training code.
  No numpy-2-removed APIs found by grep in the MuJoCo-side code.
- Sidecar `model_hero.json` (authoritative): ONNX inputs `actor_obs_lower_body` + `actor_obs_upper_body` (same 1000-d vector),
  output `action` (29). 5-frame history × 200-d frame, 20 terms sorted by name, oldest frame first. 50 Hz (`policy_dt` 0.02).
- **Correction to CLAUDE.md:** the action scale is not a flat 0.25. `q_target = q_default + action_scales * a + (q_ref - q_default)` on the
  14 arm dofs (plain `q_default + action_scales * a` on legs/waist), with per-joint `action_scales = 0.25 * effort_limit / kp`
  (0.351 / 0.548 / 0.439 / 0.075 depending on joint), `action_clip = 100`.
- **The policy never sees a world-frame palm target.** Its inputs are reference arm joints (`h12_ref_upper_dof_pos`), palm position/rotation
  *errors* (`h07`, `h08`), and future reference root pose/rot/height (`h20`–`h22`). A palm target must first be turned into a reference
  (IK) — details below.

### hoi-sim env: dependency findings (pip `--dry-run` resolution, nothing installed)
- Latest on PyPI today: mujoco 3.17.0, onnxruntime 1.31.1 (needs Python ≥3.11), mink 1.3.0 (needs mujoco ≥3.10), daqp 0.10.3.
- **Conflict:** HERO pins `numpy<2`. dex-retargeting 0.5.0 needs `numpy>=2` (and no torch); 0.4.6 is the last numpy-1 release and needs
  `torch` (unpinned) + `nlopt<2.8` + resolves to old `pin==2.7.0`. LeRobot 0.6.1 needs Python ≥3.12, `numpy>=2,<2.3`, `torch<2.12`.
- dex-retargeting needs Python <3.13. torch 2.11.0+cu128 (same as hoi-perception, has sm_120) is the newest build on the cu128 index.
- Both candidate stacks resolve wheel-only: (A) py3.11 + numpy 1.26.4 + HERO pins + dex-retargeting 0.4.6 + torch 2.11.0;
  (B) py3.12 + numpy 2.2.6 + dex-retargeting 0.5.0 + lerobot 0.6.1 + torch 2.11.0 (B overrides HERO's numpy pin).
- MuJoCo release dates: 3.13.0 (2026-09-09) … 3.16.0 (10-14), **3.17.0 (10-22) is newer than HERO's code release (10-15)**, so HERO
  cannot have tested it. The only MuJoCo version HERO names is 3.13.0 (browser demo WASM build) → pin `mujoco==3.13.0`.
- Proposed M1 install set (dry-run, py3.11): 25 packages, `numpy 1.26.4, scipy 1.17.1, mujoco 3.13.0, mink 1.3.0, daqp 0.10.3,
  onnxruntime 1.31.1` (+ pytest, imageio[ffmpeg]); **no torch, no nvidia-* packages**. Env not created yet (awaiting approval).

### How HERO runs in MuJoCo (read from code; 5 reader agents + 5 adversarial verifiers, 159/172 claims confirmed, rest corrected)
Read-only — no HERO code was executed. Key spot-checks redone by hand (finger joint types, ORT session, ONNX sha256).
- **Loading:** `sim2sim.policy_hero_export.HeroExportPolicy(onnx_dir, stand_flag=0.0, ...)` (policy_hero_export.py:1275). Picks the
  `*_hero.json` sidecar + the ONNX it names, `onnxruntime.InferenceSession(..., providers=("CPUExecutionProvider",))`, 1 thread.
  Checks only input width 1000 / output 29 (no hash check at load; example ONNX sha256 `521402c0…2c33` matches `DEMO_SHA256` in
  `export_policy_assets.py:17` ✔ verified). Sidecar > ONNX metadata > code defaults. Hard-coded in Python, NOT in the sidecar:
  obs scales `h01×0.25, h03×2.0, h10×0.05`, joint order (`hero_isaacsim.constants.DOF_NAMES`), palm offset.
- **Per 50 Hz step:** `q_target = policy.control(ref, t, st)` → builds 20 terms → scale → clip ±100 → 5-frame history (zeros in front
  after reset) → same 1000-d float32 vector to both ONNX inputs → `q = q_default + action_scale*clip(a,±100)` (+ `q_ref − q_default` on
  the 14 arm joints). `frame_major_to_term_major_perm` is done inside the ONNX graph.
- **PD is in Python, not MuJoCo actuators:** `MujocoPlant.step` computes `tau = kp(q*−q) − kd·qd`, clips to effort limits, writes
  `qfrc_applied`, 20 substeps × 1 ms. Uses the sidecar kp/kd (the bench passes `policy.kp/kd/effort_limit`). Armature from HERO tables,
  joint damping = frictionloss = 0. `plant_params.ACTION_CLIP=20` disagrees with the sidecar's 100 — use the sidecar.
- **How palm targets reach the policy:** the policy only reads a 50 fps **reference clip** (`sim2sim.reference.ClipReference`): arm
  joints at t+1 (h12 + residual), palm pose of the reference in its own pelvis frame (h07/h08 = robot palm − reference palm, both in
  the respective pelvis frame), reference root at t+{0,5,10,15,20} (h20–h22), reference pelvis height (h03). A world palm goal becomes a
  reference via **mink IK** in Python: `sim2sim.bench.replan.BenchReplanner(goal, cfg).plan_from_state(st, plant, goal_pos_w)`
  (replan.py:786) → per-frame palm path (smoothstep, reach time `clip(0.4+1.2·dist+…, 0.5, 3.0)` s) → mink/daqp IK (≤12 iters/frame,
  joint speed cap 0.08 rad/frame) on HERO's IK MJCF → FK on `g1_29dof_with_hand_rev_1_0.xml` → `ClipReference.from_arrays(fps=50)`.
  Closed loop: replan periodically from the live state + goal adjust `g ← g − 0.6·e` (≤1 cm/update, stop when |e| ≤ 1.75 cm).
  Built-in planning is point-to-point; a moving target (circle) means repeated replans or a hand-built `SegmentPlan` (untested).
- **Palm point** = `wrist_yaw_link + R·(0.0415, ±0.003, 0)` = origin of `*_hand_palm_link` in the training URDF. Dex3 finger bases are
  a further ~8 cm forward, so a grasp palm target sits ~8–12 cm short of the cup centre (browser demo uses `[0.077, ±0.055, 0]`).
- **Robot model:** the plant is built at runtime from a URDF via `MjSpec` (free joint "root" on pelvis, floor, foot-collision swap,
  robot geoms `contype=1, conaffinity=0`, no self-collision). Training URDF `g1_29dof_dex3fixed_hero.urdf`: **14 finger joints fixed**
  ✔ verified. `g1_29dof_with_hand_rev_1_0.urdf`: same 14 joints **revolute** ✔ verified, otherwise same masses/inertias/collision shapes
  (each hand 0.697 kg, 0.324 kg of it in the fingers). The browser pick-and-place demo uses the revolute one → that is our model.
- **Feet:** sidecar says training used `sonic_train` (7 capsules); bench/plant default is `sonic_box`. Must choose deliberately (log it).
- **Object terms h17–h19:** example policy only flags ~0.30 m boxes starting ≤ 0.12 m above the floor; the browser demo always feeds
  zeros. A cup on a table is out of distribution → keep h17–h19 = 0 (the default when `RobotState.object_pos_w` is None).
- **No touch sensors** anywhere in HERO's models or code → we add `<touch>` sites in our scene (or count contacts).
- **Fingers:** `MujocoPlant.step` zeroes `qfrc_applied` each substep and only drives the 29 body joints, so finger PD must live in our own
  substep loop (as `interactive_scene.DemoScene.step`: kp 1.2, kd 0.04, effort 2.45/1.4 N·m, 2 rad/s target slew) or in MuJoCo actuators
  for the 14 finger joints. The browser demo additionally triples finger kp while closed and adds a 0.6×effort squeeze — Python doesn't.
- **Scene template:** `DemoScene(table_height=0.74)` has a physical table (0.50/0.74/0.88 m only), a 0.12 kg mug, movable fingers,
  explicit hand–object contact pairs (condim 4). It **cannot be constructed in this clone**: it requires the YCB cracker-box texture
  (`assets/objects/` absent until `scripts/fetch_ycb_cracker_box.py` downloads it). Plan: copy its table/contact patterns into our own
  `spec_callback` rather than depend on it. Validated object workspace there: x 0.30–0.57 m, |y| ≤ 0.40 m in front of the robot at x=−0.03.
- **HERO's benchmark has no table** (palm may pass through it); its IK can take a `TableBox` for clearance (hand capsule ≥ 2 cm).
- **Gotchas:** quaternions are wxyz in MuJoCo/npz/IK but xyzw inside sim2sim; `stand_flag=0.0` must be passed for standing (else a fast
  reference pelvis can flip the policy into walking mode); references need ≥ 21 frames past t (indices clamp silently); the IK MJCF
  has different waist geometry than the training URDF (do physics on the training-geometry URDF only); one zero-action hold step after
  reset (`policy.reset_target_for(ref)`).
- No reference clips or benchmark corpus ship in the repo; for M1 we generate our own (standing pose via FK, then IK reaches).

### Decisions (approved 2026-10-28)
- **hoi-sim env:** Python 3.11, HERO `[bench]` pins, `mujoco==3.13.0`, no torch. Lock + constraints files in `envs/`.
- **Feet:** capsule feet (`sonic_train`) to match HERO's training. Box sole (`sonic_box`) only as a fix if standing is unstable.
- **Fingers:** HERO's movable-finger model (`g1_29dof_with_hand_rev_1_0.urdf`) + our own finger PD loop + fingertip touch sensors we add.
- **Finger retargeting:** no dex-retargeting for now. Plan: fingertip IK with mink on the same MuJoCo Dex3 hand model.
  Fallback: dex-retargeting offline in `hoi-perception`, joint angles passed via the `.npz`.
- **Palm motion:** build HERO reference clips from OUR retargeted palm path (replace the smooth-path step of `BenchReplanner`,
  reuse its IK → FK → `ClipReference` code). HERO's point-to-point replanner = fallback and ablation baseline.
- **M5:** the learned policy (trained in `hoi-policy`) is exported to ONNX and runs inside `hoi-sim` alongside HERO.
- CLAUDE.md updated to match (approach steps 3–5, HERO section, env table, `envs/` in the layout).

### ROS 2 packages visible in hoi-perception (investigation only, nothing changed)
- `mujoco-ros2-control` / `-msgs` 0.1.1 are **not** installed in the env and **not** in `~/.local`: `pip show` reports
  `Location: /opt/ros/jazzy/lib/python3.12/site-packages`. They leak in through `PYTHONPATH`, which `~/.bashrc` sets by sourcing
  `/opt/ros/jazzy/setup.bash` (lines 118 and 144). Those six ROS/Gazebo dirs come **before** the env's own site-packages in `sys.path`
  of every conda env.
- User site: only `~/.local/lib/python3.12/site-packages` exists (225 entries) → not on the path of the 3.10 / 3.11 envs.
- Collision check of the 251 top-level ROS module names against what we import (numpy, mujoco, mink, onnxruntime, yaml, cv2, torch,
  sam2, open3d, ...) and our own package names: no clash except ROS's generic `controller` and `launch` (we must not name modules that).
  ROS's compiled modules are cp312 and won't load in 3.10/3.11; pure-Python ones would.
- Mitigation used for our runs: `env -u PYTHONPATH conda run -n hoi-sim ...`. (A permanent fix — e.g. not sourcing ROS in `~/.bashrc`
  or `conda env config vars set PYTHONPATH=` per env — is the user's call.)

---

## 2026-10-29 (Thu) — M1 sub-task 2: hoi-sim env, our MuJoCo scene, HERO standing + reaching

### hoi-sim created (as approved)
- `conda create -n hoi-sim python=3.11` + `pip install -e third_party/HERO[bench] mujoco==3.13.0 pytest imageio[ffmpeg]` (dry run first:
  exactly the planned 25 packages + hero-isaacsim). Install took ~25 min only because the network was slow (imageio-ffmpeg 29 MB).
- Checks (run with `env -u PYTHONPATH`): `pip check` → "No broken requirements found."; no torch / nvidia-* / onnxruntime-gpu;
  Python 3.11.18, numpy 1.26.4, mujoco 3.13.0, mink 1.3.0, onnxruntime 1.31.1 (providers Azure + CPU);
  `HeroExportPolicy` loads: inputs `actor_obs_lower_body`, `actor_obs_upper_body` [1, 1000] → `action` [1, 29], CPUExecutionProvider.
- `envs/hoi-sim.lock.txt` (pip freeze, 25 pins) and `envs/hoi-sim.constraints.txt` (numpy, scipy, mujoco, mink, onnxruntime).

### Scene: `sim/scene.py` (`TableScene`)
- HERO's `MujocoPlant` with `g1_29dof_with_hand_rev_1_0.urdf` (14 movable finger joints) and `foot_collision="sonic_train"` (7 capsules
  per foot). Body physics untouched (HERO's armature tables, Python PD on `qfrc_applied`, 1 kHz × 20). Finger PD copied from HERO's demo
  (kp 1.2, kd 0.04, effort 2.45/1.4 N·m, 2 rad/s target slew, armature 0.001, damping 0.01), fingers held open in M1.
- Table in HERO's benchmark layout (stance centre at origin, near edge x = 0.30 m, 0.60 m deep, 1.0 m wide, 3.5 cm top, four legs).
  Cup = free solid cylinder r 4 cm, h 10 cm, 150 g, condim 4; coaster = static disc r 5 cm. Cup contact = explicit pairs with every hand
  geom, the table top and the coaster, friction priors from HERO's "glazed ceramic mug" (hand 1.0625, table 0.35; coaster 0.5 = our guess),
  softness mixed like HERO's `_add_object_contact_pairs`. Hand geoms friction 1.5 (HERO's Dex3 value). Visual-only floor/sky/light.
- Touch sensors: one MuJoCo `touch` site per distal finger link (thumb_2, index_1, middle_1, both hands), box = the link's collision
  bounding box + 2 mm.
- Sanity checks: pelvis spawns at 0.757 m (HERO's STANDING_ROOT_Z 0.7567), total mass 34.54 kg, each hand 0.6965 kg (HERO's nominal),
  nq 57 / nv 55 / no MuJoCo actuators. Cup at rest: exactly 4 contacts with the table, all from the explicit pair (μ 0.35, solref 0.014,
  condim 4) → MuJoCo does not add a duplicate mask-based contact for an explicit pair (open question from the code read, now answered).
- Touch sensors work (scratch check, not a repo result): cup held as a fixed obstacle 2 mm into each right distal link → thumb_2 29.8 N,
  index_1 73.8 N, middle_1 85.3 N on the touched link (index/middle cross-talk when the cup touches both). First attempt with a free
  falling cup sampled only every 20 ms and missed the brief contacts — not a sensor problem.

### Reach test: `sim/m1_reach_test.py` → `results/m1_reach/`
Protocol (details in the script docstring): stand 10 s on a static reference; then 6 right-hand goals in HERO's benchmark ranges
(x 0.35/0.45, y −0.12/−0.22/−0.32, palm 10 cm above the table, canonical side grasp), 10 s each, one after the other. Initial reach =
`BenchReplanner.plan_from_state` with HERO's base plan for the height (`_bench_base`: squat at 0.50 m, stand at 0.74/0.88 m); correction =
HERO's published closed-loop protocol (replan at reach end then every 3 s, base held, goal adjust, 0.3 s blend). Odometry = exact sim pose
(HERO's published runs use a noisy odometry model — deviation). Error = right palm vs the original goal, world frame.

| table | fell | pelvis z min | pelvis tilt max | final err mean / max (cm) | hold-window mean err (cm) | hold-window max (cm) |
|---|---|---|---|---|---|---|
| 0.50 m | no | 0.591 m | 15.7° | 2.17 / 4.86 | 2.46 (mean over goals) | 5.76 |
| 0.74 m | no | 0.689 m | 12.9° | 1.43 / 1.93 | 1.70 | 6.22 |
| 0.88 m | no | 0.666 m | 10.2° | 1.40 / 2.16 | 1.89 | 6.30 |

- Standing 10 s (same at every height): pelvis z 0.735–0.752 m, root xy drift 7 mm, pelvis tilt ≤ 1.2°.
- 17/18 goals end within 2.2 cm. Worst: 0.50 m table, far-right corner (0.45, −0.32) — HERO's own reach prefilter flags it as not
  reachable at max lean ("squat_prefilter_failed"); final 4.86 cm, IK terminal error 0.95 cm, 4 replans.
- IK terminal error of the initial plans: 0.11–0.95 cm → the remaining error is HERO's tracking, not the planner.
- Hold-window max (4–6 cm) is the lag right at the end of the initial reach, before the first correction.
- No body part other than feet/hands touched floor or table; no hand–table contact; cup moved 0.05 cm (settling); root xy moves up to
  12 cm during reaches (HERO's planner lets the pelvis shift toward balance).
- Runtime: 64 s of sim per height in 8–13 s wall (incl. IK), CPU only.
- Files: `reach_results.json` (all numbers above + per-goal plans), `series_h*.npz` (per-step palm/pelvis/error), `reach_h074.mp4`
  (71 s, 25 fps; gitignored as a large video).

### Code review before the final run (workflow: 3 reviewers + 3 skeptics, read-only)
Fixed after review: (1) planners now get HERO's `arm_posture` (idle arm can't drift replan after replan); (2) first reach of a height
starts with HERO's lift waypoint — the rest palm (z 0.704) is below the 0.74/0.88 m tops and a straight-line path would sweep the fingers
through the slab; (3) cup pair softness mixed as in HERO; (4) initial-plan frame timing aligned with HERO's replan convention (was 1 frame
early); (5) "torso tilt" → it is pelvis tilt; odom deviation and goal order recorded in the JSON; per-goal pelvis min / tilt / hand contact.
Refuted by the skeptic: cup touching the idle left hand.

---

## 2026-10-29 (Thu) — M1 sub-task 3: reference builder + first scripted pick-and-place

### Decisions (approved 2026-10-29)
- Sim table height **0.74 m** from now on.
- ROS: always run our envs with `env -u PYTHONPATH`; our code never depends on ROS (CLAUDE.md "Environment rules").
- Capture app is **Stray Scanner** (Record3D is paywalled). Verified on the phone: each recording has `rgb.mp4`, `camera_matrix.csv`,
  `odometry.csv`, `imu.csv`, `depth/`, `confidence/`. Format details still to be confirmed against StrayRobots/scanner `docs/format.md`
  and the test clips; parser not written yet. CLAUDE.md updated (approach, env table, layout, data conventions, M2).

### Reference builder: `sim/reference_builder.py`
- `PathPlanner(BenchReplanner)`: our dense timed palm path (positions + wxyz orientations at 50 Hz) → HERO's whole-body mink IK (same
  feet / pelvis / CoM / posture / table-clearance tasks, base blended to HERO's benchmark base plan or held) → FK on the training-geometry
  model → `ClipReference`. Only the palm path differs from HERO's `build_plan` (which draws a straight smoothstep line to one goal).
- `min_jerk_path(p0, q0, [(pos, quat, seconds), ...])`: timed waypoints → dense path (minimum-jerk position + slerp, rest at each waypoint).
- HERO's closed-loop goal correction is kept: a settle step runs HERO's `BenchReplanner.step` (GoalAdjuster `g <- g - 0.6 e`, replans
  every 0.6 s, 0.3 s blend) until the palm error < 1.5 cm.

### Scripted pick-and-place: `sim/pick_place.py` → `results/m1_pick*/`
- **Grasp** = HERO's Dex3 side grasp (browser demo `computeGraspPreview`), written in the **cup frame** (origin at the cup centre, z up,
  world-aligned x since the cup is round): palm yaw −45° (HERO's default outward wrist yaw), palm = cup − R·(0.077, 0.055, 0), palm at
  6.5 cm above the table; approach direction = palm → cup. Moving the cup moves the grasp with it.
- **Phases** (fresh HERO reference per phase from the live state): stand 2 s → lift over the table, then pregrasp 8 cm before the grasp
  (base: HERO's benchmark base plan) → settle < 1.5 cm → slow approach 2 s → settle → close: ramp closure until thumb + index/middle
  touch (> 0.3 N), then finish to HERO's demo grip (full closure; finger kp ×3 and 0.6 × effort squeeze on blocked joints, ported from
  HERO's `simulation.js` — HERO's Python demo lacks it) → lift 10 cm → transport over the coaster (≤ 8 cm/s) → lower to 1 cm above it →
  settle → open 0.6 s → retreat 10 cm back / 5 cm up.
- **Success** = cup tilt < 10°, cup centre within the coaster radius (5 cm), resting on the coaster (|dz| < 4 mm), robot never fell.
  Failures stop the run and name the phase; categories: missed/knocked cup, slip during lift, dropped while moving, fell, plus
  "palm could not reach the target" (a settle ending > 3 cm away) and "released but not on the coaster".
- **Disclosed shortcuts** (scripted oracle baseline): exact simulator state for cup and robot (no perception, odometry = truth); the
  place target uses the cup's true in-hand offset measured after the lift; layouts constrain only the grasp and nominal place palm xy to
  the measured box (x 0.35–0.45, y −0.32…−0.12).

### What happened
1. First run (fixed layout): grasp, lift, transport fine, but the run was failed in "lower" by the in-hand check. Trace: the cup was
   still held (offset drift 1.2 cm); single 20 ms touch samples flicker to 0 N because contacts chatter at 1 ms. **Fix:** a fingertip counts
   as touching if it read > 0.3 N within the last 0.2 s. Not a grasp problem → no friction / force / cup-size tuning was needed.
2. Fixed layout (cup (0.473, −0.166), coaster (0.513, −0.306)): **success**, cup 0.73 cm off the coaster centre, upright.
3. **10 random layouts (seed 0) inside the measured workspace: 10/10 success** (95 % Wilson CI 72–100 %).
4. Because 10/10 is a weak estimate and gave no failure to show: **40 more (seed 1): 40/40** → 50/50 combined (95 % CI 93–100 %).
5. **Stress test outside the measured box** (palm box x 0.28–0.55, y −0.45…−0.02, seed 2): **12/20**. Failures: 5 palm could not reach
   the target (far right y ≈ −0.40…−0.46, or far x ≈ 0.59–0.64), 2 knocked the cup on the approach (cup close to the body x ≈ 0.38–0.40
   at y ≈ −0.34…−0.42), 1 dropped while lowering (coaster far forward x 0.62: stretched reach, cup slid 4.3 cm and tilted 15°).

Numbers (successful runs, main / extended sets): cup off the coaster centre mean 1.36 / 1.07 cm, max 2.23 / 2.80 cm; palm error at the end
of each settle ≈ 1.0 cm (max 1.66); contact detected at closure 0.56–0.66; touch at grasp ≈ 25–30 N on thumb and middle (index rarely
touches the 8 cm cup); cup rises 6.1–7.5 cm for a 10 cm palm lift (palm lag + slip); slip in the hand during the lift mean 0.70 / 0.86 cm,
max 1.20 / 1.74 cm; no falls; ~21 s of sim per attempt, ~5–8 s wall incl. IK and video.

### Code review (workflow: 2 reviewers + 2 skeptics, read-only) — fixed before the final numbers
- Real: a settle that never converged did not stop the run, so out-of-reach targets were reported as "knocked cup" / "not on the coaster"
  (stress set only; every settle in the 10 + 40 runs converged). Fix: `settle_or_fail` (> 3 cm → "reach" failure); stress set rerun.
- Also: lift slip now measured relative to the palm; per-plan IK `qp_failures` is now per plan (was cumulative; all 0 anyway); protocol
  JSON discloses close mode, privileged place target and what the box constrains. Refuted: success threshold / cup-frame math issues.
- All three result sets rerun after the fixes: identical success counts (10/10, 40/40, fixed OK); stress relabelled as above.
- The M1 reach test was rerun after the scene changes (grip boost, contact groups): identical numbers.

### Files
`results/m1_pick/pick_results.json` (fixed + 10), `results/m1_pick_extended/pick_results.json` (40), `results/m1_pick_stress/pick_results.json`
(20, outside the box). Videos (gitignored): success `results/m1_pick/videos/fixed.mp4`; failures `results/m1_pick_stress/videos/random_13.mp4`
(dropped while lowering), `random_05.mp4` (knocked cup), `random_12.mp4` (out of reach).

---

## 2026-10-30 (Fri) — M2 prep: perception install plan (research only, nothing installed)

Cloned (source only, gitignored): WiLoR `fcb911312a38fa8badd30d9656a167485d61b8f9`, sam2 `2b90b9f5ceec907a1c18123530e92e794ad901a4`,
stray-scanner `ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb`. Findings from reading their code/docs (a background research agent; to be verified on
real clips):
- **hoi-perception today:** Python 3.10.22, torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.2.6 — none of WiLoR's / SAM 2's other deps installed.
  `pip --dry-run` with a constraints file: 64 packages (~173 MB), no torch / numpy / nvidia change, nothing to compile. No nvcc on the machine.
- **WiLoR:** YOLO hand detector (53.6 MB) + ViT-H reconstruction checkpoint (**2.56 GB**). Needs `MANO_RIGHT.pkl` — manual registration at
  mano.is.tue.mpg.de (user step). Weights licence CC-BY-NC-ND, MANO licence non-commercial → never commit them (`*.pkl` added to .gitignore).
  Its 3D hand is not metric in depth: translation comes from a weak-perspective camera with a fixed fake focal length (5000/256·max(H,W)) →
  plan: keep the hand shape/articulation, re-solve translation with the real intrinsics against its 2D keypoints, check against LiDAR depth.
  torch 2.11 loading quirks: needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; ultralytics must not auto-install (`YOLO_AUTOINSTALL=false`).
- **SAM 2.1:** no conflicts; CUDA extension optional (skip). base_plus checkpoint 324 MB. Video predictor wants a folder of JPEG frames.
- **Stray Scanner (docs/format.md + app source):** depth = 16-bit PNG, 256×192, millimetres; confidence 0/1/2; `camera_matrix.csv` = K of the
  RGB resolution; `odometry.csv` = timestamp, frame, x y z, qx qy qz qw (scalar-last), camera-to-world. **The app already rotates the camera to
  OpenCV axes (x right, y down, z forward)**, world = ARKit (gravity y-up). This contradicts CLAUDE.md's "ARKit y-up −z-forward" — to be
  verified on a real clip before CLAUDE.md is changed. Possible off-by-one between rgb.mp4 frames and depth/odometry rows (check counts).

### Perception env set up (2026-10-30, approved)
- Installed into `hoi-perception` with `envs/hoi-perception.constraints.txt` (pins torch 2.11.0+cu128, torchvision 0.26.0+cu128, triton,
  numpy 2.2.6, opencv-python 4.12.0.88): WiLoR's inference deps (opencv, scikit-image, smplx 0.1.28, yacs, timm, pytorch-lightning, trimesh,
  ultralytics 8.1.34, ...), `pyrender` (--no-deps; it pins PyOpenGL 3.1.0, we have 3.1.10 — the only `pip check` complaint), `chumpy` from git
  `580566e` (--no-build-isolation), SAM 2 editable from `third_party/sam2` (`SAM2_BUILD_CUDA=0`), and `dill` (the YOLO hand-detector
  checkpoint is pickled with dill; not in WiLoR's requirements). After install: torch 2.11.0+cu128, CUDA available, numpy 2.2.6 — unchanged.
  Lock: `envs/hoi-perception.lock.txt`.
- Weights (gitignored, in third_party/): WiLoR `detector.pt` 53.6 MB, `wilor_final.ckpt` 2.56 GB (HF Space rolpotamias/WiLoR),
  SAM 2.1 `sam2.1_hiera_base_plus.pt` 324 MB.
- Runtime env vars: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch 2.11 defaults to weights_only loading, which these checkpoints fail),
  `YOLO_AUTOINSTALL=false`, `PYOPENGL_PLATFORM=egl`.
- Smoke tests: all imports OK. SAM 2.1 base_plus on the GPU (bf16) segments a synthetic box exactly (score 0.996), 0.43 s, 0.74 GB VRAM.
  WiLoR hand detector on 5 frames from each of the 4 test clips: a "right" hand in every frame where the hand is visible (conf 0.73-0.83),
  none in the start/end frames without a hand.
- Still missing: `MANO_RIGHT.pkl` (user registration at mano.is.tue.mpg.de) → the full WiLoR model cannot be loaded yet.
- Noted from the test frames: the cup is a mug with a handle (sim cup is a plain cylinder) and the table is a striped lap desk.

---

## 2026-10-30 (Fri) — M2 sub-task 1: Stray Scanner format check, parser, choice of camera view

4 test clips of the same right-hand mug → coaster move, one per camera view, in `data/raw/<id>/` (unzipped; gitignored).
Parser `perception/stray.py` (numpy + Pillow + ffmpeg/ffprobe binaries); checks `perception/check_views.py` → `results/m2_views/`
(`views.json`, `table.md`, per clip `overlay.jpg`, `cup_label.jpg`, `contact_sheet.jpg`, `overlay.mp4` (gitignored)). Run in hoi-sim:
`env -u PYTHONPATH conda run -n hoi-sim python -m perception.check_views` (18 s).

### Format (docs/format.md + app source `StrayScanner/Helpers/*.swift`, checked on all 4 clips)
- Files: `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `imu.csv`, `camera_matrix.csv`; no `distortion/` (distortion columns empty).
- RGB HEVC 1920×1440. Depth uint16 PNG in **mm**, 256×192 (same 4:3 field of view → K scaled by 256/1920). ARKit fills every
  depth pixel (0 % zeros); confidence 0/1/2 = 5–8 % / 18–30 % / 64–76 % of a frame.
- `odometry.csv`: `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x/y`; one row per depth PNG.
  `camera_matrix.csv` = the last row's K exactly. fx varies per frame by up to 2 % (focus: 1402–1437 px) → use per-frame K.
- IMU 100 Hz; acceleration in **g** (|a| median 0.99), not m/s² as the doc says.
- **Off-by-one confirmed:** every clip has N packets = N depth PNGs = N odometry rows, but ffmpeg decodes N−1 frames. Cause (app
  source): the first frame is stamped PTS −1/60 s and the writer session starts at 0, so the MP4 edit list hides it (ffprobe: the only
  negative-PTS packet is the first, flagged discard). → decoded frame k = depth/odometry index k+1. Checked on data: correlation of
  RGB motion vs depth motion (high-passed) peaks at our alignment in all 4 clips (0.40 / 0.15 / 0.42 / 0.29) vs the naive
  alignment (0.01 / 0.13 / −0.01 / 0.15); clip 49aff04f7a's peak is weak but on the same side.
- **Video clock is wrong:** the app stamps frame n at n/60 s whatever the capture rate. Clip 26d4118b0c was captured at 30 Hz (8.3 s of
  odometry) but its video says 4.15 s. 4e103843db has 19 gaps in its timestamps (14 app drops visible as video PTS gaps). Use
  odometry timestamps. Parser uses `-fps_mode passthrough` so ffmpeg never duplicates frames to fill gaps (513 frames, not 527).
- **Axes:** app writes `q_WA * q_AC` with q_AC = 180° about x → camera-to-world pose with **OpenCV camera axes** (x right, y down,
  z forward), ARKit world (y up). Verified on data: table normal fitted in the camera frame from depth vs world-up from the odometry
  pose: 1.8–5.7° for this convention; ARKit axes 102–174°, transposed pose 54–92°, wxyz quaternion order 144–176°. Independent check
  with the IMU (CoreMotion axes mapped to the back camera): 1.4–5.5°. CLAUDE.md data conventions updated.
- Phone was held in **portrait**, so all frames are sideways (world-up points to image −x); `stray.upright_rot90` gives the rotation.

### Which clip is which view (from the contact sheets + gravity)
| clip | view | optical axis from vertical | arm enters from |
|---|---|---|---|
| 26d4118b0c | top-down | 1° | bottom (your side) |
| 4e103843db | "90°": square to the table edge, from your side | 37° | bottom-right (your side) |
| 49aff04f7a | 45°, camera to your **left** (likely) | 31° | right edge |
| e8825dea53 | 45°, camera to your **right** (likely) | 39° | bottom-left |
Left/right is inferred from where your right arm enters the frame; confirm with your memory of the setup.

### Comparison (`results/m2_views/table.md`)
| | 26d4118b0c top | 49aff04f7a 45° L | 4e103843db 90° | e8825dea53 45° R |
|---|---|---|---|---|
| capture rate [Hz] / dropped frames | 30 / 0 | 60 / 0 | 60 / 19 | 60 / 0 |
| camera height above table / to cup [m] | 0.39 / 0.41 | 0.34 / 0.41 | 0.36 / 0.44 | 0.36 / 0.48 |
| table plane inliers (of conf-2 px) / RMS [mm] | 81 % / 4.9 | 82 % / 2.5 | 71 % / 3.6 | 80 % / 4.0 |
| cup: depth px in outline / conf-2 share | 3337 / 25 % | 3559 / 29 % | 3284 / 17 % | 2344 / 46 % |
| cup: conf-2 height above table p5/p50/p95 [cm] | 1.1/1.4/2.0 | 1.8/6.2/8.5 | 1.2/6.1/8.6 | 1.6/6.7/8.8 |
| cup centre jitter over static frames [mm] | 2.8 (5 fr) | 2.0 (105 fr) | 2.8 (89 fr) | 1.8 (26 fr) |
| odometry max drift [mm / deg] | 13 / 2.2 | 8 / 1.3 | 5 / 2.5 | 18 / 2.7 |
| image shift, first vs last frame [px] | 19 | 22 | **8** | 46 |
| depth changed > 2 cm in last frame | 9 % | 15 % | 14 % | **46 %** |
| hand + forearm in frame (contact sheet, by eye) | yes, throughout | forearm outside the right edge during approach/grasp | **yes, throughout** | hand cut by the frame corner during grasp/lift |

Definitions: table plane = RANSAC (1 cm) + least-squares refit on the median depth of the frames before the hand enters (hand entry =
first frame where > 0.5 % of pixels change depth by > 2 cm). Cup = depth pixels inside a cup outline hand-labelled on frame 1 (mug
body, no handle; drawn back in `cup_label.jpg`). Cup centre = median of conf-2 cup points > 1 cm above the table, projected onto it.
"Image shift" = phase correlation of the first and last frame (no hand in either); the cup has moved by then, so ~10–15 % changed depth
is expected without camera motion. Hand/forearm visibility is a judgement by eye on 24 frames per clip (no hand detector run here).

### Reading the numbers
- Every view gives usable metric depth: table fit 2.5–4.9 mm RMS, static cup centre jitter ≤ 2.8 mm.
- **Top view:** depth on the mug is its *inside floor* (heights 1–2 cm), never the wall or rim → cup position only from the 2D mask;
  the lift happens along the optical axis; it was also recorded at 30 Hz. Hand always in frame.
- **45° views:** the hand / forearm leave the frame (narrow portrait field of view at ~0.4 m). e8825dea53's camera also moved (46 px,
  46 % of the depth changed) — the phone stand was probably bumped or is on the soft bed.
- **90° view (4e103843db):** the most static camera (8 px), hand + forearm visible for the whole reach → grasp → transport → release,
  cup wall seen by depth (p95 8.6 cm). Weak spot: only 17 % of cup pixels are high confidence (76 % medium) — dark glossy mug.
- Depth registration to RGB looks right in the overlays (hand silhouette edges match).

### Recommendation (for the 15–20 real demos)
**Use the 4e103843db setup: "90°", camera on your side, square to the table edge, ~35–40° from vertical.** Plus, if possible:
- Move the phone back (now 0.44 m to the cup; plan said 0.6–0.8 m) and/or record in **landscape** so cup/coaster anywhere on the desk
  plus the forearm stay in frame. Landscape orientation needs one short test clip (frames should come out upright).
- Mount the phone rigidly, not on the bed (camera motion in 3 of 4 clips: 19–46 px).
- Keep the app at 60 Hz (26d4118b0c was 30 Hz).
- Matte cup would raise the cup's high-confidence share; the glossy dark-green mug is usable (heights look right) but low confidence.

### Incident: GPU unusable after my first analysis run (my fault)
- First version of `check_views.py` called `np.linalg.svd` with the default `full_matrices=True` on ~34 k points → a 34k×34k matrix
  (~9.5 GB) per plane fit, 4 clips in parallel → host RAM exhausted at 17:59–18:00 (kernel OOM killer killed one of my workers).
- At the same moment the NVIDIA driver logged `NV_ERR_NO_MEMORY` and then `GSP unload failed at suspend` / `can't suspend`; the GPU's
  runtime PM status is now `error` and `nvidia-smi` fails ("Unable to determine the device handle for GPU0: Unknown Error").
  Very likely caused by the RAM exhaustion. Fix: `full_matrices=False` (run now uses < 1 GB per clip, 18 s). **A reboot is needed**
  to get the GPU back; nothing was changed on the driver side.
