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

### Setup 2026-10-30 — GPU check, MANO install (chumpy-free), object config
1. **GPU** (after reboot + `prime-select nvidia`): `prime-select query` = nvidia; driver 580.178.04 (CUDA 13.0), RTX 5060 Laptop GPU idle
   (P8, 13 MiB / 8151 MiB). hoi-perception: torch 2.11.0+cu128, `cuda.is_available()` True, 2048² matmul OK. PASS.
2. **MANO:** `mano_v1_2.zip` (175 MB) was inside the repo by mistake → moved to `~/Downloads` (never git-added; MANO licence forbids
   redistribution), extracted to /tmp. `third_party/WiLoR/mano_data/` held only `mano_mean_params.npz` (untouched). Copied
   `MANO_RIGHT.pkl` + an untouched copy `MANO_RIGHT_orig.pkl`. PASS.
3. **chumpy-free MANO:** WiLoR mentions chumpy only in `requirements.txt` (no code import); none of our code uses it.
   - Reference: loaded `MANO_RIGHT_orig.pkl` WITH chumpy 0.71 → `/tmp/mano_ref.npz`. Only `shapedirs` is a chumpy object.
   - `perception/tools/strip_chumpy.py` unpickles chumpy classes as stubs (chumpy blocked via `sys.modules["chumpy"] = None`).
   - **First attempt failed:** it took the only top-level numeric array of the `shapedirs` stub, which was `idxs` (23340,) int64 —
     `shapedirs` is a `chumpy.reordering.Select` view, not a plain array. Mismatch → stopped, `MANO_RIGHT.pkl` restored from the original.
   - **Fix (approved):** handle exactly two classes — plain `Ch` (value = `x`; error if any other numeric state) and `Select`
     (`resolve(a).ravel()[idxs].reshape(preferred_shape)`); anything else raises naming the class. Asserts before writing: shapedirs
     (778,3,10) float64, all expected shapes, integer arrays only in `f` / `kintree_table`.
   - **Result: all 13 keys equal the chumpy reference exactly** (`np.array_equal`, dtype included). Shapes OK: v_template (778,3),
     shapedirs (778,3,10), posedirs (778,3,135), J_regressor (16,778) sparse, weights (778,16), f (1538,3), hands_components (45,45),
     hands_mean (45,), kintree_table (2,16). Info: `a.x` is (778,3,20) and `idxs` selects exactly the first 10 of the 20 shape components.
   - **chumpy removed:** it was installed on 2026-10-29 (git `580566e`) during WiLoR setup; uninstalled today because it is unmaintained
     and judges should not need it. Lock regenerated (`pip freeze --exclude-editable`): the only diff is the removed chumpy line.
     torch 2.11.0+cu128, numpy 2.2.6 unchanged.
4. **smplx load without chumpy:** `smplx.MANO(model_path='third_party/WiLoR/mano_data', is_rhand=True, use_pca=False)` → vertices
   (1,778,3), joints (1,16,3), all finite. PASS (folder path works).
5. **Weights inventory** (size bytes, sha256):
   - detector.pt 53582271 `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033`
   - wilor_final.ckpt 2564989533 `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2`
   - sam2.1_hiera_base_plus.pt 323606802 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`
   - MANO_RIGHT.pkl (chumpy-free) 3447680 `ed3ac9bfe93c48319616e2d38010afc802405140b23079320680174652286830`
   - MANO_RIGHT_orig.pkl (official) 3821356 `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`
6. **Git hygiene:** added `*.zip` and `data/work/` to .gitignore; `git check-ignore -v` confirms third_party/, mano_data/*.pkl, *.ckpt,
   *.pt, *.zip, data/work/, data/raw/ (except the allowed `data/raw/sample/`), results/**/*.mp4. `git status` shows no weight/MANO/zip file.
7. **`config/objects.yaml`** created (mug from its base label: Ø 9.9 cm, H 9.1 cm, W with handle 14.3 cm; handle protrusion derived 4.4 cm;
   handle size unknown; mass 0.37 kg estimated; coaster Ø 10.4 cm / 5 mm estimated).
8. **Disk / RAM:** /home 410 G, 37 G free (≥ 30 G required, but only just; 91 % used). RAM 30 Gi, 24 Gi available; swap 14 Gi.
9. Cleaned /tmp/mano_unzip and /tmp/mano_ref.npz; zip kept in ~/Downloads.

---

## 2026-10-30 (Fri) — M2: perception on the 4 test clips (Track A) + mug model and oracle re-run (Track B)

### Phase 0 — shared updates and preflight
- **CLAUDE.md** updated (the M2 prompt wins over the old plan): new pitch ("From 4 iPhone LiDAR clips of my own hand ... a mug ...
  anywhere in its reachable workspace"); new "Recorded data" section (the 4 existing clips = the data set, one viewpoint each, right hand,
  near long edge, glossy dark-green mug with handle, resin coaster, lap desk 60 × 38.5 cm on a bed, phone propped by hand; secondary
  question = viewpoint effect, confounded with execution, n = 4); removed "Matte cup", "15–20 demos", "60–80 cm at 45°";
  `config/objects.yaml` declared the single source of object + desk geometry; old `.npz` schema replaced by schema v2 + table-frame definition.
- **config/objects.yaml:** `handle_height_m` 0.06 and `handle_thickness_m` 0.012 (both `estimated`); appended the `desk` block
  (0.60 × 0.385 m, `measured`).
- **Path adaptation:** the prompt says `data/raw/test/<id>`; the clips are in `data/raw/<id>/` (as in M2.1). All 4 IDs present → used as is.
- **hoi-sim:** `pip install pyyaml==6.0.3 -c envs/hoi-sim.constraints.txt` (needed by Track B to read `config/objects.yaml`); `pip check`
  clean; lock regenerated — the only diff is the added `PyYAML==6.0.3` line. Done here (shared preflight) so Track B stays inside `sim/`.
- **Inventory** (`perception/inventory.py` → `results/m2_perception/inventory.json`):

  | clip | view | odometry rows / depth / confidence | decoded video frames | duration | median dt | max dt | dt > 1.5 × median |
  |---|---|---|---|---|---|---|---|
  | 26d4118b0c | top-down | 250 / 250 / 250 | 249 | 8.300 s | 33.3 ms | 33.3 ms | 0 |
  | 4e103843db | 90° | 514 / 514 / 514 | 513 | 8.870 s | 16.7 ms | 33.4 ms | 19 |
  | 49aff04f7a | 45° | 479 / 479 / 479 | 478 | 7.970 s | 16.7 ms | 16.7 ms | 0 |
  | e8825dea53 | 45° | 474 / 474 / 474 | 473 | 7.887 s | 16.7 ms | 16.7 ms | 0 |

- **WiLoR loader** (`perception/wilor_loader.py`). Third-party facts were read from the code by a research workflow (4 readers +
  4 adversarial verifiers, read-only). Every relative path WiLoR uses at load time and how it is overridden: config yaml / checkpoint /
  detector → absolute paths; `get_config(update_cachedir=True)` would prefix MANO paths with `./pretrained_models/` → `update_cachedir=False`;
  yacs does not expand `${MANO.DATA_DIR}` → MANO `DATA_DIR` / `MODEL_PATH` / `MEAN_PARAMS` set to the absolute `mano_data/`;
  `MODEL.BACKBONE.PRETRAINED_WEIGHTS` (a training-time file) popped, as `load_wilor` does; `MODEL.BBOX_SHAPE = [192, 256]` as `load_wilor`.
  `WiLoR(cfg, init_renderer=False)` (the pyrender renderer is only for training logs), state dict loaded with `strict=False` like
  `load_wilor` (490/490 keys match, asserted), then `.eval()`. `sys.path` is extended inside Python; no chdir.
  - **weights_only:** `wilor_final.ckpt` loads with torch 2.11's default `weights_only=True` (passed explicitly). The YOLO detector does
    **not** (checked: `UnpicklingError: Weights only load failed`): ultralytics calls `torch.load(file)` itself and the file is a
    dill-pickled ultralytics model, so `weights_only` cannot be passed through. `load_detector` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
    (torch only applies it to calls that do not pass `weights_only`) — i.e. the detector is loaded with `weights_only=False`, a trusted
    local file. Also: `YOLO_AUTOINSTALL=false`, ultralytics analytics events disabled.
  - Verified facts used later: detector `conf=0.3` (WiLoR demo), classes `{0: left, 1: right}`, BGR input, imgsz 512 from the checkpoint;
    crop `rescale_factor=2.0`; outputs are rotation matrices (global_orient (1,3,3), hand_pose (15,3,3), betas (10)); joints / vertices
    have no translation (add the camera translation); 2D keypoints are normalised crop coordinates (pixels = box_centre + kp2d · box_size);
    WiLoR's `cam_crop_to_full` assumes f = 5000/256 · max(W, H) = 37 500 px for our frames and the image centre as principal point.
- **Smoke test** (`perception/wilor_smoke.py` → `results/m2_perception/smoke/`): 3 upright frames of 4e103843db (i = 150, 250, 350):
  one right-hand detection each (score 0.733 / 0.748 / 0.790); detector + WiLoR 0.559 s (incl. CUDA warm-up) / 0.156 / 0.144 s per frame;
  model load 4.5 s; **peak VRAM 2.82 GB, peak RSS 6.06 GB**. Overlays (box, WiLoR 2D keypoints, mesh with WiLoR's own demo camera) sit
  on the hand. PASS.

### Track A — S1–S3 (frames, hand, masks)
- **Shared code:** `perception/common.py` (paths, clip list + view labels, `load_config`, per-stage runner `stage_cli`: cache in
  `data/work/<clip>/<stage>/`, skip unless `--force`, `done.json` with seconds / peak RSS / peak VRAM, a watchdog that stops a stage
  above 16 GB RSS; each stage runs as its own process so the peaks are per stage and the GPU is freed between stages).
  `perception/geometry.py` (upright ↔ original maps, intrinsics scaling, projection, transforms, rotations).
  `tests/test_geometry.py`: for k = 0..3, 1000 random points projected + pixel-mapped vs rotated + projected with `upright_K` agree to
  2.3e-13 px; pixel round trips exact; the pixel map matches what `np.rot90` does to an image. PASS.
- **Adaptation (pixel centres):** `K_depth` uses the existing `stray.Recording.K_depth` (f·s, (c + 0.5)·s − 0.5), i.e. pixel-centre
  consistent scaling by 256/1920; plain c·s would shift the depth grid by 0.43 depth px (~0.7 mm at 0.4 m).
- **S1 frames** (`perception/frames.py`, all 4 clips, 6–12 s each): upright rotation k = 3 for every clip (world-up at i = 1);
  upright full-res 1440×1920 JPEG q95 and half-res 720×960 JPEG q90 (2×2 box average); dropped frames (dt > 1.5 × median) logged, not
  resampled: 4e103843db i = 5, 11, 14, 32, 94, 106, 110, 119, 128, 138, 151, 163, 174, 186, 199, 207, 238, 266, 273; none elsewhere.
- **S2 hand** (`perception/hand_wilor.py`): detector conf 0.3 (WiLoR demo), right = class 1, highest score with the 0.1 IoU tie-break,
  WiLoR on the box (demo crop), `cam_crop_to_full_K` with the real upright K (derivation in the docstring). Check on every frame: with
  WiLoR's assumed camera (f = 37 500 px, image centre) our function reproduces WiLoR's `cam_crop_to_full` to 1.5e-13 (relative).
  The phone reports fx = fy on every frame (upright fx = fy = 1406.9 px median, max |fx − fy|/fy = 0). Keypoint order verified in
  `mano_wrapper.py` (`mano_to_openpose`): OpenPose hand order, wrist 0, thumb 1–4, index 5–8, middle 9–12, ring 13–16, pinky 17–20.
  WiLoR's crop camera is near-orthographic (f = 5000 px for a 256 px crop), so kp3d + cam_t at the real focal length will not reproject
  exactly onto WiLoR's 2D keypoints; S5's plumbing check measures this.
- **S3 prompts** (`perception/prompts/<clip>.json`, `--grid` / `--preview` tools in `perception/masks_sam2.py`): HSV green rule
  (H 35–90, S ≥ 60, V ≥ 25 in OpenCV units; largest blob → box + deepest interior point). The rule finds the mug, but its box covers
  the green wall + handle and misses the white rim/interior that belongs to the body silhouette → the body box is manual in every clip;
  the rule's interior point is used as the body positive (on the green wall) in 3 clips. **Top-down 26d4118b0c:** the largest green
  blob is the HANDLE (the body shows its white interior), so the rule's box/point became the handle prompt (auto) and the body prompt
  is manual. **e8825dea53: the handle is not visible at i = 1** (hidden behind the mug; checked on a brightened zoom) → no handle prompt,
  its masks stay empty, and its body prompt has no negative handle point. Every first-frame mask was checked by eye in
  `prompt_preview.jpg` before propagating (all correct on the first try: body = white rim + green wall without the handle, handle loop,
  full coaster disc).
- **S3 SAM 2** on 4e103843db: 513 frames in 125 s, peak VRAM 0.80 GB, peak RSS 9.35 GB (the 1024² float32 video lives in host RAM with
  `offload_video_to_cpu`). Coaster mask empty in 2 frames (covered by hand + mug at t ≈ 4.7 s); body and handle never empty.
- `third_party/WiLoR`: importing `wilor` had rewritten 4 `.pyc` files that the upstream repo tracks → restored with `git checkout`,
  generated `__pycache__` files removed, and `wilor_loader` now sets `sys.dont_write_bytecode` (checkout clean at fcb9113 again).
- **S2 + S3 on all 4 clips** (one GPU process at a time; numbers from each stage's `data/work/<clip>/<stage>/done.json`):

  | clip | hand_valid frames / N | first / last hand_valid | frames with > 1 right box / IoU tie-breaks | WiLoR s/frame (median) | S2 peak VRAM / RSS | SAM 2 time / peak VRAM / RSS | empty mask frames body / handle / coaster |
  |---|---|---|---|---|---|---|---|
  | 26d4118b0c (top) | 161 / 250 | 37 / 200 | 0 / 0 | 0.31 | 2.82 / 6.06 GB | 70 s / 0.80 / 5.42 GB | 0 / 0 / 6 |
  | 4e103843db (90°) | 366 / 514 | 90 / 455 | 2 / 0 | 0.16 | 2.82 / 6.06 GB | 125 s / 0.80 / 9.35 GB | 0 / 0 / 2 |
  | 49aff04f7a (45°) | 262 / 479 | 106 / 367 | 4 / 3 | 0.22 | 2.82 / 6.06 GB | 102 s / 0.80 / 8.83 GB | 0 / 0 / 5 |
  | e8825dea53 (45°) | 318 / 474 | 39 / 356 | 0 / 0 | 0.16 | 2.82 / 6.06 GB | 87 s / 0.80 / 8.38 GB | 0 / 473 (no prompt) / 242 |

  `cam_crop_to_full_K` = WiLoR's formula to ≤ 1.8e-13 relative in every clip; fx = fy on every frame of every clip.
  e8825dea53: the coaster mask is lost at t ≈ 3.8 s while the hand carries the mug over it and stays empty — at the end the mug stands
  on the coaster and hides almost all of it from this 45° view (checked on frame i = 470). Only the initial-window coaster mask is used
  downstream (S4 exclusion, S6 circle fit), so this does not feed any gate. Its handle (no prompt) appears at the back of the mug later.

### Track B — mug model and oracle re-run (CPU, hoi-sim)
Workflow: 1 implementer → 3 adversarial reviewers (geometry/physics, protocol, statistics) → fixer → runner → independent checker that
re-derived every number from the runs files (all match). Then reviewed by me: removed an unrequested McNemar p-value (it was also stored
rounded to 0.0), saved the grasp statistics in `summary.json` (so every number below comes from a results file), and made the
all-fail video pair show both failure modes.

**Mug model** (`sim/scene.py`, `cup_model` = "mug" default | "cylinder"; all values read from `config/objects.yaml` via `mug_config()`):
- **Body:** solid cylinder r 0.0495 m, h 0.091 m (label values); its centre is the cup body's frame origin, so "cup centre" (cup_pose,
  grasp offset, success test) keeps meaning the body centre, exactly as for the M1 cylinder. The oracle only touches the outside.
- **Handle:** D-loop of 3 capsules, radius 6 mm (= thickness / 2), on the body's +x side, centred at mid-height. The 0.044 m protrusion
  is measured to the handle's OUTER surface (overall width 0.143 m = label) and the 0.06 m span is its OUTER vertical extent; the two
  arms start one capsule radius inside the wall so the loop is attached. Handle size is `estimated` in the config.
- **Mass / inertia:** one density on all 4 geoms (mass / summed geom volume), so MuJoCo derives mass, COM and inertia from the shapes:
  0.37 kg, COM 1.84 mm toward the handle, inertia diagonal (4.74, 5.25, 4.93)·10⁻⁴ kg·m² (`protocol.mass_properties` in the runs files).
  Overlap volume (arms in the wall, corners) is counted twice — under 1 %. Caveat: a real mug is a shell with a heavier handle share.
- **Friction "identical to the current cup":** every mug geom has the cup's geom settings (condim 4, solref 0.008 s, same friction
  tuple) and the same explicit pairs with every hand geom, the table top and the coaster (HERO's glazed-ceramic priors, solmix-mixed).
- **Model switch:** "cylinder" builds the M1 cup byte-identically (same geom name, params and pair names). `m1_reach_test` passes
  "cylinder"; `pick_place` has `--cup-model` (default mug) and refuses to overwrite a results file of the other model.
- **Cup height plumbing:** where the oracle used the cylinder's constant height (grasp z offset, place target, resting check) it now uses
  `scene.cup_height`, so "palm 6.5 cm above the table", "1 cm place clearance" and "|dz| < 4 mm" keep their meaning. Grasp, phases,
  touch logic and success rule are unchanged.
- **Cylinder reproducibility:** `python -m sim.pick_place --cup-model cylinder --trials 0 --out results/m1_pick_mug/cylinder_repro` on the
  final code equals `results/m1_pick` "fixed" in every field except wall time (success, cup_final (0.5165, −0.299, 0.796), 0.73 cm off
  centre, contact closure 0.56, lift rise 6.52 cm, slip 0.69 cm, 20.12 s sim, every plan / settle).

**Protocol** (`sim/m1_pick_mug.py`): M1's 50 in-workspace layouts in M1's order — `sample_layouts(grasp, 10, seed=0)` +
`sample_layouts(grasp, 40, seed=1)`, default palm box — so trial k is paired with the cylinder's run k (asserted). Handle yaw =
approach direction (`Grasp.approach_dir`, palm → cup, −9.46°) + U(−45°, +45°) from `np.random.default_rng(2026)`, drawn once; trial k has
the same yaw at every mass (asserted). Mass 0.37 kg (config, estimated) on 50 trials; sweep: trials 0–19 at 0.25 and 0.45 kg, 0.37 kg
column = trials 0–19 of the main run (not rerun). Wilson 95 % CI, z = 1.96.

**Results** (`results/m1_pick_mug/summary.json`, `table.md`):

| run | success | Wilson 95 % CI | failures by category |
|---|---|---|---|
| mug 0.37 kg, 50 layouts | **0/50** | 0–7 % | slip during lift 28, dropped while moving 21, missed/knocked cup 1 |
| cylinder 0.15 kg (M1), same 50 layouts | 50/50 | 93–100 % | – |
| mug 0.25 kg, trials 0–19 | 0/20 | 0–16 % | slip during lift 11, dropped while moving 9 |
| mug 0.37 kg, trials 0–19 (= main run) | 0/20 | 0–16 % | slip during lift 12, dropped while moving 8 |
| mug 0.45 kg, trials 0–19 | 0/20 | 0–16 % | slip during lift 14, dropped while moving 6 |

Paired with the cylinder: all 50 layouts flipped (cylinder success → mug failure). No fall, no unreachable target, every touch check at
closing passed. Grasp statistics, mean (min–max) over the trials that reached the measurement (`summary.json` → `grasp_stats`):

| | mug 0.37 kg (n = 49) | cylinder 0.15 kg (n = 50) | mug 0.25 / 0.37 / 0.45 kg, trials 0–19 |
|---|---|---|---|
| lift slip in the hand | 2.44 cm (0.91–7.73) | 0.83 cm (0.28–1.74) | 2.12 / 2.34 / 2.34 cm |
| cup rise for a 10 cm palm lift | 5.09 cm (1.43–6.83) | 7.38 cm (6.08–9.76) | 5.65 / 5.01 / 4.77 cm |
| closure fraction at first thumb + finger touch | 0.51 (0.46–0.58) | 0.57 (0.56–0.66) | 0.52 / 0.52 / 0.52 |

- The hand closes on the mug (49/50) but the mug does not stay put: lift failures exceed the existing slip (> 2 cm) or rise (< 5 cm)
  limit; the "dropped while moving" trials passed the lift and then drifted > 3 cm in the hand during transport / lowering.
- Mass in 0.25–0.45 kg does not change the outcome (0/20 each). The lower closure fraction is expected geometrically (the 9.9 cm body is
  touched earlier in the closing ramp), not a sign of a weaker grasp by itself.
- **Not separated:** the mug differs from the cylinder in mass (0.37 vs 0.15 kg), radius (4.95 vs 4.00 cm at the same palm-frame offset →
  surface ~1 cm closer to the palm), height (palm 1.95 vs 1.50 cm above the centre) and the handle. No run isolates these (outside this
  task; would need approval). The unchanged M1 oracle grasp is not a working baseline for the real mug.
- Trial 49 failed in the approach: the hand pushed the mug 2.7 cm before grasping (M1's 2 cm limit).

**Files:** `runs_mass0.37.json` (trials 0–49), `runs_mass0.25.json`, `runs_mass0.45.json` (0–19) — protocol + one row per trial (layout
seed/index, cup/coaster xy, handle yaw, mass, outcome, phase, reason, category, numbers, wall time, detail); `summary.json`, `table.md`;
`videos.json` + `videos/failure_trial00.mp4` (slip during lift) and `videos/failure_trial05.mp4` (dropped while moving), gitignored,
both re-runs identical to the batch (nothing succeeded, so two failures of different categories); `cylinder_repro/pick_results.json`.
Runtime: three parallel CPU processes; summed per-trial wall time 156 s (50 trials), 73 s and 70 s (20 each); videos via Mesa EGL on the
Intel iGPU (`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl`) while Track A held the NVIDIA GPU.

### Track A — S4–S11, gates, cross-view (all numbers: `results/m2_perception/summary.md`, `<clip>/gates.json`, `<clip>/stages/*.json`)
**How it was built.** An interface contract (file formats between stages) plus a list of implementation decisions fixed BEFORE any result
(RANSAC seeds/iterations, dilation kernels, majority = ≥ 6 of 11 frames, grid + Nelder–Mead, event definitions, gap-fill rule, G1/G4
definitions, …). Five module groups were implemented in parallel by a workflow (implement → 2 adversarial reviewers: spec compliance,
numerics/frames → fixer), each with a synthetic self-test against known ground truth (scratch, not in the repo). `perception/run_clip.py`
runs S1–S11 as one process per stage. Adaptation: S7 lives in its own module `perception/table_frame.py` (one module = one stage), and its
JSON is `data/work/<clip>/table_frame/table_frame.json` (the spec wrote `data/work/<clip>/table_frame.json`).

**Corrections I made after review, before the final runs (none chosen to pass a gate):**
- S6 silhouette: the contract named `cv2.fillConvexPoly`, which fills every pixel the hull edge touches and rounds the vertices (~0.65 px
  too large per side). Replaced by a pixel-CENTRE-inside test on the unrounded hull — the same rule that defines SAM 2's masks (logits at
  pixel centres). Check: a top-down disc covers 0.9987 × πR² px = the 72-gon/circle area ratio. This change moves G6's free-fit diameter
  UP (on the 90° clip the old rasteriser narrowly passed G6 in a reviewer probe; with the fix every clip fails G6).
- S9 Savitzky–Golay window: my contract text said 13 / 7 frames, an arithmetic slip; the spec's "odd window nearest 0.2 s" on the real
  clock (median dt 16.67 ms / 33.33 ms) is 11 frames (0.183 s) / 5 frames (0.167 s), which the code computes.
- `mano_global_orient` stored in the table frame (the spec's rule: table frame unless the name says cam).
- `report.py` copies every stage's `done.json` and `<stage>.json` into `results/m2_perception/<clip>/stages/` so all numbers are under results/.
- Cosmetic: constrained layout for events.png (clipped label), handle-direction legend / "invalid" note in topdown.png; two docstring fixes
  from the schema audit (depth_shift_m / lidar_residual_m are finite wherever S5 computed D; capacity_ml has no source flag).

**Verification.** A second workflow re-derived every gate value, pass/fail and cross-view value per clip with independent code (not
importing report.py) and checked the overlays by eye; a schema auditor checked every demo field against S10 and cross-stage consistency
(76/76 checks on three clips, 72/72 on the top-down). **No numerical or pass/fail mismatch on any clip.**

**Gates** (✔ pass, ✘ fail):

| gate | 26d4118b0c top-down | 4e103843db 90° | 49aff04f7a 45° | e8825dea53 45° |
|---|---|---|---|---|
| G1 hand coverage | ✘ events undefined | ✔ 100 %, no gap | ✔ 99.2 %, longest gap 1 frame | ✔ 99.7 %, no gap |
| G2 LiDAR residual | ✔ 0.53 cm | ✔ 0.55 cm | ✔ 0.38 cm | ✔ 0.82 cm |
| G3 plumbing reprojection | ✘ 41.8 px | ✘ 61.2 px | ✘ 81.0 px | ✘ 30.9 px |
| G4 static jitter | ✔ raw std 0.06 cm (1 hold) | ✔ 0.05 cm (1 hold) | N/A no hold | N/A no hold |
| G5 grasp geometry | ✘ g undefined | ✔ d 99.4 mm (≤ 99.5), z 47.5 mm | ✔ d 96.1 mm, z 61.8 mm | ✔ d 96.6 mm, z 62.7 mm |
| G6 mug fit | ✘ IoU 0.918, Ø 105.1 mm, h N/A | ✘ IoU 0.943, Ø 105.7, h 86.9 | ✘ IoU 0.947, Ø 106.1, h 85.8 | ✘ IoU 0.938, Ø 105.8, h 88.1 |
| G7 placement | ✔ 0.81 cm | ✔ 2.15 cm | ✔ 1.04 cm | ✔ 0.54 cm |
| G8 table frame | ✔ RMS 5.45 mm, drift ok, x 0.8° | ✘ normal drift 2.43° (x 1.7°) | ✘ x 23.9° off 45° | ✔ x 11.0° off 45° (but see below) |

**What failed and why (plainly):**
- **G3 fails on every clip (31–81 px).** WiLoR's 3D hand comes from a near-orthographic crop camera (f = 5000 px for a 256 px crop); its 2D
  keypoints agree with a weak-perspective projection of its 3D keypoints to about a pixel (reviewer + verifier probes in the scratchpad, not a
  saved result). The spec's check projects with the real perspective camera, where a hand ~30–40 cm away and a few cm deep lands tens of
  pixels off. Our transforms are consistent; the gate as defined cannot reach 8 px for close hands. Needs the user's decision.
- **G6 fails on every clip the same way:** free-fit diameter 105.1–106.1 mm vs the 99 mm label (+6–7 mm), height 85.8–88.1 mm vs 91 where
  observable. The SAM 2 body masks are systematically larger than the label-size silhouette (verifier: on the top-down final window the mask
  extends up to 16 half-res px past it on one side). Cause not identified (SAM 2's 4× lower-resolution logits, the mug's real shape vs its
  label, or both). The 2-DOF IoU is 0.92–0.95 everywhere.
- **Top-down 26d4118b0c has no events:** the palm point's closest approach to the cup axis is 0.102 m vs the contact threshold r + 5 cm =
  0.0995 m (`stages/events.json`), so contact never fires and lift / grasp / place / release are undefined → G1 and G5 fail, 7 cross-view
  columns are n/a. On the other clips the palm-to-axis distance at g is 96–99 mm, i.e. this grasp style sits at the threshold.
- **G8 drift on 4e103843db:** plane normals of the initial and end windows differ by 2.43° (> 2°) while the offset at the mug is −0.4 mm.
  The verifiers traced it to ARKit attitude drift on a static camera (the desk slot stays on the same pixels; scratch probes, not saved).
- **S7 picked junk "long edges" on both 45° clips.** The rule "any line with inlier extent > 0.42 m is a long edge" is fooled by lines whose
  inliers are two distant fragments (extent = max − min includes gaps): 49aff04f7a L2 = 0.584 m with a 0.444 m gap (cup holder + desk
  corner) beat the real edge L0 (0.50 m, gap 0.02 m) under the pre-registered tie-break (group of the longest qualifying line) → x 23.9° off;
  e8825dea53's only qualifying line L2 = 0.54 m with a 0.26 m gap runs through the mug's outline and the cup holder → its G8 pass (11°) is
  coincidental. Consequence: on these two clips the table-frame x/y quantities (coaster dx/dy, yaws, palm path in x/y) are rotated; distances,
  heights and events are not affected. The S7 hole filling did not remove the mug outline because the mug region connects to the off-desk
  region through the desk's slot / image edge. Not changed: needs the user's decision (gap-aware extent, exclude pixels near object masks, …).
- **G4 N/A** on both 45° clips (longest slow span shorter than 0.3 s).
- **e8825dea53: no handle** at i = 1 (hidden) → handle yaw invalid, approach yaw n/a, its cup frame falls back to (junk) table x. Its
  lift confirmation is uninformative (the hand occludes the mug before contact, IoU < 0.7 already).
- **Palm below the table at the first / last frames of the hand track** (e.g. −8.4 cm at i = 90 on 4e103843db, with large LiDAR depth shifts):
  edge-of-track artefacts, outside every gate window, no event moved (palm_height.png, depth_shift.png).
- Spec items that turned out vacuous / borderline: the S9 "never fill a gap that contains t_contact, g or t_lift" can never fire (those are
  metric_valid frames by definition); G8's RMS < 6 mm is nearly automatic (RMS over 1 cm-band inliers is ≤ 5.77 mm for uniform residuals);
  G1's 0.1 s run limit means ≤ 5 frames at 60 Hz (6 frames = 0.10004 s); G5 on the 90° clip passes by 0.1 mm.
- Palm normal: never flipped where decided (3 clips); undecided on the top-down clip (no g).

**Runtime / memory per stage** (`<clip>/resources.json`; top-down / 90° / 45° / 45°): frames 6–12 s, 0.1 GB; hand (WiLoR, GPU) 63–70 s,
6.06 GB RSS, 2.82 GB VRAM; masks (SAM 2, GPU) 70–125 s, 5.4–9.4 GB RSS, 0.80 GB VRAM; table 8–22 s, ≤ 0.5 GB; hand_metric 1–2 s;
objects 20–27 s; table_frame, events, palm, demo < 1 s each; report 3–7 s. Peak RSS anywhere 9.35 GB (< 16 GB). Whole S4–S11 ≈ 40–60 s/clip.

---

## 2026-10-30 (Fri) — M2 v2: why definitions changed

Phase 0 of M2 v2. v1 results stay untouched (`results/m2_perception/`, `results/m1_pick_mug/`); v2 writes to `results/m2_perception_v2/`
and `results/m1_grasp_fix/`.

**config/objects.yaml:** `handle_height_m` 0.06 (now `measured`), `handle_thickness_m` 0.010 (`measured`, was an estimated 0.012),
`mass_kg` 0.40 (`estimated`: owner's estimate, no scale; was 0.37), new `mug.ruler_check` (ruler: diameter 0.10 m, height 0.09–0.10 m —
agrees with the label within ruler resolution; label values stay authoritative). Note: the v1 Track B runs used 0.37 kg and a 12 mm handle.

**Why the definitions change (README material):**
- **G5 and the contact threshold.** The fixed `r + 5 cm` contact threshold (and G5's upper bound) was set before seeing data, assuming the
  palm point (midpoint of wrist and middle-finger base) sits 1.5–2.5 cm off the mug surface during a side wrap. The data shows a
  palm-to-axis distance of 96–102 mm in every clip (v1: 96.1 / 96.6 / 99.4 mm at the grasp frame; the top-down clip's closest approach is
  0.102 m, `results/m2_perception/26d4118b0c/stages/events.json`), so the threshold sat on the data's edge: one clip passed G5 by 0.1 mm and
  the top-down clip missed contact by a few millimetres, losing every event. v2 defines contact relative to each clip's own closest approach
  (`D_min + 1 cm`) and keeps the distance in G5 as information only.
- **G3.** The 30–80 px plumbing error may come from a near-orthographic hand model (WiLoR's crop camera) placed under the real perspective
  camera, from a hand-scale mismatch at the true LiDAR depth, or both. v2 fits a hand scale (and a sideways shift) to WiLoR's own 2D palm
  keypoints to find out; G3 becomes the palm-keypoint reprojection after that fit.
- **Desk edge (S7).** A real bug: a line's extent counted the gaps between its fragments, so two distant collinear fragments (cup holder +
  desk corner) passed the "> 0.42 m = long edge" rule on both 45° clips. v2 uses the longest gap-free run (≤ 2 cm steps) and drops contour
  points near the mug and the coaster.
- **G8.** v1 gated on the normal-angle drift between the initial and end plane fits (the 90° clip failed at 2.43° — traced to ARKit attitude
  drift on a static camera). What affects our measurements is the plane offset at the mug, so v2 gates on that (< 1 cm) and reports the
  angle as information.
- **G6.** The ruler confirms the label (10 cm vs 9.9 cm), so the +6–7 mm free-fit diameter in every clip is a perception bias to
  investigate: v2 recomputes the static-window mug masks with SAM 2's image predictor at full resolution and compares.
- **hoi-sim preflight:** `pip install matplotlib==3.11.2 -c envs/hoi-sim.constraints.txt` (B2 asks for a plot; hoi-sim had no plotting
  library). 9 wheels (matplotlib + contourpy, cycler, fonttools, kiwisolver, pyparsing, python-dateutil, six, tzdata); numpy stays 1.26.4,
  `pip check` clean; lock regenerated (the diff is exactly those 9 lines).

### Track B v2 — grasp feasibility and fix

Track B ran on the CPU only (hoi-sim, MuJoCo 3.13.0). Workflow: implementer → reviewers → fixer (re-ran B2 and B3) → an independent checker. The checker re-derived every number below from `results/m1_grasp_fix/`:
- B2 verdicts from the per-D maxima and μ;
- B3 statistics and Wilson CIs from the per-trial rows;
- R0 against M1, trial by trial.

**No mismatch.** All 80 B3 trials of the final runs equal the pre-fix runs in every field except the new substep counts and wall time (checker, against scratch copies; not a saved result).

**B1 — facts from the code** (`b1_facts.md`, with file:line for each fact). Compiled values are in `b1_model.json`, written by `sim/grasp_fix_b1.py`. That script only runs mj_kinematics / mj_forward / mj_geomDistance; simulated time never advances.
- **Old cylinder:** D 0.080 m, h 0.100 m, 0.15 kg (`sim/scene.py:54`). Geom friction (0.35, 0.0028, 0.0001), condim 4, solref (0.008, 1).
- **Palm standoff is hard-coded.** The cup centre sits at palm + R·(0.077, 0.055, 0) m in the palm frame (`pick_place.py:58`), HERO's one Dex3 side-grasp constant. No radius is used anywhere in the oracle. Palm point to cup axis: 94.63 mm for every object. Along the palm normal: 55 − r = 15.0 mm (cylinder), 5.5 mm (mug).
- **The planned palm pose is already inside the object.** At the nominal grasp (hand open) the palm hull is at −2.15 mm on the cylinder and −11.65 mm on the mug. The proximal index/middle links are at +7.05 mm (cylinder) and −2.45 mm (mug).
- **The pre-grasp is hard-coded too:** 0.08 m back along palm → cup. That direction is (0.8137, 0.5812) in the palm frame, −9.46° in world.
- **Grasp height:** the palm point is 0.065 m above the table:
  - cylinder: +0.015 m above its centre (65 % of h);
  - mug: +0.0195 m above its centre (71 %).
  - The index finger spans 0.0804–0.1064 m above the table. It overlaps only the lowest 10.6 mm of the mug wall (19.6 mm on the cylinder).
- **Finger targets** (thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1):
  - open [0, 0.7, 0, 0, 0, 0, 0] rad;
  - closed (HERO demo grip) [0, 0, −1.2, 1.57, 1.2, 1.2, 1.2] rad.
- **Finger PD:** kp 1.2 (×3 = 3.6 while closing), kd 0.04. Torque limits: 2.45 N·m for thumb_0, 1.4 N·m for the rest. A blocked flexion joint squeezes with ≥ 0.84 N·m at full closure.
- **Contacts:**
  - Hand geoms: friction 1.5, condim 3, solref 0.02.
  - Cup–hand explicit pairs: μ 1.0625, torsional 0.00425 m, condim 4, solref 0.014.
  - Check with the hand closed: 15 contacts on the cylinder and 19 on the mug, all carrying the pair's parameters. So the **effective fingertip–mug μ = 1.0625**.
- **Global options:** cone pyramidal, impratio 1.0, noslip_iterations 0, Newton solver, Euler integrator, dt 0.001 s, solref/solimp override off. These are all MuJoCo defaults.
- **Dex3 right-hand ranges (rad):**
  - thumb_0 [−1.0472, 1.0472], thumb_1 [−1.0472, 0.7243], thumb_2 [−1.7453, 0];
  - index_0 / middle_0 [0, 1.5708];
  - index_1 / middle_1 [0, 1.7453].

**B2 — Dex3 capacity check** (`b2_capacity.json`, `b2_capacity.png`).
- μ = 1.0625, so the thresholds are 90° (comfortable) and 90° − atan(μ) = 43.26° (marginal).
- The palm normal is drawn through the palm centre (centroid of the palm collision geom, palm-frame x 0.0423 m).
- The cylinder is 0.091 m tall for every D, including 80 mm.

| D (mm) | φ_thumb_max | φ_finger_max (best finger) | verdict |
|---|---|---|---|
| 60 | none (0 of 9261 touching) | 115.30 (index) | infeasible |
| 65 | none | 111.46 (index) | infeasible |
| 70 | none | 107.50 (index) | infeasible |
| 75 | none | 103.80 (middle) | infeasible |
| 80 (M1 cylinder) | none | 100.06 (middle) | infeasible |
| 85 | none | 97.14 (middle) | infeasible |
| 90 | none | 93.81 (index) | infeasible |
| 95 | none | 90.96 (middle) | infeasible |
| **99 (mug)** | none | 88.17 (index) | **infeasible** |
| 100 | none | 87.62 (index) | infeasible |
| 105 | none | 85.11 (middle) | infeasible |
| 110 | none | 82.54 (middle) | infeasible |

- **Verdict for 99 mm: infeasible.** This is the literal reading, with the palm-centre anchor.
- **Why: the thumb never counts as touching, at any D.**
  - Its base link stands 8 mm proud of the palm face (palm-frame y 0.0283 m vs 0.0203 m).
  - So once the cylinder touches the palm centre, thumb_0 is 6.2–7.1 mm inside it in all 9261 configurations, and thumb_1 is 3.2–12.5 mm inside. Both exceed the 1 mm limit.
  - The thumb tip does reach the 0–3 mm band (240 configurations at 60 mm, 25 at 80 mm, none from 90 mm up), but never without that base penetration.
- **The fingers alone reach 88.17° at 99 mm.** The index tip gets there over the top rim edge; the middle reaches 88.15° on the side.
- **The same check rates the M1 80 mm cylinder infeasible**, yet the oracle grasps it 20/20 (R0). At this anchor the kinematic check and the dynamics disagree.
- **Anchor sensitivity** (`anchor_sensitivity`, information only: the normal moves along the palm, everything else stays the same):
  - 99 mm is infeasible for x 0.000–0.055 m and marginal for x 0.060–0.085 m.
  - At the oracle's own grasp position (x 0.077 m): 99 mm gets thumb 101.85° (281 touching) and finger 77.79° (middle), so marginal. 80 mm gets thumb 113.35° and finger 89.94°, also marginal.
  - At both anchors the 80 mm and 99 mm cylinders get the same rating.
  - Side effects at other anchors: from x 0.05 m on, the palm contact point moves up to 21.3° off the normal. For x ≤ 0.015 m the cylinder overlaps the wrist_yaw link (−1.39 to −7.29 mm).

**B3 — cause separation** (`b3_causes.json`, `b3_causes.md`; raw runs in `b3_runs/R0–R3.json`).
- The oracle is unchanged. Trials 0–19 of `results/m1_pick_mug` use the same layouts (asserted).
- R3 uses the same handle yaws (asserted); R0–R2 run at yaw 0.

| run | change | success (Wilson 95 %) | failures | lift slip, mm: mean / median [n]; trials > 20 mm | rise, cm: mean / median [n] | thumb + finger + palm at lift frame | thumb / index / middle / palm touching |
|---|---|---|---|---|---|---|---|
| R0 M1 cylinder, 0.15 kg | baseline | **20/20** (84–100 %) | – | 7.8 / 6.85 [20]; 0 | 7.28 / 7.16 [20] | 12/20 = 0.60 | 20 / 15 / 20 / 12 (of 20) |
| R1 cylinder 99 × 91 mm, no handle, 0.15 kg | size | **5/20** (11–47 %) | dropped while moving 9, slip during lift 5, missed/knocked cup 1 | 22.68 / 17.0 [19]; 5 | 5.76 / 6.06 [19] | 14/20 = 0.70 | 19 / 4 / 19 / 14 (of 19) |
| R2 same cylinder, 0.40 kg | mass | **0/20** (0–16 %) | slip during lift 17, dropped while moving 3 | 22.24 / 20.9 [20]; 14 | 4.93 / 4.94 [20] | 14/20 = 0.70 | 20 / 8 / 20 / 14 (of 20) |
| R3 full mug (handle 0.06 × 0.010 m), 0.40 kg | handle | **0/20** (0–16 %) | slip during lift 12, dropped while moving 8 | 20.75 / 20.1 [20]; 10 | 5.11 / 4.98 [20] | 13/20 = 0.65 | 20 / 6 / 20 / 13 (of 20) |

What the runs show:
- **R0 = M1.** 20/20, and every trial is identical to M1's in success, grasp numbers, final cup pose and sim time (`r0_vs_m1`).
- **Size (R0 → R1): 20/20 → 5/20.**
  - Median lift slip 6.85 → 17.0 mm. Trials over the oracle's 20 mm slip limit: 0/20 → 5/19.
  - The R1 mean (22.68 mm) is pulled up by trials 5 and 10, which slipped 69.5 / 68.1 mm and rose only 1.50 / 1.85 cm.
  - The index finger touches at the lift frame in 15/20 → 4/19 trials.
  - R1 trial 0 knocked the cup on the approach (it moved 2.0 cm).
- **Mass (R1 → R2): 5/20 → 0/20.**
  - Median slip 17.0 → 20.9 mm; trials over the limit 5 → 14.
  - Median rise 6.06 → 4.94 cm, below the 5 cm the oracle requires.
  - Lift failures 5 → 17.
- **Handle (R2 → R3): no visible effect on success.** R2 is already 0/20, so there is nothing left to lose.
  - Median slip 20.9 → 20.1 mm; median rise 4.94 → 4.98 cm.
  - Failures shift from lift to transport: lift/transport 17/3 → 12/8.
- **Size causes the largest single drop, and the mass removes what is left.** Each step is one 20-trial run.
- **The contact measure does not predict success.** R0 succeeds 20/20 with thumb + finger + palm contact in only 12/20 trials. In R1–R3, 10, 14 and 13 failed trials had that contact.
- **Substep sensitivity** (information only):
  - Grasp contact in the last / any / all of the lift frame's 20 substeps: R0 12/12/12, R1 14/16/11, R2 14/14/11, R3 13/14/12.
  - Index finger, last / any substep: R0 15/18, R1 4/6, R2 8/10, R3 6/9.

**B4 decision: STOP** (main agent: applied literally; the anchor question below goes to the user before any B5 work). B2 rates 99 mm "infeasible", so Track B stops after B3, as the spec says. No B5 rung was implemented or run.
- This verdict depends on where the palm normal is drawn (open question 1). At the oracle's grasp position, 99 mm would be "marginal" and B5 would run.

**Ladder (B5) and B6:** not run, because of the B4 stop. There is no `ladder.json`, no final 50-layout evaluation and no videos.
- No friction coefficient, PD gain, torque limit, mug geometry or mass was changed. Track B v2 has no code that sets cone, impratio or noslip.
- Legacy check: R0 with the legacy settings = 20/20, identical to M1.

**What failed or is limited:**
- **B2 as specified does not separate the objects.** At the palm-centre anchor every D from 60 to 110 mm is infeasible, because the thumb base sits inside the cylinder. That includes the 80 mm cylinder the oracle grasps 20/20.
- **The oracle's planned palm pose is inside the mug** (−11.65 mm, vs −2.15 mm for the cylinder). In a run the palm presses on the mug or stops short.
- **φ_max is a lower bound**, because the grid has 21 values per joint.
- **μ was used as the literal pair value.** With the pyramidal cone, the friction limit between the cone's edges is about μ/√2 = 0.75, which would give a 53.08° threshold. This is saved as information only.
- **`sim/m1_pick_mug.py` and the new config (fixed by the main agent):** its MAIN_MASS read the config, now 0.40 kg, so the v1
  commands looked for `runs_mass0.40.json` and a default `run` would have written into the frozen v1 folder. MAIN_MASS is now pinned to
  the v1 protocol's 0.37 kg (comment in the file); `summarize` on copies of the v1 runs files reproduces the v1 success table exactly.
  (The v1 mug geometry used a 12 mm handle; the current config's 10 mm handle only matters if v1 trials are re-run.)
- **A sub-agent ran two read-only git commands** (`git status --short`, `git diff sim/scene.py`) before noticing the no-git rule. Nothing changed.

**Files.**
- Results (`results/m1_grasp_fix/`): `b1_facts.md`, `b1_model.json`, `b2_capacity.json`, `b2_capacity.png`, `b3_causes.json`, `b3_causes.md`, `b3_runs/R0.json`–`R3.json`.
- Code:
  - new: `sim/grasp_fix_b1.py`, `sim/b2_capacity.py`, `sim/m1_grasp_causes.py`;
  - changed: `sim/scene.py` (the `mug_config()` skip, `cup_model="mug_body"`, the `on_substep` hook).

**Runtime / memory** (CPU only, no GPU):
- B2: 5.8 s, peak RSS 975 MB.
- B3 (four parallel processes): R0 100.4 s / 1825 MB, R1 85.4 s / 1660 MB, R2 77.0 s / 1651 MB, R3 81.4 s / 1688 MB.
- B1: runtime and RSS were not saved to a results file.

**Adaptations** (logged; no definition changed):
- `mug_config()` skips config entries without a `value`, such as `ruler_check`.
- New `cup_model="mug_body"`: the mug's body cylinder at its config size, without the handle (for R1/R2). The "cylinder" model still builds the M1 cup unchanged.
- `TableScene.on_substep`: an optional read-only observer, None by default. R0 = M1 shows that behaviour is unchanged.
- B2 uses the sim's compiled `TableScene(cup_model="mug")` and resizes its body cylinder for each D. The 80 mm row uses h 0.091 m, so only the diameter varies.
- B2: the φ sign is fixed per digit by the hand (`WRAP_SIGN`), and closest points are labelled side / cap / rim edge.
- B3's "first 20 layout seeds" = trials 0–19 of `results/m1_pick_mug`, i.e. M1's layouts from seeds 0 and 1, as in v1.
- Information-only extras, beyond the spec's list (none changes a headline number; can be dropped):
  - B2 `anchor_sensitivity`;
  - B3 substep counts, slip medians and over-limit counts;
  - `sim/grasp_fix_b1.py` + `b1_model.json`.

**Open questions (for the user):**
1. **B2 anchor:** where along the palm should the palm normal be drawn?
   - At the palm centre (used): 99 mm is infeasible, so Track B stops.
   - At the oracle's grasp position (x 0.077 m): 99 mm is marginal, so B5 would run.
   - This decides B4.
2. **B2 μ:** the literal pair value 1.0625 was used. With the pyramidal cone the limit between the cone's edges is about μ/√2.
3. **B3 "at the lift frame":** the last 1 ms substep (used), any substep, or all 20 substeps?
4. **If B5 runs, L1 "same palm-to-surface clearance":** which clearance should stay the same, and in which direction should the grasp shift?
   - Candidates for the clearance: the palm hull at −2.15 mm, 54.63 mm along the approach line, or 15.0 mm along the palm normal.
   - Also: should B5/B6 keep the R0 cup and coaster positions fixed? `sample_layouts` derives cup positions from GRASP_OFFSET, so changing the offset changes the layouts.
5. **Keep or drop** `sim/grasp_fix_b1.py` and the information-only extras?

### M2 v2 — Track A: perception v2 (all numbers: `results/m2_perception_v2/summary.md`, `<clip>/gates.json`, `<clip>/stages/`)
**How it was built.** Shared change by me first: `common.stage_cli` takes several stages per module (`--stage`), `RESULTS` points to
`results/m2_perception_v2` (v1 read-only). A v2 contract fixed the implementation details before any v2 result (S5b: X = S5 kp3d,
c = mean of the S5-shifted verts, x0, residual in ORIGINAL pixels; re-measured LiDAR residual = median |lidar − mesh| over S5's hand
pixels within 3 cm, no new shift; A3 drop radii r + 3 cm and fitted coaster radius + 3 cm; consumers switch to `hand_fit.npz`; A4 stages
`masks_full` + `mask_test` and the S6 decision; two-pass `run_clip` because s_hand and the A4 decision need all 4 clips). One decision was
the user's: final-window A4 prompts come from that frame's video masks (box, deepest interior point, handle centroid as negative).
Four module groups were implemented in parallel (implement → 2 adversarial reviewers → fixer); then I ran the whole v2 pipeline clean
(`python -m perception.run_clip`, 499 s): every gate and cross-view value reproduced the agents' runs exactly. A separate verification
workflow re-derived every v2 gate, cross-view value, s_hand, the A4 decision and the schema-3 fields with independent code: **no mismatch**.

**Gates v2** (v1 → v2 for every gate is in summary.md):

| gate | 26d4118b0c top-down | 4e103843db 90° | 49aff04f7a 45° | e8825dea53 45° |
|---|---|---|---|---|
| G1 coverage | ✔ (v1 ✘: no events) | ✔ | ✔ | ✔ |
| G2 LiDAR residual (S5b hand) | ✔ 0.48 cm | ✔ 0.51 cm | ✔ 0.51 cm | ✔ 0.80 cm |
| G3 palm reprojection | ✘ 36.7 px | ✘ 38.5 px | ✘ 50.6 px | ✘ 12.1 px |
| G4 jitter | ✔ | ✔ | N/A | N/A |
| G5 z_grasp | ✔ 50.1 mm | ✔ 46.0 mm | ✔ 62.9 mm | ✘ 88.7 mm (g before contact) |
| G6 mug fit (video masks) | ✘ Ø 105.1 | ✘ Ø 105.7 | ✘ Ø 106.1 | ✘ Ø 105.8 |
| G7 placement | ✔ 0.81 cm | ✔ 2.15 cm | ✔ 1.04 cm | ✔ 0.54 cm |
| G8 table frame | ✔ x 1.3° | ✔ x 1.5° (v1 ✘ drift angle) | ✔ x 15.7° (v1 ✘ 23.9°) | ✔ x 7.6° |
| G9 hand scale ±10 % | ✘ +12.9 % | ✔ −3.2 % | ✘ −17.1 % | ✔ +1.4 % |

**A1 — hand scale (S5b).** s_hand = 1.0219 (pooled median over 1103 metric_valid frames). Per-clip median s [IQR]: 1.154 [1.116–1.193],
0.989 [0.966–1.025], 0.847 [0.767–1.031], 1.036 [1.019–1.060]. The fit does not reach WiLoR's 2D palm keypoints: palm reprojection after
pass 2 is 12–51 px (G3 fails everywhere; v1's all-21 plumbing numbers were 31–81 px). The hand-scale fixer's scratch probe (not a saved
result) found that WiLoR's 2D keypoints are an orthographic image of its own 3D palm to sub-pixel accuracy; no scale + sideways shift
under the real perspective camera can reproduce that, so a 12–51 px floor remains and the fitted perspective scale soaks up part of the
distortion — which plausibly explains why s varies with the view (G9 fails on two clips). Re-measured LiDAR residual: NaN on 3
metric_valid frames (4e103843db i = 90, 49aff04f7a i = 359, 367: no hand pixel left) — G2 takes the median over the frames that have a
value and names the others; it passes either way.

**A2 — contact = D_min + 1 cm.** The top-down clip now has events (D_min 106.5 mm; contact 60, lift 87). But on 3 clips the closest
approach happens AFTER the lift, while the hand carries the mug (D_min frame vs lift: 4e103843db 225 vs 214, 49aff04f7a 201 vs 182,
e8825dea53 159 vs 151; palm z there 92 / 131 / 148 mm ≥ h, flagged). On e8825dea53 this moves contact to frame 149, after g (148):
z_grasp 88.7 mm = 0.98 h, G5 fails, and the hold before the lift is labelled approach. Implemented literally; needs the user's decision.

**A3 — desk edges.** External contour + 3 cm drop around the mug and coaster + gap-aware extent: the junk lines are gone (49aff04f7a's
0.584 m gap line now has extent 0.04 m; the real 0.50 m edge decides), G8 passes on all 4 clips. Remaining imperfections seen by the
verifiers: object-outline remnants survive the 3 cm drop (top-down: the rim projects further out than the footprint), and on e8825dea53 a
short non-edge piece (L3) sits in the long-edge group (x error 7.6°, within 20°).

**A4 — mug masks.** Full-res image-predictor masks give init free-fit diameters 114.3 (top-down) / 105.8 / 106.1 / 105.7 mm vs video
105.1 / 105.7 / 106.1 / 105.8 mm → 0 of 4 within 5 mm → S6 keeps the video masks (G6 unchanged from v1). Caveat for the reading: SAM 2
resizes every input to 1024 × 1024 in both predictors (third_party/sam2 transforms), so "full-res" changes the input sampling and the
predictor, not SAM 2's working resolution; the +6–7 mm remains unexplained. On the top-down clip the 4-DOF diameter is ill-conditioned
(h unobservable but free; the verifier's Nelder–Mead with another initial simplex gives 106–113 mm at near-equal IoU). Final-window
full-res fits failed on 2 clips (IoU 0.658 / 0.609; the positive prompt point landed in the white interior, so SAM 2 segmented the rim)
— not used, since the rule uses the init window and chose video.

**Runtime / memory (v2 stages, `<clip>/resources.json`):** hand_scale ≤ 0.5 s, hand_fit 1–2 s (≤ 0.25 GB); masks_full (SAM 2 image
predictor, GPU) 4–6 s, 2.1 GB RSS, 0.71 GB VRAM; mask_test 89–119 s (≤ 0.34 GB); objects / table_frame / events / palm / demo < 1 s;
report 4–7 s. Unchanged S1–S5 as in v1 (peak RSS 9.35 GB in SAM 2 video).

**Small fixes after verification:** summary.json stores s_hand as a number (a list only if the demos disagree); wilor_smoke docstring.
**Process notes:** some sub-agents ran read-only git commands despite the no-git rule (nothing changed); `perception/inventory.py` still
writes to the v1 folder if rerun (it is the v1 Phase 0 script; not rerun).

**Open questions for the user (Track A):** (1) A2: restrict D_min (e.g. palm z < h, or frames before the lift)? (2) G3/G9: WiLoR's
orthographic 2D vs a perspective fit — change G3's definition or the S2 placement? (3) G6: the +6–7 mm bias persists with full-res
masks — accept, or investigate the mug's real profile? (4) G2: frames without a re-measured residual — exclude (current) or count as fails?
