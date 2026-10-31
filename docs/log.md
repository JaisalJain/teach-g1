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

---

## 2026-10-30 (Fri) — M2 v3: grasp fix ladder (Track B) + final perception fixes (Track A); perception is frozen after this

v1 and v2 results stay untouched; v3 writes to `results/m1_grasp_fix/v3/` and `results/m2_perception_v3/`.

**Why definitions change (user's wording):**
- **B2 anchor.** B2 v2's palm-centre anchor failed its own positive control: it rated the 80 mm cylinder infeasible although the oracle
  grasps it 20/20. The oracle's hard-coded offset is not a valid anchor for other diameters either, because it puts the palm 11.65 mm
  inside a 99 mm mug. v3 anchors at the radius-consistent grasp pose, which equals the oracle pose at 80 mm.
  (Radius-consistent pose: the object axis sits at palm-frame (0.077, r + 0.015, 0); for r = 0.040 this is the oracle's (0.077, 0.055, 0).)
- **Hand scale.** The v2 hand-scale fit absorbed WiLoR's orthographic-vs-perspective mismatch (per-view scales 0.85–1.15, pooled 1.02),
  so v3 fixes the scale at 1. The v2 scale results stay in `results/m2_perception_v2/` and the "M2 v2 — Track A" entry above as a
  documented negative result (G9 is removed in v3).
- **Contact rule.** The v2 contact rule could select a closest approach that happened after the lift. v3 takes the closest approach only
  before the palm rises near the mug (`t_rise`).
- **G3** becomes information only (palm reprojection in px and in mm at the hand: monocular hand-model limitation, orthographic model vs a
  35–45 cm perspective camera). **G6** stays reported with unchanged thresholds, marked "known bias +6–7 mm, cause unresolved"; downstream
  positions are validated by G7 and the sim uses the label geometry.

### M2 v3 — Track A: final perception fixes (all numbers: `results/m2_perception_v3/summary.md`, `<clip>/gates.json`, `<clip>/stages/`)
**How it was built.** Contract fixed before results (closed-form palm-centroid shift; re-measured residual = v2's rule; A2v3 running
near-min uses each frame's own minimum, 5 consecutive metric_valid frames including k; D_min strictly before t_rise). Three groups in a
workflow: A1v3 and A2v3 implemented by opus agents, schema 4 / gates v3 / summary / run_clip by a sonnet agent, one opus reviewer each,
fixers. The v2 scale / fit stages were deleted (git history keeps them). A clean `run_clip` (45 s; S1–S4, S6 cached) reproduced every
gate and cross-view value exactly; an independent verification workflow (sonnet, own code per clip + schema-4 audit) found **no mismatch**.

**A1v3 — palm placement (s = 1).** Per metric_valid frame: t = ((u_x − cx) P_z / fx − P_x, (u_y − cy) P_z / fy − P_y, 0), P = mean of the
5 S5-shifted palm keypoints, u = mean of their WiLoR kp2d; asserted to 1e-6 px (float64: ~3e-13 px). Median |t| 5.6 / 12.1 / 14.9 / 4.4 mm.
**A2v3 — contact rule** (near = d_axis ≤ r + 8 cm; t_rise = first near frame with z ≥ running near-min + 3 cm for 5 valid frames; D_min
before t_rise; contact = first frame ≤ D_min + 1 cm). Events (frame index; clip time of t_rise):

| clip | t_rise | D_min (frame) | contact | grasp g | lift | place | release | z_grasp |
|---|---|---|---|---|---|---|---|---|
| 26d4118b0c top-down | 89 (2.97 s) | 106.7 mm (68) | 60 | 84 | 87 | 156 | 189 | 50.0 mm |
| 4e103843db 90° | 217 (3.88 s) | 96.9 mm (215) | 162 | 211 | 214 | 345 | 414 | 44.6 mm |
| 49aff04f7a 45° | 186 (3.10 s) | 98.3 mm (164) | 147 | 179 | 182 | 284 | 356 | 62.4 mm |
| e8825dea53 45° | 145 (2.42 s) | 94.5 mm (144) | 107 | 140 | 143 | 267 | 333 | 62.7 mm |

Contact comes before g on every clip. Remaining quirk (information, no rule changed): t_rise is a 3 cm rise and the lift a 1.5 cm rise, so
t_rise follows the lift by 2–4 frames and D_min can still fall in [lift, t_rise): 4e103843db (D_min 215, lift 214) and e8825dea53 (144, 143);
contact is unaffected. On e8825dea53 the near-minimum is set at the first near frame (d_axis 0.2 mm inside r + 8 cm) while the hand still
reaches in.

**Gates v3:**

| gate | 26d4118b0c top-down | 4e103843db 90° | 49aff04f7a 45° | e8825dea53 45° |
|---|---|---|---|---|
| G1 coverage | ✔ 100 % | ✔ 100 % | ✔ 99.2 % | ✔ 99.7 % |
| G2 LiDAR residual (placed hand) | ✔ 0.47 cm | ✔ 0.57 cm | ✔ 0.47 cm (2 frames without residual) | ✔ 0.76 cm |
| G3 (information) palm reprojection | 34.75 px = 8.18 mm | 43.09 px = 10.94 mm | 51.72 px = 13.48 mm | 14.17 px = 3.52 mm |
| G4 jitter | ✔ 0.07 cm | ✔ 0.07 cm | N/A | N/A (longest slow run 0.267 s) |
| G5 z_grasp | ✔ 50.0 mm | ✔ 44.6 mm | ✔ 62.4 mm | ✔ 62.7 mm (v2 ✘) |
| G6 mug fit (known bias) | ✘ Ø 105.1 | ✘ Ø 105.7 | ✘ Ø 106.1 | ✘ Ø 105.8 |
| G7 placement | ✔ 0.81 cm | ✔ 2.15 cm | ✔ 1.04 cm | ✔ 0.54 cm |
| G8 table frame | ✔ x 1.3° | ✔ x 1.5° | ✔ x 15.7° | ✔ x 7.6° |

(G3 "mm" = median over frames of px × z_c / fx; G3 label per the user: monocular hand-model limitation, orthographic model vs a 35–45 cm
perspective camera — the verifier notes the measured palm depth is ~31–33 cm median. G9 removed; the v2 scale results remain the
documented negative result.) v2 → v3: G5 on e8825dea53 fail → pass (events back in order); G2 / G3 values moved slightly with the new
placement (palm px vs v2's scale fit: lower on 1 clip, higher on 3; vs the S5 hand before placement: lower on all 4 — 46.5 → 34.7,
46.0 → 43.1, 58.9 → 51.7, 19.4 → 14.2 px, from hand_place done.json); everything else unchanged.

**Runtime / memory:** hand_place 0.8–1.5 s (≤ 0.2 GB); table_frame / events / palm / demo < 1.5 s; report 3.6–6.9 s; whole S5→S11 v3
pass 45 s. GPU: none in v3 (S2/S3/A4 caches).
**Process notes:** sub-agents ran no git in v3 Track A (per their reports); the demo field `t_rise` is on the odometry clock (like
`t_contact`), the cross-view column uses clip time.

### Track B v3 — fix ladder

**Question.** Does any cumulative change in the order L1, L2, L3 get the full mug (B3's R3: config mug, 0.40 kg, B3's 20 fixed trials) to at least 18/20? **Answer: no.** L1 0/20, L2 0/20, L3 11/20. No rung reached 18/20, so the 50-layout final evaluation and the example videos were not run. Everything below is in `results/m1_grasp_fix/v3/`. CPU only, no GPU, hoi-sim. Every number below was re-derived from the per-trial rows and per-D maxima by an independent check; the "Checked" lines say how.

**Constraints kept.** No friction coefficient, PD gain, torque limit, mug geometry or mug mass was changed. Every change is a named `PickPlace` parameter with the legacy value as default (`grasp_offset`, `closing_targets`, `contact_model`). A compile check of legacy against L3's contact model shows that only `geom_condim` (4 right-hand geoms) and the options `cone`, `impratio` and `noslip_iterations` differ. Trials for every rung and for R0 are B3's trials 0-19 (layouts and handle yaws computed with the legacy grasp and asserted per trial), so a new grasp offset never re-samples the layouts.

#### Radius-consistent pose and its assert
`pick_place.py` kept the object axis at palm-frame (0.077, 0.055, 0). The palm-normal component 0.055 = r_old (0.040) + 0.015. The radius-consistent pose keeps x and z and sets the palm-normal component to r + 0.015. It is `radius_consistent_offset(r)`, with `PALM_SURFACE_GAP = 0.015`. The assert on import is `radius_consistent_offset(0.040)[1] == 0.055 and np.array_equal(radius_consistent_offset(0.040), GRASP_OFFSET)`. It holds exactly in float64. For the mug, r = 0.0495 m is read from the compiled `cup_geom`, which gives (0.077, 0.0645, 0). The 8 cm pre-grasp backoff is unchanged. Because the offset changed, the approach direction in the palm frame went from 35.54 deg to 39.95 deg (world (0.9864, -0.1644) to (0.9961, -0.088)); the palm offset went from (-0.0933, 0.0156, 0.015) to (-0.1001, 0.0088, 0.0195).

#### B2v3 — capacity at the radius-consistent pose (`b2v3_capacity.json`, `.png`)
Same method as v2 (grid, touching rule: tip distance in [0, 3] mm and no geom of that digit deeper than 1 mm; phi definition, mu = 1.0625 from the explicit cup x fingertip pair, thresholds comfortable >= 90 deg, marginal >= 43.26 deg = 90 - atan(1.0625)). One change: the cylinder axis sits at the radius-consistent pose for r = D/2, base on the table, height 0.091 m (as in v2). phi = 0 is the cylinder point facing the palm. An assert checks that this point lies on the surface at the radius set. The verdict is the weaker of the thumb and the best finger (the best finger is the larger of index and middle).

| D (mm) | thumb max phi (deg) | best finger max phi (deg) | thumb touching configs | verdict |
|---|---|---|---|---|
| 60 | 148.095 | 90.931 (middle) | 299 | comfortable |
| 65 | 144.033 | 86.598 | 319 | marginal |
| 70 | 138.925 | 83.837 | 327 | marginal |
| 75 | 133.697 | 79.748 | 352 | marginal |
| **80 (positive control)** | 130.113 | 77.523 | 356 | **marginal** |
| 85 | 124.906 | 73.434 | 352 | marginal |
| 90 | 121.962 | 71.129 | 344 | marginal |
| 95 | 118.712 | 68.490 | 314 | marginal |
| **99 (gate)** | 115.858 | 66.568 (index) | 285 | **marginal** |
| 100 | 113.854 | 66.136 | 290 | marginal |
| 105 | 110.784 | 64.336 | 256 | marginal |
| 110 | 108.052 | 62.101 | 203 | marginal |

- **Positive control passed:** 80 mm is marginal, not infeasible. At 80 mm the pose is exactly the oracle's `GRASP_OFFSET`. The row uses h = 0.091 m like every row, not the M1 cylinder's 0.10 m.
- **99 mm gate passed:** marginal (thumb 115.86 deg, best finger 66.57 deg, margin to infeasible 23.3 deg). Decision in the json: continue to the ladder.
- Thumb: above 90 deg at every D, and 203-356 touching configurations per D. In v2 the thumb had 0 at every D because the cylinder was slid along the palm normal until it touched the palm. The v3 axis sits at palm x = 0.077, away from the thumb base.
- Fingers: only the 60 mm row is comfortable. Fingers have only 4-6 of 441 configurations touching per D (the 0-3 mm band is thin against the joint step), so finger maxima are coarse grid lower bounds.
- At 99 mm all three digits have a touching configuration, so L2 is computable for all three. Max-wrap points: thumb on the side (tip gap 0.313 mm); index on the **top rim edge** (gap 1.218 mm); middle on the side at 66.55 deg (gap 1.132 mm). The index point is on the top rim edge at every D.
- Palm geom distance to the cylinder at the pose (information, not used by the touching rule): -1.712 mm (60 mm) to -2.504 mm (110 mm); -2.151 mm at 80 mm (equals B1's -2.15 mm for the legacy cylinder) and -2.400 mm at 99 mm. The wrist_yaw geom stays 36.3-50.9 mm clear at every D (38.989 mm at 99 mm).
- Checked: every verdict re-derived from the per-D maxima (12/12 match); thumb touching counts as above.
- Cost: 1.7 s, peak RSS 924 MB.

#### L1 — geom-distance check (`l1_check.json`)
Hand open at the planned pose; minimum signed distance over all right-hand collision geoms x all object geoms (`mj_geomDistance`).
- (a) M1 cylinder, legacy offset: **-2.151 mm**, pair palm hull `hand_contact_right_hand_palm_link_8` and `cup_geom`.
- (b) Mug at L1: **-2.400 mm**, same pair. Identical at all 20 trial handle yaws. The handle stays at least 51.98 mm from the hand.
- b minus a = -0.249 mm. That is within the 0.5 mm limit, so the check passes. Nothing was adjusted.
- Other geoms come closer by more than 0.5 mm (thumb_2 36.51 to 27.63 mm, -8.87 mm change; thumb_1 20.48 to 14.29 mm; index_0 and middle_0 7.05 to 6.59 mm, with those fingers' base links slightly less far), but none becomes the minimum.
- Important reading: both poses have the palm hull inside the object. The legacy offset on the mug would put the palm 11.65 mm inside (B1). L1 backs the palm out by 9.25 mm, to the same 2.15-2.40 mm overlap the cylinder has. So L1 reproduces the old overlap; it does not create clearance.
- The placement code reproduces B1's independent placement exactly (max difference 0.00 mm for cylinder and mug). L1's trials equal B3 R3's 20/20.
- Cost: 2.3 s, peak RSS 2989 MB (three scenes built).

#### Ladder (`ladder.json`; per-trial rows in `runs/`)
Rungs are cumulative; contact fraction at lift = thumb AND (index or middle) AND palm touching an object geom at the last substep of the first control step of the lift phase (B3's definition, denominator 20). Re-derived from the per-trial rows (success, categories, Wilson 95 % CI, slip mean/median over trials that reached the lift, rise, contact): 0 mismatches.

| rung | change (cumulative) | success | Wilson 95 % | failures | lift slip mean / median (mm) | over 20 mm | rise mean / median (cm) | contact at lift |
|---|---|---|---|---|---|---|---|---|
| B3 R3 (reference) | none | 0/20 | 0.000-0.161 | slip 12, dropped 8 | 20.75 / 20.1 (n 20) | 10 | 5.11 / 4.98 | 13/20 |
| L1 | radius-consistent palm target | 0/20 | 0.000-0.161 | slip 15, dropped 5 | 21.96 / 22.7 (n 20) | 12 | 4.99 / 4.96 | 14/20 |
| L2 | L1 + per-digit IK closing targets | 0/20 | 0.000-0.161 | slip 19, dropped 1 | 47.06 / 30.3 (n 20) | 19 | 3.21 / 4.34 | 14/20 |
| L3 | L1 + L2 + contact model | 11/20 | 0.342-0.742 | close-gate miss 9 | 6.8 / 6.6 (n 11) | 0 | 6.87 / 6.86 | 5/20 |

The stop rule (>= 18/20) is never met. `stopped_at` is null, and `next` reads "no accepted rung reached >= 18/20". L3 was the last spec rung.

**L1 (0/20).** Compared with R3 nothing moved for the better: more trials over 20 mm (12 against 10), rise 4.99 against 5.11 cm, index touching 10 against 6. 3 of the 15 lift failures (trials 2, 7, 10) had slip <= 20 mm and failed only on rise < 5 cm; 5 more trials with slip <= 20 mm failed later, in transport. The 14/20 contact fraction is 0.70 (thumb 20, index 10, middle 20, palm 14).

**L2 (0/20).** IK per digit (`l2_targets.json`, scipy `least_squares`, joints of that digit only, bounds = joint ranges), from each digit's B2v3 99 mm max-wrap configuration; target = the closest cylinder point moved 5 mm radially into the cylinder.
- Closed targets (thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1) in rad: (-1.9e-9, 0.4064, -1.1901, 0.2179, 1.2948, 0.2173, 1.2936). HERO's: (0, 0, -1.2, 1.57, 1.2, 1.2, 1.2). Open pose, ramp, touch logic, kp/kd and squeeze rule unchanged.
- IK residuals: thumb 2e-6 mm; index 0.105 mm (along palm z); middle 2.8e-5 mm. No joint at its limit; scipy status 1 (gtol) for all three. Tip geom distance at the solution: thumb_2 -5.155 mm, index_1 -5.185 mm, middle_1 -5.227 mm.
- Index caveat: its max-wrap point is on the top rim edge, so a radial 5 mm move puts the target on the top-cap plane, 0.0 mm inside the solid (thumb and middle: 5.0 mm). Implemented literally, as the contract says.
- Result: 19 slip failures, 1 dropped. Slip got worse (47.06 / 30.3 mm). In 7 trials the cup stayed essentially on the table while the hand lifted: rise <= 1.01 cm in trials 1, 6, 8, 11, 12, 14, 15 (0.08, -0.04, 1.01, 0.00, 0.69, 0.00, 0.00 cm). Median closure at first contact: 0.976 (L1: 0.488). Substep sensitivity: grasp contact in any substep 20/20, in all substeps 9/20, in the last substep 14/20.
- **Not measured:** why. A hypothesis is that the targets sit only 0.05-0.07 rad past the touching configuration (near the squeeze rule's 0.05 rad threshold), so the grip torque is low; no finger torque was logged.

**L3 (11/20, accepted by its guards, below the stop rule).** `contact_model="grasp"`: elliptic cones, `impratio` 10, `noslip_iterations` 3, condim 4 on right palm, thumb_2, index_1, middle_1 (the cup geoms and its 72 explicit pairs were already condim 4). Solver (Newton) and tolerance 1e-8 unchanged.
- Failures: 9, all "no thumb + finger contact at full closure (missed the cup)" in the close phase (trials 0, 2, 3, 4, 8, 9, 15, 17, 18). The label in the rows is M1's "missed/knocked cup", but the cup was **not** knocked: tilt <= 0.1 deg, still at z 0.7855 m, moved 1.5-10.8 mm. The 11 trials that reached the lift all passed: slip mean 6.8 / median 6.6 mm, none over 20 mm, rise 6.87 / 6.86 cm.
- Slip is no longer a failure mode, and the failure moved to the 0.3 N touch gate. Contact at lift is 5/20 (thumb 11, index 4, middle 10, palm 5 of the 11 that reached the lift).
- **Not measured:** which digit stays under 0.3 N in the 9 failures; no fingertip force was logged.
- Guards (spec): (1) `m1_reach_test` at 0.74 m with the L3 options, `reach_l3/`: no fall; pelvis z min 0.686 m (M1 0.689), tilt max 12.8 deg (M1 12.9), no body part on floor or table, drift 0.086 m (M1 0.088), palm error mean of means 1.73 cm and mean of finals 1.51 cm (M1 1.70 / 1.43). (2) R0 with L3's own change only (`L3_R0`: cylinder, 0.15 kg, B3 R0 trials, `contact_model` grasp): 20/20, Wilson 0.839-1.0, slip 3.86 / 3.25 mm, rise 7.79 / 7.94 cm, contact 10/20. Both pass, so `accepted = true`. L1 and L2 have `accepted = null` in the json (no guard in the spec). The reach test's legacy default reproduces M1's `results/m1_reach` field by field and array by array (excluding only wall-clock fields `wall_time_s` and `initial_plan.ik_s`).

**R0 at legacy settings through the new code.** 20/20 (Wilson 0.839-1.0) in `runs/R0.json`, and again after the L2 and L3 shared-code edits (`runs/R0_after_l2.json`, `runs/R0_after_l3.json`). All 20 trials are identical to B3 R0 in every row field except wall time, and identical to M1. Slip 7.8 / 6.85 mm, rise 7.28 / 7.16 cm, contact 12/20 each time.

#### MuJoCo documentation for L3 (3.13.0, doc sources at tag 3.13.0, RST markup removed)
- Modeling > Tips and tricks > Preventing slip > Slow slippage (https://mujoco.readthedocs.io/en/stable/modeling.html#cslowslippage): "a. Use elliptic cones and increase impratio. This makes the friction dimensions harder relative to the normal dimension without increasing the friction coefficient. It reduces slow slippage but does not guarantee exact sticking. [...] b. For stronger suppression, enable the NoSlip solver by setting noslip_iterations to a positive integer. A small number (1, 2 or 3) is usually sufficient. [...]"
- Modeling > Preventing slip (https://mujoco.readthedocs.io/en/stable/modeling.html#cslippage): "Increase condim to 4 or 6 and choose appropriate friction coefficients. condim 4 enables torsional friction, preventing rotation around the normal."
- Modeling > Introduction > Solver settings (https://mujoco.readthedocs.io/en/stable/modeling.html#calgorithms): "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the Newton algorithm with very small tolerance. If that is not sufficient, enable the Noslip solver."
- Overview > Softness and slip (https://mujoco.readthedocs.io/en/stable/overview.html#soft): "Thus using the Newton solver with elliptic friction cones and large value of impratio is the recommended way of reducing slip."
- XML Reference > option > impratio (https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-impratio), noslip_iterations (#option-noslip-iterations), cone (#option-cone); Computation > Constraint solver > NoSlip (https://mujoco.readthedocs.io/en/stable/computation/index.html#sonoslip). Full quotes are in `ladder.json`, `rungs[L3].contact_model_sources`.
- The values: the docs give no number for `impratio`. 10 is the value DeepMind's MuJoCo Menagerie uses with `cone="elliptic"` in robotiq_2f85, aloha, wonik_allegro and shadow_hand (LEAP uses 100). `noslip_iterations` 3 is the top of the docs' "1, 2 or 3". Both were fixed before the first L3 run, and no other values were tried.

#### Adaptations (reading the spec literally)
- L1 pre-grasp: the 8 cm backoff is kept, but its direction follows the new offset (39.95 deg against 35.54 deg in the palm frame). The retreat uses the same approach direction. B3's handle yaws stay centred on the legacy approach direction, so the handle spread is about 4.4 deg off L1's own approach.
- L2 start/target: each digit starts from its own B2v3 99 mm argmax with the other digits at 0. The fixed point is the tip-geom point that was closest in that configuration. L2 reads `l2_targets.json`, so the run uses the saved numbers.
- The L2 closed pose changes the squeeze rule's closing direction (sign of closed minus open). On the flexion joints this is unchanged; only thumb_0 goes from 0 to -1 because the IK returned -1.9e-9, and thumb_0 is not a flexion joint.
- L3's R0 guard uses L3's own change only (`contact_model` grasp). L1 is a no-op on the 80 mm cylinder and L2's targets are for the 99 mm mug.
- Parallelism: 4 processes of 5 consecutive trials, 1 onnxruntime thread each.

#### Open questions
1. `impratio` = 10 is not from the MuJoCo docs (they give no number); it is the Menagerie value for elliptic-cone hands. `noslip_iterations` = 3 is the top of the docs' range; 1 and 2 were not tried. The solver tolerance stayed 1e-8 although the docs' advice says "very small tolerance"; the spec names only cone, condim, impratio, noslip.
2. B2v3's saved `argmax` holds only the closest point on the cylinder, not the tip-side point; L2 recomputed it. Should B2v3 save it?
3. The index's max-wrap point is on the top rim edge (all D), so L2's radial 5 mm move leaves its target on the top-cap plane (0.0 mm inside). The alternatives (inward normal, or a point lower on the side) were not run. The middle reaches almost the same angle on the side (66.55 against 66.57 deg).
4. The palm hull overlaps the cylinder at the B2v3 pose (-1.7 to -2.5 mm). The touching rule's 1 mm tolerance applies only to the swept digit's geoms, so the overlap is reported, not used. The b2v3 json has no `open_questions` block saying so.
5. Rise is measured at the end of the 1.5 s lift with no settle; R0 itself rises only 7.28 cm for a 10 cm palm lift with 7.8 mm slip, so part of the shortfall may be palm tracking lag under load rather than slip. Inference only; the palm's own rise is not stored.
6. L1's pre-grasp and retreat direction (the alternative: keep the legacy approach vector) was not run.
7. L3 was run only as the cumulative L1+L2+L3 on the mug; the full cumulative on the cylinder for the R0 guard was not run. The guard also runs only the right-hand condim change; the left hand stays condim 3.
8. The legacy check of the reach test excludes wall-clock fields; `m1_reach_test` now also writes `protocol.contact_model` and a `resources` block, which M1's file predates.
9. (Fixed by the main agent) `python -m sim.b2_capacity` used to default to `--version v2` and would have rewritten the v2 result
   files; `--version` is now required (v2: `--version v2`, v3: `--version v3`). The v2 files were never rewritten (mtime 21:38).
10. `sim/pick_place.py` was edited by two agents at once (B2v3: `radius_consistent_offset`, `PALM_SURFACE_GAP`, the assert; L1: `grasp_offset`, `GRASP_OFFSETS`). The full diffs of `sim/scene.py`, `sim/pick_place.py`, `sim/m1_reach_test.py`, `sim/m1_grasp_ladder.py`, `sim/b2_capacity.py` need one review pass by the main agent. (Main agent: scene.py diff reviewed; the checker's compiled-model diff shows only
   the intended condim / option changes under contact_model="grasp", and legacy R0 is identical to M1 trial by trial after every edit.)

#### Runtime and peak RSS per step (all CPU, from the saved files)
| step | wall | peak RSS |
|---|---|---|
| B2v3 (`b2v3_capacity.json`) | 1.7 s | 924 MB |
| L1 check (`l1_check.json`) | 2.3 s | 2989 MB |
| R0 (`runs/R0.json`) | 28.4 s | 1711 MB max per process, sum 6598 MB |
| L1 (`runs/L1.json`) | 28.1 s | 1796 MB max, sum 6636 MB |
| L2 IK (`l2_targets.json`) | 0.38 s in-script | 991 MB |
| R0 after L2 | 27.7 s | 1759 MB max, sum 6755 MB |
| L2 (`runs/L2.json`) | 19.9 s | 1712 MB max, sum 6802 MB |
| reach legacy check | 12.6 s | 1712 MB |
| R0 after L3 | 29.0 s | 1657 MB max, sum 6397 MB |
| reach L3 (with video, Mesa EGL) | 49.3 s | 1648 MB |
| L3 R0 guard | 37.8 s | 1719 MB max, sum 6570 MB |
| L3 (`runs/L3.json`) | 26.8 s | 1676 MB max, sum 6573 MB |
| ladder.json rebuild | 0.0 s | 64 MB |

Each parallel run also had a parent process of 63 MB. The 16 GB cap was never approached (at most about 8.2 GB, one 4-process run plus one reach test).

#### What failed
- Nothing reached 18/20. L1 and L2 stayed at 0/20, and L2 made slip worse (47 mm mean). L3 fixed slip (0 of 11 over 20 mm) but moved the failure to the close-phase touch gate (9/20). The best result is 11/20 (95 % CI 0.34-0.74), which does not support "the mug is grasped repeatably".
- The final 50-layout evaluation and the videos were not run (no rung >= 18/20). The only video is `reach_l3/reach_h074.mp4` (the L3 reach guard; gitignored).
- The causes of the L2 slip and of the L3 close-gate misses were not measured (no fingertip force or torque logs). The spec did not ask for them, so none was built.

---

## Track B final — L1 + L3 with HERO's closing targets (mug_v1)

**Why.** The v3 ladder was cumulative, so L3 was tested on top of L2, which had made the grip weaker (lift slip 47 mm vs 22 mm). This run tests L1 + L3 with HERO's closing targets. It is one pre-registered test, not a search. CPU only, hoi-sim. Everything is in `results/m1_grasp_fix/v3_final/` (`final.json`, `summary.md`, `runs/`, `videos.json`). Every number below was re-derived by an independent check from the per-trial rows in `runs/*.json`, with 0 mismatches.

**Profile.** `sim/pick_place.py` has `GRASP_PROFILES = {"legacy": {}, "mug_v1": dict(grasp_offset="radius", closing_targets="hero", contact_model="grasp")}`. This is exactly the spec's dict, and "legacy" stays the default. mug_v1 equals L1 (radius-consistent palm target, HERO's closing targets, legacy contact) plus L3's contact model. So the only difference from the v3 L1 rung is the contact model. It differs from the v3 L3 rung in the closing targets (HERO's instead of L2's IK targets).

**Instrumentation (behaviour unchanged).** `TableScene.step` stores `self.last_finger_tau = finger_tau.copy()` after the substep loop. This is the torque of the last substep after the squeeze rule, i.e. what went to `qfrc_applied`. It is a record only. `sim/m1_grasp_final.py` has `ForcePickPlace(ContactPickPlace)`, which records at two moments: (closed) the control step in which the right hand's closure first reaches 1.0; (lift) B3's lift frame. At each moment it records:
- the normal force per part (thumb = thumb_0/1/2, index = index_0/1, middle = middle_0/1, palm): the sum of `mj_contactForce` component 0 over contacts between that part's geoms and any cup geom;
- the 7 right finger-joint torques, with a saturation flag (|tau| >= 0.99 x effort limit);
- the 3 touch sensors.
All values are read after that step's last 1 ms substep. `sim/m1_grasp_ladder.py` was refactored so its machinery is reused (`run_parallel`, `process_resources`, `same_rows`, `make_task(task_cls)`). Legacy output keys are unchanged, and the ladder now writes `ladder.json` only into its own directory.

**Reach test: not rerun.** mug_v1's global contact options are exactly L3's, and L3 passed the reach test (`results/m1_grasp_fix/v3/reach_l3/`: no fall, pelvis z min 0.686 m, tilt max 12.8 deg, drift 0.086 m). The reach test uses the cylinder and only the contact model, so the grasp offset and the closing targets play no role in it. The compiled contact settings of run1, r0 and the v3 L3 run are identical (`compiled_contact_settings_equal_l3 = true` in `final.json`): elliptic cone, impratio 10, noslip_iterations 3, Newton with tolerance 1e-8, condim 4 on the right palm, thumb_2, index_1 and middle_1.

### Run 1 — B3's R3 (full mug from config, 0.40 kg, B3's trials 0-19 built with the legacy grasp, R3's handle yaws), mug_v1, 4 worker processes
Layouts were asserted trial by trial against B3's R3 rows (checked again by the independent check: 0 differences). **Result: 20/20, Wilson 95 % CI 0.839-1.000, no failures.**

| run | success | Wilson 95 % | failures | lift slip mean / median (mm); > 20 mm | rise mean / median (cm) | contact at lift |
|---|---|---|---|---|---|---|
| **Run 1 (mug_v1)** | 20/20 | 0.839-1.000 | none | 5.3 / 5.25 (n 20); 0 | 6.81 / 6.83 | 12/20 |
| B3 R3 (ladder.json) | 0/20 | 0.000-0.161 | dropped while moving 8, slip during lift 12 | 20.75 / 20.1; 10 | 5.11 / 4.98 | 13/20 |
| v3 L1 | 0/20 | 0.000-0.161 | dropped 5, slip 15 | 21.96 / 22.7; 12 | 4.99 / 4.96 | 14/20 |
| v3 L2 (L1+L2) | 0/20 | 0.000-0.161 | dropped 1, slip 19 | 47.06 / 30.3; 19 | 3.21 / 4.34 | 14/20 |
| v3 L3 (L1+L2+L3) | 11/20 | 0.342-0.742 | missed/knocked 9 | 6.8 / 6.6 (n 11); 0 | 6.87 / 6.86 | 5/20 |

Contact at lift in Run 1 (B3's definition: thumb AND (index or middle) AND palm): 12/20. Parts touching at the lift frame: thumb 20, index 15, middle 20, palm 12 (of 20).

### Forces, torques and touch at both moments (Run 1; 20/20 trials reached both moments)
Normal force on the cup, mean / median (min-max), N:

| part | closed | lift |
|---|---|---|
| thumb (3 links summed) | 68.61 / 74.81 (23.85-102.33) | 60.28 / 64.28 (23.59-96.72) |
| index | 19.29 / 22.90 (0-35.36) | 20.44 / 26.10 (0-38.24) |
| middle | 24.98 / 24.87 (20.18-31.63) | 29.15 / 28.40 (20.58-39.14) |
| palm | 18.22 / 6.49 (0-57.89) | 22.67 / 17.02 (0-60.07) |

Finger torques, mean / median N m; saturated = |tau| >= 0.99 x limit (limits: thumb_0 2.45, all others 1.4):

| joint | closed tau | lift tau | saturated, both moments (of 20) |
|---|---|---|---|
| thumb_0 | -0.11 / -0.10 | -0.09 / -0.06 | 0 |
| thumb_1 | -1.40 / -1.40 | -1.40 / -1.40 | 20 |
| thumb_2 | -1.14 / -1.19 | -1.04 / -1.05 | 0 |
| index_0 | 1.00 / 1.40 | 0.97 / 1.40 | 12 |
| index_1 | 0.64 / 0.84 | 0.52 / 0.51 | 0 |
| middle_0 | 1.40 / 1.40 | 1.40 / 1.40 | 20 |
| middle_1 | 0.97 / 0.94 | 1.02 / 1.02 | 0 |

Touch sensors, mean / median N: thumb_2 34.11 / 35.44 (closed), 31.60 / 32.86 (lift); index_1 17.16 / 21.53 and 18.45 / 24.93; middle_1 23.80 / 24.87 and 29.10 / 28.40. In every trial thumb_1 and middle_0 sit at their torque limit, and index_0 does in 12 of 20. The thumb force sums three thumb links, so it is larger than the thumb_2 touch value. A force of 0 with a contact present is possible (a contact inside the margin with no active constraint has zero force), so B3's "touching" and force > 0 can differ.

### Checks
- **R0 with mug_v1** (M1 cylinder, 0.15 kg, B3's R0 20 trials): **20/20** (Wilson 0.839-1.000), required 20/20, PASS. Lift slip 3.86 / 3.25 mm (none > 20 mm), rise 7.79 / 7.94 cm. All 20 rows are identical to v3's `L3_R0.json` in every field except wall time. On the 80 mm cylinder mug_v1 is the same physics as L3_R0 (`radius_consistent_offset(0.040)` equals `GRASP_OFFSET`, HERO's targets, "grasp" contact), so this also shows the force recording changed nothing.
- **Legacy R0 again** (`python -m sim.m1_grasp_ladder run --rung R0 --tag after_final --out results/m1_grasp_fix/v3_final`): 20/20, identical to M1 (`results/m1_pick` + `m1_pick_extended`, random 00-19: 20/20, 0 differing fields), and identical to `b3_runs/R0.json` in every field except wall time (0 differences). Lift slip 7.8 / 6.85 mm, rise 7.28 / 7.16 cm, contact at lift 12/20. Note: the spec's literal command (without `--out`) would have written into `results/m1_grasp_fix/v3/` and rewritten `v3/ladder.json`, which was off limits, so `--out` was added. The v3/ files all have mtimes before this run (last write 23:56) and no `ladder.json` exists in v3_final.

### 50 trials (decision rule: Run 1 20/20 >= 18/20, so run)
Trials 0-49 of `sim.m1_pick_mug.trials()` with legacy-grasp layouts, R3's handle yaws, full mug 0.40 kg, mug_v1. Layouts were asserted against `results/m1_pick_mug/runs_mass0.37.json` (my check: 0 differences). **44/50, Wilson 95 % CI 0.762-0.944.**
- Failures (M1 categories): missed/knocked cup 5 (trials 34, 38, 44, 45, 49), slip during lift 1 (trial 22). 
- Lift slip 4.97 / 4.9 mm (n 45), none > 20 mm. Rise 6.81 / 6.85 cm. Contact at lift 29/50 (thumb 45, index 32, middle 45, palm 29 of the 45 that reached the lift).
- Trials 0-19 reproduce Run 1 exactly (20/20 identical rows, in different worker chunks).
- Forces over the 45 trials that reached each moment, mean / median N: thumb 63.05 / 68.47 closed, 57.84 / 62.96 lift; index 17.10 / 21.86 and 18.18 / 23.58; middle 25.55 / 24.83 and 26.81 / 27.15; palm 19.09 / 6.10 and 23.79 / 16.58. Saturated of 45 (closed / lift): thumb_1 45 / 45, middle_0 45 / 45, index_0 27 / 26, others 0.
- **Trial 22** (cup 0.5104, -0.197 m, yaw 5.22 deg): slipped only 7 mm but rose 4.3 of 10 cm, so it failed the rise check. M1 files this under "slip during lift". Forces thumb / index / middle / palm, closed: 81.84 / 18.70 / 20.68 / 0.00 N; lift: 68.31 / 19.17 / 0.00 / 0.00 N. Torques at lift: -0.13, -1.40*, -0.98, 1.40*, 0.84, 1.40*, 0.95 (* saturated). Touch at lift: 29.32 / 19.17 / 0.00 N. At the lift frame the middle finger and palm carried no load.
- **Trials 34, 38, 44, 45, 49**: the cup moved 2.4-4.9 cm with 0 deg tilt before any finger closed. Both settles converged in every one (palm error <= 1.43 cm), and none reached either force moment. All five are in trials 20-49, and none in Run 1. M1's cylinder succeeded on the same five layouts (matched by cup position to `results/m1_pick_extended` random_24, 28, 34, 35, 39: 5/5). The cause was not measured. mug_v1 changes both the palm target (approach direction 39.95 instead of 35.54 deg in the palm frame) and the contact model, and which one matters was not tested. The cups are at x 0.447-0.469 m and y -0.328 to -0.304 m, but other trials in that y range succeeded, so there is no clean split by position.
- 2 videos (gitignored, rendered on Mesa EGL on the Intel iGPU): `results/m1_grasp_fix/v3_final/videos/success_trial00.mp4` and `failure_trial22.mp4`. Each was a fresh re-run of that trial, with the same outcome and identical result fields and `grasp_forces` as the batch (`videos.json`).

### Reading it, without overclaiming
- mug_v1 = 20/20 on R3, against 0/20 for L1 (the same profile with the legacy contact model) and 11/20 for L3 (on L2's IK targets). The single difference from L1 is the contact model, and from L3 the closing targets. But this is two runs in different configurations, not an ablation. I did not test the contact model alone on legacy HERO targets with the legacy offset, so I cannot say how much of the effect comes from the offset.
- No friction, gain, torque limit, mug geometry or mug mass was changed.
- The 'closed' moment is the first control step at closure 1.0. The finger targets slew at <= 2 rad/s, so the joints may still lag the closed pose at that step. The 0.5 s hold after it is not recorded.

### What failed
- 6 of 50 trials. 5 approach knocks (before any finger closed), 1 low lift (trial 22).
- The approach knocks are unexplained (not measured).
- The spec's R0 command had to be run with `--out` (see above).
- No other failures. The reach test was not rerun, for the reason given above.

### Files and cost
- Code: `sim/pick_place.py` (GRASP_PROFILES), `sim/scene.py` (`last_finger_tau`), `sim/m1_grasp_ladder.py` (refactor), `sim/m1_grasp_final.py` (new; `run --step run1|r0_mug_v1|fifty`, `videos`, `summarize`; the fifty step refuses to run unless Run 1 >= 18/20).
- Results: `results/m1_grasp_fix/v3_final/{final.json, summary.md, videos.json}`, `runs/{run1, r0_mug_v1, R0_after_final, fifty}.json`, `videos/{success_trial00, failure_trial22}.mp4`.
- Runtime and peak RSS (from `final.json` `resources`; CPU only, no GPU for physics):

| step | wall (s) | peak RSS (MB) |
|---|---|---|
| run1 | 37.0 | max process 1679, sum 6636, parent 147 |
| r0_mug_v1 | 35.9 | max 1733, sum 6847, parent 147 |
| legacy R0 after_final | 29.3 | max 1702, sum 6641, parent 63 |
| fifty | 89.1 | max 1805, sum 7069, parent 149 |
| videos | 24.8 | 1718 |
| summarize | 0.01 | 149 |

The parent RSS is 147 MB here against 63 MB in the ladder because `m1_grasp_final` sets MUJOCO_GL=egl before importing mujoco. This has no effect on physics (the r0 rows are identical to L3_R0). The libEGL warning about PCI id 10de:2d59 is Mesa skipping the NVIDIA card.

(Main agent: the README.md write the checker saw during this run was mine — a brief run-commands README the user asked for, committed separately in bfdefb0. The legacy R0 re-check was run with `--out results/m1_grasp_fix/v3_final` because the literal command would have written into the frozen `v3/` folder; the ladder now writes ladder.json only into its own folder.)

---

## M3 — kinematic retargeting

Date: 2026-10-31. Code: `retarget/{frames,grasp,warp,naive,run_m3}.py`, `tests/test_retarget_grasp.py`, plus one change in `sim/reference_builder.py` (`plan_path(..., return_fk=False)`). All outputs are in `results/m3_retarget/`. Kinematic only: the only physics is the 2 s setup hold that `PickPlace.run` does before its first move. Perception was not touched. CPU only.

Every number below comes from a file under `results/m3_retarget/`: `summary.md` / `summary.json` (made by `python -m retarget.run_m3 summarize` from `runs/*.json`, `verify.json`, `run_resources.json`, `r0_after_m3/runs/R0_after_m3.json`, `videos.json`), `human_values.json`, `palm_axes.json`, `test_retarget_grasp.log`. The checker re-derived the tables from the per-run JSONs, found no mismatch, and re-ran the unit test.

### What was done and why

Goal: turn my 4 recorded demos into dense 50 Hz right-palm paths and ask one question per path: can HERO's whole-body IK (the same `plan_path` / FK code M4 will use to build the reference clip) follow it? "Reachable" is a kinematic property. No physics success is claimed here.

1. **Frames and axes (`frames.py`).** The human table frame maps to the robot world by the identity rotation (forward x to x, left y to y, z up) plus a per-layout xy translation; human table plane z = 0 maps to the sim table top z = 0.74. Human cup positions are base centres, the sim cup pose is the body centre (cup_height / 2 higher), converted explicitly. Human palm frame columns are [x_f, y_h, n] (n points at the cup). Mapping `R_robot = R_human @ M`, M columns (e1, e3, -e2), det(M) = 1 asserted: robot x = x_f, robot y (palm normal) = n, robot z = x_f x n. Heading of the palm normal: `atan2(R[1,1], R[0,1])` for robot rotations, `atan2(R_h[1,2], R_h[0,2])` (the n column) for human ones.
2. **Palm-axes check (gate, passed)** (`palm_axes.json`): at the oracle grasp pose the mean of the right fingertip bodies has palm-x = +8.02 cm (needs > 0) and the cup axis has palm-y = +6.28 cm (needs > 0). At that pose the plant palm is 3.77 mm / 0.08 deg from the target and 0.0003 mm from the clip's FK palm.
3. **Grasp poses (`grasp.py`).** Oracle reference (mug_v1): R_ora = Grasp.hero_side_grasp rotation, o = radius_consistent_offset(r_mug) = (0.077, 0.0645, 0), phi_ora = 45.0 deg, h_frac_ora = 0.714. Human values at the grasp frame g: phi_h (heading of n), h_frac_h = palm height above table / cup height. Two variants, both with q = (cup axis x, y, table top + h_frac_h * h) and p = q - R * o:
   - **b_level**: R = R_z(phi_h - phi_ora) @ R_ora (my palm heading, the oracle's level wrist).
   - **b_tilt**: R = R_h(g) @ M (my full wrist orientation).
   Headings equal phi_h to 1e-9 by assertion. Why two: they differ only in wrist tilt, so any reachability difference is attributable to tilt.
4. **Unit test (passed)** (`test_retarget_grasp.log`): b_level at phi_h = phi_ora and h_frac_h = h_frac_ora reproduces the oracle grasp, max |p - p_ora| = 0 m and max |R - R_ora| = 0 (tolerance 1e-9).
5. **Time scale k (my design choice per the spec).** v_h = 95th percentile of my smoothed palm speed over the approach and transport segments; v_o = the larger oracle min-jerk peak speed (pregrasp leg 0.244 m/s, transport leg 0.15 m/s) at the recorded layout with mug_v1; k = max(1, v_h / v_o). Human times are multiplied by k. Reason: my hand is faster than the oracle's, so the same path would otherwise demand faster motion than the robot's reference clips. v_o = 0.244 m/s for every clip.
6. **Dense paths (`warp.py`, `naive.py`).**
   - Approach segment a0..g, where a0 = first frame of the last contiguous run ending at g in which palm_pos_table is finite and z >= 0.02 m (first tracked frames are below the table, an entry artefact). a0 z is 0.022 to 0.068 m.
   - Prelude (robot-specific, the robot starts with arms at its sides): standing palm to (p_s.x, p_s.y, z_clear) over 1.2 s, ending at R_G, exactly like the oracle's first pregrasp waypoint. z_clear = max(clearance height, G_z) = 0.9428 m for 4e103843db.
   - **b (object-aware)**, my design choices: approach `p(s) = p~(s) + (1-w)(p_r0 - p~(0)) + w (G - p~(1))`, w = 10s^3 - 15s^4 + 6s^5, then clamp z >= table top + 0.03 m (count recorded); orientation HELD at R_G for the whole approach (the human wrist rotation during approach is not transferred); a 0.5 s hold (`i_pause`) at the first sample whose remaining arc length to G is <= 0.08 m (M4 replaces it with the oracle's closed-loop settle). Close: 1.75 s hold at G, closure 0 to 1 over the first 1.25 s. Transport: `p(s) = p~(s) + (1-s)(G - p~(0)) + s(P - p~(1))`, P = G + (coaster_xy - cup_xy, COASTER_THICKNESS + PLACE_CLEARANCE), orientation held at R_G, and if the bump above the straight G to P line exceeds LIFT_HEIGHT (0.10 m) it is scaled down uniformly (bump scale). Release and retreat as the oracle does: closure 1 to 0 over 0.6 s, 0.6 s hold, retreat -0.10 u + (0, 0, 0.05) over 1.5 s, 1.0 s hold.
   - **a (naive)**: mapped human palm position and R_h(t) @ M from a0 to t_release, time-scaled by k, with the translation fixed at the recorded layout. No warp, clamp or pause. A 1.0 s min-jerk move to the mapped pose at a0 follows the prelude. Closure ramps 0 to 1 over 1.25 s from the mapped t_contact and 1 to 0 over 0.6 s at the mapped t_release.
7. **Layouts and runs (`run_m3.py`).** Recorded layout: cup xy = PALM_BOX centre - legacy palm offset xy; coaster = cup + my (coaster - cup) offset; handle yaw = cup_init_yaw if valid. Random layouts: trials 0 to 9 of `sim.m1_pick_mug.trials()` (legacy grasp), handle yaw = trial yaw + (phi_h - phi_ora). Runs: b_level and b_tilt x 4 clips x (recorded + 10) = 88, plus a_naive x 4 clips x recorded = 4 (its path ignores the layout, so its result holds for every layout). Total 92.
8. **Per run**: set up exactly as `PickPlace.run` does before its first move, make ONE `plan_path(base="plan", base_goal=_goal(G, R_G), return_fk=True)` call, assert path sample 0 equals palm_start (< 1 mm; observed max 0.090 mm), report FK error against the path, joint-limit hits (any of 14 arm joints within 0.01 rad of its limit), and the hand-mug and hand-handle signed distances at i_G (open hand, `mj_geomDistance`). The clip-to-plant joint mapping is checked at i_G: 92/92 within 1 mm (max 0.0005 mm).
9. **Reachable** = FK error at G and at P both < 1.0 cm and < 10 deg AND path max position error < 2 cm. Written down before the run and never tuned.
10. **Checks (`verify.json`, passed).** `plan_path` default output byte-identical with and without `return_fk` (all clip arrays plus info); the setup is repeatable; the R0 rung re-run after the `plan_path` change gives 20/20 successes, identical to M1 in 20/20 trials and to B3 R0 in 20/20 (`r0_after_m3/runs/R0_after_m3.json`; written to a separate `--out` so the frozen M1 results stay untouched). verify, run and videos ran on identical code (SHA-256 of the 8 code files equal).

### Results

Per-clip human values (`human_values.json`; phi_ora = 45.0 deg and h_frac_ora = 0.714 for all):

| clip | phi_h (deg) | palm-point azimuth (info) | h_frac_h | k | a0 time (s) | clamped, recorded (b_level / b_tilt) |
|---|---|---|---|---|---|---|
| 26d4118b0c | 81.9 | -135.3 | 0.554 | 1.588 | 1.267 | 0 / 0 |
| 4e103843db | 85.8 | -122.1 | 0.480 | 1.091 | 1.801 | 0 / 12 |
| 49aff04f7a | 81.7 | -134.9 | 0.692 | 1.739 | 1.884 | 0 / 0 |
| e8825dea53 | 61.7 | -137.0 | 0.700 | 1.520 | 0.700 | 0 / 0 |

Clamped frames: 0 for b_level on every layout; b_tilt 12 for 4e103843db on every layout, 0 for the others. The bump scale is 1.000 everywhere except e8825dea53 (0.888, both b variants).

Reachability (runs reachable / runs):

| clip | a_naive | b_level | b_tilt |
|---|---|---|---|
| 26d4118b0c | 0/1 | 0/11 | 0/11 |
| 4e103843db | 0/1 | 0/11 | 0/11 |
| 49aff04f7a | 0/1 | 0/11 | 0/11 |
| e8825dea53 | 0/1 | 10/11 | 0/11 |
| all clips | 0/4 | 10/44 | 0/44 |

Which part of the rule failed (runs passing / runs):

| clip | variant | at G | at P | path max < 2 cm |
|---|---|---|---|---|
| 26d4118b0c | b_level | 0/11 | 8/11 | 0/11 |
| 26d4118b0c | b_tilt | 0/11 | 3/11 | 0/11 |
| 4e103843db | b_level | 1/11 | 6/11 | 0/11 |
| 4e103843db | b_tilt | 0/11 | 0/11 | 0/11 |
| 49aff04f7a | b_level | 5/11 | 9/11 | 0/11 |
| 49aff04f7a | b_tilt | 0/11 | 6/11 | 0/11 |
| e8825dea53 | b_level | 10/11 | 11/11 | 11/11 |
| e8825dea53 | b_tilt | 0/11 | 0/11 | 0/11 |
| all clips | a_naive | 0/4 | 0/4 | 0/4 |

Aggregates over all runs per variant: median FK position error at G / at P (cm): b_level 1.16 / 0.48, b_tilt 3.19 / 1.72, a_naive 2.67 / 2.07. Largest orientation error: b_level 0.7 deg, b_tilt 1.7 deg, a_naive 7.9 deg. QP failures: 0 in all 92 runs. Joint-limit hits: 0 in all 92 runs (minimum margin to a limit 0.032 rad, so the 0.01 rad rule is never close to triggering). Where the path maximum sits: b_level always in the robot-specific prelude (44/44, 1.90 to 2.86 cm), b_tilt always in the approach (44/44, 2.78 to 6.32 cm), a_naive always in the replay (4/4, 3.60 to 9.86 cm).

Recorded layout, FK position error max / at G / at P (cm):

| clip | a_naive | b_level | b_tilt |
|---|---|---|---|
| 26d4118b0c | 3.60 / 3.15 / 2.04 | 2.82 / 1.40 / 1.27 | 3.06 / 2.86 / 1.83 |
| 4e103843db | 5.88 / 4.83 / 4.04 | 2.86 / 1.31 / 1.70 | 6.20 / 5.83 / 4.32 |
| 49aff04f7a | 6.07 / 2.18 / 2.09 | 2.82 / 0.97 / 1.14 | 2.89 / 2.62 / 1.45 |
| e8825dea53 | 9.86 / 1.46 / 1.99 | 1.90 / 0.81 / 0.77 | 3.77 / 3.38 / 1.96 |

Only e8825dea53 b_level is reachable on its recorded layout.

Hand-mug and hand-handle distance at G (mm, open hand, negative = overlap; min over layouts for each variant over the 4 clips; per clip in `summary.md`):

| variant | hand-mug | hand-handle |
|---|---|---|
| b_level | -11.7 to -3.3 | 42.5 to 84.8 |
| b_tilt | -12.6 to +1.4 | 44.6 to 90.6 |
| a_naive | -2.7 to +10.9 | 74.8 to 89.1 |

For reference, the oracle's own exact planned pose gives -2.4 mm hand-mug and 79.7 mm hand-handle (`results/m1_grasp_fix/v3/l1_check.json`). Nearly every b run already overlaps the mug with an open hand (information for M4's settle; no M3 rule uses it).

Runtime and peak RSS (CPU only, from `run_resources.json` and `summary.md`):

| step | wall | peak RSS |
|---|---|---|
| verify | 4.8 s | 1659 MB |
| run (92 runs, 4 processes) | 84.8 s | max process 1668 MB, sum 6492 MB, parent 148 MB |
| R0 after_m3 (4 processes) | 33.9 s | max process 1674 MB, sum 6495 MB |
| videos | 31.3 s | 2185 MB |
| summarize | 0.02 s | 151 MB |

Per-run worker peak RSS in the run JSONs is a worker high-water mark so far, not an isolated per-run peak. The unit-test log's 'wall 0.00 s' times only the test body (imports are excluded).

Videos (clip 4e103843db, recorded layout): `results/m3_retarget/videos/4e103843db__{a_naive,b_level,b_tilt}__recorded.mp4` (gitignored) and the b_level GIF `results/m3_retarget/4e103843db__b_level__recorded.gif` (2.82 MB, under the 5 MB limit; `videos.json`). The robot comes from the clip's joints, fingers blend open to closed by closure, and the mug moves rigidly with the palm from the end of the close hold to the release (b: frames 281 to 403; a_naive: 232 to 406). The videos are re-plans of the saved runs and differ from the saved runs in nothing.

### What failed (stated plainly)

- Under the rule as written, 10 of 44 b_level runs and 0 of 44 b_tilt runs and 0 of 4 a_naive runs are reachable. Only clip e8825dea53 (phi_h = 61.7 deg, nearest to the oracle's 45 deg) passes with b_level; the other three clips (phi_h 81.7 to 85.8 deg) pass for none of their 44 runs.
- b_tilt never passes: taking my wrist tilt raises the error at G, with the worst at 4e103843db (5.8 to 6.0 cm, with 12 clamped frames). b_level keeps the oracle's level wrist and has smaller errors at G (per-run values in results/m3_retarget/summary.md).
- a_naive fails everywhere; its errors peak in the replay (3.60 to 9.86 cm), which fits a fixed, layout-blind human path.
- For b_level the failures come from the path-max rule, driven by the robot-specific prelude (arms-at-sides to over the table), not by my demo. The oracle's own pregrasp plan has a 1.77 cm IK path residual. By a one-liner on `summary.json` (not saved as a number): of the 77 b runs failing the path < 2 cm rule, 26 (all b_level) are under 2 cm everywhere after the prelude, and 3 of those (49aff04f7a b_level trial01, trial02, trial05) also pass at G and at P. The rule was left unchanged.
- e8825dea53 b_level trial08 fails only at G: 1.024 cm against the 1.0 cm limit. Reported as is, not tuned.
- On all 4 clips the coaster's nominal place palm lies outside PALM_BOX with either sign convention (`coaster_place_palm_in_palm_box` False in the recorded runs).
- Nothing was run in physics. These results say nothing about whether the grasp closes or lifts. That is M4.

### Adaptations (differences from the task text, all in the integration notes)

- The R0 re-check was run with `--out results/m3_retarget/r0_after_m3 --overwrite`, because the literal command without `--out` would write into the frozen `results/m1_grasp_fix/v3/` and rewrite its `ladder.json`.
- `summarize` is run after `videos` as well, since it embeds `videos.json`.
- `retarget.frames` and `retarget.warp` were re-run in the final pass so that every M3 file comes from one run of the current code. Their values equal the earlier ones.

### Open questions (literal reading implemented in each case)

1. Should the path-max < 2 cm rule exclude the robot-specific prelude? Implemented as written. b_level's path maximum is in the prelude in 44/44 runs.
2. Place palm: the spec writes coaster - legacy palm_offset[:2], but `sample_layouts` implies coaster + offset. Implemented literally; all 4 clips are outside PALM_BOX either way.
3. Naive prelude orientation: there is no R_G in (a), so the mapped pose at g, R_h(g) @ M, is used. The alternative is the pose at a0.
4. Naive release: the 1 to 0 ramp starts at the mapped t_release (plus 30 appended frames); the alternative is a ramp that ends there.
5. Naive video: a_naive has no close hold, so the mug attaches at the first frame with closure = 1 (232). Is that intended?
6. Velocity jumps at segment joins (spec followed literally): the approach starts at my hand-entry speed at a0 (0.59 to 0.97 m/s after k) right after the prelude ends at rest; transport starts at my lift speed after the close hold. Should the warp match velocities, or should a0 skip the fast entry frames?
7. Joint-limit hits are 0 in all 92 runs, essentially by construction (HERO's QP keeps joints at least 0.032 rad inside the URDF range). Is the per-run minimum margin (recorded) enough, or should the check use the QP's limits?
8. Hand-mug overlap at G (b_level -11.7 to -3.3 mm, b_tilt -12.6 to +1.4 mm; the oracle's own exact pose is -2.4 mm): information for M4's settle.
9. Cosmetic: `frames.verify_palm_axes` sets the robot with `ref.root_pos_from_joint_pos` / `root_quat_from_joint_pos`, `run_m3` with `ref.root_pos_w` / `root_quat_w` (HERO's `rollout_clip`). Both pass their palm-match checks (0.0003 mm and at most 0.0005 mm).

(Main agent: the review found only minor issues — docstring wording in warp.py / frames.py, a positional `L.insert` in run_m3.summary_md, the unit-test log timing only the test body, and runtime/RSS of palm axes / human values living in their own files. None changes a number; not fixed, to keep M3 lean. Numbers from the integration report that are not in a results file (per-variant FK-error medians, counts of b_level runs under 2 cm after the prelude) are left out of this log.)


---

## M4 — physical execution (HERO tracks the retargeted paths)

Date: 2026-10-31. Profile mug_v1 everywhere (mug from config/objects.yaml, 0.40 kg). Perception frozen, M3 outputs untouched. New outputs: `results/m3_retarget/v2/` (Step 0, 1, oracle rows, Step 3 on the oracle) and `results/m4_execute/` (Steps 2-5). CPU only, no GPU, no videos. Code, all new files: `retarget/paths_v2.py`, `retarget/collision.py`, `retarget/run_m3_v2.py`, `retarget/m4_summary.py`, `sim/m4_execute.py`. No existing file was edited, so M3's 8 code files still match the SHA-256 in `results/m3_retarget/run_resources.json`.

Commands (from the repo root, prefix `env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m`):
- `retarget.run_m3_v2 paths | oracle | summarize` (Step 0 and 1; `paths` and `oracle` refuse to overwrite without `--overwrite`)
- `sim.m4_execute run --workers 8 --out results/m4_execute` (the 600 runs) and `sim.m4_execute run --variants b_level --layouts recorded_v2 --workers 4 --out results/m4_execute` (4 runs)
- `retarget.m4_summary` (summary.md / summary.json / 3 PNGs; reads the 604 rows and fifty.json; simulates nothing)
- `sim.m4_execute rerun --row <row.json> --report <file>`, `sim.m4_execute check-oracle --trials 5 --out <dir>`, `sim.m1_grasp_ladder run --rung R0 --tag after_m4 --out <dir>`

### Step 0 — two corrections to M3 (results/m3_retarget/v2/summary.md)

**1. Place-palm sign.** The nominal oracle palm for a cup at xy c is c + the legacy palm_offset[:2] (the mapping `sample_layouts` uses; legacy offset (-0.0933, 0.0156) m). M3's spec wrote c - offset for the place-palm flag. With the correct sign, place palm = coaster + offset, and it is still outside PALM_BOX (x 0.35-0.45, y -0.32..-0.12) for all four recorded layouts. It is outside on y only; x is inside for all four. The grasp palm is at the box centre (0.400, -0.220) by construction. M3's conclusion (all four outside) stands; only the reason changes.

| clip | place palm xy (coaster + offset) | outside PALM_BOX by x / y (cm) | M3's flag (coaster - offset) |
|---|---|---|---|
| 26d4118b0c | (0.404, -0.364) | 0.0 / 4.4 | (0.591, -0.395) out |
| 4e103843db | (0.406, -0.364) | 0.0 / 4.4 | (0.593, -0.395) out |
| 49aff04f7a | (0.398, -0.357) | 0.0 / 3.7 | (0.584, -0.388) out |
| e8825dea53 | (0.374, -0.388) | 0.0 / 6.8 | (0.561, -0.419) out |

Side note (checker, scratch `tmp/m4/check/check_m3v2.out`, not in results/): the real mug_v1 oracle's own palm offset is (-0.1001, 0.0088) m, not the legacy one. With it the place palms are still outside on y only, by 5.1 / 5.1 / 4.3 / 7.4 cm. The conclusion does not depend on which offset is used.

**2. The oracle under M3's rule.** `paths_v2.oracle_path` builds the oracle's nominal waypoints through `min_jerk_path` with M3's phase names (prelude 1.2 s, pre-grasp 1.8 s, approach to G 2.0 s, close 1.75 s with a 1.25 s closure ramp, lift +0.10 m in 1.5 s, transport max(1.5, d/0.08) s, lower 1.5 s, release 0.6 + 0.6 s, retreat 1.5 s, hold 1.0 s). It ran through `run_m3.run_one` unchanged (only the module-level `build_path` was swapped, via a context manager) on M3's 10 random layouts (trials 0-9) with each trial's own handle yaw.
- Reachable 10/10 (at G 10/10, at P 10/10, path max < 2 cm 10/10). Path max is 1.69 cm in every row, always in the prelude. Error at G 0.68-0.84 cm, at P 0.12-0.50 cm, orientation max 0.4 deg. Hand-mug distance at G -2.4 to -3.7 mm (the l1_check reference is -2.4 mm).
- So the robot-specific prelude does not fail M3's rule on its own. The oracle's prelude stays at 1.69 cm; b_level's reaches 1.90-2.86 cm (it turns the hand toward my heading phi_h, 62-86 deg).

Reachability under M3's rule (at G / at P / path max < 2 cm / all three), with the oracle rows added:

| clip | b_level | b_tilt | a_naive (recorded only) |
|---|---|---|---|
| 26d4118b0c | 0 / 8 / 0 / **0/11** | 0 / 3 / 0 / **0/11** | 0/1 |
| 4e103843db | 1 / 6 / 0 / **0/11** | 0 / 0 / 0 / **0/11** | 0/1 |
| 49aff04f7a | 5 / 9 / 0 / **0/11** | 0 / 6 / 0 / **0/11** | 0/1 |
| e8825dea53 | 10 / 11 / 11 / **10/11** | 0 / 0 / 0 / **0/11** | 0/1 |
| oracle (trials 0-9) | 10/10 at G, 10/10 at P, 10/10 path, **10/10** | | |

Largest FK position error outside the prelude (information only, from each run's `max_by_phase`; min / median / max over runs; "below 2 cm" is a count, not a rule):

| variant | runs | max error after the prelude (cm) | below 2 cm | prelude max (cm) |
|---|---|---|---|---|
| a_naive | 4 | 3.60 / 5.97 / 9.86 | 0/4 | 2.67 / 2.83 / 3.26 |
| b_level | 44 | 1.12 / 1.64 / 2.32 | 37/44 | 1.90 / 2.82 / 2.86 |
| b_tilt | 44 | 2.78 / 3.48 / 6.32 | 0/44 | 2.67 / 2.83 / 3.26 |
| oracle | 10 | 0.74 / 0.81 / 0.85 | 10/10 | 1.69 / 1.69 / 1.69 |

b_level per clip below 2 cm after the prelude: 26d4118b0c 10/11, 4e103843db 5/11, 49aff04f7a 11/11, e8825dea53 11/11.

### Step 1 — path timing v2 (retarget/paths_v2.py; npz in results/m3_retarget/v2/paths/, 92 M3 runs + 10 oracle rows)

- **How it is built.** v2 calls M3's own builders (`warp.object_aware_path`, `naive.naive_path`, unchanged) and only cuts and re-times their output, so geometry, warps and k are M3's by construction. M3's 92 paths rebuilt from the saved standing palms are byte-identical to M3's npz (92/92; the checker re-ran this independently, plus the v2 public entry against the saved v2 npz 92/92 and `oracle_path` against the saved oracle npz 10/10).
- **(b) b_level / b_tilt.** The 0.5 s pause hold is removed; i_pause becomes a split point. The approach is split at i_pause and the transport at its highest palm sample (i_peak). Each of approach 1, approach 2, transport 1 and transport 2 is re-timed rest to rest: D = D0 + 0.4 s, speed factor smoothstep(t/0.4) at both ends, sigma(D) = D0 exactly. Prelude, close, release and retreat are M3's. Each b path is 55 frames longer (+4 x 20 ramp frames, -25 pause frames).
- **(a) naive.** The replay (a0 .. t_release) is re-timed as one segment (+20 frames). The closure keeps M3's rule, with contact, G and P mapped onto the new timing.
- **What v2 removes.** M3 approach 1 started at my hand-entry speed of 0.59-0.97 m/s; transport 1 at 0.11-0.26 m/s (from M3's npz, checker recount).
- **Asserts per segment** (every saved path passed): the first and last 50 Hz step each move < 1 mm (observed max 0.047 mm first, 0.022 mm last); segment end positions equal M3's to 1e-9 (observed 0). The closed-form sigma matches a numeric integral to 1.5e-11 s. The checker's own numeric-integral re-timing of every segment of all 92 runs matches the saved npz to 2.2e-10 m (the limit of its quadrature).
- **Dry run.** `v2_path` built 600/600 on M4's layouts (trials 0-49 x 4 clips x 3 variants); shortest segment D0 = 0.48 s (4e103843db transport 1), above T_R = 0.4 s.

### Step 2 — executor (sim/m4_execute.py)

`Execute(PickPlace)`, profile mug_v1, mug mass passed explicitly and asserted equal to config (0.40 kg). Setup copies `PickPlace.run`'s lines: reset, standing reference, policy reset, reset step, 2 s stand. The path is built after the stand because it starts at the standing palm.

**follow(phase, pos, quat, base, base_goal).** Plays a dense 50 Hz path slice: one `planner.plan_path`, then one `_tick` per frame. Frame 0 is bent onto the live IK palm (p0, q0) = `planner.palm_start(state)`: pos_j + (1 - w_j)(p0 - pos_0) and quat_j = slerp(q0, quat_j, w_j), with w = 10s^3 - 15s^4 + 6s^5 and s = j/(N-1). The end frame is unchanged (w = 1). Closure is not changed by follow, except in naive. Checked by instrumenting plan_path / palm_start (reviewer probes): frame 0 equals palm_start exactly and the interior equals the formula.

**b_level / b_tilt phases** (every check is the oracle's own inherited method):
1. follow prelude + approach 1 (base "plan", base_goal = `_goal(G, R_G)`); `settle_or_fail("settle_pregrasp", pause pose, max 5 s)`.
2. follow approach 2 (base "current"); `settle_or_fail("settle_grasp", G, max 3 s)`; then `check_cup_undisturbed`.
3. `close_hand()`, then `check_grasp("close")`.
4. Record in_hand_closed; follow transport 1 (base "plan", base_goal = its last pose). At the peak: rise = cup z rise since the close, slip = |cup - palm - in_hand_closed|; fail as "lift" if slip > LIFT_SLIP_FAIL_M or rise < 0.5 x (peak palm z - G z); `check_grasp("lift")`.
5. At the peak, measure cup_in_hand and set P' = cup_place - cup_in_hand (cup_place as the oracle defines it). Transport 2's end is re-warped to P': x'(s) = x(s) + s (P' - x_end), with s = sigma(t_j)/D0 of that segment (`paths_v2.sigma`). follow it (base "current"); `check_in_hand("transport")`, `settle_or_fail("settle_place", P', max 3 s)`, `check_in_hand("lower")`.
6. Open, retreat and `evaluate()`, exactly as the oracle.

**naive:** one follow of the whole path (base "plan", base_goal = the path's info G); `scene.set_hand_closure("right", closure[j])` on every tick; no settles, no grasp checks, no place correction. Then retreat -0.10 u + (0, 0, 0.05) in 1.5 s (u = horizontal unit vector palm to cup at that moment), hold 1.0 s, `evaluate()`.

**Privileged place correction, for the README, for BOTH the oracle and the b variants.** The oracle's place target (sim/pick_place.py) and the b variants' P' both read the cup's true pose in the hand from the simulator at the transport peak. This is simulator state, not something perception provides. Every b row records this in `detail.place_correction.uses`. Checker recount from the rows (scratch): over the 63 successful b runs (62 trial runs + 1 recorded_v2) the correction moved the place pose by 17.7-38.1 mm (median 25.2).

Layouts: trial00-49 use M3's handle-yaw rule: trial yaw + (phi_h - phi_ora) per clip (turn 36.94 / 40.85 / 36.68 / 16.69 deg for 26d4118b0c / 4e103843db / 49aff04f7a / e8825dea53); trials 0-9 are asserted equal to M3's layouts. recorded_v2 layout: cup = PALM_BOX centre - legacy offset - Delta/2, coaster = cup + Delta (Delta = the clip's coaster offset), handle yaw = M3's recorded yaw (4.1 / 81.79 / 72.04 / 61.69 deg). Both nominal palms are then centred on the box (checked by the checker for all four clips).

### Step 3 — collision diagnostic (retarget/collision.py; kinematic, information only, never a gate)

M3's harness single-plan IK clip of the path; frames from the prelude's last frame to the last frame with more than 1 cm of path arc length left to G; per frame the robot is set to the clip frame, both hands open (scene.reset), mug at its layout pose; minimum `mj_geomDistance` from the 8 right-hand geoms to the mug body and its 3 handle capsules. It runs after the physics run, so it cannot change an outcome. IK-to-plant mapping check passed on all 404 b rows and all 10 oracle rows. Negative = overlap.

| variant | outcome group | n | min | p25 | median | p75 | max | overlapping (< 0) |
|---|---|---|---|---|---|---|---|---|
| b_level | success | 23 | -21.9 | -16.3 | -2.1 | -0.8 | 0.6 | 21 |
| b_level | approach knock | 8 | -40.0 | -24.0 | -10.9 | -5.8 | -1.8 | 8 |
| b_level | reach (a settle gave up) | 169 | -58.4 | -53.7 | -37.4 | -25.2 | -0.2 | 169 |
| b_tilt | success | 39 | -29.8 | -15.9 | -12.3 | -8.6 | -6.1 | 39 |
| b_tilt | approach knock | 4 | -28.4 | -28.3 | -27.7 | -27.1 | -26.7 | 4 |
| b_tilt | reach (a settle gave up) | 155 | -41.1 | -29.8 | -24.2 | -14.4 | 3.4 | 141 |
| b_tilt | dropped while moving | 2 | -9.0 | -7.0 | -5.0 | -3.0 | -1.0 | 2 |
| oracle (trials 0-9, all successes) | success | 10 | 5.2 | 5.5 | 5.6 | 6.0 | 6.1 | 0 |

Oracle rows: always the palm link against the body, at the window's last frame (1.006 cm of arc before G); to the handle 58.8-94.5 mm. The oracle's six failed trials (22, 34, 38, 44, 45, 49) were not measured, so the oracle has no knock values. Reading: the b hand is already overlapping the mug body 1 cm before G in 21/23 b_level and 39/39 b_tilt successes, while the oracle never is. The window stops 1 cm before G, so a late overlap can be the intended grasp contact as well as a knock; the number does not separate successes from knocks.

### Step 4/5 — runs and results (results/m4_execute/summary.md, summary.json; 604 rows in runs/)

600 runs = b_level, b_tilt, naive x 4 clips x trials 0-49, plus 4 recorded_v2 b_level runs. All 604 rows present, 0 error rows, every row paths v2, profile mug_v1, 0.40 kg. The oracle is Track B final's 50 mug_v1 trials (`results/m1_grasp_fix/v3_final/runs/fifty.json`), not rerun. The cup and coaster xy of every variant row equal the oracle's trial row.

Success over 50 layouts (Wilson 95 % CI):

| run | view | b_level | b_tilt | naive |
|---|---|---|---|---|
| 26d4118b0c | top-down | 0/50 (0.000-0.071) | 4/50 (0.032-0.188) | 0/50 (0.000-0.071) |
| 4e103843db | 90 deg | 0/50 (0.000-0.071) | 0/50 (0.000-0.071) | 0/50 (0.000-0.071) |
| 49aff04f7a | 45 deg | 0/50 (0.000-0.071) | 18/50 (0.241-0.499) | 0/50 (0.000-0.071) |
| e8825dea53 | 45 deg | 23/50 (0.330-0.596) | 17/50 (0.224-0.478) | 0/50 (0.000-0.071) |
| 4-clip pool | - | 23/200 (0.078-0.167) | 39/200 (0.146-0.255) | 0/200 (0.000-0.019) |
| oracle (Track B final) | - | 44/50 (0.762-0.944) | | |

Total 62/600 trial-run successes, plus 1/4 on recorded_v2. Pooled CIs treat 200 runs as independent although the 4 clips reuse the same 50 layouts and are 4 separate executions (n = 4); quote the per-clip CIs next to any pooled number. Viewpoint and execution vary together, so no clip difference can be attributed to the viewpoint.

Failure categories (M1's plus **reach** = "palm could not reach the target", a failed settle):
- oracle: missed/knocked cup 5 (trials 34, 38, 44, 45, 49), slip during lift 1 (trial 22).
- b_level (pool): reach 169, missed/knocked cup 8. b_tilt (pool): reach 155, missed/knocked cup 4, dropped while moving 2. naive: all 200 "released but not on the coaster".
- Per clip: 26d4118b0c b_level reach 50; b_tilt reach 45, knocked 1. 4e103843db b_level reach 50; b_tilt reach 50. 49aff04f7a b_level reach 46, knocked 4; b_tilt reach 29, knocked 3. e8825dea53 b_level reach 23, knocked 4; b_tilt reach 31, dropped 2.
- Across all 604 rows no b run failed at the close, the lift or the lower; the failed phases are reach 327 (incl. 3 recorded_v2), approach 12, transport 2.
- Every approach knock has failed phase approach.

Which settle gave up (324 reach failures of the b trial runs; the palm was 3.01-8.57 cm from the target at give-up). Total: settle_pregrasp 73, settle_grasp 240, settle_place 11. The 3 recorded_v2 reach failures are all settle_grasp.

| clip | variant | settle_pregrasp | settle_grasp | settle_place | total |
|---|---|---|---|---|---|
| 26d4118b0c | b_level | 21 | 29 | 0 | 50 |
| 26d4118b0c | b_tilt | 2 | 40 | 3 | 45 |
| 4e103843db | b_level | 29 | 21 | 0 | 50 |
| 4e103843db | b_tilt | 21 | 28 | 1 | 50 |
| 49aff04f7a | b_level | 0 | 46 | 0 | 46 |
| 49aff04f7a | b_tilt | 0 | 22 | 7 | 29 |
| e8825dea53 | b_level | 0 | 23 | 0 | 23 |
| e8825dea53 | b_tilt | 0 | 31 | 0 | 31 |

Paired with the oracle (same 50 layouts; both succeed / oracle only / variant only / both fail; trial numbers in summary.json `paired`):

| clip | b_level | b_tilt | naive |
|---|---|---|---|
| 26d4118b0c | 0 / 44 / 0 / 6 | 4 / 40 / 0 / 6 | 0 / 44 / 0 / 6 |
| 4e103843db | 0 / 44 / 0 / 6 | 0 / 44 / 0 / 6 | 0 / 44 / 0 / 6 |
| 49aff04f7a | 0 / 44 / 0 / 6 | 17 / 27 / 1 / 5 (variant only: trial 22) | 0 / 44 / 0 / 6 |
| e8825dea53 | 22 / 22 / 1 / 5 (variant only: trial 22) | 14 / 30 / 3 / 3 (variant only: trials 38, 44, 49) | 0 / 44 / 0 / 6 |

**Reading the reach failures (checker, scratch `tmp/m4/check/check_extra.out`, not in results/m4_execute/summary).** The executor runs `settle_grasp` before `check_cup_undisturbed` (the oracle's and the spec's order), so a run where the open hand pushed the mug and the palm then ended more than 3 cm from G is filed as reach, not as a knock. From the rows, `detail.cup_final` against `cup_xy` (2 cm xy is check_cup_undisturbed's own threshold; settle_place failures excluded because the cup is in the hand): the cup had moved more than 2 cm in 124 of the 169 b_level reach failures (settle_pregrasp 6/50, settle_grasp 118/119) and in 88 of the 144 b_tilt settle_pregrasp + settle_grasp reach failures (20/23 and 68/121). So the 12 literal "approach knocks" understate how often the hand hit the mug. Together with the negative collision distances above this points at the approach path sweeping the open hand through the mug. This is a description of the rows; no cause was tested.

Observation (checker reading, two saved tables): M3's reachability rule (G and P < 1 cm, path max < 2 cm) is stricter than the executor's 3 cm settle tolerance. e8825dea53 b_level is the only kinematically reachable group (10/11) and the best b_level group in physics (23/50); but b_tilt has 0/44 kinematically reachable and 39/200 physics successes, so the kinematic table is not a clean predictor of the physics outcome.

**Failure maps** (`failure_map_b_level.png`, `failure_map_b_tilt.png`, `failure_map_naive.png`; cup xy coloured by outcome, one panel per clip plus the oracle for reference; dashed rectangle = the PALM_BOX cup region = PALM_BOX minus the legacy palm offset, x 0.4433-0.5433, y -0.3356..-0.1356). Cluster description, **descriptive, found after looking** (the splits y < -0.25 m "outer" and x >= 0.52 m "far" were chosen after looking at the maps; not a model; every number below was re-derived from the rows):
- Layout: the 50 cups form two groups in y with an empty band between them (no cup with y between -0.2795 and -0.2111 m): 28 outer, 22 inner, 10 far (x >= 0.52 m). These are properties of the layouts.
- b_level, 26d4118b0c and 4e103843db: reach failure at all 50 cups; 4e103843db also at all 50 cups for b_tilt.
- b_level, e8825dea53 (the only clip with b_level successes): the 10 far cups all succeeded (10/10); the near cups (x < 0.52) 13/40, of them 11/18 inner and 2/22 outer.
- b_level, 49aff04f7a: no success; reach failure at 46/50 cups, and the other 4 runs are approach knocks, all at far cups (4 of the 10 far cups).
- b_tilt, e8825dea53: its 17 successes are all outer (17/28 outer, 0/22 inner). b_tilt, 49aff04f7a and 26d4118b0c succeed mostly on the inner side (49aff04f7a 15/22 inner vs 3/28 outer; 26d4118b0c 4/22 inner vs 0/28 outer). So the side on which b_tilt succeeds differs between clips. Not tested why.
- Approach knocks (cup xy): b_level 49aff04f7a x 0.522-0.538, y -0.323..-0.136; b_level e8825dea53 x 0.467-0.518, y -0.302..-0.166; b_tilt 26d4118b0c x 0.510, y -0.197; b_tilt 49aff04f7a x 0.451-0.475, y -0.322..-0.299.
- The oracle's own 5 approach knocks (trials 34, 38, 44, 45, 49) all lie in the outer group at the low-x edge (cup x 0.447-0.469). In that corner b_tilt 49aff04f7a knocked the cup at trials 6, 34, 49 (34 and 49 are also oracle knock trials). Whether the layout geometry is the shared cause is not tested.
- b_tilt e8825dea53 also has 2 "dropped while moving" (trials 1, 32).
- naive: no structure, all 200 fail the same way at every cup position. The cup ended 9.9-21.1 cm (median 13.5) from the coaster centre and had moved 0.0-15.6 cm (median 3.0) from its start in xy. This is expected: naive replays at M3's fixed recorded-layout translation and ignores the trial's cup and coaster by construction.

Outcome counts in four cup regions (S success, r reach, K knocked, D dropped; outer/inner by y, near/far by x = 0.52):

| run | outer, x < 0.52 | outer, x >= 0.52 | inner, x < 0.52 | inner, x >= 0.52 |
|---|---|---|---|---|
| 26d4118b0c b_level | r 22 | r 6 | r 18 | r 4 |
| 4e103843db b_level | r 22 | r 6 | r 18 | r 4 |
| 49aff04f7a b_level | r 22 | K 2, r 4 | r 18 | K 2, r 2 |
| e8825dea53 b_level | K 2, r 18, S 2 | S 6 | K 2, r 5, S 11 | S 4 |
| 26d4118b0c b_tilt | r 22 | r 6 | K 1, r 16, S 1 | r 1, S 3 |
| 4e103843db b_tilt | r 22 | r 6 | r 18 | r 4 |
| 49aff04f7a b_tilt | K 3, r 16, S 3 | r 6 | r 6, S 12 | r 1, S 3 |
| e8825dea53 b_tilt | D 1, r 6, S 15 | r 4, S 2 | D 1, r 17 | r 4 |

### Recorded layout v2 (b_level, one run per clip)

| clip | view | outcome | detail | cup xy | coaster xy | collision min (mm) |
|---|---|---|---|---|---|---|
| 26d4118b0c | top-down | fail, reach | settle_grasp: palm 5.9 cm from the target after 3.0 s | (0.491, -0.163) | (0.495, -0.308) | -30.4 |
| 4e103843db | 90 deg | fail, reach | settle_grasp: palm 6.6 cm | (0.490, -0.164) | (0.496, -0.307) | -30.1 |
| 49aff04f7a | 45 deg | fail, reach | settle_grasp: palm 4.7 cm | (0.495, -0.167) | (0.492, -0.304) | -12.6 |
| e8825dea53 | 45 deg | **success** | cup 0.81 cm from the coaster centre, slip 0.7 cm, rise 12.85 cm, place correction 25.21 mm | (0.506, -0.152) | (0.480, -0.319) | -0.4 |

So with the layout built from my own recorded coaster offset, 1 of 4 clips completes the task in physics. This agrees with the per-clip trial results: e8825dea53 is the only clip with b_level successes.

### Determinism checks

- Saved: `results/m4_execute/determinism_checks.json` (passed): Track B mug_v1 trials 0-4 re-run through the unchanged oracle (ForcePickPlace via the ladder's `run_trials`) are 5/5 identical to fifty.json in every field except wall_s (all of `detail` and the grasp forces included); Execute's setup + 2 s stand equals PickPlace.run's own byte for byte (qpos, qvel, sim time). `results/m4_execute/r0_after_m4/runs/R0_after_m4.json`: R0 20/20, identical to B3 R0 and to M1 (20/20 each, by the ladder's own comparator). `results/m1_grasp_fix/v3/` was only read: nothing in it was rewritten.
- Re-run by the checker on the final code (scratch, `tmp/m4/check/`): R0 after_m4 20/20, identical to the saved file and to `v3/runs/R0.json`, ladder line "R0 vs B3 R0: 20/20 identical; vs M1: 20/20"; check-oracle 5/5 identical with setup identical. Reason for re-running: the saved check was written at 02:00:08, about a minute before the last edit of sim/m4_execute.py (02:01:04); neither it nor run_resources_*.json records a code hash. All 604 rows were made after that edit (02:15-02:37). Hashes of the code that made them (sha256, first 16 hex): sim/m4_execute.py 02f4dadcc28579c9, retarget/paths_v2.py 4e03aa17b2ea1711, retarget/collision.py 3c7de69975b6fa86, retarget/run_m3_v2.py ac8ae4834236c62b; they match the hashes saved in v2/paths.json and v2/oracle.json (retarget/m4_summary.py bb5de19b770e9c69).
- Four rows re-run in fresh processes with `sim.m4_execute rerun` (e8825dea53 b_level trial00, a success; 49aff04f7a b_tilt trial07; 26d4118b0c naive trial12; 4e103843db b_level recorded_v2): 4/4 identical in every field except wall_s and RSS. The runner and the executor made further reruns (3 and 7, all identical; scratch under tmp/m4/runner and tmp/m4/executor), including rows that ran second in a worker, so reusing a task across runs does not change results.
- Also re-derived by the checker: M3 outputs and earlier results were not modified. The newest mtime in `results/m3_retarget/` outside `v2/` is 01:10 (the M3 videos step) and in each earlier results directory it predates M4 (m1_grasp_fix 00:34); the only files changed since 01:12 outside `results/m4_execute/`, `results/m3_retarget/v2/` and `tmp/` are the five new code files plus CLAUDE.md, docs/log.md (01:29-01:30) and .gitignore (01:42), all before the M4 workers started (01:53). M3's 8 code files match `run_resources.json`.

### What failed

- **Reach, mostly.** 324 of the 400 b trial runs end as a failed settle: b_level 169, b_tilt 155. b_level succeeds only on e8825dea53 (23/50); b_tilt best on 49aff04f7a (18/50) and e8825dea53 (17/50), 4/50 on 26d4118b0c, 0/50 on 4e103843db. Pooled: b_level 23/200, b_tilt 39/200. Oracle on the same layouts: 44/50.
- **The hand hits the mug on the way in.** Collision minima are negative for nearly all b runs (see Step 3), and in most reach failures the cup had been pushed more than 2 cm by the time the settle gave up.
- **naive 0/200** (all "released but not on the coaster"): expected from the fixed-translation design, not informative about the human motion itself.
- **No b run failed at the close or the lift.** Once the palm reaches G the grasp works: 62 of 62 trial-run b successes passed the close, lift and place checks (lift slip 0.04-1.6 cm, median 0.56; final cup 0.26-3.7 cm from the coaster centre, median 1.32; checker recount from rows over the 63 successful b runs).
- Tuned nothing to pass: no threshold or parameter was changed; the oracle's own settle tolerances and checks were used as they are.

### Adaptations and implementation choices

- New modules instead of new functions in warp.py / run_m3.py: editing them would change their SHA-256 and `run_m3 summarize` compares those hashes.
- v2 is derived from M3's builder output by cutting and re-timing, not by recomputing geometry.
- The oracle path starts at (p_s, q_now), like M3's paths; the real oracle starts at the IK palm_start, at most 0.09 mm away. It uses the nominal cup (at rest) and the nominal in-hand offset as `warp.time_scale` does. Phase mapping: "approach" = pre-grasp leg + leg to G, "transport" = lift + transport + lower, no "pause" phase; i_pause = pre-grasp arrival (8 cm of straight arc before G), i_peak = -1.
- Naive v2: the replay is re-timed as one rest-to-rest segment, because naive has no pause and no separate transport phase.
- The Step 3 diagnostic runs after the physics run as a hook, with its own setup and one plan (M3's `set_robot_to_clip_frame`). Arc length is measured along the path targets; "right fingers at HERO's open pose" is `scene.reset` (both hands open), as in M3's step 6.
- follow phase names reuse the oracle's ('pregrasp', 'approach', 'lift', 'transport', 'replay') so failed_phase and the M1 categories line up.
- Retreat orientation (not named in the spec): b uses the path's grasp orientation, naive the path's last quaternion.
- The lift rule uses the planned peak (pos[i_peak].z - G.z), the way the oracle uses its commanded 10 cm; the live palm z is the alternative. `lift_planned_cm` and `lift_cup_rise_cm` are both in the rows.
- Place-correction progress s = `paths_v2.sigma(j/50, D0)/D0` of transport 2 (the builder's own function, one copy of the formula).
- The two Step 4 batches were run as two commands (600 runs, then 4 recorded_v2), so each has its own resource record.
- failure-map PNGs include the oracle as a fifth reference panel; the summary also has two tables not in the spec (which settle gave up; counts in four cup regions), added so the cluster sentences are backed by numbers.

### Open questions

1. **Check order and the knock label.** `settle_grasp` runs before `check_cup_undisturbed` (literal, the oracle's order), so most pushed-mug runs are labelled reach. Cross-tab above: 124/169 b_level and 88/144 b_tilt reach failures had the cup moved more than 2 cm. Should the README count those as knocks, and should summary.md carry this cross-tab? It is not in summary.md now.
2. **M3's open question 1** (should the path-max rule exclude the prelude?): the data is in. The oracle's prelude is 1.69 cm and passes; b_level's is 1.90-2.86 cm. The rule was left unchanged (literal reading); the after-prelude column is information only.
3. **Naive v2 timing.** It is re-timed as one segment. The alternative (split like b at an 8 cm-before-g point and at a transport peak) would put the naive hand at rest mid-replay while its closure ramps on time with no hold.
4. **Place-correction progress s:** time fraction j/N versus path progress sigma(t_j)/D0; they differ only in the first and last 0.4 s. sigma/D0 is implemented.
5. **Oracle collision distribution** covers only the 10 Step 0 rows (trials 0-9, all successes). The oracle's 5 approach knocks have no value; running `collision_min(oracle_path)` on all 50 layouts (about 3.5 s each) would fill it. Not done, not in the spec.
6. **The collision diagnostic does not separate knocks from successes** (b_tilt: all 39 successes overlap); the window ends 1 cm before G, so a late overlap can be the intended contact.
7. **Naive failure category.** `evaluate()` files naive as "released but not on the coaster" even when the mug was never lifted. Literal; the `reason` field has evaluate()'s exact text.
8. **Pooled CIs** treat 200 runs as independent. The oracle runs the trial's own handle yaw, the variants run trial yaw + (phi_h - phi_ora); the oracle ignores the handle, so this should be immaterial for the paired counts, but it was not tested.
9. **Follow start offsets.** In the rows, the live palm after settling is not at the path's first frame: the bend applied at the start of each follow slice is 3.6-36.5 mm for approach 2 (median 14.1, n = 331), 10.2-31.6 mm for transport 1 (median 22.2, n = 76), 9.9-49.0 mm for transport 2 (median 18.0, n = 76), 0.1 mm for the first plan (`detail.plans[*].start_offset_mm`; checker recount, scratch). Not examined further.
10. **Reviewer items left unfixed** (no fix pass was run; none occurred in the data, 0 error rows): a diagnostic exception would turn a physics row into an error row; `--skip-existing` skips error rows instead of re-running them.
11. **Per-process CPU.** Each worker used about 120 % CPU (multithreaded numerics), so 8 processes kept about 10 cores busy. The spec limited processes, not cores.

### Runtime and peak RSS (CPU only; no GPU; no videos)

| step | wall | peak RSS |
|---|---|---|
| `run_m3_v2 paths` | 8.4 s | 170 MB |
| `run_m3_v2 oracle` (1 process) | 39.7 s | 1815 MB |
| `run_m3_v2 summarize` | 0.02 s | 153 MB |
| main batch, 600 runs, 8 workers | 823.6 s (13.7 min) inside the command; per-run wall mean 10.95 s, max 18.3 s | 1531-1593 MB per worker (sum 12.5 GB), parent 65 MB |
| recorded_v2 batch, 4 runs, 4 workers | 11.8 s | 1603-1656 MB per worker, parent 63 MB |
| checker: R0 re-check (4 processes) | 28.5 s | max 1665 MB per process |
| checker: check-oracle | 25.7 s | 2732 MB |

All far below the 16 GB per-process cap.

### M4 — merge step and videos step
- **Track V** (worktree `../hoi-visual`, branch `visual-polish`, commit dae8217): visual-only polish in the new `sim/visuals.py`
  (dark-green glossy mug + 3 handle capsules, white interior as a MuJoCo *site* — a zero-mass non-colliding *geom* still changed the cup
  free joint's compiled `dof_length`, which the check caught; cream coaster; procedural wood table; neutral floor with haze; key + fill
  lights; named render camera `grasp_right`; video 1280×720). Proof `sim/check_visual_only.py`: 1656/1656 element-for-element comparisons
  identical (6 cup × contact models; spec fields + extras + 1 s of dynamics). R0 `visual` = M1 (20/20 identical).
  Before/after image: `results/visual_polish/before_after.png`; 5 s test video `results/visual_polish/r0_trial00_5s.mp4` (gitignored).
- **Merge** (`5f3bfd5`): no file changed by both tracks, no conflict. On the merged code: `check_visual_only --base 6fb6537` (pre-merge
  main) PASS 1656/1656; R0 `after_merge` 20/20 identical to M1 and B3; the 5 M4 reruns (first b_level run of each clip + 4e103843db naive
  trial00) identical to their saved rows (`results/visual_polish/merge/`). Worktree removed.
- **Videos** (`python -m sim.m4_execute videos`, new visuals, each re-run identical to its saved row; mp4 gitignored, in
  `results/m4_execute/videos/`): 4e103843db b_level recorded_v2 (fails at reach, palm 6.6 cm short), e8825dea53 b_level trial07 (approach
  knock), 4e103843db naive trial00, oracle mug_v1 trial00 (success), and e8825dea53 b_level recorded_v2 (success). GIF of that b_level
  success: `results/m4_execute/b_level_success.gif` (4.92 MB) — the 90° recorded-layout run could not be used because it failed.

## M4b — diagnose and fix: robot-specific final approach + heading test

All numbers below come from `results/m4b/summary.md` / `summary.json` (`python -m retarget.m4b_summary summarize`), `results/m4b/m4_category_v2.json`
and `results/m4b/heading_sweep/`. Profile mug_v1, my mug at 0.40 kg, trials 0-49 (the oracle's 50 layouts).

### Why
The M4 failure analysis (from M4's saved rows) found two problems:
- **The human final approach.** Most `b_level` failures pushed the mug more than 2 cm before the close. Following my final approach drives
  the Dex3's open fingers, about 8 cm beyond the palm, through the mug. The oracle's straight final approach clears the mug by about
  5.6 mm. The brief spoke of a "first overlap 6-9 cm before G", but the rows record only the DEEPEST overlap (collision_min.frame and its
  remaining_arc_to_G_m), not the first one. The independent check found the deepest overlap typically 6-9 cm before G, and up to
  12.6 cm.
- **The heading.** On top-down and 90° the pre-grasp settle also misses, at palm headings of 82-86°.

M4b tests one fix, a robot-specific final approach, and isolates the heading.

### Step 1 — category_v2 (`retarget/m4b_summary.py`, no reruns)
- **Definition:** "pushed mug before close" when the run failed before close_hand started and the mug's final xy is more than 2 cm from
  its layout xy. Otherwise category_v2 is the run's own category.
- **What "before close" means:** failed_phase "approach"; a "reach" failure of settle_pregrasp or settle_grasp (the last settle record,
  asserted to be the one named in the reason); or a fall in a pre-close phase.
- **M4 `b_level` failures re-filed as pushed:** 31/50 (26d), 25/50 (4e), 50/50 (49aff), 26/27 (e88), as the spec's numbers.
- **M4 `b_tilt`:** 92/161 failures pushed; e88 `b_tilt` 0/33.
- The run files are not edited.

### Step 2 — `b_hybrid`, `b_hybrid_h45`
- **Paths** (`retarget/paths_v2.py`: `hybrid_grasp`, `hybrid_path`):
  - b_level's grasp (h45: R_ora at my grasp height); u = M3's horizontal unit vector of (q - G); PG = G - 0.08 u.
  - My approach a0..g is warped onto p_r0 -> PG with M3's formula and z clamp, with no pause, and re-timed as one segment.
  - The path then holds the oracle's straight leg PG -> G as a nominal record (phase "final_approach"). Close, transport, release and
    retreat are built as M3 / v2 build them.
- **Self-check (on every b_hybrid run):**
  - The warp is linear in its end point, so the approach must equal M3's b_level approach plus w (PG - G). The largest difference is
    5.6e-17 m on the smoke runs.
  - The prelude and every frame from the close on are bit-identical to v2 b_level.
- **Executor** (`sim/m4_execute.py`, `_object_aware(..., hybrid=True)`):
  1. follow approach 1 (base "plan", base goal = `_goal(G, R_G)`).
  2. `settle_or_fail` at PG (5 s).
  3. The oracle's `move("approach", [(G, quat_G, 2.0)])`.
  4. `settle_grasp` (3 s), then `check_cup_undisturbed`.
  5. M4's steps 3-6 unchanged.
- **Handle yaw:** b_hybrid_h45 uses the trial's own handle yaw (`variant_layout`); b_hybrid uses M4's (trial + (phi_h - phi_ora)).
- **Collision window:** ends 1 cm of arc before PG.
- **M4 unchanged:** 3 M4 rows re-run identical (`results/m4b/checks/`). The M4 summary regenerates identically (only the input-path
  string differs). Two independent code reviews against the spec found nothing.
- **Fix needed along the way:** `retarget/m4_summary.py` imported `VARIANTS`, which now includes the hybrids. It is pinned to
  `M4_VARIANTS`.

### Step 3 — heading sweep (`sim/m4b_heading_sweep.py`)
- **Setup:** Track B final's mug_v1 oracle (`ForcePickPlace`, built as Track B built it) with `task.grasp` turned by delta = phi - 45 deg
  about the cup axis: offset, quaternion and approach_dir. Handle yaw = trial + delta.
- **Check (passed):** phi = 45 reproduces fifty.json's trials 0-19, 20/20 identical in every field except wall_s. It runs first, and
  the command stops if it fails.

### Results

| clip (phi_h) | M4 b_level | b_hybrid | b_hybrid_h45 |
|---|---|---|---|
| 26d4118b0c top-down (81.9) | 0/50 | 22/50 | 43/50 |
| 4e103843db 90° (85.8) | 0/50 | 14/50 | 46/50 |
| 49aff04f7a 45° (81.7) | 0/50 | 27/50 | 46/50 |
| e8825dea53 45° (61.7) | 23/50 | 49/50 | 44/50 |
| pool | 23/200 | 112/200 (0.49-0.63) | 179/200 (0.84-0.93) |

- **Oracle:** 44/50 on the same layouts.
- **Heading sweep, success /20:** phi 45: 20, 55: 20, 65: 20, 75: 13, 85: 8.
- **Paired comparisons:**
  - b_hybrid vs b_level: b_hybrid wins 89 layouts and loses none (22 + 14 + 27 + 26).
  - b_hybrid vs h45, on the three high-heading clips: h45 alone wins 22, 32 and 21 layouts; b_hybrid alone wins 1, 0 and 2.
  - b_hybrid vs h45, on e88: b_hybrid alone wins 5, h45 alone 0.
  - e88 b_hybrid vs the oracle: 44 both, 5 b_hybrid only, 0 oracle only.
- **Failures:**
  - b_hybrid: reach 74 (pre-grasp 49, place 20, grasp 5), knocks 14.
  - h45: knocks 20, 1 lift slip. No reach failure.
- **Collision (min mm):** b_hybrid successes median 12.6, 40 of 112 overlapping; b_hybrid failures median -27.5, 74 of 88 overlapping;
  h45 successes median 50.4, 0 of 179 overlapping.

### Reading
The comparisons are a sequential ablation (M4 b_level -> b_hybrid -> b_hybrid_h45), not a factorial design: there is no "b_level at 45 deg"
arm. So each step is tested only as the bundle of changes it makes, and the ranking of the two factors is not tested.

- **Tested: the robot-specific approach bundle helps at my heading (23/200 -> 112/200, 89 layouts won, none lost).**
  - Same in b_level and b_hybrid: the grasp, the layout, and every path frame from the close on (bit-identical, checked on all 200).
  - Different, all bundled together:
    - where the pre-grasp settle happens: PG, straight back from G, vs a point on my path 8 cm of arc before G;
    - the approach before that point: my approach warped back to PG, re-timed as one segment and up to about 1 s longer;
    - the final leg: the oracle's straight 2 s move vs my last 8 cm.
  - Which part of the bundle matters is not separated.
- **Tested: turning the grasp to the oracle's 45 deg helps on top of the hybrid approach (112/200 -> 179/200).**
  - "Heading" means the whole grasp turned about the cup axis: G, PG, P and the retreat follow it, and the handle co-rotates
    (b_hybrid: trial yaw + (phi_h - 45); h45: the trial's own yaw).
  - Height, approach shape, transport and k are the same.
  - The gain is all on the three clips with phi_h 81.7-85.8 deg (h45-only 22 / 32 / 21 layouts vs b_hybrid-only 1 / 0 / 2).
  - On e88 (61.7 deg) b_hybrid is as good or better: 49 vs 44, paired 5 / 0.
- **Not tested: which factor is larger.** Pooled, the approach step gains more (net +89 vs +67), but per clip it differs: on 4e the
  heading step gains +32, the approach step +14.
- **Tested on the scripted oracle alone:** success falls between 65 and 75 deg (20/20 -> 13/20; 85 deg: 8/20).
- **Only suggested: why high headings fail.** There are two signals and they are not separated:
  - Most b_hybrid reach failures give up at the pre-grasp (49) or place (20) settle, as at phi 85 in the sweep (5 pre-grasp, 3 place).
  - All 49 pre-grasp reach failures and all 5 grasp-settle ones overlap the mug in the kinematic diagnostic on the way to PG (median
    -36.9 mm), against 40/112 b_hybrid successes and 0/179 h45 successes. Yet none of the 49 moved the mug more than 2 cm.
  - So either the arm's reach at those headings or contact on the way to PG could be the cause.
  - In M4, 49aff (81.7 deg) had no pre-grasp reach failures at all, so the heading alone does not explain the pre-grasp misses.
- **Data role, honestly:**
  - Both hybrids drop my final ~8 cm of approach: it is replaced by the oracle's settle and straight move.
  - h45 also drops my heading; it keeps my approach shape up to PG, my grasp height, my transport and my timing.
  - b_hybrid keeps my heading as well, and matches or beats the oracle only on e88, whose heading is inside the range where the oracle
    sweep stays at 20/20 (<= 65 deg).
- **category_v2 caveat:** cup positions are saved to 0.1 mm, so labels within 0.2 mm of the 2 cm threshold depend on the rounding (7 runs,
  listed in summary.md). Two of them are in M4b: h45 26d trial36 at 20.009 mm and b_hybrid 4e trial10 at 20.082 mm, both counted as
  pushed.
- **Process:**
  - Two agent code reviews of Step 2 found nothing. They are not saved in the repo.
  - An independent re-derivation from the raw rows matched every number in summary.md (263 checks).
  - The same check flagged the overstated first draft of this Reading, rewritten as above.

### Videos and resources
- **Videos:**
  - `results/m4b/videos/` (gitignored): e88 b_hybrid trial00 (success) and 26d b_hybrid trial00 (reach failure at settle_place). Both
    re-runs are identical to their rows.
  - GIF: `results/m4b/b_hybrid_success.gif`, 3.90 MB (480 px, 10 fps).
  - Renderer: Mesa Intel.
- **Resources:**
  - Batch: 746 s on 8 workers (mean 14.9 s per run, max 21.1 s); peak RSS 1524-1627 MB per worker.
  - Sweep: 126 s, at most 1658 MB per process.
  - Render: 51 s, 1965 MB.
  - No GPU.

## M5 — learned grasp selector (overnight run, one shot with pre-set branches)

### Plan change
- M5 is a learned grasp selector, not ACT / Diffusion Policy and not LeRobot. The `hoi-policy` env is not created.
- The selector is trained with PyTorch in `hoi-perception`, exported to ONNX and run with onnxruntime in `hoi-sim`.
- Why: M4b showed my heading transfers to this arm only up to about 65 deg (heading sweep). So the thing worth learning is, per layout,
  which of my clips to use and how far from my heading the robot must move.
- The two envs exchange plain files: `results/m5/train_data.npz` going in, `results/m5/selector.onnx` coming back.
- The 50 eval layouts (trials 0-49) are test only.

### Step 0 — `b_hybrid_phi`
- `b_hybrid_phi` is b_hybrid with the grasp heading set to a given phi: `paths_v2.hybrid_grasp` uses
  `grasp_pose("b_level", cup_xy, dict(human, phi_h=phi), oracle)`.
- **Candidate headings** (`sim/m4_execute.py candidate_phi`):
  - "phih" = my phi_h.
  - "45" = phi_ora exactly. phi_ora is 0.7853981633974484 rad, which is `math.radians(45)` + 1 ulp, so the "45" candidate has to be
    phi_ora itself to reproduce b_hybrid_h45's R_ora bit for bit.
  - "55", "65", "75" = phi_ora + 10 / 20 / 30 deg, as the M4b heading sweep turned the grasp.
- **One handle yaw per layout** (`variant_layout`): the layout's own, whatever the grasp. `yaw="m4"` exists only for the check against
  b_hybrid.
- **Check passed** (`results/m5/step0_check.json`):
  - Paths are bit-identical on all 4 clips: phi_h vs b_hybrid, and 45 vs b_hybrid_h45.
  - 4 / 4 re-runs (26d trial00 / 01, each against both references) are identical to the saved M4b rows, every field except the
    identity fields (run, variant, the candidate fields, and the info fields `variant`, `phi_rad`, `check_against_b_level`).
- One M4 row and one M4b row re-run identical after the change (`results/m5/checks/`).

### Step 1 — 120 training layouts (`policy/m5.py layouts`, `results/m5/train_layouts.json`)
- **Sampling:** `sample_layouts(legacy grasp, n, seed=100)`. Handle yaw = the legacy grasp's approach angle + U(-45, 45) deg from
  `default_rng(2100)`. trials() draws its yaw around its own grasp's approach angle, the legacy one, so this is trials()'s distribution.
- **Deviation:** the spec's assert (no training cup or coaster within 1 mm of any eval cup or coaster) fails on the first 120 draws.
  - Draws 29, 74, 98 and 105 come 0.35-0.74 mm from an eval point.
  - With about 24 000 point pairs in a 10 x 20 cm box, about 4 such near-coincidences are expected by chance.
  - The breaking draws are dropped and the same seeded sampler stream continues: 125 draws, 5 dropped (draw 120 also breaks it), 120
    kept. sample_layouts with a larger n returns the same first draws (asserted).
  - Nothing here depends on any outcome; the training set is disjoint as the spec intends.

### Step 2 — 2160 candidate rollouts
- **Candidates:** 18 in total. Each of 26d, 4e and 49aff has phi_h, 75, 65, 55 and 45. e88 (phi_h 61.7 deg) has phi_h, 55 and 45.
- **Runs:** every candidate on every training layout through `sim.m4_execute run --items results/m5/rollout_items.json`, with M4's row
  format plus the candidate fields and the collision diagnostic.
- **Rate:** 111 rows after 244 s projected 79 min in total, under the 2.5 h fallback limit, so all 120 layouts ran.

### Environment change
- The ONNX export (`torch.onnx.export(..., dynamo=False)`) needs the `onnx` package, which `hoi-perception` did not have
  ("Module onnx is not installed!" on a trial export).
- `pip install --dry-run onnx -c envs/hoi-perception.constraints.txt` would add only ml_dtypes 0.6.0, onnx 1.23.2 and protobuf 7.36.2;
  torch and numpy stay unchanged. It was installed into `hoi-perception` only.
- Afterwards torch is 2.11.0+cu128 and numpy 2.2.6. `envs/hoi-perception.lock.txt` was re-frozen and the three pins added to its
  constraints. The only `pip check` complaint (pyrender vs PyOpenGL) predates this install.

### Side work done during the rollouts (background agents; nothing simulated except one identical re-run)
- **Data sample** `data/raw/sample/4e103843db/`: a byte-identical copy of the 90 deg clip (rgb.mp4 10.7 MB, so no re-encode; 23 MB in
  total).
  - `perception.stray.load` gives the same rows, timestamps, K, poses and depth / confidence on both copies, and the video decodes to
    N - 1 = 513 frames in both (`results/m5/data_sample_check.json`).
  - rgb.mp4 is committed with `git add -f` (the repo ignores *.mp4).
- **`docs/INSTALL.md`:** from the log, the lock files and the code. The commands it could not verify are marked "(unverified)". No
  `patches/` directory exists.
- **Extra visual** (`sim/hand_to_robot_video.py`, `results/extra/hand_to_robot.gif`):
  - Side by side, my e88 clip (M2 overlay) and the robot's b_hybrid success on trial00.
  - My clip is piecewise-linearly time-warped onto the robot's real-time run at 6 events (approach start, contact, lift, carry,
    place, release). My events come from the demo npz; the robot's from a re-run of the saved row, which came out identical.

### Step 2 results — branch D1 (`results/m5/candidates.md` / `.json`)
- **Runs:** 2160 in 4283 s on 8 workers (peak RSS 1568-1646 MB per worker); 0 exceptions, so **D1 = continue**; 1538 successes.
- **The heading pattern is the M4b sweep's, now on 120 training layouts per candidate.**
  - At my own heading: 26d 41/120, 4e 31/120, 49aff 35/120, e88 96/120.
  - At 65 deg: 112 / 94 / 97 (26d / 4e / 49aff).
  - At 45 deg: 111 / 115 / 111 / 112 (26d / 4e / 49aff / e88).
- **Candidate-oracle ceiling:** at least one candidate succeeds on 120/120 training layouts.
- **Fixed-best F** (highest training success, ties -> larger phi): 4e103843db at 45 deg, 115/120.

### Step 3 results — branch D2 (`results/m5/cv.json`, `selector_meta.json`, `onnx_check.json`)
- **Cross-validation:** 5 folds by layout, noisy held-out inputs.
  - AUC 0.916 and Brier 0.102 (base rate 0.712), so the selector predicts success well above chance.
  - Held-out selection success: R1 115/120, R2 109/120, F 115/120, CEIL 120/120.
  - **D2: learning does not beat the fixed best choice in CV** (R1 115 <= F 115). Continued as the branch says: no retraining, no
    other model.
  - R2 trades 6 layouts for staying closer to my heading.
- **Final model:** trained on all 120 layouts in 16 s on CPU and exported to ONNX (21.9 KB). onnxruntime matches torch to 1.2e-7 on
  1000 rows (tolerance 1e-5).
- **Test choices** (`results/m5/choices.json`, inputs noisy, seed 2026):
  - R2 picks e88 at my own heading in 33/50 trials.
  - R1 mostly picks 45 deg candidates.
  - F is 4e103843db 45 everywhere.
  - The best training clip at 45 deg is also 4e103843db, so "b_hybrid_h45 noisy" and "F noisy" are the same runs (the noisy items
    are de-duplicated: 100 noisy runs, not 150).

### Step 4 results — branch D3 (`results/m5/summary.md` / `.json`; 403 eval runs in 748 s on 8 workers, 0 exceptions)

| variant (50 test layouts) | success |
|---|---|
| R2 human-first, clean | 49/50 |
| R1 argmax, clean | 47/50 |
| F (4e103843db 45), clean | 46/50 |
| R2, noisy path | 44/50 |
| F = b_hybrid_h45 4e103843db, noisy path | 38/50 |
| b_hybrid at each layout's own yaw | 98/200 (26d 22, 4e 11, 49aff 21, e88 44) |
| oracle | 44/50 |

- **D3:** R1, R2 and F are all at least 30/50, so no "what failed" breakdown is needed.
- **Paired comparisons:**
  - R2 vs F: 45 both / 4 R2 only / 1 F only / 0 neither.
  - R2 vs the oracle: 43 / 6 / 1 / 0. That edge rests on 7 discordant layouts, too few to claim R2 is better than the oracle.
- **R2's choices:**
  - It keeps my own heading on 35/50 layouts (e88 phi_h 33, 26d phi_h 2); mean departure 3.4 deg, max 36.7.
  - By clip: e88 41, 26d 5, 49aff 4, 4e 0. It fell back to R1 on 2 layouts (no p >= 0.8).
- **Reading** (corrected after the independent check; numbers in summary.md "R2 by the clip it chose"):
  - On the 41 layouts where R2 chose e88, it used e88's own heading on 33 (55 deg on 5, 45 deg on 3). F and the oracle also succeed
    on 39 of those 41, so choosing e88 is what lets R2 keep my heading, not what produces the success.
  - R2's edge over the oracle and over F comes mostly from its 9 picks of other clips, all successes: 26d 65 / 75 / own heading; 49aff
    45 / 75. These include 4 of its 6 wins over the oracle (trials 34, 44, 45, 49, the oracle's knock layouts) and 2 of its 4 wins over
    F. Its 75 deg and own-heading picks there come from candidates that succeed on only 34-52 % of the training layouts (26d own heading 41/120, 26d 75 62/120, 49aff 75 58/120).
  - Learning does not beat the fixed best choice in CV (D2). On the test layouts, R2 vs F is 4 vs 1 discordant and R1 vs F 2 vs 1,
    too few to separate.
  - b_hybrid at the layout's own yaw (98/200) is lower than M4b's b_hybrid (112/200), which turned the handle with the grasp. So
    M4b's 112 and its "e88 beats the oracle" (49 vs 44) depend on that handle convention: at its own yaw e88 b_hybrid is 44/50.
- **Noisy path:** only the PATH is built from the noisy layout. The scene is true, and the place correction stays privileged (it
  reads the true cup pose in the hand and the true coaster).

### Step 5 / M6 / M7
- **Videos:** the R2 success (e88 at 55 deg, trial00) and the R2 failure (e88 at 55 deg, trial07, released but not on the coaster) are
  both re-runs identical to their rows; the GIF is `results/m5/r2_success.gif` (4.0 MB).
- **Gallery** (re-renders checked identical):
  - new: `results/final/oracle_success.gif` (4.9 MB) and `results/final/b_level_failure.gif` (3.4 MB, M4 e88 trial07 approach
    knock);
  - reused: `results/m4b/b_hybrid_success.gif` and `results/m5/r2_success.gif`.
  - `sim.m4_execute render` now makes a GIF whenever the re-run is identical to its row (success or failure).
- **Figures:** `results/final/ablation.png` and `heading.png` (`retarget/final_figures.py`, from saved summaries only).
- **`docs/README_draft.md`:** every number links to its file; README.md is untouched.
- **`scripts/reproduce.sh`:** one oracle trial, one b_hybrid_h45 trial, one R2 trial from the ONNX model, then the M4 / M4b / M5
  summaries.
  - Clean-clone check: git clone of HEAD, third_party/ symlinked, existing envs. ALL MATCH in 33 s, peak RSS 2.7 GB
    (`results/final/clean_clone_check.txt`).
  - D4 needed no fix.
  - The hoi-sim env's HERO is an editable install from this machine's `third_party/HERO`.
- **Hygiene** (`scripts/hygiene.py` -> `results/final/hygiene_report.md`): one tracked file over 10 MB (the sample's rgb.mp4, 10.68 MB),
  no secret-looking strings, no README link to an untracked file, no top-level LICENSE (left for the owner).
