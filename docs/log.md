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
