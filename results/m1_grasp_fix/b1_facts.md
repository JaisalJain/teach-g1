# B1: facts from the code (Track B, M2 v2)

Nothing here comes from a simulation run. Each fact cites the file and line it comes from. Values that only exist after
MuJoCo compiles the model (joint ranges, options, pair parameters) and the two kinematic checks come from
`sim/grasp_fix_b1.py`. That script builds the oracle's `TableScene` at 0.74 m for the cylinder and for the mug (config
mass 0.40 kg), runs `mj_kinematics` / `mj_forward` / `mj_geomDistance` only (simulated time never advances), and
writes `results/m1_grasp_fix/b1_model.json`. Every compiled value below is in that JSON.

Line numbers:
- `sim/*.py`: the working tree on 2026-10-30. `sim/scene.py` has uncommitted edits that this step did not make: `mug_config()`
  now skips entries without a `value` (needed since `ruler_check` was added), and there is a new `cup_model="mug_body"`.
- HERO (`third_party/HERO`): commit `416c4e0`.
- URDF = `third_party/HERO/assets/robots/g1_modified/g1_29dof_with_hand_rev_1_0.urdf`.

MuJoCo 3.13.0.

## Short answer: the palm standoff is hard-coded

The palm-to-object standoff and the pre-grasp approach are **hard-coded**. They do not depend on any object radius.
- The palm target is a fixed offset in the palm frame: the cup centre sits at (0.077, 0.055, 0) m from the palm point
  (`sim/pick_place.py:58`, used at `:92-96`).
- The offset is HERO's single Dex3 side-grasp constant, used for every object HERO side-grasps
  (`sim2sim/interactive_client/controller.js:122`, applied at `:195` and `:213`).
- No radius appears anywhere in `sim/pick_place.py`: neither `CUP_RADIUS` nor the config diameter.
- The only object dimension the oracle uses is the height: the palm stays 6.5 cm above the table (`pick_place.py:94`, `:123`).

It is not computed from the old cylinder either. But it was only ever validated on the old cylinder (r 40 mm). HERO's own
side-grasped objects are smaller: can r 28 mm (`interactive_scene.py:81-82`), bottle r 27 mm (`:83-84`) and HERO's mug,
outer r 37 mm (`:510-514`).

Consequence: the gap between the palm and the object surface shrinks one-for-one as the radius grows. At the oracle's nominal
grasp pose, the signed distance from the cup body to each hand collision geom is (negative = overlap; `b1_model.json` →
`cups.*.nominal_grasp.clearance_open_mm`, hand open):

| geom (right hand) | cylinder r 40 mm | mug r 49.5 mm |
|---|---|---|
| palm (`right_hand_palm_link`, convex hull) | **-2.15 mm** | **-11.65 mm** |
| index_0 / middle_0 (proximal) | +7.05 / +7.05 mm | **-2.45 / -2.45 mm** |
| index_1 / middle_1 (distal) | +26.42 / +26.25 mm | +17.52 / +16.75 mm |
| thumb_0 / thumb_1 / thumb_2 | +10.69 / +20.48 / +36.51 mm | +1.19 / +10.98 / +27.01 mm |

So the target palm pose already lies 2.15 mm inside the old cylinder, and 11.65 mm inside the mug. For the mug, the open
proximal finger links also overlap it. "Nominal" means the oracle's target pose. In a run the palm only gets within
1.5 cm of it (`SETTLE_STAY_M`, `pick_place.py:67`), and a settle fails only beyond 3 cm (`REACH_FAIL_M`, `:69`). So in a run the
palm presses on the object or stops short; these numbers describe the plan, not a run.

## 1. The old cylinder ("cylinder", M1)

| property | value | source |
|---|---|---|
| diameter | 0.08 m (radius 0.04) | `sim/scene.py:54` (`CUP_RADIUS`) |
| height | 0.10 m | `sim/scene.py:54` (`CUP_HEIGHT`) |
| mass | 0.15 kg (solid, inertia diag 1.85e-4, 1.85e-4, 1.20e-4 kg m², from `b1_model.json`) | `sim/scene.py:54` (`CUP_MASS`), geom at `:183-185` |
| geom friction | (0.35, 0.0028, 0.0001), condim 4, solref (0.008, 1), solimp MuJoCo default | `sim/scene.py:181-182` |
| pair with each hand geom | mu 1.0625 (sliding, both tangent directions), torsional 1.0625 x 0.004 = 0.00425 m, rolling 0.0001 | `sim/scene.py:61` (`CUP_FRICTION`), `:203-214`; HERO "glazed ceramic mug" prior `interactive_scene.py:109`, arms `:120-121` |
| pair with table top | mu 0.35, torsional 0.35 x 0.008 = 0.0028 m | same lines |
| pair with coaster | mu 0.5, torsional 0.5 x 0.008 = 0.004 m | `sim/scene.py:61` (coaster = our nominal guess) |

The mug uses the same contact parameters on every one of its geoms (`sim/scene.py:181-182`, `:203-214`). For comparison,
the mug model the sim builds from config (`b1_model.json` → `cups.mug`): body r 0.0495 m, h 0.091 m, 0.40 kg, inertia diag
(5.55, 5.21, 5.16)e-4 kg m², COM 1.29 mm toward the handle.

## 2. Palm target and pre-grasp approach

- **Palm orientation:** `rs.palm_rotation(-45 deg, 0, 0)` = a pure yaw of -45 deg (HERO's outward wrist yaw). The palm z axis is
  world z (`sim/pick_place.py:59`, `:93`; `data_tools/reach_specs.py:579-581`).
- **Palm position:** `palm = cup_centre - R (0.077, 0.055, 0) + (0, 0, 0.065 - h/2)` (`pick_place.py:58`, `:60`, `:94`,
  `:98-99`, `:208`). `cup_centre` is the cup body's frame origin, which is the centre of the (body) cylinder.
- **Palm frame:** the "palm" point is `right_hand_palm_link`, which is `wrist_yaw_link` + (0.0415, -0.003, 0)
  (URDF `:1204-1208`, `hero_isaacsim/constants.py:112-114`).
  - In this frame x points along the fingers. Index and middle flex about +z (URDF `:1320`, `:1349`, `:1378`, `:1407`, ranges ≥ 0),
    so they curl toward +y: the palm faces +y.
  - The cup centre is therefore 55 mm along the palm normal and 77 mm along the fingers, level with the finger bases
    (index_0 / middle_0 joints at x = 0.0777, URDF `:1319`, `:1377`).
- **Standoff in numbers** (arithmetic on `pick_place.py:58`; also in `b1_model.json` → `grasp`):
  - palm point to cup axis, horizontal: |(0.077, 0.055)| = 94.63 mm, the same for every object;
  - palm point to surface along that line: 94.63 - r = 54.63 mm (cylinder), 45.13 mm (mug);
  - along the palm normal: 55 - r = 15.0 mm (cylinder), 5.5 mm (mug).
- **Approach direction:** the horizontal unit vector palm → cup centre, i.e. -offset with z dropped (`pick_place.py:95-96`).
  - In the palm frame it is (0.8137, 0.5812): 35.5 deg from the finger axis toward the palm normal. In world it is -9.46 deg
    (= `approach_yaw_deg` in `results/m1_pick_mug/summary.json`). It comes from the hard-coded offset alone.
- **Pre-grasp:** `pre_p = grasp_p - 0.08 * approach_dir` (`PREGRASP_BACKOFF`, `pick_place.py:61`, `:209`). It is hard-coded
  and does not depend on the radius.
  - Sequence: lift to the table-clearing height, move to `pre_p` (1.2 s + 1.8 s), settle (< 1.5 cm), then a straight 2 s
    move to `grasp_p`, settle (max 3 s), then the "cup undisturbed" check (`pick_place.py:210-219`; `:292-295`: moved > 2 cm
    or tilt > 15 deg = failure).
  - HERO's own demo backs off 0.12 m along palm-start → object instead (`controller.js:220`; `SIDE_APPROACH_BIAS_M = 0`, `:44`).
- **Layouts:** `sample_layouts` draws palm positions in the box and sets cup = palm - offset_xy (`pick_place.py:357`). The
  M1 / M1-mug layouts therefore depend on the same hard-coded offset. Changing the offset changes which cup positions a seed
  produces.

## 3. Grasp height on the object

- The palm point is 0.065 m above the table top for every object (`GRASP_Z_ABOVE_TABLE`, `pick_place.py:60`, `:94`). This is
  HERO's side-grasp centre height for its mug and can (`controller.js:212`).
- Relative to the object:

  | | palm above cup centre | palm at this fraction of the height |
  |---|---|---|
  | cylinder (h 0.10) | +0.015 m | 65 % |
  | mug (h 0.091) | +0.0195 m | 71 % |

- Height range of each right-hand collision geom above the table at the nominal (level) palm. These ranges do not change with
  finger flexion, which is about the palm z axis (`b1_model.json` → `cups.*.nominal_grasp.hand_geom_z_above_table_m`):

  | geom | z above table [m] | vs cylinder top (0.100) | vs mug top (0.091) |
  |---|---|---|---|
  | index_0 / index_1 | 0.0806-0.1063 / 0.0804-0.1064 | top 6.4 mm above the rim | top 15.4 mm above the rim; only the lowest 10.6 mm overlaps the wall |
  | middle_0 / middle_1 | 0.0236-0.0493 / 0.0234-0.0494 | inside | inside |
  | thumb_0 / thumb_1 / thumb_2 | 0.0555-0.0745 / 0.055-0.075 / 0.0521-0.0781 | inside | inside |
  | palm | 0.0211-0.1089 | | |

  The joint origins explain these ranges: index_0 sits at palm z +0.0285 (URDF `:1377`), middle_0 at -0.0285 (`:1319`), and
  thumb_0 at z 0 (`:1232`).
  With the hand closed, the index distal link's overlap with the object equals exactly this vertical overlap: -19.60 mm on the
  cylinder, -10.60 mm on the mug (`clearance_closed_mm`). Its shortest way out is upward.

## 4. Finger closing targets, PD gains, torque limits

HERO's finger order is (thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1) (`interactive_scene.py:30`, `:280`).

| right hand | thumb_0 | thumb_1 | thumb_2 | index_0 | index_1 | middle_0 | middle_1 | source |
|---|---|---|---|---|---|---|---|---|
| open target [rad] | 0 | 0.7 | 0 | 0 | 0 | 0 | 0 | `interactive_scene.py:31`, `:282` |
| closed target ("HERO demo grip") [rad] | 0 | 0 | -1.2 | 1.57 | 1.2 | 1.2 | 1.2 | `interactive_scene.py:33`, `:283` |
| closed target after the scene's clip to the joint range | 0 | 0 | -1.2 | 1.57 | 1.2 | 1.2 | 1.2 (unchanged) | `sim/scene.py:264-265`; `b1_model.json` `closed_used_rad` |
| torque limit [N m] | 2.45 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | `interactive_scene.py:34`, `:284`; same as URDF effort `:1236`, `:1265`, `:1294`, `:1381`, `:1410`, `:1323`, `:1352` |
| squeezed when blocked | no | no | yes | yes | yes | yes | yes | `sim/scene.py:68` (`FLEXION_JOINT`) |

- **Closure blend:** goal = open·(1 - c) + closed·c, clipped to the joint ranges (`sim/scene.py:264-265`).
  - The oracle ramps c at 0.8 /s (`pick_place.py:71`, `:270-271`) until thumb_2 and (index_1 or middle_1) read > 0.3 N
    within the last 0.2 s (`:72-73`, `:285`).
  - It then finishes to c = 1 and holds 0.5 s (`:273-280`).
- **PD:** `tau = clip(kp (target - q) - kd q_dot, ±limit)`, recomputed every 1 ms substep (`sim/scene.py:273-277`).
  - kp = 1.2 N m/rad, kd = 0.04 N m s/rad (`interactive_scene.py:34`; read at `sim/scene.py:108`).
  - kp is multiplied by 3 (= 3.6) whenever the hand's closure > 0 (`sim/scene.py:67`, `:271`). This is ported from HERO's
    browser demo, `simulation.js:12`.
  - The target slews toward the goal at ≤ 2 rad/s (`sim/scene.py:275`).
- **Squeeze:** at full closure (c ≥ 1), a flexion joint that is short of its target by > 0.05 rad in the closing direction
  pushes with at least 0.6 x its limit = 0.84 N m (`sim/scene.py:67`, `:272`, `:279-281`). thumb_0 / thumb_1 are excluded.
  The closing direction is sign(closed - open): (0, -1, -1, +1, +1, +1, +1) (`sim/scene.py:112`).
- **Joint dynamics:** finger armature 0.001, damping 0.01 (`sim/scene.py:146-149` ← `interactive_scene.py:34`). Frictionloss is
  0 (compiled).
- **How deep the closed targets reach "into" the object** (nominal pose, hand at the closed target; `clearance_closed_mm`):
  - cylinder: thumb_2 -16.53, index_1 -19.60, middle_1 -47.32, palm -2.15 mm;
  - mug: thumb_1 -4.50, thumb_2 -26.03, index_1 -10.60, middle_1 -49.40, palm -11.65 mm.

  The targets lie well inside both objects, so the grip force comes from the PD error against the surface, capped by the
  limits above.

## 5. Contact settings that apply to fingertips, palm and mug

**Geoms** (compiled, `b1_model.json` → `right_hand_geoms`, `cups.*.geoms`):

| geom | type | friction (slide, torsion, roll) | condim | solref | solimp | set by |
|---|---|---|---|---|---|---|
| fingertips `right_hand_{thumb_2,index_1,middle_1}_link` | mesh (convex hull) | (1.5, 0.005, 0.0001) | 3 | (0.02, 1) | (0.9, 0.95, 0.001, 0.5, 2) | sliding 1.5: `sim/scene.py:195-201` ← `DEX3_SLIDING_FRICTION` `interactive_scene.py:101`; the rest MuJoCo defaults (nothing sets them) |
| palm `right_hand_palm_link` | mesh (convex hull) | (1.5, 0.005, 0.0001) | 3 | (0.02, 1) | default | same |
| other hand links (thumb_0, thumb_1 = box 20 x 30 x 20 mm (URDF `:1282-1287`), index_0, middle_0) | mesh / box | (1.5, 0.005, 0.0001) | 3 | (0.02, 1) | default | same |
| cup / every mug geom | cylinder (+ 3 capsules) | (0.35, 0.0028, 0.0001) | 4 | (0.008, 1) | default | `sim/scene.py:181-182` |
| wrist links `right_wrist_{roll,pitch,yaw}_link` | mesh | (1.0, 0.005, 0.0001) | 3 | (0.02, 1) | default | URDF / MuJoCo defaults; not "hand" geoms (`sim/scene.py:129`, `:195`) |

All of these geoms have solmix 1, priority 0, margin 0 and gap 0. Robot geoms have contype 1 and conaffinity 0
(`plant.py:201`).

**Explicit pairs** (`sim/scene.py:203-214`; compiled: `b1_model.json` → `cups.*.pairs`):
- Each cup geom gets one pair with each hand geom: 8 right-hand pairs for the cylinder, 32 for the mug (4 geoms x 8). All have
  one parameter set:
  - friction (1.0625, 1.0625, 0.00425, 0.0001, 0.0001), condim 4;
  - solref (0.014, 1) = the solmix-weighted mean of the cup's 0.008 and the hand's 0.02 (`:207-209`); solreffriction (0, 0),
    i.e. friction uses solref;
  - solimp (0.9, 0.95, 0.001, 0.5, 2); margin 0, gap 0.
- Table-top pairs: (0.35, 0.35, 0.0028, …). Coaster pairs: (0.5, 0.5, 0.004, …). Same condim / solref / solimp.
- The wrist links have **no** pair with the cup (`right_wrist_geoms_with_cup_pair` is empty). A wrist-cup contact would use
  MuJoCo's per-geom mixing. At the nominal grasp they are ≥ 34.8 mm from the mug.

**Which parameters MuJoCo actually applies.** Check: the cup sits at the nominal grasp pose with the hand at its closed target,
then `mj_forward` runs collision detection (`cups.*.nominal_grasp.contacts_hand_closed`).
- Every cup-hand contact carries its explicit pair's parameters (`all_match_explicit_pair: true`): 15 contacts on the cylinder,
  19 on the mug.
- None carries the per-geom max of 1.5.
- So the effective fingertip-mug and palm-mug friction is **mu = 1.0625** (sliding), torsional 0.00425 m, condim 4.

**Global options** (compiled, `b1_model.json` → `options`):

| option | value |
|---|---|
| cone | **pyramidal** |
| impratio | **1.0** |
| noslip_iterations | **0** (noslip_tolerance 1e-6) |
| solver | Newton (iterations 100, ls_iterations 50, tolerance 1e-8) |
| integrator / timestep | Euler, 0.001 s |
| global solref / solimp override | o_solref (0.02, 1), o_solimp (0.9, 0.95, 0.001, 0.5, 2), **inactive**: override flag off (enableflags 0, disableflags 0) |

Where these come from:
- All are MuJoCo's defaults. `build_model` sets only timestep and gravity (`plant.py:173-174`); the URDF's `<mujoco>` block sets
  only meshdir / discardvisual (URDF `:2-4`).
- HERO's `apply_adapted_solver` (`physics_profiles.py:119-131`) would set the same cone / impratio / noslip. It runs only for
  `physics_profile="sonic_deploy_adapted"` (`plant.py:369`, `:418-419`). `TableScene` uses the default `"hero"`
  (`sim/scene.py:99-100`, `plant.py:339`).
- For reference, since L3 notes that global options also reach the feet: the sole capsules have condim 3 and friction
  (1.0, 0.005, 0.0001) (`physics_profiles.py:20-21`), and the floor has (1.0, 0.005, 0.0001) (`plant.py:162`).

## 6. Dex3 joint names and ranges (right hand; compiled = URDF)

| joint | axis (parent link frame) | range [rad] | range [deg] | URDF lines |
|---|---|---|---|---|
| `right_hand_thumb_0_joint` | y | [-1.0472, 1.0472] | [-60, 60] | `:1231-1237` |
| `right_hand_thumb_1_joint` | z | [-1.0472, 0.7243] | [-60, 41.5] | `:1260-1266` |
| `right_hand_thumb_2_joint` | z | [-1.7453, 0] | [-100, 0] | `:1289-1295` |
| `right_hand_index_0_joint` | z | [0, 1.5708] | [0, 90] | `:1376-1382` |
| `right_hand_index_1_joint` | z | [0, 1.7453] | [0, 100] | `:1405-1411` |
| `right_hand_middle_0_joint` | z | [0, 1.5708] | [0, 90] | `:1318-1324` |
| `right_hand_middle_1_joint` | z | [0, 1.7453] | [0, 100] | `:1347-1353` |

Link layout in the palm frame (URDF joint origins):
- thumb_0 at (0.0255, 0, 0); thumb_1 at +(-0.0025, 0.0193, 0); thumb_2 at +(0, 0.0458, 0) (`:1232`, `:1261`, `:1290`). The
  thumb points along +y (out of the palm face) and curls toward +x as thumb_2 goes negative.
- index_0 at (0.0777, -0.0016, +0.0285) and middle_0 at (0.0777, -0.0016, -0.0285); each distal joint is 0.0458 further
  along x (`:1377`, `:1319`, `:1406`, `:1348`).
- The left hand is the mirror image. The oracle uses only the right hand.
