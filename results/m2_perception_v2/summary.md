# M2 perception v2: gates, hand scale, v1 -> v2, mug mask test, cross-view table

Made by `python -m perception.report --summary` from results/m2_perception_v2/<clip>/gates.json, crossview.json and stages/; v1 values from results/m2_perception/<clip>/gates.json.
Clips without a report: none.

## Gates (v2)

| gate | pass if | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|---|
| G1 | hand coverage: metric_valid >= 90 % from first hand_valid to release, no invalid run > 0.1 s in [contact - 0.5 s, place + 0.5 s], g and lift valid | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
| G2 | LiDAR residual, re-measured on the S5b hand: median over metric_valid < 1.5 cm | **pass** median 0.48 cm over 161 metric_valid frames | **pass** median 0.51 cm over 365 metric_valid frames (1 more without a residual: frames [90]) | **pass** median 0.51 cm over 258 metric_valid frames (2 more without a residual: frames [359, 367]) | **pass** median 0.80 cm over 316 metric_valid frames |
| G3 | palm-keypoint reprojection after S5b pass 2: median over metric_valid < 8 px (all-21 median = information) | **fail** palm median 36.71 px over 161 metric_valid frames; all 21 keypoints 45.38 px (information) | **fail** palm median 38.50 px over 366 metric_valid frames; all 21 keypoints 38.25 px (information) | **fail** palm median 50.63 px over 260 metric_valid frames; all 21 keypoints 50.99 px (information) | **fail** palm median 12.08 px over 316 metric_valid frames; all 21 keypoints 22.85 px (information) |
| G4 | static jitter: in every hold (smoothed speed < 2 cm/s for >= 0.3 s) raw palm std < 1 cm (N/A if no hold) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.06 cm) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.05 cm) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
| G5 | grasp height: z_grasp in [0.25 h, 0.85 h] (palm-to-axis distance at g and distance - r = information) | **pass** z_grasp 50.1 mm (pass 22.8-77.4); information: palm-to-axis distance 111.0 mm, distance - r 61.5 mm | **pass** z_grasp 46.0 mm (pass 22.8-77.4); information: palm-to-axis distance 101.8 mm, distance - r 52.3 mm | **pass** z_grasp 62.9 mm (pass 22.8-77.4); information: palm-to-axis distance 101.4 mm, distance - r 51.9 mm | **fail** z_grasp 88.7 mm (pass 22.8-77.4); information: palm-to-axis distance 89.6 mm, distance - r 40.1 mm; NOTE: g 148 comes before contact 149 (lift 151) |
| G6 | mug fit on the A4-chosen mask source: initial IoU >= 0.80, free-fit diameter and height within 5 mm of the label (height N/A if unobservable) | **fail** video masks: IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view) | **fail** video masks: IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91) | **fail** video masks: IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91) | **fail** video masks: IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91) |
| G7 | placement: horizontal distance final mug base centre to coaster centre < 2.5 cm | **pass** 0.81 cm | **pass** 2.15 cm | **pass** 1.04 cm | **pass** 0.54 cm |
| G8 | table frame: plane RMS < 6 mm, drift plane offset at the initial mug position < 1 cm (N/A if the end window < 10 frames; normal-angle drift = information), x within 20 deg of the expected reference angle | **pass** plane RMS 5.45 mm, drift offset -3.2 mm (pass), normal 1.29 deg (information), x (largest_total_extent) error 1.3 deg | **pass** plane RMS 4.48 mm, drift offset -0.4 mm (pass), normal 2.43 deg (information), x (largest_total_extent) error 1.5 deg | **pass** plane RMS 4.19 mm, drift offset 0.1 mm (pass), normal 1.46 deg (information), x (long_edge_extent) error 15.7 deg | **pass** plane RMS 4.60 mm, drift offset 1.8 mm (pass), normal 0.70 deg (information), x (largest_total_extent) error 7.6 deg |
| G9 | hand-scale consistency: this clip's median s within +-10 % of s_hand | **fail** median s 1.154 (IQR 1.116-1.193, 161 frames) vs s_hand 1.022: +12.9 % | **pass** median s 0.989 (IQR 0.966-1.025, 366 frames) vs s_hand 1.022: -3.2 % | **fail** median s 0.847 (IQR 0.767-1.031, 260 frames) vs s_hand 1.022: -17.1 % | **pass** median s 1.036 (IQR 1.019-1.060, 316 frames) vs s_hand 1.022: +1.4 % |

## Hand scale (S5b)

s_hand = 1.0219.
s_hand is the median of the pass-1 scale s over the metric_valid frames of all 4 clips pooled; the hand is scaled by it in every frame of every clip. Per clip (pass-1 s over its metric_valid frames):

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|
| median s | 1.154 | 0.989 | 0.847 | 1.036 |
| IQR (25 - 75 %) | 1.116 - 1.193 | 0.966 - 1.025 | 0.767 - 1.031 | 1.019 - 1.060 |
| metric_valid frames | 161 | 366 | 260 | 316 |
| median s / s_hand - 1 [%] (G9: within +-10 %) | +12.9 | -3.2 | -17.1 | +1.4 |

## v1 -> v2: every gate per clip

v1 = results/m2_perception (its own gate definitions); v2 = this run. Gates whose rule did not change can still move because their inputs changed (S5b hand, relative contact rule, desk-edge fix, mask source).

| gate | what changed in v2 | clip | v1 | v2 |
|---|---|---|---|---|
| G1 | same rule; inputs changed (S5b hand, contact = D_min + 1 cm) | 26d4118b0c (top-down) | **fail** coverage undefined, longest invalid run undefined, g valid undefined, lift valid undefined (undefined: contact, grasp, lift, place, release) | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
|  |  | 4e103843db (90deg) | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
|  |  | 49aff04f7a (45deg) | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True |
|  |  | e8825dea53 (45deg) | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
| G2 | residual re-measured on the S5b hand (v1: the S5 hand) | 26d4118b0c (top-down) | **pass** median 0.53 cm over 161 metric_valid frames | **pass** median 0.48 cm over 161 metric_valid frames |
|  |  | 4e103843db (90deg) | **pass** median 0.55 cm over 366 metric_valid frames | **pass** median 0.51 cm over 365 metric_valid frames (1 more without a residual: frames [90]) |
|  |  | 49aff04f7a (45deg) | **pass** median 0.38 cm over 260 metric_valid frames | **pass** median 0.51 cm over 258 metric_valid frames (2 more without a residual: frames [359, 367]) |
|  |  | e8825dea53 (45deg) | **pass** median 0.82 cm over 316 metric_valid frames | **pass** median 0.80 cm over 316 metric_valid frames |
| G3 | 5 palm keypoints after S5b pass 2 (v1: all 21 keypoints of the S5 hand, plumbing check) | 26d4118b0c (top-down) | **fail** median 41.75 px over 161 metric_valid frames | **fail** palm median 36.71 px over 161 metric_valid frames; all 21 keypoints 45.38 px (information) |
|  |  | 4e103843db (90deg) | **fail** median 61.23 px over 366 metric_valid frames | **fail** palm median 38.50 px over 366 metric_valid frames; all 21 keypoints 38.25 px (information) |
|  |  | 49aff04f7a (45deg) | **fail** median 80.98 px over 260 metric_valid frames | **fail** palm median 50.63 px over 260 metric_valid frames; all 21 keypoints 50.99 px (information) |
|  |  | e8825dea53 (45deg) | **fail** median 30.91 px over 316 metric_valid frames | **fail** palm median 12.08 px over 316 metric_valid frames; all 21 keypoints 22.85 px (information) |
| G4 | same rule; input = the palm path of the S5b hand | 26d4118b0c (top-down) | **pass** 1 hold(s), max raw std 0.06 cm (smoothed 0.06 cm) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.06 cm) |
|  |  | 4e103843db (90deg) | **pass** 1 hold(s), max raw std 0.05 cm (smoothed 0.04 cm) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.05 cm) |
|  |  | 49aff04f7a (45deg) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
|  |  | e8825dea53 (45deg) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
| G5 | z_grasp only (v1 also required the palm-to-axis distance in [r + 1, r + 5] cm); its input g moves with the relative contact rule (lift is searched after contact; g = last valid frame <= lift - 3) | 26d4118b0c (top-down) | **fail** d nan mm (pass 59.5-99.5), z_grasp nan mm (pass 22.8-77.4) (grasp frame g undefined) | **pass** z_grasp 50.1 mm (pass 22.8-77.4); information: palm-to-axis distance 111.0 mm, distance - r 61.5 mm |
|  |  | 4e103843db (90deg) | **pass** d 99.4 mm (pass 59.5-99.5), z_grasp 47.5 mm (pass 22.8-77.4) | **pass** z_grasp 46.0 mm (pass 22.8-77.4); information: palm-to-axis distance 101.8 mm, distance - r 52.3 mm |
|  |  | 49aff04f7a (45deg) | **pass** d 96.1 mm (pass 59.5-99.5), z_grasp 61.8 mm (pass 22.8-77.4) | **pass** z_grasp 62.9 mm (pass 22.8-77.4); information: palm-to-axis distance 101.4 mm, distance - r 51.9 mm |
|  |  | e8825dea53 (45deg) | **pass** d 96.6 mm (pass 59.5-99.5), z_grasp 62.7 mm (pass 22.8-77.4) | **fail** z_grasp 88.7 mm (pass 22.8-77.4); information: palm-to-axis distance 89.6 mm, distance - r 40.1 mm; NOTE: g 148 comes before contact 149 (lift 151) |
| G6 | same thresholds, on the A4-chosen mask source (v1: video masks) | 26d4118b0c (top-down) | **fail** IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view) | **fail** video masks: IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view) |
|  |  | 4e103843db (90deg) | **fail** IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91) | **fail** video masks: IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91) |
|  |  | 49aff04f7a (45deg) | **fail** IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91) | **fail** video masks: IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91) |
|  |  | e8825dea53 (45deg) | **fail** IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91) | **fail** video masks: IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91) |
| G7 | same rule; S6 input follows the A4 mask source | 26d4118b0c (top-down) | **pass** 0.81 cm | **pass** 0.81 cm |
|  |  | 4e103843db (90deg) | **pass** 2.15 cm | **pass** 2.15 cm |
|  |  | 49aff04f7a (45deg) | **pass** 1.04 cm | **pass** 1.04 cm |
|  |  | e8825dea53 (45deg) | **pass** 0.54 cm | **pass** 0.54 cm |
| G8 | drift = plane offset at the initial mug < 1 cm (v1: failed if offset > 1 cm OR normal angle > 2 deg); desk-edge fix A3 | 26d4118b0c (top-down) | **pass** plane RMS 5.45 mm, drift offset -3.2 mm, normal 1.29 deg (pass), x (largest_total_extent) error 0.8 deg | **pass** plane RMS 5.45 mm, drift offset -3.2 mm (pass), normal 1.29 deg (information), x (largest_total_extent) error 1.3 deg |
|  |  | 4e103843db (90deg) | **fail** plane RMS 4.48 mm, drift offset -0.4 mm, normal 2.43 deg (fail), x (largest_total_extent) error 1.7 deg | **pass** plane RMS 4.48 mm, drift offset -0.4 mm (pass), normal 2.43 deg (information), x (largest_total_extent) error 1.5 deg |
|  |  | 49aff04f7a (45deg) | **fail** plane RMS 4.19 mm, drift offset 0.1 mm, normal 1.46 deg (pass), x (long_edge_extent) error 23.9 deg | **pass** plane RMS 4.19 mm, drift offset 0.1 mm (pass), normal 1.46 deg (information), x (long_edge_extent) error 15.7 deg |
|  |  | e8825dea53 (45deg) | **pass** plane RMS 4.60 mm, drift offset 1.8 mm, normal 0.70 deg (pass), x (long_edge_extent) error 11.0 deg | **pass** plane RMS 4.60 mm, drift offset 1.8 mm (pass), normal 0.70 deg (information), x (largest_total_extent) error 7.6 deg |
| G9 | new in v2 | 26d4118b0c (top-down) | - | **fail** median s 1.154 (IQR 1.116-1.193, 161 frames) vs s_hand 1.022: +12.9 % |
|  |  | 4e103843db (90deg) | - | **pass** median s 0.989 (IQR 0.966-1.025, 366 frames) vs s_hand 1.022: -3.2 % |
|  |  | 49aff04f7a (45deg) | - | **fail** median s 0.847 (IQR 0.767-1.031, 260 frames) vs s_hand 1.022: -17.1 % |
|  |  | e8825dea53 (45deg) | - | **pass** median s 1.036 (IQR 1.019-1.060, 316 frames) vs s_hand 1.022: +1.4 % |

## Mug mask test (A4, both static windows)

2-DOF fit IoU and the 4-DOF free fit, on the SAM 2 video masks (half res, as v1) and on the SAM 2 image masks of the full-res frames, in the initial and the final static window. Label: diameter 99 mm, height 91 mm. The decision rule below uses the initial window only; "(fit bound)" = the free fit stopped at its bound (r 30-70 mm, h 50-130 mm).

| clip | window | mask source | IoU (2-DOF) | free-fit diameter [mm] | free-fit height [mm] |
|---|---|---|---|---|---|
| 26d4118b0c (top-down) | init | video | 0.918 | 105.1 | N/A (unobservable) |
| 26d4118b0c (top-down) | init | full_res_image | 0.926 | 114.3 | N/A (unobservable) |
| 26d4118b0c (top-down) | final | video | 0.872 | 104.6 | N/A (unobservable) |
| 26d4118b0c (top-down) | final | full_res_image | 0.879 | 104.1 | N/A (unobservable) |
| 4e103843db (90deg) | init | video | 0.943 | 105.7 | 86.9 |
| 4e103843db (90deg) | init | full_res_image | 0.945 | 105.8 | 86.2 |
| 4e103843db (90deg) | final | video | 0.901 | 107.5 | 89.7 |
| 4e103843db (90deg) | final | full_res_image | 0.658 | 107.9 | 87.1 |
| 49aff04f7a (45deg) | init | video | 0.947 | 106.1 | 85.8 |
| 49aff04f7a (45deg) | init | full_res_image | 0.950 | 106.0 | 85.3 |
| 49aff04f7a (45deg) | final | video | 0.882 | 104.2 | 97.1 |
| 49aff04f7a (45deg) | final | full_res_image | 0.609 | 105.5 | 50.0 (fit bound) |
| e8825dea53 (45deg) | init | video | 0.938 | 105.8 | 88.1 |
| e8825dea53 (45deg) | init | full_res_image | 0.942 | 105.7 | 87.8 |
| e8825dea53 (45deg) | final | video | 0.911 | 103.2 | 92.1 |
| e8825dea53 (45deg) | final | full_res_image | 0.922 | 102.3 | 92.7 |

Decision (pre-set rule, recorded by S6 in objects.json): full_res_image for all clips if |full-res init free-fit diameter - label| <= 5 mm on >= 3 of 4 clips, else video. 0 of 4 clips within 5 mm of the 99.0 mm label (none); the rule needs 3 -> keep the video masks. Chosen: **video**.
Mask source used by S6: video in every clip.

## Cross-view table

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) | min | max | max - min |
|---|---|---|---|---|---|---|---|
| coaster offset from initial cup, dx [cm] | 0.4 | 0.6 | -0.2 | -2.6 | -2.6 | 0.6 | 3.2 |
| coaster offset from initial cup, dy [cm] | -14.4 | -14.4 | -13.7 | -16.8 | -16.8 | -13.7 | 3.1 |
| coaster offset from initial cup, distance [cm] | 14.4 | 14.4 | 13.7 | 17.0 | 13.7 | 17.0 | 3.3 |
| approach yaw at g, from the handle direction [deg] | -139 | 157 | 153 | n/a | -139 | 157 | 295 |
| grasp height fraction z_grasp / h | 0.55 | 0.51 | 0.69 | 0.98 | 0.51 | 0.98 | 0.47 |
| finger-direction pitch at g [deg] | 11 | 21 | 14 | 23 | 11 | 23 | 13 |
| palm-normal elevation at g [deg] | -16 | -16 | -13 | -24 | -24 | -13 | 11 |
| lift height (max z in [lift, place] - z_grasp) [cm] | 8.9 | 6.8 | 6.9 | 9.4 | 6.8 | 9.4 | 2.6 |
| reach duration (contact - first hand_valid) [s] | 0.77 | 1.40 | 0.72 | 1.83 | 0.72 | 1.83 | 1.12 |
| transport duration (place - lift) [s] | 2.30 | 2.23 | 1.70 | 1.67 | 1.67 | 2.30 | 0.63 |
| initial handle yaw from table x [deg] | 4 | 82 | 72 | n/a | 4 | 82 | 78 |
| fitted coaster diameter [mm] | 103.4 | 103.7 | 107.2 | 103.0 | 103.0 | 107.2 | 4.2 |
| desk long-edge rule (S7 x_axis_method) | largest_total_extent | largest_total_extent | long_edge_extent | largest_total_extent | largest_total_extent: 3; long_edge_extent: 1 |  |  |
| D_min: closest horizontal palm-to-initial-cup-axis distance [mm] | 106.5 | 93.3 | 95.6 | 78.0 | 78.0 | 106.5 | 28.4 |
| hand scale: median pass-1 s over metric_valid frames | 1.154 | 0.989 | 0.847 | 1.036 | 0.847 | 1.154 | 0.307 |
| mug mask source (S6, A4 decision) | video | video | video | video | video: 4 |  |  |

Angles at g come from T_palm_in_cup_at_grasp (smoothed palm); approach yaw and handle yaw are n/a when the handle yaw is invalid. Grasp height and lift height use S8's raw palm z. The two yaw columns lie in (-180, 180] deg; their min, max and max - min are taken linearly, without the wrap at +-180 deg. D_min is S8's closest approach (contact = first frame within D_min + 1 cm); clips where the palm was at or above the mug top at that frame (S8 d_min_z_flag): 4e103843db, 49aff04f7a, e8825dea53. Clips where the D_min frame comes after the lift (the closest approach happens while the hand carries the mug): 4e103843db (D_min frame 225, lift 214), 49aff04f7a (D_min frame 201, lift 182), e8825dea53 (D_min frame 159, lift 151). Clips where the grasp frame g (last valid frame <= lift - 3) comes before contact, so z_grasp, the angles at g and phase[g] are not taken in the grasp phase (information; no rule changed): e8825dea53 (g 148, contact 149, lift 151).

## Palm-normal flip (S9)

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|
| n flipped | False | False | False | False |

The palm-normal flip is the same in every clip where it was decided. Not decided (grasp frame undefined or no S9 output): none.

**Confound.** Viewpoint and execution vary together across these clips (views: top-down, 90deg, 45deg, 45deg); n = 4.
