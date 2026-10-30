# M2 perception v3: gates, v2 -> v3, cross-view table

Made by `python -m perception.report --summary` from results/m2_perception_v3/<clip>/gates.json and crossview.json; v2 values from results/m2_perception_v2/<clip>/gates.json.
Clips without a report: none.

## Gates (v3)

G3 is information only (no threshold); G9 (hand-scale consistency) is removed in v3. G6 keeps its thresholds and is labelled as a known bias; downstream positions are validated by G7, and sim geometry uses the label values. S6 = video masks per the v2 A4 decision (the A4 mask test is unchanged and stays in results/m2_perception_v2/summary.md).

| gate | pass if | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|---|
| G1 | hand coverage: metric_valid >= 90 % from first hand_valid to release, no invalid run > 0.1 s in [contact - 0.5 s, place + 0.5 s], g and lift valid | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
| G2 | LiDAR residual, re-measured on the placed hand: median over metric_valid < 1.5 cm | **pass** median 0.47 cm over 161 metric_valid frames | **pass** median 0.57 cm over 366 metric_valid frames | **pass** median 0.47 cm over 258 metric_valid frames (2 more without a residual: frames [366, 367]) | **pass** median 0.76 cm over 316 metric_valid frames |
| G3 | information only (no threshold): palm-keypoint reprojection median over metric_valid, in px and in mm at the hand (px * z_c / fx) | **information** palm reprojection median 34.75 px; median at the hand 8.18 mm (px * z_c / fx per frame, then the median), over 161 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) | **information** palm reprojection median 43.09 px; median at the hand 10.94 mm (px * z_c / fx per frame, then the median), over 366 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) | **information** palm reprojection median 51.72 px; median at the hand 13.48 mm (px * z_c / fx per frame, then the median), over 260 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) | **information** palm reprojection median 14.17 px; median at the hand 3.52 mm (px * z_c / fx per frame, then the median), over 316 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) |
| G4 | static jitter: in every hold (smoothed speed < 2 cm/s for >= 0.3 s) raw palm std < 1 cm (N/A if no hold) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.07 cm) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.05 cm) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
| G5 | grasp height: z_grasp in [0.25 h, 0.85 h] (palm-to-axis distance at g and distance - r = information) | **pass** z_grasp 50.0 mm (pass 22.8-77.4); information: palm-to-axis distance 111.7 mm, distance - r 62.2 mm | **pass** z_grasp 44.6 mm (pass 22.8-77.4); information: palm-to-axis distance 102.1 mm, distance - r 52.6 mm | **pass** z_grasp 62.4 mm (pass 22.8-77.4); information: palm-to-axis distance 102.9 mm, distance - r 53.4 mm | **pass** z_grasp 62.7 mm (pass 22.8-77.4); information: palm-to-axis distance 96.2 mm, distance - r 46.7 mm |
| G6 | mug fit on the A4-chosen mask source: initial IoU >= 0.80, free-fit diameter and height within 5 mm of the label (height N/A if unobservable) | **fail** video masks: IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view); known bias +6–7 mm, cause unresolved | **fail** video masks: IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91); known bias +6–7 mm, cause unresolved | **fail** video masks: IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91); known bias +6–7 mm, cause unresolved | **fail** video masks: IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91); known bias +6–7 mm, cause unresolved |
| G7 | placement: horizontal distance final mug base centre to coaster centre < 2.5 cm | **pass** 0.81 cm | **pass** 2.15 cm | **pass** 1.04 cm | **pass** 0.54 cm |
| G8 | table frame: plane RMS < 6 mm, drift plane offset at the initial mug position < 1 cm (N/A if the end window < 10 frames; normal-angle drift = information), x within 20 deg of the expected reference angle | **pass** plane RMS 5.45 mm, drift offset -3.2 mm (pass), normal 1.29 deg (information), x (largest_total_extent) error 1.3 deg | **pass** plane RMS 4.48 mm, drift offset -0.4 mm (pass), normal 2.43 deg (information), x (largest_total_extent) error 1.5 deg | **pass** plane RMS 4.19 mm, drift offset 0.1 mm (pass), normal 1.46 deg (information), x (long_edge_extent) error 15.7 deg | **pass** plane RMS 4.60 mm, drift offset 1.8 mm (pass), normal 0.70 deg (information), x (largest_total_extent) error 7.6 deg |

## v2 -> v3: every gate per clip

v2 = results/m2_perception_v2 (its own gate definitions); v3 = this run. Gates whose rule did not change can still move because their inputs changed (palm placement without a scale fit, A2v3 contact rule).

| gate | what changed in v3 | clip | v2 | v3 |
|---|---|---|---|---|
| G1 | same rule; inputs changed (A1v3 palm placement, A2v3 contact rule) | 26d4118b0c (top-down) | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
|  |  | 4e103843db (90deg) | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
|  |  | 49aff04f7a (45deg) | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True |
|  |  | e8825dea53 (45deg) | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
| G2 | same rule; residual re-measured after the A1v3 placement (v2: after the scale + shift fit) | 26d4118b0c (top-down) | **pass** median 0.48 cm over 161 metric_valid frames | **pass** median 0.47 cm over 161 metric_valid frames |
|  |  | 4e103843db (90deg) | **pass** median 0.51 cm over 365 metric_valid frames (1 more without a residual: frames [90]) | **pass** median 0.57 cm over 366 metric_valid frames |
|  |  | 49aff04f7a (45deg) | **pass** median 0.51 cm over 258 metric_valid frames (2 more without a residual: frames [359, 367]) | **pass** median 0.47 cm over 258 metric_valid frames (2 more without a residual: frames [366, 367]) |
|  |  | e8825dea53 (45deg) | **pass** median 0.80 cm over 316 metric_valid frames | **pass** median 0.76 cm over 316 metric_valid frames |
| G3 | information only (v2: pass if the palm median < 8 px); no scale fit; also reported in mm at the hand | 26d4118b0c (top-down) | **fail** palm median 36.71 px over 161 metric_valid frames; all 21 keypoints 45.38 px (information) | **information** palm reprojection median 34.75 px; median at the hand 8.18 mm (px * z_c / fx per frame, then the median), over 161 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) |
|  |  | 4e103843db (90deg) | **fail** palm median 38.50 px over 366 metric_valid frames; all 21 keypoints 38.25 px (information) | **information** palm reprojection median 43.09 px; median at the hand 10.94 mm (px * z_c / fx per frame, then the median), over 366 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) |
|  |  | 49aff04f7a (45deg) | **fail** palm median 50.63 px over 260 metric_valid frames; all 21 keypoints 50.99 px (information) | **information** palm reprojection median 51.72 px; median at the hand 13.48 mm (px * z_c / fx per frame, then the median), over 260 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) |
|  |  | e8825dea53 (45deg) | **fail** palm median 12.08 px over 316 metric_valid frames; all 21 keypoints 22.85 px (information) | **information** palm reprojection median 14.17 px; median at the hand 3.52 mm (px * z_c / fx per frame, then the median), over 316 metric_valid frames; monocular hand-model limitation (orthographic model vs a 35–45 cm perspective camera) |
| G4 | same rule; input = the palm path of the A1v3 hand | 26d4118b0c (top-down) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.06 cm) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.07 cm) |
|  |  | 4e103843db (90deg) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.05 cm) | **pass** 1 hold(s), max raw std 0.07 cm (smoothed 0.05 cm) |
|  |  | 49aff04f7a (45deg) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
|  |  | e8825dea53 (45deg) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
| G5 | same rule; g moves with the A2v3 contact rule (lift is searched after contact; g = last valid frame <= lift - 3) | 26d4118b0c (top-down) | **pass** z_grasp 50.1 mm (pass 22.8-77.4); information: palm-to-axis distance 111.0 mm, distance - r 61.5 mm | **pass** z_grasp 50.0 mm (pass 22.8-77.4); information: palm-to-axis distance 111.7 mm, distance - r 62.2 mm |
|  |  | 4e103843db (90deg) | **pass** z_grasp 46.0 mm (pass 22.8-77.4); information: palm-to-axis distance 101.8 mm, distance - r 52.3 mm | **pass** z_grasp 44.6 mm (pass 22.8-77.4); information: palm-to-axis distance 102.1 mm, distance - r 52.6 mm |
|  |  | 49aff04f7a (45deg) | **pass** z_grasp 62.9 mm (pass 22.8-77.4); information: palm-to-axis distance 101.4 mm, distance - r 51.9 mm | **pass** z_grasp 62.4 mm (pass 22.8-77.4); information: palm-to-axis distance 102.9 mm, distance - r 53.4 mm |
|  |  | e8825dea53 (45deg) | **fail** z_grasp 88.7 mm (pass 22.8-77.4); information: palm-to-axis distance 89.6 mm, distance - r 40.1 mm; NOTE: g 148 comes before contact 149 (lift 151) | **pass** z_grasp 62.7 mm (pass 22.8-77.4); information: palm-to-axis distance 96.2 mm, distance - r 46.7 mm |
| G6 | same thresholds and inputs; now labelled "known bias +6–7 mm, cause unresolved" | 26d4118b0c (top-down) | **fail** video masks: IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view) | **fail** video masks: IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view); known bias +6–7 mm, cause unresolved |
|  |  | 4e103843db (90deg) | **fail** video masks: IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91) | **fail** video masks: IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91); known bias +6–7 mm, cause unresolved |
|  |  | 49aff04f7a (45deg) | **fail** video masks: IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91) | **fail** video masks: IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91); known bias +6–7 mm, cause unresolved |
|  |  | e8825dea53 (45deg) | **fail** video masks: IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91) | **fail** video masks: IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91); known bias +6–7 mm, cause unresolved |
| G7 | same rule, same inputs | 26d4118b0c (top-down) | **pass** 0.81 cm | **pass** 0.81 cm |
|  |  | 4e103843db (90deg) | **pass** 2.15 cm | **pass** 2.15 cm |
|  |  | 49aff04f7a (45deg) | **pass** 1.04 cm | **pass** 1.04 cm |
|  |  | e8825dea53 (45deg) | **pass** 0.54 cm | **pass** 0.54 cm |
| G8 | same rule; table-frame input (S7) reads the A1v3 hand | 26d4118b0c (top-down) | **pass** plane RMS 5.45 mm, drift offset -3.2 mm (pass), normal 1.29 deg (information), x (largest_total_extent) error 1.3 deg | **pass** plane RMS 5.45 mm, drift offset -3.2 mm (pass), normal 1.29 deg (information), x (largest_total_extent) error 1.3 deg |
|  |  | 4e103843db (90deg) | **pass** plane RMS 4.48 mm, drift offset -0.4 mm (pass), normal 2.43 deg (information), x (largest_total_extent) error 1.5 deg | **pass** plane RMS 4.48 mm, drift offset -0.4 mm (pass), normal 2.43 deg (information), x (largest_total_extent) error 1.5 deg |
|  |  | 49aff04f7a (45deg) | **pass** plane RMS 4.19 mm, drift offset 0.1 mm (pass), normal 1.46 deg (information), x (long_edge_extent) error 15.7 deg | **pass** plane RMS 4.19 mm, drift offset 0.1 mm (pass), normal 1.46 deg (information), x (long_edge_extent) error 15.7 deg |
|  |  | e8825dea53 (45deg) | **pass** plane RMS 4.60 mm, drift offset 1.8 mm (pass), normal 0.70 deg (information), x (largest_total_extent) error 7.6 deg | **pass** plane RMS 4.60 mm, drift offset 1.8 mm (pass), normal 0.70 deg (information), x (largest_total_extent) error 7.6 deg |
| G9 | removed in v3 (the hand scale is fixed at 1) | 26d4118b0c (top-down) | **fail** median s 1.154 (IQR 1.116-1.193, 161 frames) vs s_hand 1.022: +12.9 % | removed in v3 |
|  |  | 4e103843db (90deg) | **pass** median s 0.989 (IQR 0.966-1.025, 366 frames) vs s_hand 1.022: -3.2 % | removed in v3 |
|  |  | 49aff04f7a (45deg) | **fail** median s 0.847 (IQR 0.767-1.031, 260 frames) vs s_hand 1.022: -17.1 % | removed in v3 |
|  |  | e8825dea53 (45deg) | **pass** median s 1.036 (IQR 1.019-1.060, 316 frames) vs s_hand 1.022: +1.4 % | removed in v3 |

## Cross-view table

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) | min | max | max - min |
|---|---|---|---|---|---|---|---|
| coaster offset from initial cup, dx [cm] | 0.4 | 0.6 | -0.2 | -2.6 | -2.6 | 0.6 | 3.2 |
| coaster offset from initial cup, dy [cm] | -14.4 | -14.4 | -13.7 | -16.8 | -16.8 | -13.7 | 3.1 |
| coaster offset from initial cup, distance [cm] | 14.4 | 14.4 | 13.7 | 17.0 | 13.7 | 17.0 | 3.3 |
| approach yaw at g, from the handle direction [deg] | -139 | 156 | 153 | n/a | -139 | 156 | 296 |
| grasp height fraction z_grasp / h | 0.55 | 0.49 | 0.69 | 0.69 | 0.49 | 0.69 | 0.20 |
| finger-direction pitch at g [deg] | 11 | 21 | 14 | 22 | 11 | 22 | 11 |
| palm-normal elevation at g [deg] | -16 | -16 | -13 | -22 | -22 | -13 | 10 |
| lift height (max z in [lift, place] - z_grasp) [cm] | 8.9 | 6.9 | 6.8 | 12.1 | 6.8 | 12.1 | 5.2 |
| reach duration (contact - first hand_valid) [s] | 0.77 | 1.32 | 0.68 | 1.13 | 0.68 | 1.32 | 0.63 |
| transport duration (place - lift) [s] | 2.30 | 2.23 | 1.70 | 2.07 | 1.70 | 2.30 | 0.60 |
| initial handle yaw from table x [deg] | 4 | 82 | 72 | n/a | 4 | 82 | 78 |
| fitted coaster diameter [mm] | 103.4 | 103.7 | 107.2 | 103.0 | 103.0 | 107.2 | 4.2 |
| desk long-edge rule (S7 x_axis_method) | largest_total_extent | largest_total_extent | long_edge_extent | largest_total_extent | largest_total_extent: 3; long_edge_extent: 1 |  |  |
| t_rise: palm rises 3 cm above its lowest point near the cup (clip time) [s] | 2.97 | 3.88 | 3.10 | 2.42 | 2.42 | 3.88 | 1.47 |
| D_min: closest horizontal palm-to-initial-cup-axis distance before t_rise [mm] | 106.7 | 96.9 | 98.3 | 94.5 | 94.5 | 106.7 | 12.2 |
| mug mask source (S6, A4 decision) | video | video | video | video | video: 4 |  |  |

Angles at g come from T_palm_in_cup_at_grasp (smoothed palm); approach yaw and handle yaw are n/a when the handle yaw is invalid. Grasp height and lift height use S8's raw palm z. The two yaw columns lie in (-180, 180] deg; their min, max and max - min are taken linearly, without the wrap at +-180 deg. t_rise is the clip time (t - t[0]) of A2v3's rise frame; D_min is the closest approach BEFORE t_rise (contact = first frame within D_min + 1 cm). Clips where the D_min frame still comes after the lift (D_min is taken before t_rise, a 3 cm rise, and the lift is a 1.5 cm rise, so t_rise can follow the lift by a few frames; information, no rule changed; contact is not affected): 4e103843db (D_min frame 215, lift 214, t_rise 217, contact 162), e8825dea53 (D_min frame 144, lift 143, t_rise 145, contact 107). Clips where the grasp frame g (last valid frame <= lift - 3) comes before contact, so z_grasp, the angles at g and phase[g] are not taken in the grasp phase (information; no rule changed): none.

Mask source used by S6: video in every clip (the v2 A4 decision).

## Palm-normal flip (S9)

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|
| n flipped | False | False | False | False |

The palm-normal flip is the same in every clip where it was decided. Not decided (grasp frame undefined or no S9 output): none.

**Confound.** Viewpoint and execution vary together across these clips (views: top-down, 90deg, 45deg, 45deg); n = 4.
