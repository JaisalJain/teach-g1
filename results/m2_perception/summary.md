# M2 perception: gates and cross-view table

Made by `python -m perception.report --summary` from results/m2_perception/<clip>/gates.json and crossview.json.
Clips without a report: none.

## Gates

| gate | pass if | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|---|
| G1 | hand coverage: metric_valid >= 90 % from first hand_valid to release, no invalid run > 0.1 s in [contact - 0.5 s, place + 0.5 s], g and lift valid | **fail** coverage undefined, longest invalid run undefined, g valid undefined, lift valid undefined (undefined: contact, grasp, lift, place, release) | **pass** coverage 100.0%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True | **pass** coverage 99.2%, longest invalid run 0.017 s (1 frames), g valid True, lift valid True | **pass** coverage 99.7%, longest invalid run 0.000 s (0 frames), g valid True, lift valid True |
| G2 | LiDAR residual: median over metric_valid < 1.5 cm | **pass** median 0.53 cm over 161 metric_valid frames | **pass** median 0.55 cm over 366 metric_valid frames | **pass** median 0.38 cm over 260 metric_valid frames | **pass** median 0.82 cm over 316 metric_valid frames |
| G3 | plumbing reprojection (our transforms, not WiLoR accuracy): median over metric_valid < 8 px | **fail** median 41.75 px over 161 metric_valid frames | **fail** median 61.23 px over 366 metric_valid frames | **fail** median 80.98 px over 260 metric_valid frames | **fail** median 30.91 px over 316 metric_valid frames |
| G4 | static jitter: in every hold (smoothed speed < 2 cm/s for >= 0.3 s) raw palm std < 1 cm (N/A if no hold) | **pass** 1 hold(s), max raw std 0.06 cm (smoothed 0.06 cm) | **pass** 1 hold(s), max raw std 0.05 cm (smoothed 0.04 cm) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) | **N/A** no static hold (N/A: no run with smoothed palm speed < 2 cm/s for >= 0.3 s) |
| G5 | grasp geometry at g: palm-to-axis distance in [r + 1, r + 5] cm and z_grasp in [0.25 h, 0.85 h] | **fail** d nan mm (pass 59.5-99.5), z_grasp nan mm (pass 22.8-77.4) (grasp frame g undefined) | **pass** d 99.4 mm (pass 59.5-99.5), z_grasp 47.5 mm (pass 22.8-77.4) | **pass** d 96.1 mm (pass 59.5-99.5), z_grasp 61.8 mm (pass 22.8-77.4) | **pass** d 96.6 mm (pass 59.5-99.5), z_grasp 62.7 mm (pass 22.8-77.4) |
| G6 | mug fit: initial IoU >= 0.80, free-fit diameter and height within 5 mm of the label (height N/A if unobservable) | **fail** IoU 0.918, diameter 105.1 mm (label 99), height N/A (unobservable from this view) | **fail** IoU 0.943, diameter 105.7 mm (label 99), height 86.9 mm (label 91) | **fail** IoU 0.947, diameter 106.1 mm (label 99), height 85.8 mm (label 91) | **fail** IoU 0.938, diameter 105.8 mm (label 99), height 88.1 mm (label 91) |
| G7 | placement: horizontal distance final mug base centre to coaster centre < 2.5 cm | **pass** 0.81 cm | **pass** 2.15 cm | **pass** 1.04 cm | **pass** 0.54 cm |
| G8 | table frame: plane RMS < 6 mm, drift check passes (or N/A), x within 20 deg of the expected reference angle | **pass** plane RMS 5.45 mm, drift offset -3.2 mm, normal 1.29 deg (pass), x (largest_total_extent) error 0.8 deg | **fail** plane RMS 4.48 mm, drift offset -0.4 mm, normal 2.43 deg (fail), x (largest_total_extent) error 1.7 deg | **fail** plane RMS 4.19 mm, drift offset 0.1 mm, normal 1.46 deg (pass), x (long_edge_extent) error 23.9 deg | **pass** plane RMS 4.60 mm, drift offset 1.8 mm, normal 0.70 deg (pass), x (long_edge_extent) error 11.0 deg |

## Cross-view table

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) | min | max | max - min |
|---|---|---|---|---|---|---|---|
| coaster offset from initial cup, dx [cm] | 0.3 | 0.7 | -8.9 | -3.6 | -8.9 | 0.7 | 9.6 |
| coaster offset from initial cup, dy [cm] | -14.4 | -14.4 | -10.4 | -16.6 | -16.6 | -10.4 | 6.2 |
| coaster offset from initial cup, distance [cm] | 14.4 | 14.4 | 13.7 | 17.0 | 13.7 | 17.0 | 3.3 |
| approach yaw at g, from the handle direction [deg] | n/a | 158 | 151 | n/a | 151 | 158 | 6 |
| grasp height fraction z_grasp / h | n/a | 0.52 | 0.68 | 0.69 | 0.52 | 0.69 | 0.17 |
| finger-direction pitch at g [deg] | n/a | 21 | 14 | 22 | 14 | 22 | 8 |
| palm-normal elevation at g [deg] | n/a | -16 | -13 | -22 | -22 | -13 | 10 |
| lift height (max z in [lift, place] - z_grasp) [cm] | n/a | 6.5 | 6.6 | 11.9 | 6.5 | 11.9 | 5.3 |
| reach duration (contact - first hand_valid) [s] | n/a | 1.43 | 0.72 | 1.25 | 0.72 | 1.43 | 0.72 |
| transport duration (place - lift) [s] | n/a | 2.22 | 1.68 | 2.12 | 1.68 | 2.22 | 0.53 |
| initial handle yaw from table x [deg] | 4 | 82 | 32 | n/a | 4 | 82 | 78 |
| fitted coaster diameter [mm] | 103.4 | 103.7 | 107.2 | 103.0 | 103.0 | 107.2 | 4.2 |
| desk long-edge rule (S7 x_axis_method) | largest_total_extent | largest_total_extent | long_edge_extent | long_edge_extent | largest_total_extent: 2; long_edge_extent: 2 |  |  |

Angles at g come from T_palm_in_cup_at_grasp (smoothed palm); approach yaw and handle yaw are n/a when the handle yaw is invalid. Grasp height and lift height use S8's raw palm z. The two yaw columns lie in (-180, 180] deg; their min, max and max - min are taken linearly, without the wrap at +-180 deg.

## Palm-normal flip (S9)

| | 26d4118b0c (top-down) | 4e103843db (90deg) | 49aff04f7a (45deg) | e8825dea53 (45deg) |
|---|---|---|---|---|
| n flipped | undecided | False | False | False |

The palm-normal flip is the same in every clip where it was decided. Not decided (grasp frame undefined or no S9 output): 26d4118b0c.

**Confound.** Viewpoint and execution vary together across these clips (views: top-down, 90deg, 45deg, 45deg); n = 4.
