# M3 kinematic retargeting: reachability on HERO's whole-body IK

Made by `python -m retarget.run_m3 summarize` from results/m3_retarget/runs/*.json, verify.json, run_resources.json, r0_after_m3/runs/R0_after_m3.json and videos.json. Kinematic only: no physics beyond PickPlace.run's 2 s setup hold.

Reachable = FK error at G and at P < 1 cm and < 10 deg, AND path max position error < 2 cm. Joint-limit hit = a path frame in which any of the 14 arm joints is within 0.01 rad of its URDF limit. Distances: mj_geomDistance from the 8 right-hand collision geoms (open hand) to cup_geom / cup_handle_* at the clip's frame i_G; negative = overlap.

## Checks

- Palm axes (frames.verify_palm_axes): pass.
- plan_path(return_fk) default output byte-identical (A vs B vs C): True; setup repeatable: True.
- R0 after the plan_path change: 20/20, identical to M1 in 20/20 trials, to B3 R0 in 20/20.
- verify, run and videos ran on the current code (SHA-256 of 8 files): verify True, run True, videos True.
- Path sample 0 vs palm_start: max 0.090 mm (limit 1 mm, asserted per run).
- Clip -> plant mapping at i_G: 92/92 within 1 mm (max 0.0005 mm).
- Recorded layouts whose coaster place palm (spec formula: coaster - legacy palm offset) lies outside PALM_BOX: 26d4118b0c, 4e103843db, 49aff04f7a, e8825dea53. Information, with coaster + offset (palm = cup + offset, as sample_layouts implies): 26d4118b0c, 4e103843db, 49aff04f7a, e8825dea53.

## Per clip

| clip | phi_h (deg) | phi_ora (deg) | palm-point azimuth | h_frac_h | h_frac_ora | k | a0 time (s) | clamped frames, recorded (b_level / b_tilt) | clamped, max over layouts (b_level / b_tilt) |
|---|---|---|---|---|---|---|---|---|---|
| 26d4118b0c | 81.9 | 45.0 | -135.3 deg | 0.554 | 0.714 | 1.588 | 1.267 | 0 / 0 | 0 / 0 |
| 4e103843db | 85.8 | 45.0 | -122.1 deg | 0.480 | 0.714 | 1.091 | 1.801 | 0 / 12 | 0 / 12 |
| 49aff04f7a | 81.7 | 45.0 | -134.9 deg | 0.692 | 0.714 | 1.739 | 1.884 | 0 / 0 | 0 / 0 |
| e8825dea53 | 61.7 | 45.0 | -137.0 deg | 0.700 | 0.714 | 1.520 | 0.700 | 0 / 0 | 0 / 0 |

## Reachability (runs reachable / runs)

| clip | a_naive | b_level | b_tilt |
|---|---|---|---|
| 26d4118b0c | 0/1 | 0/11 | 0/11 |
| 4e103843db | 0/1 | 0/11 | 0/11 |
| 49aff04f7a | 0/1 | 0/11 | 0/11 |
| e8825dea53 | 0/1 | 10/11 | 0/11 |
| all clips | 0/4 | 10/44 | 0/44 |

a_naive runs only on the recorded layout: its path ignores the layout, so its reachability holds for every layout.

Which part of the rule passed (runs passing / runs): at G (< 1 cm and < 10 deg), at P (same), path (max position error < 2 cm).

| clip | variant | at G | at P | path max < 2 cm | reachable |
|---|---|---|---|---|---|
| 26d4118b0c | a_naive | 0/1 | 0/1 | 0/1 | 0/1 |
| 26d4118b0c | b_level | 0/11 | 8/11 | 0/11 | 0/11 |
| 26d4118b0c | b_tilt | 0/11 | 3/11 | 0/11 | 0/11 |
| 4e103843db | a_naive | 0/1 | 0/1 | 0/1 | 0/1 |
| 4e103843db | b_level | 1/11 | 6/11 | 0/11 | 0/11 |
| 4e103843db | b_tilt | 0/11 | 0/11 | 0/11 | 0/11 |
| 49aff04f7a | a_naive | 0/1 | 0/1 | 0/1 | 0/1 |
| 49aff04f7a | b_level | 5/11 | 9/11 | 0/11 | 0/11 |
| 49aff04f7a | b_tilt | 0/11 | 6/11 | 0/11 | 0/11 |
| e8825dea53 | a_naive | 0/1 | 0/1 | 0/1 | 0/1 |
| e8825dea53 | b_level | 10/11 | 11/11 | 11/11 | 10/11 |
| e8825dea53 | b_tilt | 0/11 | 0/11 | 0/11 | 0/11 |

Information: the largest FK position error per phase (cm), recorded layout. The prelude (arms at the sides -> over the table) is robot-specific; the other phases carry my demo.

| run | where the path max is | prelude | to_a0 | replay | approach | pause | close | transport | release | retreat |
|---|---|---|---|---|---|---|---|---|---|---|
| 26d4118b0c__a_naive__recorded | replay | 2.67 | 0.56 | 3.60 | - | - | - | - | 1.73 | - |
| 26d4118b0c__b_level__recorded | prelude | 2.82 | - | - | 1.79 | 0.32 | 1.40 | 1.33 | 1.26 | 1.19 |
| 26d4118b0c__b_tilt__recorded | approach | 2.67 | - | - | 3.06 | 0.37 | 2.86 | 2.61 | 1.83 | 1.78 |
| 4e103843db__a_naive__recorded | replay | 3.26 | 4.45 | 5.88 | - | - | - | - | 2.74 | - |
| 4e103843db__b_level__recorded | prelude | 2.86 | - | - | 1.97 | 0.37 | 1.31 | 1.71 | 1.70 | 1.62 |
| 4e103843db__b_tilt__recorded | approach | 3.26 | - | - | 6.20 | 2.47 | 5.82 | 5.35 | 4.31 | 4.25 |
| 49aff04f7a__a_naive__recorded | replay | 2.67 | 5.99 | 6.07 | - | - | - | - | 4.83 | - |
| 49aff04f7a__b_level__recorded | prelude | 2.82 | - | - | 1.19 | 0.26 | 0.97 | 1.42 | 1.12 | 1.05 |
| 49aff04f7a__b_tilt__recorded | approach | 2.67 | - | - | 2.89 | 0.37 | 2.62 | 2.32 | 1.43 | 1.35 |
| e8825dea53__a_naive__recorded | replay | 2.99 | 0.75 | 9.86 | - | - | - | - | 2.51 | - |
| e8825dea53__b_level__recorded | prelude | 1.90 | - | - | 1.16 | 0.18 | 0.81 | 0.94 | 0.76 | 0.72 |
| e8825dea53__b_tilt__recorded | approach | 2.99 | - | - | 3.77 | 0.30 | 3.38 | 3.10 | 1.96 | 1.90 |

## Hand-mug and hand-handle distances at G (mm; open hand; negative = overlap)

Reference (information): at the oracle's own planned grasp pose (mug_v1 offset, no IK error, open hand) the hand-mug distance is -2.4 mm (right_hand_palm_link), hand-handle 79.693 mm (results/m1_grasp_fix/v3/l1_check.json, b_mug_L1).

| clip | variant | hand-mug, recorded | hand-mug min / median / max | hand-handle, recorded | hand-handle min / median / max |
|---|---|---|---|---|---|
| 26d4118b0c | a_naive | 10.9 | 10.9 / 10.9 / 10.9 | 74.8 | 74.8 / 74.8 / 74.8 |
| 26d4118b0c | b_level | -7.0 | -10.2 / -8.3 / -5.4 | 50.8 | 43.7 / 57.1 / 82.2 |
| 26d4118b0c | b_tilt | -9.2 | -12.6 / -10.7 / -7.5 | 52.1 | 44.6 / 58.2 / 82.8 |
| 4e103843db | a_naive | 2.3 | 2.3 / 2.3 / 2.3 | 80.2 | 80.2 / 80.2 / 80.2 |
| 4e103843db | b_level | -7.6 | -11.7 / -9.1 / -5.9 | 80.9 | 42.5 / 66.0 / 81.5 |
| 4e103843db | b_tilt | -5.8 | -12.1 / -7.7 / -4.1 | 84.4 | 47.9 / 71.9 / 84.6 |
| 49aff04f7a | a_naive | 6.2 | 6.2 / 6.2 / 6.2 | 89.1 | 89.1 / 89.1 / 89.1 |
| 49aff04f7a | b_level | -6.4 | -9.7 / -7.7 / -4.8 | 82.5 | 45.3 / 68.0 / 83.3 |
| 49aff04f7a | b_tilt | -7.4 | -11.0 / -8.8 / -5.6 | 83.6 | 46.0 / 68.8 / 83.9 |
| e8825dea53 | a_naive | -2.7 | -2.7 / -2.7 / -2.7 | 75.4 | 75.4 / 75.4 / 75.4 |
| e8825dea53 | b_level | -4.2 | -5.7 / -4.9 / -3.3 | 83.9 | 49.0 / 69.5 / 84.8 |
| e8825dea53 | b_tilt | -1.6 | -4.4 / -2.5 / 1.4 | 87.1 | 54.9 / 75.7 / 90.6 |

## Recorded layout

| run | reachable | pos err max / G / P (cm) | ori err max / G / P (deg) | qp failures | limit-hit frames / frames | clamped | bump scale | hand-mug (mm) | hand-handle (mm) |
|---|---|---|---|---|---|---|---|---|---|
| 26d4118b0c__a_naive__recorded | no | 3.60 / 3.15 / 2.04 | 7.9 / 0.8 / 0.4 | 0 | 0 / 541 | 0 | - | 10.9 | 74.8 |
| 26d4118b0c__b_level__recorded | no | 2.82 / 1.40 / 1.27 | 0.7 / 0.2 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -7.0 | 50.8 |
| 26d4118b0c__b_tilt__recorded | no | 3.06 / 2.86 / 1.83 | 0.8 / 0.7 / 0.5 | 0 | 0 / 664 | 0 | 1.000 | -9.2 | 52.1 |
| 4e103843db__a_naive__recorded | no | 5.88 / 4.83 / 4.04 | 1.4 / 1.2 / 1.0 | 0 | 0 / 437 | 0 | - | 2.3 | 80.2 |
| 4e103843db__b_level__recorded | no | 2.86 / 1.31 / 1.70 | 0.7 / 0.2 / 0.2 | 0 | 0 / 589 | 0 | 1.000 | -7.6 | 80.9 |
| 4e103843db__b_tilt__recorded | no | 6.20 / 5.83 / 4.32 | 1.4 / 1.3 / 1.1 | 0 | 0 / 589 | 12 | 1.000 | -5.8 | 84.4 |
| 49aff04f7a__a_naive__recorded | no | 6.07 / 2.18 / 2.09 | 1.4 / 0.5 / 0.5 | 0 | 0 / 493 | 0 | - | 6.2 | 89.1 |
| 49aff04f7a__b_level__recorded | no | 2.82 / 0.97 / 1.14 | 0.7 / 0.1 / 0.1 | 0 | 0 / 603 | 0 | 1.000 | -6.4 | 82.5 |
| 49aff04f7a__b_tilt__recorded | no | 2.89 / 2.62 / 1.45 | 0.8 / 0.7 / 0.4 | 0 | 0 / 603 | 0 | 1.000 | -7.4 | 83.6 |
| e8825dea53__a_naive__recorded | no | 9.86 / 1.46 / 1.99 | 3.1 / 0.4 / 0.5 | 0 | 0 / 510 | 0 | - | -2.7 | 75.4 |
| e8825dea53__b_level__recorded | yes | 1.90 / 0.81 / 0.77 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -4.2 | 83.9 |
| e8825dea53__b_tilt__recorded | no | 3.77 / 3.38 / 1.96 | 0.9 / 0.8 / 0.6 | 0 | 0 / 640 | 0 | 0.888 | -1.6 | 87.1 |

## All runs

| run | reachable | pos err max / G / P (cm) | ori err max / G / P (deg) | qp failures | limit-hit frames / frames | clamped | bump scale | hand-mug (mm) | hand-handle (mm) |
|---|---|---|---|---|---|---|---|---|---|
| 26d4118b0c__a_naive__recorded | no | 3.60 / 3.15 / 2.04 | 7.9 / 0.8 / 0.4 | 0 | 0 / 541 | 0 | - | 10.9 | 74.8 |
| 26d4118b0c__b_level__recorded | no | 2.82 / 1.40 / 1.27 | 0.7 / 0.2 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -7.0 | 50.8 |
| 26d4118b0c__b_level__trial00 | no | 2.82 / 1.29 / 1.09 | 0.7 / 0.2 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -5.6 | 48.2 |
| 26d4118b0c__b_level__trial01 | no | 2.83 / 1.48 / 0.93 | 0.7 / 0.1 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -5.4 | 76.4 |
| 26d4118b0c__b_level__trial02 | no | 2.82 / 1.34 / 0.74 | 0.7 / 0.2 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -5.9 | 66.6 |
| 26d4118b0c__b_level__trial03 | no | 2.82 / 1.56 / 0.25 | 0.7 / 0.2 / 0.0 | 0 | 0 / 664 | 0 | 1.000 | -8.9 | 57.1 |
| 26d4118b0c__b_level__trial04 | no | 2.82 / 1.51 / 0.27 | 0.7 / 0.2 / 0.0 | 0 | 0 / 664 | 0 | 1.000 | -8.3 | 56.8 |
| 26d4118b0c__b_level__trial05 | no | 2.82 / 1.45 / 1.01 | 0.7 / 0.1 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -5.6 | 82.2 |
| 26d4118b0c__b_level__trial06 | no | 2.82 / 1.56 / 0.31 | 0.7 / 0.2 / 0.0 | 0 | 0 / 664 | 0 | 1.000 | -8.3 | 80.7 |
| 26d4118b0c__b_level__trial07 | no | 2.82 / 1.50 / 0.24 | 0.7 / 0.2 / 0.0 | 0 | 0 / 664 | 0 | 1.000 | -9.2 | 43.7 |
| 26d4118b0c__b_level__trial08 | no | 2.82 / 1.76 / 0.29 | 0.7 / 0.2 / 0.0 | 0 | 0 / 664 | 0 | 1.000 | -10.2 | 70.0 |
| 26d4118b0c__b_level__trial09 | no | 2.82 / 1.57 / 0.28 | 0.7 / 0.3 / 0.0 | 0 | 0 / 664 | 0 | 1.000 | -9.9 | 51.2 |
| 26d4118b0c__b_tilt__recorded | no | 3.06 / 2.86 / 1.83 | 0.8 / 0.7 / 0.5 | 0 | 0 / 664 | 0 | 1.000 | -9.2 | 52.1 |
| 26d4118b0c__b_tilt__trial00 | no | 3.01 / 2.82 / 1.65 | 0.8 / 0.7 / 0.4 | 0 | 0 / 664 | 0 | 1.000 | -7.8 | 49.6 |
| 26d4118b0c__b_tilt__trial01 | no | 3.03 / 2.84 / 1.47 | 0.8 / 0.7 / 0.4 | 0 | 0 / 664 | 0 | 1.000 | -7.5 | 78.2 |
| 26d4118b0c__b_tilt__trial02 | no | 3.06 / 2.87 / 1.36 | 0.8 / 0.7 / 0.4 | 0 | 0 / 664 | 0 | 1.000 | -8.1 | 68.0 |
| 26d4118b0c__b_tilt__trial03 | no | 3.18 / 2.99 / 1.07 | 0.8 / 0.8 / 0.3 | 0 | 0 / 664 | 0 | 1.000 | -11.1 | 58.2 |
| 26d4118b0c__b_tilt__trial04 | no | 3.05 / 2.86 / 1.18 | 0.8 / 0.7 / 0.3 | 0 | 0 / 664 | 0 | 1.000 | -10.9 | 57.7 |
| 26d4118b0c__b_tilt__trial05 | no | 3.04 / 2.85 / 1.60 | 0.8 / 0.7 / 0.4 | 0 | 0 / 664 | 0 | 1.000 | -7.8 | 82.8 |
| 26d4118b0c__b_tilt__trial06 | no | 3.05 / 2.86 / 0.89 | 0.8 / 0.7 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -10.7 | 80.5 |
| 26d4118b0c__b_tilt__trial07 | no | 3.15 / 2.96 / 1.21 | 0.8 / 0.7 / 0.3 | 0 | 0 / 664 | 0 | 1.000 | -11.6 | 44.6 |
| 26d4118b0c__b_tilt__trial08 | no | 3.25 / 3.07 / 0.95 | 0.8 / 0.8 / 0.2 | 0 | 0 / 664 | 0 | 1.000 | -12.6 | 71.1 |
| 26d4118b0c__b_tilt__trial09 | no | 3.25 / 3.06 / 0.99 | 0.8 / 0.8 / 0.3 | 0 | 0 / 664 | 0 | 1.000 | -12.1 | 52.2 |
| 4e103843db__a_naive__recorded | no | 5.88 / 4.83 / 4.04 | 1.4 / 1.2 / 1.0 | 0 | 0 / 437 | 0 | - | 2.3 | 80.2 |
| 4e103843db__b_level__recorded | no | 2.86 / 1.31 / 1.70 | 0.7 / 0.2 / 0.2 | 0 | 0 / 589 | 0 | 1.000 | -7.6 | 80.9 |
| 4e103843db__b_level__trial00 | no | 2.86 / 1.01 / 1.54 | 0.7 / 0.3 / 0.1 | 0 | 0 / 589 | 0 | 1.000 | -6.1 | 47.6 |
| 4e103843db__b_level__trial01 | no | 2.86 / 1.15 / 1.23 | 0.7 / 0.3 / 0.2 | 0 | 0 / 589 | 0 | 1.000 | -5.9 | 75.7 |
| 4e103843db__b_level__trial02 | no | 2.86 / 0.99 / 1.16 | 0.7 / 0.3 / 0.2 | 0 | 0 / 589 | 0 | 1.000 | -6.4 | 66.0 |
| 4e103843db__b_level__trial03 | no | 2.86 / 1.41 / 0.42 | 0.7 / 0.3 / 0.0 | 0 | 0 / 589 | 0 | 1.000 | -9.9 | 56.0 |
| 4e103843db__b_level__trial04 | no | 2.86 / 1.25 / 0.48 | 0.7 / 0.3 / 0.1 | 0 | 0 / 589 | 0 | 1.000 | -9.2 | 55.8 |
| 4e103843db__b_level__trial05 | no | 2.86 / 1.18 / 1.45 | 0.7 / 0.3 / 0.1 | 0 | 0 / 589 | 0 | 1.000 | -6.2 | 81.5 |
| 4e103843db__b_level__trial06 | no | 2.86 / 1.24 / 0.78 | 0.7 / 0.3 / 0.0 | 0 | 0 / 589 | 0 | 1.000 | -9.1 | 79.9 |
| 4e103843db__b_level__trial07 | no | 2.86 / 1.45 / 0.39 | 0.7 / 0.3 / 0.1 | 0 | 0 / 589 | 0 | 1.000 | -10.3 | 42.5 |
| 4e103843db__b_level__trial08 | no | 2.86 / 1.83 / 0.47 | 0.7 / 0.3 / 0.1 | 0 | 0 / 589 | 0 | 1.000 | -11.7 | 68.3 |
| 4e103843db__b_level__trial09 | no | 2.86 / 1.79 / 0.63 | 0.7 / 0.2 / 0.1 | 0 | 0 / 589 | 0 | 1.000 | -11.0 | 49.9 |
| 4e103843db__b_tilt__recorded | no | 6.20 / 5.83 / 4.32 | 1.4 / 1.3 / 1.1 | 0 | 0 / 589 | 12 | 1.000 | -5.8 | 84.4 |
| 4e103843db__b_tilt__trial00 | no | 6.06 / 5.70 / 4.20 | 1.4 / 1.3 / 1.0 | 0 | 0 / 589 | 12 | 1.000 | -4.2 | 54.0 |
| 4e103843db__b_tilt__trial01 | no | 6.15 / 5.63 / 4.09 | 1.7 / 1.3 / 1.0 | 0 | 0 / 589 | 12 | 1.000 | -4.1 | 81.1 |
| 4e103843db__b_tilt__trial02 | no | 6.17 / 5.80 / 4.20 | 1.4 / 1.3 / 1.0 | 0 | 0 / 589 | 12 | 1.000 | -4.4 | 71.9 |
| 4e103843db__b_tilt__trial03 | no | 6.25 / 5.88 / 4.17 | 1.4 / 1.4 / 0.9 | 0 | 0 / 589 | 12 | 1.000 | -8.7 | 61.6 |
| 4e103843db__b_tilt__trial04 | no | 6.12 / 5.75 / 4.16 | 1.4 / 1.3 / 0.9 | 0 | 0 / 589 | 12 | 1.000 | -8.1 | 61.2 |
| 4e103843db__b_tilt__trial05 | no | 6.25 / 5.87 / 4.14 | 1.7 / 1.7 / 1.0 | 0 | 0 / 589 | 12 | 1.000 | -5.2 | 84.6 |
| 4e103843db__b_tilt__trial06 | no | 6.27 / 5.90 / 4.15 | 1.4 / 1.4 / 0.9 | 0 | 0 / 589 | 12 | 1.000 | -7.7 | 83.0 |
| 4e103843db__b_tilt__trial07 | no | 6.22 / 5.85 / 4.17 | 1.5 / 1.4 / 0.9 | 0 | 0 / 589 | 12 | 1.000 | -9.9 | 47.9 |
| 4e103843db__b_tilt__trial08 | no | 6.32 / 5.97 / 3.80 | 1.5 / 1.4 / 0.9 | 0 | 0 / 589 | 12 | 1.000 | -12.1 | 72.4 |
| 4e103843db__b_tilt__trial09 | no | 6.25 / 5.88 / 4.17 | 1.5 / 1.4 / 0.9 | 0 | 0 / 589 | 12 | 1.000 | -10.7 | 55.0 |
| 49aff04f7a__a_naive__recorded | no | 6.07 / 2.18 / 2.09 | 1.4 / 0.5 / 0.5 | 0 | 0 / 493 | 0 | - | 6.2 | 89.1 |
| 49aff04f7a__b_level__recorded | no | 2.82 / 0.97 / 1.14 | 0.7 / 0.1 / 0.1 | 0 | 0 / 603 | 0 | 1.000 | -6.4 | 82.5 |
| 49aff04f7a__b_level__trial00 | no | 2.82 / 0.87 / 1.13 | 0.7 / 0.1 / 0.1 | 0 | 0 / 603 | 0 | 1.000 | -4.9 | 50.0 |
| 49aff04f7a__b_level__trial01 | no | 2.82 / 0.87 / 0.78 | 0.7 / 0.2 / 0.1 | 0 | 0 / 603 | 0 | 1.000 | -4.8 | 77.6 |
| 49aff04f7a__b_level__trial02 | no | 2.82 / 0.88 / 0.63 | 0.7 / 0.1 / 0.1 | 0 | 0 / 603 | 0 | 1.000 | -5.2 | 68.0 |
| 49aff04f7a__b_level__trial03 | no | 2.82 / 1.13 / 0.21 | 0.7 / 0.2 / 0.0 | 0 | 0 / 603 | 0 | 1.000 | -8.4 | 58.5 |
| 49aff04f7a__b_level__trial04 | no | 2.82 / 1.06 / 0.23 | 0.7 / 0.2 / 0.0 | 0 | 0 / 603 | 0 | 1.000 | -7.8 | 58.2 |
| 49aff04f7a__b_level__trial05 | no | 2.82 / 0.88 / 0.97 | 0.7 / 0.2 / 0.1 | 0 | 0 / 603 | 0 | 1.000 | -5.0 | 83.3 |
| 49aff04f7a__b_level__trial06 | no | 2.82 / 1.06 / 0.27 | 0.7 / 0.2 / 0.0 | 0 | 0 / 603 | 0 | 1.000 | -7.7 | 81.3 |
| 49aff04f7a__b_level__trial07 | no | 2.82 / 1.17 / 0.19 | 0.7 / 0.2 / 0.0 | 0 | 0 / 603 | 0 | 1.000 | -8.7 | 45.3 |
| 49aff04f7a__b_level__trial08 | no | 2.82 / 1.44 / 0.25 | 0.7 / 0.1 / 0.0 | 0 | 0 / 603 | 0 | 1.000 | -9.7 | 71.3 |
| 49aff04f7a__b_level__trial09 | no | 2.82 / 1.27 / 0.24 | 0.7 / 0.2 / 0.0 | 0 | 0 / 603 | 0 | 1.000 | -9.3 | 52.8 |
| 49aff04f7a__b_tilt__recorded | no | 2.89 / 2.62 / 1.45 | 0.8 / 0.7 / 0.4 | 0 | 0 / 603 | 0 | 1.000 | -7.4 | 83.6 |
| 49aff04f7a__b_tilt__trial00 | no | 2.87 / 2.60 / 1.64 | 0.8 / 0.7 / 0.4 | 0 | 0 / 603 | 0 | 1.000 | -5.9 | 51.0 |
| 49aff04f7a__b_tilt__trial01 | no | 2.94 / 2.67 / 1.36 | 0.8 / 0.7 / 0.4 | 0 | 0 / 603 | 0 | 1.000 | -5.6 | 78.8 |
| 49aff04f7a__b_tilt__trial02 | no | 2.78 / 2.51 / 1.22 | 0.8 / 0.6 / 0.4 | 0 | 0 / 603 | 0 | 1.000 | -6.5 | 68.8 |
| 49aff04f7a__b_tilt__trial03 | no | 3.00 / 2.75 / 0.86 | 0.8 / 0.7 / 0.2 | 0 | 0 / 603 | 0 | 1.000 | -9.6 | 59.3 |
| 49aff04f7a__b_tilt__trial04 | no | 2.96 / 2.70 / 0.79 | 0.8 / 0.7 / 0.2 | 0 | 0 / 603 | 0 | 1.000 | -8.8 | 59.3 |
| 49aff04f7a__b_tilt__trial05 | no | 2.92 / 2.65 / 1.45 | 0.8 / 0.8 / 0.4 | 0 | 0 / 603 | 0 | 1.000 | -6.2 | 83.9 |
| 49aff04f7a__b_tilt__trial06 | no | 3.02 / 2.76 / 0.69 | 0.8 / 0.7 / 0.2 | 0 | 0 / 603 | 0 | 1.000 | -8.8 | 82.1 |
| 49aff04f7a__b_tilt__trial07 | no | 3.09 / 2.83 / 0.97 | 0.8 / 0.7 / 0.2 | 0 | 0 / 603 | 0 | 1.000 | -9.8 | 46.0 |
| 49aff04f7a__b_tilt__trial08 | no | 3.06 / 2.83 / 0.88 | 0.8 / 0.7 / 0.2 | 0 | 0 / 603 | 0 | 1.000 | -11.0 | 71.8 |
| 49aff04f7a__b_tilt__trial09 | no | 3.08 / 2.83 / 0.73 | 0.8 / 0.7 / 0.2 | 0 | 0 / 603 | 0 | 1.000 | -10.6 | 53.4 |
| e8825dea53__a_naive__recorded | no | 9.86 / 1.46 / 1.99 | 3.1 / 0.4 / 0.5 | 0 | 0 / 510 | 0 | - | -2.7 | 75.4 |
| e8825dea53__b_level__recorded | yes | 1.90 / 0.81 / 0.77 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -4.2 | 83.9 |
| e8825dea53__b_level__trial00 | yes | 1.90 / 0.76 / 0.64 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -3.5 | 51.3 |
| e8825dea53__b_level__trial01 | yes | 1.90 / 0.79 / 0.53 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -3.3 | 79.1 |
| e8825dea53__b_level__trial02 | yes | 1.90 / 0.75 / 0.35 | 0.6 / 0.2 / 0.0 | 0 | 0 / 640 | 0 | 0.888 | -3.7 | 69.5 |
| e8825dea53__b_level__trial03 | yes | 1.90 / 0.89 / 0.15 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -5.3 | 61.6 |
| e8825dea53__b_level__trial04 | yes | 1.90 / 0.88 / 0.16 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -4.9 | 61.0 |
| e8825dea53__b_level__trial05 | yes | 1.90 / 0.77 / 0.57 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -3.6 | 84.8 |
| e8825dea53__b_level__trial06 | yes | 1.90 / 0.86 / 0.19 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -5.0 | 84.0 |
| e8825dea53__b_level__trial07 | yes | 1.90 / 0.95 / 0.14 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -5.3 | 49.0 |
| e8825dea53__b_level__trial08 | no | 1.90 / 1.02 / 0.19 | 0.6 / 0.1 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -5.6 | 76.1 |
| e8825dea53__b_level__trial09 | yes | 1.90 / 0.96 / 0.17 | 0.6 / 0.2 / 0.1 | 0 | 0 / 640 | 0 | 0.888 | -5.7 | 56.6 |
| e8825dea53__b_tilt__recorded | no | 3.77 / 3.38 / 1.96 | 0.9 / 0.8 / 0.6 | 0 | 0 / 640 | 0 | 0.888 | -1.6 | 87.1 |
| e8825dea53__b_tilt__trial00 | no | 3.76 / 3.36 / 1.97 | 0.9 / 0.8 / 0.5 | 0 | 0 / 640 | 0 | 0.888 | 0.3 | 58.7 |
| e8825dea53__b_tilt__trial01 | no | 3.74 / 3.35 / 1.76 | 0.8 / 0.8 / 0.5 | 0 | 0 / 640 | 0 | 0.888 | 0.3 | 84.7 |
| e8825dea53__b_tilt__trial02 | no | 3.78 / 3.39 / 1.76 | 0.9 / 0.8 / 0.5 | 0 | 0 / 640 | 0 | 0.888 | -0.1 | 75.7 |
| e8825dea53__b_tilt__trial03 | no | 3.74 / 3.34 / 1.72 | 0.9 / 0.8 / 0.4 | 0 | 0 / 640 | 0 | 0.888 | -3.4 | 66.7 |
| e8825dea53__b_tilt__trial04 | no | 3.82 / 3.43 / 1.76 | 0.9 / 0.8 / 0.4 | 0 | 0 / 640 | 0 | 0.888 | -2.5 | 66.7 |
| e8825dea53__b_tilt__trial05 | no | 3.78 / 3.39 / 1.98 | 0.9 / 0.8 / 0.5 | 0 | 0 / 640 | 0 | 0.888 | 1.4 | 90.6 |
| e8825dea53__b_tilt__trial06 | no | 3.73 / 3.34 / 1.74 | 0.9 / 0.8 / 0.4 | 0 | 0 / 640 | 0 | 0.888 | -3.0 | 87.4 |
| e8825dea53__b_tilt__trial07 | no | 3.83 / 3.44 / 1.78 | 0.9 / 0.8 / 0.4 | 0 | 0 / 640 | 0 | 0.888 | -3.2 | 54.9 |
| e8825dea53__b_tilt__trial08 | no | 3.70 / 3.31 / 1.75 | 0.9 / 0.8 / 0.4 | 0 | 0 / 640 | 0 | 0.888 | -4.4 | 79.8 |
| e8825dea53__b_tilt__trial09 | no | 3.80 / 3.41 / 1.73 | 0.9 / 0.8 / 0.4 | 0 | 0 / 640 | 0 | 0.888 | -3.8 | 61.9 |

## Videos (clip 4e103843db, recorded layout)

- 4e103843db__a_naive__recorded: results/m3_retarget/videos/4e103843db__a_naive__recorded.mp4 (gitignored); mug carried from frame 232 (first frame with closure = 1 (no 'close' phase in this path)) to 406; re-planned run differs from the saved one in: nothing
- 4e103843db__b_level__recorded: results/m3_retarget/videos/4e103843db__b_level__recorded.mp4 (gitignored); mug carried from frame 281 (last frame of the 'close' phase (end of the close hold)) to 403; re-planned run differs from the saved one in: nothing; GIF results/m3_retarget/4e103843db__b_level__recorded.gif 2.82 MB (< 5 MB: True)
- 4e103843db__b_tilt__recorded: results/m3_retarget/videos/4e103843db__b_tilt__recorded.mp4 (gitignored); mug carried from frame 281 (last frame of the 'close' phase (end of the close hold)) to 403; re-planned run differs from the saved one in: nothing

## Runtime and peak RSS (CPU only)

| step | wall (s) | peak RSS (MB) |
|---|---|---|
| verify | 4.8 | 1659 |
| run (4 processes) | 84.8 | max process 1668, sum 6492, parent 148 |
| R0 after_m3 (4 processes) | 33.9 | max process 1674, sum 6495 |
| videos | 31.3 | 2185 |
| summarize | 0.02 | 151 |
