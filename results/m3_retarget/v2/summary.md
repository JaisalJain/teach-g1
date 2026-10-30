# M3 v2: place-palm sign, the oracle under M3's rule, v2 path timing

Made by `python -m retarget.run_m3_v2 summarize` from results/m3_retarget/runs/*.json (M3, read only), v2/runs/oracle__*.json, v2/paths.json and v2/oracle.json. Kinematic only (no physics beyond PickPlace.run's 2 s setup hold).

## Checks

- M3's 92 paths rebuilt from the saved standing palms by M3's own builders: 92/92 byte-identical to the saved npz.
- sigma (closed form) vs a numeric integral of v: max diff 1.5e-11 s (pass True); sigma(D) - D0 = 0.0e+00 s.
- Per re-timed segment (asserted while building, so every saved path passed): first and last 50 Hz step < 1 mm, end positions = M3's to 1e-9.
- Dry run of v2_path on M4's layouts (trials 0-49 x 4 clips x b_level / b_tilt / naive): built 600/600, failed 0; shortest segment D0 0.48 s (T_R = 0.4 s).
- Oracle nominal legs = warp.time_scale's at each recorded layout (distances, transport time, z_clear to 1e-12): True.
- Oracle rows: path sample 0 vs palm_start max 0.090 mm (limit 1 mm); clip -> plant mapping at G 10/10, over the collision window 10/10 (< 1 mm).
- collision_min with its own setup + plan = the diagnostic on run_one's clip: True. Standing palm of the oracle rows = M3's b rows on the same trials (other handle yaw): True.
- Code unchanged since the steps ran (SHA-256): paths True, oracle True. M3's 8 code files = the hashes in results/m3_retarget/run_resources.json: True.

## Step 0.1 — place-palm flag with the correct sign (recorded layouts)

Nominal oracle palm for a cup centre at xy c = c + legacy palm_offset[:2] (what sample_layouts uses). M3 flagged coaster - offset.

| clip | cup xy | coaster xy | grasp palm xy (in box) | place palm xy = coaster + offset | in PALM_BOX | outside by x / y (cm) | M3's flag (coaster - offset) |
|---|---|---|---|---|---|---|---|
| 26d4118b0c | (0.493, -0.236) | (0.498, -0.380) | (0.400, -0.220) (yes) | (0.404, -0.364) | no | 0.0 / 4.4 | (0.591, -0.395) out |
| 4e103843db | (0.493, -0.236) | (0.500, -0.380) | (0.400, -0.220) (yes) | (0.406, -0.364) | no | 0.0 / 4.4 | (0.593, -0.395) out |
| 49aff04f7a | (0.493, -0.236) | (0.491, -0.372) | (0.400, -0.220) (yes) | (0.398, -0.357) | no | 0.0 / 3.7 | (0.584, -0.388) out |
| e8825dea53 | (0.493, -0.236) | (0.468, -0.403) | (0.400, -0.220) (yes) | (0.374, -0.388) | no | 0.0 / 6.8 | (0.561, -0.419) out |

PALM_BOX: x (0.35, 0.45), y (-0.32, -0.12) m.

## Step 0.2 — reachability under M3's rule, with the oracle's own nominal path

Reachable = FK error at G and at P < 1 cm and < 10 deg AND path max position error < 2 cm (M3's rule, unchanged). Oracle: paths_v2.oracle_path through run_m3.run_one on trials 0-9 with each trial's own handle yaw.

| clip | variant | layouts | at G | at P | path max < 2 cm | reachable |
|---|---|---|---|---|---|---|
| 26d4118b0c | a_naive | recorded | 0/1 | 0/1 | 0/1 | 0/1 |
| 26d4118b0c | b_level | recorded + trials 0-9 | 0/11 | 8/11 | 0/11 | 0/11 |
| 26d4118b0c | b_tilt | recorded + trials 0-9 | 0/11 | 3/11 | 0/11 | 0/11 |
| 4e103843db | a_naive | recorded | 0/1 | 0/1 | 0/1 | 0/1 |
| 4e103843db | b_level | recorded + trials 0-9 | 1/11 | 6/11 | 0/11 | 0/11 |
| 4e103843db | b_tilt | recorded + trials 0-9 | 0/11 | 0/11 | 0/11 | 0/11 |
| 49aff04f7a | a_naive | recorded | 0/1 | 0/1 | 0/1 | 0/1 |
| 49aff04f7a | b_level | recorded + trials 0-9 | 5/11 | 9/11 | 0/11 | 0/11 |
| 49aff04f7a | b_tilt | recorded + trials 0-9 | 0/11 | 6/11 | 0/11 | 0/11 |
| e8825dea53 | a_naive | recorded | 0/1 | 0/1 | 0/1 | 0/1 |
| e8825dea53 | b_level | recorded + trials 0-9 | 10/11 | 11/11 | 11/11 | 10/11 |
| e8825dea53 | b_tilt | recorded + trials 0-9 | 0/11 | 0/11 | 0/11 | 0/11 |
| oracle | oracle | trials 0-9 | 10/10 | 10/10 | 10/10 | 10/10 |

Oracle rows:

| run | handle yaw (deg) | reachable | pos err max / G / P (cm) | where the max is | max after prelude (cm, phase) | ori max (deg) | hand-mug at G (mm) | collision min (mm) |
|---|---|---|---|---|---|---|---|---|
| oracle__oracle__trial00 | -38.4 | yes | 1.69 / 0.70 / 0.50 | prelude | 0.77 (approach) | 0.4 | -2.5 | 6.1 |
| oracle__oracle__trial01 | 3.1 | yes | 1.69 / 0.74 / 0.39 | prelude | 0.81 (approach) | 0.4 | -2.4 | 6.1 |
| oracle__oracle__trial02 | -12.4 | yes | 1.69 / 0.68 / 0.26 | prelude | 0.74 (approach) | 0.4 | -2.6 | 6.1 |
| oracle__oracle__trial03 | -21.1 | yes | 1.69 / 0.79 / 0.12 | prelude | 0.81 (approach) | 0.4 | -3.5 | 5.4 |
| oracle__oracle__trial04 | -22.5 | yes | 1.69 / 0.75 / 0.13 | prelude | 0.77 (approach) | 0.4 | -3.3 | 5.5 |
| oracle__oracle__trial05 | 16.7 | yes | 1.69 / 0.75 / 0.46 | prelude | 0.81 (approach) | 0.4 | -2.5 | 6.0 |
| oracle__oracle__trial06 | 27.0 | yes | 1.69 / 0.78 / 0.14 | prelude | 0.81 (approach) | 0.4 | -3.3 | 5.5 |
| oracle__oracle__trial07 | -38.5 | yes | 1.69 / 0.77 / 0.12 | prelude | 0.78 (approach) | 0.4 | -3.3 | 5.5 |
| oracle__oracle__trial08 | 4.3 | yes | 1.69 / 0.84 / 0.20 | prelude | 0.85 (approach) | 0.4 | -3.4 | 5.6 |
| oracle__oracle__trial09 | -27.6 | yes | 1.69 / 0.80 / 0.17 | prelude | 0.82 (approach) | 0.4 | -3.7 | 5.2 |

### Largest FK errors outside the prelude (information only; from each run's max_by_phase)

Per run: the maximum over every phase except 'prelude'. Columns give min / median / max over the group's runs; 'below 2 cm' counts runs whose after-prelude maximum is < 2 cm (a count, not a rule).

| clip | variant | runs | pos after prelude (cm) | phase of the max (runs) | ori after prelude (deg) | prelude pos max (cm) | below 2 cm after prelude |
|---|---|---|---|---|---|---|---|
| 26d4118b0c | a_naive | 1 | 3.60 / 3.60 / 3.60 | replay 1 | 7.9 / 7.9 / 7.9 | 2.67 / 2.67 / 2.67 | 0/1 |
| 26d4118b0c | b_level | 11 | 1.70 / 1.79 / 2.24 | approach 11 | 0.3 / 0.3 / 0.3 | 2.82 / 2.82 / 2.83 | 10/11 |
| 26d4118b0c | b_tilt | 11 | 3.01 / 3.06 / 3.25 | approach 11 | 0.8 / 0.8 / 0.8 | 2.67 / 2.67 / 2.67 | 0/11 |
| 4e103843db | a_naive | 1 | 5.88 / 5.88 / 5.88 | replay 1 | 1.4 / 1.4 / 1.4 | 3.26 / 3.26 / 3.26 | 0/1 |
| 4e103843db | b_level | 11 | 1.88 / 2.02 / 2.32 | approach 11 | 0.3 / 0.3 / 0.4 | 2.86 / 2.86 / 2.86 | 5/11 |
| 4e103843db | b_tilt | 11 | 6.06 / 6.22 / 6.32 | approach 11 | 1.4 / 1.4 / 1.7 | 3.26 / 3.26 / 3.26 | 0/11 |
| 49aff04f7a | a_naive | 1 | 6.07 / 6.07 / 6.07 | replay 1 | 1.4 / 1.4 / 1.4 | 2.67 / 2.67 / 2.67 | 0/1 |
| 49aff04f7a | b_level | 11 | 1.12 / 1.27 / 1.58 | transport 2, approach 9 | 0.2 / 0.3 / 0.3 | 2.82 / 2.82 / 2.82 | 11/11 |
| 49aff04f7a | b_tilt | 11 | 2.78 / 2.96 / 3.09 | approach 11 | 0.7 / 0.8 / 0.8 | 2.67 / 2.67 / 2.67 | 0/11 |
| e8825dea53 | a_naive | 1 | 9.86 / 9.86 / 9.86 | replay 1 | 3.1 / 3.1 / 3.1 | 2.99 / 2.99 / 2.99 | 0/1 |
| e8825dea53 | b_level | 11 | 1.12 / 1.20 / 1.33 | approach 11 | 0.2 / 0.3 / 0.3 | 1.90 / 1.90 / 1.90 | 11/11 |
| e8825dea53 | b_tilt | 11 | 3.70 / 3.77 / 3.83 | approach 11 | 0.8 / 0.9 / 0.9 | 2.99 / 2.99 / 2.99 | 0/11 |
| oracle | oracle | 10 | 0.74 / 0.81 / 0.85 | approach 10 | 0.2 / 0.2 / 0.3 | 1.69 / 1.69 / 1.69 | 10/10 |

| variant (all clips) | runs | reachable | pos after prelude (cm) | ori after prelude (deg) | prelude pos max (cm) | below 2 cm after prelude |
|---|---|---|---|---|---|---|
| a_naive | 4 | 0 | 3.60 / 5.97 / 9.86 | 1.4 / 2.3 / 7.9 | 2.67 / 2.83 / 3.26 | 0/4 |
| b_level | 44 | 10 | 1.12 / 1.64 / 2.32 | 0.2 / 0.3 / 0.4 | 1.90 / 2.82 / 2.86 | 37/44 |
| b_tilt | 44 | 0 | 2.78 / 3.48 / 6.32 | 0.7 / 0.8 / 1.7 | 2.67 / 2.83 / 3.26 | 0/44 |
| oracle | 10 | 10 | 0.74 / 0.81 / 0.85 | 0.2 / 0.2 / 0.3 | 1.69 / 1.69 / 1.69 | 10/10 |

## Step 3 — collision diagnostic on the oracle rows (kinematic, information only)

Method: M3 harness single-plan IK clip (standing_palm setup, one plan_path(base='plan', base_goal=_goal(G, R_G), return_fk=True)); frames from the prelude's last frame to the last frame with remaining path arc length to G > 1 cm; per frame: scene.reset (open hands, mug at its layout pose), set_robot_to_clip_frame, mj_forward, min mj_geomDistance from the 8 right-hand geoms to cup_geom and the cup_handle_* capsules; information only, never a gate.

Min signed distance over the window (mm; negative = overlap), 10 rows: min / median / max 5.2 / 5.6 / 6.1; to the body 5.2 / 5.6 / 6.1; to the handle 58.8 / 74.2 / 94.5. Per row in the oracle table above.

## Step 1 — v2 path timing (M3's 92 runs; v2/paths/<run>.npz)

Per segment over the runs (b: 11 layouts per clip and variant; naive: recorded only). D0 = M3's duration (D = D0 + 0.4 s). M3 start / end speed = the speed of M3's first / last step of the segment: the jump v2 removes.

| clip | variant | segment | runs | D0 (s) min / median / max | first / last step max (mm) | M3 start speed (m/s) | M3 end speed (m/s) |
|---|---|---|---|---|---|---|---|
| 26d4118b0c | a_naive | replay | 1 | 8.00 / 8.00 / 8.00 | 0.047 / 0.007 | 0.973 / 0.973 / 0.973 | 0.153 / 0.153 / 0.153 |
| 26d4118b0c | b_level | approach1 | 11 | 1.34 / 1.54 / 1.58 | 0.047 / 0.010 | 0.971 / 0.971 / 0.971 | 0.143 / 0.186 / 0.199 |
| 26d4118b0c | b_level | approach2 | 11 | 0.86 / 0.90 / 1.10 | 0.010 / 0.002 | 0.146 / 0.178 / 0.199 | 0.039 / 0.039 / 0.039 |
| 26d4118b0c | b_level | transport1 | 11 | 0.84 / 0.84 / 0.84 | 0.006 / 0.003 | 0.108 / 0.115 / 0.121 | 0.034 / 0.043 / 0.062 |
| 26d4118b0c | b_level | transport2 | 11 | 2.82 / 2.82 / 2.82 | 0.003 / 0.004 | 0.021 / 0.030 / 0.069 | 0.019 / 0.081 / 0.092 |
| 26d4118b0c | b_tilt | approach1 | 11 | 1.34 / 1.54 / 1.58 | 0.047 / 0.010 | 0.971 / 0.971 / 0.971 | 0.143 / 0.186 / 0.200 |
| 26d4118b0c | b_tilt | approach2 | 11 | 0.86 / 0.90 / 1.10 | 0.010 / 0.002 | 0.145 / 0.178 / 0.199 | 0.039 / 0.039 / 0.039 |
| 26d4118b0c | b_tilt | transport1 | 11 | 0.84 / 0.84 / 0.84 | 0.006 / 0.003 | 0.108 / 0.115 / 0.121 | 0.034 / 0.043 / 0.062 |
| 26d4118b0c | b_tilt | transport2 | 11 | 2.82 / 2.82 / 2.82 | 0.003 / 0.004 | 0.021 / 0.030 / 0.069 | 0.019 / 0.081 / 0.092 |
| 4e103843db | a_naive | replay | 1 | 5.92 / 5.92 / 5.92 | 0.029 / 0.010 | 0.593 / 0.593 / 0.593 | 0.210 / 0.210 / 0.210 |
| 4e103843db | b_level | approach1 | 11 | 1.34 / 1.44 / 1.48 | 0.029 / 0.013 | 0.594 / 0.594 / 0.594 | 0.214 / 0.238 / 0.263 |
| 4e103843db | b_level | approach2 | 11 | 0.68 / 0.72 / 0.82 | 0.013 / 0.004 | 0.208 / 0.240 / 0.257 | 0.073 / 0.073 / 0.073 |
| 4e103843db | b_level | transport1 | 11 | 0.48 / 0.48 / 0.48 | 0.013 / 0.004 | 0.204 / 0.253 / 0.264 | 0.042 / 0.061 / 0.087 |
| 4e103843db | b_level | transport2 | 11 | 1.96 / 1.96 / 1.96 | 0.004 / 0.006 | 0.043 / 0.061 / 0.088 | 0.037 / 0.113 / 0.130 |
| 4e103843db | b_tilt | approach1 | 11 | 1.34 / 1.44 / 1.46 | 0.029 / 0.013 | 0.594 / 0.594 / 0.594 | 0.224 / 0.246 / 0.275 |
| 4e103843db | b_tilt | approach2 | 11 | 0.70 / 0.72 / 0.82 | 0.013 / 0.004 | 0.217 / 0.248 / 0.270 | 0.073 / 0.073 / 0.073 |
| 4e103843db | b_tilt | transport1 | 11 | 0.48 / 0.48 / 0.48 | 0.013 / 0.004 | 0.204 / 0.253 / 0.264 | 0.042 / 0.061 / 0.087 |
| 4e103843db | b_tilt | transport2 | 11 | 1.96 / 1.96 / 1.96 | 0.004 / 0.006 | 0.043 / 0.061 / 0.088 | 0.037 / 0.113 / 0.130 |
| 49aff04f7a | a_naive | replay | 1 | 7.04 / 7.04 / 7.04 | 0.033 / 0.022 | 0.682 / 0.682 / 0.682 | 0.456 / 0.456 / 0.456 |
| 49aff04f7a | b_level | approach1 | 11 | 1.16 / 1.26 / 1.30 | 0.033 / 0.013 | 0.679 / 0.679 / 0.679 | 0.218 / 0.248 / 0.267 |
| 49aff04f7a | b_level | approach2 | 11 | 0.62 / 0.66 / 0.76 | 0.013 / 0.003 | 0.213 / 0.239 / 0.262 | 0.054 / 0.054 / 0.054 |
| 49aff04f7a | b_level | transport1 | 11 | 1.32 / 1.32 / 1.32 | 0.008 / 0.008 | 0.120 / 0.149 / 0.157 | 0.042 / 0.058 / 0.161 |
| 49aff04f7a | b_level | transport2 | 11 | 1.64 / 1.64 / 1.64 | 0.009 / 0.005 | 0.057 / 0.073 / 0.176 | 0.078 / 0.088 / 0.111 |
| 49aff04f7a | b_tilt | approach1 | 11 | 1.16 / 1.26 / 1.30 | 0.033 / 0.013 | 0.679 / 0.679 / 0.679 | 0.220 / 0.251 / 0.271 |
| 49aff04f7a | b_tilt | approach2 | 11 | 0.62 / 0.66 / 0.76 | 0.013 / 0.003 | 0.216 / 0.242 / 0.266 | 0.054 / 0.054 / 0.054 |
| 49aff04f7a | b_tilt | transport1 | 11 | 1.32 / 1.32 / 1.32 | 0.008 / 0.008 | 0.120 / 0.149 / 0.157 | 0.042 / 0.058 / 0.161 |
| 49aff04f7a | b_tilt | transport2 | 11 | 1.64 / 1.64 / 1.64 | 0.009 / 0.005 | 0.057 / 0.073 / 0.176 | 0.078 / 0.088 / 0.111 |
| e8825dea53 | a_naive | replay | 1 | 7.38 / 7.38 / 7.38 | 0.030 / 0.010 | 0.614 / 0.614 / 0.614 | 0.204 / 0.204 / 0.204 |
| e8825dea53 | b_level | approach1 | 11 | 1.38 / 1.54 / 1.60 | 0.030 / 0.009 | 0.615 / 0.615 / 0.615 | 0.128 / 0.154 / 0.179 |
| e8825dea53 | b_level | approach2 | 11 | 0.88 / 0.94 / 1.10 | 0.009 / 0.002 | 0.129 / 0.153 / 0.184 | 0.050 / 0.050 / 0.050 |
| e8825dea53 | b_level | transport1 | 11 | 0.78 / 0.78 / 0.78 | 0.009 / 0.003 | 0.122 / 0.167 / 0.176 | 0.034 / 0.047 / 0.072 |
| e8825dea53 | b_level | transport2 | 11 | 2.36 / 2.36 / 2.36 | 0.004 / 0.006 | 0.035 / 0.049 / 0.073 | 0.049 / 0.110 / 0.121 |
| e8825dea53 | b_tilt | approach1 | 11 | 1.40 / 1.56 / 1.62 | 0.030 / 0.009 | 0.615 / 0.615 / 0.615 | 0.133 / 0.158 / 0.189 |
| e8825dea53 | b_tilt | approach2 | 11 | 0.86 / 0.92 / 1.08 | 0.009 / 0.002 | 0.134 / 0.156 / 0.184 | 0.050 / 0.050 / 0.050 |
| e8825dea53 | b_tilt | transport1 | 11 | 0.78 / 0.78 / 0.78 | 0.009 / 0.003 | 0.122 / 0.167 / 0.176 | 0.034 / 0.047 / 0.072 |
| e8825dea53 | b_tilt | transport2 | 11 | 2.36 / 2.36 / 2.36 | 0.004 / 0.006 | 0.035 / 0.049 / 0.073 | 0.049 / 0.110 / 0.121 |

Recorded layout, frame indices of the v2 paths (-1 = none):

| run | M3 frames | v2 frames | i_prelude_end | i_pause | i_G | i_close_end | i_peak | i_P | i_release |
|---|---|---|---|---|---|---|---|---|---|
| 26d4118b0c__a_naive__recorded | 541 | 561 | 60 | -1 | 242 | -1 | -1 | 433 | 530 |
| 26d4118b0c__b_level__recorded | 664 | 719 | 60 | 153 | 222 | 310 | 372 | 533 | 533 |
| 26d4118b0c__b_tilt__recorded | 664 | 719 | 60 | 153 | 222 | 310 | 372 | 533 | 533 |
| 4e103843db__a_naive__recorded | 437 | 457 | 60 | -1 | 228 | -1 | -1 | 353 | 426 |
| 4e103843db__b_level__recorded | 589 | 644 | 60 | 150 | 208 | 296 | 340 | 458 | 458 |
| 4e103843db__b_tilt__recorded | 589 | 644 | 60 | 150 | 208 | 296 | 340 | 458 | 458 |
| 49aff04f7a__a_naive__recorded | 493 | 513 | 60 | -1 | 216 | -1 | -1 | 368 | 482 |
| 49aff04f7a__b_level__recorded | 603 | 658 | 60 | 141 | 196 | 284 | 370 | 472 | 472 |
| 49aff04f7a__b_tilt__recorded | 603 | 658 | 60 | 142 | 196 | 284 | 370 | 472 | 472 |
| e8825dea53__a_naive__recorded | 510 | 530 | 60 | -1 | 244 | -1 | -1 | 405 | 499 |
| e8825dea53__b_level__recorded | 640 | 695 | 60 | 154 | 224 | 312 | 371 | 509 | 509 |
| e8825dea53__b_tilt__recorded | 640 | 695 | 60 | 155 | 224 | 312 | 371 | 509 | 509 |

## Runtime and peak RSS (CPU only)

| step | wall (s) | peak RSS (MB) |
|---|---|---|
| paths | 8.4 | 170 |
| oracle (1 process) | 39.7 | 1815 |
| summarize | 0.02 | 153 |
