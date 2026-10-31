# naive, b_level and b_tilt at each layout's own handle yaw

Made by `python -m retarget.own_yaw_summary` from the saved rows (nothing is simulated here). Rows: `results/final/own_yaw/runs/` (`python -m sim.m4_execute run --own-yaw`). M4 ran these variants with the handle turned with my grasp heading (trial yaw + (phi_h - phi_ora)); here the handle has the trial's own yaw, as for the oracle, b_hybrid_h45 and every M5 run. The paths and everything else are M4's. Success / 50 per clip, / 200 pooled, Wilson 95 %; the pooled interval treats the 200 runs as independent although the 4 clips reuse the same 50 layouts, so the real uncertainty is larger.

- E1 (stop if more than 1 % of the runs raised exceptions): 0/600 exceptions, branch **continue**.
- Checks: 600 rows; own_yaw true; handle_yaw_deg = trial_handle_yaw_deg = the oracle's cup_yaw_deg; cup and coaster xy = the oracle's and the M4 row's; paths v2, profile mug_v1, mass 0.4 kg (asserted).
- M4's handle turn per clip (M4 yaw - own yaw, deg): 26d4118b0c [36.9]; 4e103843db [40.8, 40.9]; 49aff04f7a [36.7]; e8825dea53 [16.7].
- Oracle (Track B final, fifty.json): 44/50 (0.76-0.94).

## Success

| variant | 26d4118b0c (top-down) | 4e103843db (90 deg) | 49aff04f7a (45 deg) | e8825dea53 (45 deg) | 4-clip pool |
|---|---|---|---|---|---|
| b_level | 0/50 | 0/50 | 0/50 | 25/50 | 25/200 (0.09-0.18) |
| b_tilt | 5/50 | 0/50 | 17/50 | 17/50 | 39/200 (0.15-0.26) |
| naive | 0/50 | 0/50 | 0/50 | 0/50 | 0/200 (0.00-0.02) |

The same variants in M4 (handle turned with the grasp), for comparison:

| variant (M4) | 26d4118b0c | 4e103843db | 49aff04f7a | e8825dea53 | 4-clip pool |
|---|---|---|---|---|---|
| b_level | 0/50 | 0/50 | 0/50 | 23/50 | 23/200 (0.08-0.17) |
| b_tilt | 4/50 | 0/50 | 18/50 | 17/50 | 39/200 (0.15-0.26) |
| naive | 0/50 | 0/50 | 0/50 | 0/50 | 0/200 (0.00-0.02) |

## Failures

category_v2 = 'pushed mug before close' if the run failed before the close and the mug ended more than 2 cm from its layout position, else the run's category (retarget/m4b_summary.py).

| variant | clip | failures | category | category_v2 |
|---|---|---|---|---|
| b_level | 26d4118b0c | 50 | palm could not reach the target 50 | pushed mug before close 31, palm could not reach the target 19 |
| b_level | 4e103843db | 50 | palm could not reach the target 50 | pushed mug before close 25, palm could not reach the target 25 |
| b_level | 49aff04f7a | 50 | palm could not reach the target 46, missed/knocked cup 4 | pushed mug before close 50 |
| b_level | e8825dea53 | 25 | palm could not reach the target 20, missed/knocked cup 5 | pushed mug before close 25 |
| b_level | pool | 175 | palm could not reach the target 166, missed/knocked cup 9 | pushed mug before close 131, palm could not reach the target 44 |
| b_tilt | 26d4118b0c | 45 | palm could not reach the target 45 | pushed mug before close 27, palm could not reach the target 18 |
| b_tilt | 4e103843db | 50 | palm could not reach the target 49, missed/knocked cup 1 | pushed mug before close 41, palm could not reach the target 9 |
| b_tilt | 49aff04f7a | 33 | palm could not reach the target 28, slip during lift 2, missed/knocked cup 2, released but not on the coaster 1 | pushed mug before close 21, palm could not reach the target 8, slip during lift 2, released but not on the coaster 1, missed/knocked cup 1 |
| b_tilt | e8825dea53 | 33 | palm could not reach the target 31, slip during lift 1, dropped while moving 1 | palm could not reach the target 31, slip during lift 1, dropped while moving 1 |
| b_tilt | pool | 161 | palm could not reach the target 153, missed/knocked cup 3, slip during lift 3, released but not on the coaster 1, dropped while moving 1 | pushed mug before close 89, palm could not reach the target 66, slip during lift 3, released but not on the coaster 1, missed/knocked cup 1, dropped while moving 1 |
| naive | 26d4118b0c | 50 | released but not on the coaster 50 | released but not on the coaster 50 |
| naive | 4e103843db | 50 | released but not on the coaster 50 | released but not on the coaster 50 |
| naive | 49aff04f7a | 50 | released but not on the coaster 50 | released but not on the coaster 50 |
| naive | e8825dea53 | 50 | released but not on the coaster 50 | released but not on the coaster 50 |
| naive | pool | 200 | released but not on the coaster 200 | released but not on the coaster 200 |

## Paired with the oracle (same 50 trials)

| variant | clip | both succeed | own yaw only | oracle only | neither | own yaw only (trials) | oracle only (trials) |
|---|---|---|---|---|---|---|---|
| b_level | 26d4118b0c | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| b_level | 4e103843db | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| b_level | 49aff04f7a | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| b_level | e8825dea53 | 24 | 1 | 20 | 5 | [22] | [2, 3, 6, 7, 9, 10, 13, 14, 20, 24, 31, 32, 33, 35, 36, 39, 41, 42, 47, 48] |
| b_level | pool | 24 | 1 | 152 | 23 |  |  |
| b_tilt | 26d4118b0c | 5 | 0 | 39 | 6 | [] | [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 39, 40, 41, 42, 43, 47, 48] |
| b_tilt | 4e103843db | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| b_tilt | 49aff04f7a | 16 | 1 | 28 | 5 | [22] | [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 19, 23, 24, 25, 28, 29, 31, 32, 33, 35, 36, 40, 43, 47, 48] |
| b_tilt | e8825dea53 | 14 | 3 | 30 | 3 | [38, 44, 49] | [0, 1, 2, 5, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 31, 32, 37, 39, 40, 43, 46, 48] |
| b_tilt | pool | 35 | 4 | 141 | 20 |  |  |
| naive | 26d4118b0c | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| naive | 4e103843db | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| naive | 49aff04f7a | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| naive | e8825dea53 | 0 | 0 | 44 | 6 | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48] |
| naive | pool | 0 | 0 | 176 | 24 |  |  |

## Paired with the same variant's M4 row (same 50 trials)

| variant | clip | both succeed | own yaw only | M4 (turned handle) only | neither | own yaw only (trials) | M4 (turned handle) only (trials) |
|---|---|---|---|---|---|---|---|
| b_level | 26d4118b0c | 0 | 0 | 0 | 50 | [] | [] |
| b_level | 4e103843db | 0 | 0 | 0 | 50 | [] | [] |
| b_level | 49aff04f7a | 0 | 0 | 0 | 50 | [] | [] |
| b_level | e8825dea53 | 23 | 2 | 0 | 25 | [18, 26] | [] |
| b_level | pool | 23 | 2 | 0 | 175 |  |  |
| b_tilt | 26d4118b0c | 4 | 1 | 0 | 45 | [2] | [] |
| b_tilt | 4e103843db | 0 | 0 | 0 | 50 | [] | [] |
| b_tilt | 49aff04f7a | 17 | 0 | 1 | 32 | [] | [4] |
| b_tilt | e8825dea53 | 17 | 0 | 0 | 33 | [] | [] |
| b_tilt | pool | 38 | 1 | 1 | 160 |  |  |
| naive | 26d4118b0c | 0 | 0 | 0 | 50 | [] | [] |
| naive | 4e103843db | 0 | 0 | 0 | 50 | [] | [] |
| naive | 49aff04f7a | 0 | 0 | 0 | 50 | [] | [] |
| naive | e8825dea53 | 0 | 0 | 0 | 50 | [] | [] |
| naive | pool | 0 | 0 | 0 | 200 |  |  |

Batch: 600 runs, 8 workers, 795.8 s, 0 errors, peak RSS per process 1557-1626 MB, none (CPU only, onnxruntime CPU).

The place correction reads privileged simulator state (the cup's true pose in the hand), as in M4 and for the oracle.
