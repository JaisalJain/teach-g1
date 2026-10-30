# M5 candidate rollouts (training layouts only)

`python -m policy.m5 candidates`; 120 training layouts x 18 candidates = 2160 runs; exceptions 0 (0.0%; D1 stops above 5 %): branch **continue**.

b_hybrid_phi = b_hybrid (my approach to PG, the oracle's final approach, my transport) with the grasp heading phi. Each layout has one handle yaw.

| clip | candidate | phi [deg] | departure from my heading [deg] | success (Wilson 95 %) | failures (category_v2) |
|---|---|---|---|---|---|
| 26d4118b0c | phih | 81.9 | 0.0 | 41/120 (0.26-0.43) | palm could not reach the target 60, pushed mug before close 12, released but not on the coaster 6, dropped while moving 1 |
| 26d4118b0c | 75 | 75.0 | 6.9 | 62/120 (0.43-0.60) | palm could not reach the target 24, pushed mug before close 23, released but not on the coaster 10, slip during lift 1 |
| 26d4118b0c | 65 | 65.0 | 16.9 | 112/120 (0.87-0.97) | released but not on the coaster 5, slip during lift 2, pushed mug before close 1 |
| 26d4118b0c | 55 | 55.0 | 26.9 | 111/120 (0.86-0.96) | pushed mug before close 8, slip during lift 1 |
| 26d4118b0c | 45 | 45.0 | 36.9 | 111/120 (0.86-0.96) | pushed mug before close 8, slip during lift 1 |
| 4e103843db | phih | 85.8 | 0.0 | 31/120 (0.19-0.34) | palm could not reach the target 83, pushed mug before close 4, released but not on the coaster 2 |
| 4e103843db | 75 | 75.0 | 10.8 | 52/120 (0.35-0.52) | palm could not reach the target 47, pushed mug before close 16, released but not on the coaster 4, slip during lift 1 |
| 4e103843db | 65 | 65.0 | 20.8 | 94/120 (0.70-0.85) | palm could not reach the target 19, pushed mug before close 3, slip during lift 3, released but not on the coaster 1 |
| 4e103843db | 55 | 55.0 | 30.8 | 103/120 (0.78-0.91) | pushed mug before close 8, palm could not reach the target 6, slip during lift 3 |
| 4e103843db | 45 | 45.0 | 40.8 | 115/120 (0.91-0.98) | pushed mug before close 3, palm could not reach the target 1, dropped while moving 1 |
| 49aff04f7a | phih | 81.7 | 0.0 | 35/120 (0.22-0.38) | palm could not reach the target 33, pushed mug before close 29, released but not on the coaster 18, slip during lift 4, missed/knocked cup 1 |
| 49aff04f7a | 75 | 75.0 | 6.7 | 58/120 (0.40-0.57) | pushed mug before close 24, released but not on the coaster 21, palm could not reach the target 10, slip during lift 7 |
| 49aff04f7a | 65 | 65.0 | 16.7 | 97/120 (0.73-0.87) | released but not on the coaster 16, slip during lift 6, pushed mug before close 1 |
| 49aff04f7a | 55 | 55.0 | 26.7 | 100/120 (0.76-0.89) | released but not on the coaster 14, pushed mug before close 4, slip during lift 2 |
| 49aff04f7a | 45 | 45.0 | 36.7 | 111/120 (0.86-0.96) | pushed mug before close 6, released but not on the coaster 2, slip during lift 1 |
| e8825dea53 | phih | 61.7 | 0.0 | 96/120 (0.72-0.86) | released but not on the coaster 19, slip during lift 3, pushed mug before close 2 |
| e8825dea53 | 55 | 55.0 | 6.7 | 97/120 (0.73-0.87) | released but not on the coaster 16, pushed mug before close 7 |
| e8825dea53 | 45 | 45.0 | 16.7 | 112/120 (0.87-0.97) | pushed mug before close 6, released but not on the coaster 2 |

Success per clip x phi (rows: clips; columns: phi):

| clip | phi_h | 75 | 65 | 55 | 45 |
|---|---|---|---|---|---|
| 26d4118b0c (81.9) | 41/120 | 62/120 | 112/120 | 111/120 | 111/120 |
| 4e103843db (85.8) | 31/120 | 52/120 | 94/120 | 103/120 | 115/120 |
| 49aff04f7a (81.7) | 35/120 | 58/120 | 97/120 | 100/120 | 111/120 |
| e8825dea53 (61.7) | 96/120 | - | - | 97/120 | 112/120 |

Candidate-oracle ceiling (at least one candidate succeeds): **120/120 (0.97-1.00)** of the training layouts.

Fixed-best candidate F (highest training success, ties -> larger phi): **4e103843db 45** (phi 45.0 deg), 115/120 (0.91-0.98).
