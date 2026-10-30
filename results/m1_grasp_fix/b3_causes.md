# B3 - cause separation (oracle unchanged, trials 0-19)

R0 -> R1 isolates the size (diameter and height), R1 -> R2 the mass, R2 -> R3 the handle.

| run | object | mass (kg) | changed vs previous run | success | Wilson 95 % CI | failures (M1 categories) | lift slip in hand, mean (median) (mm) [n]; > 20 mm | cup rise for 10 cm lift, mean (median) (cm) [n] | thumb + finger + palm at lift frame: of 20 | of trials reaching the lift | parts touching at lift frame (thumb / index / middle / palm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | M1 cylinder, D 0.080 m, h 0.100 m | 0.15 | - (M1 baseline) | 20/20 | 84-100 % | - | 7.8 (6.8) [20]; 0 | 7.28 (7.16) [20] | 12/20 | 12/20 | 20 / 15 / 20 / 12 (of 20) |
| R1 | cylinder, D 0.099 m, h 0.091 m, no handle | 0.15 | size | 5/20 | 11-47 % | dropped while moving: 9, missed/knocked cup: 1, slip during lift: 5 | 22.7 (17.0) [19]; 5 | 5.76 (6.06) [19] | 14/20 | 14/19 | 19 / 4 / 19 / 14 (of 19) |
| R2 | cylinder, D 0.099 m, h 0.091 m, no handle | 0.40 | mass | 0/20 | 0-16 % | dropped while moving: 3, slip during lift: 17 | 22.2 (20.9) [20]; 14 | 4.93 (4.94) [20] | 14/20 | 14/20 | 20 / 8 / 20 / 14 (of 20) |
| R3 | full mug (config), handle 0.06 x 0.010 m | 0.40 | handle | 0/20 | 0-16 % | dropped while moving: 8, slip during lift: 12 | 20.8 (20.1) [20]; 10 | 5.11 (4.98) [20] | 13/20 | 13/20 | 20 / 6 / 20 / 13 (of 20) |

- Lift frame: first control step of the oracle's 'lift' phase (hand fully closed after the 0.5 s close hold, lift reference just started, palm not yet risen); contacts = data.contact after that step (its last 1 ms substep).
- Contact: by geom pair in data.contact: right-hand geom -> its body (thumb_0/1/2 = thumb, index_0/1 = index, middle_0/1 = middle, palm_link = palm); object = any geom of the cup body; grasp contact = thumb AND (index OR middle) AND palm. Trials that never reached the lift count as no contact in the 'of 20' column.
- Lift slip and rise: PickPlace's own measurements, over the trials that reached the end of the lift ([n]). The mean is the spec's number; the median and the count above the oracle's 20 mm lift-slip limit show the typical trial (a few lost cups pull the mean up).
- Lift-frame contacts are a 1 ms snapshot; contacts flicker between substeps. Thumb + finger + palm in the last / any / all of the frame's 20 substeps (information only): R0 12/12/12, R1 14/16/11, R2 14/14/11, R3 13/14/12; index finger touching in the last / any substep: R0 15/18, R1 4/6, R2 8/10, R3 6/9.
- Open question: Should 'at the lift frame' mean the last 1 ms substep of that 20 ms step (used), any of its 20 substeps, or all of them? Contacts flicker between substeps; summary.*.lift_frame.substep_sensitivity has all three counts.
- Cup yaw: R0-R2: 0 deg (round, no handle; as in M1); R3: the trial's handle yaw from trials() (= results/m1_pick_mug, asserted).
- R0 vs M1 (results/m1_pick + m1_pick_extended, runs 0-19): R0 20/20, M1 20/20; 20/20 trials identical in success, contact_closure, lift_slip_in_hand_cm, lift_cup_rise_cm, cup_off_coaster_center_cm, cup_tilt_deg, cup_final, sim_seconds.
- Runtime (four parallel CPU processes, no GPU): R0 100 s / peak RSS 1825 MB, R1 85 s / peak RSS 1660 MB, R2 77 s / peak RSS 1651 MB, R3 81 s / peak RSS 1688 MB.
- All numbers: results/m1_grasp_fix/b3_causes.json (raw runs: b3_runs/R0-R3.json).
