| | 26d4118b0c | 49aff04f7a | 4e103843db | e8825dea53 |
|---|---|---|---|---|
| capture rate [Hz] / dropped frames | 30 / 0 | 60 / 0 | 60 / 19 | 60 / 0 |
| video frames decoded / depth PNGs | 249 / 250 | 478 / 479 | 513 / 514 | 473 / 474 |
| best RGB-depth lag (high-pass corr at lag 0 / -1) | 0 (0.40 / 0.01) | 0 (0.15 / 0.13) | 0 (0.42 / -0.01) | 0 (0.29 / 0.15) |
| table normal vs odometry up / IMU [deg] | 5.7 / 3.7 | 1.8 / 1.4 | 3.4 / 3.5 | 4.8 / 5.5 |
| optical axis from vertical [deg] | 1 | 31 | 37 | 39 |
| camera height above table / to cup [m] | 0.39 / 0.41 | 0.34 / 0.41 | 0.36 / 0.44 | 0.36 / 0.48 |
| table plane inliers (of conf-2 px) / RMS [mm] | 81% / 4.9 | 82% / 2.5 | 71% / 3.6 | 80% / 4.0 |
| frame conf 0/1/2 | 6%/18%/76% | 5%/24%/72% | 8%/27%/66% | 5%/30%/64% |
| cup: depth px / conf-2 share | 3337 / 25% | 3559 / 29% | 3284 / 17% | 2344 / 46% |
| cup: conf-2 height above table p5/p50/p95 [cm] | 1.1/1.4/2.0 | 1.8/6.2/8.5 | 1.2/6.1/8.6 | 1.6/6.7/8.8 |
| cup centre jitter, static frames [mm] | 2.8 (5 frames) | 2.0 (105 frames) | 2.8 (89 frames) | 1.8 (26 frames) |
| odometry max drift [mm / deg] | 13 / 2.2 | 8 / 1.3 | 5 / 2.5 | 18 / 2.7 |
| image shift first vs last frame [px] | 19 | 22 | 8 | 46 |
| depth changed > 2 cm in last frame | 9% | 15% | 14% | 46% |
