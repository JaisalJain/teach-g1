# My recordings: four Stray Scanner clips

These are the four clips this project is built on, exactly as the app saved them. Each clip is a separate execution of the same
task, filmed from a different viewpoint. Nothing was edited, re-encoded or removed.

## How they were recorded
- App: [Stray Scanner](https://github.com/StrayRobots/scanner) on an iPhone with a LiDAR sensor (RGB video, LiDAR depth, depth
  confidence, camera intrinsics, camera odometry, IMU). The phone was propped by hand.
- Task: with my right hand, I pick up my mug with a side wrap, place it on the coaster, release it and take the hand away.
  The mug is a glossy dark-green ceramic mug with a handle; the coaster is a flat resin disc (sizes in `config/objects.yaml`).
- Scene: I sat at the near long edge of a lap desk (60 cm long, about 38.5 cm deep) that lay on a bed.

| clip | view | optical axis from vertical | capture rate | rows (depth / pose) | decoded video frames | length |
|---|---|---|---|---|---|---|
| `26d4118b0c` | top-down | 1 deg | 30 Hz | 250 | 249 | 8.30 s |
| `4e103843db` | 90 deg: at my side, square to the desk edge | 37 deg | 60 Hz (19 dropped frames) | 514 | 513 | 8.87 s |
| `49aff04f7a` | 45 deg (left / right not confirmed) | 31 deg | 60 Hz | 479 | 478 | 7.97 s |
| `e8825dea53` | 45 deg (left / right not confirmed) | 39 deg | 60 Hz | 474 | 473 | 7.89 s |

Sources: `results/m2_perception/inventory.json` (rows, frames, length) and `results/m2_views/table.md` (axis angle, rate, dropped
frames). Whether each 45-degree view is on my left or my right is not confirmed.

## Files and formats (each clip folder)
| file | content |
|---|---|
| `rgb.mp4` | RGB video, 1920x1440, HEVC. The phone was held in portrait, so the frames are sideways. |
| `depth/NNNNNN.png` | LiDAR depth, uint16 in **millimetres**, 256x192, registered to the RGB image (same field of view; scale K by 256/1920). |
| `confidence/NNNNNN.png` | ARKit depth confidence, uint8 0 (low), 1 (medium), 2 (high). |
| `odometry.csv` | one row per frame: `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y`. Quaternion is scalar-last. `fx fy cx cy` are the intrinsics of that frame in RGB pixels. |
| `camera_matrix.csv` | the 3x3 intrinsics K of the last frame (RGB pixels). |
| `imu.csv` | `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`: acceleration in **g** (not m/s^2) and angular rate (read as rad/s by `perception/stray.py`), in the phone's own axes. |

- Poses are camera-to-world with **OpenCV camera axes** (x right, y down, z forward). The world is the ARKit session frame,
  **y up** (gravity along -y).
- **Frame alignment:** `depth/`, `confidence/` and `odometry.csv` have N entries (index 0 to N - 1), but `rgb.mp4` decodes to
  **N - 1** frames: the first video frame is hidden by the MP4 edit list. Decoded frame k belongs to index k + 1; index 0 has no
  image. Use the odometry timestamps for time, never the video clock (the app stamps the video as if it ran at 60 fps, also the
  30 Hz clip).
- A parser that applies all of this is `perception/stray.py`. `perception/check_data_sample.py` compares committed copies with the
  originals (sha256 of every file, the parsed rows and the frame alignment); its result for a fresh clone is
  `results/final/raw_clips_check.json`.

## Licence
My recordings in this folder are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Jaisal Jain, 2026).
The code in this repository is MIT (`LICENSE`).
