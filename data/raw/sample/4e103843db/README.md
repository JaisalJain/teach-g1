# Data sample: clip `4e103843db` (90-degree view)

One of the 4 clips this project is built on (the other three are not committed). It is the original Stray Scanner recording,
unchanged.

## How it was recorded
- App: [Stray Scanner](https://github.com/StrayRobots/scanner) on an iPhone with a LiDAR sensor (RGB video, LiDAR depth,
  depth confidence, camera intrinsics, camera odometry, IMU). The phone was propped by hand.
- Task: with my right hand, I pick up my mug with a side wrap, place it on the coaster, release it and take the hand away.
  The mug is a glossy dark-green ceramic mug with a handle; the coaster is a flat resin disc.
- Scene: I sat at the near long edge of a lap desk (60 cm long, about 38.5 cm deep) that lay on a bed.
- View: the 90-degree view. The camera is at my side, square to the desk edge, about 37 degrees from vertical.
- Length: 514 frames over 8.9 s (the app drops a few frames, so the timestamps are not perfectly even).

## Files and formats
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
- **Frame alignment:** `depth/`, `confidence/` and `odometry.csv` all have N = 514 entries (index 0 to 513), but `rgb.mp4` decodes to
  **N - 1 = 513** frames: the first video frame is hidden by the MP4 edit list. Decoded frame k belongs to index k + 1; index 0
  has no image. Use the odometry timestamps for time, never the video clock (the app stamps the video as if it ran at 60 fps).
- A parser that applies all of this is `perception/stray.py` in the repo.

## What was changed
Nothing. All files were copied byte for byte from the original recording (same sha256, checked by
`perception/check_data_sample.py`, result in `results/m5/data_sample_check.json`).
