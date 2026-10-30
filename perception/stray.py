"""Read a Stray Scanner recording (iPhone LiDAR): RGB frames, metric depth, confidence, intrinsics, camera poses.

Format: third_party/stray-scanner/docs/format.md (app commit ec3e1dc). What the doc does not say, found in the app source
(StrayScanner/Helpers/*.swift) and checked on our own clips (docs/log.md, 2026-10-30):

- Frame alignment. depth/, confidence/ and odometry.csv have one entry per saved frame, index i = 0..N-1. rgb.mp4 also
  holds N frames, but the app stamps the first one at -1/60 s, so the MP4 edit list hides it and ffmpeg decodes N-1
  frames: decoded frame k is index k+1. Index 0 has no image; `rgb_frames` starts at index 1.
- Timing. The app stamps video frame n at n/60 s whatever the capture rate, so a 30 Hz recording plays at double speed.
  Use the odometry timestamps, never the video clock.
- Camera axes. The app rotates ARKit's camera axes (x right, y up, z backward) by 180 deg about x before writing the
  pose, so poses are camera-to-world with OpenCV camera axes: x right, y down, z forward (into the scene).
  World = ARKit's session frame: y up (gravity along -y), origin and heading = where the session started.
- Quaternions are scalar-last (qx, qy, qz, qw).
- Intrinsics are per frame, in RGB pixels (they drift ~0.2 % with focus); camera_matrix.csv is only the last frame's.
- Depth is uint16 millimetres at 256x192, registered to the RGB image (same field of view, 7.5x fewer pixels).
  ARKit fills every pixel; the confidence map (0 low, 1 medium, 2 high) says which values to trust.
- IMU acceleration is in g (CoreMotion), not m/s^2 as the doc says; axes are the phone's (CoreMotion device frame).

Dependencies: numpy, Pillow and the ffmpeg / ffprobe binaries (on PATH), so it runs in any of our envs.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FIRST_RGB_INDEX = 1  # index 0 is hidden by the MP4 edit list (see above)


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions, scalar last -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found on PATH (Ubuntu: apt install ffmpeg)")
    return path


@dataclass
class Recording:
    path: Path
    t: np.ndarray        # (N,) timestamps [s], ARKit clock
    T_wc: np.ndarray     # (N, 4, 4) camera-to-world poses (OpenCV camera axes, ARKit world, metres)
    K: np.ndarray        # (N, 3, 3) per-frame intrinsics in RGB pixels
    rgb_size: tuple      # (width, height) of rgb.mp4
    depth_size: tuple    # (width, height) of the depth / confidence maps

    def __len__(self) -> int:
        return len(self.t)

    def depth(self, i: int) -> np.ndarray:
        """Depth of frame i in metres, (H, W) float32."""
        return np.asarray(Image.open(self.path / "depth" / f"{i:06d}.png"), dtype=np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        """ARKit confidence of frame i, (H, W) uint8: 0 low, 1 medium, 2 high."""
        return np.asarray(Image.open(self.path / "confidence" / f"{i:06d}.png"))

    def K_depth(self, i: int) -> np.ndarray:
        """Intrinsics of frame i for the depth map. Same field of view as the RGB image, so scale by the size ratio
        (pixel centres: u_depth + 0.5 = s * (u_rgb + 0.5))."""
        s = self.depth_size[0] / self.rgb_size[0]
        K = self.K[i].copy()
        K[0, 0] *= s
        K[1, 1] *= s
        K[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
        return K

    def rgb_frames(self, step: int = 1, size: tuple | None = None):
        """Yield (i, rgb) for every `step`-th video frame; i = depth/odometry index (starts at 1), rgb = (H, W, 3)
        uint8. `size=(w, h)` lets ffmpeg resize (much faster than piping full 1920x1440 frames)."""
        w, h = size or self.rgb_size
        # passthrough: one output frame per decoded frame (no duplicates to fill the gaps of dropped frames)
        cmd = [_binary("ffmpeg"), "-v", "error", "-i", str(self.path / "rgb.mp4"), "-fps_mode", "passthrough"]
        if size:
            cmd += ["-vf", f"scale={w}:{h}:flags=area"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        i = FIRST_RGB_INDEX
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            while len(buf := proc.stdout.read(w * h * 3)) == w * h * 3:
                if (i - FIRST_RGB_INDEX) % step == 0:
                    yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                i += 1
        if i != len(self):
            raise RuntimeError(f"{self.path}: decoded {i - FIRST_RGB_INDEX} video frames, expected {len(self) - 1}")


def load(path) -> Recording:
    """Read the metadata of one recording folder (images are read lazily)."""
    path = Path(path)
    odo = np.loadtxt(path / "odometry.csv", delimiter=",", skiprows=1, usecols=range(13), ndmin=2)
    n = len(odo)
    if not np.array_equal(odo[:, 1], np.arange(n)):
        raise ValueError(f"{path}: odometry frame column is not 0..{n - 1}")
    for sub in ("depth", "confidence"):
        n_png = len(list((path / sub).glob("*.png")))
        if n_png != n:
            raise ValueError(f"{path}: {n_png} {sub} PNGs but {n} odometry rows")

    T_wc = np.tile(np.eye(4), (n, 1, 1))
    T_wc[:, :3, :3] = quat_xyzw_to_rot(odo[:, 5:9])
    T_wc[:, :3, 3] = odo[:, 2:5]
    K = np.zeros((n, 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = odo[:, 9], odo[:, 10], odo[:, 11], odo[:, 12], 1.0

    probe = subprocess.run([_binary("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path / "rgb.mp4")],
                           capture_output=True, text=True, check=True)
    rgb_size = tuple(int(v) for v in probe.stdout.strip().split(","))
    depth_size = Image.open(path / "depth" / "000000.png").size
    return Recording(path, odo[:, 0], T_wc, K, rgb_size, depth_size)


def read_imu(path):
    """imu.csv -> t (M,) [s], acc (M, 3) [g], gyro (M, 3) [rad/s], phone axes."""
    a = np.loadtxt(Path(path) / "imu.csv", delimiter=",", skiprows=1, ndmin=2)
    return a[:, 0], a[:, 1:4], a[:, 4:7]


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Depth map (H, W) [m] + its intrinsics -> (H, W, 3) points in the camera frame (OpenCV axes)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)


def upright_rot90(R_wc: np.ndarray) -> int:
    """k for np.rot90(image, k) so that world-up points up in the image (a phone held in portrait gives sideways frames).
    Arbitrary for a camera looking straight down."""
    ux, uy, _ = R_wc.T @ np.array([0.0, 1.0, 0.0])  # world up in camera axes; only its image-plane part matters
    if abs(ux) > abs(uy):
        return 1 if ux > 0 else 3
    return 0 if uy < 0 else 2
