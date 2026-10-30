"""S1 -- frames and geometry conventions for one clip.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.frames --clip <id>

Writes data/work/<clip>/frames/:
- meta.npz: per canonical index i (= odometry row = depth/confidence file index): t, K (RGB pixels), K_depth, T_world_cam;
  plus rot_k (np.rot90 k of the upright view), image sizes and the dropped frames (dt > 1.5 x median dt; logged, not resampled).
- full/{i:06d}.jpg: upright full-res frames (JPEG q95) for WiLoR, i = 1..N-1 (i = 0 has no image).
- sam2/{j:05d}.jpg: upright half-res frames (720x960, JPEG q90) for SAM 2; SAM 2 frame j is canonical i = j + 1.

rot_k comes from world-up at i = 1 (stray.upright_rot90) and is fixed for the whole clip.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from perception import stray
from perception.common import RAW, stage_cli, stage_dir


def run(clip: str) -> dict:
    out = stage_dir(clip, "frames")
    rec = stray.load(RAW / clip)
    n = len(rec)
    assert rec.depth_size == (256, 192), rec.depth_size
    dt = np.diff(rec.t)
    dropped = np.flatnonzero(dt > 1.5 * np.median(dt)) + 1        # index i whose gap to i-1 is too long
    k = stray.upright_rot90(rec.T_wc[1, :3, :3])
    np.savez(out / "meta.npz", t=rec.t, K=rec.K, K_depth=np.stack([rec.K_depth(i) for i in range(n)]), T_world_cam=rec.T_wc,
             rot_k=k, rgb_size=np.array(rec.rgb_size), depth_size=np.array(rec.depth_size), dropped=dropped)

    (out / "full").mkdir(exist_ok=True)
    (out / "sam2").mkdir(exist_ok=True)
    n_img = 0
    for i, rgb in rec.rgb_frames():
        up = Image.fromarray(np.ascontiguousarray(np.rot90(rgb, k)))
        up.save(out / "full" / f"{i:06d}.jpg", quality=95)
        up.reduce(2).save(out / "sam2" / f"{i - 1:05d}.jpg", quality=90)     # 2x2 box average
        n_img += 1
    return {"n_frames": n, "n_images": n_img, "rot_k": int(k), "upright_size_wh": list(up.size),
            "half_size_wh": list(up.reduce(2).size), "dropped_frames": dropped.tolist(),
            "median_dt_s": float(np.median(dt)), "max_dt_s": float(dt.max())}


if __name__ == "__main__":
    stage_cli("frames", run)
