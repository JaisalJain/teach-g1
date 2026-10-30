"""Compare the 4 test camera views (same cup -> coaster pick-and-place) before recording the real demos.

    env -u PYTHONPATH conda run -n hoi-sim python -m perception.check_views

Per clip (writes results/m2_views/):
- format: file counts, decoded video frames, capture rate, dropped frames, intrinsics, depth / confidence / IMU units;
- alignment: which depth index each decoded video frame belongs to (cross-correlate RGB and depth motion);
- axis convention: table normal fitted in the camera frame vs gravity from the odometry pose (and from the IMU);
- table plane: RANSAC on the median high-confidence depth of the frames before the hand enters;
- cup depth: confidence and height above the table of the depth pixels inside a hand-labelled cup outline;
- camera motion: odometry pose spread and the image shift of every frame vs. frame 1 (phase correlation);
- images: depth-on-RGB overlay (jpg + mp4), contact sheet of 24 frames, cup label check. Frames are rotated upright.
"""

from __future__ import annotations

import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from perception import stray

RAW = Path("data/raw")
OUT = Path("results/m2_views")
CLIPS = ["26d4118b0c", "49aff04f7a", "4e103843db", "e8825dea53"]

# Cup outline (mug body without the handle) on the first frame, full-res RGB pixels, labelled by eye on a pixel grid.
# Drawn back onto the frame in <clip>/cup_label.jpg to check it.
CUP_OUTLINE = {
    "26d4118b0c": [(910 + 244 * np.cos(a), 1170 + 244 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)],
    "49aff04f7a": [(630, 720), (690, 620), (800, 584), (940, 592), (1040, 610), (1110, 670), (1150, 760), (1150, 860),
                   (1120, 940), (1060, 980), (940, 1020), (840, 1046), (740, 1030), (660, 960), (624, 860)],
    "4e103843db": [(760, 940), (800, 860), (880, 820), (1000, 810), (1120, 816), (1200, 860), (1244, 960), (1250, 1060),
                   (1224, 1150), (1160, 1200), (1040, 1230), (940, 1256), (840, 1230), (780, 1160), (756, 1050)],
    "e8825dea53": [(516, 1040), (550, 960), (620, 934), (720, 924), (840, 924), (910, 960), (940, 1060), (936, 1160),
                   (900, 1220), (800, 1250), (700, 1280), (610, 1280), (540, 1220), (516, 1130)],
}

CONF_HIGH = 2
PLANE_TOL = 0.01      # m, RANSAC inlier distance
CHANGE_M = 0.02       # m, depth change vs. the first frame that counts as "something moved here"
ENTER_FRAC = 0.005    # fraction of the image that changed = the hand has entered


def fit_plane(pts, rng, iters=300):
    """RANSAC + least-squares refit. Returns (n, d) with n.p + d = 0, n pointing to the camera side (d > 0)."""
    best = None
    for _ in range(iters):
        a, b, c = pts[rng.integers(len(pts), size=3)]   # a repeated point gives a zero normal -> skipped
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = np.count_nonzero(np.abs(pts @ n - n @ a) < PLANE_TOL)
        if best is None or count > best[0]:
            best = (count, n, -n @ a)
    _, n, d = best
    inl = pts[np.abs(pts @ n + d) < PLANE_TOL]
    centre = inl.mean(0)
    n = np.linalg.svd(inl - centre, full_matrices=False)[2][2]   # smallest singular direction = normal
    d = -n @ centre
    if d < 0:
        n, d = -n, -d
    return n, d


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def image_shift(a, b):
    """Integer translation (dx, dy) in pixels between two grey images (phase correlation)."""
    F = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))
    c = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    h, w = c.shape
    return (dx if dx <= w // 2 else dx - w), (dy if dy <= h // 2 else dy - h)


def highpass(x, win=9):
    return x - np.convolve(x, np.ones(win) / win, mode="same")


def colormap(x):
    """x in [0, 1] -> RGB uint8 (blue - cyan - yellow - red)."""
    anchors = np.array([[48, 18, 160], [30, 160, 230], [120, 230, 110], [250, 200, 40], [200, 30, 20]], float)
    pos = np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(np.clip(x, 0, 1), pos, anchors[:, c]) for c in range(3)], -1).astype(np.uint8)


def overlay(rgb, depth, conf, near, far):
    """Depth colour over the RGB frame (both at the RGB frame's size); low-confidence pixels in grey."""
    h, w = rgb.shape[:2]
    up = lambda a: np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))
    col = colormap((up(depth) - near) / (far - near)).astype(float)
    col[up(conf) == 0] = 128
    return (0.5 * rgb + 0.5 * col).astype(np.uint8)


def outline_mask(outline, rgb_size, size):
    """Rasterise an RGB-pixel outline at another resolution (pixel centres)."""
    s = size[0] / rgb_size[0]
    img = Image.new("L", size, 0)
    ImageDraw.Draw(img).polygon([((x + 0.5) * s - 0.5, (y + 0.5) * s - 0.5) for x, y in outline], fill=1)
    return np.asarray(img).astype(bool)


def analyse(clip):
    rng = np.random.default_rng(0)
    rec = stray.load(RAW / clip)
    n = len(rec)
    out = OUT / clip
    out.mkdir(parents=True, exist_ok=True)
    r = {"clip": clip}

    # ---------------- format ----------------
    dt = np.diff(rec.t)
    dt0 = float(np.median(dt))
    conf_hist = np.mean([np.bincount(rec.confidence(i).ravel(), minlength=3) for i in range(0, n, 10)], 0)
    t_imu, acc, _ = stray.read_imu(RAW / clip)
    K_csv = np.loadtxt(RAW / clip / "camera_matrix.csv", delimiter=",")
    r["format"] = {
        "n_odometry_rows": n,
        "n_depth_png": len(list((RAW / clip / "depth").glob("*.png"))),
        "n_confidence_png": len(list((RAW / clip / "confidence").glob("*.png"))),
        "distortion_dir": (RAW / clip / "distortion").exists(),
        "rgb_size": rec.rgb_size, "depth_size": rec.depth_size,
        "capture_hz": round(1 / dt0, 2),
        "duration_s": round(float(rec.t[-1] - rec.t[0]), 3),
        "dropped_frames": int(np.sum(np.round(dt / dt0) - 1)),
        "camera_matrix_csv_eq_last_odometry_K": bool(np.allclose(K_csv, rec.K[-1], atol=1e-3)),
        "fx_range": [float(rec.K[:, 0, 0].min()), float(rec.K[:, 0, 0].max())],
        "cx_cy_mean": [float(rec.K[:, 0, 2].mean()), float(rec.K[:, 1, 2].mean())],
        "depth_dtype_png": str(np.asarray(Image.open(RAW / clip / "depth/000000.png")).dtype),
        "depth_zero_frac": float(np.mean([np.mean(rec.depth(i) == 0) for i in range(0, n, 10)])),
        "confidence_frac_0_1_2": (conf_hist / conf_hist.sum()).round(3).tolist(),
        "imu_rows": len(t_imu), "imu_hz": round(1 / float(np.median(np.diff(t_imu))), 1),
        "imu_acc_norm_median": round(float(np.median(np.linalg.norm(acc, axis=1))), 3),
    }

    # ---------------- RGB frames (depth resolution, for motion + overlays) ----------------
    dw, dh = rec.depth_size
    small = {i: f for i, f in rec.rgb_frames(size=(dw * 2, dh * 2))}       # 512x384
    r["format"]["n_video_frames_decoded"] = len(small)
    depths = np.stack([rec.depth(i) for i in range(n)])
    confs = np.stack([rec.confidence(i) for i in range(n)])

    # ---------------- alignment: RGB motion vs depth motion ----------------
    gray = {i: f[::2, ::2].mean(-1) for i, f in small.items()}               # 256x192
    idx = sorted(gray)
    m_rgb = np.array([np.abs(gray[i] - gray[i - 1]).mean() for i in idx[1:]])  # motion into frame i (i >= 2)
    m_dep = np.array([np.abs(np.clip(depths[i] - depths[i - 1], -0.2, 0.2)).mean() for i in range(1, n)])  # into i
    for name, f in (("raw", lambda x: x), ("highpass", highpass)):   # high-pass = sharper peak
        a_all, b_all = f(m_rgb), f(m_dep)
        lags = {}
        for lag in range(-3, 4):   # video frame labelled i compared with depth frame i + lag
            a = [a_all[k] for k, i in enumerate(idx[1:]) if 1 <= i + lag < n]
            b = [b_all[i + lag - 1] for i in idx[1:] if 1 <= i + lag < n]
            lags[lag] = round(float(np.corrcoef(a, b)[0, 1]), 3)
        r[f"alignment_corr_by_lag_{name}"] = lags
        r[f"alignment_best_lag_{name}"] = max(lags, key=lags.get)

    # ---------------- hand entry (depth change vs. frame 1) -> static frames ----------------
    Kd = [rec.K_depth(i) for i in range(n)]
    cup = outline_mask(CUP_OUTLINE[clip], rec.rgb_size, rec.depth_size)
    both = (confs >= 1) & (confs[1] >= 1)
    changed = ((np.abs(depths - depths[1]) > CHANGE_M) & both).mean((1, 2))
    enter = int(np.argmax(changed > ENTER_FRAC))
    static = list(range(1, enter - 5))
    r["hand_enters_index"] = enter
    r["hand_enters_s"] = round(float(rec.t[enter] - rec.t[0]), 2)
    r["static_frames"] = [static[0], static[-1]]

    # plane on the median depth of the static frames (less noise)
    dmed = np.median(depths[static], 0)
    cmin = confs[static].min(0)
    K1 = Kd[static[0]]
    P = stray.backproject(dmed, K1)
    pts = P[cmin == CONF_HIGH]
    n_c, d_c = fit_plane(pts, rng)
    dist = np.abs(pts @ n_c + d_c)
    inl = dist < PLANE_TOL
    optical = np.array([0, 0, 1.0])
    r["table"] = {
        "n_points_conf2": int(len(pts)),
        "inlier_frac_of_conf2": round(float(inl.mean()), 3),
        "inlier_frac_of_image": round(float(inl.sum() / dmed.size), 3),
        "inlier_rms_mm": round(float(np.sqrt(np.mean(dist[inl] ** 2)) * 1000), 2),
        "inlier_abs_p95_mm": round(float(np.percentile(dist[inl], 95) * 1000), 2),
        "camera_height_above_table_m": round(float(d_c), 3),
        "optical_axis_to_table_normal_deg": round(angle_deg(optical, -n_c), 1),  # 0 = looking straight down
    }

    # ---------------- axis convention: table normal (camera frame, from depth) vs gravity (odometry, IMU) ----------
    R = rec.T_wc[static[0], :3, :3]
    up_w = np.array([0, 1.0, 0])
    flip = np.diag([1.0, -1, -1])
    q_xyzw = np.loadtxt(RAW / clip / "odometry.csv", delimiter=",", skiprows=1, usecols=range(5, 9))[static[0]]
    R_wxyz = stray.quat_xyzw_to_rot(np.roll(q_xyzw, -1))    # reads the columns as (w, x, y, z)
    hyp = {
        "opencv_axes_cam_to_world_xyzw (expected)": R.T @ up_w,
        "arkit_axes (y up, z back)": flip @ R.T @ up_w,
        "world_to_cam (transposed)": R @ up_w,
        "quaternion_wxyz": R_wxyz.T @ up_w,
    }
    r["axis_test_angle_table_normal_vs_up_deg"] = {k: round(angle_deg(n_c, v), 2) for k, v in hyp.items()}
    # IMU: CoreMotion device axes (x right, y top, z out of the screen, portrait) -> OpenCV camera axes of the
    # back camera: x_cv = -y_dev, y_cv = -x_dev, z_cv = -z_dev. Accelerometer at rest reads gravity, i.e. "down".
    sel = (t_imu >= rec.t[static[0]]) & (t_imu <= rec.t[static[-1]])
    a = acc[sel].mean(0)
    down_cv = np.array([-a[1], -a[0], -a[2]])
    r["imu_down_vs_table_normal_deg"] = round(angle_deg(down_cv, -n_c), 2)

    # ---------------- cup depth ----------------
    heights, centres, conf_fr = [], [], []
    for i in static:
        p = stray.backproject(depths[i], Kd[i])
        hgt = p @ n_c + d_c
        c = confs[i][cup]
        conf_fr.append(np.bincount(c, minlength=3) / c.size)
        good = cup & (confs[i] == CONF_HIGH) & (hgt > 0.01)
        heights.append(hgt[cup & (confs[i] == CONF_HIGH)])
        q = p[good]
        q = q - np.outer(q @ n_c + d_c, n_c)                     # project onto the table
        centres.append(np.median(q, 0))
    hh = np.concatenate(heights)
    centres = np.array(centres)
    r["cup"] = {
        "n_depth_px_in_outline": int(cup.sum()),
        "conf_frac_0_1_2": np.mean(conf_fr, 0).round(3).tolist(),
        "height_above_table_cm_p5_p50_p95_conf2": (np.percentile(hh, [5, 50, 95]) * 100).round(1).tolist(),
        "frac_conf2_px_above_1cm": round(float(np.mean(hh > 0.01)), 3),
        "centre_jitter_mm_std_over_static_frames": round(float(np.linalg.norm(centres.std(0)) * 1000), 2),
        "camera_to_cup_m": round(float(np.linalg.norm(np.median(centres, 0))), 3),
    }

    # ---------------- camera motion ----------------
    pos = rec.T_wc[:, :3, 3]
    rot_dev = [np.degrees(np.arccos(np.clip((np.trace(rec.T_wc[0, :3, :3].T @ rec.T_wc[i, :3, :3]) - 1) / 2, -1, 1)))
               for i in range(n)]
    # first vs. last frame: the hand is out of view in both (checked on the contact sheets), so only the camera moved
    shift = np.array(image_shift(small[1].mean(-1), small[n - 1].mean(-1))) * rec.rgb_size[0] / small[1].shape[1]
    r["camera_motion"] = {
        "odometry_max_translation_from_start_mm": round(float(np.linalg.norm(pos - pos[0], axis=1).max() * 1000), 1),
        "odometry_max_rotation_from_start_deg": round(float(max(rot_dev)), 2),
        "image_shift_first_vs_last_px_fullres": round(float(np.linalg.norm(shift)), 1),
        "depth_changed_frac_last_frame": round(float(changed[-1]), 3),   # cup moved (~0.1) + any camera motion
    }

    # ---------------- images ----------------
    k = stray.upright_rot90(R)
    r["upright_rot90_k"] = k
    near, far = np.percentile(dmed, [2, 98])
    peak = int(np.argmax(changed))                                 # hand most in view
    big = {}
    for i, f in rec.rgb_frames(size=(960, 720)):
        if i in (static[0], peak):
            big[i] = f
    Image.fromarray(np.rot90(overlay(big[peak], depths[peak], confs[peak], near, far), k)).save(
        out / "overlay.jpg", quality=88)
    lab = Image.fromarray(big[static[0]])
    ImageDraw.Draw(lab).polygon([(x / 2, y / 2) for x, y in CUP_OUTLINE[clip]], outline=(255, 0, 255), width=3)
    Image.fromarray(np.rot90(np.asarray(lab), k)).save(out / "cup_label.jpg", quality=88)

    # overlay video at the true capture rate (rgb.mp4's own clock is wrong for 30 Hz clips)
    frames = [np.rot90(overlay(small[i], depths[i], confs[i], near, far), k) for i in sorted(small)]
    hh_, ww_ = frames[0].shape[:2]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ww_}x{hh_}",
                            "-r", str(round(1 / dt0)), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "26",
                            str(out / "overlay.mp4")], stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(np.ascontiguousarray(f).tobytes())
    enc.stdin.close()
    enc.wait()

    # contact sheet: 24 frames evenly over the clip
    pick = np.linspace(1, n - 1, 24).round().astype(int)
    tiles = [np.rot90(small[i], k) for i in pick]
    th, tw = tiles[0].shape[:2]
    cols = 6 if tw > th else 8
    sheet = Image.new("RGB", (cols * tw, -(-len(tiles) // cols) * th))
    for j, (i, t) in enumerate(zip(pick, tiles)):
        tile = Image.fromarray(np.ascontiguousarray(t))
        ImageDraw.Draw(tile).text((4, 4), f"#{i} {rec.t[i] - rec.t[0]:.1f}s", fill=(255, 255, 0))
        sheet.paste(tile, ((j % cols) * tw, (j // cols) * th))
    sheet.save(out / "contact_sheet.jpg", quality=80)

    r["changed_frac_every_10th_frame"] = changed[::10].round(3).tolist()
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(len(CLIPS)) as pool:
        results = pool.map(analyse, CLIPS)
    (OUT / "views.json").write_text(json.dumps({r["clip"]: r for r in results}, indent=1, default=float))
    results = list(json.loads((OUT / "views.json").read_text()).values())   # table from exactly what was saved
    rows = [
        ("capture rate [Hz] / dropped frames", lambda r: f"{r['format']['capture_hz']:.0f} / {r['format']['dropped_frames']}"),
        ("video frames decoded / depth PNGs", lambda r: f"{r['format']['n_video_frames_decoded']} / {r['format']['n_depth_png']}"),
        ("best RGB-depth lag (high-pass corr at lag 0 / -1)", lambda r: f"{r['alignment_best_lag_highpass']} "
         f"({r['alignment_corr_by_lag_highpass']['0']:.2f} / {r['alignment_corr_by_lag_highpass']['-1']:.2f})"),
        ("table normal vs odometry up / IMU [deg]", lambda r: f"{list(r['axis_test_angle_table_normal_vs_up_deg'].values())[0]:.1f}"
         f" / {r['imu_down_vs_table_normal_deg']:.1f}"),
        ("optical axis from vertical [deg]", lambda r: f"{r['table']['optical_axis_to_table_normal_deg']:.0f}"),
        ("camera height above table / to cup [m]", lambda r: f"{r['table']['camera_height_above_table_m']:.2f} / {r['cup']['camera_to_cup_m']:.2f}"),
        ("table plane inliers (of conf-2 px) / RMS [mm]", lambda r: f"{r['table']['inlier_frac_of_conf2']:.0%} / {r['table']['inlier_rms_mm']:.1f}"),
        ("frame conf 0/1/2", lambda r: "/".join(f"{x:.0%}" for x in r["format"]["confidence_frac_0_1_2"])),
        ("cup: depth px / conf-2 share", lambda r: f"{r['cup']['n_depth_px_in_outline']} / {r['cup']['conf_frac_0_1_2'][2]:.0%}"),
        ("cup: conf-2 height above table p5/p50/p95 [cm]", lambda r: "/".join(f"{x:.1f}" for x in r["cup"]["height_above_table_cm_p5_p50_p95_conf2"])),
        ("cup centre jitter, static frames [mm]", lambda r: f"{r['cup']['centre_jitter_mm_std_over_static_frames']:.1f} "
         f"({r['static_frames'][1]} frames)"),
        ("odometry max drift [mm / deg]", lambda r: f"{r['camera_motion']['odometry_max_translation_from_start_mm']:.0f} / "
         f"{r['camera_motion']['odometry_max_rotation_from_start_deg']:.1f}"),
        ("image shift first vs last frame [px]", lambda r: f"{r['camera_motion']['image_shift_first_vs_last_px_fullres']:.0f}"),
        ("depth changed > 2 cm in last frame", lambda r: f"{r['camera_motion']['depth_changed_frac_last_frame']:.0%}"),
    ]
    table = "| | " + " | ".join(r["clip"] for r in results) + " |\n|---|" + "---|" * len(results) + "\n"
    table += "".join(f"| {name} | " + " | ".join(f(r) for r in results) + " |\n" for name, f in rows)
    (OUT / "table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
