"""Check that committed copies of Stray Scanner recordings are the original recordings, unchanged, and parse the same way.

For each clip id, loads the copy (default: data/raw/<clip> of this checkout) and the original (--orig-root/<clip>) with
perception.stray (the same loader every perception stage uses) and compares
  - sha256 of every file (the copy must be byte for byte the original),
  - row counts, timestamps, intrinsics, camera poses, IMU, image sizes,
  - the depth and confidence arrays of the first, middle and last frame (through Recording.depth / .confidence),
  - the frame alignment: the number of decoded video frames (must be N - 1 for N odometry rows, see perception/stray.py) and a
    hash of the first, middle and last decoded frame (downscaled to 192x144 by ffmpeg, so the check stays cheap).
Writes one JSON with a result per clip. Needs numpy, Pillow and the ffmpeg / ffprobe binaries (no GPU, no perception models).

An earlier one-clip version checked the M5 data sample (results/m5/data_sample_check.json). This version checks the four committed
clips of a fresh clone against the originals (results/final/raw_clips_check.json):

    python -m perception.check_data_sample --orig-root ~/hoi/data/raw [--clips ID ...] [--copy-root data/raw] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from perception import stray

REPO = Path(__file__).resolve().parents[1]
CLIPS = ("26d4118b0c", "4e103843db", "49aff04f7a", "e8825dea53")
SMALL = (192, 144)  # decode size for the video-frame check


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decoded_frames(rec: stray.Recording, picks: list[int]) -> dict:
    """Decode the whole video, count the frames and hash the ones at video positions `picks` (0-based, decode order)."""
    n, hashes = 0, {}
    for i, rgb in rec.rgb_frames(size=SMALL):
        k = i - stray.FIRST_RGB_INDEX
        if k in picks:
            hashes[k] = hashlib.sha256(rgb.tobytes()).hexdigest()
        n += 1
    return {"decoded_frames": n, "frame_hashes": hashes}


def check_clip(clip: str, orig: Path, copy: Path) -> dict:
    """Every comparison above for one clip: original folder `orig`, committed copy `copy`."""
    # 1. files, byte for byte
    files_o = sorted(p.relative_to(orig) for p in orig.rglob("*") if p.is_file())
    files_c = sorted(p.relative_to(copy) for p in copy.rglob("*") if p.is_file() and p.name != "README.md")
    mismatch = [str(f) for f in files_o if f in files_c and sha256(orig / f) != sha256(copy / f)]
    files = {"n_files_original": len(files_o), "n_files_copy": len(files_c),
             "same_file_list": files_o == files_c,
             "missing_in_copy": [str(f) for f in files_o if f not in files_c],
             "extra_in_copy": [str(f) for f in files_c if f not in files_o],
             "sha256_mismatches": mismatch, "all_byte_identical": files_o == files_c and not mismatch,
             "sha256": {str(f): sha256(copy / f) for f in files_c if f.suffix in (".csv", ".mp4")}}

    # 2. the module's own loaders
    ro, rc = stray.load(orig), stray.load(copy)
    n = len(ro)
    picks = sorted({0, n // 2, n - 1})
    loaded = {
        "n_rows_original": len(ro), "n_rows_copy": len(rc),
        "t_equal": bool(np.array_equal(ro.t, rc.t)), "t_first_last": [float(rc.t[0]), float(rc.t[-1])],
        "K_equal": bool(np.array_equal(ro.K, rc.K)), "T_wc_equal": bool(np.array_equal(ro.T_wc, rc.T_wc)),
        "rgb_size": [list(ro.rgb_size), list(rc.rgb_size)], "depth_size": [list(ro.depth_size), list(rc.depth_size)],
        "imu_equal": all(np.array_equal(a, b) for a, b in zip(stray.read_imu(orig), stray.read_imu(copy))),
        "compared_frames": picks,
        "depth_equal": {str(i): bool(np.array_equal(ro.depth(i), rc.depth(i))) for i in picks},
        "confidence_equal": {str(i): bool(np.array_equal(ro.confidence(i), rc.confidence(i))) for i in picks},
        "depth_range_m_copy": {str(i): [float(rc.depth(i).min()), float(rc.depth(i).max())] for i in picks},
    }

    # 3. frame alignment: N - 1 decoded frames for N rows, same pixels
    picks_video = [0, (n - 1) // 2, n - 2]  # video positions: decoded frame k is odometry index k + 1
    vo, vc = decoded_frames(ro, picks_video), decoded_frames(rc, picks_video)
    video = {"n_rows": n, "expected_decoded_frames": n - 1,
             "decoded_frames_original": vo["decoded_frames"], "decoded_frames_copy": vc["decoded_frames"],
             "first_rgb_index": stray.FIRST_RGB_INDEX, "decode_size": list(SMALL),
             "compared_video_positions": picks_video, "frame_hashes_equal": vo["frame_hashes"] == vc["frame_hashes"],
             "frame_hashes_copy": {str(k): v for k, v in vc["frame_hashes"].items()}}

    ok = (files["all_byte_identical"] and loaded["n_rows_original"] == loaded["n_rows_copy"] and loaded["t_equal"]
          and loaded["K_equal"] and loaded["T_wc_equal"] and loaded["imu_equal"]
          and all(loaded["depth_equal"].values()) and all(loaded["confidence_equal"].values())
          and video["decoded_frames_original"] == video["decoded_frames_copy"] == n - 1 and video["frame_hashes_equal"])
    return {"clip": clip, "original": str(orig), "copy": str(copy), "all_checks_pass": bool(ok), "files": files,
            "loaded_with_perception.stray": loaded, "video": video}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", nargs="+", default=list(CLIPS))
    ap.add_argument("--orig-root", type=Path, required=True, help="folder holding the original recordings <clip>/")
    ap.add_argument("--copy-root", type=Path, default=REPO / "data/raw", help="folder holding the copies <clip>/ (default: this checkout's data/raw)")
    ap.add_argument("--out", type=Path, default=REPO / "results/final/raw_clips_check.json")
    args = ap.parse_args()
    orig_root, copy_root = args.orig_root.expanduser().resolve(), args.copy_root.expanduser().resolve()
    if orig_root == copy_root:
        raise SystemExit("--orig-root and --copy-root are the same folder: nothing would be compared")

    results = {c: check_clip(c, orig_root / c, copy_root / c) for c in args.clips}
    ok = all(r["all_checks_pass"] for r in results.values())
    out = {"what": "committed copies of the Stray Scanner recordings vs the originals (perception/check_data_sample.py)",
           "orig_root": str(orig_root), "copy_root": str(copy_root), "all_checks_pass": ok, "clips": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"all_checks_pass": ok, **{c: {"pass": r["all_checks_pass"], "n_rows": r["video"]["n_rows"],
                                                     "decoded_frames": r["video"]["decoded_frames_copy"],
                                                     "byte_identical": r["files"]["all_byte_identical"]} for c, r in results.items()},
                      "out": str(args.out)}, indent=2))
    if not ok:
        raise SystemExit("a committed clip differs from its original")


if __name__ == "__main__":
    main()
