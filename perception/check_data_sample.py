"""Check that the committed data sample (data/raw/sample/<clip>/) is the original Stray Scanner recording, unchanged.

Loads both folders with perception.stray (the same loader every perception stage uses) and compares
  - sha256 of every file (the sample must be a byte-for-byte copy),
  - row counts, timestamps, intrinsics, camera poses, IMU, image sizes,
  - the depth and confidence arrays of the first, middle and last frame (through Recording.depth / .confidence),
  - the number of decoded video frames (must be N - 1 for N odometry rows, see perception/stray.py) and a hash of the
    first, middle and last decoded frame (downscaled to 192x144 by ffmpeg, so the check stays cheap).
Writes the comparison as JSON.

    python -m perception.check_data_sample [--clip 4e103843db] [--out results/m5/data_sample_check.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from perception import stray

REPO = Path(__file__).resolve().parents[1]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", default="4e103843db")
    ap.add_argument("--orig", type=Path, help="default: data/raw/<clip>")
    ap.add_argument("--sample", type=Path, help="default: data/raw/sample/<clip>")
    ap.add_argument("--out", type=Path, default=REPO / "results/m5/data_sample_check.json")
    args = ap.parse_args()
    orig = args.orig or REPO / "data/raw" / args.clip
    sample = args.sample or REPO / "data/raw/sample" / args.clip

    # 1. files, byte for byte
    files_o = sorted(p.relative_to(orig) for p in orig.rglob("*") if p.is_file())
    files_s = sorted(p.relative_to(sample) for p in sample.rglob("*") if p.is_file() and p.name != "README.md")
    mismatch = [str(f) for f in files_o if f in files_s and sha256(orig / f) != sha256(sample / f)]
    files = {"n_files_original": len(files_o), "n_files_sample_without_readme": len(files_s),
             "same_file_list": files_o == files_s,
             "missing_in_sample": [str(f) for f in files_o if f not in files_s],
             "extra_in_sample": [str(f) for f in files_s if f not in files_o],
             "sha256_mismatches": mismatch, "all_byte_identical": files_o == files_s and not mismatch,
             "sha256": {str(f): sha256(sample / f) for f in files_s if f.suffix in (".csv", ".mp4")}}

    # 2. the module's own loaders
    ro, rs = stray.load(orig), stray.load(sample)
    n = len(ro)
    picks = sorted({0, n // 2, n - 1})
    loaded = {
        "n_rows_original": len(ro), "n_rows_sample": len(rs),
        "t_equal": bool(np.array_equal(ro.t, rs.t)), "t_first_last": [float(rs.t[0]), float(rs.t[-1])],
        "K_equal": bool(np.array_equal(ro.K, rs.K)), "T_wc_equal": bool(np.array_equal(ro.T_wc, rs.T_wc)),
        "rgb_size": [list(ro.rgb_size), list(rs.rgb_size)], "depth_size": [list(ro.depth_size), list(rs.depth_size)],
        "imu_equal": all(np.array_equal(a, b) for a, b in zip(stray.read_imu(orig), stray.read_imu(sample))),
        "compared_frames": picks,
        "depth_equal": {str(i): bool(np.array_equal(ro.depth(i), rs.depth(i))) for i in picks},
        "confidence_equal": {str(i): bool(np.array_equal(ro.confidence(i), rs.confidence(i))) for i in picks},
        "depth_range_m_sample": {str(i): [float(rs.depth(i).min()), float(rs.depth(i).max())] for i in picks},
    }

    # 3. video: N - 1 decoded frames for N rows, same pixels
    picks_video = [0, (n - 1) // 2, n - 2]  # video positions: decoded frame k is odometry index k + 1
    vo, vs = decoded_frames(ro, picks_video), decoded_frames(rs, picks_video)
    video = {"n_rows": n, "expected_decoded_frames": n - 1,
             "decoded_frames_original": vo["decoded_frames"], "decoded_frames_sample": vs["decoded_frames"],
             "first_rgb_index": stray.FIRST_RGB_INDEX, "decode_size": list(SMALL),
             "compared_video_positions": picks_video, "frame_hashes_equal": vo["frame_hashes"] == vs["frame_hashes"],
             "frame_hashes_sample": {str(k): v for k, v in vs["frame_hashes"].items()}}

    ok = (files["all_byte_identical"] and loaded["n_rows_original"] == loaded["n_rows_sample"] and loaded["t_equal"]
          and loaded["K_equal"] and loaded["T_wc_equal"] and loaded["imu_equal"]
          and all(loaded["depth_equal"].values()) and all(loaded["confidence_equal"].values())
          and video["decoded_frames_original"] == video["decoded_frames_sample"] == n - 1 and video["frame_hashes_equal"])
    result = {"clip": args.clip, "original": str(orig.relative_to(REPO)), "sample": str(sample.relative_to(REPO)),
              "all_checks_pass": bool(ok), "files": files, "loaded_with_perception.stray": loaded, "video": video}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_checks_pass": ok, "n_rows": n, "decoded_frames": vs["decoded_frames"],
                      "byte_identical": files["all_byte_identical"], "out": str(args.out)}, indent=2))
    if not ok:
        raise SystemExit("data sample differs from the original")


if __name__ == "__main__":
    main()
