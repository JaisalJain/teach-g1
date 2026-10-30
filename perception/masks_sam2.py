"""S3 -- object masks with the SAM 2.1 video predictor: 1 = mug body (without the handle), 2 = mug handle, 3 = coaster.

    # prompt tools (not cached stages):
    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.masks_sam2 --clip <id> --grid
    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.masks_sam2 --clip <id> --preview
    # the stage (propagation over the whole clip):
    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.masks_sam2 --clip <id>

Prompts are given on SAM 2 frame j = 0 (canonical i = 1, before the hand enters), on the upright half-res frames of S1
(720x960), and stored in perception/prompts/<clip>.json (committed) with the source of every box / point:
- "auto": the mug is the only strongly green thing on the desk, so a fixed HSV rule (GREEN_HSV below) finds it; its
  bounding box and its deepest interior point (distance-transform maximum, which lies in the body, not the thin handle)
  are derived automatically (--grid prints them).
- "manual": points read off the frame with a 50 px grid drawn on it (--grid writes prompt_grid.jpg).
The body prompt has a negative point on the handle and the handle prompt a negative point on the body. --preview runs
SAM 2 on frame 0 only and draws the three masks, so each first-frame mask is checked by eye before propagating.

SAM 2 (third_party/sam2, commit 2b90b9f; API read from sam2_video_predictor.py): build_sam2_video_predictor with the SAM 2.1
base_plus config "configs/sam2.1/sam2.1_hiera_b+.yaml" (Hydra name inside the sam2 package), bf16 autocast as in the
repo's own examples, offload_video_to_cpu=True and offload_state_to_cpu=True. The CUDA extension is not built
(SAM2_BUILD_CUDA=0), so SAM 2's small-hole filling is skipped (it would only warn); we set fill_hole_area = 0 explicitly.
Mask = logit > 0 at the JPEG resolution.

Storage: masks.npz holds np.packbits of every upright half-res mask (compressed). ``load_masks`` returns any mask in the
ORIGINAL (sideways) orientation at half, full or depth resolution (nearest neighbour at pixel centres). This loader does not
import torch, so the CPU stages can use it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from perception.common import RESULTS, WORK, stage_cli, stage_dir

BODY, HANDLE, COASTER = 1, 2, 3
NAMES = {BODY: "mug body", HANDLE: "mug handle", COASTER: "coaster"}
PROMPTS = Path("perception/prompts")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = Path(__file__).resolve().parents[1] / "third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
GREEN_HSV = ((35, 60, 25), (90, 255, 255))   # OpenCV HSV (H 0-180): the dark-green glazed mug
COLORS = {BODY: (0, 200, 0), HANDLE: (255, 0, 255), COASTER: (0, 160, 255)}   # BGR


# ---------------------------------------------------------------------------------------------------------- loader
class Masks:
    """Masks of one clip. get(obj, i, res) -> (H, W) bool in the ORIGINAL orientation; i = canonical index."""

    def __init__(self, clip: str):
        f = np.load(WORK / clip / "masks" / "masks.npz")
        self.bits, self.shape_up = f["bits"], tuple(int(v) for v in f["shape_upright"])
        self.obj_ids = [int(v) for v in f["obj_ids"]]
        self.k = int(np.load(WORK / clip / "frames" / "meta.npz")["rot_k"])
        self.n = len(self.bits) + 1                     # SAM 2 frame j = canonical i - 1; i = 0 has no image
        self.half = np.rot90(np.zeros(self.shape_up, bool), -self.k).shape   # ORIGINAL half-res (H, W) = (720, 960)

    def get(self, obj: int, i: int, res: str = "half") -> np.ndarray:
        if i == 0:
            m = np.zeros(self.half, bool)
        else:
            up = np.unpackbits(self.bits[i - 1, self.obj_ids.index(obj)], count=self.shape_up[0] * self.shape_up[1])
            m = np.rot90(up.reshape(self.shape_up).astype(bool), -self.k)
        return resample(m, res)

    def area(self, obj: int) -> np.ndarray:
        """Half-res pixel count per canonical index (0 at i = 0)."""
        j = self.obj_ids.index(obj)
        counts = [int(np.unpackbits(self.bits[t, j]).sum()) for t in range(len(self.bits))]
        return np.array([0] + counts, dtype=float)


def resample(m: np.ndarray, res: str) -> np.ndarray:
    """ORIGINAL half-res mask (720, 960) -> "half", "full" (1440, 1920) or "depth" (192, 256), nearest neighbour."""
    size = {"half": m.shape, "full": (2 * m.shape[0], 2 * m.shape[1]), "depth": (192, 256)}[res]
    rows = np.floor((np.arange(size[0]) + 0.5) * m.shape[0] / size[0]).astype(int)   # source pixel under each centre
    cols = np.floor((np.arange(size[1]) + 0.5) * m.shape[1] / size[1]).astype(int)
    return m[np.ix_(rows, cols)]


def load_masks(clip: str) -> Masks:
    return Masks(clip)


# ---------------------------------------------------------------------------------------------------------- prompts
def green_prompt(img_bgr: np.ndarray) -> dict:
    """Auto prompt for the mug from the HSV rule: bounding box of the largest green blob and its deepest interior point."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, *GREEN_HSV)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green)
    blob = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    ys, xs = np.nonzero(blob)
    return {"box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "interior_point": [int(x), int(y)],
            "blob_area_px": int(blob.sum()), "mask": blob}


def draw_grid(img: np.ndarray, step: int = 50) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (255, 255, 255) if x % 100 == 0 else (160, 160, 160), 1)
        if x % 100 == 0:
            cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (255, 255, 255) if y % 100 == 0 else (160, 160, 160), 1)
        if y % 100 == 0:
            cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def load_prompts(clip: str) -> dict:
    return json.loads((PROMPTS / f"{clip}.json").read_text())


def add_prompts(predictor, state, prompts: dict):
    logits = None
    for o in prompts["objects"]:
        if not o["points"] and not o["box"]:      # object not visible on the prompt frame: no prompt, its masks stay empty
            continue
        _, ids, logits = predictor.add_new_points_or_box(
            state, frame_idx=prompts["sam2_frame_j"], obj_id=o["obj_id"],
            points=np.array(o["points"], np.float32) if o["points"] else None,
            labels=np.array(o["labels"], np.int32) if o["points"] else None,
            box=np.array(o["box"], np.float32) if o["box"] else None)
    return list(ids), logits


def build_predictor():
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cuda")
    predictor.fill_hole_area = 0
    return predictor, torch


def draw_masks(img: np.ndarray, masks: dict, prompts: dict | None = None) -> np.ndarray:
    out = img.astype(float)
    for obj, m in masks.items():
        out[m] = 0.45 * out[m] + 0.55 * np.array(COLORS[obj])
    out = out.astype(np.uint8)
    for obj, m in masks.items():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COLORS[obj], 2)
    if prompts:
        for o in prompts["objects"]:
            c = COLORS[o["obj_id"]]
            if o["box"]:
                cv2.rectangle(out, tuple(o["box"][:2]), tuple(o["box"][2:]), c, 1)
            for (x, y), lab in zip(o["points"], o["labels"]):
                cv2.drawMarker(out, (int(x), int(y)), c, cv2.MARKER_STAR if lab else cv2.MARKER_TILTED_CROSS, 16, 2)
    return out


def tools(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args(argv)
    d = stage_dir(a.clip, "masks")
    img = cv2.imread(str(WORK / a.clip / "frames" / "sam2" / "00000.jpg"))
    if a.grid:
        g = green_prompt(img)
        vis = draw_grid(img)
        vis[g["mask"] > 0] = (0.5 * vis[g["mask"] > 0] + [0, 0, 127]).astype(np.uint8)
        cv2.rectangle(vis, tuple(g["box"][:2]), tuple(g["box"][2:]), (0, 0, 255), 1)
        cv2.drawMarker(vis, tuple(g["interior_point"]), (0, 0, 255), cv2.MARKER_STAR, 16, 2)
        cv2.imwrite(str(d / "prompt_grid.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(json.dumps({k: v for k, v in g.items() if k != "mask"}))
        print(f"wrote {d / 'prompt_grid.jpg'}")
    if a.preview:
        import tempfile
        prompts = load_prompts(a.clip)
        predictor, torch = build_predictor()
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            Path(tmp, "00000.jpg").symlink_to((WORK / a.clip / "frames" / "sam2" / "00000.jpg").resolve())
            state = predictor.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
            ids, logits = add_prompts(predictor, state, prompts)
            masks = {obj: (logits[n, 0] > 0).cpu().numpy() for n, obj in enumerate(ids)}
        cv2.imwrite(str(d / "prompt_preview.jpg"), draw_masks(img, masks, prompts), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(json.dumps({NAMES[o]: int(m.sum()) for o, m in masks.items()}))
        print(f"wrote {d / 'prompt_preview.jpg'}")


# ---------------------------------------------------------------------------------------------------------- stage
def run(clip: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = stage_dir(clip, "masks")
    prompts = load_prompts(clip)
    assert prompts["sam2_frame_j"] == 0 and prompts["frame_i"] == 1
    frames = WORK / clip / "frames" / "sam2"
    n_sam = len(list(frames.glob("*.jpg")))
    predictor, torch = build_predictor()
    obj_ids = [o["obj_id"] for o in prompts["objects"]]
    bits = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
        add_prompts(predictor, state, prompts)
        for j, ids, logits in predictor.propagate_in_video(state):
            m = (logits[:, 0] > 0.0).cpu().numpy()
            if bits is None:
                shape_up = m.shape[1:]
                bits = np.zeros((n_sam, 3, (shape_up[0] * shape_up[1] + 7) // 8), np.uint8)
            for n, obj in enumerate(list(ids)):
                bits[j, obj_ids.index(obj)] = np.packbits(m[n].reshape(-1))
    np.savez_compressed(out / "masks.npz", bits=bits, shape_upright=np.array(shape_up), obj_ids=np.array(obj_ids))
    del state, predictor
    torch.cuda.empty_cache()

    masks = load_masks(clip)
    t = np.load(WORK / clip / "frames" / "meta.npz")["t"]
    areas = {obj: masks.area(obj) for obj in (BODY, HANDLE, COASTER)}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for obj, a in areas.items():
        ax.plot(t[1:] - t[0], a[1:], label=NAMES[obj], color=np.array(COLORS[obj][::-1]) / 255)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mask area [half-res px]")
    ax.set_title(f"{clip}: SAM 2 mask areas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mask_areas.png", dpi=110)
    first = draw_masks(cv2.imread(str(frames / "00000.jpg")), {o: np.rot90(masks.get(o, 1), masks.k) for o in (BODY, HANDLE, COASTER)})
    cv2.imwrite(str(out / "first_frame_masks.jpg"), first, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"n_sam2_frames": n_sam, "obj_ids": obj_ids,
            **{f"area_px_{NAMES[o].replace(' ', '_')}_first": float(a[1]) for o, a in areas.items()},
            **{f"frames_empty_{NAMES[o].replace(' ', '_')}": int(np.sum(a[1:] == 0)) for o, a in areas.items()}}


if __name__ == "__main__":
    if "--grid" in sys.argv or "--preview" in sys.argv:
        tools(sys.argv[1:])
    else:
        stage_cli("masks", run, gpu=True)
