"""Phase 0.5 smoke test: WiLoR's hand detector + WiLoR on 3 frames of the 90 deg clip, loaded via perception.wilor_loader.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.wilor_smoke

Needs S1 (perception.frames) for 4e103843db. Writes results/m2_perception/smoke/overlay_<i>.jpg (detector box, WiLoR's 2D
keypoints, and the mesh projected with WiLoR's OWN demo camera: assumed focal 5000/256 x max(W, H), principal point at the
image centre -- i.e. exactly what WiLoR's demo would draw) and smoke.json (time per frame, peak VRAM, peak RSS).
"""

from __future__ import annotations

import json
import resource
import time

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from perception.common import RESULTS, WORK
from perception.wilor_loader import DETECTOR_CONF, RESCALE_FACTOR, RIGHT_CLASS, load_detector, load_wilor

CLIP = "4e103843db"
FRAMES = (150, 250, 350)   # hand in view (reach / transport / place), from the M2.1 contact sheet
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


@torch.no_grad()
def main():
    out = RESULTS / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    detector = load_detector("cuda")
    model, cfg, faces = load_wilor("cuda")
    load_s = time.perf_counter() - t0
    from wilor.datasets.vitdet_dataset import ViTDetDataset     # importable once load_wilor put WiLoR on sys.path
    from wilor.utils.renderer import cam_crop_to_full
    rows = []
    for i in FRAMES:
        bgr = cv2.imread(str(WORK / CLIP / "frames" / "full" / f"{i:06d}.jpg"))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        det = detector(bgr, conf=DETECTOR_CONF, verbose=False)[0].boxes.data.cpu().numpy()   # x1 y1 x2 y2 conf cls
        right = det[det[:, 5] == RIGHT_CLASS]
        row = {"i": i, "n_detections": len(det), "n_right": len(right)}
        if len(right):
            best = right[np.argmax(right[:, 4])]
            ds = ViTDetDataset(cfg, bgr, best[None, :4], np.array([1.0]), rescale_factor=RESCALE_FACTOR)
            batch = {k: v.cuda() for k, v in default_collate([ds[0]]).items()}
            o = model(batch)
            torch.cuda.synchronize()
            row["seconds"] = round(time.perf_counter() - t1, 3)
            h, w = bgr.shape[:2]
            img_size = torch.tensor([[w, h]], dtype=torch.float32, device="cuda")
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(w, h)
            cam_t = cam_crop_to_full(o["pred_cam"], batch["box_center"].float(), batch["box_size"].float(), img_size, focal)[0]
            verts = (o["pred_vertices"][0] + cam_t).cpu().numpy()
            uv_mesh = verts[:, :2] / verts[:, 2:] * focal + [w / 2, h / 2]
            kp2d = (batch["box_center"].float() + o["pred_keypoints_2d"][0] * batch["box_size"].float()).cpu().numpy()
            row.update(score=round(float(best[4]), 3), box=best[:4].round(1).tolist(), demo_focal_px=focal,
                       demo_cam_t=cam_t.cpu().numpy().round(4).tolist())
            img = bgr.copy()
            for u, v in uv_mesh.astype(int):
                cv2.circle(img, (u, v), 2, (255, 180, 0), -1)
            for a, b in BONES:
                cv2.line(img, tuple(kp2d[a].astype(int)), tuple(kp2d[b].astype(int)), (0, 0, 255), 3)
            x1, y1, x2, y2 = best[:4].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"i={i} right {best[4]:.2f}  blue: mesh (WiLoR demo camera)  red: WiLoR 2D keypoints",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(out / f"overlay_{i}.jpg"), cv2.resize(img, (720, 960), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(row)
        print(row)
    summary = {"clip": CLIP, "frames": rows, "model_load_s": round(load_s, 1),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
               "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
               "note": "first frame includes CUDA warm-up"}
    (out / "smoke.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}))


if __name__ == "__main__":
    main()
