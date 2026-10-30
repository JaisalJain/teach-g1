"""S2 -- right hand per frame with WiLoR (3D MANO hand), converted to the ORIGINAL camera frame.

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m perception.hand_wilor --clip <id>

Per upright full-res frame (S1):
1. WiLoR's YOLO detector with WiLoR's demo threshold (conf 0.3); keep right-hand boxes (class 1). Pick the highest score;
   if another right box is within 0.1 of it, take the one with the larger IoU to the previous frame's chosen box.
2. WiLoR on that box (WiLoR's own crop: ViTDetDataset, rescale_factor 2.0, no flip for right hands).
3. Camera translation with the REAL upright intrinsics (cam_crop_to_full_K, below) instead of WiLoR's assumed focal length.
4. Everything is rotated back to the ORIGINAL (sideways) camera frame / pixels right away (geometry.R_up_from_orig etc.).

Keypoint order, verified in WiLoR's MANO wrapper (wilor/models/mano_wrapper.py: 16 MANO joints + 5 fingertip vertices
744/320/443/554/671, reordered by mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]):
OpenPose hand order -- wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20 (4, 8, 12, 16, 20 = tips).

WiLoR outputs (verified in wilor/models/wilor.py): pred_cam = [s, tx, ty] of a weak-perspective camera for the 256 px crop;
pred_mano_params as rotation matrices (global_orient in the crop camera's axes = the image's camera axes, the crop is not
rotated; hand_pose relative to the flat MANO hand); joints / vertices WITHOUT translation; 2D keypoints in normalised crop
coordinates (full-image pixels = box_centre + kp2d * box_size).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from perception import geometry as G
from perception.common import WORK, stage_cli, stage_dir
from perception.wilor_loader import DETECTOR_CONF, RESCALE_FACTOR, RIGHT_CLASS, load_detector, load_wilor

TIE_SCORE = 0.1


def cam_crop_to_full_K(pred_cam, box_center, box_size, K):
    """WiLoR's crop camera -> camera translation in the full image, with real intrinsics K.

    WiLoR trains a weak-perspective crop camera [s, tx, ty]: a hand of metric size X at depth tz appears in the 256 px crop
    with scale s, i.e. the crop is box_size pixels wide in the full image and the focal length f maps the 2-unit-wide
    normalised crop to s * box_size pixels: f / tz = s * box_size / 2  ->  tz = 2 f / (s * box_size).
    A point at the box centre (b_cx, b_cy) in the full image sits at crop-normalised (0, 0), and the crop offset (tx, ty)
    is in metres at depth tz. Projecting the hand centre with the real pinhole camera:
        b_cx = fx * (tx_full / tz) + cx  ->  tx_full = tx + (b_cx - cx) * tz / fx      (same for y with fy, cy)
    We use tz = 2 fx / (s * box_size) (fx and fy differ by < 0.5 % here; logged in the stage stats).
    WiLoR's cam_crop_to_full (wilor/utils/renderer.py:12-22) is the special case fx = fy = f and (cx, cy) = (W/2, H/2):
        tz = 2 f / (b s),  tx = 2 (b_cx - W/2) / (b s) + t_crop_x = t_crop_x + (b_cx - W/2) tz / f  (identical).
    """
    s, tx, ty = pred_cam
    tz = 2.0 * K[0, 0] / (s * box_size)
    return np.array([tx + (box_center[0] - K[0, 2]) * tz / K[0, 0], ty + (box_center[1] - K[1, 2]) * tz / K[1, 1], tz])


def box_iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def choose_box(right: np.ndarray, prev_box) -> tuple[np.ndarray, bool]:
    """right: (M, 6) right-hand detections [x1 y1 x2 y2 conf cls]. Returns (box row, whether the IoU tie-break decided)."""
    best = right[np.argmax(right[:, 4])]
    close = right[right[:, 4] >= best[4] - TIE_SCORE]
    if len(close) < 2 or prev_box is None:
        return best, False
    return close[np.argmax([box_iou(r[:4], prev_box) for r in close])], True


@torch.no_grad()
def run(clip: str) -> dict:
    out = stage_dir(clip, "hand")
    meta = np.load(WORK / clip / "frames" / "meta.npz")
    K_all, k, size = meta["K"], int(meta["rot_k"]), tuple(int(v) for v in meta["rgb_size"])
    n = len(meta["t"])
    R_up = G.R_up_from_orig(k)

    detector = load_detector("cuda")
    model, cfg, faces = load_wilor("cuda")
    from wilor.datasets.vitdet_dataset import ViTDetDataset     # importable once load_wilor put WiLoR on sys.path
    from wilor.utils.renderer import cam_crop_to_full

    nan = lambda *shape: np.full((n, *shape), np.nan, np.float32)
    res = dict(hand_valid=np.zeros(n, bool), score=nan(), n_right=np.zeros(n, np.int32), box_orig=nan(4),
               global_orient=nan(3, 3), hand_pose_rotmat=nan(15, 3, 3), hand_pose_aa=nan(45), betas=nan(10), cam_t=nan(3),
               kp3d=nan(21, 3), kp2d=nan(21, 2), verts=np.full((n, 778, 3), np.nan, np.float16))
    prev_box, n_tiebreak, wilor_check, seconds = None, 0, [], []
    for i in range(1, n):
        t0 = time.perf_counter()
        bgr = cv2.imread(str(WORK / clip / "frames" / "full" / f"{i:06d}.jpg"))
        det = detector(bgr, conf=DETECTOR_CONF, verbose=False)[0].boxes.data.cpu().numpy()
        right = det[det[:, 5] == RIGHT_CLASS]
        res["n_right"][i] = len(right)
        if len(right) == 0:
            prev_box = None
            continue
        best, tiebreak = choose_box(right, prev_box)
        n_tiebreak += tiebreak
        prev_box = best[:4]

        item = ViTDetDataset(cfg, bgr, best[None, :4], np.array([1.0]), rescale_factor=RESCALE_FACTOR)[0]
        o = model({key: v.cuda() for key, v in default_collate([item]).items()})
        pred_cam = o["pred_cam"][0].cpu().numpy().astype(np.float64)
        center, bsize = item["box_center"].astype(np.float64), float(item["box_size"])

        K_up = G.upright_K(K_all[i], size, k)
        cam_t_up = cam_crop_to_full_K(pred_cam, center, bsize, K_up)
        # check: with WiLoR's assumed camera our formula must give exactly WiLoR's cam_crop_to_full
        h, w = bgr.shape[:2]
        f = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(w, h)
        ours = cam_crop_to_full_K(pred_cam, center, bsize, np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]))
        theirs = cam_crop_to_full(o["pred_cam"].double(), torch.tensor(center[None]).cuda(), torch.tensor([bsize]).cuda(),
                                  torch.tensor([[w, h]], dtype=torch.float64).cuda(), f)[0].cpu().numpy()
        wilor_check.append(float(np.abs(ours - theirs).max() / np.abs(theirs).max()))

        mano = o["pred_mano_params"]
        go_up = mano["global_orient"][0, 0].cpu().numpy().astype(np.float64)
        joints = o["pred_keypoints_3d"][0].cpu().numpy().astype(np.float64)
        verts = o["pred_vertices"][0].cpu().numpy().astype(np.float64)
        kp2d_up = center + o["pred_keypoints_2d"][0].cpu().numpy() * bsize
        box_corners = G.pix_upright_to_orig(best[:4].reshape(2, 2), size, k)

        # upright camera -> ORIGINAL camera: x_orig = R_up^T x_up (row vectors: x @ R_up)
        res["hand_valid"][i] = True
        res["score"][i] = best[4]
        res["box_orig"][i] = np.concatenate([box_corners.min(0), box_corners.max(0)])
        res["global_orient"][i] = R_up.T @ go_up
        res["hand_pose_rotmat"][i] = mano["hand_pose"][0].cpu().numpy()
        res["hand_pose_aa"][i] = G.rotmat_to_axis_angle(res["hand_pose_rotmat"][i].astype(np.float64)).reshape(45)
        res["betas"][i] = mano["betas"][0].cpu().numpy()
        res["cam_t"][i] = R_up.T @ cam_t_up
        res["kp3d"][i] = (joints + cam_t_up) @ R_up
        res["verts"][i] = ((verts + cam_t_up) @ R_up).astype(np.float16)
        res["kp2d"][i] = G.pix_upright_to_orig(kp2d_up, size, k)
        seconds.append(time.perf_counter() - t0)

    np.savez_compressed(out / "hand.npz", faces=faces, **res)
    K_up_all = G.upright_K(K_all, size, k)
    fx, fy = K_up_all[:, 0, 0], K_up_all[:, 1, 1]
    valid = res["hand_valid"]
    return {"n_frames": n, "n_hand_valid": int(valid.sum()), "first_hand_valid": int(np.argmax(valid)) if valid.any() else None,
            "last_hand_valid": int(n - 1 - np.argmax(valid[::-1])) if valid.any() else None,
            "n_frames_multiple_right_boxes": int(np.sum(res["n_right"] > 1)), "n_iou_tiebreaks": n_tiebreak,
            "score_median": float(np.nanmedian(res["score"])) if valid.any() else None,
            "upright_fx_median": float(np.median(fx)), "upright_fy_median": float(np.median(fy)),
            "fx_fy_rel_diff_max": float(np.max(np.abs(fx - fy) / fy)),
            "cam_crop_to_full_K_vs_wilor_max_rel_diff": max(wilor_check) if wilor_check else None,
            "seconds_per_frame_median": float(np.median(seconds)) if seconds else None,
            "keypoint_order": "OpenPose: wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20"}


if __name__ == "__main__":
    stage_cli("hand", run, gpu=True)
