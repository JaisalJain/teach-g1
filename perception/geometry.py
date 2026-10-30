"""Geometry helpers: the upright <-> original image maps, intrinsics, projection, rigid transforms, rotations.

Upright view. The phone was held in portrait, so the stored frames are sideways. The models (WiLoR, SAM 2) run on
``np.rot90(image, k)`` (k from ``stray.upright_rot90``). Everything they return is mapped straight back to the ORIGINAL
image / camera frame with the functions below; nothing downstream uses upright coordinates.

np.rot90(img, 1) is img[:, ::-1].T: the output pixel (u', v') comes from the input pixel (u, v) = (W-1-v', u'), so
    u' = v,  v' = W-1-u          (W = input width; the output is H wide and W tall)
With u = fx X/Z + cx and v = fy Y/Z + cy this is a pinhole camera again with X' = Y, Y' = -X, Z' = Z and
    fx' = fy, fy' = fx, cx' = cy, cy' = W-1-cx.
k > 1 applies the same step k times. Pixel coordinates are pixel centres (OpenCV convention).
"""

from __future__ import annotations

import numpy as np

_R1 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # camera rotation of one np.rot90 step


def rot90_size(size_wh, k: int) -> tuple[int, int]:
    w, h = size_wh
    return (w, h) if k % 2 == 0 else (h, w)


def R_up_from_orig(k: int) -> np.ndarray:
    """3x3 rotation taking ORIGINAL camera-frame vectors to the upright camera frame of np.rot90(image, k)."""
    return np.linalg.matrix_power(_R1, k % 4)


def upright_K(K: np.ndarray, size_wh, k: int) -> np.ndarray:
    """Intrinsics of np.rot90(image, k) for an image of size (W, H) with intrinsics K (works on (..., 3, 3))."""
    K = np.array(K, dtype=np.float64)
    w, h = size_wh
    for _ in range(k % 4):
        fx, fy, cx, cy = K[..., 0, 0].copy(), K[..., 1, 1].copy(), K[..., 0, 2].copy(), K[..., 1, 2].copy()
        K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2] = fy, fx, cy, w - 1 - cx
        w, h = h, w
    return K


def pix_orig_to_upright(uv, size_wh, k: int) -> np.ndarray:
    """(..., 2) pixel coordinates of an image of size (W, H) -> the same points in np.rot90(image, k)."""
    uv = np.array(uv, dtype=np.float64)
    w, h = size_wh
    for _ in range(k % 4):
        uv = np.stack([uv[..., 1], w - 1 - uv[..., 0]], -1)
        w, h = h, w
    return uv


def pix_upright_to_orig(uv, size_wh_orig, k: int) -> np.ndarray:
    """Inverse of pix_orig_to_upright: (..., 2) upright pixels -> ORIGINAL pixels (size_wh_orig = original (W, H))."""
    return pix_orig_to_upright(uv, rot90_size(size_wh_orig, k), (4 - k % 4) % 4)


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    """Intrinsics for the same view resampled by s (pixel centres: u' + 0.5 = s (u + 0.5)). Works on (..., 3, 3)."""
    K = np.array(K, dtype=np.float64)
    K[..., 0, 0] *= s
    K[..., 1, 1] *= s
    K[..., :2, 2] = (K[..., :2, 2] + 0.5) * s - 0.5
    return K


def project(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(..., 3) camera-frame points -> (..., 2) pixels."""
    X = np.asarray(X, dtype=np.float64)
    return np.stack([K[0, 0] * X[..., 0] / X[..., 2] + K[0, 2], K[1, 1] * X[..., 1] / X[..., 2] + K[1, 2]], -1)


def backproject_pixels(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(..., 2) pixels with depth z (along the optical axis) -> (..., 3) camera-frame points."""
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0] * z, (uv[..., 1] - K[1, 2]) / K[1, 1] * z, z], -1)


def pixel_rays(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(..., 2) pixels -> (..., 3) camera-frame ray directions with z = 1."""
    return backproject_pixels(uv, np.ones(np.shape(uv)[:-1]), K)


def transform(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (..., 3) points."""
    return np.asarray(X) @ T[:3, :3].T + T[:3, 3]


def invert(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane directions (e1, e2) with e1 x e2 = n."""
    n = np.asarray(n, dtype=np.float64) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def ray_plane(origins: np.ndarray, dirs: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Intersect rays origin + s dir with the plane n.p + d = 0. (..., 3) -> (..., 3)."""
    origins, dirs = np.asarray(origins, dtype=np.float64), np.asarray(dirs, dtype=np.float64)
    s = -(origins @ n + d) / (dirs @ n)
    return origins + s[..., None] * dirs


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-12)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(x)
    Kx = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)
    th = theta[..., None]
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle, angle in [0, pi]. Stable near 0 and near pi."""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    theta = np.arccos(cos)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sin = np.sin(theta)
    out = np.where((sin > 1e-6)[..., None], w * (theta / np.maximum(2 * sin, 1e-12))[..., None], 0.5 * w)
    # near pi the antisymmetric part vanishes: take the axis from the symmetric part (R + I) / 2 = k k^T
    near_pi = theta > np.pi - 1e-3
    if np.any(near_pi):
        S = (R[near_pi] + np.eye(3)) / 2
        col = np.argmax(np.diagonal(S, axis1=-2, axis2=-1), -1)
        k = S[np.arange(len(S)), :, col]
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        k *= np.where(np.sum(k * w[near_pi], -1) < 0, -1.0, 1.0)[..., None]   # keep the sign the residual w suggests
        out[near_pi] = k * theta[near_pi][..., None]
    return out


def orthonormalize(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (SVD), det = +1. Works on (..., 3, 3)."""
    U, _, Vt = np.linalg.svd(M)
    D = np.ones(M.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def slerp_rotmat(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Rotation a fraction s of the way from R0 to R1 (shortest path)."""
    return R0 @ axis_angle_to_rotmat(s * rotmat_to_axis_angle(R0.T @ R1))
