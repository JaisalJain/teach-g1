"""Checks of the upright <-> original maps in perception/geometry.py. Plain script:

    env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python tests/test_geometry.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception import geometry as G  # noqa: E402

SIZE = (1920, 1440)                                       # original (W, H), as stored by the app
K = np.array([[1421.3, 0, 958.7], [0, 1417.9, 721.4], [0, 0, 1]])   # fx != fy and an off-centre principal point on purpose
rng = np.random.default_rng(0)

for k in range(4):
    # 1000 random points in front of the camera: project, then map the pixel to the upright image ...
    X = np.column_stack([rng.uniform(-0.4, 0.4, 1000), rng.uniform(-0.3, 0.3, 1000), rng.uniform(0.2, 1.5, 1000)])
    via_pixels = G.pix_orig_to_upright(G.project(K, X), SIZE, k)
    # ... must equal rotating the points into the upright camera frame and projecting with the upright intrinsics
    via_camera = G.project(G.upright_K(K, SIZE, k), X @ G.R_up_from_orig(k).T)
    err = np.abs(via_pixels - via_camera).max()
    assert err < 1e-6, (k, err)

    # pixel round trips are the identity
    uv = rng.uniform([0, 0], SIZE, size=(1000, 2))
    up = G.pix_orig_to_upright(uv, SIZE, k)
    assert np.abs(G.pix_upright_to_orig(up, SIZE, k) - uv).max() < 1e-9, k
    up = rng.uniform([0, 0], G.rot90_size(SIZE, k), size=(1000, 2))
    assert np.abs(G.pix_orig_to_upright(G.pix_upright_to_orig(up, SIZE, k), SIZE, k) - up).max() < 1e-9, k

    # the pixel map matches what np.rot90 really does to an image (a few marked pixels)
    img = np.zeros(SIZE[::-1], dtype=np.int32)
    marks = rng.integers([0, 0], SIZE, size=(20, 2))
    img[marks[:, 1], marks[:, 0]] = np.arange(1, 21)
    rot = np.rot90(img, k)
    assert rot.shape[::-1] == G.rot90_size(SIZE, k)
    up = np.round(G.pix_orig_to_upright(marks, SIZE, k)).astype(int)
    assert np.array_equal(rot[up[:, 1], up[:, 0]], np.arange(1, 21)), k

    # R_up_from_orig is a proper rotation
    R = G.R_up_from_orig(k)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    print(f"k={k}: max |pixel map - camera rotation| = {err:.2e} px, round trips OK, matches np.rot90")

print("all geometry checks passed")
