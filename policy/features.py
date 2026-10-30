"""M5 selector features and input noise -- numpy only, so the trainer (hoi-perception, numpy 2) and the evaluation (hoi-sim,
numpy 1) build exactly the same feature rows.

One row per (layout, candidate). Inputs: cup xy, coaster xy (sim world, metres), the handle yaw (rad), the clip index (CLIPS
order), the candidate heading phi and the clip's own heading phi_h (rad). Features (FEATURES, 14 columns): cup x, y; coaster x, y;
coaster - cup x, y; sin, cos of the handle yaw; one-hot clip (4); phi; phi_h - phi. Standardisation (training mean / std) is
part of the exported ONNX model, so these are the raw features.

Noise (spec): cup and coaster xy N(0, 1.0 cm) per axis, handle yaw N(0, 10 deg); coaster - cup is computed from the noisy
positions. ``noisy`` draws, in this order, cup (n, 2), coaster (n, 2), yaw (n,) from the generator it is given.
"""

from __future__ import annotations

import math

import numpy as np

CLIPS = ("26d4118b0c", "4e103843db", "49aff04f7a", "e8825dea53")
FEATURES = ("cup_x", "cup_y", "coaster_x", "coaster_y", "dx", "dy", "sin_yaw", "cos_yaw", *[f"clip_{c}" for c in CLIPS], "phi", "phi_h_minus_phi")
XY_NOISE_M = 0.01
YAW_NOISE_RAD = math.radians(10.0)


def features(cup_xy, coaster_xy, yaw, clip_idx, phi, phi_h) -> np.ndarray:
    """(n, 14) float64 raw feature rows (module docstring)."""
    cup_xy, coaster_xy = np.asarray(cup_xy, float).reshape(-1, 2), np.asarray(coaster_xy, float).reshape(-1, 2)
    yaw, phi, phi_h = (np.asarray(a, float).reshape(-1) for a in (yaw, phi, phi_h))
    onehot = np.eye(len(CLIPS))[np.asarray(clip_idx, int).reshape(-1)]
    return np.column_stack([cup_xy, coaster_xy, coaster_xy - cup_xy, np.sin(yaw), np.cos(yaw), onehot, phi, phi_h - phi])


def noisy(rng: np.random.Generator, cup_xy, coaster_xy, yaw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One noisy copy of n layouts' inputs (module docstring)."""
    cup_xy, coaster_xy, yaw = np.asarray(cup_xy, float).reshape(-1, 2), np.asarray(coaster_xy, float).reshape(-1, 2), np.asarray(yaw, float).reshape(-1)
    n = len(yaw)
    return (cup_xy + rng.normal(0.0, XY_NOISE_M, (n, 2)), coaster_xy + rng.normal(0.0, XY_NOISE_M, (n, 2)),
            yaw + rng.normal(0.0, YAW_NOISE_RAD, n))
