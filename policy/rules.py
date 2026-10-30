"""M5 selection rules -- numpy only (used by the cross-validation in hoi-perception and the evaluation in hoi-sim).

For one layout, the 18 candidates (clip, heading phi) have predicted success p (the selector) and a departure from my heading,
phi_h - phi >= 0 (0 for the "phih" candidates).
  R1  the candidate with the highest p.
  R2  (human-first) among the candidates with p >= R2_MIN_P, the one with the smallest departure; ties -> higher p; none -> R1.
  F   the fixed-best candidate: the highest training success rate; ties -> the larger phi (closer to the human); a remaining tie
      (only possible between clips) -> the earlier candidate in the table order.
"""

from __future__ import annotations

import numpy as np

R2_MIN_P = 0.8          # spec


def r1(p) -> int:
    return int(np.argmax(np.asarray(p, float)))           # first maximum on an exact tie


def r2(p, departure) -> int:
    p, departure = np.asarray(p, float), np.asarray(departure, float)
    ok = np.flatnonzero(p >= R2_MIN_P)
    if not len(ok):
        return r1(p)
    return int(min(ok, key=lambda i: (departure[i], -p[i], i)))


def fixed_best(success_rate, phi) -> int:
    success_rate, phi = np.asarray(success_rate, float), np.asarray(phi, float)
    return int(min(range(len(phi)), key=lambda i: (-success_rate[i], -phi[i], i)))
