"""Spec A2 unit test: b_level with phi_h = phi_ora and h_frac_h = h_frac_ora must reproduce the oracle's grasp to 1e-9. Plain script:

    env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python tests/test_retarget_grasp.py

The oracle grasp is taken from the oracle's own code (``Grasp.hero_side_grasp(...).palm_for_cup`` and HERO's
``palm_rotation(WRIST_YAW, 0, 0)``, before any quaternion round trip), for the mug's dimensions from config/objects.yaml
(r = 0.0495 m: the compiled cup_geom radius) at a few cup positions. Also checks the frame mapping M (spec A1).
"""

import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_tools import reach_specs as rs  # noqa: E402
from sim.pick_place import WRIST_YAW, Grasp, radius_consistent_offset  # noqa: E402
from sim.scene import mug_config  # noqa: E402

from retarget.frames import M, TABLE_TOP, heading_human, heading_robot, robot_rot  # noqa: E402
from retarget.grasp import grasp_pose, oracle_reference  # noqa: E402

TOL = 1e-9
t0 = time.perf_counter()

# M: det 1, robot x = x_f, robot y = n, robot z = x_f x n, for a random human palm rotation
assert np.linalg.det(M) == 1.0
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
R_h = Q * np.sign(np.linalg.det(Q))
x_f, n = R_h[:, 0], R_h[:, 2]
R = robot_rot(R_h)
assert np.allclose(R[:, 0], x_f, atol=1e-15) and np.allclose(R[:, 1], n, atol=1e-15) and np.allclose(R[:, 2], np.cross(x_f, n), atol=1e-15)
assert abs(heading_robot(R) - heading_human(R_h)) < TOL

mug = mug_config()
r, h = mug["body_outer_diameter_m"] / 2, mug["height_m"]
o = radius_consistent_offset(r)
R_oracle = rs.palm_rotation(WRIST_YAW, 0.0, 0.0)          # what Grasp.hero_side_grasp builds its quaternion from
worst_p = worst_R = 0.0
for cup_xy in [(0.4933, -0.2356), (0.45, -0.30), (0.52, -0.10), (0.0, 0.0)]:
    c = np.array([cup_xy[0], cup_xy[1], TABLE_TOP + h / 2])     # mug body centre, resting on the table
    oracle = oracle_reference(dict(r=r, h=h), cup_centre=c)
    human = dict(phi_h=oracle["phi_ora"], h_frac_h=oracle["h_frac_ora"], R_hg=np.full((3, 3), np.nan))   # R_hg unused by b_level
    p, R_G, q = grasp_pose("b_level", cup_xy, human, oracle)
    p_oracle = Grasp.hero_side_grasp(h, o).palm_for_cup(c)
    worst_p = max(worst_p, float(np.abs(p - p_oracle).max()))
    worst_R = max(worst_R, float(np.abs(R_G - R_oracle).max()))
    assert np.abs(p - p_oracle).max() < TOL, (cup_xy, p, p_oracle)
    assert np.abs(R_G - R_oracle).max() < TOL, (cup_xy, R_G, R_oracle)
    assert np.abs(q - (p_oracle + R_oracle @ o)).max() < TOL      # the axis point opposite the palm is the oracle's too

print(f"test_retarget_grasp: OK -- b_level at phi_ora / h_frac_ora = oracle grasp: max |p - p_ora| {worst_p:.2e} m, max |R - R_ora| "
      f"{worst_R:.2e} (tolerance {TOL:g}); M checks pass; phi_ora {math.degrees(oracle['phi_ora']):.4f} deg, h_frac_ora "
      f"{oracle['h_frac_ora']:.6f}; wall {time.perf_counter() - t0:.2f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
