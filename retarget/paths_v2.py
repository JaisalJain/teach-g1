"""M4 path timing v2: M3's paths with the pause removed and every human-derived segment re-timed to start and end at rest;
plus the oracle's own nominal path, built the way M3 builds a path (M4 Step 0).

Why v2. M3 keeps my hand's timing (my times x k), so a human segment starts and ends at whatever speed my hand had there:
the approach starts at my hand-entry speed right after the prelude has stopped, and the transport starts at my lift speed
right after the close hold (M3 open question 6). HERO tracks the reference clip built from the path, so such a jump is a jump
in the reference. M4 plays the path in physics and settles at the pause pose and at G, so in v2 every join is a rest-to-rest
join. The geometry, the warps and k are M3's exactly: v2 calls M3's own builders (warp.object_aware_path, naive.naive_path,
unchanged) and then only cuts and re-times their output. Nothing in it is new geometry.

(b) b_level / b_tilt, frame by frame:
  prelude      M3's, unchanged (min-jerk, ends at rest at p_r0). i_prelude_end = its last frame.
  approach 1   M3's approach samples 0 .. j_pause (p_r0 -> the pause pose, 8 cm of arc before G), re-timed. M3's 0.5 s pause
               hold is removed; i_pause is now the split point = approach 1's last frame = approach 2's first. M4 settles here.
  approach 2   M3's approach samples j_pause .. end (the pause pose -> G), re-timed. Ends at i_G.
  close        M3's, unchanged (88 frames at G, closure 0 -> 1 over 1.25 s). i_close_end = its last frame (still at G).
  transport 1  M3's transport samples 0 .. j_peak (G -> the highest palm sample of the transport), re-timed. Ends at i_peak.
  transport 2  M3's transport samples j_peak .. end (the peak -> P), re-timed. Ends at i_P (= i_release, as in M3).
  release, retreat  M3's, unchanged.
(a) naive: prelude and to_a0 are M3's (both min-jerk). The replay (my a0 .. t_release) is the one human-derived segment and is
  re-timed as a whole (M3's naive has no pause and no separate transport phase to split at). The release frames are M3's.
  The closure keeps M3's rule (0 -> 1 over 1.25 s from the contact frame, 1 -> 0 over 0.6 s from i_release), with the contact,
  grasp and place frames mapped onto the new timing (the re-timed frame where the path is where M3 had it at that frame).

Re-timing one segment (spec Step 1). x(tau), tau in [0, D0], is the segment as M3 timed it: linear interpolation of its
50 Hz samples, sample j at tau = j / 50. New duration D = D0 + T_R, T_R = 0.4 s. Speed factor v(t) = smoothstep(t / T_R) on
[0, T_R], 1 in the middle, smoothstep((D - t) / T_R) on [D - T_R, D], with smoothstep(x) = 3x^2 - 2x^3; sigma(t) = int_0^t v.
Each ramp covers T_R / 2 of M3 time in T_R of new time, so sigma(D) = D0 exactly: the whole segment is played, 0.4 s slower.
The new samples are x(sigma(j / 50)), j = 0 .. round(50 D); the last one is set exactly to x(D0). The middle keeps M3's
speed; only the first and the last 0.4 s change, from a jump to a smooth start and stop (v(0) = v'(0) = 0 at both ends).
Orientations follow the same sigma (slerp between neighbouring samples); in (b) they are constant (R_G) and stay exact.
Asserts per segment: the first and the last 50 Hz step each move < 1 mm; the segment's end positions equal M3's to 1e-9.
A segment shorter than T_R cannot be re-timed this way (the two ramps would overlap) and raises; none of M3's runs has one
(shortest: 0.48 s, 4e103843db transport 1).

Oracle path (spec Step 0, ``oracle_path``): the oracle's nominal waypoints through min_jerk_path, every leg rest to rest,
with M3's phase names: prelude to (p_s.x, p_s.y, z_clear) 1.2 s; "approach" = pre-grasp 1.8 s + to G 2.0 s; "close" 1.75 s
(closure ramp 1.25 s); "transport" = lift +0.10 m 1.5 s + to above_p over max(1.5, d / 0.08) s + lower 1.5 s; "release"
0.6 + 0.6 s; "retreat" 1.5 s + 1.0 s hold. Nominal = the cup at rest at the layout and the nominal in-hand offset
(cup - palm at G), exactly as warp.time_scale computes the oracle's legs; the real oracle measures both in the sim.

M4b hybrid paths (``hybrid_path``: b_hybrid, b_hybrid_h45): my approach warped to the pre-grasp point PG = G - 8 cm u instead
of G and re-timed as one segment, then the oracle's straight PG -> G leg (played by Execute with the oracle's move), then the
close, transport, release and retreat built as M3 / v2 build them. b_hybrid has b_level's grasp, b_hybrid_h45 the oracle's
heading at my grasp height. See hybrid_path's docstring.

Saved npz (``save_path``): M3's fields (palm_pos, palm_quat_wxyz, closure, phase, i_pause, i_G, i_P, i_release) plus the
split indices i_prelude_end, i_close_end, i_peak (-1 where a variant has none: naive has no pause, close hold or peak).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from data_tools import hero_reach_generator as gen
from sim2sim.bench.replan import slerp_wxyz
from sim.pick_place import LIFT_HEIGHT, PLACE_CLEARANCE, PREGRASP_BACKOFF, RETREAT_BACK, RETREAT_UP, TRANSPORT_SPEED
from sim.reference_builder import min_jerk_path
from sim.scene import COASTER_THICKNESS, mug_config

from retarget.frames import TABLE_TOP, base_to_centre, human_to_sim_offset, load_demo, table_to_world
from retarget.grasp import grasp_pose, human_at_grasp, oracle_reference
from retarget.naive import naive_path
from retarget.warp import (APPROACH_MIN_Z, CLOSE_RAMP_S, CLOSE_S, FPS, ORACLE_PREGRASP_S, ORACLE_TRANSPORT_MIN_S, PRELUDE_S,
                           RELEASE_HOLD_S, RELEASE_S, RETREAT_HOLD_S, RETREAT_S, PathBuilder, approach_start, event_frames, intervals,
                           min_jerk_w, n_frames, object_aware_path, prelude, quat_of, recorded_layout, resample, time_scale)

T_R = 0.4                       # s: length of the start ramp and of the stop ramp of a re-timed segment (spec)
STEP_LIMIT_M = 0.001            # assert: the first and the last 50 Hz step of a re-timed segment move < 1 mm
END_TOL_M = 1e-9                # assert: a re-timed segment's end positions equal M3's
ORACLE_APPROACH_S = 2.0         # the oracle's pre-grasp -> grasp leg
ORACLE_LIFT_S = ORACLE_LOWER_S = 1.5
B_VARIANTS = ("b_level", "b_tilt")
NAIVE_VARIANTS = ("naive", "a_naive")      # M4 calls it "naive", M3 "a_naive"
SAVE_FIELDS = ("palm_pos", "palm_quat_wxyz", "closure", "phase", "i_pause", "i_G", "i_P", "i_release", "i_prelude_end",
               "i_close_end", "i_peak")


# ---------------------------------------------------------------------------------------------------------------- re-timing
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 3 * x ** 2 - 2 * x ** 3


def _smoothstep_integral(x):
    """int_0^x smoothstep(u) du = x^3 - x^4 / 2 on [0, 1] (= 1/2 at x = 1)."""
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 - x ** 4 / 2


def sigma(t, D0: float):
    """M3 time reached at new time ``t`` (array) for a segment of M3 duration D0 re-timed to D = D0 + T_R: the integral of
    the speed factor v (module docstring), in closed form. Start ramp T_R * I(t / T_R); middle t - T_R / 2 (v = 1, the start
    ramp lost T_R / 2); stop ramp D0 - T_R * I((D - t) / T_R). The pieces meet at t = T_R and t = D - T_R; sigma(D) = D0."""
    if D0 < T_R:
        raise ValueError(f"segment of {D0:.3f} s is shorter than T_R = {T_R} s: its start and stop ramps would overlap")
    t = np.asarray(t, dtype=float)
    D = D0 + T_R
    up = T_R * _smoothstep_integral(t / T_R)
    down = D0 - T_R * _smoothstep_integral((D - t) / T_R)
    return np.where(t < T_R, up, np.where(t > D - T_R, down, t - T_R / 2))


def sigma_check(D0: float = 1.0, n: int = 200_001) -> dict:
    """Self-check of the closed form: sigma vs a fine trapezoid integral of v itself (as the spec defines it)."""
    D = D0 + T_R
    t = np.linspace(0.0, D, n)
    v = np.where(t < T_R, smoothstep(t / T_R), np.where(t > D - T_R, smoothstep((D - t) / T_R), 1.0))
    numeric = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])
    err = float(np.abs(numeric - sigma(t, D0)).max())
    return dict(D0_s=D0, samples=n, max_abs_diff_s=err, sigma_D_minus_D0_s=float(sigma(np.array([D]), D0)[0] - D0), passed=err < 1e-9)


def retime(pos: np.ndarray, quat: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Re-time one segment (module docstring). ``pos`` (n + 1, 3) / ``quat`` (n + 1, 4) = its samples as M3 timed them (50 Hz,
    sample 0 = its start). Returns the new samples (n + 21 of them: D = n / 50 + 0.4 s) and the segment's numbers."""
    pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
    n = len(pos) - 1
    D0 = n / FPS
    N = int(round(FPS * (D0 + T_R)))
    t = np.arange(N + 1) / FPS
    u = np.clip(sigma(t, D0) * FPS, 0.0, n)                 # M3 sample index (fractional) at each new frame
    idx = np.arange(n + 1)
    new_pos = np.column_stack([np.interp(u, idx, pos[:, c]) for c in range(3)])
    new_pos[-1] = pos[-1]                                    # spec: the last sample is exactly x(D0)
    if (quat == quat[0]).all():                              # (b): R_G held, keep it bit-exact
        new_quat = np.repeat(quat[:1], N + 1, axis=0)
    else:
        i = np.minimum(np.floor(u).astype(int), n - 1)
        new_quat = np.array([slerp_wxyz(quat[k], quat[k + 1], float(w)) for k, w in zip(i, u - i)])
        new_quat[-1] = quat[-1]

    step = np.linalg.norm(np.diff(new_pos, axis=0), axis=1)
    end_err = max(float(np.abs(new_pos[0] - pos[0]).max()), float(np.abs(new_pos[-1] - pos[-1]).max()))
    assert step[0] < STEP_LIMIT_M and step[-1] < STEP_LIMIT_M, f"{name}: first / last step {step[0] * 1e3:.3f} / {step[-1] * 1e3:.3f} mm"
    assert end_err <= END_TOL_M, f"{name}: end positions differ from M3's by {end_err} m"
    m3_speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS
    info = dict(name=name, m3_intervals=n, D0_s=D0, D_s=N / FPS, frames=N + 1, first_step_mm=float(step[0] * 1e3),
                last_step_mm=float(step[-1] * 1e3), end_error_m=end_err,
                m3_first_last_speed_m_s=[float(m3_speed[0]), float(m3_speed[-1])],      # the jumps v2 removes (information)
                v2_peak_speed_m_s=float(step.max() * FPS), m3_peak_speed_m_s=float(m3_speed.max()))
    return new_pos, new_quat, info


# ---------------------------------------------------------------------------------------------------------------- (b) v2
def _last(phase: np.ndarray, name: str) -> int:
    hits = np.flatnonzero(phase == name)
    if not len(hits):
        raise ValueError(f"no '{name}' frame in the path")
    return int(hits[-1])


def v2_from_m3_b(m3: dict) -> dict:
    """v2 timing of an M3 (b) path (the dict object_aware_path returns): cut out the pause, split, re-time (module docstring)."""
    pos, quat, closure, phase = (np.asarray(m3[k]) for k in ("palm_pos", "palm_quat_wxyz", "closure", "phase"))
    pos, quat, closure = pos.astype(float), quat.astype(float), closure.astype(float)
    fr = m3["info"]["frames"]
    i_pre = _last(phase, "prelude")
    # M3's approach samples 0..n_a: sample 0 = p_r0 = the prelude's last frame, then the "approach" frames ("pause" frames,
    # the 0.5 s hold, are the ones dropped). j_pause = the sample M3's i_pause points at.
    app_frames = np.concatenate([[i_pre], np.flatnonzero(phase == "approach")])
    assert len(app_frames) == fr["approach_intervals"] + 1, "approach samples do not match M3's interval count"
    pause_frames = np.flatnonzero(phase == "pause")
    assert len(pause_frames) == fr["pause"] and (pos[pause_frames] == pos[int(m3["i_pause"])]).all(), "M3's pause hold is not a hold"
    j_pause = int(np.flatnonzero(app_frames == int(m3["i_pause"]))[0])
    assert app_frames[-1] == int(m3["i_G"]), "M3's last approach sample is not i_G"
    # M3's transport samples 0..n_t: sample 0 = G = the close hold's last frame
    close_frames = np.flatnonzero(phase == "close")
    tr_frames = np.concatenate([[close_frames[-1]], np.flatnonzero(phase == "transport")])
    assert len(tr_frames) == fr["transport_intervals"] + 1 and tr_frames[-1] == int(m3["i_P"])
    j_peak = int(np.argmax(pos[tr_frames, 2]))              # the highest palm sample of the transport
    assert (closure[app_frames[1:]] == 0).all() and (closure[tr_frames[1:]] == 1).all(), "closure not constant where v2 re-times"

    segs = []
    path = PathBuilder()
    path.add(pos[:i_pre + 1], quat[:i_pre + 1], closure[:i_pre + 1], "prelude")
    i_prelude_end = path.last()
    a = app_frames
    p1, q1, s1 = retime(pos[a[:j_pause + 1]], quat[a[:j_pause + 1]], "approach1")
    i_pause = path.add(p1[1:], q1[1:], 0.0, "approach")        # sample 0 = the previous segment's last frame
    p2, q2, s2 = retime(pos[a[j_pause:]], quat[a[j_pause:]], "approach2")
    i_G = path.add(p2[1:], q2[1:], 0.0, "approach")
    i_close_end = path.add(pos[close_frames], quat[close_frames], closure[close_frames], "close")
    t = tr_frames
    p3, q3, s3 = retime(pos[t[:j_peak + 1]], quat[t[:j_peak + 1]], "transport1")
    i_peak = path.add(p3[1:], q3[1:], 1.0, "transport")
    p4, q4, s4 = retime(pos[t[j_peak:]], quat[t[j_peak:]], "transport2")
    i_P = path.add(p4[1:], q4[1:], 1.0, "transport")
    rest = np.arange(int(m3["i_P"]) + 1, len(pos))            # release + retreat, unchanged
    for k in rest:
        path.add(pos[k], quat[k], closure[k], str(phase[k]))
    out = path.arrays()

    starts = dict(approach1=i_prelude_end, approach2=i_pause, transport1=i_close_end, transport2=i_peak)
    ends = dict(approach1=i_pause, approach2=i_G, transport1=i_peak, transport2=i_P)
    m3_ends = dict(approach1=(i_pre, int(a[j_pause])), approach2=(int(a[j_pause]), int(a[-1])), transport1=(int(t[0]), int(t[j_peak])),
                   transport2=(int(t[j_peak]), int(t[-1])))
    for s in (s1, s2, s3, s4):
        n = s["name"]
        s.update(v2_frames=[starts[n], ends[n]], m3_frames=list(m3_ends[n]))
        # the spec's end assert, on the assembled path: v2's segment ends sit exactly on M3's
        err = max(float(np.abs(out["palm_pos"][starts[n]] - pos[m3_ends[n][0]]).max()), float(np.abs(out["palm_pos"][ends[n]] - pos[m3_ends[n][1]]).max()))
        assert err <= END_TOL_M, f"{n}: assembled segment ends differ from M3's by {err} m"
        segs.append(s)
    out.update(i_pause=int(i_pause), i_G=int(i_G), i_P=int(i_P), i_release=int(i_P), i_prelude_end=int(i_prelude_end),
               i_close_end=int(i_close_end), i_peak=int(i_peak))
    out["follow"] = dict(approach1=[0, int(i_pause)], approach2=[int(i_pause), int(i_G)], transport1=[int(i_close_end), int(i_peak)],
                         transport2=[int(i_peak), int(i_P)])
    info = dict(m3["info"])
    info["m3_frames"] = info.pop("frames")
    info["frames"] = dict(prelude=i_prelude_end + 1, approach1=s1["frames"] - 1, approach2=s2["frames"] - 1, close=len(close_frames),
                          transport1=s3["frames"] - 1, transport2=s4["frames"] - 1, release_retreat=len(rest), total=len(out["palm_pos"]))
    info["v2"] = dict(T_R_s=T_R, j_pause=j_pause, j_peak=j_peak, peak_z_m=float(pos[t[j_peak], 2]), pause_hold_frames_removed=len(pause_frames),
                      m3_i_pause=int(m3["i_pause"]), m3_i_G=int(m3["i_G"]), m3_i_P=int(m3["i_P"]), segments=segs)
    out["info"] = info
    return out


# ---------------------------------------------------------------------------------------------------------------- (a) v2
def _naive_closure(T: int, i_contact: int, i_release: int) -> np.ndarray:
    """naive.naive_path's closure rule: 0 -> 1 over CLOSE_RAMP_S from i_contact, 1 -> 0 over RELEASE_S from i_release."""
    i = np.arange(T)
    up = np.clip((i - i_contact) / (CLOSE_RAMP_S * FPS), 0.0, 1.0)
    down = np.clip(1.0 - (i - i_release) / n_frames(RELEASE_S), 0.0, 1.0)
    return np.minimum(up, down)


def v2_from_m3_naive(m3: dict) -> dict:
    """v2 timing of an M3 naive path (the dict naive.naive_path returns): the replay re-timed as one segment."""
    pos, quat, closure, phase = (np.asarray(m3[k]) for k in ("palm_pos", "palm_quat_wxyz", "closure", "phase"))
    pos, quat, closure = pos.astype(float), quat.astype(float), closure.astype(float)
    info3 = m3["info"]
    base = _last(phase, "to_a0")                              # = replay sample 0 (my mapped pose at a0)
    rep = np.concatenate([[base], np.flatnonzero(phase == "replay")])
    n = info3["frames"]["replay_intervals"]
    assert len(rep) == n + 1 and rep[-1] == int(m3["i_release"]) == base + n
    # the closure must be naive_path's rule at M3's frames (so recomputing it at the mapped frames is the same rule)
    assert np.array_equal(_naive_closure(len(pos), info3["i_contact"], int(m3["i_release"])), closure), "closure is not naive_path's rule"

    p, q, seg = retime(pos[rep], quat[rep], "replay")
    N = len(p) - 1
    s_new = sigma(np.arange(N + 1) / FPS, n / FPS)            # M3 replay time at each new frame

    def mapped(frame: int) -> int:
        """The new frame at which the path is where M3 had it at ``frame`` (nearest in M3 time)."""
        return base + int(np.argmin(np.abs(s_new - (frame - base) / FPS)))

    i_pre = _last(phase, "prelude")
    path = PathBuilder()
    path.add(pos[:i_pre + 1], quat[:i_pre + 1], 0.0, "prelude")
    path.add(pos[i_pre + 1:base + 1], quat[i_pre + 1:base + 1], 0.0, "to_a0")
    i_release = path.add(p[1:], q[1:], 0.0, "replay")
    rel = np.flatnonzero(phase == "release")
    assert rel[0] == rep[-1] + 1 and rel[-1] == len(pos) - 1
    path.add(pos[rel], quat[rel], 0.0, "release")
    out = path.arrays()
    i_contact, i_G, i_P = mapped(info3["i_contact"]), mapped(int(m3["i_G"])), mapped(int(m3["i_P"]))
    assert i_release == mapped(int(m3["i_release"])) == base + N
    out["closure"] = _naive_closure(len(out["palm_pos"]), i_contact, i_release)
    err = float(np.abs(out["palm_pos"][[base, i_release]] - pos[[base, int(m3["i_release"])]]).max())
    assert err <= END_TOL_M, f"replay: assembled ends differ from M3's by {err} m"
    seg.update(v2_frames=[int(base), int(i_release)], m3_frames=[int(base), int(m3["i_release"])])
    out.update(i_pause=-1, i_G=int(i_G), i_P=int(i_P), i_release=int(i_release), i_prelude_end=i_pre, i_close_end=-1, i_peak=-1)
    out["follow"] = dict(all=[0, len(out["palm_pos"]) - 1])
    info = dict(info3)
    info["m3_frames"] = info.pop("frames")
    info["frames"] = dict(prelude=out["i_prelude_end"] + 1, to_a0=base - out["i_prelude_end"], replay=N, release=len(rel), total=len(out["palm_pos"]))
    info["i_contact"] = int(i_contact)
    info["P"] = out["palm_pos"][i_P].tolist()
    info["v2"] = dict(T_R_s=T_R, segments=[seg], m3_i_contact=int(info3["i_contact"]), m3_i_G=int(m3["i_G"]), m3_i_P=int(m3["i_P"]),
                      m3_i_release=int(m3["i_release"]),
                      event_mapping="new frame = the re-timed frame whose M3 time sigma(j / 50) is nearest to the M3 frame's time")
    out["info"] = info
    return out


# ---------------------------------------------------------------------------------------------------------------- entry
@lru_cache(maxsize=None)
def _demo(clip: str) -> dict:
    return load_demo(clip)


def default_oracle(cup_height: float) -> dict:
    """The mug_v1 oracle reference from config/objects.yaml (r = body diameter / 2). M3's runs took r from the compiled
    cup_geom; run_m3_v2 `paths` checks that the two agree."""
    return oracle_reference(dict(r=mug_config()["body_outer_diameter_m"] / 2, h=cup_height))


def m3_path(clip: str, variant: str, layout: dict, p_s, q_now_wxyz, *, oracle: dict | None = None) -> dict:
    """M3's own path for this run (warp.object_aware_path / naive.naive_path, unchanged), with k at M3's recorded layout."""
    demo = _demo(clip)
    oracle = oracle or default_oracle(float(layout["cup_height"]))
    recorded = recorded_layout(demo, oracle)                  # k is computed here, exactly as M3 does
    if variant in NAIVE_VARIANTS:
        return naive_path(demo, recorded, p_s, q_now_wxyz, oracle=oracle)     # layout-blind: the recorded layout's translation
    if variant in B_VARIANTS:
        return object_aware_path(demo, variant, layout, p_s, q_now_wxyz, recorded=recorded, oracle=oracle)
    raise ValueError(f"variant must be one of {B_VARIANTS + NAIVE_VARIANTS}")


def v2_path(clip: str, variant: str, layout: dict, p_s, q_now_wxyz, *, oracle: dict | None = None) -> dict:
    """The v2 path of one run. ``layout`` = dict(name, cup_xy, coaster_xy, handle_yaw, cup_height) (sim cup CENTRE xy, as M3);
    ``p_s`` / ``q_now_wxyz`` = the standing right palm after PickPlace.run's 2 s hold (read_state). For "naive" the layout is
    ignored (M3's fixed recorded-layout translation). Returns palm_pos (T, 3), palm_quat_wxyz (T, 4), closure (T,), phase (T,),
    the indices i_prelude_end, i_pause, i_G, i_close_end, i_peak, i_P, i_release (-1 = none), ``follow`` (name -> [first,
    last] frame, inclusive, of each slice M4 follows) and ``info`` (M3's info + "v2": segments, j_pause, j_peak)."""
    m3 = m3_path(clip, variant, layout, p_s, q_now_wxyz, oracle=oracle)
    return v2_from_m3_naive(m3) if variant in NAIVE_VARIANTS else v2_from_m3_b(m3)


# ---------------------------------------------------------------------------------------------------------------- oracle
def oracle_waypoints(layout: dict, p_s, q_now_wxyz, oracle: dict) -> dict:
    """The oracle's nominal targets for a layout (PickPlace.run's formulas with the cup at rest and the nominal in-hand offset;
    the same lines as warp.time_scale's oracle legs)."""
    h = float(layout["cup_height"])
    c = base_to_centre(np.array([*np.asarray(layout["cup_xy"], float), TABLE_TOP]), h)       # cup at rest
    quat = np.asarray(oracle["quat_ora_wxyz"], float)
    G = c + oracle["palm_offset"]
    pre_p = G - PREGRASP_BACKOFF * oracle["approach_dir"]
    z_clear = max(gen._clearance_height(np.asarray(q_now_wxyz, float), quat, "right", TABLE_TOP), pre_p[2])
    p_r0 = np.array([p_s[0], p_s[1], z_clear])
    lifted = G + np.array([0.0, 0.0, LIFT_HEIGHT])
    cup_in_hand = c - G                                       # nominal: the in-hand offset at the grasp, kept through the lift
    cup_place = np.array([*np.asarray(layout["coaster_xy"], float), TABLE_TOP + COASTER_THICKNESS]) + np.array([0.0, 0.0, h / 2 + PLACE_CLEARANCE])
    place_p = cup_place - cup_in_hand
    above_p = place_p.copy()
    above_p[2] = lifted[2]
    d_tr = float(np.linalg.norm(above_p - lifted))
    back = place_p - RETREAT_BACK * oracle["approach_dir"] + np.array([0.0, 0.0, RETREAT_UP])
    return dict(cup_centre=c, quat=quat, G=G, pre_p=pre_p, z_clear=float(z_clear), p_r0=p_r0, lifted=lifted, cup_in_hand=cup_in_hand,
                cup_place=cup_place, place_p=place_p, above_p=above_p, transport_dist_m=d_tr,
                transport_s=max(ORACLE_TRANSPORT_MIN_S, d_tr / TRANSPORT_SPEED), pregrasp_dist_m=float(np.linalg.norm(pre_p - p_r0)), back=back)


def oracle_path(layout: dict, p_s, q_now_wxyz, *, oracle: dict | None = None) -> dict:
    """The oracle's nominal path for ``layout`` (module docstring), in M3's path format. Frame 0 = (p_s, q_now): the start M3's
    paths use (the real oracle starts its first move at the IK model's palm_start, <= 0.09 mm away in M3's runs).
    i_pause = the pre-grasp arrival (8 cm before G, where the oracle settles; M3's pause was also 8 cm of arc before G);
    i_peak = -1 (the oracle's transport has no single peak: it runs level at the lift height)."""
    oracle = oracle or default_oracle(float(layout["cup_height"]))
    w = oracle_waypoints(layout, p_s, q_now_wxyz, oracle)
    quat = w["quat"]
    path = PathBuilder()
    pp, pq = min_jerk_path(np.asarray(p_s, float), np.asarray(q_now_wxyz, float), [(w["p_r0"], quat, PRELUDE_S)])
    i_prelude_end = path.add(pp, pq, 0.0, "prelude")
    ap, aq = min_jerk_path(w["p_r0"], quat, [(w["pre_p"], quat, ORACLE_PREGRASP_S), (w["G"], quat, ORACLE_APPROACH_S)])
    path.add(ap[1:], aq[1:], 0.0, "approach")
    i_pregrasp = i_prelude_end + n_frames(ORACLE_PREGRASP_S)
    i_G = path.last()
    n_c = n_frames(CLOSE_S)
    i_close_end = path.add(np.repeat(w["G"][None], n_c, axis=0), quat, np.minimum(1.0, np.arange(1, n_c + 1) / (CLOSE_RAMP_S * FPS)), "close")
    tp, tq = min_jerk_path(w["G"], quat, [(w["lifted"], quat, ORACLE_LIFT_S), (w["above_p"], quat, w["transport_s"]),
                                          (w["place_p"], quat, ORACLE_LOWER_S)])
    i_P = path.add(tp[1:], tq[1:], 1.0, "transport")
    i_lifted = i_close_end + n_frames(ORACLE_LIFT_S)
    i_above = i_lifted + max(int(round(w["transport_s"] * FPS)), 1)
    n_r = n_frames(RELEASE_S)
    path.add(np.repeat(w["place_p"][None], n_r, axis=0), quat, 1.0 - np.arange(1, n_r + 1) / n_r, "release")
    path.add(np.repeat(w["place_p"][None], n_frames(RELEASE_HOLD_S), axis=0), quat, 0.0, "release")
    rp, rq = min_jerk_path(w["place_p"], quat, [(w["back"], quat, RETREAT_S)])
    path.add(rp[1:], rq[1:], 0.0, "retreat")
    path.add(np.repeat(w["back"][None], n_frames(RETREAT_HOLD_S), axis=0), quat, 0.0, "retreat")
    out = path.arrays()
    assert np.allclose(out["palm_pos"][[i_pregrasp, i_G, i_lifted, i_above, i_P]], [w["pre_p"], w["G"], w["lifted"], w["above_p"], w["place_p"]], atol=1e-12)
    out.update(i_pause=int(i_pregrasp), i_G=int(i_G), i_P=int(i_P), i_release=int(i_P), i_prelude_end=int(i_prelude_end),
               i_close_end=int(i_close_end), i_peak=-1)
    R_G = oracle["R_ora"]
    u = np.asarray(oracle["approach_dir"], float)
    out["info"] = dict(
        variant="oracle", layout=layout.get("name"), clip=None, G=w["G"].tolist(), R_G=np.asarray(R_G).tolist(), quat_G_wxyz=quat.tolist(),
        P=w["place_p"].tolist(), u=u.tolist(), retreat_to=w["back"].tolist(), p_r0=w["p_r0"].tolist(), z_clear=w["z_clear"],
        pre_p=w["pre_p"].tolist(), lifted=w["lifted"].tolist(), above_p=w["above_p"].tolist(), cup_centre_nominal=w["cup_centre"].tolist(),
        cup_in_hand_nominal=w["cup_in_hand"].tolist(), cup_place=w["cup_place"].tolist(), transport_s=w["transport_s"],
        transport_dist_m=w["transport_dist_m"], pregrasp_dist_m=w["pregrasp_dist_m"],
        frames=dict(prelude=i_prelude_end + 1, pregrasp=n_frames(ORACLE_PREGRASP_S), approach=n_frames(ORACLE_APPROACH_S), close=n_c,
                    lift=n_frames(ORACLE_LIFT_S), transport=i_above - i_lifted, lower=i_P - i_above, release=n_r + n_frames(RELEASE_HOLD_S),
                    retreat=len(rp) - 1 + n_frames(RETREAT_HOLD_S), total=len(out["palm_pos"])),
        indices=dict(pregrasp=int(i_pregrasp), lifted=int(i_lifted), above_p=int(i_above)),
        nominal="cup at rest at the layout (base on the table top), in-hand offset = cup - palm at G (warp.time_scale's legs); "
                "the real oracle reads the cup after scene.reset (1 mm above rest) and measures the in-hand offset after the lift")
    out["follow"] = {}
    return out


def save_path(path: dict, file: Path) -> None:
    """The v2 npz: M3's fields plus the split indices (SAVE_FIELDS); phase as fixed-width unicode (allow_pickle=False loads it)."""
    arrays = {k: np.asarray(path[k]) for k in SAVE_FIELDS}
    arrays["phase"] = np.asarray([str(p) for p in path["phase"]], dtype=str)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, **arrays)


# ---------------------------------------------------------------------------------------------------------------- M4b hybrid
HYBRID_VARIANTS = ("b_hybrid", "b_hybrid_h45")
PHI_VARIANT = "b_hybrid_phi"    # M5: b_hybrid with the grasp heading set to a given phi
HYBRID_TOL_M = 1e-12            # b_hybrid self-check: its approach vs M3's b_level approach + w (PG - G)


def hybrid_grasp(variant: str, cup_xy, human: dict, oracle: dict, phi: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(G, R_G, q) of a hybrid variant. b_hybrid = b_level's grasp (grasp_pose: my heading phi_h, my height h_frac_h).
    b_hybrid_h45 = the same q (my height) with the oracle's rotation R_ora: grasp_pose's b_level formula with my heading
    replaced by the oracle's, R = R_z(0) R_ora = R_ora (asserted bit for bit)."""
    if variant == "b_hybrid":
        return grasp_pose("b_level", cup_xy, human, oracle)
    if variant == "b_hybrid_h45":
        G, R, q = grasp_pose("b_level", cup_xy, dict(human, phi_h=oracle["phi_ora"]), oracle)
        assert np.array_equal(R, oracle["R_ora"]), "R_z(0) R_ora is not R_ora"
        return G, R, q
    if variant == PHI_VARIANT:          # M5: grasp_pose's b_level formula with my heading replaced by phi [rad]
        return grasp_pose("b_level", cup_xy, dict(human, phi_h=float(phi)), oracle)
    raise ValueError(f"variant must be one of {HYBRID_VARIANTS + (PHI_VARIANT,)}")


def hybrid_path(clip: str, variant: str, layout: dict, p_s, q_now_wxyz, *, oracle: dict | None = None, phi: float | None = None) -> dict:
    """M4b: the v2 path with a robot-specific final approach (``b_hybrid``, ``b_hybrid_h45``; M5: ``b_hybrid_phi``, the grasp
    heading set to ``phi`` [rad], everything else b_hybrid's). Same arguments and return format as v2_path.

    Grasp G, R_G from ``hybrid_grasp``; u = M3's horizontal unit vector of (q - G) (warp.object_aware_path's u, which is also
    the oracle's approach_dir definition: palm -> cup axis); pre-grasp point PG = G - PREGRASP_BACKOFF u.
      prelude      M3's (warp.prelude, turning to R_G).
      approach     my approach a0..g, time-scaled by k and warped as M3 does it (p~ + (1 - w)(p_r0 - p~(0)) + w(END - p~(1)),
                   then z >= table top + 3 cm), but with END = PG instead of G and no pause; re-timed rest to rest as ONE
                   segment ("approach1", Step 1's ``retime``). It ends at i_pause = PG, where Execute settles.
      final_approach  the oracle's own straight leg PG -> G (min_jerk_path, ORACLE_APPROACH_S = 2.0 s, R_G held), ends at i_G.
                   Nominal only: Execute plays it with the oracle's ``move`` from the live palm, never with follow.
      close, transport 1 / 2, release, retreat: M3's (warp.object_aware_path's lines: transport warped G -> P with the 10 cm
                   bump cap) with v2's split at the highest transport sample and v2's re-timing of both halves.
    For b_hybrid the whole path is checked against M3 / v2's own b_level output for the same run (``_check_against_b_level``):
    the warp is linear in its end point, so the approach must equal M3's b_level approach samples + w (PG - G) (PG - G is
    horizontal, so the z clamp is unchanged); the prelude and everything from the close on must be v2 b_level's, bit for bit.
    """
    if variant not in HYBRID_VARIANTS + (PHI_VARIANT,):
        raise ValueError(f"variant must be one of {HYBRID_VARIANTS + (PHI_VARIANT,)}")
    if (phi is None) != (variant != PHI_VARIANT):
        raise ValueError(f"phi is given exactly for {PHI_VARIANT}")
    demo = _demo(clip)
    h = float(layout["cup_height"])
    oracle = oracle or default_oracle(h)
    assert abs(oracle["h"] - h) < 1e-12, "layout and oracle disagree on the mug height"
    recorded = recorded_layout(demo, oracle)                  # k is computed here, exactly as M3 does
    human = human_at_grasp(demo, h)
    t = demo["t"]
    ev = event_frames(demo)
    g, a0 = ev["g"], approach_start(demo, ev["g"])
    ts = time_scale(demo, p_s, q_now_wxyz, recorded, oracle)
    k = ts["k"]

    cup_xy, coaster_xy = np.asarray(layout["cup_xy"], float), np.asarray(layout["coaster_xy"], float)
    G, R_G, q = hybrid_grasp(variant, cup_xy, human, oracle, phi)
    quat_G = quat_of(R_G)
    u = (q - G) * np.array([1.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    PG = G - PREGRASP_BACKOFF * u
    P = G + np.array([*(coaster_xy - cup_xy), COASTER_THICKNESS + PLACE_CLEARANCE])
    offset_xy = human_to_sim_offset(demo, cup_xy, h)
    p_tilde = table_to_world(demo["palm_pos_table"], offset_xy)      # my palm, my mug's base on the sim mug's base

    path = PathBuilder()
    pre_pos, pre_quat, p_r0, z_clear = prelude(p_s, q_now_wxyz, G, quat_G)
    i_prelude_end = path.add(pre_pos, pre_quat, 0.0, "prelude")

    # approach: M3's warp onto p_r0 -> PG, M3's clamp, then one rest-to-rest segment
    n_a = intervals(k, t, a0, g)
    pt, _ = resample(t, p_tilde, a0, g, n_a)
    w = min_jerk_w(np.arange(n_a + 1) / n_a)[:, None]
    app = pt + (1 - w) * (p_r0 - pt[0]) + w * (PG - pt[-1])
    low = app[:, 2] < APPROACH_MIN_Z
    app[low, 2] = APPROACH_MIN_Z
    app[0] = pre_pos[-1]                                       # sample 0 = p_r0 = the prelude's last frame (as in M3)
    p1, q1, s1 = retime(app, np.repeat(quat_G[None], n_a + 1, axis=0), "approach1")
    i_pause = path.add(p1[1:], q1[1:], 0.0, "approach")

    # the oracle's final approach (nominal)
    fp, fq = min_jerk_path(PG, quat_G, [(G, quat_G, ORACLE_APPROACH_S)])
    i_G = path.add(fp[1:], fq[1:], 0.0, "final_approach")
    n_c = n_frames(CLOSE_S)
    i_close_end = path.add(np.repeat(G[None], n_c, axis=0), quat_G, np.minimum(1.0, np.arange(1, n_c + 1) / (CLOSE_RAMP_S * FPS)), "close")

    # transport: M3's warp G -> P and bump cap; sample 0 = G (the close hold's last frame); v2's split and re-timing
    n_t = intervals(k, t, ev["lift"], ev["place"])
    pt, _ = resample(t, p_tilde, ev["lift"], ev["place"], n_t)
    s = (np.arange(n_t + 1) / n_t)[:, None]
    tr = pt + (1 - s) * (G - pt[0]) + s * (P - pt[-1])
    line_z = (1 - s[:, 0]) * G[2] + s[:, 0] * P[2]
    bump = tr[:, 2] - line_z
    bump_max = float(bump.max())
    bump_scale = LIFT_HEIGHT / bump_max if bump_max > LIFT_HEIGHT else 1.0
    tr[:, 2] = line_z + bump_scale * bump
    tr[0] = G
    j_peak = int(np.argmax(tr[:, 2]))
    tq = np.repeat(quat_G[None], n_t + 1, axis=0)
    p3, q3, s3 = retime(tr[:j_peak + 1], tq[:j_peak + 1], "transport1")
    i_peak = path.add(p3[1:], q3[1:], 1.0, "transport")
    p4, q4, s4 = retime(tr[j_peak:], tq[j_peak:], "transport2")
    i_P = path.add(p4[1:], q4[1:], 1.0, "transport")

    # release and retreat, as M3 (the oracle's)
    n_r = n_frames(RELEASE_S)
    path.add(np.repeat(P[None], n_r, axis=0), quat_G, 1.0 - np.arange(1, n_r + 1) / n_r, "release")
    path.add(np.repeat(P[None], n_frames(RELEASE_HOLD_S), axis=0), quat_G, 0.0, "release")
    back = P - RETREAT_BACK * u + np.array([0.0, 0.0, RETREAT_UP])
    rp, rq = min_jerk_path(P, quat_G, [(back, quat_G, RETREAT_S)])
    path.add(rp[1:], rq[1:], 0.0, "retreat")
    path.add(np.repeat(back[None], n_frames(RETREAT_HOLD_S), axis=0), quat_G, 0.0, "retreat")

    out = path.arrays()
    for seg, first, last in ((s1, i_prelude_end, i_pause), (s3, i_close_end, i_peak), (s4, i_peak, i_P)):
        seg.update(v2_frames=[int(first), int(last)])
    out.update(i_pause=int(i_pause), i_G=int(i_G), i_P=int(i_P), i_release=int(i_P), i_prelude_end=int(i_prelude_end),
               i_close_end=int(i_close_end), i_peak=int(i_peak))
    out["follow"] = dict(approach1=[0, int(i_pause)], approach2=[int(i_pause), int(i_G)], transport1=[int(i_close_end), int(i_peak)],
                         transport2=[int(i_peak), int(i_P)])
    out["info"] = dict(
        variant=variant, layout=layout.get("name"), clip=clip, k=k, time_scale=ts, a0=a0, a0_time_s=float(t[a0] - t[0]), events=ev,
        n_clamped=int(low.sum()), PG_clamped=bool(low[-1]), bump_max_m=bump_max, bump_scale=bump_scale,
        G=G.tolist(), R_G=R_G.tolist(), quat_G_wxyz=quat_G.tolist(), heading_deg=math.degrees(math.atan2(R_G[1, 1], R_G[0, 1])),
        phi_h_deg=math.degrees(human["phi_h"]), PG=PG.tolist(), P=P.tolist(), q=q.tolist(), u=u.tolist(), retreat_to=back.tolist(),
        p_r0=p_r0.tolist(), z_clear=z_clear, human_to_sim_offset_xy=offset_xy.tolist(),
        final_approach="the oracle's move PG -> G, min-jerk, 2.0 s (path frames i_pause..i_G = its nominal version from PG)",
        frames=dict(prelude=i_prelude_end + 1, approach1=s1["frames"] - 1, final_approach=i_G - i_pause, close=n_c,
                    transport1=s3["frames"] - 1, transport2=s4["frames"] - 1, total=len(out["palm_pos"])),
        v2=dict(T_R_s=T_R, j_peak=j_peak, peak_z_m=float(tr[j_peak, 2]), segments=[s1, s3, s4]))
    if phi is not None:
        out["info"]["phi_rad"] = float(phi)
    if variant == "b_hybrid":
        out["info"]["check_against_b_level"] = _check_against_b_level(out, clip, layout, p_s, q_now_wxyz, oracle, PG, G, app)
    return out


def _check_against_b_level(out: dict, clip: str, layout: dict, p_s, q_now_wxyz, oracle: dict, PG, G, app) -> dict:
    """b_hybrid vs M3 / v2's own b_level path of the same run (hybrid_path's docstring). Raises if anything differs."""
    m3 = m3_path(clip, "b_level", layout, p_s, q_now_wxyz, oracle=oracle)
    v2 = v2_from_m3_b(m3)
    phase3 = np.asarray(m3["phase"])
    app_frames = np.concatenate([[_last(phase3, "prelude")], np.flatnonzero(phase3 == "approach")])
    m3_app = np.asarray(m3["palm_pos"], float)[app_frames]
    n_a = len(m3_app) - 1
    assert len(app) == n_a + 1, "approach sample counts differ"
    w = min_jerk_w(np.arange(n_a + 1) / n_a)[:, None]
    app_err = float(np.abs(m3_app + w * (PG - G) - app).max())
    assert app_err <= HYBRID_TOL_M, f"approach differs from M3's + w (PG - G) by {app_err} m"
    pre, n_c = out["i_prelude_end"] + 1, n_frames(CLOSE_S)
    after = {k: (out[k][out["i_close_end"] - n_c + 1:], v2[k][v2["i_close_end"] - n_c + 1:]) for k in ("palm_pos", "palm_quat_wxyz", "closure")}
    same_after = all(np.array_equal(a, b) for a, b in after.values())
    same_pre = all(np.array_equal(out[k][:pre], v2[k][:pre]) for k in ("palm_pos", "palm_quat_wxyz"))
    assert same_after and same_pre, f"b_hybrid differs from v2 b_level: prelude identical {same_pre}, close..retreat identical {same_after}"
    return dict(approach_vs_m3_plus_w_PG_minus_G_max_m=app_err, prelude_identical=same_pre, close_to_retreat_identical=same_after,
                b_level_i_close_end=int(v2["i_close_end"]))
