"""Reference builder: OUR timed palm path -> HERO whole-body IK -> HERO reference clip.

HERO's policy does not take a palm target; it tracks a 50 fps reference clip of the whole robot. HERO's own
``BenchReplanner`` builds such clips for a straight palm move to one goal. ``PathPlanner`` keeps everything of it
(live robot state -> mink IK on HERO's IK model with feet / pelvis / CoM / posture tasks and table clearance -> FK on the
training-geometry model -> ``ClipReference``) and replaces only the palm path: the active hand follows the dense timed
path we give it (positions + orientations, one sample per 50 Hz frame).

``min_jerk_path`` turns a few timed waypoints into such a dense path. HERO's closed-loop goal correction
(``GoalAdjuster`` + replanning, ``g <- g - 0.6 e``) is still available through the inherited ``BenchReplanner.step``.

Quaternions are wxyz here (HERO's IK convention); ``RobotState`` / ``ClipReference`` accessors are xyzw.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from hero_isaacsim import constants as HC
from sim2sim.bench.replan import _LEG_IDX, FPS, SIDES, BenchReplanner, slerp_wxyz
from sim2sim.reference import ClipReference


def min_jerk_path(p0: np.ndarray, q0_wxyz: np.ndarray, waypoints: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Dense 50 Hz palm path from (p0, q0) through ``waypoints`` = [(pos, quat_wxyz, duration_s), ...].

    Each leg uses a minimum-jerk time profile (zero velocity and acceleration at both ends) for the position and the
    same profile for a slerp of the orientation. Returns ((T, 3), (T, 4) wxyz); frame 0 is (p0, q0).
    """
    pos, quat = [np.asarray(p0, dtype=float)], [np.asarray(q0_wxyz, dtype=float)]
    for p1, q1, duration in waypoints:
        a, qa = pos[-1], quat[-1]
        n = max(int(round(duration * FPS)), 1)
        for i in range(1, n + 1):
            u = i / n
            s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
            pos.append(a + (np.asarray(p1, dtype=float) - a) * s)
            quat.append(slerp_wxyz(qa, np.asarray(q1, dtype=float), s))
    return np.array(pos), np.array(quat)


class PathPlanner(BenchReplanner):
    """``BenchReplanner`` whose palm follows our path instead of a straight line to ``goal.pos_w``.

    ``self.goal`` still sets the hand, the table slab of the IK model and -- for ``base="plan"`` -- the pelvis drop / lean
    the base moves to (HERO's benchmark base plan); ``base="current"`` keeps pelvis, legs and waist where they are.
    """

    def palm_start(self, st) -> tuple[np.ndarray, np.ndarray]:
        """Palm pose (pos, quat wxyz) of the active hand in the IK model at the robot's current joints = the IK's frame-0 palm."""
        palms, _ = self.gen._palm_start(self.bundle, self.q_start_from_robot(st))
        return palms[self.goal.hand]

    def plan_path(self, st, plant, path_pos: np.ndarray, path_quat: np.ndarray, *, base: str = "current",
                  hold_frames: int = 5, hold_s: float = 12.0, return_fk: bool = False):
        """Whole-body reference whose active palm follows ``path_pos`` / ``path_quat`` (T frames at 50 Hz), then holds.

        Mirrors ``BenchReplanner.plan_from_state`` (state ingestion, task priming, IK, padding, FK, clip) with our
        palm path in place of HERO's straight smoothstep path. Frame 0 should be the robot's palm (``palm_start``).

        Returns ``(ref, info)``. With ``return_fk=True`` (M3's kinematic check) it returns ``(ref, info, (fk_pos, fk_quat))``:
        the active palm's position (T, 3) and orientation (T, 4, wxyz) for the T path frames (not the hold padding), read from
        the same training-geometry FK pass the clip is built from, so they are what HERO will be asked to track. The flag only
        adds a return value: the default output is unchanged (results/m3_retarget/verify.json checks it byte for byte).
        """
        if base not in ("current", "plan"):
            raise ValueError("base must be 'current' or 'plan'")
        state = self.state_from_robot(st, plant)
        q_start = self.q_start_from_robot(st)
        plan = self._build_path_plan(state, q_start, np.asarray(path_pos, float), np.asarray(path_quat, float), base, hold_frames)
        self._prime_tasks(q_start)
        qp_failures_before = self.solver.qp_failures
        qs, resid = self.solver.solve_plan(plan, 0, q_start, self.goal.active_hands, q_before_start=None)
        jp = np.concatenate([qs[:, :7], qs[:, self.bundle.dof_qpos_idx]], axis=1)
        n_pad = max(int(round(hold_s * FPS)) - hold_frames, 0)
        jp = np.concatenate([jp, np.repeat(jp[-1:], n_pad, axis=0)], axis=0)
        fk = self.fk.fk_all(jp)
        ref = ClipReference.from_arrays(fps=FPS, joint_pos=jp, body_pos_w=fk["body_pos_w"], body_quat_w=fk["body_quat_w"], name=self.goal.clip_name)
        n_path = len(path_pos)
        info = {"frames": int(jp.shape[0]), "path_frames": n_path,
                "ik_max_path_err_cm": float(resid[:n_path, 0].max()) * 100.0,
                "ik_terminal_palm_err_cm": float(np.linalg.norm(fk["palm_pos_w"][n_path - 1, self.goal.hand_index] - path_pos[-1])) * 100.0,
                "qp_failures": int(self.solver.qp_failures - qp_failures_before)}
        if return_fk:
            hand = self.goal.hand_index
            return ref, info, (fk["palm_pos_w"][:n_path, hand].copy(), fk["palm_quat_w"][:n_path, hand].copy())
        return ref, info

    def _build_path_plan(self, state, q_start, path_pos, path_quat, base_mode, hold_frames):
        """Same plan as ``BenchReplanner.build_plan`` (feet flat, CoM over the support, base / posture blended on a smoothstep,
        inactive palm held), except that the active palm targets are our path and the blend spans the whole path."""
        gen, rs, scfg = self.gen, self._rs(), self.solver_cfg
        hand_i = self.goal.hand_index
        n_path = len(path_pos)
        n_reach = max(n_path - 1, 1)
        n_total = n_path + hold_frames
        base = self._base(state, base_mode)
        leg_new = {s: gen._leg_prior_for(base, s, state) for s in SIDES} if base_mode == "plan" else {s: dict(state.leg_prior[s]) for s in SIDES}
        h_new, pitch_new, ydel_new = base.pelvis_height, base.pelvis_pitch, base.pelvis_yaw_delta
        shift_new = np.asarray(base.pelvis_shift, dtype=float)
        s_base = np.array([rs.smoothstep(min(t / n_reach, 1.0)) for t in range(n_total)])
        Rz = rs.rot_z(state.yaw0)
        dofs0 = q_start[self.bundle.dof_qpos_idx]
        q_from = gen._posture_from(state.leg_prior, state.waist_pitch, state.waist_yaw, state.pelvis_height, state.pelvis_pitch)
        q_to = gen._posture_from(leg_new, base.waist_pitch, base.waist_yaw, h_new, pitch_new)
        arms = list(HC.ARM_DOF_IDX)
        q_from[arms] = dofs0[arms]
        q_to[arms] = dofs0[arms] if self.arm_posture is None else self.arm_posture[arms]
        if base_mode == "current":   # legs: exactly the measured joints
            for s, idx in _LEG_IDX.items():
                for j, key in zip(idx, ("hip_pitch", "knee", "ankle_pitch")):
                    q_from[j] = q_to[j] = state.leg_prior[s][key]
        pelvis_pos, pelvis_rpy, posture_q = np.empty((n_total, 3)), np.empty((n_total, 3)), np.empty((n_total, 29))
        for t in range(n_total):
            s = float(s_base[t])
            shift = state.pelvis_shift + (shift_new - state.pelvis_shift) * s
            xy = state.stance_center + Rz[:2, :2] @ shift
            pelvis_pos[t] = [xy[0], xy[1], state.pelvis_height + (h_new - state.pelvis_height) * s]
            pelvis_rpy[t] = [0.0, state.pelvis_pitch + (pitch_new - state.pelvis_pitch) * s,
                             state.yaw0 + state.pelvis_yaw_delta + (ydel_new - state.pelvis_yaw_delta) * s]
            posture_q[t] = q_from + (q_to - q_from) * s
        posture_cost = np.repeat(gen._posture_cost(scfg, self.goal.active_hands, "")[None], n_total, axis=0)
        palms, _ = gen._palm_start(self.bundle, q_start)
        palm_pos, palm_quat, palm_cost = np.empty((n_total, 2, 3)), np.empty((n_total, 2, 4)), np.zeros((n_total, 2))
        for i, side in enumerate(SIDES):
            palm_pos[:, i], palm_quat[:, i] = palms[side]
        idx = np.minimum(np.arange(n_total), n_path - 1)   # hold frames repeat the path's last sample
        palm_pos[:, hand_i], palm_quat[:, hand_i] = path_pos[idx], path_quat[idx]
        palm_cost[:, hand_i] = scfg.palm_pos_cost
        feet = gen._foot_plans(state, base, n_reach, n_total, s_base, False)
        flat_pts = {s: gen._flat_sole_points(state.foot_pos[s], state.yaw0) for s in SIDES}
        support = [gen._support_points(feet, t, flat_pts) for t in range(n_total)]
        return gen.SegmentPlan(frames=n_total, reach_frames=n_reach, hold_frames=hold_frames, palm_pos=palm_pos, palm_quat=palm_quat,
                               palm_cost=palm_cost, pelvis_pos=pelvis_pos, pelvis_rpy=pelvis_rpy, posture_q=posture_q, posture_cost=posture_cost,
                               feet=feet, target_pos_frame=palm_pos.copy(), target_quat_frame=palm_quat.copy(),
                               target_final_pos=palm_pos[-1].copy(), target_final_quat=palm_quat[-1].copy(),
                               retarget_frame=-1, kind="reach", support_points=support)

    def _base(self, state, mode):
        """``plan``: the goal's benchmark base (pelvis drop / pitch / waist pitch); ``current``: hold the measured base."""
        saved = self.cfg
        try:
            self.cfg = dataclasses.replace(saved, base_mode="clip" if mode == "plan" else "current")
            return self.base_for(state)
        finally:
            self.cfg = saved


def quat_angle_deg(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray) -> float:
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0_wxyz, q1_wxyz))))))
