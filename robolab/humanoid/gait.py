"""Walking pattern: footsteps, swing-foot trajectories, ZMP reference and
CoM planning by Kajita's preview control on the cart-table (LIPM) model.

Reference: S. Kajita et al., "Biped walking pattern generation by using preview
control of zero-moment point", ICRA 2003.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

G = 9.81


@dataclass(frozen=True)
class GaitParams:
    step_length: float = 0.20     # distance the feet leapfrog per step, m
    step_width: float = 0.18      # lateral distance between foot centres, m
    step_height: float = 0.05     # swing-foot apex, m
    step_time: float = 0.70       # one step = double support + single support, s
    double_support: float = 0.20  # fraction of step_time with both feet down
    n_steps: int = 6
    t_start: float = 1.0          # standing before the first step, s
    t_end: float = 1.0            # standing after the last step, s
    dt: float = 0.01              # control period, s
    pelvis_height: float = 0.68   # hip-joint height, m (keeps the knees bent)


@dataclass
class FootPlan:
    t: np.ndarray            # (T,)
    left: np.ndarray         # (T, 3) sole-centre positions
    right: np.ndarray        # (T, 3)
    phase: np.ndarray        # (T,) 0 double support, 1 left stance, 2 right stance
    zmp_ref: np.ndarray      # (T, 2)


def _smooth(tau):
    """Quintic 0->1 with zero velocity and acceleration at both ends."""
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def plan_feet(p: GaitParams) -> FootPlan:
    half_w = p.step_width / 2
    t_ds = p.double_support * p.step_time
    t_ss = p.step_time - t_ds
    seg = []  # (duration, kind, data)

    x = {"l": 0.0, "r": 0.0}
    seg.append((p.t_start, "stand", None))
    prev_zmp = np.array([0.0, 0.0])
    for i in range(1, p.n_steps + 1):
        swing = "l" if i % 2 else "r"
        stance = "r" if swing == "l" else "l"
        stance_zmp = np.array([x[stance], half_w if stance == "l" else -half_w])
        seg.append((t_ds, "ds", (prev_zmp, stance_zmp)))
        x_new = (i if i < p.n_steps else i - 1) * p.step_length
        seg.append((t_ss, "ss", (swing, x[swing], x_new, stance_zmp)))
        x[swing] = x_new
        prev_zmp = stance_zmp
    mid = np.array([(x["l"] + x["r"]) / 2, 0.0])
    seg.append((t_ds, "ds", (prev_zmp, mid)))
    seg.append((p.t_end, "stand", None))

    durations = np.array([sg[0] for sg in seg])
    bounds = np.concatenate([[0.0], np.cumsum(durations)])
    n = int(round(bounds[-1] / p.dt))
    t = np.arange(n) * p.dt
    # every sample belongs to exactly one segment (robust to float round-off)
    seg_of = np.clip(np.searchsorted(bounds, t + 1e-9, side="right") - 1, 0, len(seg) - 1)
    left = np.zeros((n, 3)); left[:, 1] = half_w
    right = np.zeros((n, 3)); right[:, 1] = -half_w
    phase = np.zeros(n, dtype=int)
    zmp = np.zeros((n, 2))

    feet_x = {"l": 0.0, "r": 0.0}
    last_zmp = np.zeros(2)
    for i, (dur, kind, data) in enumerate(seg):
        m = seg_of == i
        tau = np.clip((t[m] - bounds[i]) / dur, 0.0, 1.0)
        if kind == "stand":
            zmp[m] = last_zmp
        elif kind == "ds":
            a, b = data
            zmp[m] = a + _smooth(tau)[:, None] * (b - a)
            last_zmp = b
        else:
            swing, xa, xb, stance_zmp = data
            zmp[m] = stance_zmp
            phase[m] = 2 if swing == "l" else 1
            sm = _smooth(tau)
            foot = left if swing == "l" else right
            foot[m, 0] = xa + sm * (xb - xa)
            foot[m, 2] = p.step_height * 0.5 * (1 - np.cos(2 * np.pi * tau))
        # feet that are not swinging stay where they are
        for side, foot in (("l", left), ("r", right)):
            if not (kind == "ss" and data[0] == side):
                foot[m, 0] = feet_x[side]
        if kind == "ss":
            feet_x[data[0]] = data[2]
    return FootPlan(t, left, right, phase, zmp)


def preview_control(zmp_ref, dt, zc, preview_time=1.6, q_err=1.0, r_jerk=1e-6):
    """CoM trajectory whose cart-table ZMP tracks zmp_ref (one axis or (T, 2)).

    Returns (com, com_acc, zmp_model), each shaped like zmp_ref."""
    zmp_ref = np.asarray(zmp_ref, dtype=float)
    if zmp_ref.ndim == 2:
        outs = [preview_control(zmp_ref[:, k], dt, zc, preview_time, q_err, r_jerk)
                for k in range(zmp_ref.shape[1])]
        return tuple(np.stack([o[i] for o in outs], axis=1) for i in range(3))

    a = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    b = np.array([[dt ** 3 / 6], [dt * dt / 2], [dt]])
    c = np.array([[1, 0, -zc / G]])
    # augmented system with the integrated tracking error (Katayama)
    aa = np.block([[np.eye(1), c @ a], [np.zeros((3, 1)), a]])
    bb = np.vstack([c @ b, b])
    ii = np.array([[1.0], [0], [0], [0]])
    ff = np.vstack([c @ a, a])
    qq = np.zeros((4, 4)); qq[0, 0] = q_err
    pp = solve_discrete_are(aa, bb, qq, np.array([[r_jerk]]))
    s = 1.0 / (r_jerk + (bb.T @ pp @ bb).item())
    gi = s * (bb.T @ pp @ ii).item()
    gx = s * (bb.T @ pp @ ff)
    ac = aa - s * bb @ (bb.T @ pp @ aa)
    n_prev = int(round(preview_time / dt))
    gd = np.zeros(n_prev)
    xx = -ac.T @ pp @ ii
    gd[0] = -gi
    for j in range(1, n_prev):
        gd[j] = s * (bb.T @ xx).item()
        xx = ac.T @ xx

    # Warm-up: a preview window of constant reference before t = 0, so the
    # controller starts from rest instead of reacting to steps it already sees.
    n = len(zmp_ref) + n_prev
    ref = np.concatenate([np.full(n_prev, zmp_ref[0]), zmp_ref, np.full(n_prev, zmp_ref[-1])])
    x = np.array([[zmp_ref[0]], [0.0], [0.0]])
    com, acc, zmp = np.zeros(n), np.zeros(n), np.zeros(n)
    err_sum = 0.0
    for k in range(n):
        z_k = (c @ x).item()
        com[k], acc[k], zmp[k] = x[0, 0], x[2, 0], z_k
        err_sum += z_k - ref[k]
        u = -gi * err_sum - (gx @ x).item() - gd @ ref[k + 1:k + 1 + n_prev]
        x = a @ x + b * u
    return com[n_prev:], acc[n_prev:], zmp[n_prev:]
