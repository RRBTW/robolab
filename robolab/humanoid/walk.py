"""Kinematic walking simulation of the humanoid.

Pipeline
--------
1. Footsteps and swing-foot trajectories, ZMP reference (gait.plan_feet).
2. CoM trajectory from the ZMP reference by preview control (cart-table model).
3. Pelvis trajectory so that the WHOLE-BODY CoM (legs and arms included)
   follows the planned CoM: pelvis = CoM_ref + offset, the offset is corrected
   by a few fixed-point iterations of  offset += CoM_ref - CoM_actual.
4. Leg inverse kinematics every control period (DLS on SE(3), warm start).
   The legs are independent chains, so each (gait, leg) is one problem of a
   trajectory-IK kernel call (ik_dls_traj) covering all frames.
5. Whole-trajectory FK in one batch call; multi-body ZMP and support polygon
   margins as the balance check.

simulate_walks runs several gaits in lock-step: per CoM-correction pass ONE
IK call for all gaits and both legs (Cython: OpenMP over problems; CUDA: one
GPU thread per problem, float64) and ONE FK call for all frames of all gaits.
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field

import numpy as np

from ..transforms import _backend
from .balance import foot_corners, multibody_zmp, signed_margin, support_polygons
from .gait import GaitParams, plan_feet, preview_control
from .model import LEG_JOINTS


@dataclass
class WalkResult:
    t: np.ndarray
    q: np.ndarray                # (T, nq) joint trajectories
    base: np.ndarray             # (T, 4, 4) pelvis poses
    poses: np.ndarray            # (T, L, 4, 4) all link poses
    phase: np.ndarray            # (T,) 0 double, 1 left stance, 2 right stance
    feet_ref: dict               # 'l', 'r' -> (T, 3) planned sole positions
    com_ref: np.ndarray          # (T, 2) planned CoM (preview control)
    com: np.ndarray              # (T, 3) whole-body CoM from FK
    zmp_ref: np.ndarray          # (T, 2)
    zmp_cart: np.ndarray         # (T, 2) ZMP of the cart-table model
    zmp: np.ndarray              # (T, 2) multi-body ZMP from FK
    polygons: list               # support polygons
    margin: np.ndarray           # (T,) ZMP distance to polygon edge, + inside
    com_iterations: list = field(default_factory=list)   # max |CoM error| per pass, m
    ik_failures: int = 0
    ik_iterations: int = 0
    timings: dict = field(default_factory=dict)


def arm_swing(tree, q, left_x, right_x, step_length, amplitude):
    """Counter-swing of the arms: left arm forward when the right foot leads."""
    lead = np.clip((right_x - left_x) / step_length, -1.0, 1.0)
    q = q.copy()
    q[:, tree.joint_id("l_shoulder_pitch")] = -amplitude * lead
    q[:, tree.joint_id("r_shoulder_pitch")] = amplitude * lead
    q[:, tree.joint_id("l_shoulder_roll")] = 0.12
    q[:, tree.joint_id("r_shoulder_roll")] = -0.12
    q[:, tree.joint_id("l_elbow")] = -0.35
    q[:, tree.joint_id("r_elbow")] = -0.35
    return q


def simulate_walk(tree, params: GaitParams = GaitParams(), arm_amplitude=0.35,
                  com_passes=4, ik_tol=1e-9) -> WalkResult:
    return simulate_walks(tree, [params], arm_amplitude, com_passes, ik_tol)[0]


def simulate_walks(tree, params, arm_amplitude=0.35, com_passes=4, ik_tol=1e-9,
                   device="cpu") -> list:
    """simulate_walk for a list of gaits; results are the same as one by one.

    device="cpu": IK in the current kernel (Cython runs the problems in
    parallel). device="cuda": IK on the GPU in float64. The ZMP takes second
    differences of link positions, which amplifies IK residuals by 1/dt^2 = 1e4,
    so float32 is not accurate enough here. Timings are for the whole batch."""
    params = list(params)
    if device == "cpu":
        ik = _backend.kernel.ik_dls_traj
    elif device == "cuda":
        ik = functools.partial(_backend._load_cuda().ik_dls_traj, dtype=np.float64)
    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}")
    d, m_cnt = tree.dims, len(params)
    tm = {}
    t0 = time.perf_counter()
    plans = [plan_feet(p) for p in params]
    lens = [len(pl.t) for pl in plans]
    n_max = max(lens)
    tm["plan"] = time.perf_counter() - t0

    leg_idx = {s: np.array([tree.joint_id(j) for j in LEG_JOINTS[s]]) for s in "lr"}
    qs = []
    for p, pl in zip(params, plans):           # initial guess: knees bent, arms swinging
        q = np.zeros((len(pl.t), tree.nq))
        for side in "lr":
            q[:, tree.joint_id(f"{side}_hip_pitch")] = -0.35
            q[:, tree.joint_id(f"{side}_knee")] = 0.70
            q[:, tree.joint_id(f"{side}_ankle_pitch")] = -0.35
        qs.append(arm_swing(tree, q, pl.left[:, 0], pl.right[:, 0], p.step_length, arm_amplitude))

    # IK problems 2m (left leg of gait m) and 2m + 1 (right leg); frames padded to n_max
    link = np.array([tree.link_id(f"{s}_sole") for _ in params for s in "lr"], dtype=np.int32)
    sel = np.array([leg_idx[s] for _ in params for s in "lr"], dtype=np.int32)
    n_frames = np.repeat(lens, 2).astype(np.int32)
    target = np.tile(np.eye(4), (2 * m_cnt, n_max, 1, 1))
    for m, pl in enumerate(plans):
        target[2 * m, :lens[m], :3, 3] = pl.left
        target[2 * m + 1, :lens[m], :3, 3] = pl.right
    ik_base = np.tile(np.eye(4), (2 * m_cnt, n_max, 1, 1))

    offsets = [np.zeros((n, 2)) for n in lens]
    zcs = [p.pelvis_height + 0.10 for p in params]
    com_hist = [[] for _ in params]
    ik_fail, ik_iters = np.zeros(m_cnt, dtype=int), np.zeros(m_cnt, dtype=int)
    tm["ik"] = tm["fk_batch"] = 0.0
    for _ in range(com_passes):
        bases, com_refs, zmp_carts = [], [], []
        for m, (p, pl) in enumerate(zip(params, plans)):
            com_ref, _, zmp_cart = preview_control(pl.zmp_ref, p.dt, zcs[m])
            base = np.tile(np.eye(4), (lens[m], 1, 1))
            base[:, :2, 3] = com_ref + offsets[m]
            base[:, 2, 3] = p.pelvis_height
            bases.append(base); com_refs.append(com_ref); zmp_carts.append(zmp_cart)
            ik_base[2 * m, :lens[m]] = ik_base[2 * m + 1, :lens[m]] = base

        t1 = time.perf_counter()
        q0 = np.stack([qs[m][0] for m in range(m_cnt) for _ in "lr"])
        out, its, fails = ik(tree._parent, tree._offset, tree._axis, tree._qidx, tree._jtype,
                             q0, ik_base, link, target, sel, tree._lower, tree._upper, n_frames,
                             50, ik_tol, 1e-4)
        for m in range(m_cnt):
            qs[m][:, leg_idx["l"]] = out[2 * m, :lens[m]]
            qs[m][:, leg_idx["r"]] = out[2 * m + 1, :lens[m]]
        ik_iters += its[0::2] + its[1::2]
        ik_fail += fails[0::2] + fails[1::2]
        tm["ik"] += time.perf_counter() - t1

        t1 = time.perf_counter()
        poses_all = np.split(tree.fk_batch(np.concatenate(qs), np.concatenate(bases)),
                             np.cumsum(lens)[:-1])
        tm["fk_batch"] += time.perf_counter() - t1
        coms = []
        for m in range(m_cnt):
            com = tree.com(poses_all[m])
            err = com_refs[m] - com[:, :2]
            com_hist[m].append(float(np.abs(err).max()))
            offsets[m] += err
            zcs[m] = float(com[:, 2].mean())
            coms.append(com)

    t1 = time.perf_counter()
    results = []
    for m, (p, pl) in enumerate(zip(params, plans)):
        poses = poses_all[m]
        zmp = multibody_zmp(tree.link_coms(poses), tree.masses, p.dt)
        lc = foot_corners(poses[:, tree.link_id("l_sole")], d.foot_length, d.foot_width)
        rc = foot_corners(poses[:, tree.link_id("r_sole")], d.foot_length, d.foot_width)
        polys = support_polygons(pl.phase, lc, rc)
        margin = np.array([signed_margin(zmp[k], polys[k]) for k in range(lens[m])])
        results.append(WalkResult(pl.t, qs[m], bases[m], poses, pl.phase,
                                  {"l": pl.left, "r": pl.right}, com_refs[m], coms[m], pl.zmp_ref,
                                  zmp_carts[m], zmp, polys, margin, com_hist[m],
                                  int(ik_fail[m]), int(ik_iters[m]), tm))
    tm["balance"] = time.perf_counter() - t1
    return results
