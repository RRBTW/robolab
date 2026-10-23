"""Reference kernel in pure NumPy.

This module is the *specification* of the numeric kernel. The Cython kernel
(``_core_cy.pyx``) must implement exactly the same functions with the same
semantics, and the test-suite runs every check against both.

Kernel functions are UNCHECKED: they assume float64 arrays of the right shape
and valid group elements. Validation lives one layer up (classes and the
functional API). Never call the kernel directly on untrusted data.

Conventions
-----------
* Column vectors, right-handed frames, active rotations, radians, metres.
* Twist / se(3) coordinates are ordered xi = (omega, v), as in Lynch & Park,
  "Modern Robotics": omega is the rotational part, v the translational part.
* so3_log returns the principal value: |omega| in [0, pi].
"""
from __future__ import annotations

import numpy as np

# Below this angle closed forms lose precision; use Taylor series instead.
SMALL_ANGLE = 1e-3
# Within this distance of pi the antisymmetric part of R vanishes and the
# rotation axis is recovered from the symmetric part instead.
NEAR_PI = 1e-3

_I3 = np.eye(3)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def hat3(w):
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _coeffs(theta):
    """A = sin t / t, B = (1 - cos t) / t^2, C = (t - sin t) / t^3."""
    if theta < SMALL_ANGLE:
        t2 = theta * theta
        t4 = t2 * t2
        return (1.0 - t2 / 6.0 + t4 / 120.0,
                0.5 - t2 / 24.0 + t4 / 720.0,
                1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0)
    s = np.sin(theta)
    one_minus_c = 2.0 * np.sin(0.5 * theta) ** 2   # stable 1 - cos(theta)
    return s / theta, one_minus_c / (theta * theta), (theta - s) / (theta ** 3)


def _d_coeff(theta):
    """D = (1 - A / (2B)) / t^2, the K^2 coefficient of V^-1."""
    if theta < SMALL_ANGLE:
        t2 = theta * theta
        return 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    # theta*sin/(2(1-cos)) == (theta/2) / tan(theta/2), no cancellation
    half = 0.5 * theta
    return (1.0 - half / np.tan(half)) / (theta * theta)


# --------------------------------------------------------------------------- #
# SO(3)
# --------------------------------------------------------------------------- #
def so3_exp(w):
    theta = float(np.sqrt(w @ w))
    a, b, _ = _coeffs(theta)
    k = hat3(w)
    return _I3 + a * k + b * (k @ k)


def so3_log(r):
    v = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    sin_t = float(np.sqrt(v @ v))
    cos_t = min(1.0, max(-1.0, 0.5 * (r[0, 0] + r[1, 1] + r[2, 2] - 1.0)))
    theta = float(np.arctan2(sin_t, cos_t))

    if theta < SMALL_ANGLE:                       # theta / sin(theta) series
        t2 = theta * theta
        return v * (1.0 + t2 / 6.0 + 7.0 * t2 * t2 / 360.0)

    if np.pi - theta < NEAR_PI:
        # Symmetric part: (R + R^T)/2 = cos(t) I + (1 - cos(t)) a a^T
        m = (0.5 * (r + r.T) - cos_t * _I3) / (1.0 - cos_t)
        k = int(np.argmax(np.diag(m)))
        axis = m[k] / np.sqrt(max(m[k, k], 0.0))
        axis /= np.sqrt(axis @ axis)
        if axis @ v < 0.0:                        # at exactly pi: ambiguous
            axis = -axis
        return theta * axis

    return (theta / sin_t) * v


def so3_mul(a, b):
    return a @ b


def so3_act(r, pts):
    """Rotate points; pts has shape (M, 3)."""
    return pts @ r.T


def so3_check(r, tol):
    if not np.all(np.isfinite(r)):
        return False
    if np.max(np.abs(r.T @ r - _I3)) > tol:
        return False
    return abs(np.linalg.det(r) - 1.0) <= tol


# --------------------------------------------------------------------------- #
# SE(3), single element
# --------------------------------------------------------------------------- #
def se3_exp(xi):
    w, v = xi[:3], xi[3:]
    theta = float(np.sqrt(w @ w))
    a, b, c = _coeffs(theta)
    k = hat3(w)
    kk = k @ k
    t = np.eye(4)
    t[:3, :3] = _I3 + a * k + b * kk
    t[:3, 3] = (_I3 + b * k + c * kk) @ v
    return t


def se3_log(t):
    w = so3_log(t[:3, :3])
    theta = float(np.sqrt(w @ w))
    k = hat3(w)
    v_inv = _I3 - 0.5 * k + _d_coeff(theta) * (k @ k)
    return np.concatenate([w, v_inv @ t[:3, 3]])


def se3_inv(t):
    out = np.eye(4)
    rt = t[:3, :3].T
    out[:3, :3] = rt
    out[:3, 3] = -rt @ t[:3, 3]
    return out


def se3_mul(a, b):
    out = np.eye(4)
    out[:3, :3] = a[:3, :3] @ b[:3, :3]
    out[:3, 3] = a[:3, :3] @ b[:3, 3] + a[:3, 3]
    return out


def se3_act(t, pts):
    """Transform points; pts has shape (M, 3)."""
    return pts @ t[:3, :3].T + t[:3, 3]


def se3_adjoint(t):
    """6x6 adjoint for xi = (omega, v): [[R, 0], [p^ R, R]]."""
    r, p = t[:3, :3], t[:3, 3]
    ad = np.zeros((6, 6))
    ad[:3, :3] = r
    ad[3:, 3:] = r
    ad[3:, :3] = hat3(p) @ r
    return ad


def se3_check(t, tol):
    if not np.all(np.isfinite(t)):
        return False
    if np.max(np.abs(t[3] - (0.0, 0.0, 0.0, 1.0))) > tol:
        return False
    return so3_check(t[:3, :3], tol)


# --------------------------------------------------------------------------- #
# SE(3), batches. Shapes: poses (N, 4, 4), twists (N, 6), points (M, 3).
# Binary ops broadcast only 1<->N or N<->N.
# --------------------------------------------------------------------------- #
def _bcast(na, nb):
    if na == nb or na == 1 or nb == 1:
        return max(na, nb)
    raise ValueError(f"cannot broadcast batches of length {na} and {nb}")


def se3_mul_batch(a, b):
    n = _bcast(a.shape[0], b.shape[0])
    out = np.zeros((n, 4, 4))
    out[:, :3, :3] = a[:, :3, :3] @ b[:, :3, :3]
    out[:, :3, 3] = np.einsum("nij,nj->ni", np.broadcast_to(a[:, :3, :3], (n, 3, 3)),
                              np.broadcast_to(b[:, :3, 3], (n, 3))) + a[:, :3, 3]
    out[:, 3, 3] = 1.0
    return out


def se3_inv_batch(t):
    out = np.zeros_like(t)
    rt = np.transpose(t[:, :3, :3], (0, 2, 1))
    out[:, :3, :3] = rt
    out[:, :3, 3] = -np.einsum("nij,nj->ni", rt, t[:, :3, 3])
    out[:, 3, 3] = 1.0
    return out


def se3_exp_batch(xi):
    return np.stack([se3_exp(x) for x in xi]) if len(xi) else np.zeros((0, 4, 4))


def se3_log_batch(t):
    return np.stack([se3_log(x) for x in t]) if len(t) else np.zeros((0, 6))


def se3_act_batch(t, pts):
    """Every pose applied to every point: (N,4,4) x (M,3) -> (N, M, 3)."""
    return np.einsum("nij,mj->nmi", t[:, :3, :3], pts) + t[:, None, :3, 3]


def se3_act_each(t, pts):
    """i-th pose applied to i-th point: (N,4,4) x (N,3) -> (N, 3)."""
    if pts.shape[0] != t.shape[0]:
        raise ValueError("need exactly one point per pose")
    return np.einsum("nij,nj->ni", t[:, :3, :3], pts) + t[:, :3, 3]


def se3_check_batch(t, tol):
    return all(se3_check(x, tol) for x in t)


# --------------------------------------------------------------------------- #
# Kinematic tree (reference implementation, vectorised over configurations)
# --------------------------------------------------------------------------- #
def _joint_matrices(jtype, axis, qv):
    """(N, 4, 4) joint motions for one joint over N configurations."""
    n = qv.shape[0]
    j = np.tile(np.eye(4), (n, 1, 1))
    if jtype == 1:
        k = hat3(axis)
        s, c = np.sin(qv)[:, None, None], np.cos(qv)[:, None, None]
        j[:, :3, :3] = _I3 + s * k + (1.0 - c) * (k @ k)
    elif jtype == 2:
        j[:, :3, 3] = qv[:, None] * axis
    return j


def fk_tree_batch(parent, offset, axis, qidx, jtype, q, base):
    n, nl = q.shape[0], parent.shape[0]
    if base.shape[0] not in (1, n):
        raise ValueError("base must have 1 or N poses")
    if q.shape[1] < 1:
        raise ValueError("q needs at least one column (pad with zeros if the tree is fixed)")
    out = np.empty((n, nl, 4, 4))
    base = np.broadcast_to(base, (n, 4, 4))
    for l in range(nl):
        tp = base if parent[l] < 0 else out[:, parent[l]]
        qv = q[:, qidx[l]] if qidx[l] >= 0 else np.zeros(n)
        out[:, l] = (tp @ offset[l]) @ _joint_matrices(jtype[l], axis[l], qv)
    return out


def tree_path(parent, link):
    """Link indices from the root down to ``link`` (inclusive)."""
    path = []
    while link >= 0:
        path.append(int(link))
        link = parent[link]
    return path[::-1]


def space_jacobian(poses, axis, qidx, jtype, path, nq):
    """(6, nq) space Jacobian, twist order (omega, v) at the world origin, of the
    joints on ``path``; columns of all other joints are zero."""
    jac = np.zeros((6, nq))
    chain = np.array([l for l in path if qidx[l] >= 0], dtype=int)
    if len(chain) == 0:
        return jac
    z = np.einsum("kij,kj->ki", poses[chain, :3, :3], axis[chain])
    cols, rev = qidx[chain], jtype[chain] == 1
    jac[:3, cols[rev]] = z[rev].T
    jac[3:, cols[rev]] = np.cross(poses[chain, :3, 3][rev], z[rev]).T
    jac[3:, cols[~rev]] = z[~rev].T
    return jac


# --------------------------------------------------------------------------- #
# Inverse kinematics: damped least squares on SE(3)
# --------------------------------------------------------------------------- #
def ik_dls(parent, offset, axis, qidx, jtype, q0, base, link, target, sel, lower, upper,
           iters, tol, damping):
    """Move the joints ``sel`` so that the pose of ``link`` reaches ``target``.

    q0 (nq,), base and target (4, 4), sel (ns,) indices into q, lower/upper (nq,).
    Iteration: xi = log(T_link^-1 target) (body twist error); stop if |xi| < tol;
    J_b = Ad(T_link^-1) J_space restricted to sel;
    dq = J_b^T (J_b J_b^T + damping^2 I)^-1 xi; q[sel] = clip(q[sel] + dq).
    Returns (q, converged, iterations, err): iterations = updates applied,
    err = |xi| at the last check."""
    q = np.array(q0, dtype=np.float64)
    path = tree_path(parent, link)
    lam2 = damping * damping
    err = np.inf
    for it in range(1, iters + 1):
        poses = fk_tree_batch(parent, offset, axis, qidx, jtype, q[None], base[None])[0]
        t_inv = se3_inv(poses[link])
        xi = se3_log(se3_mul(t_inv, target))
        err = float(np.sqrt(xi @ xi))
        if err < tol:
            return q, True, it - 1, err
        jb = (se3_adjoint(t_inv) @ space_jacobian(poses, axis, qidx, jtype, path, q.shape[0]))[:, sel]
        dq = jb.T @ np.linalg.solve(jb @ jb.T + lam2 * np.eye(6), xi)
        q[sel] = np.clip(q[sel] + dq, lower[sel], upper[sel])
    return q, False, iters, err


def ik_dls_traj(parent, offset, axis, qidx, jtype, q0, base, link, target, sel, lower, upper,
                n_frames, iters, tol, damping):
    """P independent IK problems, each over its own trajectory of frames.

    q0 (P, nq) start joints; base, target (P, T, 4, 4); link (P,); sel (P, ns);
    n_frames (P,) <= T. Problem p, frame k < n_frames[p]:
        q = ik_dls(..., q, base[p, k], link[p], target[p, k], sel[p], ...)
    warm-started from frame k - 1; joints outside sel[p] stay at q0[p].
    Returns q_sel (P, T, ns) = q[sel[p]] after every frame (0 past n_frames),
    iterations (P,) total updates, failures (P,) frames that did not converge."""
    n_p, n_t, ns = base.shape[0], base.shape[1], sel.shape[1]
    out = np.zeros((n_p, n_t, ns))
    its = np.zeros(n_p, dtype=np.int64)
    fails = np.zeros(n_p, dtype=np.int64)
    for p in range(n_p):
        q = np.array(q0[p], dtype=np.float64)
        for k in range(n_frames[p]):
            q, ok, it, _ = ik_dls(parent, offset, axis, qidx, jtype, q, base[p, k], link[p],
                                  target[p, k], sel[p], lower, upper, iters, tol, damping)
            out[p, k] = q[sel[p]]
            its[p] += it
            fails[p] += not ok
    return out, its, fails


# --------------------------------------------------------------------------- #
# Monte-Carlo of joint perturbations (fused: only the errors leave the kernel)
# --------------------------------------------------------------------------- #
MC_MAX_LINKS = 64  # compiled kernels keep per-link scratch in a fixed-size local array


def mc_relative_pose(parent, offset, axis, qidx, jtype, q0, base, joints, dq, ref, target,
                     coms, masses):
    """Pose and CoM errors of N perturbed configurations.

    q0 (nq,) nominal joints, base (4, 4), joints (nj,) indices into q, dq (N, nj).
    Sample i: q = q0, q[joints] += dq[i]. With T = T_ref_target (``ref`` and
    ``target`` are link indices) and nominal T0, returns
        pos (N, 3): p(T) - p(T0), in the ref frame
        ang (N,)  : rotation angle of T0^-1 T, in [0, pi]
        com (N, 3): whole-body CoM minus nominal CoM, base frame.
    Here: the definition, via full FK. Compiled kernels use the exact
    perturbation form (see mc_prepare), which keeps float32 accurate."""
    n = dq.shape[0]
    q = np.tile(np.asarray(q0, dtype=np.float64), (n + 1, 1))   # row 0 = nominal
    q[1:, joints] += dq
    poses = fk_tree_batch(parent, offset, axis, qidx, jtype, q, base[None])
    rel = se3_mul_batch(se3_inv_batch(poses[:, ref]), poses[:, target])
    pos = rel[1:, :3, 3] - rel[0, :3, 3]
    d = np.einsum("ji,njk->nik", rel[0, :3, :3], rel[1:, :3, :3])
    v = 0.5 * np.stack([d[:, 2, 1] - d[:, 1, 2], d[:, 0, 2] - d[:, 2, 0], d[:, 1, 0] - d[:, 0, 1]], 1)
    cos_t = np.clip(0.5 * (np.trace(d, axis1=1, axis2=2) - 1.0), -1.0, 1.0)
    ang = np.arctan2(np.sqrt(np.einsum("ni,ni->n", v, v)), cos_t)
    lc = np.einsum("nlij,lj->nli", poses[:, :, :3, :3], coms) + poses[:, :, :3, 3]
    c = np.einsum("nli,l->ni", lc, masses) / np.sum(masses)
    return pos, ang, c[1:] - c[0]


def mc_prepare(parent, offset, axis, qidx, jtype, q0, base, joints, ref, target, coms, masses):
    """Constant part of the perturbation form of ``mc_relative_pose`` (float64).

    Link pose T_l = N_l (I + E_l): N_l nominal, E_l an exact small correction.
    With A_l = offset_l J_l(q0_l) (so N_l = N_parent A_l) and J(q0 + d) = J(q0) J(d):
        I + E_l = A_l^-1 (I + E_parent) A_l J_l(d_l)
    Every E stays small, so float32 keeps ~1e-7 of the *error*, not of the pose.
    """
    nl = parent.shape[0]
    if nl > MC_MAX_LINKS:
        raise ValueError(f"mc_relative_pose supports trees of up to {MC_MAX_LINKS} links, got {nl}")
    a = np.empty((nl, 4, 4))
    nom = np.empty((nl, 4, 4))
    col = {int(j): k for k, j in enumerate(joints)}
    dqidx = np.full(nl, -1, dtype=np.int32)
    for l in range(nl):
        qv = float(q0[qidx[l]]) if qidx[l] >= 0 else 0.0
        a[l] = offset[l] @ _joint_matrices(jtype[l], axis[l], np.array([qv]))[0]
        nom[l] = (base if parent[l] < 0 else nom[parent[l]]) @ a[l]
        if qidx[l] >= 0 and jtype[l] != 0:
            dqidx[l] = col.get(int(qidx[l]), -1)
    return {"a": a, "nom": nom, "dqidx": dqidx,
            "relnom": se3_inv(nom[ref]) @ nom[target],
            "w": np.asarray(masses, dtype=np.float64) / np.sum(masses)}


def openmp_enabled():
    return False


def set_num_threads(n):
    pass


def get_num_threads():
    return 1


def set_parallel_min(n):
    pass
