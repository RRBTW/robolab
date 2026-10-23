# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False, nonecheck=False
"""Cython kernel. Same functions and semantics as ``_core_py`` (the spec).

All arithmetic is done on raw row-major buffers without the GIL. Batch
kernels are parallelised with OpenMP (``prange``) when the extension was
compiled with OpenMP; batches smaller than PARALLEL_MIN stay single-threaded
because thread start-up would cost more than it saves.
Kernel functions are UNCHECKED: callers must pass float64, C-contiguous
arrays of the right shape that hold valid group elements.
"""
import os

import numpy as np
from cython.parallel cimport prange

from . import _core_py as _spec
from libc.math cimport sin, cos, tan, sqrt, atan2, fabs, isfinite, M_PI

cdef extern from *:
    """
    #ifdef _OPENMP
    #define ROBOLAB_HAS_OPENMP 1
    #else
    #define ROBOLAB_HAS_OPENMP 0
    #endif
    """
    int ROBOLAB_HAS_OPENMP

cdef int _n_threads = (os.cpu_count() or 1) if ROBOLAB_HAS_OPENMP else 1
cdef Py_ssize_t _par_min = 256


def openmp_enabled():
    """True if the kernel was compiled with OpenMP."""
    return bool(ROBOLAB_HAS_OPENMP)


def set_num_threads(int n):
    """Threads for batch kernels (1 = serial). Ignored without OpenMP."""
    global _n_threads
    _n_threads = max(1, n) if ROBOLAB_HAS_OPENMP else 1


def get_num_threads():
    return _n_threads


def set_parallel_min(Py_ssize_t n):
    """Smallest batch that is processed in parallel."""
    global _par_min
    _par_min = max(1, n)

SMALL_ANGLE = 1e-3
NEAR_PI = 1e-3
cdef double _SMALL = 1e-3
cdef double _NEAR_PI = 1e-3


# --------------------------------------------------------------------------- #
# raw-buffer primitives (row-major; ld = leading dimension: 3 or 4)
# --------------------------------------------------------------------------- #
cdef inline void _coeffs(double t, double* a, double* b, double* c) noexcept nogil:
    cdef double t2, t4, s, h
    if t < _SMALL:
        t2 = t * t
        t4 = t2 * t2
        a[0] = 1.0 - t2 / 6.0 + t4 / 120.0
        b[0] = 0.5 - t2 / 24.0 + t4 / 720.0
        c[0] = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0
    else:
        s = sin(t)
        h = sin(0.5 * t)
        a[0] = s / t
        b[0] = 2.0 * h * h / (t * t)
        c[0] = (t - s) / (t * t * t)


cdef inline double _d_coeff(double t) noexcept nogil:
    cdef double t2, half
    if t < _SMALL:
        t2 = t * t
        return 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    half = 0.5 * t
    return (1.0 - half / tan(half)) / (t * t)


cdef inline void _hat(const double* w, double* k) noexcept nogil:
    k[0] = 0.0;   k[1] = -w[2]; k[2] = w[1]
    k[3] = w[2];  k[4] = 0.0;   k[5] = -w[0]
    k[6] = -w[1]; k[7] = w[0];  k[8] = 0.0


cdef inline void _poly(const double* w, double t2, double c0, double c1,
                       double c2, double* out, int ld) noexcept nogil:
    """out = c0*I + c1*K + c2*K^2 using K^2 = w w^T - t2 I."""
    cdef double k[9]
    cdef int i, j
    _hat(w, k)
    for i in range(3):
        for j in range(3):
            out[i * ld + j] = c1 * k[i * 3 + j] + c2 * w[i] * w[j]
        out[i * ld + i] += c0 - c2 * t2


cdef inline void _so3_exp(const double* w, double* r, int ld) noexcept nogil:
    cdef double t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    cdef double a, b, c
    _coeffs(sqrt(t2), &a, &b, &c)
    _poly(w, t2, 1.0, a, b, r, ld)


cdef inline void _so3_log(const double* r, int ld, double* w) noexcept nogil:
    cdef double v[3]
    cdef double m[9]
    cdef double sin_t, cos_t, theta, t2, f, n, s
    cdef int i, j, k
    v[0] = 0.5 * (r[2 * ld + 1] - r[1 * ld + 2])
    v[1] = 0.5 * (r[0 * ld + 2] - r[2 * ld + 0])
    v[2] = 0.5 * (r[1 * ld + 0] - r[0 * ld + 1])
    sin_t = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    cos_t = 0.5 * (r[0] + r[ld + 1] + r[2 * ld + 2] - 1.0)
    if cos_t > 1.0:
        cos_t = 1.0
    elif cos_t < -1.0:
        cos_t = -1.0
    theta = atan2(sin_t, cos_t)

    if theta < _SMALL:
        t2 = theta * theta
        f = 1.0 + t2 / 6.0 + 7.0 * t2 * t2 / 360.0
        for i in range(3):
            w[i] = f * v[i]
        return

    if M_PI - theta < _NEAR_PI:
        for i in range(3):
            for j in range(3):
                m[i * 3 + j] = 0.5 * (r[i * ld + j] + r[j * ld + i])
            m[i * 3 + i] -= cos_t
        for i in range(9):
            m[i] /= (1.0 - cos_t)
        k = 0
        if m[4] > m[k * 3 + k]:
            k = 1
        if m[8] > m[k * 3 + k]:
            k = 2
        s = m[k * 3 + k]
        if s < 0.0:
            s = 0.0
        s = sqrt(s)
        for i in range(3):
            w[i] = m[k * 3 + i] / s
        n = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if w[0] * v[0] + w[1] * v[1] + w[2] * v[2] < 0.0:
            n = -n
        for i in range(3):
            w[i] = theta * w[i] / n
        return

    f = theta / sin_t
    for i in range(3):
        w[i] = f * v[i]


cdef inline void _se3_exp(const double* xi, double* t) noexcept nogil:
    cdef const double* w = xi
    cdef const double* v = xi + 3
    cdef double t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    cdef double a, b, c
    cdef double vm[9]
    cdef int i
    _coeffs(sqrt(t2), &a, &b, &c)
    _poly(w, t2, 1.0, a, b, t, 4)
    _poly(w, t2, 1.0, b, c, vm, 3)
    for i in range(3):
        t[i * 4 + 3] = vm[i * 3] * v[0] + vm[i * 3 + 1] * v[1] + vm[i * 3 + 2] * v[2]
    t[12] = 0.0; t[13] = 0.0; t[14] = 0.0; t[15] = 1.0


cdef inline void _se3_log(const double* t, double* xi) noexcept nogil:
    cdef double vinv[9]
    cdef double t2
    cdef int i
    _so3_log(t, 4, xi)
    t2 = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
    _poly(xi, t2, 1.0, -0.5, _d_coeff(sqrt(t2)), vinv, 3)
    for i in range(3):
        xi[3 + i] = vinv[i * 3] * t[3] + vinv[i * 3 + 1] * t[7] + vinv[i * 3 + 2] * t[11]


cdef inline void _se3_inv(const double* t, double* o) noexcept nogil:
    cdef int i, j
    for i in range(3):
        for j in range(3):
            o[i * 4 + j] = t[j * 4 + i]
    for i in range(3):
        o[i * 4 + 3] = -(t[i] * t[3] + t[4 + i] * t[7] + t[8 + i] * t[11])
    o[12] = 0.0; o[13] = 0.0; o[14] = 0.0; o[15] = 1.0


cdef inline void _se3_mul(const double* a, const double* b, double* o) noexcept nogil:
    cdef int i, j
    for i in range(3):
        for j in range(3):
            o[i * 4 + j] = (a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j]
                            + a[i * 4 + 2] * b[8 + j])
        o[i * 4 + 3] = (a[i * 4] * b[3] + a[i * 4 + 1] * b[7]
                        + a[i * 4 + 2] * b[11] + a[i * 4 + 3])
    o[12] = 0.0; o[13] = 0.0; o[14] = 0.0; o[15] = 1.0


cdef inline void _act(const double* r, int ld, const double* p,
                      const double* x, double* y) noexcept nogil:
    """y = R x (+ p if p != NULL)."""
    cdef int i
    for i in range(3):
        y[i] = r[i * ld] * x[0] + r[i * ld + 1] * x[1] + r[i * ld + 2] * x[2]
        if p != NULL:
            y[i] += p[i * ld]


cdef inline bint _so3_check(const double* r, int ld, double tol) noexcept nogil:
    cdef int i, j
    cdef double g, det
    for i in range(3):
        for j in range(3):
            if not isfinite(r[i * ld + j]):
                return False
    for i in range(3):
        for j in range(3):
            g = r[i] * r[j] + r[ld + i] * r[ld + j] + r[2 * ld + i] * r[2 * ld + j]
            if i == j:
                g -= 1.0
            if fabs(g) > tol:
                return False
    det = (r[0] * (r[ld + 1] * r[2 * ld + 2] - r[ld + 2] * r[2 * ld + 1])
           - r[1] * (r[ld] * r[2 * ld + 2] - r[ld + 2] * r[2 * ld])
           + r[2] * (r[ld] * r[2 * ld + 1] - r[ld + 1] * r[2 * ld]))
    return fabs(det - 1.0) <= tol


cdef inline bint _se3_check(const double* t, double tol) noexcept nogil:
    cdef int i
    for i in range(16):
        if not isfinite(t[i]):
            return False
    if fabs(t[12]) > tol or fabs(t[13]) > tol or fabs(t[14]) > tol:
        return False
    if fabs(t[15] - 1.0) > tol:
        return False
    return _so3_check(t, 4, tol)


# --------------------------------------------------------------------------- #
# Python entry points: single elements
# --------------------------------------------------------------------------- #
def hat3(const double[::1] w):
    out = np.empty((3, 3))
    cdef double[:, ::1] o = out
    _hat(&w[0], &o[0, 0])
    return out


def so3_exp(const double[::1] w):
    out = np.empty((3, 3))
    cdef double[:, ::1] o = out
    with nogil:
        _so3_exp(&w[0], &o[0, 0], 3)
    return out


def so3_log(const double[:, ::1] r):
    out = np.empty(3)
    cdef double[::1] o = out
    with nogil:
        _so3_log(&r[0, 0], 3, &o[0])
    return out


def so3_mul(const double[:, ::1] a, const double[:, ::1] b):
    out = np.empty((3, 3))
    cdef double[:, ::1] o = out
    cdef int i, j
    with nogil:
        for i in range(3):
            for j in range(3):
                o[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]
    return out


def so3_act(const double[:, ::1] r, const double[:, ::1] pts):
    cdef Py_ssize_t m = pts.shape[0], i
    out = np.empty((m, 3))
    cdef double[:, ::1] o = out
    if m:
        for i in prange(m, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=m >= 64 * _par_min):
            _act(&r[0, 0], 3, NULL, &pts[i, 0], &o[i, 0])
    return out


def so3_check(const double[:, ::1] r, double tol):
    return bool(_so3_check(&r[0, 0], 3, tol))


def se3_exp(const double[::1] xi):
    out = np.empty((4, 4))
    cdef double[:, ::1] o = out
    with nogil:
        _se3_exp(&xi[0], &o[0, 0])
    return out


def se3_log(const double[:, ::1] t):
    out = np.empty(6)
    cdef double[::1] o = out
    with nogil:
        _se3_log(&t[0, 0], &o[0])
    return out


def se3_inv(const double[:, ::1] t):
    out = np.empty((4, 4))
    cdef double[:, ::1] o = out
    with nogil:
        _se3_inv(&t[0, 0], &o[0, 0])
    return out


def se3_mul(const double[:, ::1] a, const double[:, ::1] b):
    out = np.empty((4, 4))
    cdef double[:, ::1] o = out
    with nogil:
        _se3_mul(&a[0, 0], &b[0, 0], &o[0, 0])
    return out


def se3_act(const double[:, ::1] t, const double[:, ::1] pts):
    cdef Py_ssize_t m = pts.shape[0], i
    out = np.empty((m, 3))
    cdef double[:, ::1] o = out
    if m:
        # a point transform is ~10 flops: only very large clouds pay for threads
        for i in prange(m, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=m >= 64 * _par_min):
            _act(&t[0, 0], 4, &t[0, 3], &pts[i, 0], &o[i, 0])
    return out


def se3_adjoint(const double[:, ::1] t):
    out = np.zeros((6, 6))
    cdef double[:, ::1] o = out
    cdef double ph[9]
    cdef double p[3]
    cdef int i, j
    p[0] = t[0, 3]; p[1] = t[1, 3]; p[2] = t[2, 3]
    _hat(p, ph)
    for i in range(3):
        for j in range(3):
            o[i, j] = t[i, j]
            o[3 + i, 3 + j] = t[i, j]
            o[3 + i, j] = (ph[i * 3] * t[0, j] + ph[i * 3 + 1] * t[1, j]
                           + ph[i * 3 + 2] * t[2, j])
    return out


def se3_check(const double[:, ::1] t, double tol):
    return bool(_se3_check(&t[0, 0], tol))


# --------------------------------------------------------------------------- #
# Python entry points: batches (N,4,4) / (N,6) / (M,3), OpenMP-parallel
# --------------------------------------------------------------------------- #
def se3_mul_batch(const double[:, :, ::1] a, const double[:, :, ::1] b):
    cdef Py_ssize_t na = a.shape[0], nb = b.shape[0], n, i
    if not (na == nb or na == 1 or nb == 1):
        raise ValueError(f"cannot broadcast batches of length {na} and {nb}")
    n = max(na, nb)
    out = np.empty((n, 4, 4))
    cdef double[:, :, ::1] o = out
    cdef Py_ssize_t sa = 1 if na > 1 else 0, sb = 1 if nb > 1 else 0
    if n:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= _par_min):
            _se3_mul(&a[i * sa, 0, 0], &b[i * sb, 0, 0], &o[i, 0, 0])
    return out


def se3_inv_batch(const double[:, :, ::1] t):
    cdef Py_ssize_t n = t.shape[0], i
    out = np.empty((n, 4, 4))
    cdef double[:, :, ::1] o = out
    if n:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= _par_min):
            _se3_inv(&t[i, 0, 0], &o[i, 0, 0])
    return out


def se3_exp_batch(const double[:, ::1] xi):
    cdef Py_ssize_t n = xi.shape[0], i
    out = np.empty((n, 4, 4))
    cdef double[:, :, ::1] o = out
    if n:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= _par_min):
            _se3_exp(&xi[i, 0], &o[i, 0, 0])
    return out


def se3_log_batch(const double[:, :, ::1] t):
    cdef Py_ssize_t n = t.shape[0], i
    out = np.empty((n, 6))
    cdef double[:, ::1] o = out
    if n:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= _par_min):
            _se3_log(&t[i, 0, 0], &o[i, 0])
    return out


def se3_act_batch(const double[:, :, ::1] t, const double[:, ::1] pts):
    cdef Py_ssize_t n = t.shape[0], m = pts.shape[0], k, i, j
    out = np.empty((n, m, 3))
    cdef double[:, :, ::1] o = out
    if n and m:
        for k in prange(n * m, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n * m >= 64 * _par_min):
            i = k // m
            j = k - i * m
            _act(&t[i, 0, 0], 4, &t[i, 0, 3], &pts[j, 0], &o[i, j, 0])
    return out


def se3_act_each(const double[:, :, ::1] t, const double[:, ::1] pts):
    cdef Py_ssize_t n = t.shape[0], i
    if pts.shape[0] != n:
        raise ValueError("need exactly one point per pose")
    out = np.empty((n, 3))
    cdef double[:, ::1] o = out
    if n:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= 64 * _par_min):
            _act(&t[i, 0, 0], 4, &t[i, 0, 3], &pts[i, 0], &o[i, 0])
    return out


def se3_check_batch(const double[:, :, ::1] t, double tol):
    cdef Py_ssize_t n = t.shape[0], i
    cdef Py_ssize_t bad = 0
    if n:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= _par_min):
            if not _se3_check(&t[i, 0, 0], tol):
                bad += 1
    return bad == 0


# --------------------------------------------------------------------------- #
# Kinematic tree: forward kinematics of a whole robot, many configurations
# --------------------------------------------------------------------------- #
cdef inline void _joint_motion(int jtype, const double* axis, double q,
                               const double* m, double* o) noexcept nogil:
    """o = m @ J(q): J = rotation about unit axis (1), translation along
    axis (2) or identity (0). m and o are 4x4, must not alias."""
    cdef double r[9]
    cdef double s, c, v, x, y, z
    cdef int i, j
    if jtype == 1:
        s = sin(q)
        c = cos(q)
        v = 1.0 - c
        x = axis[0]; y = axis[1]; z = axis[2]
        r[0] = c + x * x * v;     r[1] = x * y * v - z * s; r[2] = x * z * v + y * s
        r[3] = y * x * v + z * s; r[4] = c + y * y * v;     r[5] = y * z * v - x * s
        r[6] = z * x * v - y * s; r[7] = z * y * v + x * s; r[8] = c + z * z * v
        for i in range(3):
            for j in range(3):
                o[i * 4 + j] = m[i * 4] * r[j] + m[i * 4 + 1] * r[3 + j] + m[i * 4 + 2] * r[6 + j]
            o[i * 4 + 3] = m[i * 4 + 3]
    elif jtype == 2:
        for i in range(3):
            for j in range(3):
                o[i * 4 + j] = m[i * 4 + j]
            o[i * 4 + 3] = m[i * 4 + 3] + q * (m[i * 4] * axis[0] + m[i * 4 + 1] * axis[1]
                                               + m[i * 4 + 2] * axis[2])
    else:
        for i in range(12):
            o[i] = m[i]
    o[12] = 0.0; o[13] = 0.0; o[14] = 0.0; o[15] = 1.0


cdef inline void _fk_one(Py_ssize_t nl, const int* parent, const double* offset,
                         const double* axis, const int* qidx, const int* jtype,
                         const double* q, const double* base, double* out) noexcept nogil:
    cdef double m[16]
    cdef Py_ssize_t l
    cdef const double* tp
    cdef double qv
    for l in range(nl):
        tp = base if parent[l] < 0 else out + 16 * parent[l]
        _se3_mul(tp, offset + 16 * l, m)
        qv = q[qidx[l]] if qidx[l] >= 0 else 0.0
        _joint_motion(jtype[l], axis + 3 * l, qv, m, out + 16 * l)


def fk_tree_batch(const int[::1] parent, const double[:, :, ::1] offset,
                  const double[:, ::1] axis, const int[::1] qidx, const int[::1] jtype,
                  const double[:, ::1] q, const double[:, :, ::1] base):
    """World poses of all links for N configurations.

    parent[l] < l (topological order, -1 = attached to base), offset[l] =
    T_parent_joint, axis[l] unit joint axis in the joint frame, qidx[l] index
    into q (-1 for fixed), jtype[l] 0 fixed / 1 revolute / 2 prismatic.
    q (N, nq) with nq >= 1, base (1 or N, 4, 4). Returns (N, L, 4, 4)."""
    cdef Py_ssize_t n = q.shape[0], nl = parent.shape[0], nb = base.shape[0], i
    if not (nb == 1 or nb == n):
        raise ValueError("base must have 1 or N poses")
    if q.shape[1] < 1:
        raise ValueError("q needs at least one column (pad with zeros if the tree is fixed)")
    out = np.empty((n, nl, 4, 4))
    cdef double[:, :, :, ::1] o = out
    cdef Py_ssize_t sb = 1 if nb > 1 else 0
    if n and nl:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= 32):
            _fk_one(nl, &parent[0], &offset[0, 0, 0], &axis[0, 0], &qidx[0], &jtype[0],
                    &q[i, 0], &base[i * sb, 0, 0], &o[i, 0, 0, 0])
    return out


# --------------------------------------------------------------------------- #
# Inverse kinematics: damped least squares on SE(3), the whole loop without the GIL
# --------------------------------------------------------------------------- #
cdef inline void _cholesky_solve6(double* a, double* b) noexcept nogil:
    """b <- A^-1 b for symmetric positive definite 6x6 A (row-major, overwritten).
    A = J J^T + damping^2 I is SPD by construction."""
    cdef int i, j, k
    cdef double s
    for j in range(6):
        s = a[j * 6 + j]
        for k in range(j):
            s -= a[j * 6 + k] * a[j * 6 + k]
        s = sqrt(s)
        a[j * 6 + j] = s
        for i in range(j + 1, 6):
            for k in range(j):
                a[i * 6 + j] -= a[i * 6 + k] * a[j * 6 + k]
            a[i * 6 + j] /= s
    for i in range(6):
        for k in range(i):
            b[i] -= a[i * 6 + k] * b[k]
        b[i] /= a[i * 6 + i]
    for i in range(5, -1, -1):
        for k in range(i + 1, 6):
            b[i] -= a[k * 6 + i] * b[k]
        b[i] /= a[i * 6 + i]


cdef int _ik_dls(Py_ssize_t npath, const int* path, const int* parent, const double* offset,
                 const double* axis, const int* qidx, const int* jtype, double* q,
                 const double* base, int link, const double* target, Py_ssize_t ns,
                 const int* sel, const int* slink, const double* lower, const double* upper,
                 int iters, double tol, double lam2, double* poses, double* jb,
                 double* err) noexcept nogil:
    """Returns the number of updates if converged, -1 otherwise. poses: (L,16)
    scratch (only links on the path are written), jb: (ns,6) scratch."""
    cdef double m[16]
    cdef double tinv[16]
    cdef double d[16]
    cdef double xi[6]
    cdef double a[36]
    cdef double w[3]
    cdef double v[3]
    cdef double u[3]
    cdef const double* t
    cdef const double* tl
    cdef const double* tp
    cdef const double* ax
    cdef double* col
    cdef double x
    cdef int it, i, j, l
    cdef Py_ssize_t s
    for it in range(1, iters + 1):
        for i in range(npath):
            l = path[i]
            if parent[l] < 0:
                tp = base
            else:
                tp = poses + 16 * parent[l]
            _se3_mul(tp, offset + 16 * l, m)
            x = q[qidx[l]] if qidx[l] >= 0 else 0.0
            _joint_motion(jtype[l], axis + 3 * l, x, m, poses + 16 * l)
        t = poses + 16 * link
        _se3_inv(t, tinv)
        _se3_mul(tinv, target, d)
        _se3_log(d, xi)
        err[0] = sqrt(xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
                      + xi[3] * xi[3] + xi[4] * xi[4] + xi[5] * xi[5])
        if err[0] < tol:
            return it - 1
        # body Jacobian column by column: Ad(T^-1) (w; v) = (R^T w; R^T (v - p x w))
        for s in range(ns):
            col = jb + 6 * s
            l = slink[s]
            if l < 0:
                for i in range(6):
                    col[i] = 0.0
                continue
            tl = poses + 16 * l
            ax = axis + 3 * l
            for i in range(3):
                u[i] = tl[i * 4] * ax[0] + tl[i * 4 + 1] * ax[1] + tl[i * 4 + 2] * ax[2]
            if jtype[l] == 1:
                w[0] = u[0]; w[1] = u[1]; w[2] = u[2]
                v[0] = tl[7] * u[2] - tl[11] * u[1]
                v[1] = tl[11] * u[0] - tl[3] * u[2]
                v[2] = tl[3] * u[1] - tl[7] * u[0]
            else:
                w[0] = 0.0; w[1] = 0.0; w[2] = 0.0
                v[0] = u[0]; v[1] = u[1]; v[2] = u[2]
            v[0] -= t[7] * w[2] - t[11] * w[1]
            v[1] -= t[11] * w[0] - t[3] * w[2]
            v[2] -= t[3] * w[1] - t[7] * w[0]
            for i in range(3):
                col[i] = t[i] * w[0] + t[4 + i] * w[1] + t[8 + i] * w[2]
                col[3 + i] = t[i] * v[0] + t[4 + i] * v[1] + t[8 + i] * v[2]
        for i in range(6):
            for j in range(i + 1):
                x = 0.0
                for s in range(ns):
                    x += jb[6 * s + i] * jb[6 * s + j]
                a[i * 6 + j] = x
                a[j * 6 + i] = x
            a[i * 6 + i] += lam2
        _cholesky_solve6(a, xi)
        for s in range(ns):
            x = 0.0
            for i in range(6):
                x += jb[6 * s + i] * xi[i]
            j = sel[s]
            x += q[j]
            if x < lower[j]:
                x = lower[j]
            if x > upper[j]:
                x = upper[j]
            q[j] = x
    return -1


def ik_dls(parent, offset, axis, qidx, jtype, q0, base, int link, target, sel, lower, upper,
           int iters, double tol, double damping):
    """DLS inverse kinematics, see _core_py.ik_dls (spec). Same result up to
    rounding: the 6x6 system is solved by Cholesky instead of LU."""
    path = np.ascontiguousarray(_spec.tree_path(parent, link), dtype=np.int32)
    on_path = {int(qidx[l]): int(l) for l in path if qidx[l] >= 0}
    sel_a = np.ascontiguousarray(sel, dtype=np.int32)
    slink_a = np.array([on_path.get(int(s), -1) for s in sel_a], dtype=np.int32)
    q = np.array(q0, dtype=np.float64)
    poses = np.empty((parent.shape[0], 16))
    jb = np.empty((max(sel_a.shape[0], 1), 6))
    cdef double err = np.inf
    cdef int res
    cdef const int[::1] pa = path
    cdef const int[::1] par = np.ascontiguousarray(parent, dtype=np.int32)
    cdef const double[:, :, ::1] off = np.ascontiguousarray(offset, dtype=np.float64)
    cdef const double[:, ::1] ax = np.ascontiguousarray(axis, dtype=np.float64)
    cdef const int[::1] qi = np.ascontiguousarray(qidx, dtype=np.int32)
    cdef const int[::1] jt = np.ascontiguousarray(jtype, dtype=np.int32)
    cdef double[::1] qv = q
    cdef const double[:, ::1] b = np.ascontiguousarray(base, dtype=np.float64)
    cdef const double[:, ::1] tg = np.ascontiguousarray(target, dtype=np.float64)
    cdef const int[::1] se = sel_a
    cdef const int[::1] sl = slink_a
    cdef const double[::1] lo = np.ascontiguousarray(lower, dtype=np.float64)
    cdef const double[::1] hi = np.ascontiguousarray(upper, dtype=np.float64)
    cdef double[:, ::1] ps = poses
    cdef double[:, ::1] jbv = jb
    with nogil:
        res = _ik_dls(pa.shape[0], &pa[0], &par[0], &off[0, 0, 0], &ax[0, 0], &qi[0], &jt[0],
                      &qv[0], &b[0, 0], link, &tg[0, 0], se.shape[0], &se[0] if se.shape[0] else NULL,
                      &sl[0] if sl.shape[0] else NULL, &lo[0], &hi[0], iters, tol,
                      damping * damping, &ps[0, 0], &jbv[0, 0], &err)
    return q, res >= 0, res if res >= 0 else iters, err


cdef void _ik_traj_one(Py_ssize_t npath, const int* path, const int* parent, const double* offset,
                       const double* axis, const int* qidx, const int* jtype, double* q,
                       const double* base, int link, const double* target, Py_ssize_t ns,
                       const int* sel, const int* slink, const double* lower,
                       const double* upper, Py_ssize_t nf, int iters, double tol, double lam2,
                       double* poses, double* jb, double* out, long long* its,
                       long long* fails) noexcept nogil:
    cdef double err
    cdef int r
    cdef Py_ssize_t k, s
    for k in range(nf):
        r = _ik_dls(npath, path, parent, offset, axis, qidx, jtype, q, base + 16 * k, link,
                    target + 16 * k, ns, sel, slink, lower, upper, iters, tol, lam2, poses, jb,
                    &err)
        if r >= 0:
            its[0] += r
        else:
            its[0] += iters
            fails[0] += 1
        for s in range(ns):
            out[k * ns + s] = q[sel[s]]


def ik_dls_traj(parent, offset, axis, qidx, jtype, q0, base, link, target, sel, lower, upper,
                n_frames, int iters, double tol, double damping):
    """Trajectory IK of P problems, see _core_py.ik_dls_traj (spec). Problems
    run in parallel (OpenMP, dynamic schedule: trajectories differ in length),
    frames of one problem in order."""
    sel_a = np.ascontiguousarray(sel, dtype=np.int32)
    cdef Py_ssize_t n_p = base.shape[0], n_t = base.shape[1], ns = sel_a.shape[1], p
    out = np.zeros((n_p, n_t, ns))
    its = np.zeros(n_p, dtype=np.int64)
    fails = np.zeros(n_p, dtype=np.int64)
    if n_p == 0 or n_t == 0 or ns == 0:
        return out, its, fails
    link_a = np.ascontiguousarray(link, dtype=np.int32)
    paths = [_spec.tree_path(parent, int(l)) for l in link_a]
    npath_a = np.array([len(x) for x in paths], dtype=np.int32)
    path_a = np.zeros((n_p, npath_a.max()), dtype=np.int32)
    slink_a = np.full((n_p, ns), -1, dtype=np.int32)
    for p in range(n_p):
        path_a[p, :npath_a[p]] = paths[p]
        on_path = {int(qidx[l]): int(l) for l in paths[p] if qidx[l] >= 0}
        slink_a[p] = [on_path.get(int(s), -1) for s in sel_a[p]]
    q = np.array(q0, dtype=np.float64, order="C")
    poses = np.empty((n_p, parent.shape[0] * 16))
    jb = np.empty((n_p, ns * 6))
    cdef const int[:, ::1] pa = path_a
    cdef const int[::1] npa = npath_a
    cdef const int[::1] par = np.ascontiguousarray(parent, dtype=np.int32)
    cdef const double[:, :, ::1] off = np.ascontiguousarray(offset, dtype=np.float64)
    cdef const double[:, ::1] ax = np.ascontiguousarray(axis, dtype=np.float64)
    cdef const int[::1] qi = np.ascontiguousarray(qidx, dtype=np.int32)
    cdef const int[::1] jt = np.ascontiguousarray(jtype, dtype=np.int32)
    cdef double[:, ::1] qv = q
    cdef const double[:, :, :, ::1] b = np.ascontiguousarray(base, dtype=np.float64)
    cdef const int[::1] lk = link_a
    cdef const double[:, :, :, ::1] tg = np.ascontiguousarray(target, dtype=np.float64)
    cdef const int[:, ::1] se = sel_a
    cdef const int[:, ::1] sl = slink_a
    cdef const double[::1] lo = np.ascontiguousarray(lower, dtype=np.float64)
    cdef const double[::1] hi = np.ascontiguousarray(upper, dtype=np.float64)
    cdef const int[::1] nfr = np.ascontiguousarray(n_frames, dtype=np.int32)
    cdef double[:, ::1] ps = poses
    cdef double[:, ::1] jbv = jb
    cdef double[:, :, ::1] ov = out
    cdef long long[::1] itv = its
    cdef long long[::1] fv = fails
    cdef double lam2 = damping * damping
    for p in prange(n_p, nogil=True, schedule="dynamic", num_threads=_n_threads,
                    use_threads_if=n_p > 1):
        _ik_traj_one(npa[p], &pa[p, 0], &par[0], &off[0, 0, 0], &ax[0, 0], &qi[0], &jt[0],
                     &qv[p, 0], &b[p, 0, 0, 0], lk[p], &tg[p, 0, 0, 0], ns, &se[p, 0], &sl[p, 0],
                     &lo[0], &hi[0], nfr[p], iters, tol, lam2, &ps[p, 0], &jbv[p, 0],
                     &ov[p, 0, 0], &itv[p], &fv[p])
    return out, its, fails


# --------------------------------------------------------------------------- #
# Monte-Carlo of joint perturbations in perturbation form (_core_py.mc_prepare).
# A "small" transform is 12 doubles: X_R row-major [0..8], X_p [9..11], and
# stands for I + X.
# --------------------------------------------------------------------------- #
cdef enum:
    _MC_MAXL = 64


cdef inline void _sm_conj(const double* x, const double* a, double* o) noexcept nogil:
    """o = A^-1 (I + x) A - I = A^-1 x A for a rigid 4x4 A."""
    cdef double t[9]
    cdef double y[3]
    cdef int i, j
    for i in range(3):
        for j in range(3):
            t[i * 3 + j] = x[i * 3] * a[j] + x[i * 3 + 1] * a[4 + j] + x[i * 3 + 2] * a[8 + j]
    for i in range(3):
        for j in range(3):
            o[i * 3 + j] = a[i] * t[j] + a[4 + i] * t[3 + j] + a[8 + i] * t[6 + j]
    for i in range(3):
        y[i] = x[i * 3] * a[3] + x[i * 3 + 1] * a[7] + x[i * 3 + 2] * a[11] + x[9 + i]
    for i in range(3):
        o[9 + i] = a[i] * y[0] + a[4 + i] * y[1] + a[8 + i] * y[2]


cdef inline void _sm_compose(const double* x, const double* y, double* o) noexcept nogil:
    """(I + x)(I + y) - I = x + y + x y; o must not alias x or y."""
    cdef int i, j
    for i in range(3):
        for j in range(3):
            o[i * 3 + j] = (x[i * 3 + j] + y[i * 3 + j] + x[i * 3] * y[j]
                            + x[i * 3 + 1] * y[3 + j] + x[i * 3 + 2] * y[6 + j])
        o[9 + i] = (x[9 + i] + y[9 + i] + x[i * 3] * y[9] + x[i * 3 + 1] * y[10]
                    + x[i * 3 + 2] * y[11])


cdef inline void _sm_joint(int jtype, const double* ax, double d, double* o) noexcept nogil:
    """J(d) - I: s K + (1 - cos d) (a a^T - I) for revolute, d a for prismatic."""
    cdef double s, h, x = ax[0], y = ax[1], z = ax[2]
    cdef int i
    for i in range(12):
        o[i] = 0.0
    if jtype == 1:
        s = sin(d)
        h = sin(0.5 * d)
        h = 2.0 * h * h
        o[0] = h * (x * x - 1.0); o[1] = h * x * y - s * z;   o[2] = h * x * z + s * y
        o[3] = h * y * x + s * z; o[4] = h * (y * y - 1.0);   o[5] = h * y * z - s * x
        o[6] = h * z * x - s * y; o[7] = h * z * y + s * x;   o[8] = h * (z * z - 1.0)
    elif jtype == 2:
        o[9] = d * x; o[10] = d * y; o[11] = d * z


cdef void _mc_one(Py_ssize_t nl, const int* parent, const int* dqidx, const int* jtype,
                  const double* axis, const double* a, const double* nom, const double* coms,
                  const double* w, const double* dq, const double* relnom, int ref, int target,
                  double* pos, double* ang, double* com) noexcept nogil:
    cdef double e[_MC_MAXL * 12]
    cdef double c[12]
    cdef double j[12]
    cdef double f[12]
    cdef double y[3]
    cdef double sin_t, cos_t
    cdef double* el
    cdef Py_ssize_t l
    cdef int i, k
    com[0] = 0.0; com[1] = 0.0; com[2] = 0.0
    for l in range(nl):
        el = e + 12 * l
        if parent[l] < 0:
            for i in range(12):
                c[i] = 0.0
        else:
            _sm_conj(e + 12 * parent[l], a + 16 * l, c)
        if dqidx[l] >= 0:
            _sm_joint(jtype[l], axis + 3 * l, dq[dqidx[l]], j)
            _sm_compose(c, j, el)
        else:
            for i in range(12):
                el[i] = c[i]
        for i in range(3):   # CoM shift of link l: w_l R_nom (e_R c_l + e_p)
            y[i] = (el[i * 3] * coms[3 * l] + el[i * 3 + 1] * coms[3 * l + 1]
                    + el[i * 3 + 2] * coms[3 * l + 2] + el[9 + i])
        for i in range(3):
            com[i] += w[l] * (nom[16 * l + i * 4] * y[0] + nom[16 * l + i * 4 + 1] * y[1]
                              + nom[16 * l + i * 4 + 2] * y[2])
    el = e + 12 * ref                  # f = (I + e_ref)^-1 - I, exact for a rigid motion
    for i in range(3):
        for k in range(3):
            f[i * 3 + k] = el[k * 3 + i]
        f[9 + i] = -(el[9 + i] + el[i] * el[9] + el[3 + i] * el[10] + el[6 + i] * el[11])
    _sm_conj(f, relnom, c)
    _sm_compose(c, e + 12 * target, j)  # T0^-1 T - I
    for i in range(3):
        pos[i] = relnom[i * 4] * j[9] + relnom[i * 4 + 1] * j[10] + relnom[i * 4 + 2] * j[11]
    y[0] = 0.5 * (j[7] - j[5]); y[1] = 0.5 * (j[2] - j[6]); y[2] = 0.5 * (j[3] - j[1])
    sin_t = sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2])
    cos_t = 1.0 + 0.5 * (j[0] + j[4] + j[8])
    if cos_t > 1.0:
        cos_t = 1.0
    elif cos_t < -1.0:
        cos_t = -1.0
    ang[0] = atan2(sin_t, cos_t)


def mc_relative_pose(parent, offset, axis, qidx, jtype, q0, base, joints, dq, int ref, int target,
                     coms, masses):
    """Fused Monte-Carlo, see _core_py.mc_relative_pose (spec) and mc_prepare."""
    cdef dict p = _spec.mc_prepare(parent, offset, axis, qidx, jtype, q0, base, joints, ref,
                                   target, coms, masses)
    dq = np.ascontiguousarray(dq, dtype=np.float64)
    cdef Py_ssize_t n = dq.shape[0], nl = parent.shape[0], i
    if dq.shape[1] == 0:
        dq = np.zeros((n, 1))
    pos, ang, com = np.empty((n, 3)), np.empty(n), np.empty((n, 3))
    cdef const int[::1] par = np.ascontiguousarray(parent, dtype=np.int32)
    cdef const int[::1] dqi = p["dqidx"]
    cdef const int[::1] jt = np.ascontiguousarray(jtype, dtype=np.int32)
    cdef const double[:, ::1] ax = np.ascontiguousarray(axis, dtype=np.float64)
    cdef const double[:, :, ::1] am = p["a"]
    cdef const double[:, :, ::1] nm = p["nom"]
    cdef const double[:, ::1] cm = np.ascontiguousarray(coms, dtype=np.float64)
    cdef const double[::1] w = p["w"]
    cdef const double[:, ::1] rn = p["relnom"]
    cdef const double[:, ::1] d = dq
    cdef double[:, ::1] po = pos
    cdef double[::1] an = ang
    cdef double[:, ::1] co = com
    if n and nl:
        for i in prange(n, nogil=True, schedule="static", num_threads=_n_threads,
                        use_threads_if=n >= 32):
            _mc_one(nl, &par[0], &dqi[0], &jt[0], &ax[0, 0], &am[0, 0, 0], &nm[0, 0, 0],
                    &cm[0, 0], &w[0], &d[i, 0], &rn[0, 0], ref, target,
                    &po[i, 0], &an[i], &co[i, 0])
    return pos, ang, com
