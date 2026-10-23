"""Array-level API (backwards compatible with robolab v0.1).

These functions take and return plain NumPy arrays and VALIDATE their inputs.
Prefer the SO3 / SE3 classes in new code: they carry frame tags and keep the
group invariant for you. These functions remain for quick scripts, for the
original acceptance tests and for interop with code that passes raw matrices.
"""
from __future__ import annotations

import numpy as np

from . import _backend
from .lie import DEFAULT_TOL, InvalidElementError, as_array
from .so3 import project_to_so3, rotx, roty, rotz  # noqa: F401  (re-exported)

_TOL = DEFAULT_TOL


def _rotation(r):
    r = as_array(r, (3, 3), "rotation")
    if not _backend.kernel.so3_check(r, _TOL):
        raise InvalidElementError("rotation must be in SO(3): R.T @ R = I and det(R) = +1")
    return r


def _transform(t):
    t = as_array(t, (4, 4), "transform")
    if not _backend.kernel.se3_check(t, _TOL):
        raise InvalidElementError("transform must be in SE(3) with last row [0, 0, 0, 1]")
    return t


def skew(v):
    """Hat operator: skew(a) @ b == cross(a, b)."""
    return _backend.kernel.hat3(as_array(v, (3,), "vector"))


def unskew(m):
    """Vee operator, inverse of skew."""
    m = as_array(m, (3, 3), "skew matrix")
    if not np.allclose(m + m.T, 0.0, atol=_TOL, rtol=0.0):
        raise ValueError("matrix must be skew-symmetric")
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def exp_so3(w):
    """Axis-angle vector (norm = radians) -> rotation matrix."""
    return _backend.kernel.so3_exp(as_array(w, (3,), "axis-angle"))


def log_so3(r):
    """Principal logarithm: axis-angle vector with norm in [0, pi]."""
    return _backend.kernel.so3_log(_rotation(r))


def exp_se3(xi):
    """Twist (omega, v) -> 4x4 transform."""
    return _backend.kernel.se3_exp(as_array(xi, (6,), "twist"))


def log_se3(t):
    """4x4 transform -> twist (omega, v)."""
    return _backend.kernel.se3_log(_transform(t))


def se3(r, p):
    """Build a 4x4 rigid transform from rotation and translation."""
    t = np.eye(4)
    t[:3, :3] = _rotation(r)
    t[:3, 3] = as_array(p, (3,), "translation")
    return t


def se3_inv(t):
    return _backend.kernel.se3_inv(_transform(t))


def project_so3(m):
    """Closest proper rotation in Frobenius norm (SVD)."""
    return project_to_so3(m)


def plot_frame(t, ax=None, length=0.2, label=None):
    from .plot import plot_frame as _plot
    return _plot(t, ax=ax, length=length, label=label)
