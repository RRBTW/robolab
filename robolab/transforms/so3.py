"""SO(3): rotations."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _backend
from .lie import DEFAULT_TOL, LieGroup, as_array, as_points, check_frame_name


class Quaternion(NamedTuple):
    """Unit quaternion, scalar part named explicitly to avoid wxyz/xyzw mix-ups."""
    w: float
    x: float
    y: float
    z: float


def _angle(theta):
    a = np.asarray(theta, dtype=float)
    if a.ndim != 0 or not np.isfinite(a):
        raise ValueError("angle must be a finite scalar in radians")
    return float(a)


def rotx(angle):
    c, s = np.cos(_angle(angle)), np.sin(_angle(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(angle):
    c, s = np.cos(_angle(angle)), np.sin(_angle(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(angle):
    c, s = np.cos(_angle(angle)), np.sin(_angle(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def project_to_so3(m):
    """Closest proper rotation in Frobenius norm (SVD, det fixed to +1)."""
    m = as_array(m, (3, 3), "matrix")
    u, _, vt = np.linalg.svd(m)
    d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    return np.ascontiguousarray(u @ np.diag([1.0, 1.0, d]) @ vt)


class SO3(LieGroup):
    """Rotation R in SO(3) = {R in R^3x3 | R^T R = I, det R = +1}.

    ``SO3(R, parent='A', child='B')`` is the orientation of frame B in frame A:
    it rotates vectors expressed in B into A.
    """

    __slots__ = ()
    DOF = 3
    MATRIX_DIM = 3

    def __init__(self, matrix, *, parent=None, child=None, tol=DEFAULT_TOL):
        self._init(as_array(matrix, (3, 3), "rotation"), parent, child, tol)

    # -- constructors --------------------------------------------------------- #
    @classmethod
    def from_matrix(cls, matrix, *, project=False, tol=DEFAULT_TOL, parent=None, child=None):
        """Validate (default) or, with project=True, repair by SVD projection.
        Use project=True for data that went through float32 or was rounded."""
        m = project_to_so3(matrix) if project else matrix
        return cls(m, parent=parent, child=child, tol=tol)

    @classmethod
    def from_axis_angle(cls, w, *, parent=None, child=None):
        """Axis-angle vector, |w| = angle in radians (same as exp)."""
        return cls.exp(w, parent=parent, child=child)

    @classmethod
    def Rx(cls, angle, *, parent=None, child=None):
        return cls._wrap(rotx(angle), check_frame_name(parent), check_frame_name(child))

    @classmethod
    def Ry(cls, angle, *, parent=None, child=None):
        return cls._wrap(roty(angle), check_frame_name(parent), check_frame_name(child))

    @classmethod
    def Rz(cls, angle, *, parent=None, child=None):
        return cls._wrap(rotz(angle), check_frame_name(parent), check_frame_name(child))

    @classmethod
    def from_quaternion(cls, *, w, x, y, z, parent=None, child=None):
        """Keyword-only on purpose: ROS/scipy use (x, y, z, w), Eigen and many
        papers use (w, x, y, z). The input is normalised."""
        q = as_array([w, x, y, z], (4,), "quaternion")
        n = float(np.sqrt(q @ q))
        if n < 1e-12:
            raise ValueError("quaternion has zero norm")
        w, x, y, z = q / n
        m = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return cls._wrap(m, check_frame_name(parent), check_frame_name(child))

    # -- kernel hooks --------------------------------------------------------- #
    @classmethod
    def _check(cls, m, tol):
        return _backend.kernel.so3_check(m, tol)

    def _compose(self, other_m):
        return _backend.kernel.so3_mul(self._m, other_m)

    def _inverse(self):
        return np.ascontiguousarray(self._m.T)

    @classmethod
    def _exp(cls, tangent):
        return _backend.kernel.so3_exp(tangent)

    # -- protocol ------------------------------------------------------------- #
    def log(self):
        """Principal axis-angle vector, angle in [0, pi]. At exactly pi the sign
        of the axis is ambiguous (both signs give the same rotation)."""
        return _backend.kernel.so3_log(self._m)

    def act(self, points):
        """Rotate a vector (3,) or vectors (M, 3)."""
        pts, single = as_points(points)
        out = _backend.kernel.so3_act(self._m, pts)
        return out[0] if single else out

    def adjoint(self):
        return self._m.copy()

    def normalized(self):
        """Re-orthonormalise (removes drift after very long composition chains)."""
        return self._wrap(project_to_so3(self._m), self._parent, self._child)

    # -- conversions ---------------------------------------------------------- #
    def angle(self):
        return float(np.linalg.norm(self.log()))

    def as_quaternion(self) -> Quaternion:
        """Unit quaternion with w >= 0 (Shepperd's method, stable for all R)."""
        r = self._m
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        cand = (tr, r[0, 0], r[1, 1], r[2, 2])
        k = int(np.argmax(cand))
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + tr)
            q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
        elif k == 1:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
        elif k == 2:
            s = 2.0 * np.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2])
            q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
        else:
            s = 2.0 * np.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2])
            q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
        q = np.array(q)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        return Quaternion(*map(float, q))
