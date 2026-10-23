"""Rigid 3D rotations and transforms, using column vectors and radians.

T_A_B maps coordinates expressed in frame B into frame A. Compose as T_A_B @ T_B_C.
All input arrays are copied; classes expose defensive copies of their matrices.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_TOL = 1e-9


def _array(x, shape, name):
    a = np.asarray(x, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a finite array of shape {shape}')
    return a.copy()


def _angle(theta):
    a = np.asarray(theta, dtype=float)
    if a.ndim != 0 or not np.isfinite(a):
        raise ValueError('angle must be a finite scalar in radians')
    return float(a)


def _rotation(r):
    r = _array(r, (3, 3), 'rotation')
    if not np.allclose(r.T @ r, np.eye(3), atol=_TOL, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=_TOL, rtol=0):
        raise ValueError('rotation must be in SO(3): R.T @ R = I and det(R) = +1')
    return r


def _transform(t):
    t = _array(t, (4, 4), 'transform')
    _rotation(t[:3, :3])
    if not np.allclose(t[3], [0, 0, 0, 1], atol=_TOL, rtol=0):
        raise ValueError('last row must be [0, 0, 0, 1]')
    return t


def rotx(angle):
    c, s = np.cos(_angle(angle)), np.sin(_angle(angle))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(angle):
    c, s = np.cos(_angle(angle)), np.sin(_angle(angle))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(angle):
    c, s = np.cos(_angle(angle)), np.sin(_angle(angle))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skew(v):
    x, y, z = _array(v, (3,), 'vector')
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def unskew(m):
    m = _array(m, (3, 3), 'skew matrix')
    if not np.allclose(m + m.T, 0, atol=_TOL, rtol=0):
        raise ValueError('matrix must be skew-symmetric')
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def exp_so3(w):
    """Exponential map of axis-angle vector (norm = radians)."""
    w = _array(w, (3,), 'axis-angle')
    theta = np.linalg.norm(w)
    k = skew(w)
    if theta < 1e-4:
        t2 = theta * theta
        a = 1 - t2/6 + t2*t2/120
        b = .5 - t2/24 + t2*t2/720
    else:
        a = np.sin(theta)/theta
        b = (1 - np.cos(theta))/(theta*theta)
    return np.eye(3) + a*k + b*(k@k)


def log_so3(r):
    """Principal logarithm: axis-angle norm in [0, pi]; pi axis sign is ambiguous."""
    r = _rotation(r)
    v = np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])/2
    sin_theta = np.linalg.norm(v)
    cos_theta = np.clip((np.trace(r)-1)/2, -1, 1)
    theta = np.arctan2(sin_theta, cos_theta)
    if theta < 1e-7:
        return v * (1 + theta*theta/6)
    if np.pi-theta < 1e-5:
        # The symmetric part is stable near pi. Eigenvector with eigenvalue 1.
        vals, vecs = np.linalg.eigh((r+r.T)/2)
        axis = vecs[:, np.argmax(vals)]
        if np.dot(axis, v) < 0:
            axis = -axis
        return theta*axis
    return (theta/sin_theta)*v


def se3(r, p):
    """Construct a 4x4 rigid transform from rotation and translation."""
    r = _rotation(r)
    p = _array(p, (3,), 'translation')
    t = np.eye(4)
    t[:3,:3], t[:3,3] = r, p
    return t


def se3_inv(t):
    t = _transform(t)
    r = t[:3,:3].T
    return se3(r, -r@t[:3,3])


def project_so3(m):
    """Closest proper rotation in Frobenius norm (SVD)."""
    m = _array(m, (3,3), 'matrix')
    u, _, vt = np.linalg.svd(m)
    d = np.linalg.det(u@vt)
    return u @ np.diag([1., 1., 1. if d >= 0 else -1.]) @ vt


def plot_frame(t, ax=None, length=0.2, label=None):
    """Plot local axes in parent frame. Returns Matplotlib 3D Axes."""
    t = _transform(t)
    length = _angle(length)
    if length <= 0:
        raise ValueError('length must be positive')
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111, projection='3d')
    origin, r = t[:3,3], t[:3,:3]
    for i, (color, name) in enumerate(zip(('r','g','b'), ('X','Y','Z'))):
        vec = length*r[:,i]
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=.15)
        ax.text(*(origin+vec), f'{label+":" if label else ""}{name}', color=color)
    return ax


@dataclass(frozen=True, init=False)
class Rotation3D:
    _matrix: np.ndarray

    def __init__(self, matrix):
        object.__setattr__(self, '_matrix', _rotation(matrix))

    @property
    def matrix(self):
        return self._matrix.copy()

    @classmethod
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def from_axis_angle(cls, w):
        return cls(exp_so3(w))

    def inverse(self):
        return Rotation3D(self._matrix.T)

    def apply(self, v):
        return self._matrix @ _array(v, (3,), 'vector')

    def __matmul__(self, other):
        if not isinstance(other, Rotation3D):
            return NotImplemented
        return Rotation3D(self._matrix @ other._matrix)


@dataclass(frozen=True, init=False)
class Transform3D:
    _matrix: np.ndarray

    def __init__(self, rotation, translation):
        r = rotation.matrix if isinstance(rotation, Rotation3D) else rotation
        object.__setattr__(self, '_matrix', se3(r, translation))

    @property
    def matrix(self):
        return self._matrix.copy()

    @property
    def rotation(self):
        return Rotation3D(self._matrix[:3,:3])

    @property
    def translation(self):
        return self._matrix[:3,3].copy()

    @classmethod
    def identity(cls):
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, matrix):
        t = _transform(matrix)
        return cls(t[:3,:3], t[:3,3])

    def inverse(self):
        return Transform3D.from_matrix(se3_inv(self._matrix))

    def apply(self, point):
        return self._matrix[:3,:3] @ _array(point, (3,), 'point') + self._matrix[:3,3]

    def __matmul__(self, other):
        if not isinstance(other, Transform3D):
            return NotImplemented
        return Transform3D.from_matrix(self._matrix @ other._matrix)
