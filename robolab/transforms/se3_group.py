r"""SE(3): rigid-body transforms.

SE(3) = { <R, p> | R in SO(3), p in R^3 }, stored as the homogeneous matrix

    T = [[R, p],
         [0, 1]]

Physical meaning (anthropomorphic robot): a logical body part such as a foot
has its own local frame, and all its sub-parts (segments, joints, contact
points) have fixed coordinates in that frame. ``p`` places the whole local
frame in the parent (e.g. world) frame; ``R`` is the orientation of the local
frame, and by applying it every internal point is carried into the parent
frame:  x_parent = R x_local + p.  ``T_world_foot.act(toe_in_foot)`` therefore
gives the toe in world coordinates.

Twist coordinates are ordered xi = (omega, v) (Lynch & Park, Modern Robotics).
"""
from __future__ import annotations

import numpy as np

from . import _backend
from .lie import DEFAULT_TOL, LieGroup, as_array, as_points, check_frame_name, readonly
from .so3 import SO3, project_to_so3


class SE3(LieGroup):
    """Rigid transform T_parent_child: maps points in ``child`` into ``parent``.

    SE3(R, p) / SE3(rotation=SO3(...), translation=p); either may be omitted
    (identity rotation, zero translation).
    """

    __slots__ = ()
    DOF = 6
    MATRIX_DIM = 4

    def __init__(self, rotation=None, translation=None, *, parent=None, child=None,
                 tol=DEFAULT_TOL):
        m = np.eye(4)
        if rotation is not None:
            m[:3, :3] = rotation.matrix if isinstance(rotation, SO3) else as_array(
                rotation, (3, 3), "rotation")
        if translation is not None:
            m[:3, 3] = as_array(translation, (3,), "translation")
        self._init(m, parent, child, tol)

    # -- constructors --------------------------------------------------------- #
    @classmethod
    def from_matrix(cls, matrix, *, project=False, tol=DEFAULT_TOL, parent=None, child=None):
        """From a 4x4 homogeneous matrix. With project=True the rotation block
        is repaired by SVD projection and the last row is reset to [0,0,0,1]."""
        m = as_array(matrix, (4, 4), "transform")
        if project:
            m[:3, :3] = project_to_so3(m[:3, :3])
            m[3] = (0.0, 0.0, 0.0, 1.0)
        obj = object.__new__(cls)
        obj._init(m, parent, child, tol)
        return obj

    @classmethod
    def from_translation(cls, p, *, parent=None, child=None):
        m = np.eye(4)
        m[:3, 3] = as_array(p, (3,), "translation")
        return cls._wrap(m, check_frame_name(parent), check_frame_name(child))

    @classmethod
    def Trans(cls, x, y, z, *, parent=None, child=None):
        return cls.from_translation([x, y, z], parent=parent, child=child)

    @classmethod
    def from_rotation(cls, rotation, *, parent=None, child=None):
        r = rotation if isinstance(rotation, SO3) else SO3(rotation)
        m = np.eye(4)
        m[:3, :3] = r.matrix
        return cls._wrap(m, check_frame_name(parent), check_frame_name(child))

    @classmethod
    def Rx(cls, angle, **frames):
        return cls.from_rotation(SO3.Rx(angle), **frames)

    @classmethod
    def Ry(cls, angle, **frames):
        return cls.from_rotation(SO3.Ry(angle), **frames)

    @classmethod
    def Rz(cls, angle, **frames):
        return cls.from_rotation(SO3.Rz(angle), **frames)

    # -- kernel hooks --------------------------------------------------------- #
    @classmethod
    def _check(cls, m, tol):
        return _backend.kernel.se3_check(m, tol)

    def _compose(self, other_m):
        return _backend.kernel.se3_mul(self._m, other_m)

    def _inverse(self):
        return _backend.kernel.se3_inv(self._m)

    @classmethod
    def _exp(cls, tangent):
        return _backend.kernel.se3_exp(tangent)

    # -- protocol ------------------------------------------------------------- #
    def log(self):
        """Twist xi = (omega, v), |omega| in [0, pi]."""
        return _backend.kernel.se3_log(self._m)

    def act(self, points):
        """Transform a point (3,) or points (M, 3) from child into parent."""
        pts, single = as_points(points)
        out = _backend.kernel.se3_act(self._m, pts)
        return out[0] if single else out

    def act_vector(self, vectors):
        """Rotate free vectors (directions, velocities): no translation."""
        return SO3._wrap(np.ascontiguousarray(self._m[:3, :3])).act(vectors)

    def adjoint(self):
        """6x6 adjoint for xi = (omega, v): maps twists from child to parent."""
        return _backend.kernel.se3_adjoint(self._m)

    def normalized(self):
        m = self._m.copy()
        m[:3, :3] = project_to_so3(m[:3, :3])
        return self._wrap(m, self._parent, self._child)

    # -- accessors ------------------------------------------------------------ #
    @property
    def rotation(self) -> SO3:
        """Orientation of child in parent (same frame tags)."""
        return SO3._wrap(np.ascontiguousarray(self._m[:3, :3]), self._parent, self._child)

    @property
    def translation(self):
        """Origin of child expressed in parent (read-only view)."""
        return readonly(self._m[:3, 3].view())

    R = property(lambda self: readonly(self._m[:3, :3].view()),
                 doc="Rotation block as a read-only 3x3 array view.")
    p = translation
