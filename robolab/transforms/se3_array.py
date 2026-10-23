"""SE3Array: a batch of transforms sharing the same frame tags.

Deliberately a separate type from SE3 (unlike spatialmath, where a pose and a
list of poses are the same class): the type always tells you what you hold.
Storage is one contiguous (N, 4, 4) array, so batch kernels run without Python
loops and, in the Cython backend, without the GIL.

Typical use: a trajectory of one body over time (T_world_foot at N instants),
or N candidate poses evaluated at once.
"""
from __future__ import annotations

from typing import Iterable, Union

import numpy as np

from . import _backend
from .lie import (DEFAULT_TOL, FrameMismatchError, InvalidElementError, as_points,
                  check_chain, check_frame_name, frame_label, readonly)
from .se3_group import SE3


class SE3Array:
    __slots__ = ("_m", "_parent", "_child")

    def __init__(self, poses: Union[np.ndarray, Iterable[SE3]], *, parent=None, child=None,
                 tol=DEFAULT_TOL):
        if isinstance(poses, np.ndarray):
            m = np.array(poses, dtype=np.float64, order="C", copy=True)
            if m.ndim != 3 or m.shape[1:] != (4, 4):
                raise ValueError(f"poses must have shape (N, 4, 4), got {m.shape}")
        else:
            poses = list(poses)
            if not all(isinstance(p, SE3) for p in poses):
                raise TypeError("expected an (N,4,4) array or an iterable of SE3")
            tags = {(p.parent, p.child) for p in poses}
            if len(tags) > 1:
                raise FrameMismatchError(f"poses carry different frames: {sorted(map(str, tags))}")
            if tags and parent is None and child is None:
                parent, child = tags.pop()
            m = (np.stack([p.matrix for p in poses]) if poses else np.zeros((0, 4, 4)))
            m = np.ascontiguousarray(m)
        if not _backend.kernel.se3_check_batch(m, tol):
            raise InvalidElementError(f"at least one matrix is not a valid SE3 element (tol={tol:g})")
        self._set(m, check_frame_name(parent), check_frame_name(child))

    @classmethod
    def _wrap(cls, m, parent, child):
        obj = object.__new__(cls)
        obj._set(m, parent, child)
        return obj

    def _set(self, m, parent, child):
        object.__setattr__(self, "_m", readonly(m))
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_child", child)

    def __setattr__(self, name, value):
        raise AttributeError("SE3Array is immutable")

    # -- constructors --------------------------------------------------------- #
    @classmethod
    def exp(cls, twists, *, parent=None, child=None):
        x = np.array(twists, dtype=np.float64, order="C", copy=True)
        if x.ndim != 2 or x.shape[1] != 6 or not np.all(np.isfinite(x)):
            raise ValueError(f"twists must be a finite (N, 6) array, got {x.shape}")
        return cls._wrap(_backend.batch_kernel(len(x)).se3_exp_batch(x),
                         check_frame_name(parent), check_frame_name(child))

    @classmethod
    def identity(cls, n, *, parent=None, child=None):
        return cls._wrap(np.tile(np.eye(4), (n, 1, 1)), check_frame_name(parent),
                         check_frame_name(child))

    # -- container protocol ---------------------------------------------------- #
    def __len__(self):
        return self._m.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._wrap(np.ascontiguousarray(self._m[i]), self._parent, self._child)
        return SE3._wrap(self._m[i].copy(), self._parent, self._child)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # -- group operations (element-wise, broadcasting 1 <-> N) ------------------ #
    def inverse(self):
        return self._wrap(_backend.batch_kernel(len(self)).se3_inv_batch(self._m),
                          self._child, self._parent)

    def log(self):
        """Twists (N, 6), ordered (omega, v)."""
        return _backend.batch_kernel(len(self)).se3_log_batch(self._m)

    def __matmul__(self, other):
        if isinstance(other, SE3):
            om = other.matrix.reshape(1, 4, 4)
        elif isinstance(other, SE3Array):
            om = other._m
        else:
            return NotImplemented
        check_chain(self._parent, self._child, other.parent, other.child)
        k = _backend.batch_kernel(max(len(self), om.shape[0]))
        return self._wrap(k.se3_mul_batch(self._m, om), self._parent, other.child)

    def __rmatmul__(self, other):
        if not isinstance(other, SE3):
            return NotImplemented
        check_chain(other.parent, other.child, self._parent, self._child)
        k = _backend.batch_kernel(len(self))
        return self._wrap(k.se3_mul_batch(other.matrix.reshape(1, 4, 4), self._m),
                          other.parent, self._child)

    def apply(self, points):
        """Every pose applied to every point: (M,3) -> (N, M, 3); (3,) -> (N, 3)."""
        pts, single = as_points(points)
        out = _backend.batch_kernel(len(self) * len(pts)).se3_act_batch(self._m, pts)
        return out[:, 0] if single else out

    def apply_each(self, points):
        """i-th pose applied to i-th point: (N, 3) -> (N, 3)."""
        pts, _ = as_points(points)
        return _backend.batch_kernel(len(self)).se3_act_each(self._m, pts)

    # -- accessors -------------------------------------------------------------- #
    @property
    def matrices(self):
        return self._m.view()

    @property
    def translations(self):
        return readonly(self._m[:, :3, 3].view())

    @property
    def parent(self):
        return self._parent

    @property
    def child(self):
        return self._child

    def __eq__(self, other):
        raise TypeError("compare SE3Array matrices explicitly with np.allclose")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self):
        return f"SE3Array(n={len(self)}, frames={frame_label(self._parent, self._child)})"
