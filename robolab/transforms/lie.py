"""Common machinery for matrix Lie groups: immutability, frames, validation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import _backend

DEFAULT_TOL = 1e-9
"""Tolerance for ||R^T R - I||_max and |det R - 1|. Strict on purpose: data
from float32 sources (ROS messages, files) must go through from_matrix(...,
project=True) or from_quaternion, so that repairing a matrix is explicit."""


class InvalidElementError(ValueError):
    """The matrix is not a valid group element (within tolerance)."""


class FrameMismatchError(ValueError):
    """Composition of transforms whose inner frames do not match."""


# --------------------------------------------------------------------------- #
# input helpers
# --------------------------------------------------------------------------- #
def as_array(x, shape, name):
    """Finite float64 C-contiguous *copy* of x with exactly the given shape."""
    a = np.array(x, dtype=np.float64, order="C", copy=True)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def as_points(x):
    """Return (points (M,3) C-contiguous float64, was_single_point)."""
    a = np.asarray(x, dtype=np.float64)
    if a.shape == (3,):
        return np.ascontiguousarray(a.reshape(1, 3)), True
    if a.ndim == 2 and a.shape[1] == 3:
        return np.ascontiguousarray(a), False
    raise ValueError(f"points must have shape (3,) or (M, 3), got {a.shape}")


def check_frame_name(name):
    if name is not None and not isinstance(name, str):
        raise TypeError(f"frame names must be str or None, got {type(name).__name__}")
    return name


def frame_label(parent, child):
    return f"T_{parent or '?'}_{child or '?'}"


def check_chain(lp, lc, rp, rc):
    """Left maps lc -> lp, right maps rc -> rp; composition needs lc == rp."""
    if lc is not None and rp is not None and lc != rp:
        raise FrameMismatchError(
            f"{frame_label(lp, lc)} @ {frame_label(rp, rc)}: inner frames differ "
            f"({lc!r} != {rp!r})")


def readonly(a):
    a.setflags(write=False)
    return a


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
class LieGroup(ABC):
    r"""Immutable element of a matrix Lie group, tagged with optional frames.

    An element ``T`` with ``parent='A'`` and ``child='B'`` is ``T_A_B``: it maps
    coordinates expressed in B into A. Composition checks that inner frames
    match: ``T_A_B @ T_B_C -> T_A_C``; ``T_A_B @ T_C_D`` raises
    FrameMismatchError. A frame left as None is not checked (gradual typing).

    Every group implements the same protocol: identity, exp, log, inverse,
    ``@`` (compose), act (apply to points), adjoint, interpolate, isclose.
    """

    __slots__ = ("_m", "_parent", "_child")
    DOF: int = 0
    MATRIX_DIM: int = 0

    # -- construction ------------------------------------------------------ #
    @classmethod
    def _wrap(cls, m, parent=None, child=None):
        """Wrap a freshly computed, trusted matrix (no validation, no copy)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "_m", readonly(m))
        object.__setattr__(obj, "_parent", parent)
        object.__setattr__(obj, "_child", child)
        return obj

    def _init(self, m, parent, child, tol):
        if not self._check(m, tol):
            raise InvalidElementError(
                f"matrix is not a valid {type(self).__name__} element (tol={tol:g}); "
                f"use {type(self).__name__}.from_matrix(..., project=True) to repair it "
                f"explicitly")
        object.__setattr__(self, "_m", readonly(m))
        object.__setattr__(self, "_parent", check_frame_name(parent))
        object.__setattr__(self, "_child", check_frame_name(child))

    @classmethod
    def unchecked(cls, matrix, *, parent=None, child=None):
        """Wrap a matrix WITHOUT validating it. Only for matrices produced by
        trusted code in hot loops; everything else should use the constructor."""
        m = np.array(matrix, dtype=np.float64, order="C", copy=True)
        return cls._wrap(m, check_frame_name(parent), check_frame_name(child))

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_rebuild, (type(self), self._m.copy(), self._parent, self._child))

    # -- group-specific kernel hooks ---------------------------------------- #
    @classmethod
    @abstractmethod
    def _check(cls, m, tol) -> bool: ...

    @abstractmethod
    def _compose(self, other_m) -> np.ndarray: ...

    @abstractmethod
    def _inverse(self) -> np.ndarray: ...

    @classmethod
    @abstractmethod
    def _exp(cls, tangent) -> np.ndarray: ...

    @abstractmethod
    def log(self) -> np.ndarray: ...

    @abstractmethod
    def act(self, points) -> np.ndarray: ...

    @abstractmethod
    def adjoint(self) -> np.ndarray: ...

    @abstractmethod
    def normalized(self): ...

    # -- protocol ------------------------------------------------------------ #
    @classmethod
    def identity(cls, *, parent=None, child=None):
        return cls._wrap(np.eye(cls.MATRIX_DIM), check_frame_name(parent),
                         check_frame_name(child))

    @classmethod
    def exp(cls, tangent, *, parent=None, child=None):
        t = as_array(tangent, (cls.DOF,), "tangent vector")
        return cls._wrap(cls._exp(t), check_frame_name(parent), check_frame_name(child))

    def inverse(self):
        return self._wrap(self._inverse(), self._child, self._parent)

    def __matmul__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        check_chain(self._parent, self._child, other._parent, other._child)
        return self._wrap(self._compose(other._m), self._parent, other._child)

    def apply(self, points):
        """Alias of act()."""
        return self.act(points)

    def interpolate(self, other, t):
        """Geodesic interpolation: self @ exp(t * log(self^-1 @ other)).
        t=0 gives self, t=1 gives other. Both must have the same frames."""
        if type(other) is not type(self):
            raise TypeError(f"cannot interpolate {type(self).__name__} with {type(other).__name__}")
        if (self._parent, self._child) != (other._parent, other._child):
            raise FrameMismatchError(
                f"interpolate needs equal frames: {frame_label(self._parent, self._child)} "
                f"vs {frame_label(other._parent, other._child)}")
        t = float(t)
        cls = type(self)
        delta = cls._wrap(cls._wrap(self._inverse())._compose(other._m))
        step = cls._exp(t * delta.log())
        return self._wrap(self._compose(step), self._parent, self._child)

    def isclose(self, other, tol=1e-9, *, check_frames=True):
        if type(other) is not type(self):
            return False
        if check_frames and (self._parent, self._child) != (other._parent, other._child):
            return False
        return bool(np.max(np.abs(self._m - other._m)) <= tol)

    def with_frames(self, parent=None, child=None):
        """Same transform with new frame tags (the matrix is shared: immutable)."""
        return self._wrap(self._m, check_frame_name(parent), check_frame_name(child))

    # -- accessors ----------------------------------------------------------- #
    @property
    def matrix(self):
        """Read-only view of the matrix (no copy; writing raises)."""
        return self._m.view()

    @property
    def parent(self) -> Optional[str]:
        return self._parent

    @property
    def child(self) -> Optional[str]:
        return self._child

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self._m, dtype=dtype, copy=True)
        return self._m.view() if dtype is None else self._m.astype(dtype)

    # Exact float equality of transforms is almost always a bug; be explicit.
    def __eq__(self, other):
        raise TypeError(f"use {type(self).__name__}.isclose(other, tol) instead of ==")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self):
        frames = ""
        if self._parent is not None or self._child is not None:
            frames = f", parent={self._parent!r}, child={self._child!r}"
        body = np.array2string(self._m, precision=4, suppress_small=True, separator=", ")
        return f"{type(self).__name__}({body}{frames})"


def _rebuild(cls, m, parent, child):
    return cls._wrap(m, parent, child)
