"""Kinematic tree: forward kinematics, Jacobians, centre of mass, inverse kinematics.

A robot is a tree of links. Each link hangs off its parent through a joint:

    T_world_link = T_world_parent @ offset @ J(q)

where ``offset`` = T_parent_joint is the fixed placement of the joint frame in
the parent link (at q = 0) and J(q) is a rotation about ``axis`` (revolute), a
translation along ``axis`` (prismatic) or the identity (fixed). The link frame
coincides with the joint frame, so a joint axis is expressed in its own link.

Forward kinematics of the whole tree for N configurations is ONE kernel call
(Cython, OpenMP over configurations), which is what makes Monte-Carlo
analyses and whole-trajectory evaluation cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..transforms import SE3
from ..transforms import _backend, _core_py
from ..transforms.lie import FrameMismatchError, as_array, readonly

JOINT_TYPES = {"fixed": 0, "revolute": 1, "prismatic": 2}


@dataclass(frozen=True)
class Link:
    """One link and the joint that connects it to its parent.

    name    : link (and joint frame) name
    parent  : parent link name, or None for a link attached to the base
    offset  : SE3, joint frame in the parent link frame at q = 0
    joint   : joint name, or None for a rigidly attached link
    axis    : joint axis in the joint/link frame (normalised automatically)
    type    : "revolute" | "prismatic" (ignored when joint is None)
    mass    : kg;  com: centre of mass in the link frame, m
    limits  : (lower, upper) joint limits in rad or m
    """
    name: str
    parent: Optional[str]
    offset: SE3
    joint: Optional[str] = None
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    type: str = "revolute"
    mass: float = 0.0
    com: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    limits: Tuple[float, float] = (-np.inf, np.inf)


@dataclass
class IKResult:
    converged: bool
    iterations: int
    error: float  # norm of the final body-twist error


class TreePoses:
    """World poses of every link for one configuration."""

    __slots__ = ("_tree", "_m")

    def __init__(self, tree: "KinematicTree", matrices: np.ndarray):
        self._tree = tree
        self._m = readonly(matrices)

    @property
    def matrices(self):
        """(L, 4, 4) read-only array, row order = tree.link_names."""
        return self._m.view()

    def __getitem__(self, name) -> SE3:
        return SE3._wrap(self._m[self._tree.link_id(name)].copy(), self._tree.base_frame, name)

    def position(self, name):
        return self._m[self._tree.link_id(name), :3, 3].copy()


class KinematicTree:
    def __init__(self, links: Iterable[Link], base_frame: str = "world"):
        links = list(links)
        self.base_frame = base_frame
        self._links = links
        self._lid: Dict[str, int] = {}
        joint_names: List[str] = []
        parent, qidx, jtype = [], [], []
        offsets, axes, masses, coms, limits = [], [], [], [], []
        for i, lk in enumerate(links):
            if lk.name in self._lid:
                raise ValueError(f"duplicate link name {lk.name!r}")
            if lk.parent is not None and lk.parent not in self._lid:
                raise ValueError(f"link {lk.name!r}: parent {lk.parent!r} must be listed before it")
            if not isinstance(lk.offset, SE3):
                raise TypeError(f"link {lk.name!r}: offset must be an SE3")
            self._lid[lk.name] = i
            parent.append(-1 if lk.parent is None else self._lid[lk.parent])
            offsets.append(lk.offset.matrix)
            ax = as_array(lk.axis, (3,), f"axis of {lk.name}")
            n = np.linalg.norm(ax)
            if lk.joint is not None and n < 1e-12:
                raise ValueError(f"joint {lk.joint!r} has a zero axis")
            axes.append(ax / n if n > 0 else ax)
            if lk.joint is None:
                qidx.append(-1)
                jtype.append(0)
            else:
                if lk.type not in ("revolute", "prismatic"):
                    raise ValueError(f"joint {lk.joint!r}: unknown type {lk.type!r}")
                if lk.joint in joint_names:
                    raise ValueError(f"duplicate joint name {lk.joint!r}")
                qidx.append(len(joint_names))
                jtype.append(JOINT_TYPES[lk.type])
                joint_names.append(lk.joint)
                limits.append(lk.limits)
            masses.append(float(lk.mass))
            coms.append(as_array(lk.com, (3,), f"com of {lk.name}"))

        self._parent = np.ascontiguousarray(parent, dtype=np.int32)
        self._qidx = np.ascontiguousarray(qidx, dtype=np.int32)
        self._jtype = np.ascontiguousarray(jtype, dtype=np.int32)
        self._offset = readonly(np.ascontiguousarray(offsets, dtype=np.float64).reshape(-1, 4, 4))
        self._axis = readonly(np.ascontiguousarray(axes, dtype=np.float64).reshape(-1, 3))
        self.masses = readonly(np.asarray(masses))
        self.coms_local = readonly(np.asarray(coms).reshape(-1, 3))
        self.limits = readonly(np.asarray(limits, dtype=float).reshape(-1, 2))
        self._lower = readonly(np.ascontiguousarray(self.limits[:, 0]))
        self._upper = readonly(np.ascontiguousarray(self.limits[:, 1]))
        self.joint_names = tuple(joint_names)
        self.link_names = tuple(lk.name for lk in links)
        self._jid = {n: i for i, n in enumerate(joint_names)}
        self._joint_link = np.array([i for i, q in enumerate(qidx) if q >= 0], dtype=int)
        # joints on the path base -> link (link indices, root first)
        self._chain = []
        for i in range(len(links)):
            path, k = [], i
            while k >= 0:
                if qidx[k] >= 0:
                    path.append(k)
                k = parent[k]
            self._chain.append(np.array(path[::-1], dtype=int))

    # -- bookkeeping ------------------------------------------------------------ #
    @property
    def links(self):
        return tuple(self._links)

    @property
    def nq(self):
        return len(self.joint_names)

    @property
    def nl(self):
        return len(self.link_names)

    @property
    def total_mass(self):
        return float(self.masses.sum())

    def link_id(self, name):
        try:
            return self._lid[name]
        except KeyError:
            raise KeyError(f"no link named {name!r}") from None

    def joint_id(self, name):
        try:
            return self._jid[name]
        except KeyError:
            raise KeyError(f"no joint named {name!r}") from None

    def chain_joints(self, link) -> np.ndarray:
        """Indices into q of the joints between the base and ``link``."""
        return self._qidx[self._chain[self.link_id(link)]].astype(int)

    def zero(self):
        return np.zeros(self.nq)

    def clip(self, q):
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])

    # -- forward kinematics -------------------------------------------------------- #
    def _q2d(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            q = q[None]
        if q.ndim != 2 or q.shape[1] != self.nq:
            raise ValueError(f"q must have shape (nq,) or (N, nq) with nq={self.nq}, got {q.shape}")
        if self.nq == 0:
            q = np.zeros((q.shape[0], 1))
        return np.ascontiguousarray(q)

    def _base3d(self, base, n):
        if base is None:
            return np.eye(4)[None]
        if isinstance(base, SE3):
            if base.child is not None and base.parent not in (None, self.base_frame):
                raise FrameMismatchError(f"base must be expressed in {self.base_frame!r}")
            return np.ascontiguousarray(base.matrix)[None]
        if hasattr(base, "matrices"):  # SE3Array
            return np.ascontiguousarray(base.matrices)
        b = np.ascontiguousarray(base, dtype=np.float64)
        b = b[None] if b.shape == (4, 4) else b
        if b.ndim != 3 or b.shape[1:] != (4, 4) or b.shape[0] not in (1, n):
            raise ValueError("base must be SE3, (4,4) or (1|N, 4, 4)")
        return b

    def fk_batch(self, q, base=None) -> np.ndarray:
        """World poses of all links: q (N, nq) -> (N, L, 4, 4). One kernel call."""
        q = self._q2d(q)
        k = _backend.batch_kernel(q.shape[0])
        return k.fk_tree_batch(self._parent, self._offset, self._axis, self._qidx,
                               self._jtype, q, self._base3d(base, q.shape[0]))

    def fk(self, q, base=None) -> TreePoses:
        return TreePoses(self, self.fk_batch(q, base)[0])

    # -- mass properties ------------------------------------------------------------- #
    def link_coms(self, poses):
        """World CoM of every link: (..., L, 4, 4) -> (..., L, 3)."""
        m = poses.matrices if isinstance(poses, TreePoses) else np.asarray(poses)
        return np.einsum("...lij,lj->...li", m[..., :3, :3], self.coms_local) + m[..., :3, 3]

    def com(self, poses):
        """Whole-body CoM: (..., L, 4, 4) -> (..., 3)."""
        return np.einsum("...li,l->...i", self.link_coms(poses), self.masses) / self.total_mass

    # -- Jacobians ------------------------------------------------------------------------ #
    def jacobian_space(self, poses, link) -> np.ndarray:
        """Space Jacobian (6, nq), twist order (omega, v) at the world origin.
        Columns of joints not on the path to ``link`` are zero."""
        m = poses.matrices if isinstance(poses, TreePoses) else np.asarray(poses)
        return _core_py.space_jacobian(m, self._axis, self._qidx, self._jtype,
                                       self._chain[self.link_id(link)], self.nq)

    def jacobian_point(self, poses, link, point=(0.0, 0.0, 0.0)) -> np.ndarray:
        """(6, nq): rows 0-2 angular velocity of the link, rows 3-5 linear
        velocity of ``point`` (given in the link frame), both in world axes."""
        m = poses.matrices if isinstance(poses, TreePoses) else np.asarray(poses)
        js = self.jacobian_space(m, link)
        t = m[self.link_id(link)]
        p = t[:3, :3] @ as_array(point, (3,), "point") + t[:3, 3]
        out = js.copy()
        out[3:] = js[3:] + np.cross(js[:3].T, p).T  # v_p = v + omega x p
        return out

    # -- inverse kinematics ------------------------------------------------------------------ #
    def ik(self, target: SE3, link: str, q0, base=None, joints: Optional[Sequence[int]] = None,
           iters: int = 100, tol: float = 1e-9, damping: float = 1e-4) -> Tuple[np.ndarray, IKResult]:
        """Damped least squares on SE(3).

        Error  : body twist  xi = log(T_world_link^-1 @ target)   (SE3.log)
        Jacobian: body Jacobian  J_b = Ad(T_world_link^-1) @ J_space   (SE3.adjoint)
        Step   : dq = J_b^T (J_b J_b^T + damping^2 I)^-1 xi, then joint limits.

        ``joints`` restricts the solve to a subset of q (default: the chain to
        ``link``); all other joints keep their values from q0. The iterations
        run in the kernel (Cython: the whole loop in C, ~1 us per iteration).
        """
        if not isinstance(target, SE3):
            raise TypeError("target must be an SE3")
        if target.parent not in (None, self.base_frame) or target.child not in (None, link):
            raise FrameMismatchError(
                f"target must be T_{self.base_frame}_{link}, got T_{target.parent}_{target.child}")
        q0 = np.asarray(q0, dtype=np.float64)
        if q0.shape != (self.nq,):
            raise ValueError(f"q0 must have shape ({self.nq},), got {q0.shape}")
        sel = self.chain_joints(link) if joints is None else np.asarray(joints, dtype=int)
        q, ok, it, err = _backend.kernel.ik_dls(
            self._parent, self._offset, self._axis, self._qidx, self._jtype, q0,
            self._base3d(base, 1)[0], self.link_id(link), target.matrix, sel,
            self._lower, self._upper, iters, tol, damping)
        return q, IKResult(bool(ok), int(it), float(err))
