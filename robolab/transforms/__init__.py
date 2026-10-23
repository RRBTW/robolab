r"""robolab.transforms: rigid-body geometry for robots.

Conventions (identical everywhere in the package)
-------------------------------------------------
* Column vectors, right-handed frames, active rotations, radians, metres.
* ``T_A_B`` maps coordinates expressed in frame B into frame A.
  Compose as ``T_A_B @ T_B_C -> T_A_C``. With frame tags this is checked.
* Twists xi = (omega, v) as in Lynch & Park, "Modern Robotics".
* so3 log is principal: angle in [0, pi]; at exactly pi the axis sign is
  ambiguous.
* Invalid matrices are rejected (tol 1e-9). Repairing is explicit:
  ``SO3.from_matrix(m, project=True)`` / ``SE3.from_matrix(m, project=True)``.

Layers
------
1. Kernel (``_core_cy`` compiled with Cython, ``_core_py`` NumPy reference):
   unchecked numeric routines on float64 arrays. Not public.
2. Groups: ``SO3``, ``SE3`` (immutable values, invariant guaranteed by the
   constructor, optional frame tags) and ``SE3Array`` (batches).
3. Array API (``functional``): validated functions on plain NumPy arrays,
   compatible with robolab v0.1.
"""
from ._backend import (available_backends, cuda_available, get_backend, get_batch_backend,
                       get_num_threads, openmp_enabled, set_backend, set_batch_backend,
                       set_num_threads)
from .functional import (exp_se3, exp_so3, log_se3, log_so3, plot_frame, project_so3,
                         rotx, roty, rotz, se3, se3_inv, skew, unskew)
from .lie import DEFAULT_TOL, FrameMismatchError, InvalidElementError, LieGroup
from .se3_group import SE3
from .se3_array import SE3Array
from .so3 import SO3, Quaternion

# robolab v0.1 names
Rotation3D = SO3
Transform3D = SE3

__all__ = [
    "SO3", "SE3", "SE3Array", "LieGroup", "Quaternion",
    "Rotation3D", "Transform3D",
    "FrameMismatchError", "InvalidElementError", "DEFAULT_TOL",
    "rotx", "roty", "rotz", "skew", "unskew", "exp_so3", "log_so3",
    "exp_se3", "log_se3", "se3", "se3_inv", "project_so3", "plot_frame",
    "get_backend", "set_backend", "available_backends",
    "openmp_enabled", "set_num_threads", "get_num_threads",
    "set_batch_backend", "get_batch_backend", "cuda_available",
]
