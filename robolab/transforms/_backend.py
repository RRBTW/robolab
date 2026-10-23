"""Kernel selection.

The compiled Cython kernel is used when it is available. Set the environment
variable ROBOLAB_PURE_PYTHON=1 (before import) to force the NumPy reference
kernel, or call set_backend() at runtime (mostly for tests and benchmarks).

Batch operations (SE3Array, KinematicTree.fk_batch) can additionally be sent
to the GPU with set_batch_backend("cuda"). Batches shorter than ``min_batch``
and all single-element operations stay on the CPU kernel. CuPy is imported
only on that call, so CPU-only machines never pay for it.
"""
import os

from . import _core_py

try:
    from . import _core_cy
except ImportError:  # not compiled: Cython or a C compiler was missing
    _core_cy = None

_KERNELS = {"python": _core_py}
if _core_cy is not None:
    _KERNELS["cython"] = _core_cy

if os.environ.get("ROBOLAB_PURE_PYTHON") == "1" or _core_cy is None:
    BACKEND = "python"
else:
    BACKEND = "cython"
kernel = _KERNELS[BACKEND]


def available_backends():
    return tuple(_KERNELS)


def set_backend(name):
    """Switch the numeric kernel: 'cython' or 'python'."""
    global kernel, BACKEND
    if name not in _KERNELS:
        raise ValueError(f"backend {name!r} is not available; have {available_backends()}")
    kernel = _KERNELS[name]
    BACKEND = name


def get_backend():
    return BACKEND


def openmp_enabled():
    """True if the compiled kernel has OpenMP (batch kernels run in parallel)."""
    return _core_cy is not None and _core_cy.openmp_enabled()


def set_num_threads(n):
    """Threads used by batch kernels of the Cython backend (1 = serial)."""
    if _core_cy is not None:
        _core_cy.set_num_threads(int(n))


def get_num_threads():
    return _core_cy.get_num_threads() if _core_cy is not None else 1


_core_cuda = None
_batch = "cpu"
_batch_min = 4096


def _load_cuda():
    global _core_cuda
    if _core_cuda is None:
        from . import _core_cuda as mod  # ImportError says what is missing
        _core_cuda = mod
    return _core_cuda


def cuda_available():
    """True if CuPy is installed and a CUDA device is visible."""
    try:
        _load_cuda()
    except ImportError:
        return False
    return True


def set_batch_backend(name, *, min_batch=None, dtype=None):
    """Where batch operations run: 'cpu' (the current kernel) or 'cuda'.

    min_batch: batches shorter than this stay on the CPU (launch + copies
    cost more than the work). dtype: CUDA compute precision, float32 or float64.
    """
    global _batch, _batch_min
    if name not in ("cpu", "cuda"):
        raise ValueError(f"batch backend must be 'cpu' or 'cuda', not {name!r}")
    if name == "cuda" or dtype is not None:
        cuda = _load_cuda()
        if dtype is not None:
            cuda.set_dtype(dtype)
    _batch = name
    if min_batch is not None:
        _batch_min = max(0, int(min_batch))


def get_batch_backend():
    return _batch


def batch_kernel(n):
    """Kernel for a batch of n elements."""
    return _core_cuda if _batch == "cuda" and n >= _batch_min else kernel
