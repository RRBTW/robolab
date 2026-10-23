import numpy as np
import pytest

from robolab.transforms import available_backends, get_backend, set_backend


@pytest.fixture(params=available_backends(), autouse=True)
def backend(request):
    """Every test runs once per available kernel (cython and python)."""
    old = get_backend()
    set_backend(request.param)
    yield request.param
    set_backend(old)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def random_unit(rng):
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


def expm(a, terms=30):
    """Reference matrix exponential: scaling and squaring + Taylor."""
    n = max(0, int(np.ceil(np.log2(max(np.abs(a).max(), 1e-300)))) + 1)
    a = a / 2 ** n
    out, term = np.eye(len(a)), np.eye(len(a))
    for k in range(1, terms):
        term = term @ a / k
        out = out + term
    for _ in range(n):
        out = out @ out
    return out


# Angles that stress every branch: series, switch points, generic, near pi, pi.
EDGE_ANGLES = [0.0, 1e-12, 1e-6, 9.99e-4, 1.001e-3, 0.1, 1.0, 2.5,
               np.pi - 2e-3, np.pi - 1.001e-3, np.pi - 9.99e-4, np.pi - 1e-6,
               np.pi - 1e-10, np.pi]
