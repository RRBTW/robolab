"""Independent cross-check against Peter Corke's spatialmath-python."""
import numpy as np
import pytest

from conftest import EDGE_ANGLES, random_unit
from robolab.transforms import SE3, SO3

sm = pytest.importorskip("spatialmath")
smb = pytest.importorskip("spatialmath.base")


def test_compose_inverse_act(rng):
    for _ in range(100):
        a, b = SE3.exp(rng.normal(size=6)), SE3.exp(rng.normal(size=6))
        sa, sb = sm.SE3(a.matrix, check=False), sm.SE3(b.matrix, check=False)
        assert np.allclose((a @ b).matrix, (sa * sb).A, atol=1e-12)
        assert np.allclose(a.inverse().matrix, sa.inv().A, atol=1e-12)
        p = rng.normal(size=3)
        assert np.allclose(a.act(p), (sa * p).ravel(), atol=1e-12)


@pytest.mark.parametrize("angle", [a for a in EDGE_ANGLES if a < np.pi - 1e-6])
def test_logs_agree(rng, angle):
    for _ in range(20):
        w = angle * random_unit(rng)
        r = SO3.exp(w)
        assert np.allclose(r.log(), smb.trlog(r.matrix, twist=True, check=False), atol=1e-9)
        xi = np.r_[w, rng.normal(size=3)]
        t = SE3.exp(xi)
        s = smb.trlog(t.matrix, twist=True, check=False)  # spatialmath order: (v, omega)
        assert np.allclose(t.log(), np.r_[s[3:], s[:3]], atol=1e-9)


def test_exp_agrees(rng):
    for _ in range(100):
        xi = rng.normal(size=6)
        ref = smb.trexp(np.r_[xi[3:], xi[:3]])  # (v, omega)
        assert np.allclose(SE3.exp(xi).matrix, ref, atol=1e-12)
