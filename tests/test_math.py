"""Mathematical properties, run against every kernel."""
import numpy as np
import pytest

from conftest import EDGE_ANGLES, expm, random_unit
from robolab.transforms import SE3, SO3, project_so3, skew


def twist_matrix(xi):
    m = np.zeros((4, 4))
    m[:3, :3] = skew(xi[:3])
    m[:3, 3] = xi[3:]
    return m


@pytest.mark.parametrize("angle", EDGE_ANGLES)
def test_so3_exp_matches_matrix_exponential(rng, angle):
    for _ in range(20):
        w = angle * random_unit(rng)
        assert np.allclose(SO3.exp(w).matrix, expm(skew(w)), atol=1e-12)


@pytest.mark.parametrize("angle", EDGE_ANGLES)
def test_se3_exp_matches_matrix_exponential(rng, angle):
    for _ in range(20):
        xi = np.r_[angle * random_unit(rng), rng.normal(size=3)]
        assert np.allclose(SE3.exp(xi).matrix, expm(twist_matrix(xi)), atol=1e-12)


@pytest.mark.parametrize("angle", EDGE_ANGLES)
def test_so3_log_roundtrip(rng, angle):
    for _ in range(20):
        w = angle * random_unit(rng)
        r = SO3.exp(w)
        back = r.log()
        assert np.linalg.norm(back) <= np.pi + 1e-12
        assert SO3.exp(back).isclose(r, 1e-12)
        if angle < np.pi - 1e-6:  # axis sign is only ambiguous at pi
            assert np.allclose(back, w, atol=1e-9)


@pytest.mark.parametrize("angle", EDGE_ANGLES)
def test_se3_log_roundtrip(rng, angle):
    for _ in range(20):
        xi = np.r_[angle * random_unit(rng), rng.normal(size=3)]
        t = SE3.exp(xi)
        assert SE3.exp(t.log()).isclose(t, 1e-11)
        if angle < np.pi - 1e-6:
            assert np.allclose(t.log(), xi, atol=1e-9)


def test_inverse_and_associativity(rng):
    a, b, c = (SE3.exp(rng.normal(size=6)) for _ in range(3))
    assert (a @ a.inverse()).isclose(SE3.identity(), 1e-12)
    assert ((a @ b) @ c).isclose(a @ (b @ c), 1e-12)
    assert (a @ b).inverse().isclose(b.inverse() @ a.inverse(), 1e-12)


def test_act_matches_homogeneous_multiplication(rng):
    t = SE3.exp(rng.normal(size=6))
    pts = rng.normal(size=(10, 3))
    hom = (t.matrix @ np.c_[pts, np.ones(10)].T).T[:, :3]
    assert np.allclose(t.act(pts), hom, atol=1e-14)
    assert np.allclose(t.act(pts[0]), hom[0], atol=1e-14)
    assert np.allclose(t.act_vector(pts[0]), t.R @ pts[0], atol=1e-14)


def test_adjoint_identity(rng):
    """T exp(xi) T^-1 == exp(Ad_T xi)."""
    for _ in range(20):
        t, xi = SE3.exp(rng.normal(size=6)), 0.5 * rng.normal(size=6)
        lhs = t @ SE3.exp(xi) @ t.inverse()
        assert lhs.isclose(SE3.exp(t.adjoint() @ xi), 1e-11)


def test_quaternion_roundtrip(rng):
    for angle in EDGE_ANGLES:
        r = SO3.exp(angle * random_unit(rng))
        q = r.as_quaternion()
        assert q.w >= 0 and abs(np.linalg.norm(q) - 1) < 1e-14
        assert SO3.from_quaternion(**q._asdict()).isclose(r, 1e-12)


def test_project_so3_repairs_noise_and_reflection(rng):
    r = SO3.exp(rng.normal(size=3)).matrix
    noisy = r + 1e-4 * rng.normal(size=(3, 3))
    assert np.allclose(project_so3(noisy), r, atol=1e-3)
    assert SO3(project_so3(noisy))  # passes strict validation
    reflected = r @ np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(project_so3(reflected)), 1.0)


def test_interpolation():
    a = SO3.identity()
    b = SO3.Rz(1.0)
    assert a.interpolate(b, 0.0).isclose(a, 1e-14)
    assert a.interpolate(b, 1.0).isclose(b, 1e-14)
    assert a.interpolate(b, 0.25).isclose(SO3.Rz(0.25), 1e-14)
    t0, t1 = SE3.identity(), SE3.exp([0, 0, 1.0, 1.0, 0, 0])
    assert t0.interpolate(t1, 0.5).isclose(SE3.exp([0, 0, 0.5, 0.5, 0, 0]), 1e-14)


def test_long_composition_drift_stays_valid():
    step = SE3.exp([0.0123, 0.0071, -0.003, 0.01, 0.0, 0.02])
    t = SE3.identity()
    for _ in range(20000):
        t = t @ step
    assert SE3.from_matrix(t.matrix)  # still passes the 1e-9 check
    assert t.normalized().isclose(t, 1e-9)
