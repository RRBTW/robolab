"""robolab v0.1 functional API keeps working unchanged."""
import numpy as np
import pytest

from robolab.transforms import (Rotation3D, Transform3D, exp_so3, log_so3, plot_frame,
                                project_so3, rotx, roty, rotz, se3, se3_inv, skew, unskew)


def test_elementary_rotations():
    assert np.allclose(rotx(np.pi / 2) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(roty(np.pi / 2) @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(rotz(np.pi / 2) @ [1, 0, 0], [0, 1, 0])
    with pytest.raises(ValueError):
        rotx([1.0, 2.0])


def test_skew_unskew(rng):
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))
    assert np.allclose(unskew(skew(a)), a)
    with pytest.raises(ValueError):
        unskew(np.eye(3))


def test_exp_log_so3(rng):
    w = rng.normal(size=3)
    w *= 2.0 / np.linalg.norm(w)
    assert np.allclose(log_so3(exp_so3(w)), w)
    with pytest.raises(ValueError):
        log_so3(2 * np.eye(3))


def test_se3_and_inverse(rng):
    r = exp_so3(rng.normal(size=3))
    t = se3(r, [1, 2, 3])
    assert np.allclose(t @ se3_inv(t), np.eye(4))
    with pytest.raises(ValueError):
        se3(np.diag([1, 1, -1]), [0, 0, 0])


def test_project():
    m = rotz(0.3) + 1e-3
    assert np.isclose(np.linalg.det(project_so3(m)), 1.0)


def test_old_class_names():
    r = Rotation3D(rotz(0.2))
    assert Rotation3D.from_axis_angle([0, 0, 0.2]).isclose(r)
    t = Transform3D(r, [1, 0, 0])
    assert np.allclose(t.apply([0, 0, 0]), [1, 0, 0])
    assert (t @ t.inverse()).isclose(Transform3D.identity())
    assert Transform3D.from_matrix(t.matrix).isclose(t)
    assert isinstance(t.rotation, Rotation3D)


def test_plot_frame_smoke():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    ax = plot_frame(se3(np.eye(3), [0, 0, 0]), label="base")
    assert ax is not None
