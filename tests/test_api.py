"""Behaviour of the public classes: invariants, immutability, frames, batches."""
import copy
import pickle

import numpy as np
import pytest

from robolab.transforms import (SE3, SO3, FrameMismatchError, InvalidElementError, SE3Array,
                                rotz)


# -- validation ---------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    np.diag([1.0, 1.0, -1.0]),               # reflection
    2 * np.eye(3),                           # scale
    np.eye(3) + 1e-6,                        # drift above tolerance
])
def test_so3_rejects_invalid(bad):
    with pytest.raises(InvalidElementError):
        SO3(bad)


def test_non_finite_and_wrong_shape_are_value_errors():
    with pytest.raises(ValueError, match="finite"):
        SO3(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="shape"):
        SO3(np.eye(4))


def test_se3_rejects_bad_last_row_and_shapes():
    m = np.eye(4)
    m[3, 0] = 0.1
    with pytest.raises(InvalidElementError):
        SE3.from_matrix(m)
    with pytest.raises(ValueError):
        SE3(np.eye(3), [1, 2])
    with pytest.raises(ValueError):
        SO3(np.eye(4))


def test_float32_data_needs_explicit_projection():
    f32 = np.float32(rotz(0.5)).astype(float)
    with pytest.raises(InvalidElementError):
        SO3(f32)
    assert SO3.from_matrix(f32, project=True).isclose(SO3.Rz(0.5), 1e-6)
    assert SO3(f32, tol=1e-6)  # or a looser tolerance, explicitly


# -- immutability ---------------------------------------------------------------- #
def test_immutable():
    src = np.eye(3)
    r = SO3(src)
    src[0, 0] = 5.0                          # input is copied
    assert r.matrix[0, 0] == 1.0
    with pytest.raises(ValueError):
        r.matrix[0, 0] = 2.0                 # returned view is read-only
    with pytest.raises(ValueError):
        r.matrix.setflags(write=True)
    with pytest.raises(AttributeError):
        r._m = np.eye(3)
    t = SE3.Trans(1, 2, 3)
    with pytest.raises(ValueError):
        t.translation[0] = 0.0
    with pytest.raises(ValueError):
        t.R[0, 0] = 0.0


def test_eq_and_hash_are_explicitly_disabled():
    with pytest.raises(TypeError):
        SO3.identity() == SO3.identity()
    with pytest.raises(TypeError):
        hash(SE3.identity())
    assert SO3.identity().isclose(SO3.identity())


def test_copy_and_pickle():
    t = SE3.Trans(1, 2, 3, parent="world", child="base")
    assert copy.copy(t) is t and copy.deepcopy(t) is t
    u = pickle.loads(pickle.dumps(t))
    assert u.isclose(t) and u.parent == "world" and u.child == "base"


def test_numpy_interop():
    t = SE3.Trans(1, 2, 3)
    assert np.asarray(t).shape == (4, 4)
    assert (np.array(t, copy=True) == t.matrix).all()


# -- frames ------------------------------------------------------------------------ #
def test_frame_chain():
    t_w_f = SE3.Trans(1, 0, 0, parent="world", child="foot")
    t_f_t = SE3.Trans(0.1, 0, 0, parent="foot", child="toe")
    t_w_t = t_w_f @ t_f_t
    assert (t_w_t.parent, t_w_t.child) == ("world", "toe")
    inv = t_w_t.inverse()
    assert (inv.parent, inv.child) == ("toe", "world")
    with pytest.raises(FrameMismatchError, match="'foot' != 'world'"):
        t_w_f @ t_w_f


def test_untagged_frames_are_not_checked():
    a = SE3.Trans(1, 0, 0, parent="world", child="foot")
    b = SE3.Trans(0, 1, 0)
    c = a @ b
    assert c.parent == "world" and c.child is None


def test_mixing_groups_is_a_type_error():
    with pytest.raises(TypeError):
        SE3.identity() @ SO3.identity()


def test_interpolate_requires_same_frames():
    a = SE3.identity(parent="w", child="a")
    b = SE3.identity(parent="w", child="b")
    with pytest.raises(FrameMismatchError):
        a.interpolate(b, 0.5)


def test_rotation_property_keeps_frames():
    t = SE3(SO3.Rx(0.3), [1, 2, 3], parent="w", child="b")
    r = t.rotation
    assert isinstance(r, SO3) and r.parent == "w" and r.child == "b"
    assert r.isclose(SO3.Rx(0.3, parent="w", child="b"))


# -- batches -------------------------------------------------------------------------- #
def test_se3_array_basics(rng):
    xi = rng.normal(size=(8, 6))
    arr = SE3Array.exp(xi, parent="world", child="foot")
    assert len(arr) == 8
    assert isinstance(arr[0], SE3) and arr[0].parent == "world"
    assert isinstance(arr[2:5], SE3Array) and len(arr[2:5]) == 3
    assert np.allclose(arr.log(), np.stack([SE3.exp(x).log() for x in xi]), atol=1e-12)
    for i, t in enumerate(arr):
        assert t.isclose(SE3.exp(xi[i], parent="world", child="foot"), 1e-14)


def test_se3_array_broadcasting(rng):
    a = SE3Array.exp(rng.normal(size=(6, 6)), parent="w", child="f")
    b = SE3Array.exp(rng.normal(size=(6, 6)), parent="f", child="t")
    s = SE3.exp(rng.normal(size=6), parent="f", child="t")
    s_left = SE3.exp(rng.normal(size=6), parent="x", child="w")
    for i in range(6):
        assert (a @ b)[i].isclose(a[i] @ b[i], 1e-12)
        assert (a @ s)[i].isclose(a[i] @ s, 1e-12)
        assert (s_left @ a)[i].isclose(s_left @ a[i], 1e-12)
        assert a.inverse()[i].isclose(a[i].inverse(), 1e-12)
    with pytest.raises(FrameMismatchError):
        a @ a
    with pytest.raises(ValueError):
        SE3Array.exp(rng.normal(size=(6, 6))) @ SE3Array.exp(rng.normal(size=(4, 6)))


def test_se3_array_apply(rng):
    arr = SE3Array.exp(rng.normal(size=(5, 6)))
    pts = rng.normal(size=(7, 3))
    out = arr.apply(pts)
    assert out.shape == (5, 7, 3)
    assert np.allclose(out[3], arr[3].act(pts), atol=1e-14)
    assert arr.apply(pts[0]).shape == (5, 3)
    each = arr.apply_each(pts[:5])
    assert np.allclose(each[2], arr[2].act(pts[2]), atol=1e-14)
    with pytest.raises(ValueError):
        arr.apply_each(pts)


def test_se3_array_from_list_and_validation():
    lst = [SE3.Trans(i, 0, 0, parent="w", child="b") for i in range(3)]
    arr = SE3Array(lst)
    assert (arr.parent, arr.child) == ("w", "b")
    with pytest.raises(FrameMismatchError):
        SE3Array([SE3.identity(parent="w", child="a"), SE3.identity(parent="w", child="b")])
    bad = np.tile(np.eye(4), (3, 1, 1))
    bad[1, 0, 0] = 2.0
    with pytest.raises(InvalidElementError):
        SE3Array(bad)
    assert len(SE3Array([])) == 0
