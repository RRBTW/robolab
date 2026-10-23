"""Kinematic tree, humanoid model and walking pipeline (both kernels)."""
import numpy as np
import pytest

from robolab.humanoid.model import LEG_JOINTS, make_humanoid
from robolab.kinematics import KinematicTree, Link
from robolab.transforms import SE3, SO3, FrameMismatchError, openmp_enabled, set_num_threads
from robolab.transforms import _core_py as P


@pytest.fixture(scope="module")
def robot():
    return make_humanoid()


def fk_reference(tree, q, base):
    """Independent FK: compose SE3 objects link by link."""
    out = []
    for lk in tree.links:
        parent = base if lk.parent is None else out[tree.link_id(lk.parent)]
        t = parent @ lk.offset
        if lk.joint is not None:
            v = q[tree.joint_id(lk.joint)]
            ax = np.asarray(lk.axis) / np.linalg.norm(lk.axis)
            if lk.type == "revolute":
                t = t @ SE3.from_rotation(SO3.exp(v * ax))
            else:
                t = t @ SE3.from_translation(v * ax)
        out.append(t)
    return np.stack([o.matrix for o in out])


def random_q(tree, rng, n=None):
    lo, hi = np.maximum(tree.limits[:, 0], -2), np.minimum(tree.limits[:, 1], 2)
    return rng.uniform(lo, hi, size=(tree.nq,) if n is None else (n, tree.nq))


def test_fk_matches_se3_composition(robot, rng):
    for _ in range(10):
        q = random_q(robot, rng)
        base = SE3.exp(rng.normal(size=6))
        assert np.allclose(robot.fk_batch(q, base)[0], fk_reference(robot, q, base), atol=1e-12)


def test_fk_batch_equals_single_and_broadcasts_base(robot, rng):
    q = random_q(robot, rng, 50)
    bases = np.stack([SE3.exp(rng.normal(size=6)).matrix for _ in range(50)])
    batch = robot.fk_batch(q, bases)
    for i in (0, 17, 49):
        assert np.allclose(batch[i], robot.fk(q[i], bases[i]).matrices, atol=1e-14)


def test_fk_kernel_parity_and_threads(robot, rng):
    C = pytest.importorskip("robolab.transforms._core_cy")
    q = np.ascontiguousarray(random_q(robot, rng, 3000))
    args = (robot._parent, robot._offset, robot._axis, robot._qidx, robot._jtype)
    base = np.eye(4)[None]
    ref = P.fk_tree_batch(*args, q, base)
    for threads in (1, 2, 4):
        set_num_threads(threads)
        assert np.allclose(C.fk_tree_batch(*args, q, base), ref, atol=1e-13)
    set_num_threads(8)


def test_prismatic_and_fixed():
    tree = KinematicTree([
        Link("a", None, SE3.Trans(1, 0, 0), "slide", axis=(0, 0, 2), type="prismatic"),
        Link("b", "a", SE3.Trans(0, 1, 0)),
    ])
    poses = tree.fk([0.5])
    assert np.allclose(poses.position("a"), [1, 0, 0.5])
    assert np.allclose(poses.position("b"), [1, 1, 0.5])
    assert poses["b"].parent == "world" and poses["b"].child == "b"


def test_tree_validation():
    with pytest.raises(ValueError, match="listed before"):
        KinematicTree([Link("b", "a", SE3.identity()), Link("a", None, SE3.identity())])
    with pytest.raises(ValueError, match="duplicate"):
        KinematicTree([Link("a", None, SE3.identity()), Link("a", None, SE3.identity())])


def test_jacobians_match_finite_differences(robot, rng):
    q = random_q(robot, rng)
    base = SE3.exp(rng.normal(size=6))
    poses = robot.fk(q, base)
    pt = np.array([0.03, -0.02, -0.05])
    jp = robot.jacobian_point(poses, "l_sole", pt)
    h = 1e-6
    for j in robot.chain_joints("l_sole"):
        dq = np.zeros(robot.nq); dq[j] = h
        a, b = robot.fk(q + dq, base)["l_sole"], robot.fk(q - dq, base)["l_sole"]
        assert np.allclose((a.act(pt) - b.act(pt)) / (2 * h), jp[3:, j], atol=1e-7)
        w = SO3((b.inverse() @ a).R).log() / (2 * h)            # body-frame angular velocity
        assert np.allclose(b.R @ w, jp[:3, j], atol=1e-6)
    off_chain = robot.joint_id("r_knee")
    assert np.allclose(jp[:, off_chain], 0)


def test_ik_reaches_target(robot, rng):
    q0 = np.zeros(robot.nq)
    for side in "lr":
        q0[robot.joint_id(f"{side}_knee")] = 0.6
        q0[robot.joint_id(f"{side}_hip_pitch")] = -0.3
        q0[robot.joint_id(f"{side}_ankle_pitch")] = -0.3
    base = SE3.Trans(0.0, 0.0, 0.68)
    joints = [robot.joint_id(j) for j in LEG_JOINTS["l"]]
    target = SE3(SO3.Rz(0.1), [0.08, 0.12, 0.03], parent="world", child="l_sole")
    q, res = robot.ik(target, "l_sole", q0, base=base, joints=joints)
    assert res.converged and res.iterations < 30
    assert robot.fk(q, base)["l_sole"].isclose(target, 1e-8)
    assert np.allclose(q[np.setdiff1d(np.arange(robot.nq), joints)],
                       q0[np.setdiff1d(np.arange(robot.nq), joints)])
    with pytest.raises(FrameMismatchError):
        robot.ik(SE3.identity(parent="world", child="r_sole"), "l_sole", q0, base=base)


@pytest.fixture(scope="module")
def walks(robot):
    """One walking simulation per kernel; the python kernel is the spec."""
    from robolab.humanoid.gait import GaitParams
    from robolab.humanoid.walk import simulate_walk
    from robolab.transforms import available_backends, get_backend, set_backend
    out, old = {}, get_backend()
    for name in available_backends():
        set_backend(name)
        out[name] = simulate_walk(robot, GaitParams(n_steps=2), com_passes=3)
    set_backend(old)
    return out


def test_walk_same_on_all_kernels(walks):
    """IK runs in the kernel (whole DLS loop in C for Cython): same result."""
    ref = walks["python"]
    for w in walks.values():
        assert w.ik_iterations == ref.ik_iterations
        assert np.abs(w.q - ref.q).max() < 1e-12
        assert np.abs(w.margin - ref.margin).max() < 1e-12


@pytest.fixture(scope="module")
def three_gaits():
    from robolab.humanoid.gait import GaitParams
    return [GaitParams(step_length=0.2, step_time=st, n_steps=2) for st in (0.4, 0.6, 0.8)]


def test_simulate_walks_equals_one_by_one(robot, three_gaits, backend):
    """Gaits of different length in one batch: identical to separate runs."""
    if backend != "cython":
        pytest.skip("batching is kernel-independent; the NumPy spec IK is slow")
    from robolab.humanoid.walk import simulate_walk, simulate_walks
    batch = simulate_walks(robot, three_gaits, com_passes=2)
    for g, w in zip(three_gaits, batch):
        one = simulate_walk(robot, g, com_passes=2)
        assert len(w.t) == len(one.t)
        assert np.array_equal(w.q, one.q) and np.array_equal(w.margin, one.margin)
        assert w.ik_iterations == one.ik_iterations and w.ik_failures == one.ik_failures == 0
    with pytest.raises(ValueError):
        simulate_walks(robot, three_gaits, device="tpu")


def test_simulate_walks_cuda_matches_cpu(robot, three_gaits, backend):
    """GPU_PLAN 3b criterion: the sweep agrees on the minimum ZMP margin to ~0.1 mm.
    Float64 on the GPU gives far more: the same IK path up to rounding."""
    from robolab.transforms import cuda_available
    if backend != "cython" or not cuda_available():
        pytest.skip("needs the Cython kernel and a CUDA device")
    from robolab.humanoid.walk import simulate_walks
    cpu = simulate_walks(robot, three_gaits, com_passes=2)
    gpu = simulate_walks(robot, three_gaits, com_passes=2, device="cuda")
    for a, b in zip(cpu, gpu):
        assert abs(a.margin.min() - b.margin.min()) < 1e-7          # 0.1 um, criterion 0.1 mm
        assert np.abs(a.q - b.q).max() < 1e-9
        assert a.ik_iterations == b.ik_iterations and b.ik_failures == 0


def test_walk_pipeline_is_balanced(robot, walks, backend):
    w = walks[backend]
    assert w.ik_failures == 0
    assert w.com_iterations[-1] < 2e-3                     # CoM follows the plan
    assert (w.margin > 0).all()                            # ZMP inside support polygon
    ls = robot.link_id("l_sole")
    assert np.allclose(w.poses[:, ls, :3, 3], w.feet_ref["l"], atol=1e-6)
    lim = robot.limits
    assert ((w.q >= lim[:, 0] - 1e-12) & (w.q <= lim[:, 1] + 1e-12)).all()


def test_encoder_mc_scales_with_resolution(robot, rng):
    from robolab.humanoid.analysis import encoder_quantization_mc
    q = np.zeros(robot.nq)
    q[robot.joint_id("l_knee")] = q[robot.joint_id("r_knee")] = 0.6
    r12 = encoder_quantization_mc(robot, q, None, 12, 4000, rng)
    r16 = encoder_quantization_mc(robot, q, None, 16, 4000, rng)
    e12 = np.linalg.norm(r12["pos"], axis=1).mean()
    e16 = np.linalg.norm(r16["pos"], axis=1).mean()
    assert 12 < e12 / e16 < 20                              # ~2**4 = 16 times smaller


def test_parallel_batch_kernels_match_serial(rng):
    C = pytest.importorskip("robolab.transforms._core_cy")
    t = P.se3_exp_batch(rng.normal(size=(5000, 6)))
    u = P.se3_exp_batch(rng.normal(size=(5000, 6)))
    pts = rng.normal(size=(300, 3))
    set_num_threads(1)
    serial = [C.se3_mul_batch(t, u), C.se3_inv_batch(t), C.se3_log_batch(t),
              C.se3_act_batch(t, pts), C.se3_exp_batch(rng.normal(size=(0, 6)))]
    set_num_threads(4)
    par = [C.se3_mul_batch(t, u), C.se3_inv_batch(t), C.se3_log_batch(t),
           C.se3_act_batch(t, pts), C.se3_exp_batch(rng.normal(size=(0, 6)))]
    for a, b in zip(serial, par):
        assert np.array_equal(a, b)
    bad = t.copy(); bad[4321, 0, 0] = 2.0
    assert C.se3_check_batch(t, 1e-9) and not C.se3_check_batch(bad, 1e-9)
    assert isinstance(openmp_enabled(), bool)
