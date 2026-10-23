"""Compiled kernels (Cython, CUDA) must reproduce the NumPy reference kernel
on every function, including edge angles and empty batches.

Tolerances: Cython 1e-13 and CUDA float64 1e-12 (rounding only); CUDA float32
1e-5 absolute + relative. CUDA tests are skipped without CuPy or a GPU."""
import functools

import numpy as np
import pytest

from conftest import EDGE_ANGLES, random_unit
from robolab.kinematics import KinematicTree, Link
from robolab.transforms import SE3
from robolab.transforms import _core_py as P

try:
    from robolab.transforms import _core_cy as C
except ImportError:
    C = None
try:
    from robolab.transforms import _core_cuda as G
except ImportError:
    G = None

needs_cy = pytest.mark.skipif(C is None, reason="Cython kernel not compiled")
needs_cuda = pytest.mark.skipif(G is None, reason="no CuPy or no CUDA device")

# float32 series / near-pi switch points of the CUDA kernel
F32_EDGES = [0.0499, 0.0501, np.pi - 0.0501, np.pi - 0.0499]


class _Cuda:
    """CUDA kernel with a fixed compute dtype."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __getattr__(self, name):
        return functools.partial(getattr(G, name), dtype=self.dtype)


@pytest.fixture(params=[pytest.param(("cython", 1e-13), marks=needs_cy, id="cython"),
                        pytest.param(("cuda-f64", 1e-12), marks=needs_cuda, id="cuda-f64"),
                        pytest.param(("cuda-f32", 1e-5), marks=needs_cuda, id="cuda-f32")])
def kern(request):
    name, tol = request.param
    if name == "cython":
        return name, C, tol
    return name, _Cuda(np.float64 if name == "cuda-f64" else np.float32), tol


@pytest.fixture(autouse=True)
def backend():  # parity tests compare kernels directly; no backend switching
    yield


@pytest.fixture(scope="module")
def robot():
    from robolab.humanoid import make_humanoid
    return make_humanoid()


@pytest.fixture(scope="module")
def mixed_tree():
    """Revolute, prismatic and fixed joints, two branches."""
    return KinematicTree([
        Link("a", None, SE3.Trans(0, 0, 0.1), "j1", (0, 0, 1), mass=2.0, com=(0, 0, 0.05)),
        Link("b", "a", SE3.Trans(0.2, 0, 0) @ SE3.Rx(0.4), "j2", (1, 0, 0), type="prismatic",
             mass=1.0, com=(0.1, 0, 0)),
        Link("c", "b", SE3.Trans(0, 0.1, 0), mass=0.5, com=(0, 0.02, 0)),
        Link("d", "a", SE3.Ry(-0.7), "j3", (0, 1, 1), mass=1.5, com=(0.15, 0, 0)),
        Link("e", "d", SE3.Trans(0.3, 0, 0), "j4", (1, 0, 0), mass=0.7, com=(0.05, 0.01, 0)),
    ])


def _tree(t):
    return t._parent, t._offset, t._axis, t._qidx, t._jtype


def _close(got, ref, tol):
    return np.allclose(got, ref, atol=tol, rtol=tol)


@needs_cy
def test_single_element_functions(rng):
    for angle in EDGE_ANGLES:
        for _ in range(20):
            xi = np.r_[angle * random_unit(rng), rng.normal(size=3)]
            r, t = P.so3_exp(xi[:3]), P.se3_exp(xi)
            pts = rng.normal(size=(4, 3))
            for name, args in [("so3_exp", (xi[:3],)), ("se3_exp", (xi,)),
                               ("so3_log", (r,)), ("se3_log", (t,)), ("se3_inv", (t,)),
                               ("se3_adjoint", (t,)), ("hat3", (xi[:3],)),
                               ("se3_mul", (t, t)), ("so3_mul", (r, r)),
                               ("se3_act", (t, pts)), ("so3_act", (r, pts))]:
                assert np.allclose(getattr(P, name)(*args), getattr(C, name)(*args),
                                   atol=1e-13, rtol=0), name
            assert P.se3_check(t, 1e-9) == C.se3_check(t, 1e-9)


def test_batch_functions(rng, kern):
    _, k, tol = kern
    t = P.se3_exp_batch(rng.normal(size=(30, 6)))
    u = P.se3_exp_batch(rng.normal(size=(30, 6)))
    one = np.ascontiguousarray(u[:1])
    pts = rng.normal(size=(9, 3))
    for name, args in [("se3_mul_batch", (t, u)), ("se3_mul_batch", (one, t)),
                       ("se3_mul_batch", (t, one)), ("se3_inv_batch", (t,)),
                       ("se3_log_batch", (t,)), ("se3_exp_batch", (P.se3_log_batch(t),)),
                       ("se3_act_batch", (t, pts)),
                       ("se3_act_each", (t, rng.normal(size=(30, 3))))]:
        assert _close(getattr(k, name)(*args), getattr(P, name)(*args), tol), name


def test_batch_edge_angles(rng, kern):
    name, k, tol = kern
    angles = np.array(EDGE_ANGLES + F32_EDGES)
    xi = np.array([np.r_[a * random_unit(rng), rng.normal(size=3)]
                   for a in angles for _ in range(20)])
    ang = np.repeat(angles, 20)
    t = P.se3_exp_batch(xi)
    assert _close(k.se3_exp_batch(xi), t, tol)

    log = k.se3_log_batch(t)
    # At pi the axis sign is ambiguous. In float32 the rounding of R decides it
    # for anything within ~1e-4 of pi, so there only exp(log(T)) == T is checked.
    amb = np.pi - ang < (1e-4 if name == "cuda-f32" else 1e-9)
    assert _close(log[~amb], P.se3_log_batch(t)[~amb], tol)
    assert _close(P.se3_exp_batch(log[amb]), t[amb], tol)


def test_empty_batches(kern, robot):
    _, k, _ = kern
    z = np.zeros((0, 4, 4))
    for kk in (P, k):
        assert kk.se3_inv_batch(z).shape == (0, 4, 4)
        assert kk.se3_log_batch(z).shape == (0, 6)
        assert kk.se3_exp_batch(np.zeros((0, 6))).shape == (0, 4, 4)
        assert kk.se3_mul_batch(z, z).shape == (0, 4, 4)
        assert kk.se3_act_batch(z, np.zeros((5, 3))).shape == (0, 5, 3)
        assert kk.se3_act_each(z, np.zeros((0, 3))).shape == (0, 3)
        assert kk.se3_check_batch(z, 1e-9)
        q = np.zeros((0, robot.nq))
        assert kk.fk_tree_batch(*_tree(robot), q, np.eye(4)[None]).shape == (0, robot.nl, 4, 4)
        out = kk.mc_relative_pose(*_mc_args(robot, np.zeros(robot.nq), np.arange(robot.nq),
                                            np.zeros((0, robot.nq))))
        assert [o.shape for o in out] == [(0, 3), (0,), (0, 3)]


def test_checks_agree_on_invalid(kern):
    _, k, _ = kern
    good = P.se3_exp_batch(np.random.default_rng(1).normal(size=(300, 6)))
    assert k.se3_check_batch(good, 1e-9)
    m = np.eye(4)
    m[3, 1] = 1e-3
    for b in [np.diag([1.0, 1.0, -1.0, 1.0]), 2 * np.eye(4), np.full((4, 4), np.nan), m]:
        mixed = good.copy()
        mixed[137] = b
        assert P.se3_check_batch(mixed, 1e-9) == k.se3_check_batch(mixed, 1e-9) == False  # noqa: E712


@needs_cy
def test_readonly_inputs_accepted():
    t = np.eye(4)
    t.setflags(write=False)
    assert np.allclose(C.se3_inv(t), np.eye(4))


@pytest.mark.parametrize("tree", ["robot", "mixed_tree"])
def test_fk_tree_batch(rng, kern, tree, request):
    _, k, tol = kern
    t = request.getfixturevalue(tree)
    q = rng.uniform(-1.0, 1.0, (300, t.nq))
    for base in (np.eye(4)[None], P.se3_exp_batch(rng.normal(size=(300, 6)))):
        ref = P.fk_tree_batch(*_tree(t), q, base)
        assert _close(k.fk_tree_batch(*_tree(t), q, base), ref, tol)


def _mc_args(t, q0, joints, dq, base=None, ref=None, target=None):
    names = t.link_names
    return (*_tree(t), q0, np.eye(4) if base is None else base, joints, dq,
            t.link_id(ref or names[-1]), t.link_id(target or names[len(names) // 2]),
            t.coms_local, t.masses)


@pytest.mark.parametrize("half_width", [np.pi / 2 ** 18, np.pi / 2 ** 10, 0.3],
                         ids=["18bit", "10bit", "0.3rad"])
@pytest.mark.parametrize("case", ["robot", "robot-left-leg", "mixed_tree"])
def test_mc_relative_pose(rng, kern, case, half_width, request):
    """Same noise in, same errors out. Tolerance is relative to the size of the
    error itself: the compiled kernels work in perturbation form, so float32
    keeps ~1e-7 of the error even when the error is 1e-5 m. For float64 the
    spec is the less accurate side (it subtracts metre-sized poses), hence 1e-8."""
    name, k, _ = kern
    t = request.getfixturevalue("mixed_tree" if case == "mixed_tree" else "robot")
    if case == "robot-left-leg":
        joints = np.array([t.joint_id(j) for j in t.joint_names if j.startswith("l_")])
        ref, target = "r_sole", "l_sole"
    else:
        joints, ref, target = np.arange(t.nq), None, None
    q0 = rng.uniform(-0.4, 0.4, t.nq)
    base = P.se3_exp(rng.normal(size=6))
    dq = rng.uniform(-half_width, half_width, (500, len(joints)))
    args = _mc_args(t, q0, joints, dq, base, ref, target)
    tol = 2e-6 if name == "cuda-f32" else 1e-8
    for got, ref_ in zip(k.mc_relative_pose(*args), P.mc_relative_pose(*args)):
        assert got.shape == ref_.shape
        assert np.abs(got - ref_).max() <= tol * np.abs(ref_).max()


def test_mc_link_limit(kern):
    _, k, _ = kern
    links = [Link(f"l{i}", None if i == 0 else f"l{i - 1}", SE3.Trans(0, 0, 0.1), f"j{i}")
             for i in range(P.MC_MAX_LINKS + 1)]
    t = KinematicTree(links)
    with pytest.raises(ValueError, match="up to"):
        k.mc_relative_pose(*_mc_args(t, np.zeros(t.nq), np.arange(t.nq), np.zeros((3, t.nq))))


def _leg_ik(robot, rng, knee, start_noise, base=None, shift=0.0):
    """IK problem for the left leg: target = FK of a random leg pose (+ shift)."""
    from robolab.humanoid.model import LEG_JOINTS
    sel = np.array([robot.joint_id(j) for j in LEG_JOINTS["l"]])
    lid = robot.link_id("l_sole")
    qt = np.zeros(robot.nq)
    qt[sel] = rng.uniform(-0.4, 0.4, len(sel))
    qt[robot.joint_id("l_knee")] = rng.uniform(*knee)
    qt = np.clip(qt, robot._lower, robot._upper)
    base = P.se3_exp(rng.normal(size=6)) if base is None else base
    target = P.fk_tree_batch(*_tree(robot), qt[None], base[None])[0, lid].copy()
    target[:3, 3] += shift
    q0 = qt.copy()
    q0[sel] += rng.normal(0, start_noise, len(sel))
    return (*_tree(robot), q0, base, lid, target, sel, robot._lower, robot._upper)


@needs_cy
def test_ik_dls_matches_spec(rng, robot):
    """Well-conditioned problems (knee bent, no limit hit): same iterations,
    same joints. The walking simulation solves exactly this kind of problem."""
    for _ in range(200):
        args = _leg_ik(robot, rng, knee=(0.4, 1.5), start_noise=0.1)
        q_p, ok_p, it_p, err_p = P.ik_dls(*args, 100, 1e-9, 1e-4)
        q_c, ok_c, it_c, err_c = C.ik_dls(*args, 100, 1e-9, 1e-4)
        assert ok_p and ok_c and it_p == it_c and err_c < 1e-9
        assert np.abs(q_c - q_p).max() < 1e-10


@needs_cy
def test_ik_dls_hard_cases(rng, robot):
    """Near the straight-knee singularity and at joint limits the iteration is
    ill-conditioned: rounding (Cholesky vs LU) may shift convergence by a few
    iterations, so here each kernel is checked on its own: an honest verdict,
    limits respected, an unreachable target never reported as reached."""
    for shift in (0.0, 2.0):                    # 2 m away: unreachable
        for _ in range(15):
            args = _leg_ik(robot, rng, knee=(0.0, 0.1), start_noise=0.4, shift=shift)
            for k in (P, C):
                q, ok, it, err = k.ik_dls(*args, 60, 1e-9, 1e-4)
                assert ((q >= robot._lower) & (q <= robot._upper)).all()
                assert ok == (err < 1e-9)
                assert ok or it == 60
                assert not (ok and shift)


@needs_cy
def test_ik_dls_prismatic_and_off_chain_joints(rng, mixed_tree, robot):
    t = mixed_tree
    lid = t.link_id("c")                       # behind revolute j1 and prismatic j2
    qt = np.array([0.4, 0.15, -0.3, 0.2])
    target = P.fk_tree_batch(*_tree(t), qt[None], np.eye(4)[None])[0, lid]
    sel = np.array([0, 1, 2])                  # j3 is on another branch: zero column
    lo, hi = np.full(t.nq, -np.inf), np.full(t.nq, np.inf)
    q0 = np.array([0.3, 0.1, -0.1, 0.2])
    for k in (P, C):
        q, ok, it, err = k.ik_dls(*_tree(t), q0, np.eye(4), lid, target, sel, lo, hi, 100, 1e-9, 1e-4)
        assert ok and q[2] == q0[2] and q[3] == q0[3]
        assert np.allclose(q[:2], qt[:2], atol=1e-9)
    args = _leg_ik(robot, rng, knee=(0.5, 1.0), start_noise=0.1)
    sel_extra = np.r_[args[9], robot.joint_id("r_knee")]
    q, ok, *_ = C.ik_dls(*args[:9], sel_extra, *args[10:], 100, 1e-9, 1e-4)
    assert ok and q[robot.joint_id("r_knee")] == args[5][robot.joint_id("r_knee")]


@pytest.fixture(params=[pytest.param("cython", marks=needs_cy),
                        pytest.param("cuda-f64", marks=needs_cuda)])
def ik_kern(request):
    """Trajectory IK: float64 only (IK residuals feed the twice-differentiated ZMP)."""
    return C if request.param == "cython" else G


def _leg_trajectories(robot, rng, n_p, n_t):
    """Smooth leg-joint trajectories -> foot targets; problems alternate legs."""
    from robolab.humanoid.model import LEG_JOINTS
    t = np.linspace(0, 1, n_t)[:, None]
    q0 = np.zeros((n_p, robot.nq))
    base = np.tile(np.eye(4), (n_p, n_t, 1, 1))
    target = np.tile(np.eye(4), (n_p, n_t, 1, 1))
    sel, link = [], []
    for p in range(n_p):
        side = "lr"[p % 2]
        s = np.array([robot.joint_id(j) for j in LEG_JOINTS[side]])
        a, b = rng.uniform(-0.3, 0.3, (2, len(s)))
        qt = np.zeros((n_t, robot.nq))
        qt[:, s] = a + (b - a) * t
        qt[:, robot.joint_id(f"{side}_knee")] = 0.6 + 0.4 * t[:, 0]
        base[p, :, :3, 3] = [0.0, 0.0, 0.68] + 0.05 * t * [1.0, 0.2, 0.0]
        target[p] = P.fk_tree_batch(*_tree(robot), qt, base[p])[:, robot.link_id(f"{side}_sole")]
        q0[p, s] = qt[0, s] + rng.normal(0, 0.05, len(s))
        sel.append(s)
        link.append(robot.link_id(f"{side}_sole"))
    return q0, base, np.array(link), target, np.array(sel)


def test_ik_dls_traj(rng, robot, ik_kern):
    q0, base, link, target, sel = _leg_trajectories(robot, rng, 6, 25)
    n_frames = np.array([25, 25, 10, 25, 1, 0])
    args = (*_tree(robot), q0, base, link, target, sel, robot._lower, robot._upper, n_frames,
            50, 1e-9, 1e-4)
    out_p, its_p, fails_p = P.ik_dls_traj(*args)
    out, its, fails = ik_kern.ik_dls_traj(*args)
    assert out.shape == (6, 25, 6) and its.shape == fails.shape == (6,)
    assert np.abs(out - out_p).max() < 1e-10
    assert (its == its_p).all() and (fails == fails_p).all() and fails.sum() == 0
    assert (out[2, 10:] == 0).all() and (out[5] == 0).all()     # past n_frames
    # every frame of a problem is the single-frame IK warm-started from the last one
    q = q0[3].copy()
    for k in range(25):
        q, ok, *_ = P.ik_dls(*_tree(robot), q, base[3, k], link[3], target[3, k], sel[3],
                             robot._lower, robot._upper, 50, 1e-9, 1e-4)
        assert np.abs(q[sel[3]] - out[3, k]).max() < 1e-10


def test_ik_dls_traj_empty(robot, ik_kern):
    z = np.zeros((0, 5, 4, 4))
    out, its, fails = ik_kern.ik_dls_traj(*_tree(robot), np.zeros((0, robot.nq)), z, np.zeros(0, int),
                                          z, np.zeros((0, 6), int), robot._lower, robot._upper,
                                          np.zeros(0, int), 50, 1e-9, 1e-4)
    assert out.shape == (0, 5, 6) and its.shape == (0,)


@needs_cuda
def test_cuda_device_arrays_stay_on_device(rng):
    import cupy as cp
    xi = rng.normal(size=(100, 6))
    host = G.se3_exp_batch(xi)
    assert isinstance(host, np.ndarray) and host.dtype == np.float64
    dev = G.se3_exp_batch(cp.asarray(xi))
    assert isinstance(dev, cp.ndarray) and dev.dtype == G.get_dtype()
    dev64 = G.se3_exp_batch(cp.asarray(xi), dtype=np.float64)
    assert dev64.dtype == np.float64 and np.allclose(cp.asnumpy(dev64), host, atol=1e-5)
    assert isinstance(G.se3_act_each(dev, xi[:, 3:]), cp.ndarray)  # mixed -> device


@needs_cuda
def test_batch_backend_routing(rng, robot):
    from robolab.transforms import SE3Array, _backend, get_batch_backend, set_batch_backend
    old_min = _backend._batch_min
    xi = rng.normal(size=(1000, 6))
    q = rng.uniform(-0.5, 0.5, (500, robot.nq))
    cpu, fk_cpu, fk1_cpu = SE3Array.exp(xi), robot.fk_batch(q), robot.fk_batch(q[0])
    try:
        set_batch_backend("cuda", min_batch=100)
        assert get_batch_backend() == "cuda"
        gpu = SE3Array.exp(xi)
        assert np.allclose(gpu.matrices, cpu.matrices, atol=1e-5)
        assert np.allclose((gpu @ gpu.inverse()).matrices, np.eye(4), atol=1e-5)
        assert np.allclose(robot.fk_batch(q), fk_cpu, atol=1e-5)
        assert np.array_equal(robot.fk_batch(q[0]), fk1_cpu)  # below min_batch: CPU
    finally:
        set_batch_backend("cpu", min_batch=old_min)
    with pytest.raises(ValueError):
        set_batch_backend("opencl")
