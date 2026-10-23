"""Encoder Monte-Carlo. The fused CPU path reproduces the earlier
SE3Array-based computation sample by sample; the CUDA path agrees with the CPU
statistically (GPU_PLAN stage 2: mean and 95th percentile within 2 %)."""
import numpy as np
import pytest

from robolab.humanoid import make_humanoid
from robolab.humanoid.analysis import encoder_quantization_mc, mc_summary
from robolab.transforms import SE3, SE3Array, cuda_available


@pytest.fixture(scope="module")
def robot():
    return make_humanoid()


@pytest.fixture(scope="module")
def pose(robot):
    q = np.zeros(robot.nq)
    q[robot.joint_id("l_knee")] = q[robot.joint_id("r_knee")] = 0.6
    q[robot.joint_id("l_hip_pitch")] = -0.4
    return q, SE3.Trans(0, 0, 0.68)


def _mc_before_fusion(tree, q, base, bits, n_samples, rng, chunk, ref_link="r_sole",
                      target_link="l_sole"):
    """encoder_quantization_mc as it was before the fused kernel (v0.4)."""
    joints = np.arange(tree.nq)
    lsb = 2 * np.pi / 2 ** bits
    nominal = tree.fk(q, base)
    rel_nom = nominal[ref_link].inverse() @ nominal[target_link]
    com_nom = tree.com(nominal)
    ir, it = tree.link_id(ref_link), tree.link_id(target_link)
    pos, ang, com = [], [], []
    for start in range(0, n_samples, chunk):
        m = min(chunk, n_samples - start)
        qs = np.tile(q, (m, 1))
        qs[:, joints] += rng.uniform(-lsb / 2, lsb / 2, size=(m, len(joints)))
        poses = tree.fk_batch(qs, base)
        t_ref = SE3Array(poses[:, ir], parent=tree.base_frame, child=ref_link)
        t_tgt = SE3Array(poses[:, it], parent=tree.base_frame, child=target_link)
        rel = t_ref.inverse() @ t_tgt
        pos.append(rel.translations - rel_nom.translation)
        ang.append(np.linalg.norm((rel_nom.inverse() @ rel).log()[:, :3], axis=1))
        com.append(tree.com(poses) - com_nom)
    return np.concatenate(pos), np.concatenate(ang), np.concatenate(com)


@pytest.mark.parametrize("bits", [10, 18])
def test_cpu_reproduces_results_before_fusion(robot, pose, bits):
    q, base = pose
    old = _mc_before_fusion(robot, q, base, bits, 3000, np.random.default_rng(5), chunk=1000)
    new = encoder_quantization_mc(robot, q, base, bits, 3000, np.random.default_rng(5), chunk=1000)
    for key, o in zip(("pos", "ang", "com"), old):
        assert isinstance(new[key], np.ndarray)
        assert np.abs(new[key] - o).max() <= 1e-8 * np.abs(o).max(), key


def test_mc_summary(robot, pose):
    r = encoder_quantization_mc(robot, *pose, 12, 5000, np.random.default_rng(0))
    s = mc_summary(r, bins=np.logspace(-3.5, 1.2, 120))
    assert s["n"] == 5000 and 4990 <= s["hist"].sum() <= 5000
    assert 0 < s["foot_mean"] < s["foot95"] < s["foot_max"]
    e = np.linalg.norm(r["pos"], axis=1) * 1000
    assert np.isclose(s["foot95"], np.percentile(e, 95))
    with pytest.raises(ValueError):
        encoder_quantization_mc(robot, *pose, 12, 10, np.random.default_rng(0), device="tpu")


@pytest.mark.skipif(not cuda_available(), reason="no CuPy or no CUDA device")
@pytest.mark.parametrize("bits", [12, 18])
def test_cuda_matches_cpu_statistically(robot, pose, bits, backend):
    if backend != "cython":
        pytest.skip("the CPU reference uses the compiled kernel (200k samples)")
    import cupy as cp
    q, base = pose
    n = 200_000
    cpu = mc_summary(encoder_quantization_mc(robot, q, base, bits, n, np.random.default_rng(1)))
    r = encoder_quantization_mc(robot, q, base, bits, n, np.random.default_rng(2), device="cuda")
    assert isinstance(r["pos"], cp.ndarray) and r["pos"].dtype == np.float32
    gpu = mc_summary(r)
    for key in ("foot_mean", "foot95", "ang95", "com95"):
        assert abs(gpu[key] / cpu[key] - 1) < 0.02, (key, gpu[key], cpu[key])
