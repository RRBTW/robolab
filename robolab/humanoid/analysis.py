"""Sensitivity of foot placement to joint encoder quantisation (Monte-Carlo).

An absolute encoder with b bits reports the angle to within +-1/2 LSB,
LSB = 2*pi / 2**b. With all joints quantised independently, where does the
swing foot end up relative to the stance foot, and how far does the CoM move?

Each sample is one thread of a fused kernel (``mc_relative_pose``): FK of the
whole tree, relative pose and CoM, and only the errors are written out.
On the CPU that is the current kernel (Cython or NumPy); with device="cuda"
the noise is drawn on the GPU and the results stay there, so reduce them with
``mc_summary`` and only numbers come back.
"""
from __future__ import annotations

import functools

import numpy as np

from ..transforms import _backend

CHUNK = {"cpu": 20_000, "cuda": 1 << 20}


def encoder_quantization_mc(tree, q, base, bits, n_samples, rng, ref_link="r_sole",
                            target_link="l_sole", joints=None, chunk=None, device="cpu",
                            dtype=None):
    """Returns dict: 'pos' (N,3) position error of target in the ref frame [m],
    'ang' (N,) orientation error [rad], 'com' (N,3) CoM error [m], 'lsb' [rad].

    device="cpu": NumPy arrays; the noise is drawn from ``rng`` chunk by chunk
    exactly as before the fused kernel, so earlier results are reproduced.
    device="cuda": CuPy arrays computed in ``dtype`` (default float32); the noise
    comes from a GPU generator seeded by ``rng``, so CPU and GPU agree
    statistically, not sample by sample."""
    q = np.asarray(q, dtype=float)
    joints = np.arange(tree.nq) if joints is None else np.asarray(joints)
    lsb = 2 * np.pi / 2 ** bits
    if device == "cpu":
        xp, mc, draw = np, _backend.kernel.mc_relative_pose, rng.uniform
    elif device == "cuda":
        import cupy as xp
        cuda = _backend._load_cuda()
        dt = cuda.get_dtype() if dtype is None else np.dtype(dtype)
        mc = functools.partial(cuda.mc_relative_pose, dtype=dt)
        draw = functools.partial(xp.random.RandomState(int(rng.integers(2 ** 63))).uniform, dtype=dt)
    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}")
    chunk = chunk or CHUNK[device]

    head =(tree._parent, tree._offset, tree._axis, tree._qidx, tree._jtype, q,
            tree._base3d(base, 1)[0], joints)
    tail = (tree.link_id(ref_link), tree.link_id(target_link), tree.coms_local, tree.masses)
    sizes = [min(chunk, n_samples - s) for s in range(0, n_samples, chunk)] or [0]
    parts = [mc(*head, draw(-lsb / 2, lsb / 2, size=(m, len(joints))), *tail) for m in sizes]
    pos, ang, com = (xp.concatenate([p[i] for p in parts]) for i in range(3))
    return {"pos": pos, "ang": ang, "com": com, "lsb": lsb}


def mc_summary(r, bins=None):
    """Statistics of an ``encoder_quantization_mc`` result, computed where the
    data lives (NumPy or CuPy); only numbers come back. Foot and CoM in mm,
    orientation in degrees. With ``bins`` (mm) also a histogram of the foot error."""
    if isinstance(r["pos"], np.ndarray):
        xp = np
    else:
        import cupy as xp
    foot = xp.linalg.norm(r["pos"], axis=1) * 1000.0
    com = xp.linalg.norm(r["com"], axis=1) * 1000.0
    out = {"n": int(foot.shape[0]), "foot_mean": float(foot.mean()),
           "foot95": float(xp.percentile(foot, 95)), "foot_max": float(foot.max()),
           "ang95": float(np.degrees(float(xp.percentile(r["ang"], 95)))),
           "com95": float(xp.percentile(com, 95))}
    if bins is not None:
        counts = xp.histogram(foot, bins=xp.asarray(bins))[0]
        out["hist"] = counts if xp is np else counts.get()
    return out
