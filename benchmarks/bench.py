"""Kernel and API timings: Cython vs NumPy reference vs spatialmath.

    python benchmarks/bench.py
"""
import timeit

import numpy as np

from robolab.transforms import SE3, SE3Array, available_backends, set_backend
from robolab.transforms import _core_py

try:
    from robolab.transforms import _core_cy
except ImportError:
    _core_cy = None
try:
    import spatialmath as sm
    import spatialmath.base as smb
except ImportError:
    sm = None


def us(stmt, n=2000):
    return min(timeit.repeat(stmt, number=n, repeat=5)) / n * 1e6


def main():
    rng = np.random.default_rng(0)
    xi = rng.normal(size=6)
    t = _core_py.se3_exp(xi)
    u = _core_py.se3_exp(rng.normal(size=6))
    big = _core_py.se3_exp_batch(rng.normal(size=(10_000, 6)))
    big2 = _core_py.se3_exp_batch(rng.normal(size=(10_000, 6)))
    cloud = rng.normal(size=(100_000, 3))
    twists = rng.normal(size=(10_000, 6))

    kernels = [("numpy", _core_py)] + ([("cython", _core_cy)] if _core_cy else [])
    rows = []
    for label, f, n in [
        ("se3_exp", lambda k: k.se3_exp(xi), 2000),
        ("se3_log", lambda k: k.se3_log(t), 2000),
        ("se3_mul", lambda k: k.se3_mul(t, u), 2000),
        ("se3_inv", lambda k: k.se3_inv(t), 2000),
        ("se3_check", lambda k: k.se3_check(t, 1e-9), 2000),
        ("exp  x10k", lambda k: k.se3_exp_batch(twists), 3),
        ("log  x10k", lambda k: k.se3_log_batch(big), 3),
        ("mul  x10k", lambda k: k.se3_mul_batch(big, big2), 20),
        ("act 100k pts", lambda k: k.se3_act(t, cloud), 20),
    ]:
        rows.append((label, [us(lambda: f(k), n) for _, k in kernels]))

    print(f"{'kernel op':<14}" + "".join(f"{name:>12}" for name, _ in kernels)
          + ("   speed-up" if _core_cy else ""))
    for label, times in rows:
        line = f"{label:<14}" + "".join(f"{x:>10.2f}us" for x in times)
        if _core_cy:
            line += f"   x{times[0] / times[1]:.1f}"
        print(line)

    print("\nfull API (validation + objects), one composition a @ b:")
    for be in available_backends():
        set_backend(be)
        a, b = SE3.exp(xi), SE3.exp(rng.normal(size=6))
        print(f"  robolab[{be}] SE3 @ SE3        {us(lambda: a @ b):8.2f} us")
        print(f"  robolab[{be}] SE3.exp          {us(lambda: SE3.exp(xi)):8.2f} us")
        arr = SE3Array.exp(rng.normal(size=(10_000, 6)))
        print(f"  robolab[{be}] SE3Array@SE3 10k {us(lambda: arr @ a, 20):8.0f} us")
    if sm is not None:
        sa, sb = sm.SE3(t, check=False), sm.SE3(u, check=False)
        print(f"  spatialmath   SE3 * SE3        {us(lambda: sa * sb):8.2f} us")
        print(f"  spatialmath   trexp            {us(lambda: smb.trexp(xi)):8.2f} us")
        sarr = sm.SE3([sm.SE3(m, check=False) for m in big])
        print(f"  spatialmath   SE3[10k] * SE3   {us(lambda: sarr * sa, 3):8.0f} us")


if __name__ == "__main__":
    main()
