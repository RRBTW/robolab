"""GPU benchmark behind ``robolab bench --device cuda`` (GPU_PLAN stage 4).

For every operation and batch size N it measures, per call:
  cpu1, cpuN  Cython kernel on 1 thread / on all threads
  gpu32, gpu64  CUDA kernel with the data already on the GPU (CuPy arrays),
                CUDA events around a run of back-to-back calls. For tiny N this
                is the launch overhead of a CuPy call, not the kernel itself.
  h2d, d2h    copies of the float32 inputs / outputs over PCIe
  e2e32       NumPy float64 in, NumPy float64 out (casts + copies + kernel)
The kernel columns are what carries over to a Jetson, where CPU and GPU share
one memory; the copy columns are specific to a PCIe card.
"""
from __future__ import annotations

import time
import timeit

import numpy as np

OPS = ("compose", "inverse", "exp", "log", "fk", "mc")
MAX_N = {"fk": 100_000}          # FK writes N x 26 x 4 x 4: 1.7 GB in float32 at N = 1e6


def cpu_time(fn, min_time=0.05, repeat=5):
    """Seconds per call: best of ``repeat`` runs, each long enough to time.
    Five runs: 16 OpenMP threads on 8 cores (WSL, desktop apps) jitter a lot."""
    fn()
    t0 = time.perf_counter()
    fn()
    number = max(1, int(min_time / max(time.perf_counter() - t0, 1e-7)))
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def gpu_time(fn, min_time=0.02, repeat=3):
    """Seconds per call on the device: CUDA events around back-to-back calls."""
    import cupy as cp
    fn()
    cp.cuda.Device().synchronize()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record(); fn(); stop.record(); stop.synchronize()
    number = max(1, int(min_time / max(cp.cuda.get_elapsed_time(start, stop) / 1e3, 1e-6)))
    best = np.inf
    for _ in range(repeat):
        start.record()
        for _ in range(number):
            fn()
        stop.record()
        stop.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(start, stop) / 1e3 / number)
    return best


def _cases(robot, n, rng):
    """op -> (call(kernel, *arrays), arrays). Arrays are float64 NumPy."""
    from .transforms import _core_cy as C
    tree = (robot._parent, robot._offset, robot._axis, robot._qidx, robot._jtype)
    t = C.se3_exp_batch(rng.normal(size=(n, 6)))
    q = rng.uniform(-0.5, 0.5, (n, robot.nq))
    q0 = np.zeros(robot.nq)
    q0[robot.joint_id("l_knee")] = q0[robot.joint_id("r_knee")] = 0.6
    lsb = 2 * np.pi / 2 ** 14
    mc_head = (*tree, q0, np.eye(4), np.arange(robot.nq))
    mc_tail = (robot.link_id("r_sole"), robot.link_id("l_sole"), robot.coms_local, robot.masses)
    tree_dev = {}

    def tree_for(k, qq):
        """The tree arrays where q lives: no per-call uploads when q is on the GPU."""
        if isinstance(qq, np.ndarray):
            return tree
        if qq.dtype not in tree_dev:
            import cupy as cp
            p, o, a, qi, jt = tree
            tree_dev[qq.dtype] = (cp.asarray(p), cp.asarray(o, dtype=qq.dtype),
                                  cp.asarray(a, dtype=qq.dtype), cp.asarray(qi), cp.asarray(jt))
        return tree_dev[qq.dtype]

    return {
        "compose": (lambda k, a, b: k.se3_mul_batch(a, b), (t, C.se3_exp_batch(rng.normal(size=(n, 6))))),
        "inverse": (lambda k, a: k.se3_inv_batch(a), (t,)),
        "exp": (lambda k, x: k.se3_exp_batch(x), (rng.normal(size=(n, 6)),)),
        "log": (lambda k, a: k.se3_log_batch(a), (t,)),
        # base is passed like a batch so that it lives where q lives
        "fk": (lambda k, qq, b: k.fk_tree_batch(*tree_for(k, qq), qq, b), (q, np.eye(4)[None])),
        "mc": (lambda k, d: k.mc_relative_pose(*mc_head, d, *mc_tail),
               (rng.uniform(-lsb / 2, lsb / 2, (n, robot.nq)),)),
    }


def run(robot, sizes, ops=OPS, log=print):
    """Rows of dicts: op, n and the columns described in the module docstring (s)."""
    import cupy as cp

    from .transforms import _core_cuda as G
    from .transforms import _core_cy as C
    n_cpu = C.get_num_threads()
    rows = []
    old_dtype = G.get_dtype()
    try:
        for op in ops:
            for n in sizes:
                if n > MAX_N.get(op, n):
                    continue
                call, arrays = _cases(robot, n, np.random.default_rng(n))[op]
                r = {"op": op, "n": n}
                C.set_num_threads(1)
                r["cpu1"] = cpu_time(lambda: call(C, *arrays))
                C.set_num_threads(n_cpu)
                r["cpuN"] = cpu_time(lambda: call(C, *arrays))
                for bits, dt in ((32, np.float32), (64, np.float64)):
                    G.set_dtype(dt)
                    dev = [cp.asarray(a, dtype=dt) for a in arrays]
                    r[f"gpu{bits}"] = gpu_time(lambda: call(G, *dev))
                    if bits == 32:
                        host32 = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
                        r["h2d"] = sum(cpu_time(lambda h=h: (cp.asarray(h), cp.cuda.Device().synchronize()))
                                       for h in host32)
                        out = call(G, *dev)
                        outs = out if isinstance(out, tuple) else (out,)
                        r["d2h"] = sum(cpu_time(lambda o=o: cp.asnumpy(o)) for o in outs)
                        r["e2e32"] = cpu_time(lambda: call(G, *arrays))
                    del dev
                cp.get_default_memory_pool().free_all_blocks()
                rows.append(r)
                log(r)
    finally:
        G.set_dtype(old_dtype)
        C.set_num_threads(n_cpu)
    return rows


MARGIN = 1.5


def thresholds(rows, column, margin=MARGIN):
    """op -> smallest measured N from which ``column`` is at least ``margin``
    times faster than the best CPU column, for this and every larger N (None: never).
    The margin keeps fixed costs out of the verdict: below N ~ 1e4 both sides
    are dominated by them (CuPy launch ~30 us, OpenMP wake-up 10-50 us), and
    their ratio jitters from run to run."""
    out = {}
    for op in dict.fromkeys(r["op"] for r in rows):
        rr = sorted((r for r in rows if r["op"] == op), key=lambda r: r["n"])
        win = [min(r["cpu1"], r["cpuN"]) >= margin * r[column] for r in rr]
        out[op] = next((r["n"] for i, r in enumerate(rr) if all(win[i:])), None)
    return out


def launch_overhead():
    """Seconds per CuPy kernel call with nothing to do (N = 1, data on the GPU)."""
    import cupy as cp

    from .transforms import _core_cuda as G
    t = cp.asarray(np.eye(4)[None], dtype=cp.float32)
    return gpu_time(lambda: G.se3_inv_batch(t, dtype=np.float32))
