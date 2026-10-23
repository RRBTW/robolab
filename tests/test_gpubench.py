"""robolab.gpubench: the verdict logic (no GPU needed) and a short real run."""
import pytest

from robolab import gpubench


@pytest.fixture(autouse=True)
def backend():  # the benchmark picks its kernels itself
    yield


def _row(op, n, cpu1, cpu_n, gpu):
    return {"op": op, "n": n, "cpu1": cpu1, "cpuN": cpu_n, "gpu32": gpu}


def test_thresholds_need_margin_and_must_hold_for_larger_n():
    rows = [
        _row("a", 1_000, 1.0, 0.5, 0.40),      # faster, but less than 1.5x: no
        _row("a", 10_000, 9.0, 3.0, 1.0),      # 3x: yes
        _row("a", 100_000, 90, 30, 5.0),
        _row("b", 1_000, 1.0, 1.0, 0.1),       # wins at 1e3 ...
        _row("b", 10_000, 2.0, 2.0, 3.0),      # ... loses at 1e4: the verdict starts later
        _row("b", 100_000, 20, 20, 1.0),
        _row("c", 1_000, 1.0, 1.0, 2.0),
        _row("c", 10_000, 1.0, 1.0, 2.0),
    ]
    t = gpubench.thresholds(rows, "gpu32")
    assert t == {"a": 10_000, "b": 100_000, "c": None}
    assert gpubench.thresholds(rows[:1], "gpu32", margin=1.0) == {"a": 1_000}   # vs best CPU


def test_short_real_run():
    from robolab.transforms import cuda_available
    pytest.importorskip("robolab.transforms._core_cy")
    if not cuda_available():
        pytest.skip("no CuPy or no CUDA device")
    from robolab.humanoid import make_humanoid
    rows = gpubench.run(make_humanoid(), [1_000], ops=("inverse", "mc"), log=lambda r: None)
    assert [r["op"] for r in rows] == ["inverse", "mc"]
    for r in rows:
        for k in ("cpu1", "cpuN", "gpu32", "gpu64", "h2d", "d2h", "e2e32"):
            assert 0 < r[k] < 1.0, (r["op"], k)
    assert 0 < gpubench.launch_overhead() < 1e-2
