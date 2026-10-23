"""Headless runner for machines without a desktop (Raspberry Pi, servers, SSH).

    robolab info                      system, kernel, OpenMP, threads
    robolab walk     [--gif]          walking simulation, balance check, joint CSV
    robolab sweep    [--quick] [--device cuda]
                                      where the LIPM plan stops being balanced
    robolab encoders [--samples N] [--device cuda]
                                      Monte-Carlo: encoder resolution vs foot error
    robolab bench    [--quick] [--device cuda]
                                      FK and SE(3) kernels: loop vs batch vs threads (vs GPU)
    robolab all      [--quick] [--gif]

Tables go to the terminal and to <out>/report.md, figures to <out>/*.png,
trajectories to <out>/joint_trajectory.csv and <out>/walk.npz.
Matplotlib runs with the Agg backend: no display is needed.

Language: Russian by default, English on the raw Linux console (TERM=linux),
where Cyrillic often has no font; override with --lang ru|en.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #
MSG = {
    "ru": {
        "sys": "Система", "board": "плата", "cpu": "ядер CPU", "py": "Python", "kernel": "ядро",
        "omp": "OpenMP", "threads": "потоков", "temp": "температура CPU", "freq": "частота CPU",
        "mem": "память", "gov": "governor",
        "walk": "Симуляция шага", "params": "параметры", "sim_time": "время симуляции",
        "stage": "этап", "seconds": "с", "ik": "IK", "ik_fail": "неудач IK",
        "com_conv": "ошибка ЦМ по проходам коррекции таза, мм", "balance": "Баланс",
        "margin_min": "мин. запас ZMP", "inside": "ZMP внутри опоры", "of_time": "% времени",
        "zmp_dev": "макс. отклонение многозвенного ZMP от опорного (x, y)",
        "naive": "ЦМ = таз (без коррекции)", "corrected": "с коррекцией таза",
        "com_err": "макс. ошибка ЦМ, мм", "min_margin_mm": "мин. запас ZMP, мм",
        "joints": "Суставы (левая нога)", "joint": "сустав", "range": "диапазон, °",
        "vel": "макс |ω|, °/с", "rpm": "макс |ω|, об/мин",
        "saved": "сохранено", "sweep": "Перебор шага и периода: мин. запас многозвенного ZMP, мм",
        "sweep_note": "отрицательное значение = ZMP выходит за опору (робот опрокинется)",
        "step": "шаг", "period": "период",
        "enc": "Монте-Карло: разрядность энкодеров (поза в верхней точке переноса левой ноги)",
        "bits": "бит", "lsb": "LSB, °", "foot_mean": "стопа ср., мм", "foot95": "стопа 95%, мм",
        "foot_max": "стопа макс, мм", "ang95": "ориент. 95%, °", "com95": "ЦМ 95%, мм",
        "samples": "выборок", "bench": "Скорость", "fk_title": "FK всего тела",
        "configs": "конфигураций", "method": "способ", "time_ms": "время, мс",
        "speedup": "ускорение", "per_s": "конф./с", "loop": "цикл SE3-объектов",
        "np_kernel": "батч, NumPy-ядро", "cy_n": "батч, Cython, {n} пот.",
        "ops_title": "Батчевые ядра SE(3), {n} элементов", "op": "операция",
        "compose": "композиция", "inverse": "обратное", "done": "Готово. Результаты в",
        "view": "Посмотреть с ноутбука: python3 -m http.server 8000 --directory {out}  ->  http://<IP-платы>:8000",
        "gif": "анимация", "gif_hint": "анимацию можно включить флагом --gif",
        "rendering": "рендер", "frames": "кадров",
        "plot_t": "t, с", "plot_m": "м", "zmp_ref": "ZMP опорный", "zmp_mb": "ZMP многозвенный",
        "zmp_cart": "ZMP модели", "com": "ЦМ", "top": "вид сверху", "margin_mm": "запас ZMP, мм",
        "angles": "углы левой ноги, °", "speeds": "скорости, °/с", "foot_err": "ошибка переносимой стопы, мм",
        "axis": "ось", "gpu": "GPU", "warmup": "без компиляции и прогрева",
        "gpu_bench": "GPU: батчевые ядра, время на вызов, мс",
        "gpu_note": "GPU f32/f64: данные уже на GPU (так работает Jetson с общей памятью); "
                    "H2D/D2H: копии float32 по PCIe; GPU+копии: NumPy float64 на входе и выходе",
        "launch": "накладные расходы вызова GPU (N = 1, данные на GPU)",
        "gpu_wins": "С какого N GPU выгоден (в ≥1.5 раза быстрее лучшего CPU)", "on_gpu": "данные на GPU",
        "with_copies": "с копиями", "best": "ускорение при макс. N (данные на GPU)",
        "never": "не быстрее", "op_fk": "FK, 26 звеньев", "op_mc": "Монте-Карло (fused)",
        "e2e": "GPU+копии",
    },
    "en": {
        "sys": "System", "board": "board", "cpu": "CPU cores", "py": "Python", "kernel": "kernel",
        "omp": "OpenMP", "threads": "threads", "temp": "CPU temperature", "freq": "CPU frequency",
        "mem": "memory", "gov": "governor",
        "walk": "Walking simulation", "params": "parameters", "sim_time": "simulation time",
        "stage": "stage", "seconds": "s", "ik": "IK", "ik_fail": "IK failures",
        "com_conv": "CoM error per pelvis-correction pass, mm", "balance": "Balance",
        "margin_min": "min ZMP margin", "inside": "ZMP inside support", "of_time": "% of time",
        "zmp_dev": "max deviation of multi-body ZMP from reference (x, y)",
        "naive": "CoM = pelvis (no correction)", "corrected": "pelvis corrected",
        "com_err": "max CoM error, mm", "min_margin_mm": "min ZMP margin, mm",
        "joints": "Joints (left leg)", "joint": "joint", "range": "range, deg",
        "vel": "max |w|, deg/s", "rpm": "max |w|, rpm",
        "saved": "saved", "sweep": "Step length x period sweep: min multi-body ZMP margin, mm",
        "sweep_note": "negative = ZMP leaves the support polygon (the robot tips over)",
        "step": "step", "period": "period",
        "enc": "Monte-Carlo: encoder resolution (pose at the apex of the left swing)",
        "bits": "bits", "lsb": "LSB, deg", "foot_mean": "foot mean, mm", "foot95": "foot 95%, mm",
        "foot_max": "foot max, mm", "ang95": "orient 95%, deg", "com95": "CoM 95%, mm",
        "samples": "samples", "bench": "Speed", "fk_title": "Whole-body FK",
        "configs": "configurations", "method": "method", "time_ms": "time, ms",
        "speedup": "speed-up", "per_s": "configs/s", "loop": "loop of SE3 objects",
        "np_kernel": "batch, NumPy kernel", "cy_n": "batch, Cython, {n} thr",
        "ops_title": "SE(3) batch kernels, {n} elements", "op": "operation",
        "compose": "compose", "inverse": "inverse", "done": "Done. Results in",
        "view": "View from a laptop: python3 -m http.server 8000 --directory {out}  ->  http://<board-IP>:8000",
        "gif": "animation", "gif_hint": "add --gif to render the animation",
        "rendering": "rendering", "frames": "frames",
        "plot_t": "t, s", "plot_m": "m", "zmp_ref": "ZMP reference", "zmp_mb": "ZMP multi-body",
        "zmp_cart": "ZMP cart-table", "com": "CoM", "top": "top view", "margin_mm": "ZMP margin, mm",
        "angles": "left-leg angles, deg", "speeds": "velocities, deg/s", "foot_err": "swing-foot error, mm",
        "axis": "axis", "gpu": "GPU", "warmup": "excluding compile and warm-up",
        "gpu_bench": "GPU: batch kernels, time per call, ms",
        "gpu_note": "GPU f32/f64: data already on the GPU (how a Jetson with shared memory works); "
                    "H2D/D2H: float32 copies over PCIe; GPU+copies: NumPy float64 in and out",
        "launch": "GPU call overhead (N = 1, data on the GPU)",
        "gpu_wins": "From which N the GPU pays off (>= 1.5x faster than the best CPU)", "on_gpu": "data on GPU",
        "with_copies": "with copies", "best": "speed-up at max N (data on GPU)",
        "never": "never", "op_fk": "FK, 26 links", "op_mc": "Monte-Carlo (fused)",
        "e2e": "GPU+copies",
    },
}
L = MSG["ru"]


def _pick_lang(arg):
    if arg:
        return arg
    enc = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    if enc != "utf8" or os.environ.get("TERM") == "linux":
        return "en"
    return "ru"


# --------------------------------------------------------------------------- #
# report: console + markdown
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self, out: Path):
        self.out = out
        self.md = []

    def h(self, title):
        print(f"\n== {title} " + "=" * max(0, 70 - len(title)), flush=True)
        self.md.append(f"\n## {title}\n")

    def p(self, text):
        print(text, flush=True)
        self.md.append(text + "  ")

    def kv(self, pairs):
        w = max(len(k) for k, _ in pairs)
        for k, v in pairs:
            print(f"  {k:<{w}}  {v}", flush=True)
        self.md.append("\n| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in pairs) + "\n")

    def table(self, headers, rows):
        rows = [[str(c) for c in r] for r in rows]
        w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        line = "  ".join(f"{h:>{w[i]}}" if i else f"{h:<{w[i]}}" for i, h in enumerate(headers))
        print("  " + line + "\n  " + "-" * len(line))
        for r in rows:
            print("  " + "  ".join(f"{c:>{w[i]}}" if i else f"{c:<{w[i]}}" for i, c in enumerate(r)))
        sys.stdout.flush()
        self.md.append("\n| " + " | ".join(map(str, headers)) + " |\n|" + "---|" * len(headers) + "\n"
                       + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")

    def file(self, path):
        print(f"  -> {L['saved']}: {path}", flush=True)
        self.md.append(f"![{Path(path).stem}]({Path(path).name})\n" if str(path).endswith((".png", ".gif"))
                       else f"`{Path(path).name}`\n")

    def save(self):
        (self.out / "report.md").write_text("# robolab report\n" + "\n".join(self.md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def _read(path, conv=str):
    try:
        return conv(Path(path).read_text().strip().strip("\x00"))
    except (OSError, ValueError):
        return None


def _per_call(fn, number=3, repeat=3):
    import timeit
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


SIDE_COLOR = {"l": "C2", "r": "C1"}


def draw_robot(ax, robot, poses, alpha=1.0, lw=2.2):
    from .humanoid import SKELETON
    d = robot.dims
    p = poses[:, :3, 3]
    for a, b in SKELETON:
        sides = {n[0] for n in (a, b) if n[:2] in ("l_", "r_")}
        color = SIDE_COLOR[sides.pop()] if len(sides) == 1 else "0.2"
        i, j = robot.link_id(a), robot.link_id(b)
        ax.plot(*zip(p[i], p[j]), color=color, alpha=alpha, lw=lw)
    h = poses[robot.link_id("head")]
    ax.scatter(*(h[:3, :3] @ [0.02, 0, 0.12] + h[:3, 3]), s=120, color="0.2", alpha=alpha)
    for side in "lr":
        s = poses[robot.link_id(f"{side}_sole")]
        c = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [1, 1, 0]]) * \
            [d.foot_length / 2, d.foot_width / 2, 0]
        ax.plot(*(c @ s[:3, :3].T + s[:3, 3]).T, color=SIDE_COLOR[side], alpha=alpha, lw=1.2)


def _equal_3d(ax, center, r):
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        set_lim(c - r, c + r)
    ax.set_box_aspect((1, 1, 1))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_info(args, rep):
    import robolab.transforms as rt
    mem = None
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal"):
            mem = f"{int(line.split()[1]) / 1024 / 1024:.1f} GB"
    temp = _read("/sys/class/thermal/thermal_zone0/temp", int)
    freq = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", int)
    rep.h(L["sys"])
    rep.kv([
        (L["board"], _read("/proc/device-tree/model") or platform.machine()),
        (L["cpu"], os.cpu_count()),
        (L["mem"], mem or "?"),
        (L["temp"], f"{temp / 1000:.1f} °C" if temp else "?"),
        (L["freq"], f"{freq / 1000:.0f} MHz" if freq else "?"),
        (L["gov"], _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "?"),
        (L["py"], f"{platform.python_version()} ({platform.machine()}), numpy {np.__version__}"),
        (L["kernel"], rt.get_backend()),
        (L["omp"], rt.openmp_enabled()),
        (L["threads"], rt.get_num_threads()),
    ] + ([(L["gpu"], _gpu_label(args))] if args.device == "cuda" else []))


def _cpu_label():
    import robolab.transforms as rt
    return f"{rt.get_backend()}, {rt.get_num_threads()} {L['threads']}"


def _gpu_label(args):
    import cupy
    from robolab.transforms import _backend
    cuda = _backend._load_cuda()
    return f"{cuda.device_name()}, {args.dtype or cuda.get_dtype().name}, CuPy {cupy.__version__}"


def cmd_walk(args, rep):
    from .humanoid import GaitParams, LEG_JOINTS, make_humanoid, simulate_walk
    from .humanoid.balance import foot_corners
    plt = _plt()
    robot = make_humanoid()
    params = GaitParams(step_length=args.step_length, step_time=args.step_time, n_steps=args.steps)
    rep.h(L["walk"])
    rep.p(f"{L['params']}: step_length={params.step_length} m, step_time={params.step_time} s, "
          f"n_steps={params.n_steps}, dt={params.dt} s; robot {robot.nq} DoF, {robot.total_mass:.1f} kg")
    t0 = time.perf_counter()
    walk = simulate_walk(robot, params, com_passes=4)
    total = time.perf_counter() - t0
    rep.kv([
        (L["sim_time"], f"{total:.2f} {L['seconds']} ({len(walk.t)} x {params.dt * 1000:.0f} ms)"),
        *[(f"  {k}", f"{v:.3f} {L['seconds']}") for k, v in walk.timings.items()],
        (L["ik"], f"{walk.ik_iterations} it, {L['ik_fail']}: {walk.ik_failures}"),
        (L["com_conv"], " -> ".join(f"{e * 1000:.2f}" for e in walk.com_iterations)),
    ])

    rep.h(L["balance"])
    naive = simulate_walk(robot, params, com_passes=1)
    rep.kv([
        (L["margin_min"], f"{walk.margin.min() * 1000:.1f} mm"),
        (L["inside"], f"{100 * (walk.margin > 0).mean():.1f} {L['of_time']}"),
        (L["zmp_dev"], f"{np.abs(walk.zmp - walk.zmp_ref).max(0).round(3)} m"),
    ])
    rep.table(["", L["com_err"], L["min_margin_mm"]], [
        [L["naive"], f"{naive.com_iterations[-1] * 1000:.1f}", f"{naive.margin.min() * 1000:.1f}"],
        [L["corrected"], f"{walk.com_iterations[-1] * 1000:.1f}", f"{walk.margin.min() * 1000:.1f}"],
    ])

    rep.h(L["joints"])
    qd = np.gradient(walk.q, params.dt, axis=0)
    ids = [robot.joint_id(j) for j in LEG_JOINTS["l"]]
    rep.table([L["joint"], L["range"], L["vel"], L["rpm"]], [
        [robot.joint_names[j], f"{np.degrees(walk.q[:, j].min()):.0f} .. {np.degrees(walk.q[:, j].max()):.0f}",
         f"{np.degrees(np.abs(qd[:, j]).max()):.0f}", f"{np.abs(qd[:, j]).max() * 60 / (2 * np.pi):.1f}"]
        for j in ids])

    out = rep.out
    with open(out / "joint_trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [f"{n}_rad" for n in robot.joint_names])
        for k in range(len(walk.t)):
            w.writerow([f"{walk.t[k]:.3f}"] + [f"{v:.6f}" for v in walk.q[k]])
    rep.file(out / "joint_trajectory.csv")
    np.savez_compressed(out / "walk.npz", t=walk.t, q=walk.q, base=walk.base, com=walk.com,
                        com_ref=walk.com_ref, zmp=walk.zmp, zmp_ref=walk.zmp_ref, margin=walk.margin,
                        phase=walk.phase, joint_names=np.array(robot.joint_names))
    rep.file(out / "walk.npz")

    # plan: ZMP and CoM over time
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    for k, lab in enumerate("xy"):
        ax[k].plot(walk.t, walk.zmp_ref[:, k], color="0.6", lw=3, label=L["zmp_ref"])
        ax[k].plot(walk.t, walk.zmp_cart[:, k], "C3--", label=L["zmp_cart"])
        ax[k].plot(walk.t, walk.com[:, k], "C0", label=L["com"])
        ax[k].set_title(f"{L['axis']} {lab}"); ax[k].set_xlabel(L["plot_t"]); ax[k].set_ylabel(L["plot_m"])
    ax[0].legend(); fig.tight_layout(); fig.savefig(out / "plan.png"); plt.close(fig)
    rep.file(out / "plan.png")

    # balance: top view + margin
    d = robot.dims
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1.4, 1]})
    for side in "lr":
        corners = foot_corners(walk.poses[:, robot.link_id(f"{side}_sole")], d.foot_length, d.foot_width)
        for k in np.flatnonzero(walk.feet_ref[side][:, 2] < 1e-9)[::25]:
            ax[0].fill(*corners[k].T, color=SIDE_COLOR[side], alpha=0.06, lw=0)
    ax[0].plot(*walk.zmp_ref.T, color="0.5", lw=2.5, label=L["zmp_ref"])
    ax[0].plot(*walk.zmp.T, "C3", lw=1.2, label=L["zmp_mb"])
    ax[0].plot(*walk.com[:, :2].T, "C0", lw=1.5, label=L["com"])
    ax[0].set_aspect("equal"); ax[0].set_title(L["top"]); ax[0].legend(loc="upper left")
    ax[1].plot(walk.t, walk.margin * 1000, "k"); ax[1].axhline(0, color="r", lw=1)
    for ph, col in ((1, "C2"), (2, "C1")):
        ax[1].fill_between(walk.t, -10, 120, where=walk.phase == ph, color=col, alpha=0.12, lw=0)
    ax[1].set_ylim(-10, 120); ax[1].set_ylabel(L["margin_mm"]); ax[1].set_xlabel(L["plot_t"])
    fig.tight_layout(); fig.savefig(out / "balance.png"); plt.close(fig)
    rep.file(out / "balance.png")

    # joints
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for j in ids:
        ax[0].plot(walk.t, np.degrees(walk.q[:, j]), label=robot.joint_names[j][2:])
        ax[1].plot(walk.t, np.degrees(qd[:, j]))
    ax[0].set_title(L["angles"]); ax[1].set_title(L["speeds"])
    for a in ax:
        a.set_xlabel(L["plot_t"])
    ax[0].legend(ncol=3, fontsize=8, loc="upper right"); fig.tight_layout()
    fig.savefig(out / "joints.png"); plt.close(fig)
    rep.file(out / "joints.png")

    # stroboscopic picture of the whole walk
    fig = plt.figure(figsize=(13, 5.5)); ax = fig.add_subplot(projection="3d")
    ks = np.linspace(0, len(walk.t) - 1, 12).astype(int)
    for i, k in enumerate(ks):
        draw_robot(ax, robot, walk.poses[k], alpha=0.25 + 0.75 * i / (len(ks) - 1), lw=1.6)
    ax.plot(*np.c_[walk.zmp, np.zeros(len(walk.t))].T, "C3", lw=1)
    x0, x1 = walk.com[:, 0].min() - 0.3, walk.com[:, 0].max() + 0.3
    ax.set_xlim(x0, x1); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0, 1.5)
    ax.set_box_aspect((x1 - x0, 0.6, 1.5), zoom=1.1); ax.view_init(10, -78)
    ax.set_yticks([]); ax.set_xlabel("x, m")
    fig.subplots_adjust(0, 0.02, 1, 1); fig.savefig(out / "robot_strobe.png"); plt.close(fig)
    rep.file(out / "robot_strobe.png")

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        frames = range(0, len(walk.t), 4)
        print(f"  {L['rendering']} {L['gif']}: {len(frames)} {L['frames']}...", flush=True)
        fig = plt.figure(figsize=(6, 5.5)); ax = fig.add_subplot(projection="3d")

        def update(k):
            ax.cla()
            draw_robot(ax, robot, walk.poses[k])
            ax.plot(*np.c_[walk.zmp[:k + 1], np.zeros(k + 1)].T, "C3", lw=1)
            poly = np.vstack([walk.polygons[k], walk.polygons[k][:1]])
            ax.plot(poly[:, 0], poly[:, 1], 0, color="C2", lw=1.5)
            _equal_3d(ax, (walk.com[k, 0], 0, 0.72), 0.72); ax.view_init(10, -55)
            ax.set_title(f"t = {walk.t[k]:.2f} s   ZMP margin {walk.margin[k] * 1000:.0f} mm", fontsize=10)

        FuncAnimation(fig, update, frames=frames).save(out / "walk.gif", writer=PillowWriter(fps=25), dpi=70)
        plt.close(fig)
        rep.file(out / "walk.gif")
    else:
        rep.p(f"  ({L['gif_hint']})")


def cmd_sweep(args, rep):
    from .humanoid import GaitParams, make_humanoid
    from .humanoid.walk import simulate_walks
    plt = _plt()
    robot = make_humanoid()
    times = [0.4, 0.55, 0.7] if args.quick else [0.4, 0.5, 0.6, 0.7, 0.8]
    lengths = [0.15, 0.25] if args.quick else [0.15, 0.20, 0.25, 0.30]
    rep.h(L["sweep"])
    gaits = [GaitParams(step_length=sl, step_time=st, n_steps=4) for sl in lengths for st in times]
    if args.device == "cuda":
        simulate_walks(robot, gaits[:1], com_passes=1, device="cuda")  # compile the IK kernel
    t0 = time.perf_counter()
    walks = simulate_walks(robot, gaits, com_passes=3, device=args.device)
    elapsed = time.perf_counter() - t0
    grid = np.array([w.margin.min() * 1000 for w in walks]).reshape(len(lengths), len(times))
    for g, v in zip(gaits, grid.ravel()):
        print(f"  {L['step']} {g.step_length:.2f} m, {L['period']} {g.step_time:.2f} s: {v:7.1f} mm")
    where = "CUDA float64" if args.device == "cuda" else f"CPU {_cpu_label()}"
    rep.p(f"{grid.size} x simulate_walk ({L['ik']}: {where}): {elapsed:.1f} {L['seconds']}, "
          f"{L['ik']} {walks[0].timings['ik']:.2f} {L['seconds']}; {L['sweep_note']}")
    rep.table([f"{L['step']} \\ {L['period']}"] + [f"{t:.2f} s" for t in times],
              [[f"{sl:.2f} m"] + [f"{v:.0f}" for v in grid[i]] for i, sl in enumerate(lengths)])
    fig, ax = plt.subplots(figsize=(1.4 * len(times) + 2, 0.8 * len(lengths) + 1.6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-60, vmax=40, aspect="auto")
    ax.set_xticks(range(len(times)), [f"{t:.2f} s" for t in times])
    ax.set_yticks(range(len(lengths)), [f"{s:.2f} m" for s in lengths])
    for i in range(len(lengths)):
        for j in range(len(times)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center")
    ax.set_xlabel(L["period"]); ax.set_ylabel(L["step"]); ax.grid(False)
    fig.colorbar(im, label=L["min_margin_mm"]); fig.tight_layout()
    fig.savefig(rep.out / "sweep.png"); plt.close(fig)
    rep.file(rep.out / "sweep.png")


def cmd_encoders(args, rep):
    from .humanoid import GaitParams, make_humanoid, simulate_walk
    from .humanoid.analysis import encoder_quantization_mc, mc_summary
    plt = _plt()
    robot = make_humanoid()
    walk = simulate_walk(robot, GaitParams(n_steps=2), com_passes=3)
    k = int(np.argmax(walk.feet_ref["l"][:, 2]))
    rng = np.random.default_rng(0)
    bins = np.logspace(-3.5, 1.2, 120)
    rep.h(L["enc"])

    def mc(bits, n, g):
        return encoder_quantization_mc(robot, walk.q[k], walk.base[k], bits, n, g,
                                       device=args.device, dtype=args.dtype)

    mc_summary(mc(12, 1000, np.random.default_rng(1)), bins)  # compile kernels, warm caches
    rows, hists = [], {}
    t0 = time.perf_counter()
    for bits in (10, 12, 14, 16, 18):
        r = mc(bits, args.samples, rng)
        s = mc_summary(r, bins)
        hists[bits] = s["hist"]
        rows.append([bits, f"{np.degrees(r['lsb']):.4f}", f"{s['foot_mean']:.3g}", f"{s['foot95']:.3g}",
                     f"{s['foot_max']:.3g}", f"{s['ang95']:.3g}", f"{s['com95']:.3g}"])
    where = f"CUDA {args.dtype or 'float32'}" if args.device == "cuda" else f"CPU {_cpu_label()}"
    rep.p(f"5 x {args.samples} {L['samples']}, {where}: {time.perf_counter() - t0:.2f} {L['seconds']}"
          f" ({L['warmup']})")
    rep.table([L["bits"], L["lsb"], L["foot_mean"], L["foot95"], L["foot_max"], L["ang95"], L["com95"]], rows)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for bits, h in hists.items():
        ax.stairs(h, bins, lw=1.5, label=f"{bits} {L['bits']}")
    ax.set_xscale("log"); ax.set_xlabel(L["foot_err"]); ax.legend(); fig.tight_layout()
    fig.savefig(rep.out / "encoders.png"); plt.close(fig)
    rep.file(rep.out / "encoders.png")


def cmd_bench(args, rep):
    import robolab.transforms as rt
    from robolab.transforms import SE3, SO3
    from .humanoid import make_humanoid
    plt = _plt()
    robot = make_humanoid()
    n_cpu = os.cpu_count() or 1
    threads = sorted({1, 2, n_cpu} & set(range(1, n_cpu + 1))) if rt.openmp_enabled() else [1]
    n = 5_000 if args.quick else 20_000
    q = np.random.default_rng(1).uniform(-0.5, 0.5, (n, robot.nq))
    base = SE3.Trans(0, 0, 0.68)

    def fk_naive(qq):
        out = []
        for lk in robot.links:
            t = (base if lk.parent is None else out[robot.link_id(lk.parent)]) @ lk.offset
            if lk.joint is not None:
                t = t @ SE3.from_rotation(SO3.exp(qq[robot.joint_id(lk.joint)] * np.asarray(lk.axis)))
            out.append(t)
        return out

    rep.h(f"{L['bench']}: {L['fk_title']}, {n} {L['configs']} x {robot.nl} links")
    res = {}
    m = 100 if args.quick else 200
    res[L["loop"]] = _per_call(lambda: [fk_naive(x) for x in q[:m]], 1, 2) / m * n
    if "python" in rt.available_backends():
        rt.set_backend("python")
        res[L["np_kernel"]] = _per_call(lambda: robot.fk_batch(q, base), 1, 2)
    if "cython" in rt.available_backends():
        rt.set_backend("cython")
        for th in threads:
            rt.set_num_threads(th)
            res[L["cy_n"].format(n=th)] = _per_call(lambda: robot.fk_batch(q, base))
        rt.set_num_threads(n_cpu)
    names, vals = list(res), np.array(list(res.values()))
    rep.table([L["method"], L["time_ms"], L["speedup"], L["per_s"]],
              [[k, f"{v * 1000:.1f}", f"{vals[0] / v:.0f}x", f"{n / v:,.0f}"] for k, v in res.items()])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(res) + 1.2))
    ax.barh(names[::-1], vals[::-1] * 1000, color="C0")
    ax.set_xscale("log"); ax.set_xlabel(f"{L['time_ms']} ({n} {L['configs']})"); fig.tight_layout()
    fig.savefig(rep.out / "bench_fk.png"); plt.close(fig)
    rep.file(rep.out / "bench_fk.png")

    if "cython" in rt.available_backends():
        from robolab.transforms import _core_cy as K
        ne = 50_000 if args.quick else 200_000
        rng = np.random.default_rng(2)
        tt = np.ascontiguousarray(K.se3_exp_batch(rng.normal(size=(ne, 6))))
        xx = np.ascontiguousarray(rng.normal(size=(ne, 6)))
        rep.h(L["ops_title"].format(n=ne))
        rows = []
        for name, fn in [(L["compose"], lambda: K.se3_mul_batch(tt, tt)), (L["inverse"], lambda: K.se3_inv_batch(tt)),
                         ("exp", lambda: K.se3_exp_batch(xx)), ("log", lambda: K.se3_log_batch(tt))]:
            times = []
            for th in threads:
                rt.set_num_threads(th)
                times.append(_per_call(fn))
            rows.append([name] + [f"{t * 1000:.2f}" for t in times] + [f"{times[0] / times[-1]:.2f}x"])
        rt.set_num_threads(n_cpu)
        rep.table([L["op"]] + [f"{th} {L['threads']}, ms" for th in threads] + [L["speedup"]], rows)

    if args.device == "cuda":
        _bench_cuda(args, rep, robot, plt)


def _bench_cuda(args, rep, robot, plt):
    from . import gpubench
    sizes = [1_000, 10_000, 100_000] if args.quick else [1_000, 10_000, 100_000, 1_000_000]
    names = {"compose": L["compose"], "inverse": L["inverse"], "exp": "exp", "log": "log",
             "fk": L["op_fk"], "mc": L["op_mc"]}
    rep.h(f"{L['gpu_bench']} ({_gpu_label(args).split(',')[0]})")
    rep.p(L["gpu_note"])
    rep.p(f"{L['launch']}: {gpubench.launch_overhead() * 1e6:.0f} µs")
    rows = gpubench.run(robot, sizes, log=lambda r: print(
        f"  {names[r['op']]}, N = {r['n']:,}: CPU {r['cpuN'] * 1e3:.3g} ms, GPU {r['gpu32'] * 1e3:.3g} ms",
        flush=True))
    import robolab.transforms as rt
    n_cpu = rt.get_num_threads()
    cols = [("cpu1", "CPU 1"), ("cpuN", f"CPU {n_cpu}"), ("gpu32", "GPU f32"), ("gpu64", "GPU f64"),
            ("h2d", "H2D"), ("d2h", "D2H"), ("e2e32", L["e2e"])]
    for op in gpubench.OPS:
        rr = [r for r in rows if r["op"] == op]
        rep.p(f"\n**{names[op]}**")
        rep.table(["N"] + [c[1] for c in cols],
                  [[f"{r['n']:,}"] + [f"{r[k] * 1e3:.3g}" for k, _ in cols] for r in rr])
    on_gpu, copies = gpubench.thresholds(rows, "gpu32"), gpubench.thresholds(rows, "e2e32")
    rep.h(L["gpu_wins"])
    best = {op: [r for r in rows if r["op"] == op][-1] for op in gpubench.OPS}
    rep.table([L["op"], L["on_gpu"], L["with_copies"], L["best"]],
              [[names[op], f"{on_gpu[op]:,}" if on_gpu[op] else L["never"],
                f"{copies[op]:,}" if copies[op] else L["never"],
                f"{best[op]['cpuN'] / best[op]['gpu32']:.1f}x (N = {best[op]['n']:,})"]
               for op in gpubench.OPS])

    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, op in zip(axs.ravel(), gpubench.OPS):
        rr = [r for r in rows if r["op"] == op]
        n = [r["n"] for r in rr]
        for k, lab, st in (("cpu1", "CPU 1", "C7:"), ("cpuN", f"CPU {n_cpu}", "C7-"),
                           ("gpu32", "GPU f32", "C2-"), ("gpu64", "GPU f64", "C2--"),
                           ("e2e32", cols[-1][1], "C3-")):
            ax.plot(n, [r[k] * 1e3 for r in rr], st, marker="o", ms=3, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(names[op]); ax.set_xlabel("N")
    axs[0, 0].set_ylabel(L["time_ms"]); axs[1, 0].set_ylabel(L["time_ms"]); axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rep.out / "bench_gpu.png"); plt.close(fig)
    rep.file(rep.out / "bench_gpu.png")


# --------------------------------------------------------------------------- #
def main(argv=None):
    global L
    ap = argparse.ArgumentParser(prog="robolab", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["info", "walk", "sweep", "encoders", "bench", "all"])
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    ap.add_argument("--threads", type=int, default=None, help="OpenMP threads (default: all cores)")
    ap.add_argument("--quick", action="store_true", help="smaller sweep and benchmark")
    ap.add_argument("--gif", action="store_true", help="render walk.gif (slow on a Pi: ~1-2 min)")
    ap.add_argument("--samples", type=int, default=100_000, help="Monte-Carlo samples per resolution")
    ap.add_argument("--step-length", type=float, default=0.20)
    ap.add_argument("--step-time", type=float, default=0.70)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lang", choices=["ru", "en"], default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="where encoders (Monte-Carlo) and sweep (batched IK) run; with cuda, "
                         "bench adds GPU tables. cuda needs CuPy and an NVIDIA GPU")
    ap.add_argument("--dtype", choices=["float32", "float64"], default=None,
                    help="CUDA precision (default float32; float64 is for checks, slow on GeForce/Jetson)")
    args = ap.parse_args(argv)
    L = MSG[_pick_lang(args.lang)]

    import robolab.transforms as rt
    if args.threads:
        rt.set_num_threads(args.threads)
    if args.device == "cuda":
        try:
            rt._backend._load_cuda()
        except ImportError as exc:
            ap.error(f"--device cuda: {exc}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report(out)
    t0 = time.perf_counter()
    cmds = {"info": [cmd_info], "walk": [cmd_info, cmd_walk], "sweep": [cmd_sweep],
            "encoders": [cmd_encoders], "bench": [cmd_info, cmd_bench],
            "all": [cmd_info, cmd_walk, cmd_sweep, cmd_encoders, cmd_bench]}[args.command]
    try:
        for c in cmds:
            c(args, rep)
    finally:
        rep.save()
    print(f"\n{L['done']} {out.resolve()}/ ({time.perf_counter() - t0:.0f} {L['seconds']}), report.md")
    print(L["view"].format(out=out))


if __name__ == "__main__":
    main()
