"""Matplotlib helpers (optional dependency, imported lazily)."""
from __future__ import annotations

import numpy as np

from .se3_group import SE3
from .se3_array import SE3Array


def plot_frame(t, ax=None, length=0.2, label=None):
    """Draw the child frame's axes (X red, Y green, Z blue) in the parent frame.

    ``t`` is an SE3 or a 4x4 matrix. The label defaults to the child frame name.
    Returns the Matplotlib 3D Axes.
    """
    if not isinstance(t, SE3):
        t = SE3.from_matrix(t)
    length = float(length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be a positive finite number")
    if label is None:
        label = t.child
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111, projection="3d")
    origin, r = t.translation, t.R
    for i, (color, name) in enumerate(zip(("r", "g", "b"), ("X", "Y", "Z"))):
        vec = length * r[:, i]
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.15)
        ax.text(*(origin + vec), f"{label + ':' if label else ''}{name}", color=color)
    return ax


def plot_trajectory(poses: SE3Array, ax=None, length=0.05, every=1):
    """Draw the path of the frame origins and a frame every ``every`` poses."""
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111, projection="3d")
    p = poses.translations
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "k-", lw=1)
    for i in range(0, len(poses), max(1, int(every))):
        plot_frame(poses[i], ax=ax, length=length, label="")
    return ax
