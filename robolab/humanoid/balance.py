"""Balance metrics: multi-body ZMP and the support polygon."""
from __future__ import annotations

import numpy as np

from .gait import G


def multibody_zmp(link_com, masses, dt):
    """ZMP of a set of point masses on flat ground.

        p_x = sum m ((z'' + g) x - z x'') / sum m (z'' + g)   (same for y)

    link_com (T, L, 3) world CoM trajectories of the links, masses (L,).
    Link rotational inertia is neglected (point-mass model), accelerations are
    central finite differences."""
    acc = np.gradient(np.gradient(link_com, dt, axis=0, edge_order=2), dt, axis=0, edge_order=2)
    x, y, z = link_com[..., 0], link_com[..., 1], link_com[..., 2]
    ax, ay, az = acc[..., 0], acc[..., 1], acc[..., 2]
    den = ((az + G) * masses).sum(axis=1)
    px = (((az + G) * x - z * ax) * masses).sum(axis=1) / den
    py = (((az + G) * y - z * ay) * masses).sum(axis=1) / den
    return np.stack([px, py], axis=1)


def foot_corners(sole_poses, length, width):
    """(T, 4, 2) ground corners of the foot soles from their (T, 4, 4) poses."""
    local = np.array([[length / 2, width / 2, 0], [length / 2, -width / 2, 0],
                      [-length / 2, -width / 2, 0], [-length / 2, width / 2, 0]])
    world = np.einsum("tij,cj->tci", sole_poses[:, :3, :3], local) + sole_poses[:, None, :3, 3]
    return world[..., :2]


def _hull(points):
    """Convex hull (Andrew's monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, points))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def support_polygons(phase, left_corners, right_corners):
    """List of CCW polygons: both feet in double support (0), one foot otherwise."""
    polys = []
    for k, ph in enumerate(phase):
        if ph == 1:
            pts = left_corners[k]
        elif ph == 2:
            pts = right_corners[k]
        else:
            pts = np.vstack([left_corners[k], right_corners[k]])
        polys.append(_hull(pts))
    return polys


def signed_margin(point, poly):
    """Distance from point to the polygon boundary: > 0 inside, < 0 outside."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    dist = np.linalg.norm(a + t[:, None] * ab - point, axis=1).min()
    inside = np.all(ab[:, 0] * (point[1] - a[:, 1]) - ab[:, 1] * (point[0] - a[:, 0]) >= 0)
    return dist if inside else -dist
