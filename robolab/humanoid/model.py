"""A 21-DoF humanoid (~1.45 m, ~37 kg) for kinematic studies.

Frames: x forward, y left, z up. All joint frames are aligned with the pelvis
at q = 0 (standing straight, arms down), so joint axes are simply x, y or z:

    pelvis (floating base)
    ├─ legs  (x2): hip_yaw z, hip_roll x, hip_pitch y, knee y, ankle_pitch y,
    │              ankle_roll x, + fixed sole frame on the ground contact
    ├─ waist_yaw z ─ torso
    │   ├─ arms (x2): shoulder_pitch y, shoulder_roll x, elbow y, + fixed hand
    │   └─ neck_yaw z, neck_pitch y ─ head

Proportions and masses are typical of research humanoids of this size; they
are parameters of make_humanoid(), not measurements of a specific robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics import KinematicTree, Link
from ..transforms import SE3

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
DEG = np.pi / 180.0


@dataclass(frozen=True)
class HumanoidDims:
    hip_width: float = 0.09        # lateral offset of each hip from the pelvis centre
    thigh: float = 0.36
    shank: float = 0.36
    ankle_height: float = 0.06     # ankle axis above the sole
    foot_length: float = 0.22
    foot_width: float = 0.10
    foot_ahead: float = 0.03       # sole centre ahead of the ankle axis
    waist_height: float = 0.10
    shoulder_height: float = 0.40  # above the waist joint
    shoulder_width: float = 0.18
    upper_arm: float = 0.26
    forearm: float = 0.24
    neck_height: float = 0.48      # above the waist joint
    m_pelvis: float = 5.0
    m_thigh: float = 4.0
    m_shank: float = 2.5
    m_foot: float = 1.0
    m_torso: float = 10.0
    m_upper_arm: float = 1.5
    m_forearm: float = 1.0
    m_head: float = 2.5

    @property
    def leg_length(self):
        return self.thigh + self.shank + self.ankle_height


def _leg(side, d: HumanoidDims):
    s = 1.0 if side == "l" else -1.0
    roll_out = (-25 * DEG, 45 * DEG) if side == "l" else (-45 * DEG, 25 * DEG)
    I = SE3.identity
    return [
        Link(f"{side}_hip_yaw", "pelvis", SE3.Trans(0, s * d.hip_width, 0), f"{side}_hip_yaw", Z,
             limits=(-45 * DEG, 45 * DEG)),
        Link(f"{side}_hip_roll", f"{side}_hip_yaw", I(), f"{side}_hip_roll", X, limits=roll_out),
        Link(f"{side}_thigh", f"{side}_hip_roll", I(), f"{side}_hip_pitch", Y,
             mass=d.m_thigh, com=(0, 0, -d.thigh / 2), limits=(-120 * DEG, 45 * DEG)),
        Link(f"{side}_shank", f"{side}_thigh", SE3.Trans(0, 0, -d.thigh), f"{side}_knee", Y,
             mass=d.m_shank, com=(0, 0, -d.shank / 2), limits=(0 * DEG, 150 * DEG)),
        Link(f"{side}_ankle_pitch", f"{side}_shank", SE3.Trans(0, 0, -d.shank), f"{side}_ankle_pitch", Y,
             limits=(-75 * DEG, 45 * DEG)),
        Link(f"{side}_foot", f"{side}_ankle_pitch", I(), f"{side}_ankle_roll", X,
             mass=d.m_foot, com=(d.foot_ahead, 0, -d.ankle_height / 2), limits=(-30 * DEG, 30 * DEG)),
        Link(f"{side}_sole", f"{side}_foot", SE3.Trans(d.foot_ahead, 0, -d.ankle_height)),
    ]


def _arm(side, d: HumanoidDims):
    s = 1.0 if side == "l" else -1.0
    roll_out = (-10 * DEG, 120 * DEG) if side == "l" else (-120 * DEG, 10 * DEG)
    return [
        Link(f"{side}_shoulder", "torso", SE3.Trans(0, s * d.shoulder_width, d.shoulder_height),
             f"{side}_shoulder_pitch", Y, limits=(-180 * DEG, 90 * DEG)),
        Link(f"{side}_upper_arm", f"{side}_shoulder", SE3.identity(), f"{side}_shoulder_roll", X,
             mass=d.m_upper_arm, com=(0, 0, -d.upper_arm / 2), limits=roll_out),
        Link(f"{side}_forearm", f"{side}_upper_arm", SE3.Trans(0, 0, -d.upper_arm), f"{side}_elbow", Y,
             mass=d.m_forearm, com=(0, 0, -d.forearm / 2), limits=(-150 * DEG, 0 * DEG)),
        Link(f"{side}_hand", f"{side}_forearm", SE3.Trans(0, 0, -d.forearm)),
    ]


def make_humanoid(dims: HumanoidDims = HumanoidDims()) -> KinematicTree:
    d = dims
    links = [Link("pelvis", None, SE3.identity(), mass=d.m_pelvis, com=(0, 0, 0.03))]
    links += _leg("l", d) + _leg("r", d)
    links += [
        Link("torso", "pelvis", SE3.Trans(0, 0, d.waist_height), "waist_yaw", Z,
             mass=d.m_torso, com=(0, 0, 0.22), limits=(-60 * DEG, 60 * DEG)),
    ]
    links += _arm("l", d) + _arm("r", d)
    links += [
        Link("neck", "torso", SE3.Trans(0, 0, d.neck_height), "neck_yaw", Z, limits=(-80 * DEG, 80 * DEG)),
        Link("head", "neck", SE3.identity(), "neck_pitch", Y,
             mass=d.m_head, com=(0.02, 0, 0.10), limits=(-40 * DEG, 40 * DEG)),
    ]
    tree = KinematicTree(links, base_frame="world")
    tree.dims = d
    return tree


LEG_JOINTS = {side: [f"{side}_hip_yaw", f"{side}_hip_roll", f"{side}_hip_pitch", f"{side}_knee",
                     f"{side}_ankle_pitch", f"{side}_ankle_roll"] for side in "lr"}

# Line segments for drawing a stick figure: pairs of link names.
SKELETON = [
    ("pelvis", "l_hip_yaw"), ("pelvis", "r_hip_yaw"),
    ("l_hip_yaw", "l_shank"), ("l_shank", "l_foot"), ("l_foot", "l_sole"),
    ("r_hip_yaw", "r_shank"), ("r_shank", "r_foot"), ("r_foot", "r_sole"),
    ("pelvis", "torso"), ("torso", "neck"), ("neck", "head"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_forearm"), ("l_forearm", "l_hand"),
    ("r_shoulder", "r_forearm"), ("r_forearm", "r_hand"),
]
