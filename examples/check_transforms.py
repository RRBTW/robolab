"""Worked example: a leg as a chain of frames.

    python examples/check_transforms.py
"""
import numpy as np

from robolab.transforms import SE3, SE3Array, SO3, FrameMismatchError, get_backend

np.set_printoptions(precision=4, suppress=True)
print(f"kernel: {get_backend()}\n")

# Pelvis in world, hip yaw, knee pitch, foot offset. Each transform is T_parent_child.
T_world_pelvis = SE3(SO3.Rz(np.deg2rad(30)), [0.0, 0.0, 0.9], parent="world", child="pelvis")
T_pelvis_hip = SE3(SO3.Rz(np.deg2rad(10)), [0.0, -0.1, 0.0], parent="pelvis", child="hip")
T_hip_knee = SE3(SO3.Ry(np.deg2rad(20)), [0.0, 0.0, -0.42], parent="hip", child="knee")
T_knee_foot = SE3(SO3.Ry(np.deg2rad(-20)), [0.0, 0.0, -0.40], parent="knee", child="foot")

# Indices cancel: world<-pelvis<-hip<-knee<-foot
T_world_foot = T_world_pelvis @ T_pelvis_hip @ T_hip_knee @ T_knee_foot
print("T_world_foot:", T_world_foot.parent, "<-", T_world_foot.child)
print(T_world_foot.matrix, "\n")

# The toe has fixed coordinates in the foot frame; p and R carry it into the world.
toe_in_foot = np.array([0.15, 0.0, -0.05])
print("toe in world:", T_world_foot.act(toe_in_foot))

# Wrong order is caught immediately instead of producing silent garbage.
try:
    T_world_pelvis @ T_hip_knee
except FrameMismatchError as e:
    print("caught:", e)

# Round trips through the Lie algebra.
xi = T_world_foot.log()
print("\ntwist (omega, v):", xi)
print("exp(log(T)) == T:", SE3.exp(xi, parent="world", child="foot").isclose(T_world_foot))

# A step trajectory: 50 poses interpolated geodesically, then the toe path.
T_end = SE3.Trans(0.3, 0.0, 0.0, parent="world", child="world") @ T_world_foot
traj = SE3Array([T_world_foot.interpolate(T_end, s) for s in np.linspace(0, 1, 50)])
toe_path = traj.apply(toe_in_foot)          # (50, 3)
print(f"\n{traj}; toe moved by {np.linalg.norm(toe_path[-1] - toe_path[0]):.3f} m")
