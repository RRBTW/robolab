# se3.py — ИСХОДНЫЙ ФРАГМЕНТ (прислан в чате, обрезан на докстринге to_matrix).
# Сохранён как есть для истории и бенчмарков; рабочая копия с дописанным
# to_matrix встроена в notebooks/benchmark_transforms.ipynb (ячейка se3_ilya.py).
# ВНИМАНИЕ: докстринг с LaTeX не raw-строка -> SyntaxWarning на Python 3.12+.
import numpy as np


class SE3:
    """
    Special Euclidean group in 3D (SE(3)) representation.

    Represents a rigid body transformation (rotation and translation)
    in 3D space. Mathematically, an element of SE(3) is an ordered pair 
    defined as:

    $$ SE(3) = \{ \langle R, p \rangle \mid R \in SO(3), p \in \mathbb{R}^3 \} $$

    Here, $R$ is the rotation matrix and $p$ is the translation vector.
    
    The rotation matrix $R$ belongs to the Special Orthogonal group $SO(3)$.
    $SO(3)$ is the group of all valid 3D rotations. It is defined as the set 
    of all $3 \times 3$ orthogonal matrices with a determinant of 1, which 
    ensures that the transformation preserves distances and orientation 
    (no reflections):

    $$ SO(3) = \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \det(R) = 1 \} $$

    The ordered pair $\langle R, p \rangle$ is commonly represented as a 
    4x4 homogeneous transformation matrix, where the bottom-right element 
    is the scalar 1 (a $1 \times 1$ identity matrix):

    $$ T = \begin{bmatrix} R & p \\ 0_{1 \times 3} & 1 \end{bmatrix} $$

    Physical Interpretation
    -----------------------
    In anthropomorphic robotics, we consider a logical body part (e.g., a leg 
    or a foot) that has its own local coordinate system. Inside this local 
    frame, all its sub-parts (e.g., specific foot segments or joints) are 
    described by fixed local coordinates.

    The translation vector $p$ describes the action of moving the *entire* 
    local coordinate system to a specific location within the global coordinate 
    system. It dictates where the whole logical body part is placed globally.

    The rotation matrix $R \in SO(3)$ describes the orientation of this local 
    coordinate system, in which all its internal parts are defined. By using $R$, 
    we compute how to transform and move each of these internal elements into the 
    global coordinates, effectively describing the global action and position of 
    every part inside the logical body part.

    Instance Attributes
    -------------------
    R : numpy.ndarray
        3x3 rotation matrix (belongs to SO(3)).
    p : numpy.ndarray
        3-element translation vector.
    """

    def __init__(self, R: np.ndarray, p: np.ndarray):
        R = np.asarray(R, dtype=float)
        p = np.asarray(p, dtype=float).flatten()
        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix R must be 3x3, got {R.shape}")
        if p.shape != (3,):
            raise ValueError(f"Translation vector p must be of size 3, got {p.shape}")
        self.R = R
        self.p = p

    @classmethod
    def identity(cls) -> 'SE3':
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'SE3':
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"Matrix T must be 4x4, got {T.shape}")
        return cls(T[:3, :3], T[:3, 3])

    def to_matrix(self) -> np.ndarray:
        """
        Convert to a 4x4 homogeneous transformation matrix.
        """
        # --- фрагмент в чате обрывается здесь ---
        raise NotImplementedError
