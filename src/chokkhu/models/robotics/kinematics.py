"""Robotics Kinematics and Differentiable Manipulator Geometry.

Formulated from first principles using Denavit-Hartenberg (DH) parameterization,
geometric & analytical Jacobians, and Damped Least Squares (DLS) Inverse Kinematics.
"""

import numpy as np
from typing import Optional, Tuple, List


class RobotArmKinematics:
    """Multi-DoF Serial Manipulator Forward & Inverse Kinematics Engine.

    Parameters
    ----------
    dh_params : np.ndarray of shape (N, 4)
        Standard Denavit-Hartenberg parameters per link:
        each row is [a (link length), alpha (twist angle), d (link offset), theta_offset].
    joint_limits : Optional[List[Tuple[float, float]]], default=None
        Per-joint minimum and maximum angle constraints (radians).
    """

    def __init__(
        self,
        dh_params: np.ndarray,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.dh_params = np.asarray(dh_params, dtype=float)
        if self.dh_params.ndim != 2 or self.dh_params.shape[1] != 4:
            raise ValueError(
                f"dh_params must have shape (N, 4), got {self.dh_params.shape}"
            )

        self.num_joints = self.dh_params.shape[0]
        if joint_limits is not None:
            if len(joint_limits) != self.num_joints:
                raise ValueError(
                    f"joint_limits length {len(joint_limits)} != num_joints {self.num_joints}"
                )
            self.joint_limits = joint_limits
        else:
            self.joint_limits = [(-np.pi, np.pi) for _ in range(self.num_joints)]

    @staticmethod
    def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute standard 4x4 Denavit-Hartenberg homogeneous transformation matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        T = np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return T

    def forward_kinematics(
        self, joint_angles: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Compute end-effector homogeneous transformation and all intermediate joint frames.

        Parameters
        ----------
        joint_angles : np.ndarray of shape (num_joints,)
            Current joint angles in radians.

        Returns
        -------
        T_end : np.ndarray of shape (4, 4)
            End-effector 4x4 homogeneous transformation matrix in base frame.
        transforms : List[np.ndarray]
            List of (num_joints + 1) transformation matrices from base frame to each frame i.
        """
        angles = np.asarray(joint_angles, dtype=float).flatten()
        if len(angles) != self.num_joints:
            raise ValueError(
                f"joint_angles length {len(angles)} != num_joints {self.num_joints}"
            )

        T_current: np.ndarray = np.eye(4, dtype=float)
        transforms = [T_current.copy()]

        for i in range(self.num_joints):
            a_i, alpha_i, d_i, theta_off = self.dh_params[i]
            theta_i = angles[i] + theta_off
            T_i = self.dh_transform(a_i, alpha_i, d_i, theta_i)
            T_current = np.dot(T_current, T_i)
            transforms.append(T_current.copy())

        return T_current, transforms

    def get_end_effector_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """Extract 3D Cartesian position [x, y, z] of the end-effector."""
        T_end, _ = self.forward_kinematics(joint_angles)
        return T_end[:3, 3]

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute the 6 x N geometric Jacobian matrix J = [J_v; J_omega].

        Parameters
        ----------
        joint_angles : np.ndarray of shape (num_joints,)
            Joint angles in radians.

        Returns
        -------
        J : np.ndarray of shape (6, num_joints)
            Geometric Jacobian relating joint velocities to end-effector linear and angular velocities.
        """
        T_end, transforms = self.forward_kinematics(joint_angles)
        p_e = T_end[:3, 3]
        J = np.zeros((6, self.num_joints), dtype=float)

        for i in range(self.num_joints):
            T_i_prev = transforms[i]
            z_i_prev = T_i_prev[:3, 2]  # Rotation axis of joint i
            p_i_prev = T_i_prev[:3, 3]  # Origin of joint frame i-1

            # Linear velocity Jacobian component: J_v = z_{i-1} x (p_e - p_{i-1})
            J[:3, i] = np.cross(z_i_prev, p_e - p_i_prev)
            # Angular velocity Jacobian component: J_omega = z_{i-1}
            J[3:, i] = z_i_prev

        return J

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_angles: Optional[np.ndarray] = None,
        max_iters: int = 100,
        tolerance: float = 1e-4,
        damping: float = 0.05,
        step_size: float = 0.5,
    ) -> Tuple[np.ndarray, bool, int]:
        """Solve Inverse Kinematics using Damped Least Squares (DLS / Levenberg-Marquardt).

        Parameters
        ----------
        target_position : np.ndarray of shape (3,)
            Desired [x, y, z] Cartesian position.
        target_orientation : Optional[np.ndarray] of shape (3, 3), default=None
            Desired 3x3 rotation matrix. If None, position-only IK is solved.
        initial_angles : Optional[np.ndarray] of shape (num_joints,), default=None
            Starting joint configuration. Defaults to zeros.
        max_iters : int, default=100
            Maximum iteration steps.
        tolerance : float, default=1e-4
            Convergence Euclidean error threshold.
        damping : float, default=0.05
            Damping parameter lambda for singularity robustness.
        step_size : float, default=0.5
            Newton step size scaling.

        Returns
        -------
        joint_angles : np.ndarray of shape (num_joints,)
            Solved joint angles in radians.
        converged : bool
            Whether target was reached within tolerance.
        iterations : int
            Number of iterations executed.
        """
        target_pos = np.asarray(target_position, dtype=float).flatten()[:3]
        if initial_angles is None:
            current_angles = np.zeros(self.num_joints, dtype=float)
        else:
            current_angles = np.asarray(initial_angles, dtype=float).flatten().copy()

        for it in range(max_iters):
            T_end, transforms = self.forward_kinematics(current_angles)
            current_pos = T_end[:3, 3]
            pos_err = target_pos - current_pos

            if target_orientation is not None:
                # Orientation error via rotation matrix cross-product
                R_curr = T_end[:3, :3]
                R_err = np.dot(target_orientation, R_curr.T)
                # Small angle axis-angle approximation
                rot_err = 0.5 * np.array(
                    [
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ]
                )
                error = np.concatenate([pos_err, rot_err])
                J = self.compute_jacobian(current_angles)
            else:
                error = pos_err
                J = self.compute_jacobian(current_angles)[:3, :]

            err_norm = np.linalg.norm(error)
            if err_norm < tolerance:
                return current_angles, True, it

            # Damped Least Squares: delta_theta = J^T * (J * J^T + lambda^2 * I)^(-1) * error
            m = J.shape[0]
            JJT_damped = np.dot(J, J.T) + (damping**2) * np.eye(m)
            dls_inv = np.dot(J.T, np.linalg.solve(JJT_damped, error))

            current_angles += step_size * dls_inv

            # Apply joint limits
            for j in range(self.num_joints):
                low, high = self.joint_limits[j]
                current_angles[j] = np.clip(current_angles[j], low, high)

        final_pos = self.get_end_effector_position(current_angles)
        converged = bool(np.linalg.norm(target_pos - final_pos) < tolerance)
        return current_angles, converged, max_iters

    @classmethod
    def planar_2d(cls, link_lengths: List[float] = [1.0, 1.0]) -> "RobotArmKinematics":
        """Factory constructor for standard 2-link planar revolute robot arm."""
        dh = []
        for l_len in link_lengths:
            dh.append([l_len, 0.0, 0.0, 0.0])
        return cls(np.array(dh, dtype=float))

    @classmethod
    def puma560(cls) -> "RobotArmKinematics":
        """Factory constructor for standard 6-DoF Puma 560 industrial manipulator."""
        puma_dh = [
            [0.0, np.pi / 2, 0.0, 0.0],
            [0.4318, 0.0, 0.0, 0.0],
            [0.0203, -np.pi / 2, 0.150, 0.0],
            [0.0, np.pi / 2, 0.4318, 0.0],
            [0.0, -np.pi / 2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        return cls(np.array(puma_dh, dtype=float))
