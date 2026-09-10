"""3D Gaussian Splatting for Real-Time Radiance Field Rendering.

Pure NumPy implementation of Kerbl et al. (2023) "3D Gaussian Splatting for Real-Time Radiance Field Rendering":
- 3D Gaussian Primitive Representation: Mean, Covariance (Scale + Quaternion Rotation), Opacity, Color
- 2D Projected Covariance Matrix computation via Jacobian projection
- Tile-based / Point-based Alpha-Blending Splatting Renderer
"""

from typing import Tuple
import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q_norm[..., 0], q_norm[..., 1], q_norm[..., 2], q_norm[..., 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return np.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1).reshape(
        q.shape[:-1] + (3, 3)
    )


class GaussianSplatting3D:
    r"""3D Gaussian Scene Representation.

    Each 3D Gaussian primitive is defined by:
    - Center mean position :math:`\mu \in \mathbb{R}^3`
    - Scale vector :math:`s \in \mathbb{R}^3`
    - Quaternion rotation :math:`q \in \mathbb{R}^4`
    - Opacity :math:`\alpha \in [0, 1]`
    - RGB color / Spherical Harmonic :math:`c \in [0, 1]^3`

    Covariance: :math:`\Sigma = R S S^T R^T`
    """

    def __init__(self, num_gaussians: int) -> None:
        self.num_gaussians: int = num_gaussians
        self.means: np.ndarray = np.random.uniform(
            -1.0, 1.0, size=(num_gaussians, 3)
        ).astype(np.float32)
        self.scales: np.ndarray = np.random.uniform(
            0.01, 0.05, size=(num_gaussians, 3)
        ).astype(np.float32)
        self.quats: np.ndarray = np.zeros((num_gaussians, 4), dtype=np.float32)
        self.quats[:, 0] = 1.0
        self.opacities: np.ndarray = np.random.uniform(
            0.5, 0.9, size=(num_gaussians, 1)
        ).astype(np.float32)
        self.colors: np.ndarray = np.random.uniform(
            0.0, 1.0, size=(num_gaussians, 3)
        ).astype(np.float32)

    def compute_3d_covariances(self) -> np.ndarray:
        """Compute (N, 3, 3) 3D covariance matrices Sigma = R S S^T R^T."""
        rot = quaternion_to_rotation_matrix(self.quats)
        scale_mat: np.ndarray = np.zeros((self.num_gaussians, 3, 3), dtype=np.float32)
        for i in range(3):
            scale_mat[:, i, i] = self.scales[:, i]

        m = rot @ scale_mat
        cov3d = m @ np.swapaxes(m, -1, -2)
        return cov3d

    def project_to_2d(
        self,
        camera_pose: np.ndarray,
        focal_length: float,
        img_width: int,
        img_height: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D Gaussians to 2D image plane."""
        cov3d = self.compute_3d_covariances()

        w2c = np.linalg.inv(np.asarray(camera_pose, dtype=np.float32))
        r_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]

        means_cam = self.means @ r_w2c.T + t_w2c
        depths = means_cam[:, 2]

        x_cam, y_cam, z_cam = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
        z_safe = np.maximum(z_cam, 1e-4)

        u = (x_cam / z_safe) * focal_length + img_width * 0.5
        v = (y_cam / z_safe) * focal_length + img_height * 0.5
        means_2d = np.stack([u, v], axis=-1)

        j: np.ndarray = np.zeros((self.num_gaussians, 2, 3), dtype=np.float32)
        j[:, 0, 0] = focal_length / z_safe
        j[:, 0, 2] = -focal_length * x_cam / (z_safe**2)
        j[:, 1, 1] = focal_length / z_safe
        j[:, 1, 2] = -focal_length * y_cam / (z_safe**2)

        jw = j @ r_w2c
        cov2d = jw @ cov3d @ np.swapaxes(jw, -1, -2)

        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3

        return means_2d, cov2d, depths

    def render(
        self,
        camera_pose: np.ndarray,
        focal_length: float,
        img_width: int = 64,
        img_height: int = 64,
    ) -> np.ndarray:
        """Render RGB image via front-to-back alpha-blending splatting."""
        means_2d, cov2d, depths = self.project_to_2d(
            camera_pose, focal_length, img_width, img_height
        )

        valid = depths > 0.1
        if not np.any(valid):
            return np.zeros((img_height, img_width, 3), dtype=np.float32)

        sort_idx = np.argsort(depths)
        sorted_means = means_2d[sort_idx]
        sorted_covs = cov2d[sort_idx]
        sorted_opacities = self.opacities[sort_idx].ravel()
        sorted_colors = self.colors[sort_idx]

        image: np.ndarray = np.zeros((img_height, img_width, 3), dtype=np.float32)
        transmittance: np.ndarray = np.ones((img_height, img_width), dtype=np.float32)

        grid_y: np.ndarray
        grid_x: np.ndarray
        grid_y, grid_x = np.meshgrid(
            np.arange(img_height), np.arange(img_width), indexing="ij"
        )
        pixels = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

        for i in range(len(sort_idx)):
            if not valid[sort_idx[i]]:
                continue

            mu = sorted_means[i]
            cov = sorted_covs[i]
            alpha_base = sorted_opacities[i]
            color = sorted_colors[i]

            det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
            if det <= 1e-6:
                continue

            inv_cov = (
                np.array(
                    [[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]], dtype=np.float32
                )
                / det
            )

            diff = pixels - mu[None, None, :]
            power = -0.5 * (
                diff[..., 0]
                * (diff[..., 0] * inv_cov[0, 0] + diff[..., 1] * inv_cov[1, 0])
                + diff[..., 1]
                * (diff[..., 0] * inv_cov[0, 1] + diff[..., 1] * inv_cov[1, 1])
            )

            gaussian_weights = np.exp(np.clip(power, -10.0, 0.0))
            alpha = np.clip(alpha_base * gaussian_weights, 0.0, 0.99)

            weight = alpha * transmittance
            image += weight[..., None] * color[None, None, :]
            transmittance *= 1.0 - alpha

            if np.max(transmittance) < 0.001:
                break

        return np.clip(image, 0.0, 1.0)
