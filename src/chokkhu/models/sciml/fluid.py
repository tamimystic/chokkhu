"""Differentiable Smoothed Particle Hydrodynamics (SPH) Fluid Simulation.

Formulated from first principles using cubic spline/Wendland smoothing kernels,
Tait equation of state, Navier-Stokes pressure/viscosity forces, and symplectic integration.
"""

import numpy as np
from typing import Optional, Tuple


class DifferentiableParticleFluid:
    """Smoothed Particle Hydrodynamics (SPH) Differentiable Simulation Engine.

    Parameters
    ----------
    num_particles : int, default=100
        Number of simulated fluid particles.
    dim : int, default=2
        Spatial dimensions (2 for 2D or 3 for 3D).
    rest_density : float, default=1000.0
        Rest mass density rho_0 (kg/m^3).
    stiffness : float, default=200.0
        Gas constant / stiffness parameter k in Tait equation of state.
    viscosity : float, default=0.1
        Dynamic fluid viscosity coefficient mu (Pa.s).
    smoothing_length : float, default=0.1
        Support radius h of SPH kernel.
    particle_mass : float, default=0.02
        Mass m of individual fluid particle (kg).
    dt : float, default=0.005
        Integration time step delta_t (seconds).
    gravity : Optional[np.ndarray], default=None
        External acceleration vector. Defaults to [0, -9.81].
    bounds : Optional[Tuple[float, float, float, float]], default=None
        Simulation domain boundaries [xmin, xmax, ymin, ymax].
    seed : int, default=42
        Random seed for particle initialization.
    """

    def __init__(
        self,
        num_particles: int = 100,
        dim: int = 2,
        rest_density: float = 1000.0,
        stiffness: float = 200.0,
        viscosity: float = 0.1,
        smoothing_length: float = 0.1,
        particle_mass: float = 0.02,
        dt: float = 0.005,
        gravity: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        seed: int = 42,
    ) -> None:
        self.num_particles = int(num_particles)
        self.dim = int(dim)
        self.rest_density = float(rest_density)
        self.stiffness = float(stiffness)
        self.viscosity = float(viscosity)
        self.h = float(smoothing_length)
        self.mass = float(particle_mass)
        self.dt = float(dt)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        if gravity is None:
            self.gravity = np.zeros(self.dim)
            if self.dim >= 2:
                self.gravity[1] = -9.81
        else:
            self.gravity = np.asarray(gravity, dtype=float)

        self.bounds = bounds if bounds is not None else (-1.0, 1.0, -1.0, 1.0)
        self.h2 = self.h * self.h

        # Normalization constants for 2D kernels
        if self.dim == 2:
            self.poly6_coeff = 315.0 / (64.0 * np.pi * (self.h**9))
            self.spiky_grad_coeff = -45.0 / (np.pi * (self.h**6))
            self.visc_lap_coeff = 45.0 / (np.pi * (self.h**6))
        else:
            self.poly6_coeff = 315.0 / (64.0 * np.pi * (self.h**9))
            self.spiky_grad_coeff = -45.0 / (np.pi * (self.h**6))
            self.visc_lap_coeff = 45.0 / (np.pi * (self.h**6))

    def compute_density(self, positions: np.ndarray) -> np.ndarray:
        """Compute particle density field rho_i via Poly6 smoothing kernel.

        Parameters
        ----------
        positions : np.ndarray of shape (N, dim)
            Current particle coordinates.

        Returns
        -------
        densities : np.ndarray of shape (N,)
            Mass density per particle.
        """
        # Pairwise displacement vectors r_ij = x_i - x_j
        diff = positions[:, None, :] - positions[None, :, :]  # (N, N, dim)
        r2 = np.sum(diff**2, axis=-1)  # (N, N)

        # Kernel W_poly6(r, h) = (h^2 - r^2)^3
        in_range = r2 < self.h2
        kernel_vals = np.where(in_range, self.poly6_coeff * ((self.h2 - r2) ** 3), 0.0)

        densities = self.mass * np.sum(kernel_vals, axis=1)
        return np.maximum(densities, self.rest_density * 0.1)

    def compute_pressure(self, densities: np.ndarray) -> np.ndarray:
        """Compute pressure via Tait equation of state: P_i = k * (rho_i - rho_0)."""
        return np.maximum(0.0, self.stiffness * (densities - self.rest_density))

    def compute_forces(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        """Compute total net forces (pressure + viscosity + gravity) per particle.

        Parameters
        ----------
        positions : np.ndarray of shape (N, dim)
            Particle positions.
        velocities : np.ndarray of shape (N, dim)
            Particle velocities.

        Returns
        -------
        forces : np.ndarray of shape (N, dim)
            Net force vectors per particle.
        """
        densities = self.compute_density(positions)
        pressures = self.compute_pressure(densities)

        diff = positions[:, None, :] - positions[None, :, :]  # (N, N, dim)
        r = np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)  # (N, N)
        in_range = (r < self.h) & (r > 1e-7)

        # 1. Pressure gradient force: F_p = -sum_j m * (P_i + P_j)/(2 * rho_j) * grad_W_spiky
        # grad_W_spiky = -45 / (pi * h^6) * (h - r)^2 * (r_ij / r)
        spiky_factor = np.where(
            in_range, self.spiky_grad_coeff * ((self.h - r) ** 2) / r, 0.0
        )
        p_term = (pressures[:, None] + pressures[None, :]) / (
            2.0 * densities[None, :] + 1e-12
        )
        grad_w = spiky_factor[:, :, None] * diff  # (N, N, dim)
        f_pressure = -self.mass * np.sum(p_term[:, :, None] * grad_w, axis=1)

        # 2. Viscosity force: F_v = mu * sum_j m * (v_j - v_i)/rho_j * lap_W_visc
        # lap_W_visc = 45 / (pi * h^6) * (h - r)
        visc_lap = np.where(in_range, self.visc_lap_coeff * (self.h - r), 0.0)
        v_diff = velocities[None, :, :] - velocities[:, None, :]  # (N, N, dim)
        f_viscosity = (
            self.viscosity
            * self.mass
            * np.sum(
                (visc_lap[:, :, None] / (densities[None, :, None] + 1e-12)) * v_diff,
                axis=1,
            )
        )

        # 3. External Gravity Force
        f_gravity = self.mass * self.gravity[None, :]

        total_forces = f_pressure + f_viscosity + f_gravity
        return total_forces

    def step(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one step of symplectic Euler time integration with boundary collisions.

        Parameters
        ----------
        positions : np.ndarray of shape (N, dim)
            Current positions.
        velocities : np.ndarray of shape (N, dim)
            Current velocities.

        Returns
        -------
        new_positions : np.ndarray of shape (N, dim)
        new_velocities : np.ndarray of shape (N, dim)
        """
        forces = self.compute_forces(positions, velocities)
        accelerations = forces / self.mass

        # Symplectic Euler update
        new_velocities = velocities + accelerations * self.dt
        new_positions = positions + new_velocities * self.dt

        # Boundary collision handling (elastic restitution)
        if self.bounds is not None and self.dim == 2:
            xmin, xmax, ymin, ymax = self.bounds
            restitution = 0.5

            # X boundaries
            mask_xmin = new_positions[:, 0] < xmin
            new_positions[mask_xmin, 0] = xmin
            new_velocities[mask_xmin, 0] *= -restitution

            mask_xmax = new_positions[:, 0] > xmax
            new_positions[mask_xmax, 0] = xmax
            new_velocities[mask_xmax, 0] *= -restitution

            # Y boundaries
            mask_ymin = new_positions[:, 1] < ymin
            new_positions[mask_ymin, 1] = ymin
            new_velocities[mask_ymin, 1] *= -restitution

            mask_ymax = new_positions[:, 1] > ymax
            new_positions[mask_ymax, 1] = ymax
            new_velocities[mask_ymax, 1] *= -restitution

        return new_positions, new_velocities

    def simulate(
        self,
        initial_positions: np.ndarray,
        initial_velocities: Optional[np.ndarray] = None,
        num_steps: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate fluid trajectory over multiple time steps.

        Returns
        -------
        pos_history : np.ndarray of shape (num_steps + 1, N, dim)
        vel_history : np.ndarray of shape (num_steps + 1, N, dim)
        """
        pos = initial_positions.copy()
        if initial_velocities is None:
            vel = np.zeros_like(pos)
        else:
            vel = initial_velocities.copy()

        pos_hist = [pos.copy()]
        vel_hist = [vel.copy()]

        for _ in range(num_steps):
            pos, vel = self.step(pos, vel)
            pos_hist.append(pos.copy())
            vel_hist.append(vel.copy())

        return np.array(pos_hist), np.array(vel_hist)
