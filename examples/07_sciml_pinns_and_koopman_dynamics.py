"""Example 7: Scientific Machine Learning (SciML), PINNs, FNO & Koopman Dynamics."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.models.sciml import (
    BurgersPINN,
    DynamicModeDecomposition,
    ExtendedDMD,
    FourierNeuralOperator2D,
)


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 7: SCIENTIFIC ML, PINNS & DYNAMICAL SYSTEMS")
    print("=" * 70)

    # 1. Physics-Informed Neural Networks (PINN) for Burgers Equation
    print("[1] Physics-Informed Neural Network (Burgers PINN):")
    pinn = BurgersPINN(nu=0.01 / np.pi, hidden_layers=[32, 32])
    t_colloc = np.random.uniform(0, 1, size=(50, 1))
    x_colloc = np.random.uniform(-1, 1, size=(50, 1))
    tx_colloc = np.column_stack([t_colloc, x_colloc])
    pde_res = pinn.pde_residual(tx_colloc)
    print(f"  Spacetime collocation points : {tx_colloc.shape}")
    print(f"  Burgers PDE residual loss norm: {np.mean(pde_res**2):.6f}")

    # 2. Fourier Neural Operator 2D (FNO)
    print("\n[2] Fourier Neural Operator (2D FNO):")
    fno = FourierNeuralOperator2D(modes1=8, modes2=8, hidden_dim=16)
    spatial_field = np.random.randn(2, 32, 32, 1)
    fno_out = fno.forward(spatial_field)
    print(f"  Input PDE coefficient field shape: {spatial_field.shape}")
    print(f"  FNO predicted solution field shape : {fno_out.shape}")

    # 3. Dynamic Mode Decomposition (DMD)
    print("\n[3] Dynamic Mode Decomposition (DMD) of Fluid / Wave Dynamics:")
    t_span = np.linspace(0, 4 * np.pi, 60)
    x_grid = np.linspace(-5, 5, 20)
    T_mat, X_mat = np.meshgrid(t_span, x_grid)
    mode1 = (1.0 / np.cosh(X_mat)) * np.exp(1j * 2.5 * T_mat)
    mode2 = np.tanh(X_mat) * (1.0 / np.cosh(X_mat)) * np.exp(1j * 4.0 * T_mat)
    snapshots = np.real(mode1 + mode2)

    dmd = DynamicModeDecomposition(rank=4)
    dmd.fit(snapshots)
    print(f"  Snapshot matrix shape           : {snapshots.shape}")
    print(f"  Extracted coherent DMD modes    : {dmd.modes.shape}")
    print(f"  DMD continuous eigenvalues (freq): {np.round(dmd.eigenvalues, 3)}")

    # 4. Extended DMD with Non-Linear Observables
    edmd = ExtendedDMD(polynomial_degree=2)
    edmd.fit(snapshots)
    next_step = edmd.predict_step(snapshots[:, -1])
    print(f"  Extended DMD predicted next state step: {np.round(next_step[:4], 3)}...")
    print("=" * 70)


if __name__ == "__main__":
    main()
