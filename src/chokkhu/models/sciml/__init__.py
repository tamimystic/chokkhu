"""Scientific Machine Learning, PINNs & Symbolic Modeling Package.

Pure NumPy implementations of:
- PINN, BurgersPINN, HeatPINN, WavePINN, HarmonicOscillatorPINN
- SymbolicRegressor: Sparse equation discovery (SINDy)
- NeuralODE: Continuous-depth ordinary differential equation models
- SpectralConv2d, FourierNeuralOperator2D: 2D Fourier Neural Operator for PDEs
- SparseGaussianProcessRegression, VariationalSparseGP: Inducing point sparse GPs
"""

from .pinn import (
    PINN,
    BurgersPINN,
    HeatPINN,
    WavePINN,
    HarmonicOscillatorPINN,
)
from .symbolic_regression import SymbolicRegressor
from .neural_ode import NeuralODE
from .fluid import DifferentiableParticleFluid
from .fno import SpectralConv2d, FourierNeuralOperator2D
from .sparse_gp import SparseGaussianProcessRegression, VariationalSparseGP
from .koopman import DynamicModeDecomposition, ExtendedDMD

__all__ = [
    "PINN",
    "BurgersPINN",
    "HeatPINN",
    "WavePINN",
    "HarmonicOscillatorPINN",
    "SymbolicRegressor",
    "NeuralODE",
    "DifferentiableParticleFluid",
    "SpectralConv2d",
    "FourierNeuralOperator2D",
    "SparseGaussianProcessRegression",
    "VariationalSparseGP",
    "DynamicModeDecomposition",
    "ExtendedDMD",
]
