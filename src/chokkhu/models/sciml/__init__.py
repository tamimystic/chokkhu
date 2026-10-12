"""Scientific Machine Learning, PINNs & Symbolic Modeling Package.

Pure NumPy implementations of:
- PINN, BurgersPINN, HeatPINN, WavePINN, HarmonicOscillatorPINN
- SymbolicRegressor: Sparse equation discovery (SINDy)
- NeuralODE: Continuous-depth ordinary differential equation models
- SpectralConv2d, FourierNeuralOperator2D: 2D Fourier Neural Operator for PDEs
- SparseGaussianProcessRegression, VariationalSparseGP: Inducing point sparse GPs
"""

from .fluid import DifferentiableParticleFluid
from .fno import FourierNeuralOperator2D, SpectralConv2d
from .koopman import DynamicModeDecomposition, ExtendedDMD
from .neural_ode import NeuralODE
from .pinn import (
    PINN,
    BurgersPINN,
    HarmonicOscillatorPINN,
    HeatPINN,
    WavePINN,
)
from .sparse_gp import SparseGaussianProcessRegression, VariationalSparseGP
from .symbolic_regression import SymbolicRegressor

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
