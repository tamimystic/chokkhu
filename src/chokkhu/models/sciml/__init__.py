"""Scientific Machine Learning, PINNs & Symbolic Modeling Package.

Pure NumPy implementations of:
- PINN, BurgersPINN, HeatPINN, WavePINN, HarmonicOscillatorPINN
- SymbolicRegressor: Sparse equation discovery (SINDy)
- NeuralODE: Continuous-depth ordinary differential equation models
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

__all__ = [
    "PINN",
    "BurgersPINN",
    "HeatPINN",
    "WavePINN",
    "HarmonicOscillatorPINN",
    "SymbolicRegressor",
    "NeuralODE",
]
