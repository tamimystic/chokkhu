"""Example 8: Zero-Heavy-Dependency Architecture Benchmarks & Stress Tests."""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.evaluation.metrics import r2_score
from chokkhu.core.tensor import Tensor
from chokkhu.models.ml import RidgeRegression, GradientBoosting


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 8: SOVEREIGN ZERO-DEPENDENCY BENCHMARKS")
    print("=" * 70)

    print("Benchmarking pure NumPy sovereign operators at scale...")

    # 1. Autograd Tensor Graph Throughput
    print("\n[1] Autograd Reverse-Mode Backprop Benchmark (100k elements):")
    N = 100_000
    x_np = np.random.randn(N)
    w_np = np.random.randn(N)

    t0 = time.perf_counter()
    x = Tensor(x_np, requires_grad=True)
    w = Tensor(w_np, requires_grad=True)
    y = (x * w + (x**2)).sum()
    y.backward()
    elapsed_autograd = (time.perf_counter() - t0) * 1000.0

    print(f"  Vector size         : {N:,} elements")
    print(f"  Forward + Backward  : {elapsed_autograd:.2f} ms")
    print(
        f"  Autograd throughput : {N / (elapsed_autograd / 1000.0) / 1e6:.2f} M elements/sec"
    )

    # 2. Ridge Analytical Linear Solver
    print("\n[2] Analytical OLS / Ridge Solver Benchmark (5,000 samples, 50 features):")
    X_big = np.random.randn(5000, 50)
    y_big = X_big @ np.random.randn(50) + np.random.randn(5000) * 0.1

    t0 = time.perf_counter()
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_big, y_big)
    elapsed_ridge = (time.perf_counter() - t0) * 1000.0
    r2 = r2_score(y_big, ridge.predict(X_big))
    print(f"  Fit time (Cholesky / SVD Closed Form): {elapsed_ridge:.2f} ms")
    print(f"  R2 score on training set             : {r2:.4f}")

    # 3. Sovereign GBDT Fast Split Finding
    print(
        "\n[3] GBDT Tree Construction Benchmark (1,000 samples, 10 features, 5 trees):"
    )
    X_cls = np.random.randn(1000, 10)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)

    t0 = time.perf_counter()
    gb = GradientBoosting(n_estimators=5, max_depth=3)
    gb.fit(X_cls, y_cls)
    elapsed_gb = (time.perf_counter() - t0) * 1000.0
    print(f"  GBDT ensemble training time: {elapsed_gb:.2f} ms")

    print("\n" + "=" * 70)
    print("  ALL BENCHMARKS COMPLETED WITH STRICT ZERO EXTERNAL DEPENDENCIES")
    print("=" * 70)


if __name__ == "__main__":
    main()
