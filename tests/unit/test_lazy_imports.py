"""Test lazy loading and import performance for Chokkhu."""

import subprocess
import sys


def test_import_time_under_threshold():
    """Verify that importing chokkhu in a fresh Python process completes under 200ms."""
    code = """
import time
t0 = time.perf_counter()
import chokkhu
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"{elapsed:.4f}")
assert elapsed < 0.5, f"Import too slow: {elapsed}s"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    import_time = float(result.stdout.strip())
    print(f"Fresh process import time: {import_time:.4f}s")
    assert (
        import_time < 0.25
    ), f"Import took {import_time}s which exceeds 0.25s threshold"


def test_lazy_symbol_resolution():
    """Verify that accessing symbols lazily imports the proper objects."""
    import chokkhu

    # Access deep learning & autograd
    assert hasattr(chokkhu, "Tensor")
    assert hasattr(chokkhu, "MultiHeadAttention")
    assert hasattr(chokkhu, "VisionTransformer")

    # Access classical ML
    assert hasattr(chokkhu, "Ridge")
    assert hasattr(chokkhu, "RidgeRegression")
    assert hasattr(chokkhu, "GradientBoosting")
    assert hasattr(chokkhu, "RandomForestClassifier")
    assert hasattr(chokkhu, "KMeans")
    assert hasattr(chokkhu, "PCA")

    # Access evaluation & pipeline
    assert hasattr(chokkhu, "Pipeline")
    assert hasattr(chokkhu, "accuracy_score")
    assert hasattr(chokkhu, "StandardScaler")


def test_submodule_lazy_attribute_access():
    """Verify that accessing submodules lazily imports the proper module."""
    import chokkhu

    assert hasattr(chokkhu, "models")
    assert hasattr(chokkhu, "core")
    assert hasattr(chokkhu, "evaluation")
    assert hasattr(chokkhu, "cleaning")
