"""Example 4: Time Series Forecasting, PatchTST, Kalman & Particle Filtering."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.models.timeseries import (
    ExtendedKalmanFilter,
    ParticleFilter,
    PatchTST,
)
from chokkhu.pipeline import dispatch_pipeline


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 4: TIME SERIES FORECASTING & STATE FILTERING")
    print("=" * 70)

    # 1. Generate non-linear trend + seasonal signal
    np.random.seed(42)
    T_steps = 150
    t = np.linspace(0, 15, T_steps)
    trend = 0.05 * t
    seasonality = np.sin(2 * np.pi * t / 3.0) + 0.5 * np.cos(2 * np.pi * t / 1.5)
    noise = np.random.randn(T_steps) * 0.15
    y_series = trend + seasonality + noise

    df_ts = pd.DataFrame({"timestamp": t, "energy_demand": y_series})

    # 2. Universal Time Series Pipeline Dispatcher
    print("[1] Running End-to-End Time Series Pipeline Dispatcher:")
    ts_res = dispatch_pipeline(
        data=df_ts,
        target="energy_demand",
        task="timeseries_forecast",
        lags=6,
        test_size=0.2,
        verbose=False,
    )
    print(ts_res.summary())
    forecast_vals = ts_res.forecast(steps=8)
    print(f"  Next 8-step out-of-sample forecast: {np.round(forecast_vals, 3)}")

    # 3. PatchTST Transformer Architecture
    print("\n[2] PatchTST Patch-Based Transformer Forward Pass:")
    patch_tst = PatchTST(
        input_length=24, horizon=6, patch_len=8, stride=4, embed_dim=32
    )
    sample_seq = y_series[:24].reshape(1, -1)
    patch_pred = patch_tst.forward(sample_seq)
    print(f"  PatchTST 6-step horizon prediction shape: {patch_pred.shape}")

    # 4. Extended Kalman Filter (EKF) Non-Linear Tracking
    print("\n[3] Extended Kalman Filter State Estimation:")

    def f_nonlin(x, u=None):
        return np.array([x[0] + 0.1 * x[1], x[1] - 0.01 * np.sin(x[0])])

    def h_nonlin(x):
        return np.array([x[0]])

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1, f=f_nonlin, h=h_nonlin)
    ekf.x = np.array([0.0, 1.0])
    ekf.predict()
    ekf.update(z=np.array([0.15]))
    print(f"  EKF updated state estimate: {np.round(ekf.x, 4)}")

    # 5. Non-Linear Particle Filter
    print("\n[4] Sequential Monte Carlo Particle Filter:")
    pf = ParticleFilter(dim_x=2, dim_z=1, n_particles=200, f=f_nonlin, h=h_nonlin)
    pf.predict()
    mean_est, _ = pf.update(z=np.array([0.15]))
    print(f"  Particle Filter estimated mean state: {np.round(mean_est, 4)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
