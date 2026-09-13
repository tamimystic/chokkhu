"""Example 5: Causal Inference, Heterogeneous Treatment Effects & Survival Analysis."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.models.causal import (
    DoublyRobustLearner,
    XLearner,
)
from chokkhu.models.survival import CoxPHRegression, KaplanMeierFitter
from chokkhu.pipeline import dispatch_pipeline


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 5: CAUSAL INFERENCE & SURVIVAL ANALYSIS")
    print("=" * 70)

    # 1. Causal Inference Synthetic DGP
    np.random.seed(42)
    N = 300
    W = np.random.randn(N, 4)
    propensity = 1.0 / (1.0 + np.exp(-(0.5 * W[:, 0] - 0.3 * W[:, 1])))
    T_assign = (np.random.rand(N) < propensity).astype(int)
    true_cate = 2.0 + 1.5 * W[:, 0]
    baseline = 1.0 + 2.0 * W[:, 0] + W[:, 1]
    Y_outcome = baseline + T_assign * true_cate + np.random.randn(N) * 0.5

    df_causal = pd.DataFrame(W, columns=[f"w_{i+1}" for i in range(4)])
    df_causal["treatment"] = T_assign
    df_causal["outcome"] = Y_outcome

    print("[1] Universal Causal Pipeline Dispatcher:")
    causal_res = dispatch_pipeline(
        data=df_causal,
        target="outcome",
        treatment="treatment",
        task="causal_inference",
        verbose=False,
    )
    print(causal_res.summary())

    # 2. Doubly Robust Learner & X-Learner
    print("\n[2] Doubly Robust & Meta-Learner CATE Estimation:")
    dr = DoublyRobustLearner()
    dr.fit(W, T_assign, Y_outcome)
    dr_ate = dr.estimate_ate()
    print(f"  True ATE                     : {np.mean(true_cate):.4f}")
    print(f"  Doubly Robust Estimated ATE  : {dr_ate:.4f}")

    x_learner = XLearner()
    x_learner.fit(W, T_assign, Y_outcome)
    cate_preds = x_learner.predict_cate(W[:5])
    print(f"  X-Learner Sample CATE Preds  : {np.round(cate_preds, 3)}")

    # 3. Survival Analysis: Kaplan-Meier & Cox Proportional Hazards
    print("\n[3] Survival Analysis (Kaplan-Meier & Cox PH):")
    true_durations = np.random.exponential(scale=20.0, size=N)
    censor_times = np.random.exponential(scale=25.0, size=N)
    durations = np.minimum(true_durations, censor_times)
    events = (true_durations <= censor_times).astype(int)

    km = KaplanMeierFitter()
    km.fit(durations, events)
    median_surv = km.median_survival_time_
    print(
        f"  Observed events: {np.sum(events)} / {N} (Censoring rate: {1.0 - np.mean(events):.1%})"
    )
    print(f"  Kaplan-Meier Median Survival : {median_surv:.2f} time units")

    cox = CoxPHRegression()
    cox.fit(W, durations, events)
    hr = cox.hazard_ratios_
    print(f"  Cox PH Estimated Hazard Ratios: {np.round(hr, 3)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
