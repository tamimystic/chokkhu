from __future__ import annotations

import os
import numpy as np
import pandas as pd

import chokkhu as ck
from chokkhu.reports import HTMLReportBuilder, create_diagnostic_dashboard


def test_dashboard_classification(tmp_path):
    df = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feat_2": [10.0, 20.0, 15.0, 30.0, 25.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )

    result = ck.pipeline(
        data=df,
        target="target",
        task="classification",
        model="logistic_regression",
        test_size=0.3,
        random_state=42,
        verbose=False,
    )

    save_path = str(tmp_path / "dashboard_cls.html")
    out = create_diagnostic_dashboard(result, save_path=save_path)
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        html = f.read()
    assert "Confusion Matrix" in html
    assert "Top Feature Importances" in html
    assert "CLASSIFICATION" in html


def test_dashboard_regression_with_conformal_bands(tmp_path):
    n = 30
    df = pd.DataFrame(
        {
            "x1": np.linspace(0, 10, n),
            "x2": np.random.randn(n),
            "target": np.linspace(0, 10, n) * 2.5 + np.random.randn(n) * 0.2,
        }
    )

    result = ck.pipeline(
        data=df,
        target="target",
        task="regression",
        model="linear_regression",
        conformal_interval=0.95,
        test_size=0.3,
        random_state=42,
        verbose=False,
    )

    save_path = str(tmp_path / "dashboard_reg_conformal.html")
    out = create_diagnostic_dashboard(result, save_path=save_path)
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        html = f.read()
    assert "Residuals vs. Fitted" in html
    assert "REGRESSION" in html


def test_svg_edge_cases():
    # Empty inputs
    assert HTMLReportBuilder._generate_confusion_svg(np.array([]), np.array([])) == ""
    assert HTMLReportBuilder._generate_residual_svg(np.array([]), np.array([])) == ""
    assert HTMLReportBuilder._generate_feature_importance_svg([]) == ""
    assert (
        HTMLReportBuilder._generate_conformal_svg(
            np.array([1.0]), np.array([1.0]), np.array([0.5]), np.array([1.5])
        )
        == ""
    )

    # Constant predictions
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    res_svg = HTMLReportBuilder._generate_residual_svg(y_true, y_pred)
    assert "Residuals vs. Fitted" in res_svg
