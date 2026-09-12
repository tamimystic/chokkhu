"""Unit tests for standalone Dark-Mode HTML diagnostic report builder."""

import os
import numpy as np
import pandas as pd

from chokkhu.pipeline import dispatch_pipeline
from chokkhu.reports import HTMLReportBuilder, create_diagnostic_dashboard


def test_html_diagnostic_dashboard_generation(tmp_path):
    """Test generating standalone dark-mode HTML diagnostic dashboard."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feat_a": np.random.randn(80),
            "feat_b": np.random.randn(80),
            "target": np.random.choice([0, 1], size=80),
        }
    )

    res = dispatch_pipeline(
        data=df,
        target="target",
        task="classification",
        verbose=False,
    )

    out_file = str(tmp_path / "dashboard.html")
    report_path = create_diagnostic_dashboard(res, save_path=out_file)

    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "<!DOCTYPE html>" in html
    assert "Chokkhu AI Universal Diagnostic Dashboard" in html
    assert "Confusion Matrix" in html or "kpi-card" in html
    assert "CLASSIFICATION" in html


def test_html_report_builder_basic(tmp_path):
    """Test HTMLReportBuilder.build on a directory."""
    save_dir = str(tmp_path / "reports")
    os.makedirs(save_dir, exist_ok=True)

    out_file = HTMLReportBuilder.build(save_dir, title="Test Report")
    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        html = f.read()
    assert "Test Report" in html
