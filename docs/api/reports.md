# Interactive Standalone SVG & Dashboard Reporting API Reference

The `chokkhu.reports` package generates self-contained, interactive, zero-dependency dark-mode HTML diagnostic dashboards containing pure inline SVG charts without external JavaScript or CSS libraries.

---

## 1. Creating the Dashboard

```python
import chokkhu as ck
from chokkhu.reports import create_diagnostic_dashboard

# 1. Run any Chokkhu pipeline
result = ck.pipeline(
    data="dataset.csv",
    target="label",
    task="classification",
    conformal_interval=0.95
)

# 2. Build standalone dark-mode dashboard
dashboard_file = create_diagnostic_dashboard(
    result,
    save_path="diagnostic_dashboard.html",
    title="Chokkhu Sovereign AI Diagnostic Report"
)
```

---

## 2. Integrated Standalone Diagnostic SVGs

| Diagnostic Chart | SVG Generator | Description |
| :--- | :--- | :--- |
| **Confusion Matrix** | `HTMLReportBuilder._generate_confusion_svg` | Multi-class heatmapped grid with exact counts and overall accuracy metrics. |
| **Residuals vs. Fitted** | `HTMLReportBuilder._generate_residual_svg` | Scatter plot with dashed zero-error reference line and mean residual readout. |
| **Feature Importances** | `HTMLReportBuilder._generate_feature_importance_svg` | Horizontal normalized bar chart sorted by actual model feature importance or TreeSHAP scores. |
| **Conformal Uncertainty Bands** | `HTMLReportBuilder._generate_conformal_svg` | Shaded polygon polygon band displaying $95\%$ empirical prediction coverage over test samples. |

---

## 3. Key Design Properties
- **Zero External Network Requests**: Standalone HTML works 100% offline in air-gapped secure enterprise environments.
- **Pure Vector SVGs**: Crisp rendering at any screen resolution or DPI without pixelation.
- **Mobile Responsive**: Built using CSS CSS Grid with automatic layout wrapping.
