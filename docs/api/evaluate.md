# Model Evaluation (ck.evaluate)

Evaluate your trained models instantly with comprehensive mathematical metrics and visual heatmaps.

## Parameters Configuration

- **Default usage:** `ck.evaluate(model, X_test, y_test)`
- **Strict Parameters:**
  - `model` (Object): The trained Chokkhu model instance.
  - `X_test` (np.ndarray): Testing features matrix.
  - `y_test` (np.ndarray): True target labels.
- **Dynamic Parameters (Changeable):**
  - `task` (str): Default `"auto"`. Resolves to `"classification"` or `"regression"`.
  - `average` (str): Default `"macro"`. Averaging method for multi-class metrics. Options: `"macro"`, `"weighted"`, `"micro"`.
  - `save_reports` (bool): Default `False`. If True, saves evaluation visualizations to disk.
  - `save_dir` (str): Default `"./chokkhu_outputs/eval_reports"`. Directory for saved images.

## Example

```python
import numpy as np
from chokkhu.evaluation import Metrics

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.3])

metrics = Metrics()
accuracy = metrics.accuracy(y_true, y_pred > 0.5)
print(f"Accuracy: {accuracy}")
```
