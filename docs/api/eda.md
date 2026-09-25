# Exploratory Data Analysis (ck.eda)

Chokkhu provides powerful, automated EDA engines for both tabular and image datasets.

## Tabular EDA Parameters

- **Default usage:** `ck.eda.tabular(df, target_col="price")`
- **Strict Parameters:**
  - `dataset_path` (str or pd.DataFrame): The path to the dataset or the DataFrame object itself.
- **Dynamic Parameters (Changeable):**
  - `target_col` (str): Default `None`. If provided, bivariate analysis will compute IV, WoE, and T-Tests.
  - `save_reports` (bool): Default `False`. If True, saves all figures and generates an HTML report.
  - `save_dir` (str): Default `"./chokkhu_outputs/tabular_reports"`. Directory to save the generated reports.

## Image EDA Parameters

- **Default usage:** `ck.eda.image("images_folder/")`
- **Strict Parameters:**
  - `dataset_path` (str): The path to the root directory containing class subdirectories of images.
- **Dynamic Parameters (Changeable):**
  - `save_reports` (bool): Default `False`. If True, saves all metric figures and generates an HTML report.
  - `save_dir` (str): Default `"./chokkhu_outputs/image_reports"`. Directory to save the generated reports.

## Example

```python
import numpy as np
from chokkhu.eda import univariate

data = np.random.randn(1000)
stats = univariate.summary_statistics(data)
print(stats)
```
