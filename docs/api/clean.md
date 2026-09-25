# Data Cleaning (ck.clean)

The data cleaning pipeline in Chokkhu handles missing values, outliers, and duplicates simultaneously.

## Default Behavior
If you run `ck.clean(df)` without any parameters, the default behavior is:
- Missing values are filled using the `"median"` value of the column.
- Columns with more than 50% missing data are dropped completely.
- Outliers are detected using the `"iqr"` (Interquartile Range) method and are removed.
- Duplicate rows are detected and removed.
- Data types are automatically optimized (e.g., float64 to float32).

---

## Detailed Parameters Configuration

You can customize everything. Here are the detailed parameters:

### Missing Value Parameters
- **`missing`** (str): Default `"median"`. 
  - *Dynamic Options:* 
    - `"mean"`: Replaces with average.
    - `"mode"`: Replaces with most frequent value.
    - `"knn"`: Uses K-Nearest Neighbors to predict missing values based on similar rows.
    - `"iterative"`: Uses other features to predict missing values in a loop (MICE).
    - `None`: Skips missing value treatment.
- **`missing_threshold`** (float): Default `0.5`. Any column with a missing ratio above this is dropped.
- **`fill_value`** (Any): Default `0`. Used if you select a constant fill strategy.
- **`knn_k`** (int): Default `5`. Only active if `missing="knn"`. Number of neighbors to use.
- **`iterative_max_iter`** (int): Default `10`. Only active if `missing="iterative"`. Number of imputation rounds.

### Outlier Parameters
- **`outliers`** (str): Default `"iqr"`.
  - *Dynamic Options:*
    - `"iqr"`: Values outside 1.5 * IQR are flagged.
    - `"zscore"`: Values with Z-score > threshold are flagged.
    - `"isolation_forest"`: Anomaly detection using a tree-based machine learning model.
    - `None`: Skips outlier treatment.
- **`outlier_action`** (str): Default `"remove"`. 
  - *Dynamic Options:* 
    - `"remove"`: Deletes the outlier rows entirely.
    - `"clip"`: Replaces outlier values with the upper/lower threshold boundaries.
- **`outlier_threshold`** (float): Default `1.5`. Multiplier used for IQR.
- **`zscore_threshold`** (float): Default `3.0`. Standard deviation limit used for Z-score.

### Duplicate Parameters
- **`duplicates`** (bool): Default `True`. If True, removes identical rows.
- **`duplicate_keep`** (str): Default `"first"`. Options: `"first"`, `"last"`, `False` (drops all duplicates).

### Data Type Parameters
- **`fix_data_types`** (bool): Default `True`. Automatically downcasts numerical columns to save memory and converts objects to categorical types.

## Example

```python
import numpy as np
from chokkhu.cleaning.missing import handle_missing

data = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
cleaned_data = handle_missing(data, strategy="mean")
print(cleaned_data)
```
