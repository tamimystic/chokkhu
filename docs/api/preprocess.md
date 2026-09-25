# Data Preprocessing (ck.preprocess)

The preprocessing module prepares your cleaned data for machine learning algorithms by scaling numerical features and encoding categorical features.

## Default Behavior
If you run `ck.preprocess(df, target="price")`, the default behavior is:
- The column "price" is excluded from scaling/encoding and saved as the target.
- All numerical columns are scaled using `"standard"` scaling (mean=0, variance=1).
- All categorical columns are encoded using `"onehot"` encoding.
- The function returns the new DataFrame and a `state` object.

---

## Detailed Parameters Configuration

### Target Handling (Strict)
- **`target`** (str): Default `None`. You must pass the name of your target column (e.g., the value you want to predict). The pipeline will ignore this column so it doesn't get incorrectly scaled.

### Scaling Parameters (Dynamic)
- **`scale`** (str): Default `"standard"`.
  - *Options:*
    - `"standard"`: Removes mean and scales to unit variance (Z-score normalization).
    - `"minmax"`: Scales features to lie between 0 and 1.
    - `"robust"`: Uses median and quantiles, making it robust to extreme outliers.
    - `"maxabs"`: Scales by dividing through the largest maximum value.
    - `None`: Disables numerical scaling.

### Encoding Parameters (Dynamic)
- **`encode`** (str): Default `"onehot"`.
  - *Options:*
    - `"onehot"`: Creates binary columns (0 or 1) for each category. Best for non-ordinal data.
    - `"label"`: Converts categories to integers (0, 1, 2...). Best for ordinal data or tree-based models.
    - `"frequency"`: Replaces categories with their frequency percentage in the dataset.
    - `None`: Disables categorical encoding.

### Feature Selection Parameters (Dynamic)
- **`select_features`** (str): Default `None`.
  - *Options:*
    - `"correlation"`: Drops features that are highly correlated with each other to reduce multicollinearity.
    - `"mutual_info"`: Selects features that have the highest mutual information score with the target.
- **`top_k_features`** (int): Default `10`. Only active if `select_features` is used. Specifies how many top features to keep.

## The State Object
The function returns `df_processed, state`. 
The `state` dictionary contains the trained scalers and encoders. You must save this state if you want to preprocess new, unseen testing data later using the exact same scaling parameters!

## Example

```python
import numpy as np
from chokkhu.preprocessing.scaling import StandardScaler

X = np.random.rand(50, 3) * 10
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mean:", np.mean(X_scaled, axis=0))
print("Std:", np.std(X_scaled, axis=0))
```
