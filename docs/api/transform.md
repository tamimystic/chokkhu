# Transformation & Augmentation (ck.transform)

Handles advanced dataset transformations including dimensionality reduction (PCA, t-SNE), class imbalance resampling (SMOTE), polynomial feature engineering, and image augmentations.

## Tabular Transformation Parameters

- **Default usage:** `ck.transform(df, target="price", pca=10)`
- **Strict Parameters:**
  - `data` (pd.DataFrame): The input DataFrame.
- **Dynamic Parameters (Changeable):**
  - `target` (str): Default `None`. Required if you are doing resampling (like SMOTE).
  - `pca` (int): Default `None`. Retains the specified number of principal components.
  - `tsne` (int): Default `None`. Retains the specified number of components (usually 2 or 3).
  - `resample` (str): Default `None`. Handles class imbalance. Options: `"smote"`, `"adasyn"`, `"over"`, `"under"`.
  - `polynomial` (int): Default `None`. Generates polynomial features up to the specified degree.

## Image Augmentation Parameters

- **Default usage:** `ck.transform(images_dict, augment=True, augment_factor=2)`
- **Strict Parameters:**
  - `data` (dict): Must contain `"images"` and `"labels"`.
- **Dynamic Parameters (Changeable):**
  - `augment` (bool): Default `False`. Must be True to enable augmentation.
  - `augment_techniques` (list): Default `[]` (randomly applies all). Options: `"horizontal_flip"`, `"rotate"`, `"brightness"`, `"noise"`, `"crop"`, `"blur"`, `"cutout"`.
  - `augment_factor` (int): Default `1`. The multiplier for the dataset size.

## Example

```python
import numpy as np
from chokkhu.transformation.pca import PCA

X = np.random.rand(100, 10)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("Reduced shape:", X_reduced.shape)
```
