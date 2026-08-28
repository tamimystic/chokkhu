# CHOKKHU — End-to-End Research-Grade ML/DL Pipeline
## Complete Implementation Blueprint

> **"Minimalistic Code. Maximum Output. Zero Heavy Dependencies."**
>
> chokkhu (চক্ষু) — The eye that sees through your data.

---

## Vision & Core Philosophy

**chokkhu** is NOT just an EDA library. It is a **complete, end-to-end, research-grade Machine Learning and Deep Learning pipeline framework** built entirely from scratch using only core-level Python libraries.

### Design Principles
1. **One-Line API** — Every task should be achievable in a single function call.
2. **Smart Defaults, Full Control** — Every function works with zero config, but exposes every hyperparameter for advanced users.
3. **Zero Heavy Dependencies** — No scikit-learn, no TensorFlow, no PyTorch. Everything from scratch.
4. **Research-Grade** — Every algorithm implemented with mathematical rigor.
5. **Beautiful Reports** — Every step auto-generates publication-quality HTML reports.

### Allowed Dependencies (STRICT)
| Library | Purpose | Status |
|---------|---------|--------|
| `numpy` | Core numerical computation | Allowed |
| `pandas` | Data manipulation and I/O | Allowed |
| `scipy` | Statistical functions, optimization | Allowed |
| `matplotlib` | Plotting engine | Allowed |
| `seaborn` | Statistical visualizations | Allowed |
| `opencv-python-headless` | Image processing | Allowed |
| `tqdm` | Progress bars | Allowed |
| `scikit-learn` | BANNED | BANNED |
| `tensorflow` / `keras` | BANNED | BANNED |
| `pytorch` | BANNED | BANNED |
| `xgboost` / `lightgbm` | BANNED | BANNED |
| `shap` / `lime` (package) | BANNED | BANNED |

---

## Architecture Overview

```
chokkhu/
├── __init__.py                    # Top-level API: chokkhu.load(), chokkhu.clean(), etc.
├── core/
│   ├── logger.py                  # Unified logging (DONE)
│   ├── visualizer.py              # Plot utilities (DONE)
│   └── config.py                  # Global defaults and configuration
│
├── io/                            # Phase 1: Data Loading
│   ├── loader.py                  # chokkhu.load()
│   └── exporters.py               # chokkhu.save()
│
├── eda/                           # Phase 2: EDA (DONE)
│   ├── tabular/                   # Tabular EDA engine
│   └── image/                     # Image EDA engine
│
├── cleaning/                      # Phase 3: Data Cleaning
│   ├── missing.py                 # Missing value handlers
│   ├── outliers.py                # Outlier detection and removal
│   ├── duplicates.py              # Duplicate detection and removal
│   └── dtype_fixer.py             # Automatic data type correction
│
├── preprocessing/                 # Phase 4: Data Preprocessing
│   ├── scaling.py                 # Scalers (Standard, MinMax, Robust, etc.)
│   ├── encoding.py                # Encoders (Label, OneHot, Target, etc.)
│   └── feature_selection.py       # Feature selectors
│
├── transformation/                # Phase 5: Data Transformation
│   ├── decomposition.py           # PCA, LDA, SVD (from scratch)
│   ├── resampling.py              # SMOTE, Under/Oversampling
│   ├── augmentation.py            # Image augmentation
│   └── feature_engineering.py     # Polynomial, Interaction features
│
├── splitting/                     # Phase 6: Data Splitting
│   └── splitter.py                # Train/Test/Val split, KFold, Stratified
│
├── models/                        # Phase 7 and 8: Modeling
│   ├── ml/                        # Classical ML (from scratch)
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── naive_bayes.py
│   │   ├── gradient_boosting.py
│   │   ├── kmeans.py
│   │   ├── dbscan.py
│   │   └── hierarchical.py
│   │
│   ├── dl/                        # Deep Learning (from scratch, numpy only)
│   │   ├── layers.py              # Dense, Conv2D, MaxPool, Flatten, Dropout, BatchNorm
│   │   ├── activations.py         # ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU
│   │   ├── losses.py              # MSE, CrossEntropy, BinaryCE, Hinge
│   │   ├── optimizers.py          # SGD, Adam, RMSProp, AdaGrad
│   │   ├── initializers.py        # Xavier, He, Random, Zeros
│   │   ├── regularizers.py        # L1, L2, Dropout
│   │   ├── nn.py                  # Sequential model builder
│   │   └── callbacks.py           # EarlyStopping, LRScheduler, ModelCheckpoint
│   │
│   └── registry.py               # Model name to class mapping
│
├── evaluation/                    # Phase 9: Evaluation
│   ├── classification.py          # Accuracy, Precision, Recall, F1, ROC, AUC, CM
│   ├── regression.py              # MSE, RMSE, MAE, R2, Adjusted R2, MAPE
│   ├── clustering.py              # Silhouette, Davies-Bouldin, Calinski-Harabasz
│   └── cross_validation.py        # K-Fold CV, Stratified CV
│
├── explainability/                # Phase 10: Explainable AI
│   ├── feature_importance.py      # Permutation Importance, MDI
│   ├── shap_scratch.py            # SHAP from scratch (KernelSHAP approximation)
│   ├── lime_scratch.py            # LIME from scratch
│   ├── gradcam.py                 # GradCAM / GradCAM++ for DL models
│   ├── pdp.py                     # Partial Dependence Plots
│   └── ice.py                     # Individual Conditional Expectation
│
├── pipeline/                      # Phase 11: Full Pipeline
│   └── auto_pipeline.py           # chokkhu.pipeline() one function does everything
│
└── reports/                       # Phase 12: HTML Reports
    └── html_builder.py            # Auto HTML report generator (DONE, will be extended)
```

---

## Top-Level API Design (chokkhu/__init__.py)

This is how users will interact with chokkhu. Every function is one line.

```python
import chokkhu

# PHASE 1: DATA LOADING
data = chokkhu.load("data.csv")
data = chokkhu.load("data.json", format="json")
images = chokkhu.load("images/", type="image", img_size=(224, 224))

# PHASE 2: EDA (ALREADY DONE)
chokkhu.eda.tabular(data, target="price", save_reports=True)
chokkhu.eda.image("path/to/images/", save_reports=True)

# PHASE 3: DATA CLEANING
cleaned = chokkhu.clean(data, missing="median", outliers="iqr", duplicates=True)

# PHASE 4: PREPROCESSING
processed, processor = chokkhu.preprocess(data, scale="standard", encode="onehot")

# PHASE 5: TRANSFORMATION
transformed = chokkhu.transform(data, pca=2, resample="smote", target="label")

# PHASE 6: DATA SPLITTING
X_train, X_test, y_train, y_test = chokkhu.split(data, target="price", test_size=0.2)

# PHASE 7: MODEL TRAINING (ML)
model = chokkhu.train(
    model="random_forest",
    X_train=X_train, y_train=y_train,
    task="classification",
    n_estimators=100, max_depth=None
)

# PHASE 8: MODEL TRAINING (DL)
model = chokkhu.train(
    model="neural_network",
    X_train=X_train, y_train=y_train,
    layers=[128, 64, 32], activation="relu",
    optimizer="adam", learning_rate=0.001, epochs=100
)

# PHASE 9: EVALUATION
results = chokkhu.evaluate(model, X_test, y_test, save_reports=True)

# PHASE 10: EXPLAINABLE AI
chokkhu.explain(model, X_test, y_test, method="shap", save_reports=True)

# PHASE 11: FULL PIPELINE (One function does EVERYTHING)
chokkhu.pipeline(data="data.csv", target="price", model="random_forest", save_reports=True)
```

---
---

## PHASE 1: DATA LOADING — chokkhu.load()

### Purpose
Universal data loader that auto-detects file format and loads any type of data with a single function call.

### Module: chokkhu/io/loader.py

### API Signature
```python
chokkhu.load(
    path: str,                          # Path to file or directory

    # Format Control
    format: str = "auto",               # "auto", "csv", "json", "parquet", "excel", "tsv", "image"
    type: str = "tabular",              # "tabular" or "image"

    # CSV/Tabular Options
    sep: str = ",",                     # Delimiter for CSV
    header: int = 0,                    # Row number for header
    encoding: str = "utf-8",            # File encoding
    na_values: list = None,             # Additional strings to recognize as NA
    usecols: list = None,               # Specific columns to load
    nrows: int = None,                  # Number of rows to read (for previewing large files)
    dtype: dict = None,                 # Column data types

    # Image Options
    img_size: tuple = None,             # Resize all images to (H, W), e.g. (224, 224)
    color_mode: str = "rgb",            # "rgb", "grayscale", "bgr"
    flatten: bool = False,              # Flatten images to 1D vectors
    normalize: bool = False,            # Normalize pixel values to [0, 1]
    extensions: list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],

    # General
    verbose: bool = True                # Print loading summary
) -> pd.DataFrame | dict
```

### Supported Formats and Auto-Detection Logic
| Extension | Format | Reader |
|-----------|--------|--------|
| .csv | CSV | pandas.read_csv() |
| .tsv | TSV | pandas.read_csv(sep='\t') |
| .json | JSON | pandas.read_json() |
| .parquet | Parquet | pandas.read_parquet() |
| .xlsx / .xls | Excel | pandas.read_excel() |
| .feather | Feather | pandas.read_feather() |
| Directory (folder) | Image Folder | Custom OpenCV loader |

### Image Loading (when type="image")
When path is a directory of class-wise sub-folders (e.g., images/cat/, images/dog/):
- Returns a dictionary: {"X": np.ndarray, "y": np.ndarray, "class_names": list, "file_paths": list}
- Each sub-folder name becomes the class label
- Images auto-resized to img_size
- Optional: normalize to [0,1], convert to grayscale, flatten

### Implementation Notes
- Pure pandas I/O for tabular (no extra dependency)
- cv2.imread() + cv2.resize() for images
- Progress bar via tqdm for large image directories
- Auto-detection works by checking file extension or if path is a directory

---

## PHASE 2: EDA — chokkhu.eda (COMPLETED)

### Status: FULLY IMPLEMENTED

### What Is Already Built:

#### Tabular EDA (chokkhu.eda.tabular())
- **Global Analysis:** Shape, Memory, Duplicates, Data Types, Index Integrity
- **Missing Data:** Missing Density, Nullity Correlation, Zero Inflation, Imputation Shift, Imputation Recommendations
- **Univariate:** Auto type detection (Numerical/Categorical), Descriptive Stats, Distribution Fitting, Skewness, Kurtosis, Outlier Detection (IQR, Z-score)
- **Bivariate:** Numerical-Numerical (Scatter + Pearson), Numerical-Categorical (Box + ANOVA), Categorical-Categorical (Stacked Bar + Chi-squared)
- **Multivariate:** Pearson and Spearman Correlation (Clustermap), VIF, Cramer's V, Mutual Information, PCA 2D Embedding, Mahalanobis Outliers, Dataset Drift (PSI)

#### Image EDA (chokkhu.eda.image())
- **Metadata:** Width, Height, Aspect Ratio, File Size, Total Pixels
- **Color:** RGB Histograms, Brightness, Contrast
- **Texture:** GLCM Contrast, GLCM Homogeneity, Edge Density, Average Images
- **Quality:** Shannon Entropy, Blur Score (Laplacian Variance), SNR
- **Duplicates:** Perceptual Hash (pHash)
- **Objects:** Face Detection (Haar Cascade), Face Count

---

## PHASE 3: DATA CLEANING — chokkhu.clean()

### Purpose
Automatically detect and fix data quality issues: missing values, outliers, duplicates, and incorrect data types.

### Module: chokkhu/cleaning/

### API Signature
```python
chokkhu.clean(
    data: pd.DataFrame,

    # --- Missing Value Handling ---
    missing: str = "median",
    # Options:
    #   "mean"          — Fill with column mean (numerical only)
    #   "median"        — Fill with column median (numerical only)
    #   "mode"          — Fill with most frequent value (works for categorical too)
    #   "knn"           — K-Nearest Neighbors imputation (from scratch)
    #   "interpolate"   — Linear/polynomial interpolation for time-series
    #   "ffill"         — Forward fill (previous value)
    #   "bfill"         — Backward fill (next value)
    #   "constant"      — Fill with a constant value (use fill_value param)
    #   "drop_rows"     — Drop rows with any missing values
    #   "drop_cols"     — Drop columns exceeding missing_threshold
    #   "iterative"     — Iterative imputation (regression-based, from scratch)
    #   "auto"          — Automatic: chokkhu decides best strategy per column
    #   None            — Do nothing

    # --- Missing Value Sub-Parameters ---
    missing_threshold: float = 0.5,     # Drop columns with missing pct above this (for "drop_cols")
    fill_value: any = 0,                # Constant value for missing="constant"
    knn_k: int = 5,                     # K neighbors for missing="knn"
    interpolate_method: str = "linear", # "linear", "polynomial", "spline" for missing="interpolate"
    interpolate_order: int = 2,         # Polynomial/spline order
    iterative_max_iter: int = 10,       # Max iterations for missing="iterative"

    # --- Outlier Detection and Removal ---
    outliers: str = "iqr",
    # Options:
    #   "iqr"               — Interquartile Range method
    #   "zscore"            — Z-Score method
    #   "modified_zscore"   — Modified Z-Score (using MAD)
    #   "percentile"        — Remove values outside [p_low, p_high] percentiles
    #   "winsorize"         — Cap outliers at percentile boundaries instead of removing
    #   "isolation"         — Isolation Forest-like detection (from scratch)
    #   "log_transform"     — Apply log transformation to reduce outlier impact
    #   None                — Do nothing

    # --- Outlier Sub-Parameters ---
    outlier_threshold: float = 1.5,     # IQR multiplier (default 1.5, strict = 3.0)
    zscore_threshold: float = 3.0,      # Z-score threshold
    outlier_columns: list = None,       # Apply only to specific columns (None = all numeric)
    percentile_low: float = 0.01,       # Lower percentile for "percentile" method
    percentile_high: float = 0.99,      # Upper percentile for "percentile" method
    outlier_action: str = "remove",     # "remove" (drop rows), "cap" (winsorize), "nan" (replace with NaN)

    # --- Duplicate Handling ---
    duplicates: bool = True,            # Remove exact duplicate rows
    duplicate_subset: list = None,      # Check duplicates only on these columns
    duplicate_keep: str = "first",      # "first", "last", or False (remove all)

    # --- Data Type Fixing ---
    fix_dtypes: bool = True,            # Auto-detect and fix data types
    category_threshold: int = 20,       # Unique values below this treated as category
    date_formats: list = None,          # Custom date format strings to try

    # --- General ---
    inplace: bool = False,              # Modify original DataFrame or return copy
    verbose: bool = True,               # Print cleaning summary
    save_report: bool = False,          # Generate HTML cleaning report
    report_dir: str = "./chokkhu_reports/"
) -> pd.DataFrame
```

### 3.1 Missing Value: "mean" Imputation
**What it does:** Fill each numerical column's NaN values with that column's arithmetic mean.
```
For each numerical column with NaN:
    mean_val = column.dropna().sum() / column.dropna().count()
    column[isnan] = mean_val
```
**Implementation:** Pure numpy — np.nanmean()
**Scope:** Numerical columns only. Categorical columns are skipped.
**When to use:** When data is roughly normally distributed (symmetric, no extreme skew).

### 3.2 Missing Value: "median" Imputation
**What it does:** Fill each numerical column's NaN values with that column's median value.
```
For each numerical column with NaN:
    sorted_vals = sort(column.dropna())
    n = len(sorted_vals)
    median_val = sorted_vals[n//2] if n is odd
                 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    column[isnan] = median_val
```
**Implementation:** np.nanmedian()
**Scope:** Numerical columns only.
**When to use:** When data is skewed or has outliers (more robust than mean).

### 3.3 Missing Value: "mode" Imputation
**What it does:** Fill each column's NaN values with the most frequently occurring value.
```
For each column with NaN:
    value_counts = count occurrences of each unique value
    mode_val = value with highest count
    column[isnan] = mode_val
```
**Implementation:** pandas.Series.value_counts().index[0]
**Scope:** All columns (numerical + categorical).
**When to use:** Best for categorical columns. Also works for numerical.

### 3.4 Missing Value: "knn" Imputation (FROM SCRATCH)
**What it does:** For each missing value, find the K most similar rows (using available features) and fill with the weighted average of their values.
```
For each row with missing value in column C:
    1. Select all rows where column C is NOT missing
    2. For the incomplete row, compute Euclidean distance to all complete rows
       using only the features that are available (not NaN) in BOTH rows
    3. Select K nearest neighbors (smallest distances)
    4. Imputed value = weighted average of K neighbors' values for column C
       (weights = 1/distance, so closer neighbors have more influence)
```
**Implementation:** Pure NumPy distance computation
**Key Parameter:** knn_k (default: 5)
**Distance Metric:** Euclidean (with NaN-aware pairwise computation)
**Complexity:** O(n_squared x d) — warning will be printed for datasets > 10000 rows
**When to use:** When feature relationships matter and simple mean/median loses too much info.

### 3.5 Missing Value: "interpolate" (Time-Series)
**What it does:** Fill gaps by drawing curves through known data points. Preserves temporal patterns.
```
For each column:
    If method == "linear":
        Draw straight line between known points, fill gaps on the line
    If method == "polynomial":
        Fit polynomial of degree `order` through known points, fill gaps on the curve
    If method == "spline":
        Fit smooth cubic/quadratic spline through known points
```
**Implementation:** scipy.interpolate.interp1d for linear, numpy.polyfit for polynomial
**Key Parameters:** interpolate_method ("linear", "polynomial", "spline"), interpolate_order (2)
**When to use:** Time-series data where the order of rows matters (stock prices, sensor data).

### 3.6 Missing Value: "iterative" Imputation (FROM SCRATCH)
**What it does:** Uses other columns to predict missing values via iterative regression. Each column's missing values are predicted using all other columns as features.
```
1. Initialize all missing values with column median (starting guess)
2. For iteration = 1 to max_iter:
    For each column C with missing values:
        a. Use all OTHER columns as features (X), column C as target (y)
        b. Train a simple linear regression (numpy.linalg.lstsq) using only
           the rows where column C is NOT missing
        c. Predict the missing values of C using the trained regression
        d. Update column C's missing values with new predictions
3. Repeat until convergence (values stop changing) or max_iter reached
```
**Implementation:** Pure NumPy linear algebra (np.linalg.lstsq)
**Key Parameter:** iterative_max_iter (default: 10)
**When to use:** When columns are correlated and you want the most accurate imputation.

### 3.7 Missing Value: "auto" (Smart Strategy)
**What it does:** Chokkhu automatically picks the best imputation strategy for each column based on data characteristics.
```
For each column:
    missing_pct = percentage of missing values
    if missing_pct > missing_threshold (default 50%):
        -> Drop the column entirely (too much missing data)
    elif column is numerical:
        if column is normally distributed (Shapiro-Wilk test p > 0.05):
            -> Use mean imputation
        else:
            -> Use median imputation (more robust for skewed data)
    elif column is categorical:
        -> Use mode imputation
    elif column appears time-ordered (index is datetime or sequential):
        -> Use forward fill
```
**Implementation:** Combines scipy.stats.shapiro with heuristics
**When to use:** When you do not know which method to pick. Let chokkhu decide.

### 3.8 Outlier Detection: "iqr" (Interquartile Range)
**What it does:** Flags values that fall too far below Q1 or above Q3.
```
For each numerical column:
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    lower_bound = Q1 - (outlier_threshold x IQR)
    upper_bound = Q3 + (outlier_threshold x IQR)
    outliers = values < lower_bound OR values > upper_bound
```
**Implementation:** np.percentile()
**Key Parameter:** outlier_threshold (default: 1.5 for standard, 3.0 for extreme outliers only)
**When to use:** General purpose. Works well for most distributions.

### 3.9 Outlier Detection: "zscore"
**What it does:** Flags values whose Z-score exceeds a threshold.
```
For each numerical column:
    mean = column.mean()
    std = column.std()
    z_scores = (values - mean) / std
    outliers = |z_scores| > zscore_threshold
```
**Key Parameter:** zscore_threshold (default: 3.0)
**When to use:** When data is approximately normally distributed.

### 3.10 Outlier Detection: "modified_zscore" (MAD-based)
**What it does:** Uses Median Absolute Deviation instead of standard deviation. Much more robust to outliers themselves affecting the threshold.
```
For each numerical column:
    median = np.median(column)
    MAD = np.median(|values - median|)
    modified_z = 0.6745 x (values - median) / MAD
    outliers = |modified_z| > zscore_threshold
```
**When to use:** When data is heavily skewed or already has many outliers.

### 3.11 Outlier Detection: "isolation" (Scratch Isolation Forest)
**What it does:** Builds random trees that try to isolate each data point. Outliers are easier to isolate (shorter path = more anomalous).
```
Build ensemble of random isolation trees:
    For each tree:
        1. Randomly select a feature
        2. Randomly select a split value between min and max of that feature
        3. Recursively partition data left/right based on the split
        4. Record path length (number of splits to isolate each point)
    Anomaly score = average path length across all trees
    Outliers = points with the SHORTEST average path length
```
**Implementation:** Pure Python/NumPy recursive tree building
**When to use:** When outliers are multi-dimensional and cannot be caught by looking at one column at a time.

### 3.12 Data Type Auto-Fixing
**What it does:** Scans every column and auto-corrects incorrectly stored data types.
```
For each column:
    1. Try converting to numeric (pd.to_numeric with errors='coerce')
       If more than 80% values successfully convert -> treat as numeric (int or float)
    2. Try converting to datetime (try multiple format patterns)
       If more than 80% values match -> treat as datetime
    3. Check if values are boolean-like ("yes"/"no", "true"/"false", "1"/"0")
       -> Convert to bool
    4. If unique values < category_threshold (default 20):
       -> Convert to pd.Categorical (saves memory)
```
**When to use:** Always. Raw data from CSVs often stores numbers as strings.

---

## PHASE 4: DATA PREPROCESSING — chokkhu.preprocess()

### Purpose
Scale numerical features, encode categorical features, and select the most important features — all in one call. Returns both the processed data AND a state object so you can apply the same transformation to test data.

### Module: chokkhu/preprocessing/

### API Signature
```python
result, processor = chokkhu.preprocess(
    data: pd.DataFrame,
    target: str = None,                 # Target column name (excluded from processing)

    # --- Scaling (Numerical Columns) ---
    scale: str = None,
    # Options:
    #   "standard"      — StandardScaler: (x - mean) / std -> mean=0, std=1
    #   "minmax"        — MinMaxScaler: (x - min) / (max - min) -> [0, 1]
    #   "robust"        — RobustScaler: (x - median) / IQR -> resistant to outliers
    #   "maxabs"        — MaxAbsScaler: x / |max| -> [-1, 1]
    #   "log"           — Log transform: log(1 + x) -> reduce skewness
    #   "power"         — Box-Cox or Yeo-Johnson power transform
    #   "l2"            — L2 Normalization: x / ||x||2 -> unit norm
    #   None            — No scaling

    # --- Scaling Sub-Parameters ---
    scale_columns: list = None,         # Apply scaling only to these columns (None = all numeric)
    feature_range: tuple = (0, 1),      # Range for minmax scaling
    power_method: str = "yeo-johnson",  # "box-cox" (positive only) or "yeo-johnson" (any)

    # --- Encoding (Categorical Columns) ---
    encode: str = None,
    # Options:
    #   "label"         — Label Encoding: map each category to integer (0, 1, 2, ...)
    #   "onehot"        — One-Hot Encoding: create binary column per category
    #   "ordinal"       — Ordinal Encoding: map categories with explicit order
    #   "target"        — Target Encoding: replace category with mean of target variable
    #   "frequency"     — Frequency Encoding: replace category with its frequency count
    #   "binary"        — Binary Encoding: encode as binary digits (fewer columns than one-hot)
    #   "hash"          — Hash Encoding: hash categories to fixed number of dimensions
    #   None            — No encoding

    # --- Encoding Sub-Parameters ---
    encode_columns: list = None,        # Apply encoding only to these columns (None = all categorical)
    onehot_drop: str = "first",         # "first" (drop first to avoid multicollinearity) or None
    onehot_max_categories: int = 20,    # If unique values > this, skip one-hot (too many columns)
    ordinal_order: dict = None,         # {"column": ["low", "medium", "high"]} for ordinal
    hash_n_components: int = 8,         # Number of hash dimensions

    # --- Feature Selection ---
    select_features: str = None,
    # Options:
    #   "variance"          — Remove features with variance below threshold
    #   "correlation"       — Remove highly correlated features (keep one from each pair)
    #   "mutual_info"       — Select features with highest mutual information with target
    #   "chi2"              — Chi-squared test for categorical features vs target
    #   "anova"             — ANOVA F-test for numerical features vs categorical target
    #   "recursive"         — Recursive Feature Elimination using a base estimator (from scratch)
    #   None                — No feature selection

    # --- Feature Selection Sub-Parameters ---
    select_k: int = None,               # Keep top K features (None = auto)
    variance_threshold: float = 0.01,   # Minimum variance to keep a feature
    correlation_threshold: float = 0.95,# Drop features correlated above this
    select_columns: list = None,        # Apply selection only from these columns

    # --- General ---
    verbose: bool = True,
    save_report: bool = False,
    report_dir: str = "./chokkhu_reports/"

) -> Tuple[pd.DataFrame, PreprocessorState]
```

### 4.1 StandardScaler (FROM SCRATCH)
**What it does:** Centers data to mean=0 and scales to std=1.
```python
# Fit (learn parameters from training data):
mean = np.mean(X, axis=0)           # Mean of each column
std = np.std(X, axis=0)             # Std deviation of each column

# Transform (apply to any data):
X_scaled = (X - mean) / (std + 1e-8)  # epsilon to avoid division by zero

# Inverse Transform (convert back to original scale):
X_original = X_scaled * std + mean
```
**When to use:** Most ML algorithms. Especially important for SVM, KNN, Neural Networks.

### 4.2 MinMaxScaler (FROM SCRATCH)
**What it does:** Scales all values to a fixed range (default [0, 1]).
```python
# Fit:
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)

# Transform:
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
X_scaled = X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
```
**When to use:** Neural Networks (especially with sigmoid/tanh activation), image pixel values.

### 4.3 RobustScaler (FROM SCRATCH)
**What it does:** Uses median and IQR instead of mean and std. Resistant to outliers.
```python
# Fit:
median = np.median(X, axis=0)
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1

# Transform:
X_scaled = (X - median) / (IQR + 1e-8)
```
**When to use:** When your data has outliers that you do NOT want to remove.

### 4.4 Label Encoding (FROM SCRATCH)
**What it does:** Maps each unique category to a unique integer.
```python
# For column ["cat", "dog", "bird", "cat", "dog"]:
# Result: [0, 1, 2, 0, 1]

unique_values = sorted(column.unique())
mapping = {val: idx for idx, val in enumerate(unique_values)}
encoded = column.map(mapping)
```
**When to use:** Ordinal data or tree-based models that can handle integer categories.

### 4.5 One-Hot Encoding (FROM SCRATCH)
**What it does:** Creates a new binary (0/1) column for each unique category.
```python
# For column "color" with values ["red", "blue", "green"]:
# Result: color_red=[1,0,0], color_blue=[0,1,0], color_green=[0,0,1]

for category in unique_categories:
    new_col_name = f"{column_name}_{category}"
    new_col_values = (column == category).astype(int)
    # Add new column to DataFrame
# Drop original column
# Optionally drop first category column (to avoid dummy variable trap / multicollinearity)
```
**Key Parameter:** onehot_drop="first" (recommended for linear models), onehot_max_categories=20
**When to use:** Nominal categories (no natural order). Linear models, Neural Networks.

### 4.6 Target Encoding (FROM SCRATCH)
**What it does:** Replaces each category with the mean of the target variable for that category.
```python
# For column "city" and target "price":
# If mean price for "Dhaka" = 50000, "Chittagong" = 35000:
# "Dhaka" -> 50000, "Chittagong" -> 35000

for category in unique_categories:
    mask = (column == category)
    target_mean = target[mask].mean()
    column[mask] = target_mean

# Smoothing (to prevent overfitting on rare categories):
# smoothed = (count x category_mean + m x global_mean) / (count + m)
# where m = smoothing parameter (default: number of classes)
```
**When to use:** High-cardinality categories (hundreds of unique values). Tree-based models.

### 4.7 Frequency Encoding (FROM SCRATCH)
**What it does:** Replaces each category with its frequency (count or proportion).
```python
# For column ["cat", "dog", "cat", "bird", "cat"]:
# Counts: cat=3, dog=1, bird=1
# Result: [3, 1, 3, 1, 3] (or normalized: [0.6, 0.2, 0.6, 0.2, 0.6])
freq_map = column.value_counts().to_dict()
encoded = column.map(freq_map)
```
**When to use:** When frequency of a category is informative (e.g., popular vs rare items).

### 4.8 Binary Encoding (FROM SCRATCH)
**What it does:** Converts category integers to binary representation. Uses fewer columns than one-hot.
```python
# For 8 categories: one-hot needs 8 columns, binary needs only 3 (log2(8))
# Category 0 -> [0, 0, 0]
# Category 1 -> [0, 0, 1]
# Category 5 -> [1, 0, 1]
# Category 7 -> [1, 1, 1]

label_encoded = label_encode(column)
n_bits = int(np.ceil(np.log2(len(unique_values) + 1)))
for bit in range(n_bits):
    new_col = (label_encoded >> bit) & 1
```
**When to use:** High-cardinality categories where one-hot creates too many columns.

### 4.9 Feature Selection: Variance Threshold (FROM SCRATCH)
**What it does:** Removes features that have very low variance (nearly constant).
```python
for each column:
    variance = np.var(column)
    if variance < variance_threshold:
        drop column
```
**When to use:** Always as a first pass. Constant or near-constant features add noise.

### 4.10 Feature Selection: Correlation Filter (FROM SCRATCH)
**What it does:** When two features are highly correlated (redundant), drop the one that has lower correlation with the target.
```python
corr_matrix = np.corrcoef(X.T)
for each pair (i, j) where |corr[i][j]| > correlation_threshold:
    corr_with_target_i = |correlation(feature_i, target)|
    corr_with_target_j = |correlation(feature_j, target)|
    drop the feature with LOWER correlation to target
```
**Key Parameter:** correlation_threshold (default: 0.95)
**When to use:** When you suspect multicollinearity.

### 4.11 Feature Selection: Mutual Information (FROM SCRATCH)
**What it does:** Measures how much knowing a feature reduces uncertainty about the target. Works for both linear and nonlinear relationships.
```python
# Already implemented in our multivariate.py — reuse the same logic
# Discretize continuous features into bins
# For each feature X and target Y:
MI(X, Y) = sum_x sum_y P(x,y) x log2( P(x,y) / (P(x) x P(y)) )
# Sort features by MI score, select top K
```
**When to use:** When relationships between features and target may be nonlinear.

### 4.12 Feature Selection: ANOVA F-test (FROM SCRATCH)
**What it does:** Tests if the mean of a numerical feature is significantly different across target classes.
```python
# For each numerical feature:
#   Group feature values by target class
#   Compute between-group variance / within-group variance = F statistic
#   Higher F = more discriminative feature
# scipy.stats.f_oneway() for computation

groups = [feature_values[target == c] for c in classes]
F, p_value = scipy_f_oneway(*groups)
# Select features with smallest p-values (most significant)
```
**When to use:** Numerical features with categorical target (classification tasks).

### PreprocessorState Object
```python
class PreprocessorState:
    """Stores fitted parameters so test/new data can be transformed identically."""
    scaler_params: dict    # {mean, std, min, max, median, IQR — depends on method}
    encoder_maps: dict     # {column_name: {category: encoded_value}}
    selected_features: list # Names of features that passed selection
    
    def transform(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the exact same preprocessing to new/test data."""
        # Uses stored parameters, does NOT refit
    
    def inverse_transform(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Convert processed data back to original scale."""
```
**Why this matters:** You MUST apply the SAME transformation to test data. If you fit a scaler on training data (mean=50, std=10), you must use those exact values for test data, not recompute mean/std on test data. The PreprocessorState ensures this.

---

## PHASE 5: DATA TRANSFORMATION — chokkhu.transform()

### Purpose
Dimensionality reduction, data augmentation, resampling for imbalanced datasets, and feature engineering.

### Module: chokkhu/transformation/

### API Signature
```python
transformed = chokkhu.transform(
    data: pd.DataFrame | dict,          # DataFrame or image dict from chokkhu.load()
    target: str = None,                 # Target column (required for SMOTE, LDA)

    # --- Dimensionality Reduction ---
    pca: int = None,                    # Number of PCA components (None = skip)
    lda: int = None,                    # Number of LDA components (None = skip)
    svd: int = None,                    # Number of SVD components (None = skip)
    tsne: int = None,                   # t-SNE dimensions (2 or 3) (None = skip)

    # --- PCA Sub-Parameters ---
    pca_variance: float = None,         # Keep components explaining this pct variance (e.g., 0.95)
    pca_whiten: bool = False,           # Whiten components (unit variance)

    # --- t-SNE Sub-Parameters ---
    tsne_perplexity: float = 30.0,      # Perplexity (5-50 typical)
    tsne_learning_rate: float = 200.0,
    tsne_n_iter: int = 1000,

    # --- Resampling (Imbalanced Data) ---
    resample: str = None,
    # Options:
    #   "smote"             — SMOTE: Synthetic Minority Oversampling (FROM SCRATCH)
    #   "random_oversample" — Randomly duplicate minority class samples
    #   "random_undersample"— Randomly remove majority class samples
    #   "smote_tomek"       — SMOTE + Tomek Links cleaning
    #   "adasyn"            — Adaptive Synthetic Sampling (FROM SCRATCH)
    #   None                — No resampling

    # --- Resampling Sub-Parameters ---
    resample_ratio: float = 1.0,        # Target ratio (minority/majority). 1.0 = balanced
    smote_k: int = 5,                   # K neighbors for SMOTE
    random_state: int = None,

    # --- Image Augmentation (when data is image dict) ---
    augment: bool = False,
    augment_techniques: list = None,    # Specific techniques (None = all below)
    # Available techniques:
    #   "horizontal_flip"   — Flip left-right
    #   "vertical_flip"     — Flip top-bottom
    #   "rotate"            — Random rotation within range
    #   "brightness"        — Random brightness adjustment
    #   "contrast"          — Random contrast adjustment
    #   "noise"             — Add Gaussian noise
    #   "crop"              — Random crop and resize back
    #   "blur"              — Random Gaussian blur
    #   "cutout"            — Random rectangular black patch (regularization)
    #   "mixup"             — Mix two images with alpha blending
    augment_factor: int = 2,            # How many augmented copies per image
    rotate_range: tuple = (-30, 30),    # Rotation angle range in degrees
    brightness_range: tuple = (0.7, 1.3),
    noise_std: float = 0.05,

    # --- Feature Engineering (Tabular) ---
    polynomial: int = None,             # Generate polynomial features up to degree N
    interaction: bool = False,          # Generate interaction features (A x B)
    log_features: list = None,          # Apply log(1+x) to these columns
    binning: dict = None,              # {"column": n_bins} discretize continuous to bins

    # --- General ---
    verbose: bool = True,
    save_report: bool = False,
    report_dir: str = "./chokkhu_reports/"
) -> pd.DataFrame | dict
```

### 5.1 PCA — Principal Component Analysis (FROM SCRATCH)
**What it does:** Finds the directions (principal components) along which data varies the most. Projects data onto fewer dimensions while preserving maximum variance.

**Mathematical Foundation:**
```
1. Center the data: X_centered = X - mean(X)
2. Covariance matrix: C = (1/n) * X_centered.T @ X_centered
3. Eigen decomposition: C = V * Lambda * V.T
   where Lambda = diagonal matrix of eigenvalues
         V = matrix of eigenvectors (principal component directions)
4. Sort eigenvectors by eigenvalue (largest first)
5. Select top k eigenvectors -> W (projection matrix)
6. Project: X_pca = X_centered @ W
7. Explained variance ratio = eigenvalue_i / sum(all eigenvalues)
```

**Implementation:**
```python
def pca(X, n_components):
    # 1. Center
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # 3. Eigen decomposition (use eigh for symmetric matrices — faster and more stable)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort by eigenvalue descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Select top n_components
    W = eigenvectors[:, :n_components]
    
    # 6. Project
    X_pca = X_centered @ W
    
    # 7. Explained variance
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_pca, explained_variance_ratio, W
```
**When to use:** Reduce high-dimensional data. Visualization (project to 2D/3D). Remove noise.

### 5.2 LDA — Linear Discriminant Analysis (FROM SCRATCH)
**What it does:** Like PCA but supervised. Finds directions that maximize separation BETWEEN classes while minimizing spread WITHIN classes.

**Mathematical Foundation:**
```
1. Compute within-class scatter matrix: S_W = sum_c (X_c - mean_c).T @ (X_c - mean_c)
2. Compute between-class scatter matrix: S_B = sum_c n_c * (mean_c - overall_mean) @ (mean_c - overall_mean).T
3. Solve: S_W^{-1} @ S_B -> eigenvalues, eigenvectors
4. Select top k eigenvectors
5. Project data
```

**Implementation:**
```python
def lda(X, y, n_components):
    classes = np.unique(y)
    overall_mean = np.mean(X, axis=0)
    d = X.shape[1]
    S_W = np.zeros((d, d))  # Within-class scatter
    S_B = np.zeros((d, d))  # Between-class scatter
    
    for c in classes:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        diff = X_c - mean_c
        S_W += diff.T @ diff
        n_c = X_c.shape[0]
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)
    
    A = np.linalg.inv(S_W) @ S_B
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = eigenvalues.argsort()[::-1]
    W = eigenvectors[:, idx[:n_components]].real
    return X @ W, W
```
**When to use:** Classification tasks. Better than PCA when you want class separation.

### 5.3 t-SNE (FROM SCRATCH)
**What it does:** Non-linear dimensionality reduction for VISUALIZATION. Preserves local neighborhoods — similar points stay close in 2D/3D.

**Implementation (exact algorithm, suitable for n < 5000):**
```python
def tsne(X, n_components=2, perplexity=30, n_iter=1000, lr=200):
    n = X.shape[0]
    
    # 1. Compute pairwise Euclidean distances
    dists = compute_pairwise_distances(X)
    
    # 2. Compute joint probabilities P using Gaussian kernels
    #    Binary search for sigma (bandwidth) to match desired perplexity
    P = compute_joint_probabilities(dists, perplexity)
    
    # 3. Initialize low-dimensional embedding randomly
    Y = np.random.randn(n, n_components) * 0.01
    
    # 4. Gradient descent with momentum
    for t in range(n_iter):
        # Compute Q (Student t-distribution in low-dimensional space)
        dists_Y = compute_pairwise_distances(Y)
        Q = (1 + dists_Y) ** (-1)
        np.fill_diagonal(Q, 0)
        Q = Q / Q.sum()
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradient of KL divergence
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[i] * (1 + dists_Y[i]) ** (-1))[:, np.newaxis] * diff, axis=0)
        
        # Update with learning rate
        Y = Y - lr * grad
    
    return Y
```
**Key Parameters:** perplexity (5-50, affects neighborhood size), n_iter (1000+)
**When to use:** ONLY for visualization. Not for preprocessing before training.
**Warning:** Will be slow for large datasets. Chokkhu will print a warning for n > 5000.

### 5.4 SMOTE — Synthetic Minority Oversampling (FROM SCRATCH)
**What it does:** Creates synthetic (fake but realistic) samples for the minority class by interpolating between existing minority samples and their nearest neighbors.

**Algorithm:**
```
For each minority class that needs more samples:
    1. Pick a random minority sample
    2. Find its K nearest neighbors (within the minority class)
    3. Pick one neighbor randomly
    4. Create a synthetic point on the line between the sample and the neighbor:
       synthetic = sample + random(0, 1) x (neighbor - sample)
    5. Repeat until desired ratio is achieved
```

**Implementation:**
```python
def smote(X, y, k=5, ratio=1.0):
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    target_count = int(majority_count * ratio)
    
    X_resampled = list(X)
    y_resampled = list(y)
    
    for c in classes:
        class_count = counts[classes == c][0]
        if class_count >= target_count:
            continue
        
        n_synthetic = target_count - class_count
        X_minority = X[y == c]
        
        for i in range(n_synthetic):
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority[idx]
            
            # Compute distances to all other minority samples
            dists = np.linalg.norm(X_minority - sample, axis=1)
            nearest_indices = np.argsort(dists)[1:k+1]
            
            # Pick a random neighbor
            nn_idx = nearest_indices[np.random.randint(0, min(k, len(nearest_indices)))]
            neighbor = X_minority[nn_idx]
            
            # Interpolate
            lam = np.random.random()
            synthetic = sample + lam * (neighbor - sample)
            
            X_resampled.append(synthetic)
            y_resampled.append(c)
    
    return np.array(X_resampled), np.array(y_resampled)
```
**Key Parameters:** smote_k=5, resample_ratio=1.0
**When to use:** Imbalanced classification (e.g., 95% class A, 5% class B).

### 5.5 Image Augmentation (Using OpenCV Only)
**What it does:** Creates modified copies of training images to increase dataset size and reduce overfitting. All augmentations use cv2 only (no albumentations, no imgaug).

**Available Techniques:**
```python
def horizontal_flip(img):
    return cv2.flip(img, 1)

def vertical_flip(img):
    return cv2.flip(img, 0)

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def adjust_brightness(img, factor):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor):
    mean = np.mean(img)
    return np.clip(mean + factor * (img - mean), 0, 255).astype(np.uint8)

def add_gaussian_noise(img, std):
    noise = np.random.normal(0, std * 255, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def random_crop(img, crop_fraction=0.8):
    h, w = img.shape[:2]
    new_h, new_w = int(h * crop_fraction), int(w * crop_fraction)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    cropped = img[y:y+new_h, x:x+new_w]
    return cv2.resize(cropped, (w, h))

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def cutout(img, size=50):
    h, w = img.shape[:2]
    x = np.random.randint(0, w - size)
    y = np.random.randint(0, h - size)
    img_copy = img.copy()
    img_copy[y:y+size, x:x+size] = 0
    return img_copy

def mixup(img1, img2, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    return np.clip(lam * img1 + (1 - lam) * img2, 0, 255).astype(np.uint8)
```
**When to use:** Image classification tasks. Especially when training data is small.

### 5.6 Polynomial Feature Engineering (FROM SCRATCH)
**What it does:** Creates new features by multiplying existing features together.
```python
# For features [a, b] with degree=2:
# Output: [a, b, a*a, a*b, b*b]

def polynomial_features(X, degree=2):
    from itertools import combinations_with_replacement
    n_features = X.shape[1]
    new_features = [X]
    
    for d in range(2, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            new_col = np.ones(X.shape[0])
            for idx in combo:
                new_col *= X[:, idx]
            new_features.append(new_col.reshape(-1, 1))
    
    return np.hstack(new_features)
```
**Key Parameter:** polynomial degree (2 = quadratic, 3 = cubic)
**When to use:** When relationships between features and target are nonlinear. Use with caution — creates many columns.

---

## PHASE 6: DATA SPLITTING — chokkhu.split()

### Purpose
Split data into train/test/validation sets with optional stratification and cross-validation support.

### Module: chokkhu/splitting/splitter.py

### API Signature
```python
chokkhu.split(
    data: pd.DataFrame,
    target: str,                        # Target column name

    # --- Split Configuration ---
    test_size: float = 0.2,             # Proportion for test set
    val_size: float = None,             # Proportion for validation set (None = no val set)
    stratify: bool = False,             # Stratified split (preserve class proportions)
    shuffle: bool = True,               # Shuffle before splitting
    random_state: int = None,           # Reproducibility seed

    # --- Cross-Validation ---
    cv: int = None,                     # Number of K-Fold splits (None = simple split)
    cv_stratified: bool = True,         # Stratified K-Fold

    # --- Time-Series Split ---
    time_series: bool = False,          # Time-series aware split (no shuffling, temporal order)
    n_splits: int = 5,                  # Number of time-series splits
) -> tuple | generator
```

### 6.1 Simple Train-Test Split (FROM SCRATCH)
```python
def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

### 6.2 Stratified Split (FROM SCRATCH)
```python
def stratified_split(X, y, test_size=0.2, random_state=None):
    classes = np.unique(y)
    train_indices, test_indices = [], []
    for c in classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        split = int(len(class_indices) * (1 - test_size))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```
**When to use:** Classification with imbalanced classes. Ensures each split has same class ratio.

### 6.3 K-Fold Cross-Validation (FROM SCRATCH)
```python
def kfold(X, y, k=5):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
```

### 6.4 Time-Series Split (FROM SCRATCH)
```python
def time_series_split(X, y, n_splits=5):
    n = len(X)
    fold_size = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, n)
        yield X[:train_end], X[val_start:val_end], y[:train_end], y[val_start:val_end]
```
**When to use:** Time-series data where future cannot leak into past.

---

## PHASE 7: MODELING (ML) — chokkhu.train()

### Purpose
Train any classical ML model with a single function call. Every model is implemented FROM SCRATCH using only NumPy/SciPy.

### Module: chokkhu/models/ml/

### API Signature
```python
model = chokkhu.train(
    model: str,                         # Model name (see table below)
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "auto",                 # "classification", "regression", "clustering", "auto"
    random_state: int = None,
    verbose: bool = True,
    save_report: bool = False,
    **kwargs                            # Model-specific hyperparameters (see each model below)
) -> ChokkhuModel
```

### Model Registry
| Model Name (string) | Algorithm | Task | Module |
|---|---|---|---|
| "linear_regression" | Linear Regression (OLS + GD) | Regression | ml/linear_regression.py |
| "ridge" | Ridge Regression (L2) | Regression | ml/linear_regression.py |
| "lasso" | Lasso Regression (L1) | Regression | ml/linear_regression.py |
| "elastic_net" | ElasticNet (L1 + L2) | Regression | ml/linear_regression.py |
| "logistic_regression" | Logistic Regression | Classification | ml/logistic_regression.py |
| "knn" | K-Nearest Neighbors | Both | ml/knn.py |
| "decision_tree" | Decision Tree (CART) | Both | ml/decision_tree.py |
| "random_forest" | Random Forest | Both | ml/random_forest.py |
| "gradient_boosting" | Gradient Boosting (GBM) | Both | ml/gradient_boosting.py |
| "svm" | Support Vector Machine | Both | ml/svm.py |
| "naive_bayes" | Naive Bayes (Gaussian) | Classification | ml/naive_bayes.py |
| "kmeans" | K-Means Clustering | Clustering | ml/kmeans.py |
| "dbscan" | DBSCAN Clustering | Clustering | ml/dbscan.py |
| "hierarchical" | Agglomerative Clustering | Clustering | ml/hierarchical.py |

---

### 7.1 Linear Regression (FROM SCRATCH)

**Math:** y_hat = Xw + b, Loss = (1/2n) * sum((y - y_hat)^2)

**All Hyperparameters:**
```python
chokkhu.train(model="linear_regression", X_train=X, y_train=y,
    method = "ols",                # "ols" (closed-form) or "gradient_descent"
    regularization = None,         # None, "l1" (Lasso), "l2" (Ridge), "elastic_net"
    alpha = 1.0,                   # Regularization strength
    l1_ratio = 0.5,                # ElasticNet mixing (1.0 = pure L1, 0.0 = pure L2)
    learning_rate = 0.01,          # For gradient descent
    max_iter = 1000,               # For gradient descent
    tol = 1e-6,                    # Convergence tolerance
    fit_intercept = True,
    normalize_features = False,
)
```

**OLS (Closed Form):**
```python
def fit_ols(X, y):
    X_b = np.column_stack([np.ones(X.shape[0]), X])
    w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y   # Normal equation
    return w[0], w[1:]  # bias, weights
```

**Gradient Descent:**
```python
def fit_gd(X, y, lr=0.01, max_iter=1000, alpha=0, reg="l2"):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    for _ in range(max_iter):
        y_pred = X @ w + b
        error = y_pred - y
        dw = (1/n) * X.T @ error
        db = (1/n) * np.sum(error)
        if reg == "l2":    dw += (alpha / n) * w
        elif reg == "l1":  dw += (alpha / n) * np.sign(w)
        w -= lr * dw
        b -= lr * db
    return b, w
```

---

### 7.2 Logistic Regression (FROM SCRATCH)

**Math:** P(y=1|x) = sigmoid(Xw + b) = 1 / (1 + exp(-(Xw+b)))

**All Hyperparameters:**
```python
chokkhu.train(model="logistic_regression", X_train=X, y_train=y,
    learning_rate = 0.01,
    max_iter = 1000,
    tol = 1e-6,
    regularization = "l2",         # None, "l1", "l2", "elastic_net"
    alpha = 0.01,
    l1_ratio = 0.5,
    multi_class = "ovr",           # "ovr" (One-vs-Rest) or "multinomial" (Softmax)
    fit_intercept = True,
    class_weight = None,           # {class: weight} for imbalanced data
)
```

**Implementation:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def fit(X, y, lr=0.01, max_iter=1000, alpha=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    for _ in range(max_iter):
        z = X @ w + b
        y_pred = sigmoid(z)
        dw = (1/n) * X.T @ (y_pred - y) + (alpha/n) * w
        db = (1/n) * np.sum(y_pred - y)
        w -= lr * dw
        b -= lr * db
    return b, w
# Multi-class: OVR = train K separate binary classifiers (one per class)
# Predict class with highest probability
```

---

### 7.3 K-Nearest Neighbors (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="knn", X_train=X, y_train=y, task="classification",
    n_neighbors = 5,               # K value
    weights = "uniform",           # "uniform" or "distance" (inverse distance weighting)
    metric = "euclidean",          # "euclidean", "manhattan", "cosine", "minkowski"
    p = 2,                         # Minkowski parameter (2=euclidean, 1=manhattan)
)
```

**Implementation:**
```python
def predict(X_train, y_train, X_test, k=5, weights="uniform", metric="euclidean"):
    predictions = []
    for test_point in X_test:
        if metric == "euclidean":
            dists = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        elif metric == "manhattan":
            dists = np.sum(np.abs(X_train - test_point), axis=1)
        elif metric == "cosine":
            dot = X_train @ test_point
            norms = np.linalg.norm(X_train, axis=1) * np.linalg.norm(test_point)
            dists = 1 - dot / (norms + 1e-8)
        nearest_idx = np.argsort(dists)[:k]
        nearest_labels = y_train[nearest_idx]
        nearest_dists = dists[nearest_idx]
        if weights == "uniform":
            prediction = np.bincount(nearest_labels.astype(int)).argmax()
        elif weights == "distance":
            w = 1 / (nearest_dists + 1e-8)
            weighted_votes = np.bincount(nearest_labels.astype(int), weights=w)
            prediction = weighted_votes.argmax()
        predictions.append(prediction)
    return np.array(predictions)
```

---

### 7.4 Decision Tree — CART (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="decision_tree", X_train=X, y_train=y, task="classification",
    max_depth = None,              # Maximum tree depth (None = unlimited)
    min_samples_split = 2,         # Minimum samples to split a node
    min_samples_leaf = 1,          # Minimum samples in a leaf
    max_features = None,           # "sqrt", "log2", int, float, None (all)
    criterion = "gini",            # "gini", "entropy" (classification) or "mse", "mae" (regression)
    min_impurity_decrease = 0.0,   # Minimum impurity decrease to split
    class_weight = None,           # {class: weight}
)
```

**Core Implementation:**
```python
class DecisionTreeNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None     # Leaf prediction
        self.is_leaf = False

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def best_split(X, y, criterion_func, max_features):
    best_gain = -np.inf
    best_feature, best_threshold = None, None
    parent_impurity = criterion_func(y)
    feature_indices = select_features(X.shape[1], max_features)
    for feature_idx in feature_indices:
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            n = len(y)
            gain = parent_impurity - (
                (left_mask.sum()/n) * criterion_func(y[left_mask]) +
                (right_mask.sum()/n) * criterion_func(y[right_mask])
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    return best_feature, best_threshold, best_gain

def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2):
    node = DecisionTreeNode()
    if (max_depth and depth >= max_depth) or len(y) < min_samples_split or len(np.unique(y)) == 1:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()  # majority vote
        return node
    feature, threshold, gain = best_split(X, y, gini_impurity, None)
    if gain <= 0:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node
    node.feature_idx = feature
    node.threshold = threshold
    left_mask = X[:, feature] <= threshold
    node.left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    node.right = build_tree(X[~left_mask], y[~left_mask], depth+1, max_depth)
    return node
```

---

### 7.5 Random Forest (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="random_forest", X_train=X, y_train=y, task="classification",
    n_estimators = 100,            # Number of trees
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = "sqrt",         # "sqrt", "log2", int, float
    criterion = "gini",
    bootstrap = True,              # Sample with replacement
    max_samples = 1.0,             # Fraction of samples per tree
    oob_score = False,             # Out-of-bag score estimation
    class_weight = None,
    random_state = None,
)
```

**Implementation:**
```python
def fit_random_forest(X, y, n_estimators=100, max_depth=None, max_features="sqrt", bootstrap=True):
    trees = []
    for i in range(n_estimators):
        if bootstrap:
            indices = np.random.choice(len(X), size=len(X), replace=True)
        else:
            indices = np.arange(len(X))
        X_boot, y_boot = X[indices], y[indices]
        tree = build_tree(X_boot, y_boot, max_depth=max_depth, max_features=max_features)
        trees.append(tree)
    return trees

def predict_random_forest(trees, X, task="classification"):
    all_preds = np.array([predict_tree(tree, X) for tree in trees])
    if task == "classification":
        return np.array([np.bincount(all_preds[:, i].astype(int)).argmax() for i in range(len(X))])
    else:
        return np.mean(all_preds, axis=0)
```

---

### 7.6 Gradient Boosting (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="gradient_boosting", X_train=X, y_train=y, task="classification",
    n_estimators = 100,            # Number of boosting rounds
    learning_rate = 0.1,           # Shrinkage (step size)
    max_depth = 3,                 # Depth of each tree (usually shallow)
    min_samples_split = 2,
    min_samples_leaf = 1,
    subsample = 1.0,               # Fraction of samples per tree (stochastic GB)
    max_features = None,
    loss = "deviance",             # "deviance" (log-loss) or "exponential" (AdaBoost)
    n_iter_no_change = None,       # Early stopping
    validation_fraction = 0.1,
    tol = 1e-4,
)
```

**Implementation:**
```python
def fit_gradient_boosting(X, y, n_estimators=100, lr=0.1, max_depth=3):
    # Initialize prediction with mean (regression) or log-odds (classification)
    if task == "regression":
        F = np.full(len(y), np.mean(y))
    else:
        p = np.mean(y)
        F = np.full(len(y), np.log(p / (1 - p + 1e-8)))
    trees = []
    for m in range(n_estimators):
        # Compute pseudo-residuals (negative gradient of loss)
        if task == "regression":
            residuals = y - F
        else:
            probs = 1 / (1 + np.exp(-F))
            residuals = y - probs
        # Fit a shallow tree to residuals
        tree = build_tree(X, residuals, max_depth=max_depth)
        predictions = predict_tree(tree, X)
        F += lr * predictions
        trees.append(tree)
    return trees
```

---

### 7.7 Support Vector Machine (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="svm", X_train=X, y_train=y, task="classification",
    kernel = "rbf",                # "linear", "rbf", "polynomial", "sigmoid"
    C = 1.0,                       # Regularization parameter
    gamma = "scale",               # "scale" (1/(n_features*var)), "auto" (1/n_features), or float
    degree = 3,                    # Polynomial kernel degree
    coef0 = 0.0,                   # Independent term in polynomial/sigmoid kernel
    max_iter = 1000,
    tol = 1e-3,
    multi_class = "ovr",           # "ovr" or "ovo" (one-vs-one)
)
```

**Kernel Functions:**
```python
def linear_kernel(x1, x2):
    return x1 @ x2.T

def rbf_kernel(x1, x2, gamma):
    dist_sq = np.sum(x1**2, axis=1).reshape(-1,1) + np.sum(x2**2, axis=1) - 2 * x1 @ x2.T
    return np.exp(-gamma * dist_sq)

def polynomial_kernel(x1, x2, degree, gamma, coef0):
    return (gamma * x1 @ x2.T + coef0) ** degree
```

**Implementation (Simplified SMO):**
Uses Sequential Minimal Optimization to find optimal Lagrange multipliers (alphas).

---

### 7.8 Naive Bayes — Gaussian (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="naive_bayes", X_train=X, y_train=y,
    variant = "gaussian",          # "gaussian", "multinomial", "bernoulli"
    var_smoothing = 1e-9,          # Stability for variance calculation
    alpha_smoothing = 1.0,         # Laplace smoothing for Multinomial/Bernoulli
    class_prior = None,            # Prior probabilities of classes
)
```

**Implementation:**
```python
def fit_gaussian_nb(X, y):
    classes = np.unique(y)
    params = {}
    for c in classes:
        X_c = X[y == c]
        params[c] = {
            'mean': np.mean(X_c, axis=0),
            'var': np.var(X_c, axis=0) + 1e-9,
            'prior': len(X_c) / len(X)
        }
    return params

def predict_gaussian_nb(params, X):
    predictions = []
    for x in X:
        posteriors = {}
        for c, p in params.items():
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * p['var']) + (x - p['mean'])**2 / p['var']
            )
            posteriors[c] = np.log(p['prior']) + log_likelihood
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)
```

---

### 7.9 K-Means Clustering (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="kmeans", X_train=X, task="clustering",
    n_clusters = 3,
    init = "kmeans++",             # "kmeans++", "random"
    max_iter = 300,
    tol = 1e-4,
    n_init = 10,                   # Number of initializations (best selected)
    random_state = None,
)
```

**K-Means++ Initialization:**
```python
def kmeans_plus_plus_init(X, k):
    centroids = [X[np.random.randint(len(X))]]
    for _ in range(1, k):
        dists = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(X[np.random.choice(len(X), p=probs)])
    return np.array(centroids)
```

**Main Loop:**
```python
def fit_kmeans(X, k, max_iter=300, tol=1e-4):
    centroids = kmeans_plus_plus_init(X, k)
    for _ in range(max_iter):
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return centroids, labels
```

---

### 7.10 DBSCAN Clustering (FROM SCRATCH)

**All Hyperparameters:**
```python
chokkhu.train(model="dbscan", X_train=X, task="clustering",
    eps = 0.5,                     # Maximum distance between neighbors
    min_samples = 5,               # Minimum points to form a dense region
    metric = "euclidean",          # "euclidean", "manhattan", "cosine"
)
```

**Implementation:**
```python
def fit_dbscan(X, eps=0.5, min_samples=5):
    n = len(X)
    labels = np.full(n, -1)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    dist_matrix = pairwise_distances(X)
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dist_matrix[i] <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1   # Noise
            continue
        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = np.where(dist_matrix[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1
    return labels
```

---

## PHASE 8: MODELING (DL) — Deep Learning FROM SCRATCH

### Purpose
Build and train neural networks (Dense, CNN) using ONLY NumPy. No TensorFlow. No PyTorch. Pure math.

### Module: chokkhu/models/dl/

### API — Dense Network
```python
model = chokkhu.train(
    model="neural_network",
    X_train=X, y_train=y, task="classification",
    layers = [128, 64, 32],        # Hidden layer sizes
    activation = "relu",           # "relu", "sigmoid", "tanh", "leaky_relu", "gelu"
    output_activation = "auto",    # Auto: "softmax" for classification, "linear" for regression
    optimizer = "adam",            # "sgd", "adam", "rmsprop", "adagrad"
    learning_rate = 0.001,
    epochs = 100,
    batch_size = 32,
    dropout = 0.0,                 # Dropout rate (0.0 = no dropout)
    l2_lambda = 0.0,               # L2 weight decay
    batch_norm = False,
    lr_schedule = None,            # "step", "exponential", "cosine", None
    early_stopping = False,
    patience = 10,
    weight_init = "he",            # "he", "xavier", "random", "zeros"
    loss = "auto",                 # "mse", "cross_entropy", "binary_cross_entropy"
    validation_split = 0.0,
    X_val = None, y_val = None,
)
```

### API — CNN
```python
model = chokkhu.train(
    model="cnn",
    X_train=X_images, y_train=y, task="classification",
    architecture = [
        {"type": "conv2d", "filters": 32, "kernel_size": 3, "stride": 1, "padding": "same"},
        {"type": "relu"},
        {"type": "maxpool", "pool_size": 2, "stride": 2},
        {"type": "conv2d", "filters": 64, "kernel_size": 3},
        {"type": "relu"},
        {"type": "maxpool", "pool_size": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 128},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
        {"type": "dense", "units": 10},
        {"type": "softmax"}
    ],
    optimizer = "adam",
    learning_rate = 0.001,
    epochs = 50,
    batch_size = 32,
)
```

### 8.1 Activation Functions (dl/activations.py)
```python
def relu(x):           return np.maximum(0, x)
def relu_grad(x):      return (x > 0).astype(float)
def sigmoid(x):        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_grad(x):   s = sigmoid(x); return s * (1 - s)
def tanh(x):           return np.tanh(x)
def tanh_grad(x):      return 1 - np.tanh(x) ** 2
def leaky_relu(x, a=0.01):       return np.where(x > 0, x, a * x)
def leaky_relu_grad(x, a=0.01):  return np.where(x > 0, 1, a)
def gelu(x):           return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
```

### 8.2 Loss Functions (dl/losses.py)
```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_true, y_pred):  # y_true: one-hot
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
def cross_entropy_grad(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
```

### 8.3 Optimizers (dl/optimizers.py)
```python
class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr, self.momentum, self.velocities = lr, momentum, {}
    def update(self, name, param, grad):
        if name not in self.velocities:
            self.velocities[name] = np.zeros_like(param)
        self.velocities[name] = self.momentum * self.velocities[name] - self.lr * grad
        return param + self.velocities[name]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0
    def update(self, name, param, grad):
        self.t += 1
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
        m_hat = self.m[name] / (1 - self.beta1**self.t)
        v_hat = self.v[name] / (1 - self.beta2**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSProp:
    def __init__(self, lr=0.001, decay=0.99, eps=1e-8):
        self.lr, self.decay, self.eps, self.cache = lr, decay, eps, {}
    def update(self, name, param, grad):
        if name not in self.cache:
            self.cache[name] = np.zeros_like(param)
        self.cache[name] = self.decay * self.cache[name] + (1 - self.decay) * grad**2
        return param - self.lr * grad / (np.sqrt(self.cache[name]) + self.eps)
```

### 8.4 Layers (dl/layers.py)
```python
class DenseLayer:
    def __init__(self, input_size, output_size, init="he"):
        if init == "he":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif init == "xavier":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros((1, output_size))
    def forward(self, X):
        self.input = X
        return X @ self.W + self.b
    def backward(self, dout):
        self.dW = self.input.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T

class Conv2DLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.b = np.zeros(out_channels)
        self.stride, self.padding = stride, padding
    def forward(self, X):
        # X shape: (batch, channels, height, width)
        # im2col-based convolution for efficiency
        pass  # Full implementation with forward and backward passes
    def backward(self, dout):
        pass  # Compute gradients for W, b, and input

class MaxPool2DLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size, self.stride = pool_size, stride
    def forward(self, X):
        N, C, H, W = X.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                window = X[:, :, i*self.stride:i*self.stride+self.pool_size,
                                  j*self.stride:j*self.stride+self.pool_size]
                out[:, :, i, j] = np.max(window, axis=(2, 3))
        return out

class FlattenLayer:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.shape)

class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate
    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1-self.rate, size=X.shape) / (1-self.rate)
            return X * self.mask
        return X
    def backward(self, dout):
        return dout * self.mask

class BatchNormLayer:
    def __init__(self, n_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
    def forward(self, X, training=True):
        if training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
            X_norm = (X - mean) / np.sqrt(var + 1e-5)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + 1e-5)
        return self.gamma * X_norm + self.beta
```

### 8.5 Sequential Model Builder (dl/nn.py)
```python
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss_fn = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training) if hasattr(layer, 'training') else layer.forward(X)
        return X
    
    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def fit(self, X, y, epochs=100, batch_size=32, X_val=None, y_val=None):
        n = len(X)
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0
            for i in range(0, n, batch_size):
                X_batch = X[indices[i:i+batch_size]]
                y_batch = y[indices[i:i+batch_size]]
                y_pred = self.forward(X_batch, training=True)
                loss = self.loss_fn.forward(y_batch, y_pred)
                epoch_loss += loss
                loss_grad = self.loss_fn.backward(y_batch, y_pred)
                self.backward(loss_grad)
                for layer in self.layers:
                    if hasattr(layer, 'W'):
                        layer.W = self.optimizer.update(str(id(layer))+'W', layer.W, layer.dW)
                        layer.b = self.optimizer.update(str(id(layer))+'b', layer.b, layer.db)
            self.history["train_loss"].append(epoch_loss / max(1, n // batch_size))
    
    def predict(self, X):
        return self.forward(X, training=False)
```

---

## PHASE 9: EVALUATION — chokkhu.evaluate()

### Purpose
Comprehensive model evaluation with all standard metrics, auto-generated visualizations, and HTML reports. All metrics are implemented FROM SCRATCH.

### Module: chokkhu/evaluation/

### API Signature
```python
results = chokkhu.evaluate(
    model,                              # Trained ChokkhuModel object
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "auto",                 # "classification", "regression", "clustering", "auto"
    average: str = "weighted",          # "micro", "macro", "weighted", "binary" for multi-class
    threshold: float = 0.5,             # Decision threshold for binary classification
    class_names: list = None,           # Human-readable class names
    cv: int = None,                     # K-Fold CV (report mean +/- std)
    cv_stratified: bool = True,
    verbose: bool = True,
    save_report: bool = False,
    report_dir: str = "./chokkhu_reports/",
    plot: bool = True,
) -> dict
```

### 9.1 Classification Metrics (ALL FROM SCRATCH)

| Metric | Formula | Notes |
|--------|---------|-------|
| Accuracy | correct / total | np.mean(y_pred == y_true) |
| Precision | TP / (TP + FP) | Per-class + macro/micro/weighted average |
| Recall | TP / (TP + FN) | Per-class + averages |
| F1-Score | 2 * P * R / (P + R) | Harmonic mean |
| Specificity | TN / (TN + FP) | Per-class |
| Matthews Correlation | (TP*TN - FP*FN) / sqrt(...) | Single scalar, balanced metric |
| Cohen's Kappa | (p0 - pe) / (1 - pe) | Agreement vs chance |
| Log Loss | -sum(y * log(y_hat)) | Requires probability output |
| ROC-AUC | Area under ROC curve | Trapezoidal integration |
| PR-AUC | Area under PR curve | Trapezoidal integration |
| Confusion Matrix | (n_classes x n_classes) | CM[true][pred] += 1 |

**Implementation:**
```python
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm

def precision_recall_f1(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    results = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        results[c] = {"precision": precision, "recall": recall, "f1": f1}
    return results

def roc_auc(y_true, y_scores):
    sorted_indices = np.argsort(-y_scores)
    y_sorted = y_true[sorted_indices]
    tpr_list, fpr_list = [0], [0]
    tp, fp = 0, 0
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1: tp += 1
        else: fp += 1
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)
    auc = np.trapz(tpr_list, fpr_list)
    return auc, fpr_list, tpr_list
```

**Auto-Generated Plots (Classification):**
1. Confusion Matrix Heatmap
2. ROC Curve (per-class with AUC values)
3. Precision-Recall Curve
4. Class Distribution (predicted vs actual)
5. Calibration Curve (reliability diagram)

### 9.2 Regression Metrics (ALL FROM SCRATCH)

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | (1/n) * sum((y - y_hat)^2) | Mean Squared Error |
| RMSE | sqrt(MSE) | Root Mean Squared Error |
| MAE | (1/n) * sum(abs(y - y_hat)) | Mean Absolute Error |
| MAPE | (100/n) * sum(abs((y-y_hat)/y)) | Mean Absolute Percentage Error |
| R2 Score | 1 - SS_res/SS_tot | Coefficient of determination |
| Adjusted R2 | 1 - (1-R2)(n-1)/(n-p-1) | Accounts for number of features |
| Max Error | max(abs(y - y_hat)) | Worst case error |
| Median AE | median(abs(y - y_hat)) | Robust error metric |

**Auto-Generated Plots (Regression):**
1. Actual vs Predicted scatter plot with ideal line
2. Residual Plot (residuals vs predicted values)
3. Residual Distribution (histogram + KDE)
4. Q-Q Plot (check normality of residuals)

### 9.3 Clustering Metrics (ALL FROM SCRATCH)

| Metric | Description | Notes |
|--------|-------------|-------|
| Silhouette Score | How similar a point is to its cluster vs others | (b - a) / max(a, b) |
| Davies-Bouldin | Average similarity ratio with most similar cluster | Lower = better |
| Calinski-Harabasz | Between/within cluster dispersion ratio | Higher = better |
| Inertia | Sum of squared distances to nearest centroid | For elbow method |

**Auto-Generated Plots (Clustering):**
1. Cluster Visualization (2D PCA projection with colors)
2. Silhouette Plot (per-sample coefficients)
3. Elbow Curve (inertia vs number of clusters)

---

## PHASE 10: EXPLAINABLE AI — chokkhu.explain()

### Purpose
Make any model interpretable. Understand WHY the model made each prediction. All methods FROM SCRATCH.

### Module: chokkhu/explainability/

### API Signature
```python
explanation = chokkhu.explain(
    model,                              # Trained ChokkhuModel
    X: np.ndarray,
    y: np.ndarray = None,
    method: str = "feature_importance",
    # Options:
    #   "feature_importance"    — Permutation-based feature importance
    #   "shap"                  — KernelSHAP (model-agnostic Shapley values)
    #   "lime"                  — Local Interpretable Model-agnostic Explanations
    #   "pdp"                   — Partial Dependence Plots
    #   "ice"                   — Individual Conditional Expectation
    #   "gradcam"               — GradCAM (for CNN/DL models only)
    #   "gradcam++"             — GradCAM++ (for CNN/DL models only)
    #   "all"                   — Run all applicable methods

    # Feature Importance Parameters
    n_repeats: int = 10,
    scoring: str = "auto",

    # SHAP Parameters
    shap_n_samples: int = 100,
    shap_n_features: int = None,

    # LIME Parameters
    lime_n_samples: int = 5000,
    lime_n_features: int = 10,
    lime_kernel_width: float = 0.75,

    # PDP/ICE Parameters
    pdp_features: list = None,          # Features to plot (None = top 6)
    pdp_grid_resolution: int = 50,

    # GradCAM Parameters
    gradcam_layer: int = -1,            # Which conv layer (-1 = last)
    gradcam_class: int = None,
    gradcam_images: list = None,

    verbose: bool = True,
    save_report: bool = False,
    report_dir: str = "./chokkhu_reports/",
    plot: bool = True,
) -> dict
```

### 10.1 Permutation Feature Importance (FROM SCRATCH)
**What it does:** Measures how much model performance drops when a feature is randomly shuffled.
```python
def permutation_importance(model, X, y, scoring_fn, n_repeats=10):
    base_score = scoring_fn(y, model.predict(X))
    importances = {}
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, col])
            permuted_score = scoring_fn(y, model.predict(X_permuted))
            scores.append(base_score - permuted_score)
        importances[col] = {"mean": np.mean(scores), "std": np.std(scores)}
    return importances
```
**Plot:** Horizontal bar chart sorted by importance with error bars.

### 10.2 KernelSHAP (FROM SCRATCH)
**What it does:** Approximates Shapley values — the fair contribution of each feature to the prediction.
```python
def kernel_shap(model, x_explain, X_background, n_samples=100):
    M = x_explain.shape[0]
    coalitions = np.random.binomial(1, 0.5, size=(n_samples, M))
    predictions = []
    for z in coalitions:
        bg_sample = X_background[np.random.randint(len(X_background))]
        x_masked = np.where(z, x_explain, bg_sample)
        predictions.append(model.predict(x_masked.reshape(1, -1))[0])
    predictions = np.array(predictions)
    # Compute Shapley kernel weights
    weights = []
    for z in coalitions:
        s = z.sum()
        if s == 0 or s == M:
            weights.append(1e6)
        else:
            w = (M - 1) / (scipy.special.comb(M, s) * s * (M - s))
            weights.append(w)
    weights = np.array(weights)
    W = np.diag(weights)
    shap_values = np.linalg.lstsq(coalitions.T @ W @ coalitions,
                                   coalitions.T @ W @ predictions, rcond=None)[0]
    return shap_values
```
**Plots:** Force Plot (waterfall), Summary Plot (bee swarm), Dependence Plot.

### 10.3 LIME (FROM SCRATCH)
**What it does:** Explains individual predictions by fitting a simple interpretable model locally.
```python
def lime_explain(model, x_explain, n_samples=5000, n_features=10, kernel_width=0.75):
    d = x_explain.shape[0]
    # Generate perturbations
    perturbation_mask = np.random.binomial(1, 0.5, size=(n_samples, d))
    perturbed_data = np.tile(x_explain, (n_samples, 1))
    for i in range(n_samples):
        off_features = np.where(perturbation_mask[i] == 0)[0]
        for f in off_features:
            perturbed_data[i, f] = np.random.normal(x_explain[f], 0.1)
    predictions = model.predict(perturbed_data)
    distances = np.sqrt(np.sum((perturbation_mask - 1) ** 2, axis=1))
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    # Weighted Lasso regression (from scratch) to get sparse local explanation
    # coefficients = feature importance
    return coefficients, intercept
```
**Plot:** Bar chart showing top N feature contributions (positive/negative).

### 10.4 Partial Dependence Plots (FROM SCRATCH)
**What it does:** Shows the marginal effect of a feature on the prediction.
```python
def partial_dependence(model, X, feature_idx, grid_resolution=50):
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), grid_resolution)
    pd_values = []
    for val in feature_values:
        X_modified = X.copy()
        X_modified[:, feature_idx] = val
        avg_prediction = np.mean(model.predict(X_modified))
        pd_values.append(avg_prediction)
    return feature_values, pd_values
```
**Plot:** Line plots showing how each feature affects prediction.

### 10.5 GradCAM / GradCAM++ (FOR DL MODELS)
**What it does:** Highlights which parts of an image the CNN focused on for its prediction.
```python
def gradcam(model, image, target_class, conv_layer_idx=-1):
    # 1. Forward pass: get activation maps from target conv layer
    activations = model.forward_to_layer(image, conv_layer_idx)
    # 2. Backward pass: get gradients of target class score
    gradients = model.backward_from_class(target_class, conv_layer_idx)
    # 3. Global average pooling of gradients = weights
    weights = np.mean(gradients, axis=(1, 2))
    # 4. Weighted sum of activation maps
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
    # 5. ReLU + normalize to [0, 1]
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # 6. Resize to original image size
    cam_resized = cv2.resize(cam, (image.shape[2], image.shape[1]))
    return cam_resized
```
**Plot:** Original image with GradCAM heatmap overlay (jet colormap).

---

## PHASE 11: FULL PIPELINE — chokkhu.pipeline()

### Purpose
THE ULTIMATE ONE-LINER. A single function that runs the entire ML pipeline from raw data to validated predictions with **ZERO DATA LEAKAGE**.

### Zero Data Leakage Architecture (CRITICAL)
In traditional flawed ML pipelines, preprocessing (standardization, encoding, PCA, SMOTE) is mistakenly run across the entire dataset before train/test splitting, causing test-set distribution statistics and synthetic samples to leak into training data.

**Chokkhu Pipeline enforces strict Zero Data Leakage execution order:**
```
Raw Data
   │
   ▼
[1. Load & Sanitation]  ──> Basic type fixes & duplicate removal
   │
   ▼
[2. Split BEFORE Processing] ──> X_train, X_test (and X_val) strictly isolated
   │
   ├────────────────────────────────────────┐
   │ (Train Pipeline)                       │ (Test / Inference Pipeline)
   ▼                                        ▼
[3. Preprocess (Fit & Transform)]     [Transform Only] ──> Uses fitted PreprocessorState
   │                                        │
   ▼                                        ▼
[4. Feature Transforms (PCA/LDA)]     [Transform Only] ──> Uses fitted TransformationState
   │                                        │
   ▼                                        │ (NEVER resample Test/Val!)
[5. Resample (SMOTE/ADASYN)]                │
   │                                        │
   ▼                                        │
[6. Model Training / Auto Selection]        │
   │                                        │
   ▼                                        ▼
[7. Final Evaluation on Untouched Test Set] ◄┘
```

### API Signature
```python
results = chokkhu.pipeline(
    data: str | pd.DataFrame,           # Path to data file OR DataFrame
    target: str,                        # Target column name
    clean: bool | str | dict = "auto",  # Cleaning config (missing, outliers, duplicates)
    preprocess: bool | str | dict = "auto",  # Preprocessing config (scale, encode, select)
    transform: dict = None,             # Transformation config (pca, lda, polynomial)
    resample: str = None,               # Resampling (smote, adasyn, oversample, undersample)
    resample_ratio: float = 1.0,        # Minority class target ratio
    smote_k: int = 5,                   # Number of neighbors for SMOTE
    model: str | list = "auto",         # Model name(s). "auto" = evaluate candidates and pick best
    task: str = "auto",                 # "classification", "regression", "auto"
    test_size: float = 0.2,             # Test set fraction (split BEFORE processing)
    val_size: float = None,             # Optional validation set fraction
    stratify: bool = True,              # Stratified split for classification
    random_state: int = 42,             # Reproducibility seed
    evaluate: bool = True,              # Run evaluation metrics on untouched test set
    save_reports: bool = False,         # Save evaluation charts to disk
    save_dir: str = "chokkhu_reports",  # Reports output folder
    verbose: bool = True,
    **kwargs                            # Model-specific hyperparameters
) -> PipelineResult
```

### PipelineResult Object
```python
class PipelineResult:
    data_raw: pd.DataFrame              # Original data
    data_cleaned: pd.DataFrame          # After cleaning
    splits: dict                        # Train/test/val array splits
    preprocessor_state: PreprocessorState  # Fitted scalers, encoders, selectors
    transformation_state: TransformationState  # Fitted PCA/LDA/Polynomial transformers
    model: ChokkhuModel                 # Best trained model
    model_name: str                     # Name of best model
    task: str                           # "classification" or "regression"
    target_col: str                     # Target column name
    evaluation: dict                    # All evaluation metrics on untouched test set
    cv_scores: dict                     # Candidate model scores during auto-selection
    feature_names: list[str]            # Final feature column names
    
    def predict(self, new_data) -> np.ndarray:
        """Predict on new data (auto-applies fitted preprocessor + transformation pipeline)."""
    def predict_proba(self, new_data) -> np.ndarray:
        """Predict probabilities on new data if classification model."""
    def summary(self) -> str:
        """Print a human-readable summary of the entire pipeline."""
    def save(self, path: str) -> str:
        """Save complete pipeline artifact to disk."""
    @classmethod
    def load(cls, path: str) -> PipelineResult:
        """Load a saved pipeline from disk."""
```

### Example — ONE LINE does EVERYTHING (Leak-Free):
```python
import chokkhu

# Execute complete pipeline
result = chokkhu.pipeline("house_prices.csv", target="price", model="auto", save_reports=True)

# Behind the scenes:
# 1. Loads dataset
# 2. Sanitizes & cleans data
# 3. Splits into train/test BEFORE feature engineering (Zero Data Leakage)
# 4. Fits encoders, scalers, and selectors STRICTLY on train set
# 5. Transforms train and test sets using fitted state
# 6. Tries multiple models (Linear, Ridge, RF, DT, KNN, GB)
# 7. Selects best model via validation score
# 8. Evaluates on untouched test set
# 9. Ready for inference!

print(result.summary())
print(f"Best Model: {result.model_name}")

# Predict on brand-new raw input (automatically scaled & encoded)
predictions = result.predict(new_raw_df)
```

---

## PHASE 12: HTML REPORTS (Extended)

### Current Status: Basic HTML with embedded base64 plots (DONE)
### Extended: Full dashboard-style report

### Report Sections (for each pipeline stage):
1. Data Overview — Shape, types, memory, sample rows
2. EDA Summary — Key statistics, distributions, correlations
3. Cleaning Log — What was cleaned, how many rows/values affected
4. Preprocessing Log — What was scaled, encoded, selected
5. Model Training — Architecture, hyperparameters, training curves (loss/accuracy per epoch)
6. Evaluation Dashboard — All metrics, confusion matrix, ROC, PR curves
7. Explainability — Feature importance, SHAP plots, PDP
8. Executive Summary — Auto-generated one-paragraph summary of key findings

### Tech Stack:
- Pure HTML + CSS (no JavaScript frameworks)
- Base64-encoded images (self-contained single file)
- Responsive design
- Collapsible sections for easy navigation
- Print-friendly CSS

---

## IMPLEMENTATION PRIORITY AND ROADMAP

| Priority | Phase | Estimated Effort | Dependencies |
|----------|-------|------------------|--------------|
| P0 | Phase 1: Data Loading | 1 week | None |
| P0 | Phase 3: Data Cleaning | 2 weeks | Phase 1 |
| P0 | Phase 4: Preprocessing | 2 weeks | Phase 1 |
| P1 | Phase 6: Splitting | 3 days | Phase 1 |
| P1 | Phase 5: Transformation | 2 weeks | Phase 4 |
| P0 | Phase 7: ML Modeling | 4 weeks | Phase 6 |
| P1 | Phase 9: Evaluation | 2 weeks | Phase 7 |
| P2 | Phase 10: Explainability | 3 weeks | Phase 7, 9 |
| P2 | Phase 8: DL Modeling | 4 weeks | Phase 7 |
| P3 | Phase 11: Pipeline | 1 week | All above |
| P3 | Phase 12: Reports (Extended) | 2 weeks | All above |

### Total Estimated Effort: ~6 months (working incrementally)

---

## SUMMARY

chokkhu will be the ONLY Python package where:
- Every algorithm is built FROM SCRATCH (NumPy only)
- Every step requires ONE LINE of code
- Every hyperparameter has a SMART DEFAULT but is FULLY CUSTOMIZABLE
- Every step generates BEAUTIFUL, PUBLICATION-READY reports
- The ENTIRE ML pipeline (Load -> EDA -> Clean -> Process -> Train -> Evaluate -> Explain) can be done with a SINGLE FUNCTION CALL

"Give it your data. Get back everything." — That is chokkhu.
