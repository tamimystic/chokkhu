# Model Training (ck.train)

The `ck.train()` function is the heart of Chokkhu's machine learning capabilities. Instead of importing dozens of different classes, you simply pass the name of the model you want to train. 

Every model has **Strict Parameters** (things that must be provided and cannot be changed, like the training data) and **Dynamic Parameters** (hyperparameters you can tweak via `**kwargs`).

---

## Supervised Learning Models

### 1. Linear Regression
Used for continuous target variables.
- **Default usage:** `ck.train("linear_regression", X_train, y_train)`
- **Strict Parameters:** 
  - `model="linear_regression"`
  - `X_train`, `y_train` (Must be numerical matrices/arrays)
- **Dynamic Parameters (Changeable):**
  - `method` (str): Default `"ols"`. Options: `"ols"` (Ordinary Least Squares), `"gd"` (Gradient Descent).
  - `regularization` (str): Default `None`. Options: `"ridge"`, `"lasso"`, `"elasticnet"`.
  - `alpha` (float): Default `1.0`. Regularization penalty.
  - `learning_rate` (float): Default `0.01`. Used if method is gradient descent.
  - `epochs` (int): Default `1000`. Number of iterations.

### 2. Logistic Regression
Used for binary classification tasks.
- **Default usage:** `ck.train("logistic_regression", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train` (Target must be 0 or 1).
- **Dynamic Parameters (Changeable):**
  - `learning_rate` (float): Default `0.01`.
  - `epochs` (int): Default `1000`.
  - `regularization` (str): Default `None`. Options: `"l1"`, `"l2"`.
  - `alpha` (float): Default `1.0`. Regularization penalty.

### 3. K-Nearest Neighbors (KNN)
Used for both classification and regression based on proximity.
- **Default usage:** `ck.train("knn", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train`.
- **Dynamic Parameters (Changeable):**
  - `n_neighbors` (int): Default `5`. The number of neighbors to vote.
  - `task` (str): Default `"classification"`. Options: `"classification"`, `"regression"`.

### 4. Naive Bayes (Gaussian)
A probabilistic classifier based on Bayes' theorem.
- **Default usage:** `ck.train("naive_bayes", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train`.
- **Dynamic Parameters (Changeable):**
  - *(None. Gaussian Naive Bayes calculates priors and variances automatically from the data.)*

### 5. Support Vector Machine (SVM)
Used for binary classification by finding the optimal hyperplane.
- **Default usage:** `ck.train("svm", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train` (Target must be binary).
- **Dynamic Parameters (Changeable):**
  - `learning_rate` (float): Default `0.001`.
  - `lambda_param` (float): Default `0.01`. Regularization parameter.
  - `epochs` (int): Default `1000`.

### 6. Decision Tree
Tree-based model for classification.
- **Default usage:** `ck.train("decision_tree", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train`.
- **Dynamic Parameters (Changeable):**
  - `max_depth` (int): Default `None` (grows until leaves are pure). Maximum depth of the tree.
  - `min_samples_split` (int): Default `2`. Minimum samples required to split a node.
  - `criterion` (str): Default `"gini"`. Options: `"gini"`, `"entropy"`.

### 7. Random Forest
An ensemble of decision trees to prevent overfitting.
- **Default usage:** `ck.train("random_forest", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train`.
- **Dynamic Parameters (Changeable):**
  - `n_estimators` (int): Default `100`. Number of trees in the forest.
  - `max_depth` (int): Default `None`.
  - `min_samples_split` (int): Default `2`.
  - `max_features` (str): Default `"sqrt"`. Number of features to consider per split.

### 8. Gradient Boosting Machine (GBM)
Sequential ensemble of trees to correct previous errors.
- **Default usage:** `ck.train("gradient_boosting", X_train, y_train)`
- **Strict Parameters:** 
  - `X_train`, `y_train`.
- **Dynamic Parameters (Changeable):**
  - `n_estimators` (int): Default `100`.
  - `learning_rate` (float): Default `0.1`.
  - `max_depth` (int): Default `3`.

---

## Unsupervised Learning Models

### 1. K-Means Clustering
- **Default usage:** `ck.train("kmeans", X_train)`
- **Strict Parameters:** 
  - `X_train` (Unlabeled data).
- **Dynamic Parameters (Changeable):**
  - `n_clusters` (int): Default `8`. Number of clusters to form.
  - `max_iter` (int): Default `300`. Maximum number of iterations.

### 2. DBSCAN
Density-based spatial clustering.
- **Default usage:** `ck.train("dbscan", X_train)`
- **Strict Parameters:** 
  - `X_train`.
- **Dynamic Parameters (Changeable):**
  - `eps` (float): Default `0.5`. The maximum distance between two samples.
  - `min_samples` (int): Default `5`. Minimum samples in a neighborhood.

### 3. Hierarchical Clustering
- **Default usage:** `ck.train("hierarchical", X_train)`
- **Strict Parameters:** 
  - `X_train`.
- **Dynamic Parameters (Changeable):**
  - `n_clusters` (int): Default `2`.
  - `linkage` (str): Default `"ward"`. Options: `"ward"`, `"complete"`, `"average"`.

---

## Reinforcement Learning

### 1. Q-Learning
- **Default usage:** `ck.train("q_learning", env=my_env)`
- **Strict Parameters:** 
  - `env` (An environment object with reset() and step() functions).
- **Dynamic Parameters (Changeable):**
  - `episodes` (int): Default `1000`.
  - `learning_rate` (float): Default `0.1`.
  - `discount_factor` (float): Default `0.99`.
  - `epsilon` (float): Default `1.0`. Exploration rate.
  - `epsilon_decay` (float): Default `0.995`.

## Example

```python
import numpy as np
from chokkhu.models.engine import Trainer
from chokkhu.models.ml import NeuralNetwork

model = NeuralNetwork(layers=[10, 5, 1])
trainer = Trainer(model, optimizer="sgd", lr=0.01)

X = np.random.rand(100, 10)
y = np.random.rand(100, 1)
trainer.fit(X, y, epochs=10)
print("Training completed.")
```
