from __future__ import annotations

from .dbscan import DBSCAN
from .decision_tree import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from .gradient_boosting import (
    GradientBoosting,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .hierarchical import HierarchicalClustering
from .kmeans import KMeans
from .knn import KNN
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .random_forest import (
    RandomForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .svm import SVM
from .neural_network import (
    NeuralNetwork,
    MLPClassifier,
    MLPRegressor,
)

from .adaboost import AdaBoostClassifier, AdaBoostRegressor
from .extra_trees import ExtraTreesClassifier, ExtraTreesRegressor
from .kmedoids import KMedoids
from .optics import OPTICS
from .isolation_forest import IsolationForest
from .hist_gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from .svm_kernels import SVC, SVR, OneClassSVM, LinearSVC
from .nearest_neighbors import (
    NearestCentroid,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from .glm import (
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    HuberRegressor,
    BayesianRidge,
    ARDRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
)
from .naive_bayes_extended import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    LDAClassifier,
    QDAClassifier,
)
from .clustering_advanced import (
    MiniBatchKMeans,
    GaussianMixture,
    SpectralClustering,
)
from .knn_spatial import (
    KNNClassifier,
    KNNRegressor,
)

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNN",
    "NaiveBayes",
    "KMeans",
    "SVM",
    "DecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForest",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoosting",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "DBSCAN",
    "HierarchicalClustering",
    "NeuralNetwork",
    "MLPClassifier",
    "MLPRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "KMedoids",
    "OPTICS",
    "IsolationForest",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "SVC",
    "SVR",
    "OneClassSVM",
    "LinearSVC",
    "NearestCentroid",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "HuberRegressor",
    "BayesianRidge",
    "ARDRegression",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "ComplementNB",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "LDAClassifier",
    "QDAClassifier",
    "MiniBatchKMeans",
    "GaussianMixture",
    "SpectralClustering",
    "KNNClassifier",
    "KNNRegressor",
]
