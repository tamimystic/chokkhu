from __future__ import annotations

from .adaboost import AdaBoostClassifier, AdaBoostRegressor
from .clustering_advanced import (
    GaussianMixture,
    MiniBatchKMeans,
    SpectralClustering,
)
from .dbscan import DBSCAN
from .decision_tree import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from .extra_trees import ExtraTreesClassifier, ExtraTreesRegressor
from .glm import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    LassoRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    RidgeRegression,
)
from .gradient_boosting import (
    GradientBoosting,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .hierarchical import HierarchicalClustering
from .hist_gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from .isolation_forest import IsolationForest
from .kmeans import KMeans
from .kmedoids import KMedoids
from .knn import KNN
from .knn_spatial import (
    KNNClassifier,
    KNNRegressor,
)
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .naive_bayes_extended import (
    BernoulliNB,
    ComplementNB,
    GaussianNB,
    LDAClassifier,
    LinearDiscriminantAnalysis,
    MultinomialNB,
    QDAClassifier,
    QuadraticDiscriminantAnalysis,
)
from .nearest_neighbors import (
    NearestCentroid,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from .neural_network import (
    MLPClassifier,
    MLPRegressor,
    NeuralNetwork,
)
from .optics import OPTICS
from .random_forest import (
    RandomForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .svm import SVM
from .svm_kernels import SVC, SVR, LinearSVC, OneClassSVM

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
