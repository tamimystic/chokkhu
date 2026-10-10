import numpy as np
import pytest

from chokkhu.models.ml import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    HierarchicalClustering,
    KMedoids,
    LDAClassifier,
    MultinomialNB,
    QDAClassifier,
)


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    c1 = np.random.randn(20, 2) + np.array([5.0, 5.0])
    c2 = np.random.randn(20, 2) + np.array([-5.0, -5.0])
    return np.vstack([c1, c2])


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_dbscan_dynamic_params(cluster_data):
    X = cluster_data
    for metric in ["euclidean", "manhattan", "cosine"]:
        db = DBSCAN(eps=1.5, min_samples=3, metric=metric)
        labels = db.fit_predict(X)
        assert len(labels) == len(X)
        assert db.labels_ is not None


def test_hierarchical_dynamic_params(cluster_data):
    X = cluster_data
    for linkage in ["ward", "single", "complete", "average"]:
        hc = HierarchicalClustering(n_clusters=2, linkage=linkage)
        labels = hc.fit_predict(X)
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 2

    agg = AgglomerativeClustering(distance_threshold=10.0, n_clusters=None)
    agg_labels = agg.fit_predict(X)
    assert len(agg_labels) == len(X)


def test_kmedoids_dynamic_params(cluster_data):
    X = cluster_data
    km = KMedoids(n_clusters=2, init="random", max_iter=50, random_state=42)
    labels = km.fit_predict(X)
    assert len(labels) == len(X)
    assert km.inertia_ >= 0.0
    preds = km.predict(X[:5])
    assert len(preds) == 5


def test_optics_dynamic_params(cluster_data):
    X = cluster_data
    opt = OPTICS(min_samples=3, eps=1.5)
    labels = opt.fit_predict(X)
    assert len(labels) == len(X)
    assert len(opt.ordering_) == len(X)


def test_gaussian_nb_dynamic_params(classification_data):
    X, y = classification_data
    gnb = GaussianNB(priors=np.array([0.5, 0.5]), var_smoothing=1e-8)
    gnb.fit(X, y)
    preds = gnb.predict(X)
    probs = gnb.predict_proba(X)
    assert len(preds) == len(y)
    assert probs.shape == (len(y), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_discrete_nb_dynamic_params():
    np.random.seed(42)
    X = np.random.randint(0, 10, size=(40, 5))
    y = np.random.choice([0, 1], size=40)

    mnb = MultinomialNB(alpha=0.5, fit_prior=False)
    mnb.fit(X, y)
    assert len(mnb.predict(X)) == len(y)

    bnb = BernoulliNB(binarize=4.0, alpha=0.5)
    bnb.fit(X, y)
    assert len(bnb.predict(X)) == len(y)

    cnb = ComplementNB(norm=True, alpha=0.5)
    cnb.fit(X, y)
    assert len(cnb.predict(X)) == len(y)

    cat_nb = CategoricalNB(alpha=0.5)
    cat_nb.fit(X, y)
    assert len(cat_nb.predict(X)) == len(y)


def test_discriminant_analysis_dynamic_params(classification_data):
    X, y = classification_data
    lda = LDAClassifier(priors=np.array([0.5, 0.5]), tol=1e-5)
    lda.fit(X, y)
    assert len(lda.predict(X)) == len(y)

    qda = QDAClassifier(reg_param=1e-3, tol=1e-5)
    qda.fit(X, y)
    assert len(qda.predict(X)) == len(y)
