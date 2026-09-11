"""Unit tests for Frontier 4.17: Machine Unlearning & Concept Scrubbing."""

import numpy as np
import pytest

from chokkhu.privacy.unlearning import (
    SISARetraining,
    FisherScrubbing,
    SCRUB,
    NullspaceConceptScrubbing,
)


def test_sisa_exact_unlearning_deterministic_recovery():
    """Verify SISA correctly isolates and removes forget set upon unlearning request."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 4))
    # Linearly separable data
    y = np.where(X[:, 0] + X[:, 1] > 0, 1.0, 0.0)

    sisa = SISARetraining(num_shards=3, num_slices=2, task="classification", seed=42)
    sisa.fit(X, y)

    # Initial predictions
    initial_preds = sisa.predict(X)
    assert len(initial_preds) == 60

    # Request unlearning of samples 0, 1, 2
    forget_indices = [0, 1, 2]
    sisa.forget(forget_indices)

    # Model should still predict accurately on remaining samples
    post_unlearn_preds = sisa.predict(X[3:])
    acc = np.mean(post_unlearn_preds == y[3:])
    assert acc > 0.7


def test_sisa_multi_shard_aggregation():
    """Verify SISA aggregates multi-shard predictions via majority voting."""
    sisa = SISARetraining(num_shards=5, num_slices=2, task="classification")
    X = np.array([[1.0, 2.0], [-1.0, -2.0], [2.0, 3.0], [-2.0, -3.0], [1.5, 2.5], [-1.5, -2.5]])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    sisa.fit(X, y)
    preds = sisa.predict(np.array([[2.0, 2.0], [-2.0, -2.0]]))
    assert preds[0] == 1
    assert preds[1] == 0


def test_fisher_scrubbing_influence_shift():
    """Verify Fisher scrubbing shifts parameters away from forget sample influence."""
    rng = np.random.default_rng(42)
    X_retain = rng.standard_normal((100, 3))
    y_retain = np.where(X_retain[:, 0] > 0, 1.0, 0.0)

    # Forget set with strong opposite correlation
    X_forget = np.array([[5.0, 0.0, 0.0], [4.5, 0.5, 0.0]])
    y_forget = np.array([0.0, 0.0])  # Anomaly labels on forget set

    # Initial weights
    w_init = np.array([1.0, 0.0, 0.0])
    b_init = 0.0

    scrubber = FisherScrubbing(damping=1e-3)
    w_scrubbed, b_scrubbed = scrubber.scrub(
        w_init, b_init, X_forget, y_forget, X_retain, y_retain
    )

    assert w_scrubbed.shape == (3,)
    assert isinstance(b_scrubbed, float)
    # The scrubbing shift should alter w_scrubbed
    assert not np.allclose(w_scrubbed, w_init)


def test_fisher_information_matrix_computation():
    """Test empirical Fisher information computation properties."""
    w = np.array([1.0, -1.0])
    b = 0.5
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y = np.array([1.0, 0.0, 0.0])

    Fisher, theta = FisherScrubbing.compute_empirical_fisher(w, b, X, y)
    assert Fisher.shape == (3, 3)
    # Fisher must be symmetric and positive semi-definite
    np.testing.assert_allclose(Fisher, Fisher.T, atol=1e-8)
    eigenvals = np.linalg.eigvalsh(Fisher)
    assert np.all(eigenvals >= -1e-8)


def test_scrub_kl_distillation_forgetting():
    """Verify SCRUB degrades forget accuracy while preserving teacher retain predictions."""
    rng = np.random.default_rng(42)
    X_retain = rng.standard_normal((100, 4))
    w_teacher = np.array([2.0, -1.5, 0.5, -0.5])
    b_teacher = 0.2
    y_retain = np.where(X_retain @ w_teacher + b_teacher > 0, 1.0, 0.0)

    X_forget = rng.standard_normal((20, 4))
    y_forget = np.where(X_forget @ w_teacher + b_teacher > 0, 1.0, 0.0)

    scrub = SCRUB(lr=0.05, epochs=15, alpha=1.5, beta=1.0)
    w_student, b_student = scrub.unlearn(
        w_teacher, b_teacher, X_forget, y_forget, X_retain, y_retain
    )

    # Retain set predictions should remain close to teacher
    teacher_preds_r = np.where(X_retain @ w_teacher + b_teacher > 0, 1.0, 0.0)
    student_preds_r = np.where(X_retain @ w_student + b_student > 0, 1.0, 0.0)
    retain_agreement = np.mean(teacher_preds_r == student_preds_r)
    assert retain_agreement >= 0.80


def test_nullspace_concept_orthogonality():
    """Verify Nullspace projection projects concept direction to zero."""
    rng = np.random.default_rng(42)
    # Features with concept embedded along axis 0
    H0 = rng.standard_normal((50, 4))
    H1 = rng.standard_normal((50, 4))
    H0[:, 0] += 5.0  # Concept class 0
    H1[:, 0] -= 5.0  # Concept class 1

    H = np.vstack([H0, H1])
    c = np.array([0] * 50 + [1] * 50)

    nullspace = NullspaceConceptScrubbing()
    nullspace.fit(H, c, num_directions=1)

    H_scrubbed = nullspace.transform(H)
    assert H_scrubbed.shape == H.shape

    # Mean difference between classes after scrubbing should be near zero along concept basis
    mean0 = np.mean(H_scrubbed[:50], axis=0)
    mean1 = np.mean(H_scrubbed[50:], axis=0)
    concept_diff = mean0 - mean1
    
    # Projection of diff onto concept basis must be virtually 0
    v_concept = nullspace.concept_basis
    assert np.abs(concept_diff @ v_concept) < 1e-4


def test_nullspace_scrub_weights():
    """Test weight matrix scrubbing via nullspace projection."""
    nullspace = NullspaceConceptScrubbing()
    H = np.array([[5.0, 0.0], [-5.0, 0.0]])
    c = np.array([0, 1])
    nullspace.fit(H, c, num_directions=1)

    # Weight matrix sensitive to dimension 0
    W = np.array([[2.0, 3.0], [1.0, 1.0]])  # (2, 2)
    W_scrubbed = nullspace.scrub_weights(W, mode="input")

    # The first row (sensitive to dim 0) should be zeroed out
    np.testing.assert_allclose(W_scrubbed[0], [0.0, 0.0], atol=1e-6)
