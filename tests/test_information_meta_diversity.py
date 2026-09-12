"""Unit tests for Frontier 4.25: Information Bottleneck, Meta-Learning & Quality Diversity."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from chokkhu.information.bottleneck import DeepVariationalInformationBottleneck
from chokkhu.models.meta.maml import MAML
from chokkhu.models.meta.map_elites import MAPElites


def test_variational_information_bottleneck_fit_and_compress():
    """Verify VIB compresses input, reduces total loss, and produces valid predictions."""
    rng = np.random.default_rng(42)
    N = 120
    # 2 informative features, 4 noise features
    X_info = rng.standard_normal((N, 2))
    y = (X_info[:, 0] + X_info[:, 1] > 0.0).astype(np.int64)
    X_noise = rng.standard_normal((N, 4))
    X = np.hstack([X_info, X_noise])

    vib = DeepVariationalInformationBottleneck(
        input_dim=6,
        latent_dim=4,
        hidden_dim=32,
        num_classes=2,
        beta=0.005,
        learning_rate=0.01,
        random_state=42,
    )

    history = vib.fit(X, y, epochs=25, batch_size=32)

    assert len(history["total_loss"]) == 25
    assert history["total_loss"][-1] < history["total_loss"][0]

    # Test latent encoding
    mu = vib.encode(X, sample=False)
    assert isinstance(mu, np.ndarray)
    assert mu.shape == (N, 4)

    # Test stochastic sampling
    z_sample = vib.encode(X, sample=True)
    assert isinstance(z_sample, tuple)
    assert len(z_sample) == 3
    assert z_sample[0].shape == (N, 4)

    # Test predictions
    probs = vib.predict_proba(X, num_samples=5)
    assert probs.shape == (N, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    preds = vib.predict(X)
    assert preds.shape == (N,)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.70


def test_maml_few_shot_regression_adaptation():
    """Verify MAML fast inner-loop gradient adaptation on few-shot regression tasks."""
    rng = np.random.default_rng(42)

    # Generate synthetic sinusoidal tasks: y = A * sin(x + phi)
    tasks = []
    for _ in range(8):
        amp = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0.0, np.pi)

        x_supp = rng.uniform(-4.0, 4.0, size=(10, 1))
        y_supp = amp * np.sin(x_supp + phase)

        x_query = rng.uniform(-4.0, 4.0, size=(10, 1))
        y_query = amp * np.sin(x_query + phase)

        tasks.append((x_supp, y_supp, x_query, y_query))

    maml = MAML(
        layer_sizes=[1, 32, 32, 1],
        task_type="regression",
        inner_lr=0.02,
        meta_lr=0.005,
        inner_steps=5,
        random_state=42,
    )

    losses = maml.meta_fit(tasks, epochs=15, batch_size=4)
    assert len(losses) == 15
    assert losses[-1] < losses[0]

    # Evaluate adaptation on a test task
    test_amp = 2.0
    test_phase = 0.5
    x_test_supp = rng.uniform(-4.0, 4.0, size=(10, 1))
    y_test_supp = test_amp * np.sin(x_test_supp + test_phase)
    x_test_query = rng.uniform(-4.0, 4.0, size=(20, 1))
    y_test_query = test_amp * np.sin(x_test_query + test_phase)

    # Initial query loss before adaptation
    unadapted_pred = maml.predict(x_test_query)
    unadapted_mse = np.mean((unadapted_pred[:, np.newaxis] - y_test_query) ** 2)

    # Adapted query loss after fast inner-loop updates
    adapted_params = maml.adapt(x_test_supp, y_test_supp, inner_steps=10, inner_lr=0.05)
    adapted_pred = maml.predict(x_test_query, params=adapted_params)
    adapted_mse = np.mean((adapted_pred[:, np.newaxis] - y_test_query) ** 2)

    assert adapted_mse < unadapted_mse


def test_maml_few_shot_classification_adaptation():
    """Verify MAML classification meta-training and probability outputs."""
    rng = np.random.default_rng(42)

    tasks = []
    for _ in range(6):
        # 2-way classification with rotating decision boundaries
        theta = rng.uniform(0.0, np.pi)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        X_s = rng.standard_normal((12, 2)) @ rot
        y_s = (X_s[:, 0] > 0.0).astype(np.int64)

        X_q = rng.standard_normal((12, 2)) @ rot
        y_q = (X_q[:, 0] > 0.0).astype(np.int64)

        tasks.append((X_s, y_s, X_q, y_q))

    maml = MAML(
        layer_sizes=[2, 16, 2],
        task_type="classification",
        inner_lr=0.05,
        meta_lr=0.01,
        inner_steps=3,
        random_state=42,
    )

    losses = maml.meta_fit(tasks, epochs=10, batch_size=3)
    assert len(losses) == 10

    # Test adaptation on classification task
    x_test_s, y_test_s, x_test_q, y_test_q = tasks[0]
    adapted = maml.adapt(x_test_s, y_test_s, inner_steps=5, inner_lr=0.05)
    probs = maml.predict_proba(x_test_q, params=adapted)
    assert probs.shape == (12, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    preds = maml.predict(x_test_q, params=adapted)
    assert preds.shape == (12,)


def test_map_elites_quality_diversity_illumination():
    """Verify MAP-Elites populates behavior space grid and optimizes diverse elite solutions."""

    # 4-dimensional sphere function with 2D behavior coordinates (mean of first 2 and last 2 dimensions)
    def eval_fn(genome: np.ndarray) -> Tuple[float, np.ndarray]:
        # Target center at 0.5
        fitness = float(-np.sum((genome - 0.5) ** 2))
        # Behavior descriptors: bounded in [0, 1]
        b1 = float(np.clip(np.mean(genome[:2]), 0.0, 1.0))
        b2 = float(np.clip(np.mean(genome[2:]), 0.0, 1.0))
        return fitness, np.array([b1, b2], dtype=np.float64)

    map_elites = MAPElites(
        genome_dim=4,
        behavior_dims=(6, 6),  # 36 total niche cells
        behavior_bounds=[(0.0, 1.0), (0.0, 1.0)],
        genome_bounds=(0.0, 1.0),
        mutation_sigma=0.1,
        crossover_prob=0.5,
        random_state=42,
    )

    res = map_elites.optimize(
        eval_fn,
        num_iterations=120,
        initial_samples=30,
        batch_size=8,
    )

    assert res.coverage > 0.25
    assert res.total_evaluations > 100
    assert len(res.history_coverage) == 120
    assert res.max_fitness > -0.5

    # Check elite retrieval
    elites = map_elites.get_elites()
    assert len(elites) == len(res.archive)

    best_ind = map_elites.get_best_overall()
    assert best_ind is not None
    assert best_ind.fitness == res.max_fitness

    # Check grid representation
    grid = map_elites.get_archive_grid()
    assert grid.shape == (6, 6)
