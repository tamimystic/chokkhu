from __future__ import annotations

import numpy as np

from chokkhu import train
from chokkhu.models.rl import GridWorld, QLearning


def test_qlearning_default_env():
    # Should use the default GridWorld environment
    model = train("q_learning", episodes=10, random_state=42, verbose=False)
    assert isinstance(model, QLearning)
    assert isinstance(model.env, GridWorld)

    # Predict best action for state 0 (start)
    X_test = np.array([0])
    preds = model.predict(X_test)
    assert len(preds) == 1
    assert preds[0] in [0, 1, 2, 3]


def test_qlearning_custom_env():
    class DummyEnv:
        def __init__(self):
            self.observation_space = 2
            self.action_space = [0, 1]

        def reset(self):
            return 0

        def step(self, action):
            return 1, 1.0, True

    custom_env = DummyEnv()
    model = train(
        "q_learning", env=custom_env, episodes=5, random_state=42, verbose=False
    )

    assert isinstance(model.env, DummyEnv)

    # Predict for state 0
    X_test = np.array([0, 1])
    preds = model.predict(X_test)
    assert len(preds) == 2
