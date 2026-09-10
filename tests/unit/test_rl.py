from __future__ import annotations

import numpy as np

from chokkhu.models.rl import (
    GridWorld,
    QLearning,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    DQN,
    DoubleDQN,
    DuelingDQN,
    REINFORCE,
    ActorCritic,
    PPO,
    SAC,
    EpsilonGreedyBandit,
    UCB1Bandit,
    ThompsonSamplingBandit,
    LinUCBBandit,
    DecisionTransformer,
)


def test_gridworld_and_q_learning() -> None:
    env = GridWorld(size=4)
    state = env.reset()
    assert state == 0
    next_state, reward, done = env.step(1)  # Right
    assert next_state == 1
    assert reward == -0.01
    assert not done

    agent = QLearning(
        env=env, episodes=20, learning_rate=0.1, discount_factor=0.9, epsilon=0.2
    )
    agent.fit()
    assert agent.q_table is not None
    assert agent.q_table.shape == (16, 4)
    preds = agent.predict(np.array([0, 1, 2]))
    assert len(preds) == 3


def test_replay_buffers() -> None:
    # Uniform Replay Buffer
    buf = ReplayBuffer(capacity=100, state_dim=4)
    for i in range(10):
        s = np.array([i, i, i, i], dtype=np.float32)
        s_next = s + 1.0
        buf.push(s, i % 2, 1.0, s_next, False)
    assert len(buf) == 10

    states, actions, rewards, next_states, dones = buf.sample(batch_size=4)
    assert states.shape == (4, 4)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4)
    assert dones.shape == (4,)

    # Prioritized Replay Buffer
    per = PrioritizedReplayBuffer(capacity=100, state_dim=4)
    for i in range(10):
        s = np.array([i, i, i, i], dtype=np.float32)
        s_next = s + 1.0
        per.push(s, i % 2, 1.0, s_next, False)
    assert len(per) == 10

    s_b, a_b, r_b, ns_b, d_b, indices, weights = per.sample(batch_size=4)
    assert s_b.shape == (4, 4)
    assert indices.shape == (4,)
    assert weights.shape == (4,)
    assert np.all(weights > 0.0)

    per.update_priorities(indices, np.array([0.5, 0.2, 0.9, 0.1]))


def test_dqn_and_double_dqn() -> None:
    state_dim = 4
    action_dim = 2
    dqn = DQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=16, lr=0.01)

    s = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    a = dqn.select_action(s, evaluate=True)
    assert 0 <= a < action_dim

    states = np.random.randn(8, state_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, size=8)
    rewards = np.random.randn(8).astype(np.float32)
    next_states = np.random.randn(8, state_dim).astype(np.float32)
    dones = np.zeros(8, dtype=np.float32)

    loss, td_err = dqn.train_step(states, actions, rewards, next_states, dones)
    assert isinstance(loss, float)
    assert td_err.shape == (8,)

    # Double DQN
    ddqn = DoubleDQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=16, lr=0.01)
    loss2, td_err2 = ddqn.train_step(states, actions, rewards, next_states, dones)
    assert isinstance(loss2, float)
    assert td_err2.shape == (8,)


def test_dueling_dqn() -> None:
    state_dim = 4
    action_dim = 3
    model = DuelingDQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=16)

    s = np.array([0.5, -0.1, 0.2, 0.9], dtype=np.float32)
    q, feat, v, a = model.forward(s)
    assert q.shape == (1, action_dim)
    assert feat.shape == (1, 16)
    assert v.shape == (1, 1)
    assert a.shape == (1, action_dim)

    act = model.select_action(s, evaluate=True)
    assert 0 <= act < action_dim

    states = np.random.randn(6, state_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, size=6)
    rewards = np.ones(6, dtype=np.float32)
    next_states = np.random.randn(6, state_dim).astype(np.float32)
    dones = np.zeros(6, dtype=np.float32)

    loss = model.train_step(states, actions, rewards, next_states, dones)
    assert isinstance(loss, float)


def test_reinforce_and_actor_critic() -> None:
    state_dim = 3
    action_dim = 2

    # REINFORCE
    agent = REINFORCE(state_dim=state_dim, action_dim=action_dim, lr=0.01)
    for _ in range(5):
        s = np.random.randn(state_dim).astype(np.float32)
        a = agent.select_action(s)
        agent.store_transition(s, a, 1.0)
    total_r = agent.finish_episode()
    assert total_r == 5.0
    assert len(agent.states) == 0

    # ActorCritic (A2C)
    ac = ActorCritic(
        state_dim=state_dim, action_dim=action_dim, lr_actor=0.01, lr_critic=0.01
    )
    s = np.array([0.2, -0.5, 0.8], dtype=np.float32)
    s_next = np.array([0.3, -0.4, 0.7], dtype=np.float32)
    a = ac.select_action(s)
    act_loss, crit_loss = ac.train_step(s, a, 1.0, s_next, False)
    assert isinstance(act_loss, float)
    assert isinstance(crit_loss, float)


def test_ppo() -> None:
    state_dim = 4
    action_dim = 2
    ppo = PPO(state_dim=state_dim, action_dim=action_dim, epochs=2)

    for _ in range(10):
        s = np.random.randn(state_dim).astype(np.float32)
        a, lp, val = ppo.select_action(s)
        ppo.store_transition(s, a, 1.0, val, lp, False)

    actor_loss, critic_loss = ppo.update(next_value=0.0)
    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert len(ppo.states) == 0


def test_sac() -> None:
    state_dim = 3
    action_dim = 2
    sac = SAC(state_dim=state_dim, action_dim=action_dim, hidden_dim=16)

    s = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    a, log_prob = sac.sample_action(s)
    assert a.shape == (action_dim,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)

    states = np.random.randn(4, state_dim).astype(np.float32)
    actions = np.random.uniform(-1.0, 1.0, size=(4, action_dim)).astype(np.float32)
    rewards = np.ones(4, dtype=np.float32)
    next_states = np.random.randn(4, state_dim).astype(np.float32)
    dones = np.zeros(4, dtype=np.float32)

    crit_loss, act_loss = sac.train_step(states, actions, rewards, next_states, dones)
    assert isinstance(crit_loss, float)
    assert isinstance(act_loss, float)


def test_multi_armed_bandits() -> None:
    n_arms = 4

    # Epsilon-Greedy
    eps_b = EpsilonGreedyBandit(n_arms=n_arms, epsilon=0.5, decay=0.9)
    arm = eps_b.select_arm()
    assert 0 <= arm < n_arms
    eps_b.update(arm, 1.0)
    assert eps_b.counts[arm] == 1

    # UCB1
    ucb = UCB1Bandit(n_arms=n_arms)
    arm_u = ucb.select_arm()
    assert 0 <= arm_u < n_arms
    ucb.update(arm_u, 0.5)

    # Thompson Sampling
    ts = ThompsonSamplingBandit(n_arms=n_arms)
    arm_t = ts.select_arm()
    assert 0 <= arm_t < n_arms
    ts.update(arm_t, 1.0)
    assert ts.alpha[arm_t] == 2.0

    # LinUCB
    lin_ucb = LinUCBBandit(n_arms=n_arms, n_features=3, alpha=1.0)
    context = np.array([1.0, 0.5, -0.2])
    arm_l = lin_ucb.select_arm(context)
    assert 0 <= arm_l < n_arms
    lin_ucb.update(arm_l, context, 1.0)


def test_decision_transformer() -> None:
    state_dim = 4
    act_dim = 2
    dt = DecisionTransformer(
        state_dim=state_dim, act_dim=act_dim, hidden_dim=16, max_length=5
    )

    B, T = 2, 4
    rtg = np.random.randn(B, T, 1).astype(np.float32)
    states = np.random.randn(B, T, state_dim).astype(np.float32)
    actions = np.random.randn(B, T, act_dim).astype(np.float32)
    timesteps = np.zeros((B, T), dtype=np.int64)

    preds = dt.forward(rtg, states, actions, timesteps)
    assert preds.shape == (B, T, act_dim)

    # Test inference get_action
    single_action = dt.get_action(rtg[0], states[0], actions[0], timesteps[0])
    assert single_action.shape == (act_dim,)
