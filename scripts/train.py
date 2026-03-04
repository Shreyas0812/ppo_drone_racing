"""
PPO (Proximal Policy Optimization) training script for drone racing using the HoverAviary environment.
This script implements a complete PPO training loop for training a neural network policy to control
a quadrotor drone in a hover task. It uses an Actor-Critic architecture with a rollout buffer to
collect experience, computes Generalized Advantage Estimation (GAE) for better gradient estimates,
and performs policy updates via PPO.
Key Components:
- HoverAviary: Physics simulation environment for drone control
- ActorCritic: Neural network with separate actor (policy) and critic (value) heads
- RolloutBuffer: Collects trajectories and computes returns/advantages
- PPO Update: Clipped objective function to prevent overly large policy changes
Training Loop:
1. Reset environment and collect n_steps of experience
2. Compute returns and advantages using GAE
3. Calculate explained variance (critic quality metric)
4. Perform PPO policy and value function updates
5. Clear buffer and repeat
Parameters:
- max_iterations: Number of training iterations (1000)
- n_steps: Trajectory length per iteration (2048)
- learning_rate: Adam optimizer learning rate (3e-4)
- explained_var: Measures how well the critic predicts returns (should approach 1.0)
Returns:
- Trained policy network saved implicitly through optimizer updates
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from gym_pybullet_drones.envs import HoverAviary
import torch

from actor_critic import ActorCritic
from ppo import update
from rollout_buffer import RolloutBuffer

env = HoverAviary(gui=False, record=False)
obs, info = env.reset()

obs_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]

policy = ActorCritic(obs_dim, action_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
buffer = RolloutBuffer(n_steps=2048, obs_dim=obs_dim, action_dim=action_dim)

max_iterations = 1000

for iteration in range(max_iterations):
    # Collect experience
    obs, info = env.reset()
    for t in range(buffer.n_steps):
        action, log_prob, value = policy.get_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        buffer.store(obs, action, reward, done, value, log_prob)
        obs = next_obs if not bool(done) else env.reset()[0]

    # Compute returns and advantages - GAE
    _, _, last_value = policy.get_action(obs)
    buffer.compute_returns_and_advantages(last_value=last_value)

    explained_var = 1 - (buffer.returns - buffer.values).var() / (buffer.returns.var() + 1e-8)

    """
    explained_var
    Early in training it will be near 0, and should climb toward 1 as the critic learns. If it stays near 0 or goes negative, the critic network is too small or the learning rate is wrong.

    ~1 => Critic predicts returns almost perfectly (good)
    ~0 => Critic is no better than predicting the mean return (bad)
    <0 => Critic is worse than predicting the mean return (very bad)
    """

    # Update policy using PPO
    policy_loss, value_loss, entropy, approx_kl = update(policy, optimizer, buffer)

    # Logging
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}/{max_iterations} completed.")
        print(f"mean reward: {buffer.rewards.mean():.3f} mean value: {buffer.values.mean():.3f} mean advantage: {buffer.advantages.mean():.3f}")
        print(f"policy_loss: {policy_loss:.4f}  value_loss: {value_loss:.4f}  entropy: {entropy:.4f}  approx_kl: {approx_kl:.4f}  explained_var: {explained_var:.4f}")

    buffer.clear()  # Clear buffer for the next iteration

env.close()