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
    ep_len = 0
    ep_lengths = []
    with torch.no_grad():
        for t in range(buffer.n_steps):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            buffer.store(obs, action, reward, done, value, log_prob)
            ep_len += 1

            if bool(done):
                ep_lengths.append(ep_len)
                ep_len = 0
                obs = env.reset()[0]
            else:
                obs = next_obs

    """
    Adding ep_len and ep_lengths to track episode lengths during training. This is useful for monitoring how long episodes last, which can indicate learning progress (e.g., longer episodes may suggest better performance in a task like hover). We reset ep_len at the end of each episode and append it to ep_lengths for logging purposes.
    """

    # Compute returns and advantages - GAE
    with torch.no_grad():
        _, _, last_value = policy.get_action(obs)

    """
    Two no_grad blocks — one around the rollout loop, one around the bootstrap value — because both are inference-only. 
    This stops PyTorch from building and storing computation graphs during collection, which saves memory proportional to n_steps (2048 steps here) and speeds up the rollout phase. 
    
    
    torch.no-grad: It tells PyTorch: "don't build a computation graph for anything inside this block."
    We don't need the graph during rollout because we're not doing backpropagation until the update phase.
    """
    
    buffer.compute_returns_and_advantages(last_value=last_value)

    explained_var = 1 - (buffer.returns - buffer.values).var() / (buffer.returns.var() + 1e-8)

    """
    explained_var
    Early in training it will be near 0, and should climb toward 1 as the critic learns. If it stays near 0 or goes negative, the critic network is too small or the learning rate is wrong.

    ~1 => Critic predicts returns almost perfectly (good)
    ~0 => Critic is no better than predicting the mean return (bad)
    <0 => Critic is worse than predicting the mean return (very bad)
    """

    """
    mean_ep_len
    -> Short and increasing: drone is surviving longer as it learns, which is the expected healthy pattern
    -> Stuck at low number: drone is crashing early and policy not improving
    -> Always equal to n_steps: episodes never terminate naturally; the environment's done signal may not be firing, or the hover task has no terminal condition
    """

    """
    clip_frac
    0.0 - 0.1: PPO updates are mostly within the clipping range, which is good for stable learning
    >0.2 consistently: policy is changing too much during updates, which can lead to instability; consider reducing learning rate or number of epochs
    ~0.0 consistently: policy is barely updating, which may indicate a learning rate that's too low or insufficient epochs or vanishing gradients
    """
    # Update policy using PPO
    policy_loss, value_loss, entropy, approx_kl, clip_frac = update(policy, optimizer, buffer)

    # Logging
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}/{max_iterations} completed.")
        mean_ep_len = sum(ep_lengths) / len(ep_lengths) if ep_lengths else float('nan')
        print(f"mean reward: {buffer.rewards.mean():.3f} mean value: {buffer.values.mean():.3f} mean advantage: {buffer.advantages.mean():.3f} mean_ep_len: {mean_ep_len:.1f}")
        print(f"policy_loss: {policy_loss:.4f}  value_loss: {value_loss:.4f}  entropy: {entropy:.4f}  approx_kl: {approx_kl:.4f}  clip_frac: {clip_frac:.3f}  explained_var: {explained_var:.4f}")

    buffer.clear()  # Clear buffer for the next iteration

torch.save(policy.state_dict(), "policy.pt")
env.close()