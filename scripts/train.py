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

    # Update policy using PPO
    policy_loss, value_loss, entropy, approx_kl = update(policy, optimizer, buffer)

    # Logging
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}/{max_iterations} completed.")
        print(f"mean reward: {buffer.rewards.mean():.3f} mean value: {buffer.values.mean():.3f} mean advantage: {buffer.advantages.mean():.3f}")
        print(f"policy_loss: {policy_loss:.4f}  value_loss: {value_loss:.4f}  entropy: {entropy:.4f}  approx_kl: {approx_kl:.4f}")

    buffer.clear()  # Clear buffer for the next iteration

env.close()