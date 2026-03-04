import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from gym_pybullet_drones.envs import HoverAviary
import torch
from actor_critic import ActorCritic

env = HoverAviary(gui=True)
obs_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]

policy = ActorCritic(obs_dim, action_dim)
policy.load_state_dict(torch.load("policy.pt"))


"""
Rebuilds the same network architecture used during training, then loads the saved weights from policy.pt into it. 
"""

policy.eval()

"""
policy.eval() switches off any training-specific behavior (like dropout) — not strictly necessary here since ActorCritic doesn't use those, but it's good practice.
"""

obs, _ = env.reset()
for _ in range(5000):
    with torch.no_grad():
        action, _, _ = policy.get_action(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done:
        obs, _ = env.reset()

"""
At each step:
1. Feed the current observation into the policy to get an action (the _, _ discards log_prob and value, which are only needed during training)
2. Step the environment with that action, get the next observation
3. If the drone crashes/episode ends, reset and keep watching
"""

"""
torch.no_grad() is here for the same reason as in training — we're doing pure inference, so no need to build a computation graph.
"""

env.close()
