import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from gym_pybullet_drones.envs import HoverAviary
import torch
from actor_critic import ActorCritic

env = HoverAviary(gui=False, record=False)
obs_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]

policy = ActorCritic(obs_dim, action_dim)
if os.path.exists("policy.pt"):
    policy.load_state_dict(torch.load("policy.pt"))
else:
    checkpoint = torch.load("checkpoint.pt")
    policy.load_state_dict(checkpoint["policy"])
    print(f"policy.pt not found, loaded checkpoint from iteration {checkpoint['iteration'] + 1}")


"""
Rebuilds the same network architecture used during training, then loads the saved weights from policy.pt into it. 
"""

policy.eval()

"""
policy.eval() switches off any training-specific behavior (like dropout) — not strictly necessary here since ActorCritic doesn't use those, but it's good practice.
"""

obs, _ = env.reset()
for step in range(5000):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action = policy.actor(obs_tensor)  # deterministic mean — no sampling noise

    obs, reward, done, truncated, _ = env.step(action.clamp(-1.0, 1.0))

    if step % 50 == 0:
        x, y, z = obs[0, 0], obs[0, 1], obs[0, 2]
        vx, vy, vz = obs[0, 3], obs[0, 4], obs[0, 5]
        act = action.squeeze().tolist()
        print(f"step {step:4d} | pos: ({x:.3f}, {y:.3f}, {z:.3f}) | vel: ({vx:.3f}, {vy:.3f}, {vz:.3f}) | reward: {reward:.3f} | action: [{act[0]:.2f}, {act[1]:.2f}, {act[2]:.2f}, {act[3]:.2f}]")

    crashed = float(obs[0, 2]) < 0.05
    if done or crashed:
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
