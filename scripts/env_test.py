from gym_pybullet_drones.envs import HoverAviary

env = HoverAviary()
obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Sample observation:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"reward: {reward} done: {done} truncated: {truncated}")

env.close()