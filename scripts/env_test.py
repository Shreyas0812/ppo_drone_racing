from gym_pybullet_drones.envs import HoverAviary

env = HoverAviary()
obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Sample observation:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"reward: {reward:.3f} done: {done}")

env.close()


"""
OUTPUT:

Observation space: Box(
    [[-inf -inf   0.  -inf -inf -inf -inf -inf -inf -inf -inf -inf  -1.  -1. -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. -1.  -1.]], 
    [[ inf  inf   inf inf   inf  inf  inf  inf  inf  inf  inf  inf   1.   1.   1.   1.   1.   1.  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.]], 
    (1, 72), float32)

Idea is : 

x_min, y_min, z_min, 3 velocity components, 3 angular rate components, 60 history components (velocity and angular rate history)
x_max, y_max, z_max, 3 velocity components, 3 angular rate components, 60 history components (velocity and angular rate history)

---
  
Action space: Box(-1.0, 1.0, (1, 4), float32)

Idea is:

4 rotor commands normalized between -1 and 1

---

Sample observation: [[ 0.      0.      0.1125  0.     -0.      0.      0.      0.      0.       0.      0.      0.      0.      0.      0.      0.      0.      0.  0.      0.      0.      0.      0.      0.      0.      0.      0.       0.      0.      0.      0.      0.      0.      0.      0.      0.  0.      0.      0.      0.      0.      0.      0.      0.      0.       0.      0.      0.      0.      0.      0.      0.      0.      0.  0.      0.      0.      0.      0.      0.      0.      0.      0.       0.      0.      0.      0.      0.      0.      0.      0.      0.    ]]

Idea is:

x_value, y_value, z_value, x_velocity, y_velocity, z_velocity, x_angular_rate, y_angular_rate, z_angular_rate, 60 history components (velocity and angular rate history)

Reward based on proximity to target position (0, 0, 1) and penalizing for high velocities and angular rates.

reward: 1.379 done: False
reward: 1.377 done: False
reward: 1.375 done: False
reward: 1.373 done: False
reward: 1.373 done: False

"""