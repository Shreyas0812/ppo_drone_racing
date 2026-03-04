import torch

"""
Rollout buffer to store trajectories of experience for PPO training.
Stores 
observations (T, obs_dim)
actions (T, action_dim)
rewards (T,)
dones (T,)
values (T,)
log_probs (T,)

GAE:

delta_t     = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t         = delta_t + gamma * lambda * A_{t+1}
returns_t   = A_t + V(s_t)
"""

class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim):
        self.n_steps = n_steps
        self.obs = torch.zeros((n_steps, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((n_steps, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(n_steps, dtype=torch.float32)
        self.dones = torch.zeros(n_steps, dtype=torch.float32)
        self.values = torch.zeros(n_steps, dtype=torch.float32)
        self.log_probs = torch.zeros(n_steps, dtype=torch.float32)
        
        self.advantages = torch.zeros(n_steps, dtype=torch.float32)
        self.returns = torch.zeros(n_steps, dtype=torch.float32)

        self.ptr = 0
    
    def store(self, obs, action, reward, done, value, log_prob):
        # Squeeze the drone batch dimension (1, dim) -> (dim,)
        if hasattr(obs, 'ndim') and obs.ndim > 1:
            obs = obs.squeeze(0)
        if hasattr(action, 'ndim') and action.ndim > 1:
            action = action.squeeze(0)
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value.squeeze() if hasattr(value, 'squeeze') else value
        self.log_probs[self.ptr] = log_prob.squeeze() if hasattr(log_prob, 'squeeze') else log_prob
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute GAE advantages and returns for the stored trajectory. looks backward through the trajectory to compute advantages and returns based on rewards, values, and done flags.
        """
        gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get(self):
        """
        Return normalized advantages and the rest of the data for training.
        """
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # Normalize advantages
        return self.obs, self.actions, self.log_probs, adv, self.returns
    
    def clear(self):
        self.ptr = 0
    
    def mean_reward(self):
        return self.rewards.mean().item()
    
        