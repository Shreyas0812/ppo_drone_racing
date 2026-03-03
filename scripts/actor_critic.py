import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=[128, 128]):
        super().__init__()

        # Actor: obs -> action mean
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ELU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.Linear(hidden[1], action_dim),
            nn.Tanh()  # Assuming action space is normalized between -1 and 1
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std

        # Critic: obs -> state value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )