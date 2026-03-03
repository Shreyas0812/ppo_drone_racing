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
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std -- a vector of 4 log-standard deviations for each action dimension (one per rotor command) 

        """
        log_std does not change with the observation, it is a fixed parameter that is learned during training through gradient updates. 

        Together with actor mean, it defines a Gaussian policy. The mean is determined by the actor network based on the current observation, while the log_std is a constant that is optimized to find the right level of exploration (how much randomness in action selection) during training.
 
        π(a|s) = N(μθ(s), e^σ)
        """

        # Critic: obs -> state value -- estimates the value of being in a given state, output is a single scalar value V(s) - estimate of expected return from that state
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

        """
        Higher capacity for critic because it needs to learn a more complex function (mapping from state to expected return) compared to the actor which only needs to learn the mean action.
        """