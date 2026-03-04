import torch

def update(policy, optimizer, buffer, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01, n_epochs=5, n_minibatches=4):
    
    obs, actions, old_log_probs, returns, advantages = buffer.get()

    # Normalize advantages for better training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.size(0)
    minibatch_size = batch_size // n_minibatches
    