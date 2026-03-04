import torch

def update(policy, optimizer, buffer, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01, n_epochs=5, n_minibatches=4):
    
    obs, actions, old_log_probs, returns, advantages = buffer.get()

    # Normalize advantages for better training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.size(0)
    minibatch_size = batch_size // n_minibatches

    for epoch in range(n_epochs):
        # Shuffle the indices for each epoch to ensure different mini-batches
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]

            # Re-evaluate the policy for the current mini-batch
            log_probs, entropy, values = policy.evaluate(obs[minibatch_indices], actions[minibatch_indices])

            # Policy loss with clipping
            ratio = (log_probs - old_log_probs[minibatch_indices]).exp()  # Importance sampling ratio
            surr1 = ratio * advantages[minibatch_indices]
            surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[minibatch_indices]
            policy_loss = -torch.min(surr1, surr2).mean()  # PPO objective (maximize the minimum of the two surrogate losses)

            