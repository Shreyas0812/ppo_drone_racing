import torch

def update(policy, optimizer, buffer, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01, n_epochs=5, n_minibatches=4):

    obs, actions, old_log_probs, advantages, returns = buffer.get()

    batch_size = obs.size(0)
    minibatch_size = batch_size // n_minibatches

    total_policy_loss, total_value_loss, total_entropy, total_kl, total_clip_frac = 0.0, 0.0, 0.0, 0.0, 0.0
    n_updates = 0

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

            # Value function loss
            value_loss = (values - returns[minibatch_indices]).pow(2).mean()

            # Total loss   
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()

            # Update the policy network
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)  # Gradient clipping for stability
            optimizer.step()

            # Prevent log_std from drifting: caps std between exp(-2)=0.14 and exp(0.5)=1.65
            with torch.no_grad():
                policy.log_std.clamp_(-2.0, 0.5)

            approx_kl = (old_log_probs[minibatch_indices] - log_probs).mean()
            clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_kl += approx_kl.item()
            total_clip_frac += clip_frac.item()
            n_updates += 1

    return total_policy_loss / n_updates, total_value_loss / n_updates, total_entropy / n_updates, total_kl / n_updates, total_clip_frac / n_updates