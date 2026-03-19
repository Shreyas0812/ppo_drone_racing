# PPO Drone Hover — ESE 6510 Physical Intelligence

This branch (`gpb-ppo-hover`) is a personal learning sandbox for understanding PPO from first principles before writing the main implementation. The goal was to build a clean, well-commented PPO training loop and get a quadrotor to hover using `gym-pybullet-drones`, with each line of code directly traceable back to the math in my handwritten notes.

---

## What This Branch Is

A from-scratch implementation of **Proximal Policy Optimization (PPO)** applied to the `HoverAviary` drone environment. It is intentionally not optimized — the priority was understanding, not performance. Every design decision is explained inline and mapped to the relevant equations in `notes_to_code_mapping.md`.

---

## Project Structure

```
scripts/
├── actor_critic.py    # Actor-Critic neural network (Gaussian policy + value function)
├── rollout_buffer.py  # Trajectory storage + GAE advantage computation
├── ppo.py             # PPO clipped surrogate objective update step
├── train.py           # Main training loop
└── eval.py            # Evaluation / inference (deterministic policy)

notes_to_code_mapping.md   # Line-by-line mapping of code to handwritten notes
params.md                  # All hyperparameters, metrics, and tuning guide
```

---

## Algorithm Overview

The implementation follows the standard PPO-clip algorithm:

1. **Collect rollout** — run the current policy for `n_steps=2048` steps in `HoverAviary`, storing observations, actions, rewards, values, and log-probabilities in a `RolloutBuffer`
2. **Compute advantages** — use Generalized Advantage Estimation (GAE, γ=0.95, λ=0.95) to compute per-step advantage estimates and returns
3. **PPO update** — perform `n_epochs=5` passes over the buffer in `n_minibatches=4` mini-batches, optimizing the clipped surrogate objective:

   L_CLIP = E[ min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

   where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) and ε = 0.2

4. **Total loss** = policy loss + `0.5` × value loss − `0.03` × entropy bonus
5. Repeat for `max_iterations=2000` iterations

---

## Neural Network Architecture

### Actor (policy)
A 2-layer MLP outputting the **mean** of a Gaussian policy. Standard deviation is a **learnable parameter** independent of state:

   π_θ(a|s) = N(μ_ψ(s), exp(σ))

```
LayerNorm → Linear(obs, 128) → ELU → Linear(128, 128) → ELU → Linear(128, action_dim) → Tanh
```

### Critic (value function)
A slightly larger MLP mapping state to a scalar value estimate V(s):

```
LayerNorm → Linear(obs, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 1)
```

The critic is larger than the actor because fitting the value function is a harder regression problem.

---

## Reward Shaping

The raw `HoverAviary` reward is augmented with:

```python
reward = reward
       + 0.3 * min(altitude, 1.0)               # bonus for gaining altitude up to target
       + 0.3 * max(0, vz) if altitude < 1.0      # bonus for upward velocity below target
```

This gives the policy a dense signal during early training when the drone is on the ground and the sparse hover reward is uninformative.

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_steps` | 2048 | Rollout length per iteration |
| `lr` | 3e-4 | Adam optimizer, shared for actor and critic |
| `clip_epsilon` | 0.2 | PPO trust region |
| `n_epochs` | 5 | Update passes per rollout |
| `n_minibatches` | 4 | Mini-batch count per epoch (batch size = 512) |
| `gamma` | 0.95 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `entropy_coef` | 0.03 | Entropy bonus weight |
| `max_norm` | 0.5 | Gradient clipping norm |

See [params.md](params.md) for the full hyperparameter reference, metric interpretations, and a diagnostic tuning workflow.

---

## Running

**Train:**
```bash
cd scripts
python train.py
```

Training saves a checkpoint to `checkpoint.pt` every 100 iterations and a final `policy.pt` at completion. If `checkpoint.pt` exists, training automatically resumes from where it left off.

**Evaluate:**
```bash
cd scripts
python eval.py
```

Loads `policy.pt` (or falls back to `checkpoint.pt`) and runs the deterministic policy (mean action, no sampling noise) for 5000 steps, printing position, velocity, and motor commands every 50 steps.

---

## Dependencies

- `gym-pybullet-drones`
- `torch`
- `numpy`

---

## Notes & References

- [notes_to_code_mapping.md](notes_to_code_mapping.md) — maps every major code block to its corresponding equation in the handwritten course notes (pages referenced throughout)
- [params.md](params.md) — complete hyperparameter table, metric interpretation guide, and step-by-step tuning diagnostic
