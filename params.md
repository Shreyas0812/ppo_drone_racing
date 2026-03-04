# PPO Hyperparameters & Metrics Reference

## Hyperparameters

### train.py

| Parameter | Location | Default | Description |
|---|---|---|---|
| `lr` | `Adam(..., lr=3e-4)` | `3e-4` | Learning rate for both actor and critic |
| `n_steps` | `RolloutBuffer(n_steps=2048)` | `2048` | Steps collected per iteration before an update |
| `max_iterations` | `max_iterations = 1000` | `1000` | Total number of training iterations |

### ppo.py — `update()` arguments

| Parameter | Default | Description |
|---|---|---|
| `clip_epsilon` | `0.2` | PPO clipping range — keeps policy updates within `[1-ε, 1+ε]` of the old policy |
| `value_loss_coef` | `0.5` | Weight of the critic loss in the total loss |
| `entropy_coef` | `0.01` | Weight of the entropy bonus — encourages exploration |
| `n_epochs` | `5` | Number of full passes over the buffer per update |
| `n_minibatches` | `4` | Buffer is split into this many minibatches per epoch |

### rollout_buffer.py — `compute_returns_and_advantages()` arguments

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `0.99` | Discount factor — how much future rewards are valued |
| `lam` | `0.95` | GAE lambda — trades off bias vs variance in advantage estimates. Lower = more bias, less variance |

### ppo.py — Gradient clipping

| Parameter | Location | Default | Description |
|---|---|---|---|
| `max_norm` | `clip_grad_norm_(..., max_norm=0.5)` | `0.5` | Max allowed gradient norm before clipping. Prevents exploding gradients |

**`max_norm` tuning:**
- `policy_loss` spiking → lower to `0.3`
- Learning very slow, gradients small → raise to `1.0`

### actor_critic.py — Network Architecture

| Parameter | Location | Default | Description |
|---|---|---|---|
| `hidden` (actor) | `ActorCritic(..., hidden=[128, 128])` | `[128, 128]` | Actor hidden layer sizes. Increase for more complex tasks |
| Critic hidden sizes | Hardcoded in `__init__` | `[256, 128]` | Critic hidden layers. Currently not exposed as an argument |
| `log_std` init | `nn.Parameter(torch.zeros(action_dim))` | `0.0` → `std=1.0` | Initial action distribution spread. Controls early exploration width |
| Actor output activation | `nn.Tanh()` | `[-1, 1]` | Bounds action mean. Assumes environment action space is normalized to `[-1, 1]` |

**`log_std` init tuning:**
- `torch.zeros` → `std = 1.0` — default, wide initial exploration
- `torch.full(..., -0.5)` → `std ≈ 0.6` — tighter initial actions, less random early on
- `torch.full(..., 0.5)` → `std ≈ 1.6` — more aggressive early exploration

**Actor hidden size tuning:**
- `[128, 128]` → default, sufficient for hover
- `[256, 256]` → more capacity for complex maneuvers (racing gates)
- Critic is intentionally larger than actor (harder function to learn)

### Derived parameters (not tunable directly)

| Parameter | Formula | Value | Description |
|---|---|---|---|
| `minibatch_size` | `n_steps / n_minibatches` | `512` | Samples per gradient update. Smaller = noisier gradients |
| `total_updates_per_iter` | `n_epochs × n_minibatches` | `20` | Total gradient steps per iteration |
| `initial_std` | `exp(log_std_init)` | `1.0` | Initial action standard deviation |

---

## Metrics

### Rollout Metrics (environment performance)

| Metric | Healthy range | What it tells you |
|---|---|---|
| `mean reward` | Increasing over time | Primary learning signal |
| `mean value` | Tracks mean reward roughly | Critic's average state value estimate |
| `mean advantage` | Near 0 | Should be zero-centered after normalization |
| `mean_ep_len` | Short early, increasing | How long each episode lasts before termination |

**`mean_ep_len` interpretation:**
- Short and increasing → drone surviving longer, healthy learning
- Stuck at a low number → drone crashing early, policy not improving
- Always equal to `n_steps` (2048) → `done` signal never fires; no natural episode termination

---

### Training Metrics (optimizer health)

| Metric | Healthy range | What it tells you |
|---|---|---|
| `policy_loss` | Decreasing and stabilizing | How much the clipped PPO objective is being optimized |
| `value_loss` | Decreasing over time | How well the critic predicts returns |
| `entropy` | Slowly decreasing | Exploration level of the policy |
| `approx_kl` | < 0.02 | How much the policy changed in one update cycle |
| `clip_frac` | 0.0 – 0.1 | Fraction of steps where the PPO ratio was actually clipped |
| `explained_var` | Approaching 1.0 | How well the critic explains the variance in returns |

**`entropy` interpretation:**
- Slowly decreasing → healthy, policy is exploiting while retaining some exploration
- Drops to near 0 early → policy collapsed to deterministic; raise `entropy_coef`
- Never decreases → too much forced randomness; lower `entropy_coef`

**`approx_kl` interpretation:**
- Consistently > 0.02 → policy updating too aggressively; lower `lr`, reduce `n_epochs`, lower `clip_epsilon`
- Near 0 → policy barely moving; raise `lr` or `n_epochs`

**`clip_frac` interpretation:**
- 0.0 – 0.1 → healthy, most updates within trust region
- Consistently > 0.2 → policy changing too fast; lower `lr`, reduce `n_epochs`
- Always ~0.0 → policy barely updating; raise `lr`, raise `n_epochs`

**`explained_var` interpretation:**
- ~1.0 → critic predicts returns almost perfectly
- ~0.0 → critic no better than predicting the mean return
- < 0.0 → critic worse than mean prediction; increase critic size, raise `value_loss_coef`, lower `lr`

---

## When to Tune — Diagnostic Workflow

### Step 1: Check if training is even running correctly (iterations 1–10)

| What to check | Expected | Problem if... |
|---|---|---|
| `value_loss` | High but decreasing | Stays constant → critic not learning at all |
| `entropy` | High (~2–4 for 4D action) | Near 0 immediately → policy collapsed on step 1 |
| `approx_kl` | Small but nonzero | Exactly 0 → gradients not flowing |
| `mean reward` | Any value (can be negative) | `nan` or `inf` → numerical issue, check reward scaling |

If anything is `nan`/`inf` — stop immediately. Fix numerical issues before tuning anything.

---

### Step 2: Check early learning signal (iterations 10–50)

| What to check | Expected | Action if wrong |
|---|---|---|
| `explained_var` | Rising from 0 toward 0.3+ | Stuck at 0 or negative → fix critic first (see cheat sheet) |
| `mean reward` | Slight upward trend or less negative | Flat or decreasing → reward may need shaping |
| `mean_ep_len` | Any value > 1 | Always 1 → drone crashing on first step |
| `clip_frac` | 0.05 – 0.2 | Always 0 → policy not updating; always >0.3 → too aggressive |

---

### Step 3: Check mid-training progress (iterations 50–200)

| What to check | Expected | Action if wrong |
|---|---|---|
| `mean reward` | Clearly increasing | Plateau → you've hit a local optimum, adjust `lr` or `entropy_coef` |
| `explained_var` | > 0.5 | Still low → critic is the bottleneck, increase size or `value_loss_coef` |
| `entropy` | Slowly declining | Sudden collapse → raise `entropy_coef` |
| `approx_kl` | Stable < 0.02 | Increasing over time → reduce `lr` or `n_epochs` |

---

### Step 4: Check convergence (iterations 200+)

| What to check | Expected | Action if wrong |
|---|---|---|
| `mean reward` | Approaching task success | Plateau well below goal → may need reward shaping or larger network |
| `explained_var` | > 0.8 | Still < 0.5 → critic hasn't converged, training will be unstable |
| `entropy` | Low but nonzero | Zero → policy is fully deterministic, may be stuck |
| `clip_frac` | Near 0 (< 0.05) | Still high → updates too large for a converging policy |

---

### Diagnostic Priority Order

When something looks wrong, investigate in this order:

```
1. Is mean reward moving at all?
      No  → check entropy (collapsed?) and approx_kl (zero?)
      Yes → continue

2. Is explained_var rising?
      No  → fix critic before touching anything else
      Yes → continue

3. Is entropy decaying too fast?
      Yes → raise entropy_coef
      No  → continue

4. Is approx_kl > 0.02 or clip_frac > 0.2?
      Yes → lower lr or n_epochs
      No  → continue

5. Is reward plateauing?
      Yes → try reward shaping, larger network, or adjusted gamma/lam
```

---

## Tuning Cheat Sheet

| Symptom | What to change |
|---|---|
| Reward not increasing | Check `mean_ep_len`, `explained_var`, and `entropy` first |
| Drone crashing immediately | Reward shaping, lower `lr`, larger network |
| `explained_var` stuck near 0 | Increase critic hidden size, raise `value_loss_coef` |
| `approx_kl` > 0.02 | Lower `lr`, reduce `n_epochs`, lower `clip_epsilon` |
| `clip_frac` > 0.2 | Same as above |
| `entropy` collapses early | Raise `entropy_coef` |
| `value_loss` not decreasing | Lower `lr`, raise `value_loss_coef` |
| Policy loss exploding | Lower `lr`, lower `max_norm` (try `0.3`) |
| Policy learning slowly from start | Raise `log_std` init for more early exploration |
| Actions too random at start | Lower `log_std` init (e.g. `torch.full(..., -0.5)`) |
| Complex task not learning | Increase actor hidden to `[256, 256]`, expose critic hidden as arg and increase too |
| Gradients vanishing, no learning | Raise `max_norm` to `1.0`, check `lr` |
