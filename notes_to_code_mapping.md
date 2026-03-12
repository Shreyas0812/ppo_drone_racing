# Notes ↔ Code: Complete Mapping

## Your Codebase at a Glance

Your project implements **PPO (Proximal Policy Optimization)** with an **Actor-Critic** architecture to train a drone to hover. The code lives in 4 main files:

| File | Role | Primary Algorithm |
|---|---|---|
| `actor_critic.py` | Neural network (policy + value function) | Stochastic Gaussian Policy (Notes p.6, 21-22) |
| `rollout_buffer.py` | Trajectory storage + GAE computation | GAE — Generalized Advantage Estimation (Notes p.19) |
| `ppo.py` | PPO clipped objective update | PPO v2 — Clipped Version (Notes p.28) |
| `train.py` | Training loop orchestrator | Full PPO Algorithm (Notes p.29) |
| `eval.py` | Inference / evaluation | Deterministic policy deployment |

---

## 1. `actor_critic.py` — The Policy and Critic Networks

### 1a. Stochastic Gaussian Policy (Actor)

**Notes reference:** Pages 6, 21–22

Your notes on page 6 define a stochastic policy:

> π_θ(a|s) is a probability distribution → sample from this distribution is the action

And on page 21, you specifically work out the Gaussian case:

> π_θ(a|s) = N(μ_ψ(s), σ²)

**In your code**, this is exactly what `ActorCritic` implements:

```python
# actor_critic.py
self.actor = nn.Sequential(
    nn.LayerNorm(obs_dim),
    nn.Linear(obs_dim, hidden[0]), nn.ELU(),
    nn.Linear(hidden[0], hidden[1]), nn.ELU(),
    nn.Linear(hidden[1], action_dim),
    nn.Tanh()  # bounds mean to [-1, 1]
)
self.log_std = nn.Parameter(torch.zeros(action_dim))
```

The actor network **is** μ_ψ(s) from your notes — it takes in the state and outputs the mean action. The `log_std` parameter **is** log(σ) — a learnable but state-independent standard deviation.

Together they define: **π_θ(a|s) = N(actor(s), exp(log_std)²)**

This matches your notes page 21 exactly:

> π_θ(a|s) = (1/√(2πσ²)) exp(-(a - μ_ψ(s))² / 2σ²)

### 1b. The Critic (Value Function)

**Notes reference:** Pages 13, 15–17

Your notes on page 13 establish that the optimal baseline is approximately the value function:

> b* ≈ E[R(z)] = V^π(s₀) — value function of state at zero

And on page 15, you note that in practice we approximate it with a neural network:

> V^π(s_t) ≈ V̂_w(s_t) ← Neural Network (CRITIC)

**In your code:**

```python
# actor_critic.py
self.critic = nn.Sequential(
    nn.LayerNorm(obs_dim),
    nn.Linear(obs_dim, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 1)
)
```

This is V̂_w(s) — it takes a state and outputs a single scalar value estimate. Your notes on page 15 explicitly say "we train CRITIC & POLICY at the same time," which is exactly what your training loop does.

**Why the critic is bigger than the actor (256-128 vs 128-128):** Your notes page 16 discuss how fitting the value function is hard because targets have high variance for small t and low rewards for large t. A larger network helps the critic learn this complex mapping.

### 1c. `get_action()` — Sampling from the Policy

**Notes reference:** Pages 6, 9 (REINFORCE step 1)

Your notes on page 9, REINFORCE step 1:

> 1. Sample {τ}^N from π(a_t|s_t, θ)

**In your code:**

```python
def get_action(self, obs):
    mean = self.actor(obs)
    std = self.log_std.exp()
    dist = Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    value = self.critic(obs).squeeze(-1)
    return action, log_prob, value
```

This does exactly what REINFORCE step 1 says: sample an action from the current policy distribution. The `log_prob` is ∇_θ log π(a_t|s_t, θ) which you'll need for the policy gradient (page 8-9). The `value` is V̂_w(s_t) from the critic.

### 1d. `evaluate()` — Re-evaluating Old Actions Under Current Policy

**Notes reference:** Pages 25–26 (Importance Sampling)

Your notes on page 25 introduce importance sampling for off-policy correction:

> L^IS(θ) = E_{τ_old} [ p(τ_new)/p(τ_old) · Σ log π(a_t|s_t) · Â_t ]

The `evaluate()` method re-computes log probabilities of *old* actions under the *current* (updated) policy parameters. This is essential for computing the importance sampling ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) in PPO.

### 1e. Policy Gradient for Gaussian — The Score Function

**Notes reference:** Pages 21–22

Your notes derive the partial derivatives for a Gaussian policy:

> g_μ := ∂/∂μ log π_θ(a|s) = (a - μ) / σ²
> g_σ := ∂/∂σ log π_θ(a|s) = ((a - μ)² - σ²) / σ³

And on page 22:

> g_ψ = (a - μ)/σ² · ∂μ/∂ψ · R_t

In your code, you never manually compute these — PyTorch's autograd handles it when you call `loss.backward()`. But this is exactly what's happening under the hood when gradients flow through `dist.log_prob(action)`.

The intuition from page 22 is important: if R_t is positive, μ gets pushed toward action a (the mean shifts to make that good action more likely). If R_t is negative, μ moves away. And σ controls exploration — it shrinks when the advantage is positive (we're confident in this action) and grows otherwise.

---

## 2. `rollout_buffer.py` — Trajectory Collection and GAE

### 2a. Trajectory Storage

**Notes reference:** Pages 6, 9

Your notes page 6 define a trajectory:

> τ = (s₀, a₀, s₁, a₁, ......)

And the cumulative reward for a trajectory (page 6):

> R(τ) = Σ_{t=0}^{T-1} r(s_t, a_t)

**In your code**, the `RolloutBuffer` stores exactly these components for 2048 timesteps:
- `obs` — the states s_t
- `actions` — the actions a_t
- `rewards` — the rewards r_t
- `dones` — episode termination flags
- `values` — V̂_w(s_t) from the critic
- `log_probs` — log π_θ(a_t|s_t) for importance sampling later

### 2b. GAE Computation — The Heart of Variance Reduction

**Notes reference:** Pages 18–19 (GAE), Pages 10–14 (building up to it)

This is where your notes build up beautifully to the code. Let me trace the full path:

**Step 1 — Reward-to-go (Page 10):**
Your notes show that the policy gradient can use reward-to-go instead of full trajectory return:

> g_θ = E [ Σ_t ∇_θ log π(a_t|s_t, θ) · Σ_{j=t}^{T-1} r_j ]

This is called the "empirical Q function" or "reward to go."

**Step 2 — Baseline subtraction (Pages 10–13):**
You prove that subtracting a baseline b(s_t) doesn't change the expected gradient but reduces variance:

> g_θ = E [ Σ_t ∇_θ log π(a_t|s_t, θ) · (Σ_{j=t}^{T-1} r_j - b(s_t)) ]

And the optimal baseline is approximately the value function V^π(s_t), giving us the **advantage**:

> A^π(s_t, a_t) = Σ_{j=t}^{T-1} r_j - V^π(s_t)

**Step 3 — TD estimates and bootstrapping (Page 18):**
Instead of using the full Monte Carlo return, you can bootstrap with the critic:

> TD(0): A_t ≈ r_t + V̂_w(s_{t+1}) - V̂_w(s_t) — one-step estimate (low variance, high bias)
> TD(n): A_t ≈ r_t + r_{t+1} + ... + r_{t+n} + V̂_w(s_{t+n+1}) - V̂_w(s_t) — n-step estimate

**Step 4 — GAE: Don't choose n, blend them all (Page 19):**

> GAE: A_t(θ) = Σ_{n=1}^{T-t-1} w_n · TD(n) / Σ w_n, where w_n = λ^{n-1}

Your notes say: "This simply does an average [of TD estimates] rather than choosing 1."

**In your code**, this becomes the elegant backward pass:

```python
def compute_returns_and_advantages(self, last_value, gamma=0.95, lam=0.95):
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
```

Line by line:

| Code | Equation (from notes) |
|---|---|
| `delta = r[t] + γ * V(s_{t+1}) * (1-done) - V(s_t)` | δ_t = r_t + γ V(s_{t+1}) - V(s_t) — the TD residual (Notes p.18, "TD(0)") |
| `gae = delta + γ * λ * (1-done) * gae` | Â_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ... — recursive GAE formula (Notes p.19) |
| `returns = advantages + values` | G_t = Â_t + V(s_t) — the target for critic training |

The `(1 - self.dones[t])` term zeros out bootstrapping across episode boundaries — if the episode ended, we don't bootstrap from the "next state" because there isn't one in the same episode.

**The key parameters:**
- `gamma = 0.95` — **Discount factor** (Notes p.14). Your notes say γ controls the "effective horizon" ≈ 1/(1-γ) = 20 steps. In racing, you note "we need to make sure it at least makes it to the next gate."
- `lam = 0.95` — **GAE lambda** (Notes p.19). λ=0 gives pure TD(0) (low variance, high bias). λ=1 gives Monte Carlo (no bias, high variance). λ=0.95 is the standard blend.

### 2c. Advantage Normalization

**Not explicitly in your notes, but important:**

```python
def get(self):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

This normalizes advantages to have zero mean and unit variance. This is a practical trick that stabilizes training — it ensures roughly half the actions are "good" (positive advantage) and half are "bad" (negative), preventing the gradient from being dominated by scale.

---

## 3. `ppo.py` — The PPO Update

### 3a. The Importance Sampling Ratio

**Notes reference:** Pages 25–26

Your notes page 25 introduce importance sampling:

> E_{x~p(x)}[f(x)] = E_{x~q(x)}[p(x)/q(x) · f(x)]

And define the importance-sampled loss:

> L^IS(θ) = E_{τ_old}[p(τ_new)/p(τ_old) · Σ log π(a_t|s_t) · Â_t]

**In your code:**

```python
ratio = (log_probs - old_log_probs[mb]).exp()
```

This computes r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t). Note the clever use of `exp(log_new - log_old) = new/old` to avoid numerical issues with dividing small probabilities.

### 3b. The PPO Clipped Objective

**Notes reference:** Page 28

Your notes define PPO v2 (clipped version):

> L^CLIP = E_{τ_old} [ min( r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]

And explain: "If policy gradient pushes above a point, we say gradient is zero & estimate is not valid."

**In your code:**

```python
surr1 = ratio * advantages[mb]                                           # r_t(θ) · Â_t
surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[mb]  # clip(r_t, 1-ε, 1+ε) · Â_t
policy_loss = -torch.min(surr1, surr2).mean()                           # -L^CLIP (negative because we minimize)
```

The `clip_epsilon = 0.2` means the ratio is clamped to [0.8, 1.2], exactly as drawn in your notes' graph on page 28. When Â_t > 0, this prevents the policy from becoming too much more likely than before. When Â_t < 0, it prevents the policy from becoming too much less likely.

### 3c. The Value Function (Critic) Loss

**Notes reference:** Pages 16–17, 20

Your notes page 16 define the critic loss:

> L(w) = (1/2) Σ ||V̂_w(s_t^i) - y_t^i||²

Where y_t is the target (Monte Carlo return or bootstrapped return).

And page 20 shows the critic update:

> w^new = w^old - α · ∂L/∂w

**In your code:**

```python
value_loss = (values - returns[mb]).pow(2).mean()
```

This is exactly L(w) from your notes. The `returns` were computed by GAE as `advantages + values`, which gives us the bootstrapped target y_t.

### 3d. The Total Loss with Entropy Bonus

**Notes reference:** Page 31

Your notes page 31 show the total loss for PPO:

> L^TOT(θ) = L^CLIP(θ) - H(θ) + L^SUP(θ)

Where H(θ) is entropy (you note "we usually add entropy" and "the aim is to maximize it").

**In your code:**

```python
loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()
```

| Code term | Notes term | Purpose |
|---|---|---|
| `policy_loss` | -L^CLIP | Clipped PPO objective (negated for minimization) |
| `value_loss_coef * value_loss` | L^SUP (or critic loss) | Critic training |
| `-entropy_coef * entropy` | -H(θ) | Entropy bonus — encourages exploration |

The entropy bonus prevents the policy from collapsing to a deterministic policy too early (σ → 0). Your notes page 22 discuss how σ "controls exploration."

### 3e. Multiple Epochs and Minibatches

**Notes reference:** Page 29

Your notes on page 29 show the PPO algorithm:

> Do SGD on L^CLIP(θ) for some epochs

**In your code:**

```python
for epoch in range(n_epochs):          # 5 passes over the data
    indices = torch.randperm(batch_size)
    for start in range(0, batch_size, minibatch_size):  # 4 minibatches per epoch
        ...
```

This gives 5 × 4 = 20 gradient updates per iteration. The clipping mechanism (from 3b) is what makes this safe — without it, multiple passes would cause the policy to diverge too far from the data-collecting policy.

### 3f. Gradient Clipping

**Not in your notes explicitly, but relates to stability (Notes p.42):**

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
```

This prevents exploding gradients — if the gradient norm exceeds 0.5, all gradients are scaled down proportionally.

### 3g. log_std Clamping

```python
with torch.no_grad():
    policy.log_std.clamp_(-2.0, 0.5)
```

This keeps σ between exp(-2)≈0.14 and exp(0.5)≈1.65. Without this, σ could collapse to near-zero (no exploration) or explode (random actions). This relates to your notes page 22 about σ controlling exploration.

---

## 4. `train.py` — The Full PPO Training Loop

### 4a. Overall Structure

**Notes reference:** Page 29 (PPO algorithm)

Your notes page 29 lay out:

> PPO:
> for iter i:
>   Compute N Trajectories
>   Compute A_t^n ∀ t, n
>   Do SGD on L^CLIP(θ) for some epochs
>   θ^old ← θ
> end for

**In your code**, this is exactly the structure:

```python
for iteration in range(max_iterations):
    # 1. Collect N trajectories (2048 steps)
    with torch.no_grad():
        for t in range(buffer.n_steps):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            buffer.store(obs, action, reward, done, value, log_prob)
    
    # 2. Compute advantages (GAE)
    buffer.compute_returns_and_advantages(last_value=last_value)
    
    # 3. Do SGD on L^CLIP for some epochs
    policy_loss, value_loss, entropy, approx_kl, clip_frac = update(policy, optimizer, buffer)
    
    # 4. θ^old ← θ (implicit — old data is thrown away, new data collected next iteration)
    buffer.clear()
```

### 4b. On-Policy Nature

**Notes reference:** Pages 23–24

Your notes page 23 explicitly state:

> On-policy algorithm → can be done only once & then need to collect data again

And page 24 explains *why*:

> L^PG(θ) uses empirical advantage at θ̄, so if θ changes, we get θ_new, hence approximation fails at other points

This is why your code collects fresh trajectories every iteration and calls `buffer.clear()`. The data from iteration i cannot be reused in iteration i+1 because the policy has changed. PPO's clipping allows a few SGD passes on the same data (the `n_epochs`), but not across iterations.

### 4c. Reward Shaping

**Not directly in your RL theory notes, but relates to the reward design problem:**

```python
reward = reward + 0.3 * min(altitude, 1.0) + (0.3 * max(0.0, upward_velocity) if altitude < 1.0 else 0.0)
```

This adds bonuses for altitude (encourage going up) and upward velocity when below target. Your course slides mention that for drone racing "it's not clear how to write a smooth cost function specifying the desired behavior" — this reward shaping is your attempt to guide learning.

### 4d. Explained Variance

```python
explained_var = 1 - (buffer.returns - buffer.values).var() / (buffer.returns.var() + 1e-8)
```

This measures how well the critic predicts returns. It connects to your notes page 13 about choosing the value function as the baseline — if V̂_w is perfect, returns - values would be zero everywhere, and explained_var = 1.

---

## 5. `eval.py` — Deterministic Deployment

**Notes reference:** Page 6 (stochastic vs deterministic)

During evaluation, you use the *mean* of the policy without sampling:

```python
action = policy.actor(obs_tensor)  # deterministic mean — no sampling noise
```

This is μ_θ(s) only — no sampling from N(μ, σ²). During training, stochasticity is essential for exploration (your notes page 22: "σ controls exploration"). During evaluation, you want the best action, which is the mean.

---

## 6. What Your Notes Cover That ISN'T in Your Code

Your notes cover several algorithms and concepts beyond what's implemented:

| Notes Topic | Pages | In Your Code? |
|---|---|---|
| Vanilla REINFORCE | 9 | **No** — PPO is an improvement over this |
| TRPO (Trust Region Policy Optimization) | 25–27 | **No** — PPO replaces TRPO with clipping |
| Importance Sampling (theory) | 25 | **Partially** — the ratio in PPO uses this concept |
| KL Divergence constraint | 25–26 | **No** — PPO v2 uses clipping instead of KL constraint |
| PPO v1 (adaptive KL penalty) | 27–28 | **No** — you implemented PPO v2 (clipped) |
| KL Scheduler for learning rate | 29 | **No** — you use a fixed learning rate |
| Parallel Actors (A3C) | 30 | **No** — single actor, sequential collection |
| MAPPO / IPPO | 32 | **No** — single agent |
| Self-Play | 32 | **No** |
| Population Based Training | 32 | **No** |
| Value / Q-Learning | 34–39 | **No** — you use policy gradient, not value-based |
| Double Q-Learning | 39–40 | **No** |
| Experience Replay Buffer | 40 | **No** — on-policy doesn't use replay |
| SAC (Soft Actor-Critic) | 44–47 | **No** — different algorithm entirely |
| Dynamic Programming | 36 | **No** |
| Fitted Value / Q Iteration | 37 | **No** |

---

## 7. The Big Picture: How Everything Connects

Here's the flow of one training iteration, mapped to your notes:

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: COLLECT TRAJECTORIES                           │
│  Notes: p.9 REINFORCE step 1, p.6 trajectory defn      │
│  Code: train.py rollout loop                            │
│                                                         │
│  For 2048 steps:                                        │
│    a ~ π_θ(·|s)     ← actor_critic.get_action()        │
│    s', r = env(a)   ← env.step()                       │
│    Store (s, a, r, done, V(s), log π)                   │
│                      ← rollout_buffer.store()           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: COMPUTE ADVANTAGES (GAE)                       │
│  Notes: p.10 reward-to-go, p.13 advantage,              │
│         p.18 TD estimates, p.19 GAE                     │
│  Code: rollout_buffer.compute_returns_and_advantages()  │
│                                                         │
│  δ_t = r_t + γ V(s_{t+1}) - V(s_t)   ← TD residual    │
│  Â_t = Σ (γλ)^l δ_{t+l}              ← GAE             │
│  G_t = Â_t + V(s_t)                   ← returns        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: PPO UPDATE                                     │
│  Notes: p.28 PPO clipped, p.31 total loss               │
│  Code: ppo.py update()                                  │
│                                                         │
│  For 5 epochs × 4 minibatches:                          │
│    r(θ) = π_new/π_old       ← importance ratio          │
│    L^CLIP = min(r·Â, clip(r)·Â)  ← clipped objective   │
│    L = L^CLIP + 0.5·L_critic - 0.03·H  ← total loss    │
│    θ ← θ - α·∇L            ← gradient descent          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: CLEAR BUFFER, REPEAT                           │
│  Notes: p.23 "on-policy → collect data again"           │
│  Code: buffer.clear()                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Key Equations Summary — Notes → Code

| Equation | Notes Page | Code Location | Code Line |
|---|---|---|---|
| π_θ(a\|s) = N(μ_θ(s), σ²) | p.21 | `actor_critic.py` | `dist = Normal(mean, std)` |
| log π_θ(a\|s) | p.21 | `actor_critic.py` | `dist.log_prob(action).sum(-1)` |
| V^π(s_t) ≈ V̂_w(s_t) | p.15 | `actor_critic.py` | `self.critic(obs)` |
| δ_t = r_t + γV(s_{t+1}) - V(s_t) | p.18 | `rollout_buffer.py` | `delta = rewards[t] + gamma * next_value * (1-done) - values[t]` |
| Â^GAE_t = Σ(γλ)^l δ_{t+l} | p.19 | `rollout_buffer.py` | `gae = delta + gamma * lam * (1-done) * gae` |
| r_t(θ) = π_θ/π_θ_old | p.25-28 | `ppo.py` | `ratio = (log_probs - old_log_probs).exp()` |
| L^CLIP = min(r·Â, clip(r,1±ε)·Â) | p.28 | `ppo.py` | `torch.min(surr1, surr2).mean()` |
| L^TOT = L^CLIP - H + L^critic | p.31 | `ppo.py` | `loss = policy_loss + vf_coef*value_loss - ent_coef*entropy` |
| L(w) = ½\|\|V̂_w(s) - y\|\|² | p.16, 20 | `ppo.py` | `(values - returns).pow(2).mean()` |
| θ^new = θ^old + α·ĝ | p.9, 20, 29 | `ppo.py` | `optimizer.step()` |
| w^new = w^old - α·∂L/∂w | p.20 | `ppo.py` | `optimizer.step()` (same optimizer) |
| R(τ) = Σ r(s_t, a_t) | p.6 | `rollout_buffer.py` | `self.rewards` accumulated |
| J(θ) = E[Σ γ^t r(s_t, a_t)] | p.14 | `train.py` | reward signal from environment |

