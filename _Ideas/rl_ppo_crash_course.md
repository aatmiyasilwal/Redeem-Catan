# Reinforcement Learning & PPO Crash Course
## From Supervised Learning to Game-Playing Agents

---

## Part 1: Why Reinforcement Learning is Fundamentally Different

### The Supervised Learning Paradigm (What You Know)

In supervised learning, you have:
- **Input → Output pairs** `(X, y)`
- A **loss function** that tells you exactly how wrong each prediction is
- **Independent, identically distributed (i.i.d.)** data points

```python
# Supervised: clear target, clear error
pred = model(X_batch)
loss = cross_entropy(pred, y_batch)  # exact gradient direction
loss.backward()
optimizer.step()
```

The model is told **exactly** what the right answer should be for every input.

### The Reinforcement Learning Paradigm

In RL, you have:
- **States** `s` — the agent's current observation of the world
- **Actions** `a` — what the agent can do
- **Rewards** `r` — a scalar signal that says "that was good/bad" — but **not what the right action was**
- **Trajectories** — sequences of `(s, a, r, s')` that are **correlated**, not i.i.d.

```python
# RL: no target label, only a reward signal
action = policy(state)
next_state, reward, done, _ = env.step(action)
# But was this action the right one? We don't know.
# We only know the reward we got. We have to infer quality.
```

**The fundamental difference**: In supervised learning, the loss function is a **teacher**. In RL, the reward signal is at best a **hint** — it tells you the outcome was good or bad, but not what action would have been optimal.

---

## Part 2: The RL Formalism — Markov Decision Processes

Every RL problem is framed as a **Markov Decision Process (MDP)**:

| Component | Symbol | Catan Example |
|-----------|--------|---------------|
| State space | `S` | Board layout, hands, dev cards, settlements, longest road, largest army |
| Action space | `A` | Build road, build settlement, buy dev card, trade, roll, move robber |
| Transition function | `P(s' \| s, a)` | Dice rolls, card draws, opponent actions (stochastic) |
| Reward function | `R(s, a, s')` | +1 for VP, +bonus for longest road, negative for being robbed |
| Discount factor | `γ` | How much you value future VP vs. immediate VP (typically 0.99) |

### The Goal: Find an Optimal Policy

A **policy** `π(a | s)` is a mapping from states to action probabilities. The goal is to find `π*` that maximizes the expected **cumulative discounted reward**:

```
J(π) = E[Σ γ^t · r_t]  over t=0 to T
```

In Catan: maximize expected total VP over the course of a game, with a slight preference for getting VP sooner rather than later.

---

## Part 3: The Two Approaches — Value-Based vs. Policy-Based

### Value-Based Methods (Q-Learning, DQN)

Learn a **value function** `Q(s, a)` that estimates "how good is it to take action `a` in state `s`, then follow the optimal policy thereafter?"

```
Q(s, a) ≈ expected return from (s, a) onward
```

The policy is then **derived** from the Q-function:
```
π(s) = argmax_a Q(s, a)
```

**Problems for Catan:**
- The action space is **huge** (~3000 discrete actions). You'd need to evaluate Q for all 3000 at every step.
- Categorical actions (build on node 42 vs. node 43) don't generalize well — the network doesn't learn that "building near wheat is generally good" across different node IDs.

### Policy-Based Methods (REINFORCE, PPO)

Learn the **policy directly** `π_θ(a | s)` — a neural network that outputs action probabilities given a state.

```python
logits = policy_network(state)      # shape: (action_space_size,)
probs = softmax(logits)
action = sample(probs)
```

**Advantages for Catan:**
- Handles **large, discrete** action spaces naturally (output layer = action space size)
- Can learn **stochastic** policies (sometimes explore, sometimes exploit)
- Works with **continuous** action spaces too (not needed here, but good to know)

---

## Part 4: The Core Challenge — Credit Assignment

This is the central problem of RL:

> The agent receives a reward of +1 VP at turn 80 for building a city. But which of the ~200 actions taken in the previous 79 turns **deserve credit** for that VP?

The road built at turn 3 that connected to a wheat hex? The trade at turn 15 that converted excess brick into needed sheep? The decision NOT to build a settlement at turn 7 because the hex numbers were bad?

**Q-learning** addresses this with the Bellman equation:
```
Q(s, a) = r + γ · max_a' Q(s', a')
```

The value of the current action is the immediate reward plus the discounted value of the best future action. This **bootstraps** — each Q-value is updated toward the next state's Q-value, propagating credit backward through time.

But this is **unstable** with neural networks. Enter Policy Gradients.

---

## Part 5: Policy Gradient Methods — REINFORCE

The policy gradient theorem states that the gradient of the expected return with respect to policy parameters `θ` is:

```
∇_θ J(π) = E[Σ ∇_θ log π(a_t | s_t) · G_t]
```

Where `G_t = Σ γ^(k-t) · r_k` is the **return** — the total discounted reward from time `t` onward.

In plain English:
1. Run a full episode (game)
2. For each action, compute the **total reward** the agent got from that point forward
3. If the return was high, **increase the probability** of those actions via gradient ascent
4. If the return was low, **decrease the probability**

```python
# Simplified REINFORCE
log_probs = []
rewards = []
done = False
while not done:
    action, log_prob = policy.select_action(state)
    next_state, reward, done, _ = env.step(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    state = next_state

# Compute discounted returns
returns = []
G = 0
for r in reversed(rewards):
    G = r + γ * G
    returns.insert(0, G)

# Update: increase probability of actions that led to high returns
loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns))
loss.backward()
optimizer.step()
```

### Problems with Vanilla REINFORCE

1. **High variance**: `G_t` includes rewards from many timesteps. Two games with identical early decisions can have wildly different returns because of late-game dice luck. This means noisy gradients and slow convergence.

2. **No baseline**: If every return is positive (which it often is — Catan VP is always ≥ 0), the gradient always pushes probabilities up. There's no way to say "this action was worse than average."

3. **Sample inefficiency**: Each episode is used once and discarded. In Catan, a single game takes ~80-100 timesteps. You need thousands of games to converge.

---

## Part 6: Actor-Critic Methods — The Best of Both Worlds

Actor-critic methods combine policy-based and value-based approaches:

| Component | Role | In Catan |
|-----------|------|----------|
| **Actor** | The policy `π_θ(a | s)` — decides what action to take | The neural network that outputs action probabilities |
| **Critic** | The value function `V_φ(s)` — evaluates how good a state is | A separate head that predicts "expected final VP from this state" |

The critic provides a **baseline** that dramatically reduces variance:

```
Advantage: A(s, a) = Q(s, a) - V(s)
```

Instead of using the raw return `G_t`, we use the **advantage** — how much better was this action than the average for this state?

```python
# With critic baseline
advantage = returns - values  # how much better/worse than expected
loss = -sum(log_prob * advantage for log_prob, advantage in zip(log_probs, advantages))
```

If an action yielded 5 VP but the critic predicted 5 VP for that state, the advantage is ~0 — the action was neither good nor bad, it was expected. If the critic predicted 2 VP and the action led to 5 VP, the advantage is +3 — this was a genuinely good decision.

---

## Part 7: PPO — Proximal Policy Optimization

PPO is the current gold-standard policy gradient algorithm. It solves three critical problems of earlier actor-critic methods.

### Problem 1: Policy Updates Can Be Too Large

In vanilla policy gradient, a single bad batch of data can **destroy** a well-trained policy with one large gradient step. The policy might suddenly assign near-zero probability to actions it previously liked, and it can never recover.

**PPO's solution: Clipped Surrogate Objective**

```python
ratio = π_new(a|s) / π_old(a|s)  # how much did the policy change?

# Two terms: the normal policy gradient, and a clipped version
unclipped = ratio * advantage
clipped = clip(ratio, 1-ε, 1+ε) * advantage  # ε = 0.2

# Take the minimum — prevents overconfident updates
loss = -min(unclipped, clipped).mean()
```

The `clip` function ensures that the new policy can never deviate more than `ε` (typically 0.2, meaning 20%) from the old policy in probability ratio. This creates a **trust region** — the policy can only move a small, safe step at a time.

Visual intuition:
```
Advantage > 0 (good action):
  ratio > 1+ε  →  clip to 1+ε  (don't get too excited)
  ratio < 1+ε  →  use as-is

Advantage < 0 (bad action):
  ratio < 1-ε  →  clip to 1-ε  (don't get too scared)
  ratio > 1-ε  →  use as-is
```

### Problem 2: Sample Inefficiency

PPO runs the policy in the environment for a fixed number of steps (e.g., 2048), then performs **multiple epochs** (e.g., 4) of gradient descent on that same batch of data. This is possible because the clipping mechanism prevents the policy from drifting too far from the data it was collected on.

```python
# PPO training loop
for iteration in range(total_iterations):
    # Collect data with current policy
    states, actions, rewards, old_log_probs = collect_trajectories(env, policy, steps=2048)
    
    # Compute advantages using Generalized Advantage Estimation (GAE)
    advantages = compute_gae(rewards, values, γ=0.99, λ=0.95)
    
    # Multiple epochs of updates on the SAME data
    for epoch in range(4):
        for batch in minibatches(states, actions, advantages, old_log_probs, batch_size=64):
            ratio = exp(new_log_prob - old_log_prob)
            loss = -min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage).mean()
            loss -= 0.01 * entropy_bonus  # encourage exploration
            loss.backward()
            optimizer.step()
```

### Problem 3: Exploration vs. Exploitation

PPO includes an **entropy bonus** in the loss:

```python
loss = policy_loss - 0.01 * entropy_loss + 0.5 * value_loss
```

Entropy measures how uniform the policy's action distribution is. A high-entropy policy is uncertain (exploring); a low-entropy policy is confident (exploiting). The entropy bonus penalizes the policy for becoming too confident too quickly, forcing it to keep exploring diverse actions.

### Generalized Advantage Estimation (GAE)

PPO doesn't use raw returns for advantage estimation. It uses **GAE**, which smoothly interpolates between:
- **Monte Carlo returns** (low bias, high variance): `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...`
- **TD(0) residuals** (high bias, low variance): `A_t = r_t + γ·V(s_{t+1}) - V(s_t)`

```
GAE(λ) = Σ (γλ)^k · δ_{t+k}
where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD residual)
```

The `λ` parameter (typically 0.95) controls the bias-variance tradeoff. `λ=1` → full Monte Carlo (high variance). `λ=0` → pure TD(0) (high bias). `λ=0.95` → a sweet spot.

---

## Part 8: Action Masking — Critical for Catan

Standard PPO outputs probabilities over the entire action space. In Catan, most actions are **invalid** at any given state (you can't build a city without resources, you can't move the robber without rolling a 7, etc.).

**Without masking**: The agent wastes probability mass on invalid actions, diluting learning signal.

**With masking**: Invalid actions are zeroed out before the softmax, and the agent is penalized for trying them.

```python
def mask_fn(env):
    valid_actions = env.unwrapped.get_valid_actions()  # e.g., [5, 12, 47, 203]
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return mask

# Inside PPO:
logits = policy(state)
masked_logits = logits + log(mask)  # -inf for invalid actions
probs = softmax(masked_logits)
```

This is why you use `sb3_contrib.MaskablePPO` instead of standard `stable_baselines3.PPO`.

---

## Part 9: Why PPO for Catan? Comparing Alternatives

### PPO vs. MCTS (Monte Carlo Tree Search)

| Aspect | PPO | MCTS |
|--------|-----|------|
| **How it works** | Learns a policy via gradient descent on rollouts | Builds a search tree via simulated rollouts at each decision |
| **At inference** | Single forward pass → instant action | Thousands of rollouts per decision → slow |
| **Training time** | Hours to days of parallel simulation | No training, but expensive per-move computation |
| **Handles partial information** | Yes, naturally through observation space | Poorly — MCTS assumes full observability |
| **Stochastic environments** | Learns the distribution | Needs many rollouts to average over dice randomness |
| **Catan suitability** | **Excellent** — fast inference, handles dice, learns from experience | **Poor** — Catan's branching factor is too high for effective search |

**Why MCTS fails for Catan:**
- Catan's branching factor is ~10-50 at each step (vs. chess's ~35, Go's ~250)
- A full game has ~300-400 total actions across all players
- MCTS needs thousands of playouts per decision to converge — at 300 decisions × 1000 playouts each, that's 300,000 full game simulations **per move**
- Even AlphaGo (which uses MCTS + neural nets) only works because it's a two-player, deterministic, perfect-information game. Catan's 4-player, dice-driven, hidden-hand nature breaks MCTS's assumptions.

### PPO vs. GRPO (Group Relative Policy Optimization)

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Baseline** | Critic network `V(s)` estimates expected return | Group of rollouts from same state provides relative comparison |
| **Critic needed?** | Yes — adds parameters and training instability | No — eliminates the value network entirely |
| **Sample efficiency** | Moderate (needs critic training) | Lower (needs many rollouts per state for reliable groups) |
| **Maturity** | Battle-tested (2017), used in AlphaStar, Dota 2, ChatGPT | Experimental (2024), mostly LLM research |
| **Catan suitability** | **Proven** in board game RL | **Unproven** — no evidence it handles multi-agent stochastic games well |

**Why GRPO is not the right choice:**
- GRPO is designed for LLM alignment where you can generate many completions from the same prompt cheaply. In Catan, each "completion" is a full game simulation — generating 8-16 per state is computationally prohibitive.
- GRPO's group-based advantage is noisy when the group size is small. In a high-variance game like Catan (dice can swing VP by 3-4 in a single turn), you'd need very large groups to get reliable signal — which is exactly the sample efficiency problem GRPO was supposed to solve.
- The critic in PPO, once trained, provides a **smooth, generalizable** estimate of state value. GRPO's per-state group averages don't generalize — they're computed independently for each state.

### PPO vs. DQN (Deep Q-Network)

| Aspect | PPO | DQN |
|--------|-----|-----|
| **Policy type** | Direct policy `π(a|s)` | Value function `Q(s,a)`, policy is greedy |
| **Stochastic policies** | Yes | No (always greedy) |
| **Continuous actions** | Yes | No |
| **Large discrete actions** | Handles naturally | Struggles — must eval Q for all actions |
| **Catan suitability** | **Good** | **Poor** — 3000 actions means Q-head is huge, and greedy play is suboptimal |

**Why DQN fails for Catan:**
- The Q-network must output a value for every possible action. With ~3000 actions, the final layer is enormous and training is unstable.
- DQN is inherently greedy — `argmax Q(s,a)`. In Catan, sometimes you want to explore (try a non-obvious placement) to discover better strategies.
- DQN doesn't handle the multi-agent nature well — it assumes a stationary environment, but Catan's other players are adapting.

### PPO vs. SAC (Soft Actor-Critic)

| Aspect | PPO | SAC |
|--------|-----|-----|
| **Action space** | Discrete or continuous | Continuous only (primarily) |
| **Off-policy** | No (on-policy) | Yes (off-policy) |
| **Sample efficiency** | Moderate | High |
| **Catan suitability** | **Good** | **Poor** — Catan has discrete actions |

SAC is designed for **continuous control** (robotics, locomotion). Catan's action space is discrete — you build on node 42 or you don't. SAC's continuous action distribution (Gaussian) doesn't map naturally to this.

### PPO vs. A3C (Asynchronous Advantage Actor-Critic)

| Aspect | PPO | A3C |
|--------|-----|-----|
| **Parallelism** | Synchronous (wait for all workers) | Asynchronous (workers update independently) |
| **Stability** | High (clipped updates) | Lower (stale gradients from async workers) |
| **Catan suitability** | **Good** — stable, reproducible | **Okay** — but `DummyVecEnv` with PPO is more stable |

A3C was superseded by PPO. The asynchronous updates in A3C lead to **stale gradients** — a worker computes a gradient based on a policy version that's already been updated by other workers. PPO's synchronous approach avoids this.

---

## Part 10: Challenges and Issues with PPO in Catan

### 1. Reward Sparsity

In Catan, the agent gets **zero reward** for most actions. Rolling a 6 and getting resources? No direct reward. Building a road? No reward. The only meaningful rewards are:
- +1 VP for building a settlement (indirect)
- +1 VP for building a city (indirect)
- +2 VP for longest road / largest army
- Game end: +1 if won, 0 if lost

The environment must **densify** rewards (e.g., reward VP gains, penalize being robbed, reward resource acquisition) or the agent won't learn.

**Your solution**: `catanatron_gym` already provides shaped rewards. The agent gets small positive rewards for resource gains, small negative rewards for losses, and large rewards for VP gains.

### 2. Multi-Agent Non-Stationarity

In single-agent RL, the environment is stationary — the transition function `P(s'|s,a)` doesn't change. In multi-agent Catan, the "environment" includes 3 other players whose strategies may evolve. This violates the MDP assumption.

**Your solution**: By using profile-aware observations, the agent treats opponent identity as **part of the state**. If opponent A is aggressive and opponent B is passive, those are different states. The policy learns: "in state (board, hands, opponent=A), build roads. In state (board, hands, opponent=B), buy dev cards."

### 3. Partial Observability

In Catan, you can't see other players' hands. This is technically a **Partially Observable MDP (POMDP)**, not an MDP. The agent must maintain an implicit belief about hidden information.

**Your solution**: The observation space includes all visible information (board, public VP, visible dev cards played). The RNN or multi-step history in the base Catanatron observation helps the agent track patterns.

### 4. Dice Variance

Two identical games can diverge wildly based on early dice rolls. This adds enormous variance to the return signal, slowing convergence.

**Your mitigation**: 500-game evaluation runs average out dice variance. During training, the parallel environments and GAE(λ=0.95) help smooth the signal.

### 5. Sample Complexity

PPO needs hundreds of thousands to millions of environment steps to converge. At ~80 steps per Catan game, 500k steps = ~6,250 games. With `DummyVecEnv` × 4 workers, that's still ~1,500 serial games.

**This is why `DummyVecEnv` with 4 parallel environments is critical** — it cuts wall-clock time by 4x.

---

## Part 11: Your Pipeline's Architecture — Putting It All Together

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                            │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────┐    ┌────────────┐ │
│  │ profiles.npy  │───▶│ OpponentProfile     │───▶│ Catanatron │ │
│  │ (5 players ×  │    │ Wrapper             │    │ Gym Env    │ │
│  │  50 features) │    │                     │    │            │ │
│  └──────────────┘    │ - Appends profile   │    │ - Board    │ │
│                      │   vectors to obs    │    │ - Actions  │ │
│                      │ - 3 modes:          │    │ - Rewards  │ │
│                      │   baseline/aware/   │    │            │ │
│                      │   shuffled          │    └─────┬──────┘ │
│                      └──────────┬──────────┘          │        │
│                                 │                     │        │
│                      ┌──────────▼──────────┐          │        │
│                      │ ActionMasker         │          │        │
│                      │ (masks invalid       │          │        │
│                      │  actions to -inf)    │          │        │
│                      └──────────┬──────────┘          │        │
│                                 │                     │        │
│                      ┌──────────▼──────────┐          │        │
│                      │ MaskablePPO          │◀─────────┘        │
│                      │                      │                  │
│                      │ - Actor: π(a|s)      │── steps ──────▶  │
│                      │ - Critic: V(s)       │                  │
│                      │ - GAE(λ=0.95)        │                  │
│                      │ - Clipped surrogate   │                  │
│                      │ - 4 parallel envs    │                  │
│                      └──────────────────────┘                  │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────┐    ┌────────────┐ │
│  │ eval.py      │    │ parse_logs.ipynb    │    │cluster_bots│ │
│  │ (500 games,  │───▶│ (extracts profiles  │───▶│ (K-Means + │ │
│  │  writes txt   │    │  from game logs)    │    │  PCA viz)  │ │
│  │  logs)        │    │                     │    │            │ │
│  └──────────────┘    └─────────────────────┘    └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 12: Key Takeaways

1. **PPO is the right choice** because it handles large discrete action spaces, supports stochastic policies, is sample-efficient enough for board games, and is battle-tested (AlphaStar, OpenAI Five, ChatGPT's RLHF all use PPO variants).

2. **The profile wrapper is your innovation** — it transforms the RL problem from "learn Catan from scratch" to "learn Catan given prior knowledge of opponent behavior." This is analogous to how human players improve faster when they know their opponents' tendencies.

3. **The three-mode design** (baseline/aware/shuffled) provides a rigorous ablation study: baseline proves profiles help, aware proves the agent can generalize, shuffled proves the signal is semantic not just dimensional.

4. **PPO's limitations in Catan** (reward sparsity, multi-agent non-stationarity, partial observability, dice variance) are all mitigated by your design choices: shaped rewards, profile-augmented state, and 500-game evaluation runs.

5. **MCTS, GRPO, DQN, SAC are all worse choices** for this problem domain — MCTS is computationally intractable for Catan's branching factor, GRPO is unproven and sample-hungry, DQN can't handle the action space, and SAC is designed for continuous control.
