# How to Interpret PPO Training & Evaluation Output

## Overview
Every time you run `train.py` and `eval.py`, stable-baselines3 emits two blocks: a **training metrics table** and an **evaluation summary**. This document explains every field in depth, what ranges are healthy, what signals trouble, and how to act on each metric.

---

## Part 1: Training Metrics Table

```
-----------------------------------------
| time/                   |             |
|    fps                  | 547         |
|    iterations           | 62          |
|    time_elapsed         | 926         |
|    total_timesteps      | 507904      |
| train/                  |             |
|    approx_kl            | 0.008179251 |
|    clip_fraction        | 0.0687      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.248      |
|    explained_variance   | 0.886       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0369     |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.00817    |
|    value_loss           | 0.00219     |
-----------------------------------------
```

### 1.1 `time/` Block

#### `fps` (Frames Per Second)
**What it is**: Number of environment steps processed per second across all parallel workers.
**In Catan context**: One "frame" = one game decision (place road, buy dev card, propose trade, etc.). Catan has high branching factor (~20-50 legal actions per turn), so each step involves computing legal actions, querying the policy network, and advancing the game state.
**Healthy range**: 400–800 fps on Apple Silicon. Lower values indicate CPU-bound simulation or excessive action-mask computation.
**Troubleshooting**:
- Below 200: The action mask computation (`get_valid_actions()`) or board geometry checks are too slow. Profile `catanatron`'s state machine.
- Above 1000: Likely cached or skipping steps; verify rewards are actually being computed.

#### `iterations`
**What it is**: Number of PPO rollout collections completed. Each iteration gathers `n_steps` (default 2048) timesteps from the environment before running a gradient update.
**In Catan context**: 1 Catan game averages ~200-300 decisions across all players. At 2048 steps/iteration, each iteration represents ~7-10 full games across the 4 parallel environments.
**Healthy range**: For 500k total_timesteps, expect ~244 iterations (500000 / 2048 ≈ 244).
**Note**: This value is cumulative across all logging intervals.

#### `time_elapsed`
**What it is**: Wall-clock seconds since training started.
**Use**: Track training velocity. 500k steps in ~926s = ~540 timesteps/sec.
**Target**: If you want faster iteration cycles, increase parallel environments (currently 4 in `DummyVecEnv`) or reduce `n_steps` for quicker (but noisier) updates.

#### `total_timesteps`
**What it is**: Cumulative number of environment steps processed.
**In Catan context**: This is NOT number of games. One game has 200-300 steps. 500k timesteps ≈ 1,700-2,500 games of experience.
**Target**: The pipeline trains for 500,000 by default. This is a moderate training budget; competitive Catan bots typically need 1M-5M timesteps for convergence.

---

### 1.2 `train/` Block

#### `approx_kl` (Approximate KL Divergence)
**What it is**: Measures how much the updated policy diverges from the old policy (before the gradient step). PPO uses clipping to keep this small.
**Healthy range**: 0.001 – 0.02.
**Interpretation**:
- **Below 0.001**: Updates are too conservative. The policy isn't learning fast enough. Consider increasing learning rate or reducing clipping.
- **Above 0.03**: Updates are too aggressive. The policy is making large jumps that might destabilize training. Reduce learning rate or tighten `clip_range`.
- **0.008 (your value)**: Excellent. Stable learning with meaningful policy improvement.

#### `clip_fraction`
**What it is**: Fraction of policy updates that were clipped by the PPO clip mechanism. When the new policy deviates too far from the old one, the advantage is clipped to prevent destructive updates.
**Healthy range**: 0.05 – 0.20.
**Interpretation**:
- **Below 0.02**: Clipping rarely triggers. Learning rate may be too low; the policy is changing too slowly.
- **Above 0.30**: Clipping triggers frequently. The policy is trying to change faster than PPO allows. This wastes compute and slows convergence. Lower the learning rate.
- **0.0687 (your value)**: Good. Clipping is active but not dominant, meaning the learning rate is well-calibrated.

#### `clip_range`
**What it is**: The epsilon parameter for PPO clipping (default 0.2). This is a hyperparameter, not a learned value.
**Interpretation**: Fixed at 0.2 unless you change `clip_range` in the PPO config. It means the new policy probability ratio is clipped to `[0.8, 1.2]` of the old policy.
**When to adjust**:
- Increase to 0.3 for faster initial learning (riskier).
- Decrease to 0.1 for fine-tuning a pre-trained model.

#### `entropy_loss`
**What it is**: Negative of the policy's entropy. Entropy measures how "random" or "uncertain" the policy is. High entropy = exploring many actions. Low entropy = committing to specific actions.
**In Catan context**: Early in training, entropy should be high (trying different settlement placements, trade strategies). Late in training, entropy should decrease as the policy converges on optimal actions.
**Healthy range**: Starts around -0.5 to -1.0, gradually trends toward -2.0 to -4.0 by the end.
**Interpretation**:
- **Above -0.5**: Policy is very uncertain/random. Too early in training, or entropy coefficient is too high.
- **Below -5.0**: Policy is overly deterministic. Risk of converging to a local optimum. May need more exploration.
- **-0.248 (your value)**: Still early/mid-training. The policy is actively exploring, which is healthy.

#### `explained_variance`
**What it is**: How well the value network predicts actual returns. Ranges from -∞ to 1.0. A value of 1.0 means perfect predictions; 0 means predictions are no better than the mean; negative means worse than the mean.
**In Catan context**: The value network estimates "how good is this board state for me?" Catan has high variance outcomes (dice rolls), so perfect prediction is impossible.
**Healthy range**: 0.5 – 0.95 for mature training.
**Interpretation**:
- **Below 0.3**: Value network is failing to learn the state-value function. The reward signal may be too sparse or noisy. Consider reward shaping.
- **0.5 – 0.7**: Acceptable for early-mid training.
- **Above 0.8**: Excellent. The value network has a good understanding of position evaluation.
- **Above 0.95**: Suspicious. May indicate overfitting to specific game states or reward leakage.
- **0.886 (your value)**: Very good. The value network reliably estimates position advantage.

#### `learning_rate`
**What it is**: Current learning rate for the optimizer. With default `learning_rate=3e-4`, this stays constant unless you use a learning rate schedule.
**Interpretation**: 0.0003 is the standard default for PPO. It works well for most environments. If training plateaus, try a cosine decay schedule starting at 3e-4 and decaying to 1e-5.

#### `loss` (Total Loss)
**What it is**: Combined loss = `value_loss * vf_coef + policy_gradient_loss - entropy_loss * ent_coef`. It's the aggregate objective the optimizer minimizes.
**Interpretation**: The raw number is less meaningful than its trend. A gradually decreasing loss is good. Erratic oscillation suggests instability.
**Note**: Don't obsess over the absolute value. It's a composite of three competing objectives.

#### `n_updates`
**What it is**: Total number of gradient update steps performed. Each iteration (rollout collection) typically triggers 1-4 epochs of updates.
**In Catan context**: With default `n_epochs=4`, each of the ~244 iterations produces 4 updates, yielding ~976 total updates over 500k timesteps.
**Your value (610)**: Consistent with mid-training progress.

#### `policy_gradient_loss`
**What it is**: The loss component that updates the policy network to increase probability of actions with positive advantage and decrease probability of actions with negative advantage.
**Interpretation**:
- **Negative**: Policy is improving (taking actions that yield higher-than-expected returns).
- **Near zero**: Policy has converged or is not finding better actions.
- **Positive**: Policy is degrading (taking worse actions). This can happen temporarily during exploration but should not persist.
- **-0.00817 (your value)**: Slightly negative, indicating gradual improvement.

#### `value_loss`
**What it is**: Mean squared error between the value network's predictions and the actual discounted returns (GAE targets).
**Interpretation**:
- **Decreasing over time**: Value network is learning to predict returns accurately.
- **Stuck above 1.0**: Value network isn't learning. Catan's sparse rewards (win/loss only at game end) make this hard. Consider intermediate rewards (VP gained, resource advantage, longest road).
- **0.00219 (your value)**: Extremely low. The value network is making very accurate predictions. This is excellent for Catan.

---

## Part 2: Evaluation Summary

```
==============================
EVALUATION RESULTS
==============================
Model    : models/baseline_ppo_014.zip
Games    : 500
Win Rate : 52.4%
Avg VP   : 7.41
Saved to : results/0411_003540_baseline_014.csv
==============================
```

### 2.1 `Win Rate`
**What it is**: Percentage of games where the trained agent finished with the most Victory Points.
**Catan context**: With 4 players, random play yields ~25% win rate. A competent bot should achieve 35-45%. Strong bots reach 50-65%.
**Benchmarks**:
| Win Rate | Interpretation |
|----------|---------------|
| 25-30%   | Near-random play |
| 30-40%   | Basic heuristic bot |
| 40-50%   | Solid mid-tier bot |
| 50-60%   | Strong bot (your current level) |
| 60-70%   | Expert-level bot |
| 70%+     | Approaching ceiling against these opponents |

**Your value (52.4%)**: The bot wins more than half its games. This is strong performance, indicating the PPO agent has learned effective strategies beyond baseline heuristics.

### 2.2 `Avg VP` (Average Victory Points)
**What it is**: Mean victory points across all 500 evaluation games, regardless of win/loss.
**Catan context**: Winning requires 10 VP. Average games end between 7-9 VP for non-winners and 10-12 VP for winners.
**Benchmarks**:
| Avg VP | Interpretation |
|--------|---------------|
| < 6.0  | Agent consistently falls behind |
| 6.0-7.0 | Agent competes but struggles to close |
| 7.0-7.5 | Agent frequently reaches 8-9 VP (one dev card away from winning) |
| 7.5-8.5 | Strong positioning; agent wins when it secures the last 1-2 VP efficiently |
| 8.5+   | Agent dominates games |

**Your value (7.41)**: The agent consistently reaches competitive VP totals. Combined with 52.4% win rate, this means the agent wins when it hits the final VP threshold efficiently, and loses when it stalls at 8-9 VP.

### 2.3 Combined Interpretation

| Win Rate | Avg VP | Diagnosis |
|----------|--------|-----------|
| Low (<35%) | Low (<6) | Agent is outplayed; needs more training or reward shaping |
| Low (<35%) | High (7.5+) | Agent reaches good positions but loses to end-game execution (largest army, longest road races, or opponent steals) |
| High (50%+) | Low (<6.5) | Agent wins by opponents self-destructing, not by strong play. Unlikely to generalize to better opponents |
| High (50%+) | High (7.5+) | Strong, well-rounded agent |

**Your profile (52.4% / 7.41)**: Healthy strong agent. To push further:
1. **If Win Rate stagnates but Avg VP rises**: Agent is competitive but can't close games. Train longer, or add specific end-game reward shaping.
2. **If both plateau**: You've hit the skill ceiling of these opponents. Switch to harder opponents (`aware` or `shuffled` mode).

### 2.4 CSV Output
The file `results/0411_003540_baseline_014.csv` contains per-game results:
```csv
game_id,won,vp
1,1,10
2,0,7
3,1,11
...
```
- **Use it for**: Computing variance, streaks, and distribution analysis. A bot that wins 52% of games but has high variance (lots of 4-VP blowouts and 11-VP stomps) is less consistent than one with tight 7-8 VP spreads.
- **Filename format**: `MMDD_HHMMSS_mode_players.csv` — sortable by timestamp.

---

## Part 3: Red Flags & Quick Diagnostics

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `explained_variance` drops below 0.3 | Sparse rewards, value network can't learn | Add intermediate rewards (VP deltas, resource advantage) |
| `clip_fraction` > 0.4 consistently | Learning rate too high | Reduce LR to 1e-4 or add LR schedule |
| `entropy_loss` stays above -0.3 after 500k steps | Policy never commits to actions | Increase entropy coefficient decay |
| `approx_kl` spikes above 0.05 occasionally | Normal; PPO clips these updates | No action needed if it recovers |
| Win Rate oscillates wildly between eval runs | Small eval sample size (500 games) | Increase `n_games` to 1000+ for stable estimates |
| Avg VP > 8.0 but Win Rate < 40% | Agent hogs resources but doesn't convert to VP | Reward VP acquisition directly, not just resource accumulation |
| Training loss goes negative and keeps decreasing | Healthy; policy and value both improving | Continue training |
| Training loss goes positive | Policy degradation | Reduce LR, increase `n_epochs`, or check reward function |

---

## Part 4: Training Progression Expectations

For a 500k timestep training run:

| Timesteps | Expected State |
|-----------|---------------|
| 0-50k | Random exploration; metrics are noisy |
| 50-150k | Policy learns basic heuristics; `explained_variance` crosses 0.5 |
| 150-300k | Strategic patterns emerge (port usage, development card timing); win rate climbs |
| 300-500k | Fine-tuning; metrics stabilize; diminishing returns |

If your agent looks "done" at 200k, consider early stopping. If metrics are still climbing at 500k, extend to 1M.
