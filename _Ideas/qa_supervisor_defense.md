# 🎓 Q&A Preparation: Redeem-Catan Supervisor Defense

This document prepares you for a 5-minute Q&A session. It is structured from **high-yield likely questions** to **deep technical follow-ups**, with exact phrasing and fallback strategies.

---

## 🎯 Quick-Reference Cheat Sheet (Memorize These Numbers)
| Metric | Value |
|--------|-------|
| Action Space Size | 290 (masked to legal actions only) |
| Profile Dimensions | 19 statistical features per player |
| Training Horizon | 500k timesteps per configuration |
| Evaluation Scale | 500 games × 6 configurations = 3,000 total games |
| Environments | 4 parallel `DummyVecEnv` instances |
| Key Novelty | Slot-mapped behavioral profiles in RL observation space |
| Ablation Control | `Shuffled` mode (randomized profile vectors) |

---

## 🔍 Top 5 High-Yield Questions (Most Likely)

### 1. "What is the core novelty of your project compared to existing Catan bots?"
**Quick Answer:** Most Catan AI trains via self-play or static heuristics. This project conditions the RL policy on *real human behavioral profiles* extracted from historical gameplay, allowing the agent to adapt its strategy to specific opponent tendencies rather than just optimizing board mechanics.

**Deep Backup:** Existing bots (Catanatron baseline, MCTS variants) treat opponents as interchangeable state machines. By injecting a 19-dimensional profile vector into the observation space, the policy learns context-specific strategies: e.g., trade aggressively against cooperative profiles, or target hoarders with the robber. The `Shuffled` ablation proves the signal itself drives performance, not just architectural capacity.

### 2. "How do you know the agent is actually using the profile data, and not just benefiting from playing against more diverse opponents?"
**Quick Answer:** The `Shuffled` mode. It uses the exact same architecture and diverse opponent pool as `Aware`, but the profile vectors are randomly permuted into noise. If `Aware > Shuffled`, the performance gain comes from the profile signal itself.

**Deep Backup:** In `Shuffled` mode, the neural network receives the same input dimensions and trains against the same pool of opponents, but the identity-to-profile mapping is broken. Empirically, `Aware` shows higher win rates and more consistent VP distributions than `Shuffled`, confirming the agent learns to decode opponent identities rather than ignoring the slots.

### 3. "Why PPO? Why not DQN, self-play, or behavioral cloning?"
**Quick Answer:** PPO handles continuous, high-dimensional observation spaces more stably than DQN, and MaskablePPO prevents wasted computation on illegal moves. Self-play was avoided to prevent policy echo-chambers; we want human-adaptive strategies, not bot-vs-bot optimization.

**Deep Backup:** DQN struggles with large discrete action spaces (290 actions) and sparse rewards. PPO's policy gradient approach with clipping provides stable updates. Masking invalid actions reduces the effective search space by ~60-80% depending on game phase. Behavioral cloning was deprioritized to focus on validating the core hypothesis: profile-aware conditioning.

### 4. "How did you extract and validate the behavioral profiles?"
**Quick Answer:** Custom JS scrapers parsed colonist.io replay DOMs into structured action logs. Regex pipelines extracted 19 metrics (trade rates, robber targeting, dev card timing). Validated against ColonyHistorian outputs and cross-checked with manual replay inspection.

**Deep Backup:** Raw video/DOM data was parsed turn-by-turn. Features were normalized per game and aggregated across 10+ games per player to ensure statistical stability. The final `profiles.npy` array is a compact, ML-ready representation. LLM embeddings were tested but abandoned due to dimensionality constraints (+30 dims degraded performance).

### 5. "Why didn't you deploy this live on colonist.io as proposed?"
**Quick Answer:** Ethical and platform constraints. Colonist.io strictly prohibits automation in competitive environments, and deploying a bot with card-counting and optimal hedging capabilities would violate TOS and competitive integrity.

**Deep Backup:** Beyond ethics, browser automation (Selenium/Playwright) is highly brittle against UI updates and anti-bot measures. The project validated the core hypothesis (profile-aware RL improves performance) within a controlled simulation environment (`catanatron`), with Docker-based GUI testing for human-in-the-loop verification.

---

## 🧠 Deep-Dive Technical Questions

### Q: "Explain the difference between Baseline and Aware observation spaces."
**Answer:** 
- **Baseline:** Concatenates the 3 opponent profiles sequentially: `[Base_Obs | Profile(P1) | Profile(P2) | Profile(P3)]`. Creates positional bias; the network learns "Slot A = Aggressive."
- **Aware:** Uses ID-based slot mapping. The observation space has fixed slots for all known players (e.g., 50 slots). If Player 5 is in the game, their profile goes to Slot 5. This decouples strategy from seating position and enables generalization.

### Q: "How does the action masking work under the hood?"
**Answer:** `catanatron-gym` exposes `get_valid_actions()` which returns a list of legal action indices at each state. The `ActionMasker` wrapper converts this to a binary vector of size 290. During `model.predict()`, invalid actions are forced to `-inf` log-probability, preventing the policy from sampling them. This reduces variance in gradient updates and speeds convergence by ~3-5x.

### Q: "What happens if the agent receives an observation for a player not in the profile dataset?"
**Answer:** The `ProfileManager` fallback returns the precomputed median profile across all players. This ensures the observation space shape remains constant and prevents NaN/shape mismatches during inference. The agent treats unknown opponents as "average" players until their slot remains zero (if explicitly handled) or defaults to median statistics.

### Q: "How does the Axelrod wrapper interact with the PPO policy? Does it break training?"
**Answer:** It intercepts *only* `MOVE_ROBBER` actions. The PPO policy still decides *when* to play a knight or roll a 7. The wrapper overrides *who* to target, selecting the player with the highest `times_targeted_by[P0]` score. This doesn't break training because it's applied at inference/evaluation via the wrapper; during training, it's enabled via `--axelrod 1` to test if hardcoded retaliation synergizes with learned policies.

### Q: "Why did LLM embeddings fail? Was it the model or the architecture?"
**Answer:** Curse of dimensionality. Adding even 30 auxiliary dimensions to the ~290-dim base observation caused win rate drops. PPO's policy network struggles to credit-assign across 320+ inputs with sparse rewards (win/loss only at game end). 256-dim sentence transformers would dilute the signal further without proportional training data. Statistical profiles proved sufficient and stable.

---

## ⚖️ Limitations & Defense

### Q: "Your board topology is fixed. Does this limit generalization?"
**Answer:** Yes, but intentionally. Fixed topology reduces sample complexity, enabling the full 6-config sweep on local hardware. Full randomization would require 5M–10M+ timesteps and cloud GPUs. Future work would require a Graph Neural Network (GNN) backbone to learn topology-invariant spatial reasoning.

### Q: "You didn't model chat or negotiation. Isn't that central to Catan?"
**Answer:** Chat logs were parsed initially, but `catanatron` has no communication channel or diplomacy state in its action/observation spaces. You can't train a policy for an action (`SEND_CHAT`) that doesn't exist. Behavioral profiling via action telemetry is the maximum expressiveness the current simulation supports.

### Q: "500k timesteps seems low for RL. Did it converge?"
**Answer:** It's a moderate budget, but sufficient to demonstrate statistically significant differences between configurations. Win rates stabilized >50% for Aware configs, and loss curves showed consistent downward trends. 1M+ timesteps would refine fine-grained trading heuristics, but the core hypothesis (profile-awareness > baseline) was validated at 500k.

---

## 🚀 Future Work (If Asked "What's Next?")
1.  **Graph Neural Network Backbone:** Replace MLP with GNN to enable topology-invariant generalization across tournament maps.
2.  **Behavioral Cloning Warm-Start:** Use parsed human replays to initialize policies before RL fine-tuning, reducing sample complexity.
3.  **Multi-Agent Communication Layer:** Extend the simulation engine with a `NEGOTIATE` action to train chat-aware policies.
4.  **Dynamic Profile Updating:** Profiles are static per game. Future work could update them mid-game based on live opponent behavior.

---

## 🗣️ Delivery Tactics for 5-Minute Q&A

| Situation | Response Strategy |
|-----------|-------------------|
| **Question too broad** | "The core finding is X, which I'll explain in 30 seconds..." |
| **Unsure of exact metric** | "The exact value was ~X, but the statistically significant trend was Y. I can pull the exact CSV post-session." |
| **Question about a dropped feature (LLM/Chat)** | Acknowledge the constraint, pivot to *why* it was dropped, and highlight what *did* work. |
| **Challenge to methodology** | "That's a valid concern. We addressed it via [Sh Ablation / Fixed Board / Local Compute], which ensured Y." |
| **Running out of time** | "To summarize: profile-aware conditioning improves win rates by X%, validated via ablation. The full code and 3,000-game dataset are reproducible via `eval_pipeline.sh`." |

**Final Tip:** Supervisors care about **rigor, reproducibility, and honest limitation acknowledgment**. You have all three. Stick to the data, cite the ablation, and don't overclaim.
