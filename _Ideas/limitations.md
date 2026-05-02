# Project Limitations & Technical Justifications

## Overview
This document outlines the key limitations encountered during the development of the opponent-aware Catan AI, along with the technical and engineering justifications for each design decision. These limitations reflect the trade-offs made to balance academic rigor, computational feasibility, and ethical constraints within the project timeline.

---

## 1. Statistical vs. Semantic Opponent Profiling

### Original Proposal
The project initially intended to incorporate Large Language Model (LLM)-generated descriptions of opponent behavior, embedded into high-dimensional vectors (e.g., 256+ dimensions via sentence transformers), and concatenated with the reinforcement learning agent's observation space.

### Implementation
Profiles are strictly statistical: 19-dimensional vectors capturing frequencies of trading, robber aggression, development card timing, and resource hoarding tendencies derived from historical gameplay logs.

### Justification
**Curse of Dimensionality in RL.** Preliminary experiments demonstrated that adding even 30 auxiliary dimensions to the agent's observation space caused a measurable decline in performance. In reinforcement learning with sparse rewards (win/loss at game end), the policy network struggles to disentangle relevant signals from noise when the input dimensionality grows. 

- **Signal Dilution:** Catanatron's base observation space is already ~290 dimensions. Adding 256+ embedding dimensions effectively doubles the input size, forcing the neural network to learn irrelevant correlations unless provided with orders-of-magnitude more training data.
- **Credit Assignment Difficulty:** When rewards are sparse, high-dimensional inputs make it harder for the gradient signal to attribute success to specific features. If 30 dimensions degraded performance, 256 dimensions would almost certainly collapse learning entirely without a massive increase in compute budget.
- **Empirical Validation:** The statistical profiles proved sufficient to capture the behavioral variations necessary for opponent-aware conditioning, without destabilizing the PPO training dynamics.

---

## 2. Chat & Negotiation Modeling

### Original Proposal
Analyze colonist.io chat logs to model negotiation dynamics (e.g., hostility, bluffing, coalition formation) and incorporate these signals into the agent's policy.

### Implementation
Chat analysis was omitted. The agent relies solely on action-based telemetry (trades proposed, robber placements, etc.) to infer opponent styles.

### Justification
**Simulation Environment Mismatch.** While colonist.io allows human chat, Catanatron is strictly a game-mechanics engine. It models resources, board states, and physical actions, but does not support a communication channel or "diplomacy state."

- **Missing Action Space:** Catanatron's action space includes only physical moves (`BUILD_ROAD`, `MARITIME_TRADE`, `MOVE_ROBBER`). There is no `SEND_MESSAGE` or `NEGOTIATE` action available to the agent.
- **Missing Observation Space:** The environment has no representation of social signals like "opponent promised not to rob me" or "opponent is hostile." Even if chat logs were parsed, there is no mechanism to feed that information into the RL policy or train the agent to respond to it.
- **Conclusion:** Behavioral cloning from chat data would require extending the simulation engine to support multi-agent communication protocols, which falls outside the scope of this project.

---

## 3. Fixed Board Topology & Compute Constraints

### Original Proposal
Train agents on fully randomized board configurations (number placements, resource tiles, harbor locations) to maximize generalization across diverse games.

### Implementation
Agents were trained on Catanatron's `BASE` map template with fixed number placements. Only resource tile locations were randomized across episodes.

### Justification
**Sample Complexity vs. Hardware Budget.** This decision served two critical purposes:

1. **Canonical Observation Space:** By keeping the board topology and number placements constant, the agent can efficiently learn spatial heuristics (e.g., recognizing that a 6-tile intersection is high-value) without expending neural capacity on re-learning geometry every episode. This dramatically reduces the sample complexity required for convergence.
2. **Local Compute Feasibility:** The reduction in sample complexity allowed the full training sweep (500k timesteps across 6 configurations: baseline/aware/shuffled × axelrod/no-axelrod) to complete on local Apple Silicon hardware. Had we introduced full board randomization, the agent would have required 5M–10M+ timesteps to generalize across topologies, necessitating the cloud GPU resources (e.g., Azure) outlined in the original proposal, which exceeded the project's budget and timeline.

**Impact:** The model's learned strategies are currently tied to this specific board layout. Performance on TOURNAMENT maps or fully randomized boards may differ, and future work would require a Graph Neural Network (GNN) backbone to achieve topology-invariant generalization.

---

## 4. No Live Deployment on Colonist.io

### Original Proposal
Develop an execution framework enabling the agent to play in online environments (colonist.io) for practical testing, using Selenium/LangGraph for state inference and action mapping.

### Implementation
The execution framework was dropped. Evaluation was conducted entirely within Catanatron's simulation environment and local Docker-based GUI testing.

### Justification
**Ethical & Platform Constraints.** Colonist.io enforces strict rules against botting, particularly in competitive lobbies and ranked play. Automating actions to gain an unfair advantage violates the platform's Terms of Service and undermines competitive integrity.

- **Risk of Unfair Advantage:** Even if the agent is imperfect, features like card counting, optimal strategy hedging, and tireless computation would provide a significant edge over human players.
- **Technical Fragility:** Browser automation (Selenium/Playwright) is highly brittle against UI updates, DOM changes, and anti-botting measures. Maintaining a live bot would divert engineering effort away from the core research (opponent profiling).
- **Alternative Validation:** The agent was tested via `catanatron-play` with a human-in-the-loop GUI (`scripts/play_gui.sh`), allowing for visual verification of strategy (including the Axelrod tit-for-tat mechanic) without violating platform rules.

---

## 5. Training Horizon & Convergence

### Original Proposal
Train agents until convergence, with the option to scale compute resources if needed.

### Implementation
All models were trained for 500k timesteps. Evaluation was conducted with 500 games per configuration.

### Justification
**Compute Budget & Diminishing Returns.** 500k timesteps is a moderate training budget for a complex multi-agent environment like Catan. While larger budgets (1M–5M+ timesteps) typically yield stronger convergence, the 500k sweep was sufficient to demonstrate:

- Baseline agents reaching competitive win rates (>50%).
- Measurable differences between aware, shuffled, and baseline configurations.
- Statistically significant results in opponent-aware conditioning.

Extending training would have required cloud compute resources, which were constrained by the project's budget (under 1000 HKD).

---

## 6. Behavioral Cloning vs. Pure RL

### Original Proposal
Apply behavioral cloning to subsets of colonist.io games to initialize policies that mimic human play before further RL fine-tuning.

### Implementation
This was deferred. Agents were trained from scratch using PPO with random initialization.

### Justification
**Time Constraints & Prioritization.** The project timeline prioritized direct RL training to validate the core hypothesis: that opponent profiling improves performance. Behavioral cloning would have required:

1. Parsing raw colonist.io replays into exact Catanatron-compatible action traces.
2. Implementing a supervised learning pre-training pipeline.
3. Fine-tuning via RL (a two-stage process prone to distribution shift issues).

Given the limited timeline, training from scratch allowed faster iteration on the profile injection mechanism itself. Future work could leverage behavioral cloning as a warm-start to accelerate convergence.

---

## Conclusion

These limitations do not invalidate the project's core finding—that incorporating opponent behavioral profiles into the observation space of a reinforcement learning agent yields measurable performance improvements over baseline models. Rather, they highlight the trade-offs necessary to deliver a functional, empirically validated system within realistic computational, ethical, and timeline constraints. Future iterations would benefit from increased compute budgets, expanded simulation environments with communication channels, and topology-invariant architectures like Graph Neural Networks.
