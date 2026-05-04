# Model Configuration Modes: Baseline vs. Aware vs. Shuffled

This document details the architectural and strategic differences between the three training configurations used in the project. Understanding these distinctions is critical for interpreting the evaluation results and the ablation study.

---

## 1. Baseline Mode (`--mode b`)

**"The Specialist"**

### Architecture
In Baseline mode, the agent is trained against a fixed set of three opponents (e.g., players indexed 0, 1, and 3). 
- **Observation Space:** The opponent profile vectors are simply concatenated and appended to the standard board-state observation.
- **Vector Layout:** `[Base_Observation | Profile(P0) | Profile(P1) | Profile(P2)]`

### Behavior
- **Positional Bias:** The neural network learns to associate the *position* of the profile vector in the input array with specific strategies. It might learn "The vector at index X always corresponds to the aggressive trader, so I should avoid trading."
- **Overfitting Risk:** The agent may memorize the specific playstyles of these three individuals rather than learning general concepts of "aggression" or "hoarding." If you were to change the order of opponents or swap in a new player, a Baseline-trained agent might fail to adapt because the input layout has changed.

### Use Case
Establishes the performance floor. It answers: *"What is the best win rate we can achieve if we just train a bot specifically to beat these three friends?"*

---

## 2. Aware Mode (`--mode a`)

**"The Generalist"**

### Architecture
In Aware mode, the agent is trained against a variable pool of opponents sampled randomly from the dataset.
- **Observation Space:** The observation space expands to accommodate slots for *all* potential players (e.g., 50 slots).
- **Vector Layout:** `[Base_Observation | ... | Profile(P_ID_A) ... | Profile(P_ID_B) ...]`
- **Slot Mechanism:** Instead of concatenating vectors sequentially, each player's profile is placed into a specific slot corresponding to their unique ID. If Player A is not in the game, their slot remains zeros.

### Behavior
- **Identity Recognition:** The agent learns to look at specific slots to identify who it is playing against. It decouples "Player Style" from "Seating Position."
- **Generalization:** Because the agent learns that "Slot 5 contains an aggressive player" regardless of where that player sits at the table, it can adapt its strategy dynamically.
- **Strategic Adaptation:** If it encounters a profile vector in a slot that indicates a "Resource Hoarder," it learns a generalized policy to counter hoarders (e.g., aggressive robber placement on their high-yield tiles), even if it hasn't played that exact player before.

### Use Case
The core novelty of the project. It proves that *profiling*—understanding who you are playing against—is a valuable signal for RL agents in imperfect information games.

---

## 3. Shuffled Mode (`--mode s`)

**"The Control (Ablation)"**

### Architecture
Shuffled mode uses the exact same architecture as **Aware Mode** (variable opponents, slot-based layout), but with one critical modification: **The profile vectors are randomly permuted.**
- **Vector Layout:** The data entering the slots is noise. The vector intended for Slot 5 might end up in Slot 12, or be mixed with values from a completely different player.

### Behavior
- **Signal vs. Noise:** This mode tests whether the performance gains in **Aware Mode** are actually due to the *information* contained in the profiles, or simply due to the increased capacity of the neural network (more input dimensions) and the robustness gained from playing against variable opponents.
- **Expected Outcome:** The agent should perform worse than **Aware Mode** because it is receiving contradictory or useless data regarding opponent identities. It essentially has to ignore the "profile slots" and rely only on the base board state.

### Use Case
Scientific rigor. If `Aware > Shuffled`, we prove that the *content* of the profiles matters. If `Aware ≈ Shuffled`, it implies the agent isn't actually using the profile data effectively, and the gains might just be from training against diverse opponents.

---

## Summary Comparison

| Feature | Baseline (`-m b`) | Aware (`-m a`) | Shuffled (`-m s`) |
|:---|:---:|:---:|:---:|
| **Opponent Selection** | Fixed Trio | Random Pool | Random Pool |
| **Profile Injection** | Concatenated (Sequential) | Slot-Mapped (ID-based) | Slot-Mapped (Randomized) |
| **Agent Learns...** | To beat *these specific people* | To recognize *playstyles* | To ignore noise / Generalize |
| **Generalization** | Low | High | High (but no profile signal) |
| **Role in Project** | Lower Bound Performance | **Primary Hypothesis** | **Ablation Control** |

---

## 4. The Axelrod Overlay (`--axelrod 1`)

The Axelrod flag is an orthogonal setting that can be applied to **any** of the three modes. It does not change the observation space or training mode; instead, it intercepts the agent's actions.

- **Logic:** Whenever the agent decides to move the Robber (either via a rolled 7 or a Knight card), the wrapper overrides the model's choice to target the specific opponent who has targeted the agent the most in the current game.
- **Effect:** It forces a "Tit-for-Tat" retaliation strategy, regardless of what the RL policy thinks is optimal. This tests if a hardcoded social strategy (retaliation) can boost performance even when combined with learned policies.
