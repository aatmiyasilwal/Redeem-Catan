# 🎯 Catan FYP 2-Week Coding Sprint (Mar 25 - Apr 13)

## 📋 Phase 1: Data Pipeline (Mar 25–27)
### Mar 25: Setup & Inventory
- [ ] Install/test ColonyHistorian extension and viewer on 2–3 fresh games
- [ ] List all friend usernames + target games (ranked/casual, 4p only)
- [ ] Decide data format (`data/raw/*.json`)
- [ ] Setup project repo: `data/raw/`, `data/processed/`, `src/data/`, `src/rl/`, `src/profiling/`

### Mar 26: Log Collection
- [ ] Download historical replays for friend group (batch download)
- [ ] Pull API/profile stats if available (last 100 games endpoint)
- [ ] Write Python script: loop files → validate JSON → print sample events
- [ ] Confirm schema understanding (dice, builds, trades)

### Mar 27: Event Schema & Processing
- [ ] Define `TurnEvent` schema (`game_id`, `turn_idx`, `player_id`, `action_type`, etc.)
- [ ] Implement `parse_game(game_json) -> list[TurnEvent]`
- [ ] Save to `data/processed/game_events_<id>.parquet`
- [ ] Manual validation: 2–3 games vs viewer/replay

***

## 📊 Phase 2: Opponent Profiling (Mar 28–30)
### Mar 28: Feature Design
- [ ] Finalize feature list (trade freq, robber agg, resource prefs, etc.)
- [ ] Define vector layout (`np.array` length N with indices doc'd)

### Mar 29: Aggregation Implementation
- [ ] `build_player_stats(all_game_events) -> dict[player_id -> stats]`
- [ ] `stats -> feature_vector (np.array)` with normalisation
- [ ] Save `data/processed/player_profiles.json` + `player_profile_vectors.npy`

### Mar 30: EDA & Validation
- [ ] Jupyter notebook: plot/print vectors for you + friends
- [ ] Quick clustering (sklearn k-means) to check non-degeneracy
- [ ] Confirm intuitive differences (e.g., someone trades more)

***

## ⚙️ Phase 3: Catanatron Baseline (Mar 31–Apr 2)
### Mar 31: Env Wiring
- [ ] Install Catanatron + `catanatron-gym`
- [ ] Minimal SB3 training script (`gym.make("catanatron/Catanatron-v0")`)
- [ ] Train 50k steps, save/load model
- [ ] Quick eval vs random/heuristic bots

### Apr 1: Stable Baseline
- [ ] Train longer baseline (500k–1M steps)
- [ ] Tournament script: N games vs default bots → CSV metrics
- [ ] Log win rate, avg VP

### Apr 2: Baseline Cleanup
- [ ] Modularise: `src/rl/env.py`, `src/rl/train.py`, `src/rl/eval.py`
- [ ] Save `models/baseline_ppo.zip` + config

***

## 🔗 Phase 4: Opponent-Aware RL (Apr 3–6)
### Apr 3: State Design
- [ ] Define opponent profile injection (concat to obs)
- [ ] Map colonist `player_id` → Catanatron indices

### Apr 4: Gym Wrapper
- [ ] `OpponentProfileWrapper(gym.ObservationWrapper)`
- [ ] `reset()`: attach opponent vectors
- [ ] `observation(obs)`: concatenate profiles
- [ ] Test shapes with random agent

### Apr 5: First Aware Training
- [ ] Train PPO in wrapped env (300k–500k steps)
- [ ] Confirm stability/logging works

### Apr 6: Quick Comparison
- [ ] Small tournament: baseline vs aware vs friend-profile opponents
- [ ] Check qualitative differences

***

## 🤖 Phase 5: LLM Profiling (Apr 7–9) *optional but high-impact*
### Apr 7: Text Profiles
- [ ] Prompt → LLM for each player’s stats → 2–5 sentence summary
- [ ] Save textual profiles

### Apr 8: Embeddings
- [ ] Embed text profiles (OpenAI/local model)
- [ ] Combine numeric + embedding vectors
- [ ] Update wrapper

### Apr 9: Ablation Setup
- [ ] 3 configs ready to launch:
  1) Baseline PPO
  2) PPO + numeric profiles
  3) PPO + numeric + embeddings

***

## 🏆 Phase 6: Training + Evaluation (Apr 10–13)
### Apr 10: Long Training
- [ ] Launch best configs (prioritise baseline + numeric profiles)
- [ ] Work on eval code while training

### Apr 11: Tournaments
- [ ] Each model: 1000 games vs default bots + friend profiles
- [ ] Metrics: win rate, VP, trades, robber targets

### Apr 12: Ablation Analysis
- [ ] Notebook: compare 3 configs, plot win rates + CIs
- [ ] Check opponent-aware does something different

### Apr 13: Buffer/Cleanup
- [ ] Fix brittle pipeline parts
- [ ] Add minimal docstrings
- [ ] Export key plots/tables for report

***

**✅ Success criteria by Apr 13:**
- ✅ **3 trained models** (baseline, numeric-aware, LLM-aware)
- ✅ **Tournament results** comparing them (CSV + plots)
- ✅ **Working pipeline** from colonist.io → profiles → training → eval
- ✅ **Modular code** ready for documentation

**💡 Quick tips:**
- **Daily checkpoint**: End each day with a quick tournament run to see progress.
- **Compute**: Use Colab T4 for training; local for parsing/EDA.
- **Backup**: Push to GitHub daily (`git commit -m "Day X: [milestone]"`).
- **If behind**: Skip LLM embeddings, focus on numeric profiles + baseline.
